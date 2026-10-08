"""
CNN Suphx (échelle réduite ~7M params). Input : (N_CHANNELS, 34, 1).

Référence : Suphx paper §2.2, Fig. 4-5.
    Tronc   : 3x1 conv 256 -> 16 blocs résiduels (3x1 conv 256)
    Discard : 1x1 conv 1 -> flatten -> 34 logits (Fig. 4)
    Binaire : 3x1 conv 32 -> flatten -> 2 FC -> N sorties (Fig. 5)

[RL] Support de l'oracle guiding (Suphx §3.3) :
    - in_channels peut être étendu de N_CHANNELS (250) à
      IN_CHANNELS_ORACLE (386) pour inclure les 136 canaux parfaits
    - use_oracle=True dans build_model pour le faire automatiquement
    - _load_partial_sl (dans oracle_agent.py) permet d'initialiser
      les poids du tronc avec le SL : les canaux normaux sont copiés,
      les canaux oracle sont initialisés à 0.
"""
import torch
import torch.nn as nn

from mahjong_ai.envs.suphx_features import N_CHANNELS

# Dimensions oracle : 3 mains adverses (3*34=102) + mur caché (34) = 136
# 4 canaux oracle : 3 mains adverses + 1 mur cache
ORACLE_DIM = 4
IN_CHANNELS_ORACLE = N_CHANNELS + ORACLE_DIM   # 254

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + r)


class SuphxTrunk(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, channels=256, n_blocks=16):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, channels, (3, 1),
                                     padding=(1, 0), bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_blocks)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_in(self.input_conv(x)))
        for b in self.blocks:
            x = b(x)
        return x


class DiscardHead(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x).squeeze(-1).squeeze(1)   # (B, 34)


class BinaryHead(nn.Module):
    def __init__(self, in_channels=256, n_out=2, mid_channels=32, hidden=256, n_fc=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, (3, 1),
                               padding=(1, 0), bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        flat = mid_channels * 34
        layers, prev = [], flat
        for _ in range(n_fc):
            layers += [nn.Linear(prev, hidden), nn.ReLU(inplace=True)]
            prev = hidden
        layers.append(nn.Linear(prev, n_out))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x))).flatten(1)
        return self.fc(x)


class SuphxModel(nn.Module):
    def __init__(self, head_type, in_channels=N_CHANNELS, channels=256,
                 n_blocks=16, binary_hidden=256, binary_mid=32, binary_fc=2):
        super().__init__()
        assert head_type in ('discard', 'riichi', 'chow', 'pong', 'kong')
        self.head_type = head_type
        self.trunk = SuphxTrunk(in_channels, channels, n_blocks)
        if head_type == 'discard':
            self.head = DiscardHead(channels)
        else:
            n_out = 4 if head_type == 'chow' else 2
            self.head = BinaryHead(channels, n_out=n_out, mid_channels=binary_mid,
                                    hidden=binary_hidden, n_fc=binary_fc)

    def forward(self, x):
        if x.dim() == 3:                    # (B, C, 34)
            x = x.unsqueeze(-1)
        return self.head(self.trunk(x))


def build_model(head_type, use_oracle=False, **kwargs):
    """Build Suphx model.

    Args:
        head_type  : 'discard' / 'riichi' / 'chow' / 'pong' / 'kong'
        use_oracle : si True, force in_channels=IN_CHANNELS_ORACLE (386).
                     Sinon, utilise in_channels du kwargs (defaut N_CHANNELS=250).
    """
    if use_oracle:
        kwargs.setdefault('in_channels', IN_CHANNELS_ORACLE)
    return SuphxModel(head_type, **kwargs)