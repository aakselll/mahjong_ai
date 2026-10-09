"""
Agent RL Suphx : policy (discard) + critic + oracle channels.

Conforme au papier §3.3 (oracle guiding) :
    - input : [x_n(s) ; δ_t · x_o(s)]  (386 canaux)
    - δ_t ~ Bernoulli(γ_t) sur les 136 canaux oracle
    - γ_t décroît de 1 à 0 sur gamma_decay_steps

Une fois γ=0, l'agent devient "normal" (250 canaux effectifs, les 136
autres à zéro). C'est ce qui permet de déployer l'agent sans info parfaite.
"""
import numpy as np
import torch
import torch.nn as nn

from mahjong_ai.agents.suphx_cnn import (
    build_model, N_CHANNELS, ORACLE_DIM, IN_CHANNELS_ORACLE,
)


class Critic(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS_ORACLE, in_tiles=34, hidden=512):
        super().__init__()
        flat = in_channels * in_tiles
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class OracleDiscardAgent:
    """Policy (discard) + critic + SL figés (riichi/chow/pong/kong)."""

    def __init__(self, sl_discard_path, sl_frozen_dir, device='cpu',
                 policy_lr=1e-4, critic_lr=1e-3):
        self.device = device

        # Policy avec canaux oracle
        self.policy = build_model('discard', use_oracle=True).to(device)
        ck = torch.load(sl_discard_path, map_location=device, weights_only=False)
        # Charge partiellement (les 250 premiers canaux d'entrée matchent)
        self._load_partial_sl(ck['model_state_dict'])
        print(f"[oracle_agent] discard SL chargé (val_acc={ck.get('val_acc','?')})")

        # Critic
        self.critic = Critic().to(device)

        # SL figés
        self.frozen_sl = {}
        for head in ['riichi', 'chow', 'pong', 'kong']:
            path = f"{sl_frozen_dir}/{head}.pt"
            try:
                m = build_model(head).to(device)
                ck = torch.load(path, map_location=device, weights_only=False)
                m.load_state_dict(ck['model_state_dict'])
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                self.frozen_sl[head] = m
            except Exception as e:
                print(f"[oracle_agent] {head} non chargé : {e}")

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _load_partial_sl(self, state_dict):
        """Charge partiellement : le trunk (conv_in) attend 386 canaux mais
        le SL en a 250. On étend les poids d'entrée en initialisant les
        canaux oracle à 0.
        """
        my_state = self.policy.state_dict()
        for k, v in state_dict.items():
            if k not in my_state:
                continue
            if my_state[k].shape == v.shape:
                my_state[k].copy_(v)
            elif k == 'trunk.input_conv.weight':
                # (256, 386, 3, 1) attendu, on a (256, 250, 3, 1)
                new_w = torch.zeros_like(my_state[k])
                new_w[:, :v.shape[1]] = v
                my_state[k].copy_(new_w)
                print(f"[oracle_agent] {k} etendu de {v.shape} a {new_w.shape}")
            # Les autres mismatch sont ignores (rare)
        self.policy.load_state_dict(my_state)

    # ------------------------------------------------------------------ #
    def forward_with_dropout(self, obs_normal, obs_oracle, gamma_t):
        """Construit l'input avec dropout oracle et forward.

        obs_normal : (B, N_CHANNELS, 34)
        obs_oracle : (B, ORACLE_DIM, 34)
        gamma_t    : float dans [0, 1]

        Retourne (logits_discard, value, oracle_mask).
        """
        B = obs_normal.shape[0]
        # Mask Bernoulli sur les canaux oracle
        if gamma_t <= 0:
            delta = torch.zeros(B, ORACLE_DIM, 1, device=self.device)
        elif gamma_t >= 1:
            delta = torch.ones(B, ORACLE_DIM, 1, device=self.device)
        else:
            delta = (torch.rand(B, ORACLE_DIM, 1, device=self.device)
                     < gamma_t).float()
        oracle_masked = obs_oracle * delta
        obs_full = torch.cat([obs_normal, oracle_masked], dim=1)  # (B, 386, 34)
        logits = self.policy(obs_full)
        value = self.critic(obs_full)
        return logits, value, delta

    # ------------------------------------------------------------------ #
    def select_action(self, obs_normal, obs_oracle, mask, gamma_t,
                       deterministic=False):
        """Forward + sample pour la collecte."""
        on = torch.as_tensor(obs_normal, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        oo = torch.as_tensor(obs_oracle, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        logits, value, _ = self.forward_with_dropout(on, oo, gamma_t)
        masked = logits + (m - 1.0) * 1e9
        dist = torch.distributions.Categorical(logits=masked)
        a = torch.argmax(masked, -1) if deterministic else dist.sample()
        return (int(a.item()),
                float(dist.log_prob(a).item()),
                float(value.item()),
                float(dist.entropy().item()))

    # ------------------------------------------------------------------ #
    def evaluate_actions(self, obs_normal, obs_oracle, masks, actions,
                          gamma_t):
        """Batch forward pour l'update. gamma_t controle le dropout."""
        logits, values, _ = self.forward_with_dropout(obs_normal, obs_oracle,
                                                        gamma_t)
        masked = logits + (masks - 1.0) * 1e9
        dist = torch.distributions.Categorical(logits=masked)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, values, entropy

    # ------------------------------------------------------------------ #
    def save(self, path, step=None, alpha=None):
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'step': step,
            'alpha': alpha,          # <-- AJOUT
        }, path)