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
import os
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
    """Policy (discard) + critic + SL figés (riichi/chow/pong/kong).

    [FIX 2026-10-14] L'architecture est lue depuis la config sauvegardee
    dans chaque checkpoint (train_policy_sl.py stocke {'config': ...})
    au lieu des defauts de build_model. Sans ca :
    - les tetes legeres (128 canaux / 6 blocs) ne chargent pas ;
    - la policy RL (defaut 16 blocs) ne warm-start que partiellement
      depuis un SL a 12 blocs -- les blocs 12-15 restent aleatoires.
    """
    def __init__(self, sl_discard_path, sl_frozen_dir, device='cpu',
                 policy_lr=1e-4, critic_lr=1e-3):
        self.device = device

        # --- Policy : archi lue depuis le checkpoint SL discard ---
        ck = torch.load(sl_discard_path, map_location=device,
                        weights_only=False)
        cfg = ck.get('config', {}) if isinstance(ck, dict) else {}
        channels = cfg.get('channels', 256)
        n_blocks = cfg.get('n_blocks', 16)

        self.policy = build_model('discard', use_oracle=True,
                                  channels=channels,
                                  n_blocks=n_blocks).to(device)
        self._load_partial_sl(ck['model_state_dict'])
        print(f"[oracle_agent] discard SL charge "
              f"(val_acc={ck.get('val_acc','?')}, "
              f"arch={channels}x{n_blocks})", flush=True)

        # --- Critic ---
        self.critic = Critic().to(device)

        # --- SL figes : archi lue depuis CHAQUE checkpoint ---
        self.frozen_sl = {}
        for head in ['riichi', 'chow', 'pong', 'kong']:
            path = os.path.join(sl_frozen_dir, f'{head}.pt')
            try:
                hck = torch.load(path, map_location=device,
                                 weights_only=False)
                hcfg = hck.get('config', {}) if isinstance(hck, dict) else {}
                m = build_model(
                    head,
                    channels=hcfg.get('channels', 256),
                    n_blocks=hcfg.get('n_blocks', 16),
                    binary_hidden=hcfg.get('binary_hidden', 256),
                    in_channels=hcfg.get('in_channels', N_CHANNELS),
                ).to(device)
                m.load_state_dict(hck['model_state_dict'])
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                self.frozen_sl[head] = m
                print(f"[oracle_agent] {head} charge "
                      f"(arch={hcfg.get('channels','?')}x"
                      f"{hcfg.get('n_blocks','?')})", flush=True)
            except Exception as e:
                print(f"[oracle_agent] {head} non charge : {e}", flush=True)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(),
                                             lr=policy_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=critic_lr)

        # [FIX BATCHNORM] La policy contient des BatchNorm2d (tronc + 16
        # blocs). Un module fraichement construit est en mode train : en
        # rollout (select_action, batch de 1) la BN normaliserait alors
        # chaque canal sur les seuls 34 emplacements de l'etat courant au
        # lieu d'utiliser les stats du SL, et evaluate_actions (batch de
        # ~1000) utiliserait d'autres stats encore -> la policy qui joue
        # n'est pas celle du checkpoint SL et le ratio PPO est du bruit.
        # En eval, les stats SL sont figees (les gradients, eux, passent
        # normalement, et gamma/beta de la BN restent entrainables).
        self.policy.eval()

    def set_eval_mode(self):
        """Remet la policy en eval (BN figee). A rappeler apres tout
        load_state_dict / resume : load_state_dict ne change pas le mode,
        mais un .train() externe l'aurait change."""
        self.policy.eval()
        self.critic.eval()

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
    def select_action(self, obs_normal, obs_oracle, mask, gamma_t):
        """Retourne (action, logp, value, masked_obs_oracle).
        
        [FIX ORACLE GUIDING] Le masque Bernoulli appliqué à l'oracle fait
        partie de l'ETAT vu par la policy. Pour que l'importance sampling
        soit correct (ratio = 1 quand les poids n'ont pas change), il faut
        que l'evaluation utilise LE MEME masque que la generation.
        
        On retourne donc l'oracle MASQUE pour qu'il soit stocke dans la
        transition et reutilise tel quel pendant l'entrainement.
        """
        on = torch.as_tensor(obs_normal, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        oo = torch.as_tensor(obs_oracle, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32,
                                device=self.device).unsqueeze(0)

        # Appliquer le dropout Bernoulli per-element
        if gamma_t < 1.0:
            dropout = torch.bernoulli(
                torch.full_like(oo, gamma_t))
            oo_masked = oo * dropout
        else:
            oo_masked = oo

        x = torch.cat([on, oo_masked], dim=1)
        with torch.no_grad():
            logits = self.policy(x)
            masked_logits = logits + (mask_t - 1.0) * 1e9
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            
            # Critic (inutilise avec value_coef=0 mais garde pour compat)
            value = self.critic(x).item()

        # [FIX] Retourner l'oracle MASQUE (pas le brut)
        masked_obs_oracle = oo_masked.squeeze(0).cpu().numpy()
        
        return int(action.item()), logp.item(), value, masked_obs_oracle

    # ------------------------------------------------------------------ #
    def evaluate_actions(self, obs_n, obs_o, masks, actions, gamma_t):
        """Evalue les log-probs des actions sous la policy courante.
        
        [FIX ORACLE GUIDING] obs_o est DEJA MASQUE (stocke depuis la
        generation). On ne re-applique PAS de nouveau dropout -- sinon le
        ratio d'importance sampling serait faux (memes poids != ratio 1
        car les entrees different).
        
        gamma_t est garde en parametre pour compatibilite mais IGNORE :
        le masque est deja "bake in" dans obs_o.
        """
        on = torch.as_tensor(obs_n, dtype=torch.float32,
                            device=self.device)
        oo = torch.as_tensor(obs_o, dtype=torch.float32,
                            device=self.device)
        # [FIX] PAS de nouveau dropout -- obs_o est deja masque
        x = torch.cat([on, oo], dim=1)
        
        logits = self.policy(x)
        mask_t = torch.as_tensor(masks, dtype=torch.float32,
                                device=self.device)
        masked_logits = logits + (mask_t - 1.0) * 1e9
        dist = torch.distributions.Categorical(logits=masked_logits)
        logp = dist.log_prob(actions)
        
        values = self.critic(x)
        # Entropie sur les actions valides
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return logp, values.squeeze(-1), entropy

    # ------------------------------------------------------------------ #
    def save(self, path, step=None, alpha=None):
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'step': step,
            'alpha': alpha,          # <-- AJOUT
        }, path)