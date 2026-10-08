"""
Parametric Monte-Carlo Policy Adaptation (pMCPA).

Reference : Suphx paper section 3.4, Eq. 6.

    theta_a = arg max_theta  Sum_{tau ~ theta_o . T(h)}  R(tau) * p(tau;theta) / p(tau;theta_o)

Au debut de chaque manche :
    1. Simulations : K trajectoires avec la policy offline (theta_o)
       -> adversaires et mur tires aleatoirement (hors notre main)
    2. Adaptation  : gradient updates avec importance sampling pondere par R(tau)
    3. Inference   : utilise theta_a pour cette manche

Apres la manche, on reset sur theta_o.

Note : s'applique UNIQUEMENT au modele discard (comme pour le RL).

API utilisee par le sniffer :
    pmcpa = PMCPA(models, head_sizes, device, k=50, lr=1e-4, grad_steps=3)
    pmcpa.reset()                             # au debut de chaque manche
    pmcpa.adapt_from_sniffer_state(state)     # apres avoir recu notre main
    is_ready = pmcpa.is_adapted
    probs = pmcpa.run_all_heads_adapted(obs_normal, obs_oracle)
"""
import copy
import os
import threading
import time

import numpy as np
import torch
import torch.nn as nn


# ====================================================================== #
# Constantes
# ====================================================================== #
N_TILES = 34
HAND_SIZE = 13
COPIES_PER_TILE = 4
LIVE_WALL_SIZE = 70
DEAD_WALL_SIZE = 14
RINSHAN_SIZE = 4

# Reward proxy par manche : score brut (en points / 1000)
# Le papier ne detaille pas R(tau) ; on utilise le score de la manche.
# Alternatives : rang a la fin de la manche, delta de score, etc.
DEFAULT_ROUND_REWARD_SCALE = 1.0 / 1000.0


# ====================================================================== #
# Simulation d'une manche isolee
# ====================================================================== #
def _sample_round_state(our_hand_counts, rng):
    """Genere un etat initial de manche avec notre main fixe.

    Args:
        our_hand_counts : np.array (34,) int -- main de notre agent (13 tuiles)
        rng             : np.random.RandomState

    Retourne un dict avec :
        hands   : liste de 4 np.array (34,) int -- indexee par siege REEL
        wall    : liste de tuiles restantes (70)
        dead_wall : liste de tuiles du mur mort (14)
        dora_indicators : liste d'indicators
        seat    : notre siege (0 par convention)
    """
    # Pool de tuiles restantes : 136 - notre main
    pool = []
    for t in range(N_TILES):
        remaining = COPIES_PER_TILE - int(our_hand_counts[t])
        if remaining > 0:
            pool.extend([t] * remaining)
    rng.shuffle(pool)

    # Distribution : 3 adversaires x 13 = 39, dead wall = 14, wall = 83
    # (on tronque le mur vivant a 70 comme dans env)
    idx = 0
    opp_hands = []
    for _ in range(3):
        h = np.zeros(N_TILES, dtype=np.int32)
        for t in pool[idx:idx + HAND_SIZE]:
            h[t] += 1
        idx += HAND_SIZE
        opp_hands.append(h)

    dead_wall = pool[idx:idx + DEAD_WALL_SIZE]
    idx += DEAD_WALL_SIZE

    # On garde tout le reste comme mur vivant (le nombre exact varie,
    # mais MultiAgentMahjongEnv gere deja la fin sur mur vide).
    live_wall = pool[idx:]

    dora_indicators = [dead_wall[0]] if dead_wall else []
    ura_dora = [dead_wall[1]] if len(dead_wall) > 1 else []

    hands = [
        our_hand_counts.astype(np.int32),   # siege 0 (nous)
        opp_hands[0], opp_hands[1], opp_hands[2],
    ]

    return {
        'hands': hands,
        'wall': live_wall,
        'dead_wall': dead_wall,
        'dora_indicators': dora_indicators,
        'ura_dora_indicators': ura_dora,
    }


# ====================================================================== #
# Agent pMCPA
# ====================================================================== #
class PMCPA:
    """Adaptation run-time de la policy discard a la main initiale."""

    def __init__(self, models, head_sizes, device='cpu',
                 k=50, lr=1e-4, grad_steps=3, timeout_s=20.0,
                 reward_scale=DEFAULT_ROUND_REWARD_SCALE,
                 verbose=True):
        """
        Args:
            models      : dict {head_name: nn.Module} (discard/riichi/chow/pong/kong)
            head_sizes  : dict {action_type: int} (pour la mask Categorical)
            device      : 'cpu' ou 'cuda'
            k           : nombre de simulations (papier : "does not need to be very large")
            lr          : learning rate pour l'adaptation
            grad_steps  : nombre d'updates de gradient
            timeout_s   : timeout max pour l'adaptation (en secondes)
            reward_scale: facteur de conversion du score en reward
            verbose     : logs
        """
        self.device = device
        self.k = int(k)
        self.lr = float(lr)
        self.grad_steps = int(grad_steps)
        self.timeout_s = float(timeout_s)
        self.reward_scale = float(reward_scale)
        self.verbose = bool(verbose)

        # Modeles (on garde des references vers le discard seul)
        self.models = models
        self.discard = models['discard']
        self.frozen = {h: models[h] for h in
                        ['riichi', 'chow', 'pong', 'kong']
                        if h in models}

        # Sauvegarde des poids OFFLINE (pour reset apres adaptation)
        self._offline_discard_state = copy.deepcopy(
            self.discard.state_dict()
        )

        # Optimizer pour l'adaptation (cree a la demande)
        self._adapt_optim = None

        # Etat
        self.is_adapted = False
        self._initial_hand = None   # np.array (34,) count
        self._thread = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def reset(self):
        """Restaure les poids OFFLINE. Appele au debut de chaque manche."""
        with self._lock:
            self.discard.load_state_dict(self._offline_discard_state)
            self.discard.eval()
            self.is_adapted = False
            self._initial_hand = None
            self._adapt_optim = None
        if self.verbose:
            print("[pMCPA] reset sur poids offline", flush=True)

    # ------------------------------------------------------------------ #
    def adapt_from_sniffer_state(self, state):
        """Lance l'adaptation a la main initiale.

        Args:
            state : GameState du sniffer (doit contenir state.hand,
                    state.my_seat, state.melds, state.dora_indicators, ...)

        Lance l'adaptation dans un thread separe pour ne pas bloquer le
        sniffer. `is_adapted` passe a True a la fin.
        """
        # Extrait la main depuis le sniffer (13 tuiles au debut de manche)
        if not state.hand:
            if self.verbose:
                print("[pMCPA] main vide, adaptation ignoree", flush=True)
            return

        hand_counts = np.zeros(N_TILES, dtype=np.int32)
        for t in state.hand:
            if 0 <= t < N_TILES:
                hand_counts[t] += 1

        if int(hand_counts.sum()) != HAND_SIZE:
            if self.verbose:
                print(f"[pMCPA] main invalide ({hand_counts.sum()} tuiles, "
                      f"attendu {HAND_SIZE}), adaptation ignoree", flush=True)
            return

        self._initial_hand = hand_counts
        self.is_adapted = False

        # Lance dans un thread daemon
        self._thread = threading.Thread(
            target=self._adapt_worker,
            args=(hand_counts, state),
            daemon=True,
        )
        self._thread.start()

    def compute_contextual_reward(self, sniffer_state, round_score):
        """Reward contextuel pour pMCPA.

        - Si pas la derniere manche : score brut (positif = bien).
        - Derniere manche : evaluer l'impact sur le rang final.
        """
        round_idx = sniffer_state.round_idx  # 0..3
        is_last_round = (round_idx >= 3)

        # Recupere nos scores et notre siege
        my_seat = sniffer_state.my_seat
        scores = list(sniffer_state.scores)   # deja centres sur le baseline

        # Calcule le rang actuel
        def _rank_of(seat, sc):
            return 1 + sum(1 for s in sc if s > sc[seat])

        current_rank = _rank_of(my_seat, scores)

        if not is_last_round:
            # Manches East 1-3 : le score brut est un bon proxy
            return round_score

        # Derniere manche : context important
        new_scores = scores.copy()
        new_scores[my_seat] += round_score
        new_rank = _rank_of(my_seat, new_scores)

        # Score de transition de rang
        # +1 place gagnee = +5, -1 place perdue = -5, etc.
        rank_delta = current_rank - new_rank

        if current_rank == 1:
            # Proteger la 1ere place : perdre le rang est TRES mauvais
            if new_rank == 1:
                return +15.0   # on garde la 1ere place
            else:
                return -15.0   # on la perd, tres mauvais
        elif current_rank == 4:
            # Doit gagner gros pour sortir de la 4eme
            if new_rank == 4:
                return -15.0   # toujours dernier
            else:
                return +10.0 * rank_delta   # recompense proportionnelle
        else:
            # Rangs 2-3 : reward par place gagnee
            return 5.0 * rank_delta
    # ------------------------------------------------------------------ #
    def _adapt_worker(self, hand_counts, state):
        """Thread d'adaptation."""
        t_start = time.time()
        try:
            if self.verbose:
                print(f"[pMCPA] debut adaptation (main={hand_counts.sum()} "
                      f"tuiles, K={self.k})", flush=True)

            # 1. Generer les K simulations
            trajectories = self._simulate_k_rounds(hand_counts, state)

            if not trajectories:
                if self.verbose:
                    print("[pMCPA] aucune trajectoire generee", flush=True)
                return

            # 2. Adaptation (gradient updates)
            self._adapt_with_importance_sampling(trajectories)

            self.is_adapted = True

            if self.verbose:
                dt = time.time() - t_start
                print(f"[pMCPA] adaptation terminee en {dt:.1f}s "
                      f"({len(trajectories)} trajectoires)", flush=True)

        except Exception as e:
            if self.verbose:
                print(f"[pMCPA] erreur adaptation : {e}", flush=True)
                import traceback
                traceback.print_exc()
            self.is_adapted = False

    # ------------------------------------------------------------------ #
    def _simulate_k_rounds(self, our_hand, sniffer_state):
        """Simule K manches avec la policy offline.

        [FIX 2025-XX] Cree l'env UNE SEULE FOIS et le passe a chaque sim
        (evite la re-init des arrays numpy). Timeout via t_start unique.
        """
        from mahjong_ai.envs.multi_agent import MultiAgentMahjongEnv

        trajectories = []
        rng = np.random.RandomState(int(time.time() * 1000) & 0x7fffffff)

        offline_state = copy.deepcopy(self.discard.state_dict())
        self.discard.eval()

        # [FIX] Cree UNE SEULE FOIS l'env, avec look-ahead desactive
        base_env = MultiAgentMahjongEnv(
            seed=int(rng.randint(0, 2**31 - 1)),
            compute_lookahead=False,
        )

        t_start = time.time()

        for k in range(self.k):
            # Timeout unique
            if time.time() - t_start > self.timeout_s:
                if self.verbose:
                    print(f"[pMCPA] timeout sims a k={k}/{self.k} "
                        f"({time.time()-t_start:.1f}s)", flush=True)
                break

            try:
                traj = self._simulate_one_round(
                    our_hand, sniffer_state, rng, offline_state, base_env)
                if traj is not None:
                    trajectories.append(traj)
                    if self.verbose and (k + 1) % 5 == 0:
                        print(f"[pMCPA] sim {k+1}/{self.k} OK "
                            f"({time.time()-t_start:.1f}s, "
                            f"{len(traj['trajectory'])} transitions, "
                            f"reward={traj['reward']:+.1f})", flush=True)
            except Exception as e:
                if self.verbose:
                    print(f"[pMCPA] sim {k} echouee : {e}", flush=True)
                continue

        self.discard.train()
        return trajectories

    # ------------------------------------------------------------------ #
    def _simulate_one_round(self, our_hand, sniffer_state, rng,
                            offline_state, base_env):
        """Simule UNE manche complete en reutilisant base_env.

        [FIX 2025-XX] Prend base_env en argument au lieu de creer un nouvel
        env (le look-ahead est deja desactive dans base_env).
        """
        from mahjong_ai.envs.multi_agent import (
            DISCARD, PONG, CHOW, KONG, RIICHI, RON,
        )

        MAX_STEPS = 500

        print(f"[pMCPA-DBG] debut sim, hand={int(our_hand.sum())} tuiles",
            flush=True)

        # [FIX] Reutilise base_env au lieu d'en creer un nouveau
        env = base_env
        env.rng = np.random.RandomState(int(rng.randint(0, 2**31 - 1)))

        # Remplace _deal pour injecter la main
        sampled = _sample_round_state(our_hand, rng)
        print(f"[pMCPA-DBG] pool sample OK, wall={len(sampled['wall'])} tuiles",
            flush=True)

        original_deal = env._deal

        def custom_deal():
            original_deal()
            env.hands[0] = sampled['hands'][0].copy()
            env.hands[1] = sampled['hands'][1].copy()
            env.hands[2] = sampled['hands'][2].copy()
            env.hands[3] = sampled['hands'][3].copy()
            env.wall = list(sampled['wall'])
            env.rinshan_pool = list(sampled['dead_wall'][:RINSHAN_SIZE])
            env.dora_indicators = list(sampled['dora_indicators'])
            env.ura_dora_indicators = list(sampled['ura_dora_indicators'])
            env._initial_wall_size = len(env.wall)

        env._deal = custom_deal

        trajectory = []
        t0 = time.time()
        final_reward = 0.0
        step_count = 0

        try:
            print(f"[pMCPA-DBG] reset_round...", flush=True)
            req = env.reset_round()
            print(f"[pMCPA-DBG] premier req : seat={req['seat']} "
                f"atype={req['action_type']}", flush=True)

            while True:
                step_count += 1
                if step_count > MAX_STEPS:
                    print(f"[pMCPA-DBG] MAX_STEPS atteint ({MAX_STEPS})",
                        flush=True)
                    break

                seat = req['seat']
                obs = req.get('obs')
                mask = req.get('mask')
                atype = req.get('action_type')

                if atype == DISCARD:
                    model = self.discard
                elif atype == KONG:
                    model = self.frozen['kong']
                elif atype == RIICHI:
                    model = self.frozen['riichi']
                elif atype == PONG:
                    model = self.frozen['pong']
                elif atype == CHOW:
                    model = self.frozen['chow']
                elif atype == RON:
                    action = (1 if mask is not None and len(mask) > 1
                            and mask[1] > 0.5 else 0)
                    result = env.step(action)
                    if result.get('round_done'):
                        fr = result.get('final_rewards', {}) or {}
                        final_reward = self.compute_contextual_reward(
                            sniffer_state, float(fr.get(0, 0.0)))
                        break
                    req = result['next_request']
                    continue
                else:
                    result = env.step(None)
                    if result.get('round_done'):
                        fr = result.get('final_rewards', {}) or {}
                        final_reward = self.compute_contextual_reward(
                            sniffer_state, float(fr.get(0, 0.0)))
                        break
                    req = result['next_request']
                    continue

                if obs is None or mask is None:
                    result = env.step(None)
                    if result.get('round_done'):
                        fr = result.get('final_rewards', {}) or {}
                        final_reward = self.compute_contextual_reward(
                            sniffer_state, float(fr.get(0, 0.0)))
                        break
                    req = result['next_request']
                    continue

                # [FIX] Concat 4 canaux oracle zeros pour le discard
                if atype == DISCARD:
                    from mahjong_ai.agents.suphx_cnn import (
                        IN_CHANNELS_ORACLE, N_CHANNELS as _NC,
                    )
                    ORACLE_DIM = IN_CHANNELS_ORACLE - _NC
                    if obs.shape[0] == _NC:
                        zeros = np.zeros(
                            (ORACLE_DIM, obs.shape[1]), dtype=obs.dtype)
                        obs_model = np.concatenate([obs, zeros], axis=0)
                    else:
                        obs_model = obs
                else:
                    obs_model = obs

                obs_t = torch.as_tensor(obs_model, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)

                with torch.no_grad():
                    logits = model(obs_t)
                    masked = logits + (mask_t - 1.0) * 1e9
                    dist = torch.distributions.Categorical(logits=masked)
                    action = dist.sample()
                    logp = dist.log_prob(action).item()

                action_int = int(action.item())

                if atype == DISCARD:
                    trajectory.append({
                        'obs': obs.copy(),
                        'mask': mask.copy(),
                        'action': action_int,
                        'logp': logp,
                    })

                result = env.step(action_int)

                if result.get('round_done'):
                    fr = result.get('final_rewards', {}) or {}
                    final_reward = self.compute_contextual_reward(
                        sniffer_state, float(fr.get(0, 0.0)))
                    break
                req = result['next_request']

            print(f"[pMCPA-DBG] sim terminee, {step_count} steps, "
                f"{len(trajectory)} transitions, "
                f"reward={final_reward:+.2f}", flush=True)

            return {
                'trajectory': trajectory,
                'reward': final_reward,
                't0': t0,
            }

        except Exception as e:
            print(f"[pMCPA] sim echouee apres {step_count} steps : {e}",
                flush=True)
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------ #
    def _adapt_with_importance_sampling(self, trajectories):
        """Effectue grad_steps updates avec IS pondere par R(tau).

        [FIX 2025-XX] Le modele reste en eval() pendant TOUT le bloc
        d'adaptation. En mode train(), les BatchNorm mettent a jour leurs
        buffers running_mean/running_var inplace a chaque forward, ce qui
        casse autograd quand plusieurs forwards precedent un backward
        (l'erreur "variable modified by an inplace operation").

        En eval(), les BN utilisent leurs stats figees (pas de mutation) et
        .backward() fonctionne normalement sur les parametres requires_grad.
        """
        if self._adapt_optim is None:
            self._adapt_optim = torch.optim.Adam(
                self.discard.parameters(), lr=self.lr)

        # [FIX] Force eval mode : pas de mutation inplace des BN buffers
        self.discard.eval()

        for step_i in range(self.grad_steps):
            self._adapt_optim.zero_grad()

            total_loss = 0.0
            n_valid = 0

            for traj in trajectories:
                transitions = traj['trajectory']
                if not transitions:
                    continue

                R = float(traj['reward'])
                if R == 0.0:
                    continue

                # Log proba sous la policy actuelle
                logp_sum = None
                for t in transitions:
                    # [FIX] Concat des 4 canaux oracle zeros pour le discard
                    from mahjong_ai.agents.suphx_cnn import (
                        IN_CHANNELS_ORACLE, N_CHANNELS as _NC,
                    )
                    ORACLE_DIM = IN_CHANNELS_ORACLE - _NC
                    obs_data = t['obs']
                    if obs_data.shape[0] == _NC:
                        zeros = np.zeros(
                            (ORACLE_DIM, obs_data.shape[1]),
                            dtype=obs_data.dtype)
                        obs_data = np.concatenate([obs_data, zeros], axis=0)
                    obs_t = torch.as_tensor(
                        obs_data, dtype=torch.float32,
                        device=self.device).unsqueeze(0)
                    mask_t = torch.as_tensor(
                        t['mask'], dtype=torch.float32,
                        device=self.device).unsqueeze(0)

                    logits = self.discard(obs_t)
                    masked = logits + (mask_t - 1.0) * 1e9
                    dist = torch.distributions.Categorical(logits=masked)
                    a_t = torch.tensor([t['action']], dtype=torch.long,
                                        device=self.device)
                    step_logp = dist.log_prob(a_t).sum()
                    if logp_sum is None:
                        logp_sum = step_logp
                    else:
                        logp_sum = logp_sum + step_logp

                if logp_sum is None:
                    continue

                # Importance sampling
                logp_old = sum(t['logp'] for t in transitions)
                log_ratio = logp_sum - logp_old
                ratio = torch.exp(log_ratio)

                # Loss : -R(tau) * ratio * logp_sum
                total_loss = total_loss - R * ratio * logp_sum
                n_valid += 1

            if n_valid == 0:
                break

            total_loss = total_loss / n_valid
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discard.parameters(), 5.0)
            self._adapt_optim.step()

        # Reste en eval pour l'inference (run_all_heads_adapted)
        self.discard.eval()

    # ------------------------------------------------------------------ #
    def run_all_heads_adapted(self, obs_normal, obs_oracle=None):
        """Forward avec la policy ADAPTEE (theta_a).

        Retourne un dict {head_name: probs}.
        """
        # Si pas adapte, fallback sur offline
        # (le caller devrait gerer, mais securite)
        obs_t = torch.as_tensor(obs_normal, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)

        # Prep pour discard : ajoute les canaux oracle si necessaire
        from mahjong_ai.agents.suphx_cnn import (
            IN_CHANNELS_ORACLE, N_CHANNELS,
        )
        if obs_normal.shape[0] == N_CHANNELS:
            oracle_zeros = np.zeros(
                (IN_CHANNELS_ORACLE - N_CHANNELS, obs_normal.shape[1]),
                dtype=obs_normal.dtype,
            )
            obs_d = np.concatenate([obs_normal, oracle_zeros], axis=0)
        else:
            obs_d = obs_normal
        obs_d_t = torch.as_tensor(obs_d, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)

        results = {}
        with torch.no_grad():
            for head in ['discard', 'riichi', 'chow', 'pong', 'kong']:
                if head not in self.models:
                    continue
                if head == 'discard':
                    logits = self.discard(obs_d_t)[0]
                else:
                    logits = self.models[head](obs_t)[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                results[head] = probs
        return results

    # ------------------------------------------------------------------ #
    def wait_adaptation(self, timeout=None):
        """Bloque jusqu'a ce que l'adaptation soit terminee (ou timeout)."""
        if self._thread is not None:
            self._thread.join(timeout=timeout or self.timeout_s)