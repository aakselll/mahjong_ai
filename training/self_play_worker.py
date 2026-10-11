"""
Worker self-play : utilise GameEnv (multi-round) au lieu de
MultiAgentMahjongEnv (single round).

[FIX 2025-XX] L'ancien code utilisait MultiAgentMahjongEnv, ce qui faisait
qu'une "partie" etait en realite UNE seule manche. Maintenant : 4 rounds
East 1-4 + repeat dealer + GRP reward.

[STATS] Ajout d'un stats_proxy partage (Manager dict) qui accumule les
statistiques de jeu (rangs, longueur, outcomes).
"""
import time
import numpy as np
import torch
import os

from mahjong_ai.training.game_env import GameEnv, PredictorHolder
from mahjong_ai.agents.oracle_agent import OracleDiscardAgent
from mahjong_ai.training.parameter_server import fetch_from_state
from mahjong_ai.training.replay_buffer import buffer_add
from mahjong_ai.envs.multi_agent import DISCARD
from mahjong_ai.agents.reward_predictor import RewardPredictor, FEATURE_DIM


def worker_loop(worker_id, state, lock, buffer_data, buffer_max,
                 stats_proxy,                    # <-- NOUVEAU
                 sl_discard_path, sl_frozen_dir,
                 grp_path=None,
                 n_games_per_cycle=10, device='cpu'):

    # [FIX] Limite les threads PyTorch par worker pour eviter
    # l'oversubscription.
    n_workers_total = int(os.environ.get("RL_N_WORKERS", "1"))
    threads_per_worker = max(1, torch.get_num_threads() // max(n_workers_total, 1))
    torch.set_num_threads(threads_per_worker)
    print(f"[worker {worker_id}] torch threads = {threads_per_worker}",
          flush=True)

    agent = OracleDiscardAgent(sl_discard_path, sl_frozen_dir, device=device)

    # --- GRP ---
    predictor_holder = PredictorHolder()
    if grp_path is not None:
        try:
            grp = RewardPredictor(input_dim=FEATURE_DIM, hidden_dim=128,
                                    num_layers=2).to(device)
            ck = torch.load(grp_path, map_location=device, weights_only=False)
            grp.load_state_dict(ck)
            grp.eval()
            predictor_holder.model = grp
            print(f"[worker {worker_id}] GRP charge", flush=True)
        except Exception as e:
            print(f"[worker {worker_id}] GRP load error : {e}", flush=True)

    # --- GameEnv 4-round East ---
    env = GameEnv(seed=worker_id, predictor_holder=predictor_holder,
                  game_log_buffer=None, n_rounds=4,
                  game_format="tonpuu")
    env.round_env.frozen_sl = agent.frozen_sl

    print(f"[worker {worker_id}] demarre (n_rounds=4)", flush=True)

    while True:
        policy_state, critic_state, step, alpha, gamma_t = \
            fetch_from_state(state, lock)
        if policy_state is not None:
            try:
                agent.policy.load_state_dict(policy_state)
                agent.critic.load_state_dict(critic_state)
            except Exception as e:
                print(f"[worker {worker_id}] load error : {e}", flush=True)

        for _ in range(n_games_per_cycle):
            transitions, game_stats = _play_one_game(env, agent, gamma_t)
            if transitions:
                buffer_add(buffer_data, buffer_max, transitions)
            # [STATS] mise a jour du proxy partage
            if game_stats is not None:
                _update_stats(stats_proxy, lock, game_stats)

        time.sleep(0.01)


def _update_stats(stats_proxy, lock, game_stats):
    """Met a jour le proxy de stats apres une partie.

    Utilise le meme lock que le parameter server pour eviter la contention.
    Chaque compteur est lu et ecrit sous le lock pour l'atomicite.
    """
    rank = game_stats['rank_of_seat0']
    rank_key = (f'n_{rank}st' if rank == 1 else
                 f'n_{rank}nd' if rank == 2 else
                 f'n_{rank}rd' if rank == 3 else 'n_4th')
    with lock:
        stats_proxy['n_games'] = stats_proxy.get('n_games', 0) + 1
        stats_proxy[rank_key] = stats_proxy.get(rank_key, 0) + 1
        stats_proxy['sum_rank'] = stats_proxy.get('sum_rank', 0.0) + rank
        stats_proxy['n_rounds_total'] = (
            stats_proxy.get('n_rounds_total', 0) + game_stats['n_rounds'])
        stats_proxy['n_honba_total'] = (
            stats_proxy.get('n_honba_total', 0) + game_stats['n_honba'])
        stats_proxy['n_tsumo'] = (
            stats_proxy.get('n_tsumo', 0) + game_stats['n_tsumo'])
        stats_proxy['n_ron'] = (
            stats_proxy.get('n_ron', 0) + game_stats['n_ron'])
        stats_proxy['n_exhaustive'] = (
            stats_proxy.get('n_exhaustive', 0) + game_stats['n_exhaustive'])
        stats_proxy['n_abortive'] = (
            stats_proxy.get('n_abortive', 0) + game_stats['n_abortive'])


def _play_one_game(env, agent, gamma_t):
    """Joue une partie COMPLETE (4 rounds + repeats).

    [FIX 2025-XX] Reward de la manche k attribue a la DERNIERE transition
    du siege DANS LA MANCHE k.

    [FIX 2025-XX-2] Tracking de max_honba (le honba final est souvent 0
    apres un dernier round non-repeat). On prend le maximum atteint
    pendant la partie pour les stats.

    Retourne (transitions, stats) ou stats est un dict des stats de la
    partie.
    """
    transitions = []
    pending_reward = {s: 0.0 for s in range(4)}
    last_idx_per_seat = {s: None for s in range(4)}
    max_honba = 0

    # [STATS] Compteurs locaux
    stats = {
        'n_rounds': 0,
        'n_tsumo': 0,
        'n_ron': 0,
        'n_exhaustive': 0,
        'n_abortive': 0,
    }

    req = env.reset_round()
    while True:
        seat = req['seat']
        action = None
        logp = value = 0.0
        obs_n = obs_o = mask = None

        if req['action_type'] == DISCARD:
            obs_n = req['obs']
            obs_o = req['oracle']
            mask = req['mask']
            action, logp, value, _ = agent.select_action(
                obs_n, obs_o, mask, gamma_t)

        result = env.step(action)

        if action is not None:
            transitions.append({
                'seat': seat,
                'obs_normal': obs_n,
                'obs_oracle': obs_o,
                'mask': mask,
                'action': action,
                'logp': logp,
                'value': value,
                'reward': pending_reward[seat],
                'done': False,
            })
            pending_reward[seat] = 0.0
            last_idx_per_seat[seat] = len(transitions) - 1

        if result['round_done']:
            # [STATS] categorisation du outcome
            outcome = result.get('outcome', 'unknown')
            if outcome == 'tsumo':
                stats['n_tsumo'] += 1
            elif outcome in ('ron', 'multi_ron'):
                stats['n_ron'] += 1
            elif outcome == 'exhaustive':
                stats['n_exhaustive'] += 1
            elif outcome.startswith('abortive'):
                stats['n_abortive'] += 1
            stats['n_rounds'] += 1

            # [FIX] Tracker le honba max atteint pendant la partie.
            # env.honba est incremente dans GameEnv.step() a la fin
            # de la manche si le dealer repeat.
            max_honba = max(max_honba, env.honba)

            round_rewards = result['final_rewards']
            for s in range(4):
                reward_s = round_rewards.get(s, 0.0)
                reward_s = max(-20.0, min(20.0, reward_s))
                if reward_s == 0.0:
                    continue
                idx = last_idx_per_seat[s]
                if idx is not None:
                    transitions[idx]['reward'] += reward_s
                else:
                    pending_reward[s] += reward_s

            last_idx_per_seat = {s: None for s in range(4)}

            if result.get('game_done', False):
                for s in range(4):
                    if pending_reward[s] != 0.0:
                        for t in reversed(transitions):
                            if t['seat'] == s:
                                t['reward'] += pending_reward[s]
                                break
                        pending_reward[s] = 0.0
                break
            req = env.reset_round()
        else:
            req = result['next_request']

    if transitions:
        transitions[-1]['done'] = True

    # [STATS] Calcul du rang final de seat 0
    scores = env.game_scores   # list de 4 floats (par siege REEL)
    order = sorted(range(4), key=lambda p: (-scores[p], p))
    rank_of_seat0 = order.index(0) + 1

    stats['rank_of_seat0'] = rank_of_seat0
    stats['n_honba'] = max_honba    # <-- FIX : max au lieu de env.honba

    return transitions, stats