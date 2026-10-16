"""
Worker self-play : utilise GameEnv (multi-round) au lieu de
MultiAgentMahjongEnv (single round).

[FIX 2025-XX] L'ancien code utilisait MultiAgentMahjongEnv, ce qui faisait
qu'une "partie" etait en realite UNE seule manche. Maintenant : 4 rounds
East 1-4 + repeat dealer + GRP reward.

[NIVEAU 2 / FIX IPC 2026-10-14] Les transitions passent par une mp.Queue
    au lieu d'une liste proxy Manager :
    - l'ancien chemin tuait le serveur Manager sous Windows (~75 min
      observes) ;
    - la Queue est le mecanisme prevu pour le transfert de donnees
      volumineuses (pipes dedies, backpressure via maxsize) ;
    - put(timeout=30) : si le trainer est mort, le worker s'arrete
      proprement au lieu de bloquer indefiniment sur une Queue pleine.

    Signature changee vs Niveau 1 : (buffer_data, buffer_max) ->
    transitions_queue. Le trainer passe la Queue en 4e argument.

[STATS] stats_proxy partage (Manager dict) qui accumule les statistiques
    de jeu (rangs, longueur, outcomes). Volume leger, frequence moderee :
    reste tolerable pour le Manager.
"""
import os
import time

import numpy as np
import torch

from mahjong_ai.training.game_env import GameEnv, PredictorHolder
from mahjong_ai.agents.oracle_agent import OracleDiscardAgent
from mahjong_ai.training.parameter_server import fetch_from_state
from mahjong_ai.envs.multi_agent import DISCARD
from mahjong_ai.agents.reward_predictor import RewardPredictor, FEATURE_DIM


def worker_loop(worker_id, state, lock, transitions_queue,
                stats_proxy,
                sl_discard_path, sl_frozen_dir,
                grp_path=None,
                n_games_per_cycle=10, device='cpu'):
    """
    [NIVEAU 2] Les transitions passent par une mp.Queue au lieu d'une
    liste proxy Manager.

    Args :
        worker_id          : id du worker (pour logs + seed GameEnv)
        state, lock        : proxies Manager du parameter server
        transitions_queue  : mp.Queue partagee avec le trainer
        stats_proxy        : proxy Manager dict pour les stats de jeu
        sl_discard_path    : chemin du checkpoint SL discard (warm-start)
        sl_frozen_dir      : dossier des tetes SL gelees
        grp_path           : chemin du checkpoint GRP (optionnel)
        n_games_per_cycle  : parties jouees avant re-sync des poids
        device             : cpu
    """
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
            ck = torch.load(grp_path, map_location=device, weights_only=False)
            # [FIX] Le checkpoint est un dict avec metadonnees -- extraire
            # le state_dict et lire l'archi dedans (hidden_dim=64 sauvegarde,
            # ne pas hardcoder 128).
            state_dict = ck.get("model_state_dict", ck)
            hidden_dim = ck.get("hidden_dim", 64)
            input_dim = ck.get("feature_dim", FEATURE_DIM)
            grp = RewardPredictor(input_dim=input_dim, hidden_dim=hidden_dim,
                                  num_layers=2).to(device)
            grp.load_state_dict(state_dict)
            grp.eval()
            predictor_holder.model = grp
            print(f"[worker {worker_id}] GRP charge (hidden={hidden_dim}, "
                  f"val_loss={ck.get('val_loss', '?')})", flush=True)
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
                # [NIVEAU 2] put au lieu de buffer_add.
                # timeout : ne pas bloquer indefiniment si le trainer est
                # mort (Queue pleine -> exception -> arret propre).
                try:
                    transitions_queue.put(transitions, timeout=30.0)
                except Exception:
                    print(f"[worker {worker_id}] Queue indisponible -- "
                          f"arret du worker", flush=True)
                    return
            # [STATS] mise a jour du proxy partage
            if game_stats is not None:
                _update_stats(stats_proxy, lock, game_stats)


def _update_stats(stats_proxy, lock, game_stats):
    """Accumule les stats d'une partie dans le proxy partage.

    [FIX 2026-10-14] Aligne sur le format REEL de _play_one_game :
      - rang          : cle 'rank_of_seat0' (pas 'rank')
      - outcomes      : compteurs CUMULES par partie (n_tsumo/n_ron/
                        n_exhaustive/n_abortive somment les manches),
                        pas une cle 'outcome' unique
      - n_rounds/n_honba : cles deja correctes
    """
    if game_stats is None:
        return
    with lock:
        stats_proxy['n_games'] = stats_proxy.get('n_games', 0) + 1
        rank = game_stats.get('rank_of_seat0', 0)
        if rank == 1:
            stats_proxy['n_1st'] = stats_proxy.get('n_1st', 0) + 1
        elif rank == 2:
            stats_proxy['n_2nd'] = stats_proxy.get('n_2nd', 0) + 1
        elif rank == 3:
            stats_proxy['n_3rd'] = stats_proxy.get('n_3rd', 0) + 1
        elif rank == 4:
            stats_proxy['n_4th'] = stats_proxy.get('n_4th', 0) + 1
        stats_proxy['sum_rank'] = stats_proxy.get('sum_rank', 0.0) + rank
        stats_proxy['n_rounds_total'] = stats_proxy.get('n_rounds_total', 0) \
            + game_stats.get('n_rounds', 0)
        stats_proxy['n_honba_total'] = stats_proxy.get('n_honba_total', 0) \
            + game_stats.get('n_honba', 0)
        # [FIX] Outcomes : compteurs cumules par manche dans la partie --
        # on ajoute directement les compteurs de la partie.
        stats_proxy['n_tsumo'] = stats_proxy.get('n_tsumo', 0) \
            + game_stats.get('n_tsumo', 0)
        stats_proxy['n_ron'] = stats_proxy.get('n_ron', 0) \
            + game_stats.get('n_ron', 0)
        stats_proxy['n_exhaustive'] = stats_proxy.get('n_exhaustive', 0) \
            + game_stats.get('n_exhaustive', 0)
        stats_proxy['n_abortive'] = stats_proxy.get('n_abortive', 0) \
            + game_stats.get('n_abortive', 0)

def _play_one_game(env, agent, gamma_t):
    """Joue une partie COMPLETE (4 rounds + repeats).

    [FIX MULTI-SEAT REWARD] Le reward de la manche k est attribue a
    TOUTES les transitions du siege dans cette manche (papier : "assign
    it to each state-action pair in round k"). L'ancien code ne
    l'assignait qu'a la DERNIERE transition -- 98 pourcent des
    transitions avaient reward=0, diluant le signal d'apprentissage par
    ~50x.
    """
    transitions = []
    pending_reward = {s: 0.0 for s in range(4)}
    # [FIX] Tous les indices par siege pour la manche courante
    round_indices_per_seat = {s: [] for s in range(4)}
    max_honba = 0

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
            action, logp, value, masked_obs_o = agent.select_action(
                obs_n, obs_o, mask, gamma_t)

        result = env.step(action)

        if action is not None:
            transitions.append({
                'seat': seat,
                'obs_normal': obs_n,
                'obs_oracle': masked_obs_o,
                'mask': mask,
                'action': action,
                'logp': logp,
                'value': value,
                'reward': 0.0,          # [FIX] rempli a la fin de la manche
                'done': False,
            })
            # [FIX] Enregistrer TOUTES les transitions du siege
            round_indices_per_seat[seat].append(len(transitions) - 1)

        if result['round_done']:
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

            max_honba = max(max_honba, env.honba)

            # [FIX] Attribuer le reward de la manche a TOUTES les
            # transitions de chaque siege dans cette manche
            round_rewards = result['final_rewards']
            for s in range(4):
                reward_s = round_rewards.get(s, 0.0)
                reward_s = max(-20.0, min(20.0, reward_s))
                if reward_s == 0.0:
                    continue
                # [FIX] Boucle sur TOUS les indices du siege,
                # pas seulement le dernier
                for idx in round_indices_per_seat[s]:
                    transitions[idx]['reward'] += reward_s
                if not round_indices_per_seat[s]:
                    pending_reward[s] += reward_s

            # [FIX] Reset pour la prochaine manche
            round_indices_per_seat = {s: [] for s in range(4)}

            if result.get('game_done', False):
                # Reward pending (siege sans action dans la derniere manche)
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

    scores = env.game_scores
    order = sorted(range(4), key=lambda p: (-scores[p], p))
    rank_of_seat0 = order.index(0) + 1

    stats['rank_of_seat0'] = rank_of_seat0
    stats['n_honba'] = max_honba

    return transitions, stats