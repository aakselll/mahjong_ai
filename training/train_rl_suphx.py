"""
Boucle principale d'entrainement RL Suphx.

[FIDELITE PAPIER §3.3] Oracle guiding en 3 phases :
    Phase 1 (oracle)     : gamma_t = 1 constant, LR normal
    Phase 2 (dropout)    : gamma_t : 1 -> 0 lineaire, LR normal
    Phase 3 (normal)     : gamma_t = 0, LR / 10, IW rejection

[FIX Windows spawn] Les workers recoivent UNIQUEMENT les proxies
Manager (dict, lock, list) - pas les objets ParameterServer/ReplayBuffer
qui contiennent le Manager (non picklable).

[STATS] Ajout d'un stats_proxy partage pour monitorer :
    - win rate (rank de seat 0 sur N parties)
    - distribution 1st/2nd/3rd/4th
    - longueur moyenne des parties (rounds + honba)
    - distribution des fins de manche (tsumo/ron/exh/abort)

Usage :
    py -3.14 -m mahjong_ai.training.train_rl_suphx ^
        --sl-discard checkpoints/sl_2015_run1/discard.pt ^
        --sl-frozen checkpoints/sl_2015_run1 ^
        --grp-checkpoint checkpoints/grp_sl_east4.pt ^
        --total-steps 100000 --n-workers 4 ^
        --output checkpoints/rl_discard.pt
"""
import argparse
import multiprocessing as mp
import os
import time
import collections

import numpy as np
import torch

from mahjong_ai.agents.oracle_agent import OracleDiscardAgent
from mahjong_ai.training.parameter_server import (
    ParameterServer, publish_to_state,
)
from mahjong_ai.training.replay_buffer import ReplayBuffer
from mahjong_ai.training.self_play_worker import worker_loop


# ---------------------------------------------------------------------- #
# Stats helpers
# ---------------------------------------------------------------------- #
STATS_KEYS_INIT = {
    'n_games': 0,
    'n_1st': 0, 'n_2nd': 0, 'n_3rd': 0, 'n_4th': 0,
    'sum_rank': 0.0,
    'n_rounds_total': 0,
    'n_honba_total': 0,
    'n_tsumo': 0, 'n_ron': 0, 'n_exhaustive': 0,
    'n_abortive': 0,
    'n_riichi': 0,
}


def reset_stats_proxy(stats_proxy):
    """Remet les compteurs a zero (utilise au changement de phase)."""
    for k, v in STATS_KEYS_INIT.items():
        stats_proxy[k] = v


def format_game_stats(stats_proxy):
    """Formate la ligne GAME pour les logs."""
    n_games = stats_proxy.get('n_games', 0)
    if n_games == 0:
        return None
    n1 = stats_proxy.get('n_1st', 0)
    n2 = stats_proxy.get('n_2nd', 0)
    n3 = stats_proxy.get('n_3rd', 0)
    n4 = stats_proxy.get('n_4th', 0)
    mean_rank = stats_proxy.get('sum_rank', 0.0) / n_games
    win_rate = n1 / n_games
    mean_rounds = stats_proxy.get('n_rounds_total', 0) / n_games
    mean_honba = stats_proxy.get('n_honba_total', 0) / n_games
    n_tsumo = stats_proxy.get('n_tsumo', 0)
    n_ron = stats_proxy.get('n_ron', 0)
    n_exh = stats_proxy.get('n_exhaustive', 0)
    n_ab = stats_proxy.get('n_abortive', 0)
    total_ends = n_tsumo + n_ron + n_exh + n_ab
    tsumo_pct = 100 * n_tsumo / max(total_ends, 1)
    ron_pct = 100 * n_ron / max(total_ends, 1)
    exh_pct = 100 * n_exh / max(total_ends, 1)
    ab_pct = 100 * n_ab / max(total_ends, 1)

    return (
        f"n={n_games} "
        f"[{n1}/{n2}/{n3}/{n4}] "
        f"wr={win_rate:.3f} "
        f"mean_rank={mean_rank:.2f} "
        f"rounds={mean_rounds:.2f} "
        f"honba={mean_honba:.2f} "
        f"tsumo%={tsumo_pct:.0f} "
        f"ron%={ron_pct:.0f} "
        f"exh%={exh_pct:.0f} "
        f"ab%={ab_pct:.0f}"
    )


def _purge_old_checkpoints(base_path, keep_n):
    """Supprime les checkpoints _step* en gardant les keep_n plus recents."""
    import glob
    pattern = f"{base_path}_step*.pt"
    files = sorted(glob.glob(pattern))
    if len(files) > keep_n:
        for f in files[:-keep_n]:
            try:
                os.remove(f)
                print(f"[purge] supprime {f}", flush=True)
            except OSError:
                pass


# ---------------------------------------------------------------------- #
# Drift tracking helpers
# ---------------------------------------------------------------------- #
def compute_drift_metrics(batch):
    """Calcule les metriques de suivi de derive depuis un batch."""
    if not batch:
        return None

    actions = [t['action'] for t in batch if t.get('seat') == 0]
    if actions:
        counter = collections.Counter(actions)
        top3 = counter.most_common(3)
        top3_str = " ".join(f"{a}:{c}" for a, c in top3)
        counts = np.array(list(counter.values()), dtype=np.float32)
        probs = counts / counts.sum()
        action_entropy = float(-(probs * np.log(probs + 1e-8)).sum())
        max_freq = float(probs.max())
        action_div = 1.0 - max_freq
    else:
        top3_str = "-"
        action_entropy = 0.0
        action_div = 0.0

    advs = np.array([t['advantage'] for t in batch], dtype=np.float32)
    vals = np.array([t['value'] for t in batch], dtype=np.float32)
    rets = np.array([t['return'] for t in batch], dtype=np.float32)

    return {
        'actions_top3':   top3_str,
        'action_entropy': action_entropy,
        'action_div':     action_div,
        'adv_mean':       float(advs.mean()),
        'adv_std':        float(advs.std()),
        'adv_abs_mean':   float(np.abs(advs).mean()),
        'val_mean':       float(vals.mean()),
        'val_std':        float(vals.std()),
        'ret_mean':       float(rets.mean()),
        'n_valid':        len(batch),
    }


def check_drift_alerts(metrics, phase):
    if metrics is None:
        return []
    alerts = []
    if metrics['action_div'] < 0.2:
        alerts.append(f"COLLAPSE_ACTION (div={metrics['action_div']:.2f})")
    if metrics['action_entropy'] < 0.5:
        alerts.append(f"LOW_ACTION_H ({metrics['action_entropy']:.2f})")
    if metrics['adv_abs_mean'] > 10.0:
        alerts.append(f"BIG_ADV ({metrics['adv_abs_mean']:.1f})")
    if metrics['val_std'] < 0.1 and phase >= 2:
        alerts.append(f"FLAT_VALUE (std={metrics['val_std']:.3f})")
    gap = abs(metrics['val_mean'] - metrics['ret_mean'])
    if gap > 5.0:
        alerts.append(f"VAL_RET_GAP ({gap:.1f})")
    return alerts


# ---------------------------------------------------------------------- #
# Schedule 3 phases
# ---------------------------------------------------------------------- #
def get_phase_and_gamma(step, total_steps,
                         oracle_frac=0.20, transition_frac=0.50):
    p1_end = int(total_steps * oracle_frac)
    p2_end = int(total_steps * (oracle_frac + transition_frac))
    if step < p1_end:
        return 1, 1.0, 1.0
    elif step < p2_end:
        frac = (step - p1_end) / max(p2_end - p1_end, 1)
        return 2, max(0.0, 1.0 - frac), 1.0
    else:
        return 3, 0.0, 0.1


# ---------------------------------------------------------------------- #
# GAE
# ---------------------------------------------------------------------- #
def compute_advantages(transitions, gamma=0.99, lam=0.95):
    gae = 0.0
    next_v = 0.0
    for t in reversed(range(len(transitions))):
        tr = transitions[t]
        if tr['done']:
            next_v = 0.0
            delta = tr['reward'] - tr['value']
            gae = delta
        else:
            delta = tr['reward'] + gamma * next_v - tr['value']
            gae = delta + gamma * lam * gae
        tr['advantage'] = gae
        tr['return'] = gae + tr['value']
        next_v = tr['value']
    advs = np.array([tr['advantage'] for tr in transitions], dtype=np.float32)
    if len(advs) > 1 and advs.std() > 1e-6:
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        
    for tr in transitions:
        tr['return'] = max(-30.0, min(30.0, tr['return']))
    return advs



# ---------------------------------------------------------------------- #
# PG + IS loss
# ---------------------------------------------------------------------- #
def pg_is_loss(agent, transitions, advantages, gamma_t,
                clip_eps=0.2, iw_threshold=None, phase=1):
    obs_n = torch.as_tensor(np.stack([t['obs_normal'] for t in transitions]),
                             dtype=torch.float32, device=agent.device)
    obs_o = torch.as_tensor(np.stack([t['obs_oracle'] for t in transitions]),
                             dtype=torch.float32, device=agent.device)
    masks = torch.as_tensor(np.stack([t['mask'] for t in transitions]),
                             dtype=torch.float32, device=agent.device)
    actions = torch.as_tensor([t['action'] for t in transitions],
                                dtype=torch.long, device=agent.device)
    old_logp = torch.as_tensor([t['logp'] for t in transitions],
                                 dtype=torch.float32, device=agent.device)
    returns = torch.as_tensor([t['return'] for t in transitions],
                                dtype=torch.float32, device=agent.device)
    advs = torch.as_tensor(advantages, dtype=torch.float32, device=agent.device)

    new_logp, values, entropy = agent.evaluate_actions(
        obs_n, obs_o, masks, actions, gamma_t)
    ratio = torch.exp(new_logp - old_logp)

    if phase == 3 and iw_threshold is not None:
        keep = (ratio <= iw_threshold).float()
        n_kept = keep.sum().item()
        if n_kept < 2:
            return None, None, None
        keep = keep / keep.sum() * keep.numel()
    else:
        keep = torch.ones_like(ratio)

    if clip_eps is not None:
        s1 = ratio * advs
        s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs
        policy_loss = -(torch.min(s1, s2) * keep).mean()
    else:
        policy_loss = -(ratio * advs * keep).mean()

    value_loss = ((values - returns) ** 2 * keep).mean()
    entropy_bonus = entropy.mean()
    return policy_loss, value_loss, entropy_bonus


# ---------------------------------------------------------------------- #
# Boucle principale
# ---------------------------------------------------------------------- #
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"[train_rl] GPU detected : {torch.cuda.get_device_name(0)}")
        print(f"[train_rl] VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    os.environ["RL_N_WORKERS"] = str(args.n_workers)

    torch.set_num_threads(2)

    print(f"[train_rl] device={device}, n_workers={args.n_workers}",
          flush=True)
    print(f"[train_rl] schedule 3 phases : "
          f"oracle={args.oracle_frac*100:.0f}%, "
          f"transition={args.transition_frac*100:.0f}%, "
          f"normal={(1-args.oracle_frac-args.transition_frac)*100:.0f}%",
          flush=True)

    # --- Parameter server + buffer ---
    ps = ParameterServer()
    rb = ReplayBuffer(max_size=args.buffer_size, manager=ps.manager)

    # --- [STATS] Proxy partage pour les stats de jeu ---
    stats_proxy = ps.manager.dict()
    reset_stats_proxy(stats_proxy)

    # --- Agent ---
    agent = OracleDiscardAgent(
        sl_discard_path=args.sl_discard,
        sl_frozen_dir=args.sl_frozen,
        device=device,
        policy_lr=args.lr,
        critic_lr=args.critic_lr,
    )

    # --- Resume (optionnel) ---
    start_step = 0
    start_alpha = args.alpha_resume if args.alpha_resume is not None \
                  else args.alpha_init
    if args.resume and os.path.isfile(args.resume):
        print(f"[train_rl] resume depuis {args.resume}", flush=True)
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        try:
            agent.policy.load_state_dict(ck['policy'])
            agent.critic.load_state_dict(ck['critic'])
            start_step = ck.get('step', 0) or 0
            start_alpha = ck.get('alpha', args.alpha_init) or args.alpha_init

            # Extraction depuis le nom si step=0
            if start_step == 0:
                import re
                m = re.search(r'step(\d+)', args.resume)
                if m:
                    start_step = int(m.group(1))
                    print(f"[train_rl]   step extrait du nom : {start_step}",
                          flush=True)

            print(f"[train_rl]   -> step={start_step}, alpha={start_alpha:.4f}",
                  flush=True)
        except Exception as e:
            print(f"[train_rl] ECHEC resume : {e}", flush=True)

    # --- Publish initial ---
    publish_to_state(
        ps.state, ps.lock,
        agent.policy.state_dict(),
        agent.critic.state_dict(),
        step=0, alpha=args.alpha_init, gamma_t=1.0,
    )

    # --- Workers ---
    workers = []
    for i in range(args.n_workers):
        p = mp.Process(
            target=worker_loop,
            args=(
                i,
                ps.state, ps.lock,
                rb.data, args.buffer_size,
                stats_proxy,                        # <-- NOUVEAU
                args.sl_discard, args.sl_frozen,
                args.grp_checkpoint,
                args.games_per_cycle, device
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)
    print(f"[train_rl] {args.n_workers} workers lances", flush=True)

    # --- Boucle ---
    step = start_step
    alpha = start_alpha
    recent_entropies = []
    # Si on resume dans une phase avancee, on demarre dans cette phase
    current_phase, _, _ = get_phase_and_gamma(
        step, args.total_steps, args.oracle_frac, args.transition_frac)
    base_lr = args.lr
    t_start = time.time()
    n_updates = 0

    while step < args.total_steps:
        if len(rb) < args.batch_episodes:
            time.sleep(1.0)
            continue

        batch = rb.sample(args.batch_episodes)
        if batch is None:
            continue
        if len(batch) > args.max_batch_transitions:
            idx = np.random.choice(len(batch), args.max_batch_transitions,
                                    replace=False)
            batch = [batch[i] for i in idx]

        phase, gamma_t, lr_factor = get_phase_and_gamma(
            step, args.total_steps, args.oracle_frac, args.transition_frac)

        # --- Changement de phase ---
        if phase != current_phase:
            print(f"[step {step:7d}] PHASE {current_phase} -> {phase} "
                  f"(gamma={gamma_t:.3f}, lr_factor={lr_factor})",
                  flush=True)
            current_phase = phase

            # Policy LR : base * lr_factor (1.0 en phase 2, 0.1 en phase 3)
            for pg in agent.policy_optim.param_groups:
                pg['lr'] = base_lr * lr_factor

            # Critic LR : on reduit independamment de la policy
            #   phase 1 : 1.0  x critic_lr = 1e-2
            #   phase 2 : 0.5  x critic_lr = 5e-3
            #   phase 3 : 0.1  x critic_lr = 1e-3
            critic_factor = {1: 1.0, 2: 0.5, 3: 0.1}[phase]
            for pg in agent.critic_optim.param_groups:
                pg['lr'] = args.critic_lr * critic_factor

            # [STATS] reset pour mesurer par phase
            reset_stats_proxy(stats_proxy)
            print(f"[step {step}] stats reset pour phase {phase}",
                  flush=True)

        # --- Update ---
        advantages = compute_advantages(batch, gamma=args.gamma, lam=args.lam)

        # Multi-epoch PPO : plusieurs passes de gradient par batch
        PPO_EPOCHS = args.ppo_epochs
        policy_loss_last = None
        value_loss_last = None
        entropy_last = None

        policy_gnorm_last = None
        critic_gnorm_last = None

        for epoch_i in range(PPO_EPOCHS):
            agent.policy_optim.zero_grad()
            agent.critic_optim.zero_grad()
            iw_thresh = args.iw_threshold if phase == 3 else None
            policy_loss, value_loss, entropy = pg_is_loss(
                agent, batch, advantages, gamma_t,
                clip_eps=args.clip_eps, iw_threshold=iw_thresh, phase=phase)
            if policy_loss is None:
                break
            loss = policy_loss + args.value_coef * value_loss - alpha * entropy
            loss.backward()
            # [FIX] capture les grad norms pour le log DRIFT
            policy_gnorm_last = torch.nn.utils.clip_grad_norm_(
                agent.policy.parameters(), 5.0)
            critic_gnorm_last = torch.nn.utils.clip_grad_norm_(
                agent.critic.parameters(), 5.0)
            agent.policy_optim.step()
            agent.critic_optim.step()
            policy_loss_last = policy_loss
            value_loss_last = value_loss
            entropy_last = entropy

        if policy_loss_last is None:
            continue
        policy_loss = policy_loss_last
        value_loss = value_loss_last
        entropy = entropy_last
        policy_gnorm = policy_gnorm_last if policy_gnorm_last is not None \
            else torch.tensor(0.0)
        critic_gnorm = critic_gnorm_last if critic_gnorm_last is not None \
            else torch.tensor(0.0)

        if policy_loss_last is None:
            continue
        policy_loss = policy_loss_last
        value_loss = value_loss_last
        entropy = entropy_last

        step += len(batch)
        n_updates += 1

        # --- Alpha (Eq. 3) ---
        recent_entropies.append(entropy.item())
        if len(recent_entropies) > 50:
            recent_entropies.pop(0)
        if len(recent_entropies) >= 3:
            h_bar = np.mean(recent_entropies)
            alpha = max(0.0, alpha + args.beta * (args.h_target - h_bar))

        # --- Publish ---
        if step % args.publish_every < len(batch):
            publish_to_state(
                ps.state, ps.lock,
                agent.policy.state_dict(),
                agent.critic.state_dict(),
                step=step, alpha=alpha, gamma_t=gamma_t,
            )

        # --- Log + Drift + GAME stats ---
        if step % args.log_every < len(batch):
            elapsed = time.time() - t_start
            metrics = compute_drift_metrics(batch)
            alerts = check_drift_alerts(metrics, phase)

            print(f"[step {step:7d} | phase {phase}] "
                  f"pl={policy_loss.item():+.4f} "
                  f"vl={value_loss.item():.4f} "
                  f"H={entropy.item():.4f} "
                  f"alpha={alpha:.4f} "
                  f"gamma={gamma_t:.3f} "
                  f"buf={len(rb)} "
                  f"({elapsed:.0f}s)", flush=True)

            if metrics is not None:
                print(
                    f"    | DRIFT: "
                    f"act_H={metrics['action_entropy']:.2f} "
                    f"act_div={metrics['action_div']:.2f} "
                    f"adv_abs={metrics['adv_abs_mean']:.2f} "
                    f"adv_std={metrics['adv_std']:.2f} "
                    f"val_std={metrics['val_std']:.2f} "
                    f"val_mean={metrics['val_mean']:+.2f} "
                    f"ret_mean={metrics['ret_mean']:+.2f} "
                    f"| top3=[{metrics['actions_top3']}] "
                    f"pgn={policy_gnorm.item():.2f} "
                    f"cgn={critic_gnorm.item():.2f}",
                    flush=True,
                )

            # [STATS] Ligne GAME
            game_line = format_game_stats(stats_proxy)
            if game_line is not None:
                print(f"    | GAME: {game_line}", flush=True)

            for a in alerts:
                print(f"    | !! {a}", flush=True)


        if step % args.save_every < len(batch):
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            base = args.output[:-3] if args.output.endswith('.pt') else args.output
            step_path = f"{base}_step{step:07d}.pt"
            latest_path = f"{base}_latest.pt"
            agent.save(step_path)
            agent.save(latest_path)
            if args.keep_last_n > 0:
                _purge_old_checkpoints(base, args.keep_last_n)
            print(f"[step {step}] save {step_path} + {latest_path}",
                  flush=True)


    agent.save(args.output, step=step)
    print(f"[train_rl] termine. Checkpoint : {args.output}", flush=True)
    for w in workers:
        w.terminate()
    for w in workers:
        w.join(timeout=2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sl-discard', required=True)
    p.add_argument('--sl-frozen', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--total-steps', type=int, default=1000000)
    p.add_argument('--n-workers', type=int, default=4)
    p.add_argument('--games-per-cycle', type=int, default=10)
    p.add_argument('--buffer-size', type=int, default=500)
    p.add_argument('--batch-episodes', type=int, default=20)
    p.add_argument('--max-batch-transitions', type=int, default=1000)

    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--critic-lr', type=float, default=3e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--clip-eps', type=float, default=0.2)
    p.add_argument('--value-coef', type=float, default=0.5)
    p.add_argument('--h-target', type=float, default=2.0)
    p.add_argument('--alpha-init', type=float, default=0.01)
    p.add_argument('--beta', type=float, default=0.005)

    p.add_argument('--oracle-frac', type=float, default=0.20)
    p.add_argument('--transition-frac', type=float, default=0.50)
    p.add_argument('--grp-checkpoint', type=str,
                   default='checkpoints/grp_sl_east4.pt')

    p.add_argument('--iw-threshold', type=float, default=5.0)
    p.add_argument('--keep-last-n', type=int, default=0)

    p.add_argument('--publish-every', type=int, default=2000)
    p.add_argument('--save-every', type=int, default=50000)
    p.add_argument('--log-every', type=int, default=2000)

    p.add_argument('--resume', type=str, default=None,
                   help='chemin vers un checkpoint _latest.pt pour reprendre')
    p.add_argument('--ppo-epochs', type=int, default=3,
                   help='nombre d\'epochs PPO par batch (defaut 3)')

    p.add_argument('--alpha-resume', type=float, default=None,
                   help='valeur de alpha au resume (defaut: args.alpha_init)')
    
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    mp.freeze_support()
    main()