"""
Entrainement supervise du Global Reward Predictor (GRP) sur des logs humains.

[SUIVI DE SUPHX] GRU 2 couches + 2 FC, entraine par MSE sur le rank reward.

[FORMATS] Chaque partie hanchan produit deux vues: une vue hanchan complete
et une vue tonpuu limitee a l'Est. Le format est encode dans chaque feature,
afin que la meme GRU apprenne deux horizons et deux cibles distinctes.

[FIX 2025-XX-2] Loss ponderee : les rewards extremes (±15) pesent 3x plus
que les rewards moderes (±5).

Usage :
    python -m mahjong_ai.agents.train_grp_sl --data-dir data/tenhou/2015 --max-files 100000 --epochs 15
"""
import argparse
import glob
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mahjong_ai.agents.reward_predictor import RewardPredictor, FEATURE_DIM
from mahjong_ai.sl.mjai_parser import parse_mjai_zst_file


# ====================================================================== #
# Constantes (identiques a training/game_env.py)
# ====================================================================== #
N_ROUNDS_REF = 8             # reference hanchan; tonpuu utilise 4
REWARD_SCALE = 1.0 / 1000.0
RANK_REWARDS = {1: 15.0, 2: 5.0, 3: -5.0, 4: -15.0}
MAX_ROUNDS = 12              # E1-S4 + W1-W4 extension
MAX_PHYSICAL_ROUNDS = 12     # preserve observable extension rounds

EXTREME_REWARD_THRESHOLD = 10.0
EXTREME_REWARD_WEIGHT = 3.0


# ====================================================================== #
# Extraction
# ====================================================================== #
def _rank_from_scores(scores):
    order = sorted(range(4), key=lambda p: (-scores[p], p))
    ranks = [0] * 4
    for rank, player in enumerate(order, start=1):
        ranks[player] = rank
    return ranks


def extract_games_from_events(events, game_format="hanchan"):
    """Extrait les features par manche pour les 4 joueurs.

    Le format determine l'horizon et la cible. Une vue tonpuu d'un hanchan
    s'arrete a l'entree en Sud; une vue hanchan conserve les manches Sud.
    """
    if game_format not in ("hanchan", "tonpuu"):
        raise ValueError(f"format inconnu: {game_format}")
    normal_rounds = 4 if game_format == "tonpuu" else 8
    format_flags = (1.0, 0.0) if game_format == "tonpuu" else (0.0, 1.0)
    scores = [25000.0] * 4
    current_riichi = [False] * 4
    oya = 0
    round_idx = 0
    n_rounds = 0
    normal_round_index = 0
    is_last_normal_round = False
    is_extension = False
    honba = 0
    kyotaku = 0
    feats_by_player = [[] for _ in range(4)]

    def _emit_round(deltas, dealer_seat):
        nonlocal scores, current_riichi, round_idx
        for p in range(4):
            scores[p] += deltas[p]
        scores_arr = np.array(scores, dtype=np.float32)
        ranks = _rank_from_scores(scores_arr)
        leader = float(scores_arr.max())
        last = float(scores_arr.min())

        for p in range(4):
            feat = np.array([
                deltas[p] * REWARD_SCALE,
                scores[p] * REWARD_SCALE,
                1.0 if p == dealer_seat else 0.0,
                min(round_idx + 1, normal_rounds)
                / float(normal_rounds),
                1.0 if current_riichi[p] else 0.0,
                ranks[p] / 4.0,
                (leader - scores[p]) * REWARD_SCALE,
                (scores[p] - last) * REWARD_SCALE,
                format_flags[0],
                format_flags[1],
                1.0 if is_last_normal_round else 0.0,
                1.0 if is_extension else 0.0,
                min(max(honba, 0), 4) / 4.0,
                min(max(kyotaku, 0), 4) / 4.0,
                max(normal_rounds - normal_round_index, 0)
                / float(normal_rounds),
            ], dtype=np.float32)
            feats_by_player[p].append(feat)
        round_idx += 1

    for ev in events:
        ev_type = ev.get("type")

        if ev_type == "start_game":
            scores = [25000.0] * 4
            continue

        if ev_type == "start_kyoku":
            if game_format == "tonpuu" and ev.get("bakaze") != "E":
                break
            if n_rounds >= MAX_PHYSICAL_ROUNDS:
                break
            s = ev.get("scores")
            if isinstance(s, list) and len(s) == 4:
                scores = [float(x) for x in s]
            oya = int(ev.get("oya", 0) or 0)
            kyoku = int(ev.get("kyoku", 1) or 1)
            if ev.get("bakaze") == "E":
                normal_round_index = kyoku
            elif ev.get("bakaze") == "S":
                normal_round_index = 4 + kyoku
            else:
                normal_round_index = 8 + kyoku
            is_last_normal_round = normal_round_index == normal_rounds
            is_extension = normal_round_index > normal_rounds
            honba = int(ev.get("honba", 0) or 0)
            kyotaku = int(ev.get("kyotaku", 0) or 0)
            current_riichi = [False] * 4
            n_rounds += 1
            continue

        if ev_type == "reach":
            actor = ev.get("actor")
            if actor is not None and 0 <= actor < 4:
                current_riichi[actor] = True
            continue

        if ev_type in ("hora", "ryukyoku"):
            deltas = ev.get("deltas")
            if not (isinstance(deltas, list) and len(deltas) == 4):
                deltas = [0] * 4
            _emit_round([float(x) for x in deltas], oya)
            continue

    if round_idx == 0:
        return None
    if game_format == "tonpuu" and round_idx < 4:
        return None
    if any(len(f) == 0 for f in feats_by_player):
        return None

    final_ranks = _rank_from_scores([float(x) for x in scores])
    rewards_final = [RANK_REWARDS[final_ranks[p]] for p in range(4)]

    return feats_by_player, rewards_final


# ====================================================================== #
# Dataset PyTorch
# ====================================================================== #
class GRPDataset(Dataset):
    def __init__(self, data_dir, max_files=None, shuffle_files=True,
                 cache_path=None, max_rounds=MAX_ROUNDS):
        self.max_rounds = max_rounds
        self.samples = []

        if cache_path and os.path.isfile(cache_path):
            print(f"[dataset] chargement du cache : {cache_path}")
            t0 = time.time()
            data = np.load(cache_path, allow_pickle=True).item()
            if data.get("feature_dim") != FEATURE_DIM:
                raise RuntimeError(
                    f"Cache GRP incompatible: "
                    f"feature_dim={data.get('feature_dim')}, "
                    f"attendu={FEATURE_DIM}. Utilisez un nouveau "
                    "cache-path."
                )
            self.samples = data["samples"]
            print(f"[dataset] {len(self.samples)} echantillons charges en "
                  f"{time.time()-t0:.1f}s")
            return

        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.zst"),
                                  recursive=True))
        print(f"[dataset] {len(files)} fichiers trouves dans {data_dir}")
        if not files:
            raise RuntimeError(f"Aucun fichier .zst dans {data_dir}")

        if shuffle_files:
            random.Random(42).shuffle(files)
        if max_files:
            files = files[:max_files]

        print(f"[dataset] parsing de {len(files)} fichiers...")
        t0 = time.time()
        n_games = 0
        n_failed = 0
        n_skipped_incomplete = 0

        for i, path in enumerate(files):
            if (i + 1) % max(1, len(files) // 10) == 0:
                print(f"  ... {i+1}/{len(files)} fichiers  "
                      f"({n_games} parties, {len(self.samples)} samples, "
                      f"{n_skipped_incomplete} parties incompletes) "
                      f"[{time.time()-t0:.0f}s]")
            try:
                events = parse_mjai_zst_file(path)
                if not events:
                    n_failed += 1
                    continue
                hanchan_result = extract_games_from_events(
                    events, game_format="hanchan")
                tonpuu_result = extract_games_from_events(
                    events, game_format="tonpuu")
                if hanchan_result is None and tonpuu_result is None:
                    n_skipped_incomplete += 1
                    n_failed += 1
                    continue
                for result in (hanchan_result, tonpuu_result):
                    if result is None:
                        continue
                    feats_by_player, rewards_final = result
                    for p in range(4):
                        feats = feats_by_player[p]
                        T = len(feats)
                        if T == 0:
                            continue
                        if T > self.max_rounds:
                            feats = feats[:self.max_rounds]
                        feat_arr = np.stack(feats, axis=0).astype(np.float32)
                        self.samples.append(
                            (feat_arr, float(rewards_final[p])))
                n_games += 1
            except Exception:
                n_failed += 1
                continue

        print(f"[dataset] {len(self.samples)} echantillons charges "
              f"en {time.time()-t0:.1f}s "
              f"({n_games} parties, "
              f"{n_skipped_incomplete} parties incompletes, "
              f"{n_failed} echecs)")

        if cache_path:
            print(f"[dataset] sauvegarde du cache -> {cache_path}")
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.save(cache_path, {
                "feature_dim": FEATURE_DIM,
                "samples": self.samples,
            }, allow_pickle=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        feats, reward = self.samples[i]
        return torch.from_numpy(feats), torch.tensor(reward, dtype=torch.float32)


def make_batch_collate(max_rounds=MAX_ROUNDS):
    def collate(batch):
        B = len(batch)
        T_max = max(f.shape[0] for f, _ in batch)
        T_max = min(T_max, max_rounds)

        features = torch.zeros(B, T_max, FEATURE_DIM, dtype=torch.float32)
        rewards = torch.zeros(B, dtype=torch.float32)
        masks = torch.zeros(B, T_max, dtype=torch.float32)

        for i, (f, r) in enumerate(batch):
            T = min(f.shape[0], T_max)
            features[i, :T] = f[:T]
            rewards[i] = r
            masks[i, :T] = 1.0
        return features, rewards, masks
    return collate


# ====================================================================== #
# Boucle d'entrainement
# ====================================================================== #
def train_grp(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[grp] device = {device}")

    ds = GRPDataset(
        data_dir=args.data_dir,
        max_files=args.max_files,
        cache_path=args.cache_path,
    )
    if len(ds) == 0:
        raise RuntimeError("Aucun echantillon extrait.")

    n_total = len(ds)
    n_val = max(int(n_total * args.val_fraction), 1)
    indices = np.random.RandomState(42).permutation(n_total)
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    print(f"[grp] split : {len(train_idx)} train / {len(val_idx)} val")

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    collate = make_batch_collate(max_rounds=MAX_ROUNDS)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, collate_fn=collate)

    model = RewardPredictor(
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[grp] modele : {n_params/1e6:.3f}M parametres "
          f"(hidden={args.hidden}, layers={args.layers}, dropout={args.dropout})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * max(len(train_loader), 1))

    def _compute_weight(rewards, preds):
        w = 1.0 + (EXTREME_REWARD_WEIGHT - 1.0) * \
            (rewards.abs() > EXTREME_REWARD_THRESHOLD).float()
        return w.unsqueeze(1).expand_as(preds)

    def evaluate(loader):
        model.eval()
        total_loss, total_n = 0.0, 0.0
        with torch.no_grad():
            for feats, rewards, masks in loader:
                feats = feats.to(device)
                rewards = rewards.to(device)
                masks = masks.to(device)
                preds = model(feats)
                r_exp = rewards.unsqueeze(1).expand_as(preds)
                weight = _compute_weight(rewards, preds)
                sq = (preds - r_exp) ** 2 * masks * weight
                total_loss += sq.sum().item()
                total_n += (masks * weight).sum().item()
        model.train()
        return total_loss / max(total_n, 1.0)

    print(f"[grp] entrainement sur {args.epochs} epoques...")
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        total_loss, total_n = 0.0, 0.0
        n_batches = 0

        for feats, rewards, masks in train_loader:
            feats = feats.to(device)
            rewards = rewards.to(device)
            masks = masks.to(device)

            preds = model(feats)
            r_exp = rewards.unsqueeze(1).expand_as(preds)

            weight = _compute_weight(rewards, preds)
            sq = (preds - r_exp) ** 2 * masks * weight
            loss = sq.sum() / (masks * weight).sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += sq.sum().item()
            total_n += (masks * weight).sum().item()
            n_batches += 1

        train_loss = total_loss / max(total_n, 1.0)
        val_loss = evaluate(val_loader)

        var_rank = float(np.var(list(RANK_REWARDS.values())))
        r2 = 1.0 - val_loss / var_rank

        dt = time.time() - t0
        print(f"[epoch {epoch+1:2d}/{args.epochs}] "
              f"train_loss={train_loss:7.4f}  "
              f"val_loss={val_loss:7.4f}  "
              f"R²={r2:5.3f}  "
              f"({dt:.1f}s, {n_batches} batches)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.output)
            print(f"    -> nouveau meilleur modele sauvegarde (val_loss={val_loss:.4f})")

    print(f"[grp] entrainement termine. Meilleur val_loss = {best_val:.4f}")
    print(f"[grp] checkpoint : {args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--cache-path", type=str, default="grp_dataset_cache.npy")
    p.add_argument("--output", type=str, default="grp_sl.pt")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val-fraction", type=float, default=0.1)
    args = p.parse_args()

    train_grp(args)


if __name__ == "__main__":
    main()