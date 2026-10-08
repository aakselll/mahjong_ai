"""
Entrainement supervise du Global Reward Predictor (GRP) sur des logs humains.

[SUIVI DE SUPHX] Dans le papier Suphx (Section 3.2), le reward predictor Phi
est entraine en SL sur des logs de joueurs humains 10-dan (100M+ manches),
puis FIGE pendant tout le RL. C'est ce qui lui permet d'apprendre les
strategies contextuelles des humains ("fold quand 1er avec avance", "attaque
quand 4e") -- patterns qu'un GRP entraine sur du self-play stEREotype ne peut
pas capturer.

Ce script :
    1. Parse les fichiers .mjai.json.zst du dataset NikkeTryHard
    2. Extrait pour chaque partie les features par manche par joueur
       (identiques a training/game_env.py:_build_features)
    3. Calcule le rang final de chaque joueur
    4. Entraine le RewardPredictor GRU en SL (MSE sur la cible rank reward)
    5. Sauvegarde un checkpoint chargeable avec --warm-start-predictor

Usage :
    python -m mahjong_ai.agents.train_grp_sl --data-dir data/tenhou/2024 --max-files 50000 --epochs 10
    python -m mahjong_ai.agents.train_grp_sl --data-dir data/tenhou/2024 --max-files 5000 --epochs 5 --output grp_sl.pt

Puis dans training/train.py :
    --warm-start-predictor grp_sl.pt --no-predictor-retrain
"""
import argparse
import glob
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mahjong_ai.agents.reward_predictor import RewardPredictor, FEATURE_DIM
from mahjong_ai.sl.mjai_parser import parse_mjai_zst_file


# ====================================================================== #
# Constantes (identiques a training/game_env.py)
# ====================================================================== #
N_ROUNDS_REF = 4             # denominateur de progression (comme game_env.py)
REWARD_SCALE = 1.0 / 1000.0  # points Tenhou -> unites reward
RANK_REWARDS = {1: 15.0, 2: 5.0, 3: -5.0, 4: -15.0}
MAX_ROUNDS = 16
MAX_PHYSICAL_ROUNDS = 8           # plafond de padding (au-dela, on tronque)


# ====================================================================== #
# Extraction des parties depuis les evenements MJAI
# ====================================================================== #
def _rank_from_scores(scores):
    """Rang 1-4 pour chaque joueur (1 = plus haut score). Egalites tranchees
    par index de siege (comme game_env._rank_rewards_from_scores)."""
    order = sorted(range(4), key=lambda p: (-scores[p], p))
    ranks = [0] * 4
    for rank, player in enumerate(order, start=1):
        ranks[player] = rank
    return ranks


def extract_games_from_events(events):
    """Extrait les features par manche pour les 4 joueurs d'une partie.

    Reproduit exactement la logique de training/game_env.py:step() pour
    construire le vecteur x^k de chaque manche.

    Retourne une liste de tuples (features_par_joueur, reward_final_par_joueur)
    ou chaque element est :
        features_par_joueur[p] : np.array (T_p, 8) -- T_p = nb de manches
        reward_final_par_joueur[p] : float (rank reward final)

    Retourne None si la partie est invalide (pas assez de manches, etc.).
    """
    # --- Etat courant ---
    scores = [25000.0] * 4           # points Tenhou absolus
    current_riichi = [False] * 4
    oya = 0
    round_idx = 0                    # 0-indexe (1ere manche = 0)
    n_rounds = 0

    # --- Accumulateur de features par joueur ---
    feats_by_player = [[] for _ in range(4)]

    def _emit_round(deltas, dealer_seat):
        """Construit les features pour la manche qui vient de se terminer."""
        nonlocal scores, current_riichi, round_idx
        for p in range(4):
            scores[p] += deltas[p]
        scores_arr = np.array(scores, dtype=np.float32)
        ranks = _rank_from_scores(scores_arr)
        leader = float(scores_arr.max())
        last = float(scores_arr.min())

        for p in range(4):
            feat = np.array([
                deltas[p] * REWARD_SCALE,                      # 0: score manche
                scores[p] * REWARD_SCALE,                      # 1: score cumule
                1.0 if p == dealer_seat else 0.0,              # 2: dealer
                min(round_idx + 1, N_ROUNDS_REF) / float(N_ROUNDS_REF),  # 3: progression
                1.0 if current_riichi[p] else 0.0,             # 4: riichi
                ranks[p] / 4.0,                                # 5: rang
                (leader - scores[p]) * REWARD_SCALE,           # 6: gap leader
                (scores[p] - last) * REWARD_SCALE,             # 7: gap last
            ], dtype=np.float32)
            feats_by_player[p].append(feat)
        round_idx += 1

    # --- Parcours des evenements ---
    for ev in events:
        ev_type = ev.get("type")

        if ev_type == "start_game":
            scores = [25000.0] * 4
            continue

        if ev_type == "start_kyoku":
            if n_rounds >= MAX_PHYSICAL_ROUNDS:
                break   # on a deja 4 rounds, on ignore la suite (South)
            s = ev.get("scores")
            if isinstance(s, list) and len(s) == 4:
                scores = [float(x) for x in s]
            oya = int(ev.get("oya", 0) or 0)
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
                # Certains fichiers n'ont pas de deltas sur ryukyoku : on met 0
                deltas = [0] * 4
            _emit_round([float(x) for x in deltas], oya)
            continue

    if n_rounds < 4:
        return None
    if any(len(f) == 0 for f in feats_by_player):
        return None

    # --- Reward final de chaque joueur (basé sur le rang final) ---
    final_ranks = _rank_from_scores([float(x) for x in scores])
    rewards_final = [RANK_REWARDS[final_ranks[p]] for p in range(4)]

    return feats_by_player, rewards_final


# ====================================================================== #
# Dataset PyTorch
# ====================================================================== #
class GRPDataset(Dataset):
    """Dataset de (features, reward_final) pour UN joueur d'UNE partie.

    Pour chaque partie, on cree 4 echantillons (un par joueur), car chaque
    joueur a sa propre perspective de la meme partie (features differentes
    car les rangs / dealers / riichi different selon le joueur).
    """

    def __init__(self, data_dir, max_files=None, shuffle_files=True,
                 cache_path=None, max_rounds=MAX_ROUNDS):
        self.max_rounds = max_rounds
        self.samples = []   # liste de (features (T, 8), reward float)

        # --- Cache : evite de reparser a chaque run ---
        if cache_path and os.path.isfile(cache_path):
            print(f"[dataset] chargement du cache : {cache_path}")
            t0 = time.time()
            data = np.load(cache_path, allow_pickle=True).item()
            self.samples = data["samples"]
            print(f"[dataset] {len(self.samples)} echantillons charges en "
                  f"{time.time()-t0:.1f}s")
            return

        # --- Parse des fichiers ---
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
        n_rounds_total = 0

        for i, path in enumerate(files):
            if (i + 1) % max(1, len(files) // 10) == 0:
                print(f"  ... {i+1}/{len(files)} fichiers  "
                      f"({n_games} parties, {len(self.samples)} samples) "
                      f"[{time.time()-t0:.0f}s]")
            try:
                events = parse_mjai_zst_file(path)
                if not events:
                    n_failed += 1
                    continue
                result = extract_games_from_events(events)
                if result is None:
                    n_failed += 1
                    continue
                feats_by_player, rewards_final = result

                # Creer 4 echantillons (un par joueur)
                for p in range(4):
                    feats = feats_by_player[p]
                    T = len(feats)
                    if T == 0:
                        continue
                    # Tronque si trop long
                    if T > self.max_rounds:
                        feats = feats[:self.max_rounds]
                        T = self.max_rounds
                    feat_arr = np.stack(feats, axis=0).astype(np.float32)
                    self.samples.append((feat_arr, float(rewards_final[p])))
                n_games += 1
            except Exception:
                n_failed += 1
                continue

        print(f"[dataset] {len(self.samples)} echantillons charges "
              f"en {time.time()-t0:.1f}s "
              f"({n_games} parties, {n_failed} echecs)")

        # --- Sauvegarde du cache ---
        if cache_path:
            print(f"[dataset] sauvegarde du cache -> {cache_path}")
            np.save(cache_path, {"samples": self.samples}, allow_pickle=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        feats, reward = self.samples[i]
        return torch.from_numpy(feats), torch.tensor(reward, dtype=torch.float32)


def make_batch_collate(max_rounds=MAX_ROUNDS):
    """Collate avec padding dynamique + mask pour la loss."""
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

    # --- Dataset ---
    ds = GRPDataset(
        data_dir=args.data_dir,
        max_files=args.max_files,
        cache_path=args.cache_path,
    )
    if len(ds) == 0:
        raise RuntimeError("Aucun echantillon extrait.")

    # --- Split train / val (10 % de val) ---
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

    # --- Modele ---
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

    def evaluate(loader):
        model.eval()
        total_loss, total_n = 0.0, 0.0
        with torch.no_grad():
            for feats, rewards, masks in loader:
                feats = feats.to(device)
                rewards = rewards.to(device)
                masks = masks.to(device)
                preds = model(feats)                          # (B, T)
                r_exp = rewards.unsqueeze(1).expand_as(preds) # (B, T)
                sq = (preds - r_exp) ** 2 * masks             # zero sur padding
                total_loss += sq.sum().item()
                total_n += masks.sum().item()
        model.train()
        return total_loss / max(total_n, 1.0)

    # --- Boucle ---
    print(f"[grp] entrainement sur {args.epochs} epoques...")
    best_val = float("inf")
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
            sq = (preds - r_exp) ** 2 * masks
            loss = sq.sum() / masks.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += sq.sum().item()
            total_n += masks.sum().item()
            n_batches += 1

        train_loss = total_loss / max(total_n, 1.0)
        val_loss = evaluate(val_loader)

        # Metrique R^2 (sur la variance des rank_rewards)
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


# ====================================================================== #
# Main
# ====================================================================== #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True,
                   help="dossier racine des fichiers .mjai.json.zst")
    p.add_argument("--max-files", type=int, default=None,
                   help="limiter le nombre de fichiers a parser")
    p.add_argument("--cache-path", type=str, default="grp_dataset_cache.npy",
                   help="cache .npy des echantillons (evite le re-parse)")
    p.add_argument("--output", type=str, default="grp_sl.pt",
                   help="chemin de sauvegarde du modele")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val-fraction", type=float, default=0.1)
    args = p.parse_args()

    train_grp(args)


if __name__ == "__main__":
    main()