"""
Dataset SL pour un modele Suphx.

Les obs produites par mjai_parser sont deja en forme (N_CHANNELS, 34).
Split train/val PAR FICHIER (pas par echantillon) pour eviter la fuite.
"""
import glob
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import time
from mahjong_ai.sl.mjai_parser import parse_mjai_zst_file, events_to_samples_all_povs
from mahjong_ai.envs.multi_agent import DISCARD, PONG, CHOW, KONG, RIICHI
from mahjong_ai.envs.suphx_features import N_CHANNELS


HEAD_TO_ACTION_TYPE = {
    'discard': DISCARD,
    'riichi': RIICHI,
    'chow': CHOW,
    'pong': PONG,
    'kong': KONG,
}


def discover_files(data_dir, max_files=None, seed=42):
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.zst'), recursive=True))
    if not files:
        raise RuntimeError(f'Aucun .zst trouve dans {data_dir}')
    random.Random(seed).shuffle(files)
    if max_files:
        files = files[:max_files]
    return files

def _format_duration(seconds):
    """Formate une duree en '1h23m', '5m12s', ou '12.3s'."""
    if seconds < 0 or seconds != seconds:  # NaN ou negatif
        return '?'
    if seconds < 60:
        return f'{seconds:.1f}s'
    if seconds < 3600:
        return f'{int(seconds//60)}m{int(seconds%60):02d}s'
    return f'{int(seconds//3600)}h{int((seconds%3600)//60):02d}m'

class PolicyDataset(Dataset):
    """Echantillons (obs (N_CHANNELS, 34), label, file_id) pour UNE tete."""

    def __init__(self, files, head_type, cache_path=None, verbose=True):
        assert head_type in HEAD_TO_ACTION_TYPE, f'Tete inconnue : {head_type}'
        self.head_type = head_type
        self.action_type = HEAD_TO_ACTION_TYPE[head_type]
        self.samples = []

        if cache_path and os.path.isfile(cache_path):
            if verbose:
                print(f'[dataset/{head_type}] cache charge : {cache_path}')
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
            if verbose:
                print(f'[dataset/{head_type}] {len(self.samples)} echantillons')
            return

        n_files = len(files)
        if verbose:
            print(f'[dataset/{head_type}] parsing {n_files} fichiers...', flush=True)

        t_start = time.time()
        # Petit rappel ETA tous les N fichiers (en plus du log par fichier)
        eta_every = max(1, min(50, n_files // 20))
        recent_times = []
        recent_window = 20
        skipped = 0

        for file_id, path in enumerate(files):

            fname = os.path.basename(path)
            if verbose:
                print(f'  [{file_id+1:5d}/{n_files}] {fname} ', end='', flush=True)

            t_file0 = time.time()
            n_this = 0
            failed = False
            try:
                events = parse_mjai_zst_file(path)
                if events:
                    for s in events_to_samples_all_povs(events):
                        obs, atype, action, _mask = s
                        if atype != self.action_type:
                            continue
                        self.samples.append((obs, int(action), file_id))
                        n_this += 1
            except Exception as e:
                failed = True
                if verbose:
                    print(f'FAILED ({type(e).__name__}: {e})', flush=True)
                skipped += 1

            if failed:
                continue

            dt_file = time.time() - t_file0
            recent_times.append(dt_file)
            if len(recent_times) > recent_window:
                recent_times.pop(0)

            if verbose:
                print(f'-> {n_this:5d} samples  {dt_file*1000:6.0f}ms', flush=True)

            # Bilan ETA tous les eta_every fichiers
            if verbose and (file_id + 1) % eta_every == 0:
                now = time.time()
                elapsed = now - t_start
                files_done = file_id + 1
                files_left = n_files - files_done
                rate_recent = (len(recent_times)
                                / max(sum(recent_times), 1e-6))
                rate_global = files_done / max(elapsed, 1e-6)
                rate = rate_recent if len(recent_times) >= 5 else rate_global
                eta_sec = files_left / max(rate, 1e-6)
                print(
                    f'  ---> [{files_done}/{n_files}] '
                    f'elapsed={_format_duration(elapsed)}  '
                    f'ETA={_format_duration(eta_sec)}  '
                    f'{rate:.2f} f/s  '
                    f'total={len(self.samples)}  '
                    f'skipped={skipped}',
                    flush=True,
                )

        elapsed_total = time.time() - t_start
        if verbose:
            print(
                f'[dataset/{head_type}] total {len(self.samples)} echantillons '
                f'en {_format_duration(elapsed_total)} '
                f'({n_files/elapsed_total:.2f} fichiers/s moyens, '
                f'{skipped} echecs)',
                flush=True,
            )

        if verbose:
            print(f'[dataset/{head_type}] total {len(self.samples)} echantillons')

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f, protocol=4)
            if verbose:
                print(f'[dataset/{head_type}] cache ecrit : {cache_path}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        obs, label, _file_id = self.samples[i]
        return torch.from_numpy(np.ascontiguousarray(obs)), \
               torch.tensor(label, dtype=torch.long)

    def split_by_file(self, val_fraction=0.1, seed=42):
        file_ids = sorted({s[2] for s in self.samples})
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(file_ids))
        n_val = max(int(len(file_ids) * val_fraction), 1)
        val_files = set(np.array(file_ids)[perm[:n_val]].tolist())

        train_idx, val_idx = [], []
        for i, (_, _, fid) in enumerate(self.samples):
            (val_idx if fid in val_files else train_idx).append(i)
        return train_idx, val_idx

    def make_subsets(self, val_fraction=0.1, seed=42):
        tr, va = self.split_by_file(val_fraction, seed)
        return Subset(self, tr), Subset(self, va)

