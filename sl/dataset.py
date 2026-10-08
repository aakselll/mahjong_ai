"""
Dataset SL partage entre les 5 tetes Suphx, avec cache incremental par shards.

[FIX 2025-XX-3] Systeme de shards pour reprise sur crash :
  - Le parse est decoupe en shards de SHARD_SIZE fichiers chacun.
  - Chaque shard est ecrit atomiquement (.tmp puis os.replace).
  - Si crash, on reprend au dernier shard ecrit (pas de re-parse).
  - Signature des fichiers verifiee au resume (evite les incoherences
    si data-dir / max-files change).

[FIX 2025-XX-4] Migration automatique de l'ancien samples_all.pkl
vers le format shards (transparent, aucune action utilisateur).

[FIX 2025-XX-5] Log time-based pendant le parse : affichage au moins
toutes les LOG_INTERVAL_S secondes, plus un log immediat a chaque shard
ecrit. Evite les longues plages de silence.

Layout du cache :
  cache/sl_2015/
  +-- _shards/
  |   +-- shard_000000_000100.pkl   (fichiers 0..99)
  |   +-- shard_000100_000200.pkl   (fichiers 100..199)
  |   +-- ...
  +-- _meta.pkl                     (signature pour valider le resume)
  +-- samples_all.pkl               (legacy, migre auto)

Le filtrage par tete se fait pendant le chargement : la memoire au
chargement est bornee a 1 shard + les samples filtres.
"""
import glob
import hashlib
import os
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

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

SHARD_SIZE = 100        # nombre de fichiers par shard
SIG_SAMPLE = 1000       # nombre de fichiers utilises pour la signature
LOG_INTERVAL_S = 20     # intervalle max entre deux logs de progression


# ====================================================================== #
# Helpers top-level (picklables pour multiprocessing Windows)
# ====================================================================== #
def _parse_one_file(task):
    """Parse UN fichier -> (file_id, samples_list_or_None)."""
    file_id, path = task
    try:
        events = parse_mjai_zst_file(path)
        if not events:
            return file_id, []
        out = []
        for s in events_to_samples_all_povs(events):
            obs, atype, action, _mask = s
            obs_u8 = np.clip(
                np.round(np.asarray(obs, dtype=np.float32) * 255.0),
                0, 255,
            ).astype(np.uint8)
            out.append((obs_u8, int(atype), int(action), file_id))
        return file_id, out
    except Exception:
        return file_id, None


def _format_duration(seconds):
    if seconds < 0 or seconds != seconds:
        return '?'
    if seconds < 60:
        return f'{seconds:.1f}s'
    if seconds < 3600:
        return f'{int(seconds//60)}m{int(seconds%60):02d}s'
    return f'{int(seconds//3600)}h{int((seconds%3600)//60):02d}m'


def _files_signature(files):
    """Hash deterministe sur les SIG_SAMPLE premiers basenames."""
    h = hashlib.md5()
    for f in files[:SIG_SAMPLE]:
        h.update(os.path.basename(f).encode())
        h.update(b'\n')
    return h.hexdigest()


def discover_files(data_dir, max_files=None, seed=42):
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.zst'), recursive=True))
    if not files:
        raise RuntimeError(f'Aucun .zst trouve dans {data_dir}')
    random.Random(seed).shuffle(files)
    if max_files:
        files = files[:max_files]
    return files


def _has_shards(shard_dir):
    if not os.path.isdir(shard_dir):
        return False
    return any(f.startswith('shard_') and f.endswith('.pkl')
               for f in os.listdir(shard_dir))


def _scan_last_file_id(shard_dir):
    """Renvoie l'ID exclusif du dernier fichier parse (0 si aucun shard)."""
    if not os.path.isdir(shard_dir):
        return 0
    shards = sorted(glob.glob(os.path.join(shard_dir, 'shard_*.pkl')))
    if not shards:
        return 0
    name = os.path.basename(shards[-1]).replace('.pkl', '')
    # format : shard_XXXXXX_YYYYYY.pkl, YYYYYY = exclusif
    return int(name.split('_')[2])


def _count_shards(shard_dir):
    if not os.path.isdir(shard_dir):
        return 0
    return len(glob.glob(os.path.join(shard_dir, 'shard_*.pkl')))


def _migrate_pkl_to_shards(pkl_path, shard_dir, shard_size=SHARD_SIZE, verbose=True):
    """Convertit un samples_all.pkl legacy en shards. Supprime le pkl."""
    if verbose:
        print(f'[migrate] chargement {pkl_path}...', flush=True)
    t0 = time.time()
    with open(pkl_path, 'rb') as f:
        samples = pickle.load(f)
    if verbose:
        print(f'[migrate] {len(samples)} samples charges en '
              f'{time.time()-t0:.1f}s', flush=True)

    by_fid = {}
    for s in samples:
        by_fid.setdefault(s[3], []).append(s)
    fids = sorted(by_fid.keys())
    if verbose:
        print(f'[migrate] {len(fids)} file_ids distincts '
              f'({fids[0]}..{fids[-1]})', flush=True)

    os.makedirs(shard_dir, exist_ok=True)
    for i in range(0, len(fids), shard_size):
        chunk = fids[i:i+shard_size]
        start, end = chunk[0], chunk[-1] + 1
        shard_samples = []
        for fid in chunk:
            shard_samples.extend(by_fid[fid])
        name = f'shard_{start:06d}_{end:06d}.pkl'
        path = os.path.join(shard_dir, name)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(shard_samples, f, protocol=4)
        os.replace(tmp, path)
        if verbose:
            print(f'  [migrate] {name}: {len(shard_samples)} samples',
                  flush=True)

    os.remove(pkl_path)
    if verbose:
        print(f'[migrate] termine, {pkl_path} supprime', flush=True)


def _load_all_shards_filtered(shard_dir, action_type, verbose=False,
                                head_type='?'):
    """Charge tous les shards, filtre par action_type, renvoie la liste."""
    shard_files = sorted(glob.glob(os.path.join(shard_dir, 'shard_*.pkl')))
    out = []
    total_raw = 0
    for i, sf in enumerate(shard_files):
        with open(sf, 'rb') as f:
            shard = pickle.load(f)
        total_raw += len(shard)
        for obs_u8, atype, act, fid in shard:
            if atype == action_type:
                out.append((obs_u8, act, fid))
        if verbose and (i + 1) % 20 == 0:
            print(f'  [load/{head_type}] {i+1}/{len(shard_files)} shards '
                  f'charges, {len(out)} samples filtres', flush=True)
    if verbose:
        print(f'  [load/{head_type}] {len(shard_files)} shards, '
              f'{total_raw} samples bruts, {len(out)} apres filtre',
              flush=True)
    return out


# ====================================================================== #
# Dataset
# ====================================================================== #
class PolicyDataset(Dataset):
    """Echantillons (obs (N_CHANNELS, 34) uint8, label, file_id) pour UNE tete.

    Cache partage entre les 5 tetes : contient tous les types d'action.
    Filtre en memoire au chargement.
    """

    def __init__(self, files, head_type, cache_path=None, verbose=True,
                 parse_workers=1, shard_size=SHARD_SIZE):
        assert head_type in HEAD_TO_ACTION_TYPE, f'Tete inconnue : {head_type}'
        self.head_type = head_type
        self.action_type = HEAD_TO_ACTION_TYPE[head_type]
        self._parse_workers = max(1, int(parse_workers))
        self._shard_size = max(1, int(shard_size))

        if cache_path is None:
            # Pas de cache -> parse en memoire sans shards
            self.samples = self._parse_no_cache(files, verbose)
            return

        cache_dir = os.path.dirname(cache_path) or '.'
        shard_dir = os.path.join(cache_dir, '_shards')
        meta_path = os.path.join(cache_dir, '_meta.pkl')
        os.makedirs(cache_dir, exist_ok=True)

        # 1) Migration auto si pkl legacy et pas de shards
        if os.path.isfile(cache_path) and not _has_shards(shard_dir):
            if verbose:
                print(f'[dataset/{head_type}] migration legacy pkl -> shards',
                      flush=True)
            _migrate_pkl_to_shards(cache_path, shard_dir,
                                    shard_size=self._shard_size, verbose=verbose)

        # 2) Etat actuel
        n_files_total = len(files)
        start_file_id = _scan_last_file_id(shard_dir)

        if verbose:
            n_shards = _count_shards(shard_dir)
            if n_shards:
                print(f'[dataset/{head_type}] etat cache : '
                      f'{n_shards} shards, {start_file_id}/{n_files_total} '
                      f'fichiers parses', flush=True)
            else:
                print(f'[dataset/{head_type}] cache vide, parse de '
                      f'{n_files_total} fichiers', flush=True)

        # 3) Parse le reste si necessaire
        if start_file_id < n_files_total:
            self._parse_range(files, start_file_id, n_files_total,
                              shard_dir, meta_path, verbose)
        elif start_file_id > n_files_total:
            if verbose:
                print(f'[dataset/{head_type}] ATTENTION : {start_file_id} '
                      f'fichiers deja parses mais seulement {n_files_total} '
                      f'demandes (le cache contient plus de donnees que '
                      f'necessaire, on charge tout)', flush=True)

        # 4) Chargement filtre
        if verbose:
            t0 = time.time()
            print(f'[dataset/{head_type}] chargement + filtre des shards...',
                  flush=True)
        self.samples = _load_all_shards_filtered(
            shard_dir, self.action_type, verbose=verbose,
            head_type=head_type,
        )
        if verbose:
            print(f'[dataset/{head_type}] {len(self.samples)} echantillons '
                  f'apres filtre en {time.time()-t0:.1f}s', flush=True)

    # ------------------------------------------------------------------ #
    def _parse_no_cache(self, files, verbose):
        """Parse sans cache (mode in-memory)."""
        if verbose:
            print(f'[dataset/{self.head_type}] parse sans cache '
                  f'({self._parse_workers} worker(s))...', flush=True)
        t0 = time.time()
        tasks = list(enumerate(files))
        out = []
        if self._parse_workers <= 1:
            for task in tasks:
                _, samples = _parse_one_file(task)
                if samples:
                    for obs_u8, atype, act, fid in samples:
                        if atype == self.action_type:
                            out.append((obs_u8, act, fid))
        else:
            import multiprocessing as mp
            with mp.Pool(processes=self._parse_workers) as pool:
                for _, samples in pool.imap(_parse_one_file, tasks, chunksize=1):
                    if samples:
                        for obs_u8, atype, act, fid in samples:
                            if atype == self.action_type:
                                out.append((obs_u8, act, fid))
        if verbose:
            print(f'[dataset/{self.head_type}] {len(out)} samples '
                  f'en {_format_duration(time.time()-t0)}', flush=True)
        return out

    # ------------------------------------------------------------------ #
    def _parse_range(self, files, start_id, end_id, shard_dir, meta_path, verbose):
        """Parse files[start_id:end_id] et ecrit les shards."""
        n = end_id - start_id
        if n <= 0:
            return

        # Verification de la signature
        sig = _files_signature(files)
        os.makedirs(shard_dir, exist_ok=True)
        if os.path.isfile(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            if meta.get('signature') != sig:
                raise RuntimeError(
                    f'Signature mismatch !\n'
                    f'  Cache cree pour signature : {meta.get("signature")}\n'
                    f'  Fichiers actuels signature : {sig}\n'
                    f'Le data-dir ou les fichiers ont change. Supprimez '
                    f'{shard_dir}/ et {meta_path} pour reparcourir, ou '
                    f'revenez a la config d\'origine.'
                )
        else:
            with open(meta_path, 'wb') as f:
                pickle.dump({'signature': sig, 'created': time.time()}, f)

        if verbose:
            print(f'[dataset/{self.head_type}] parsing {n} fichiers '
                  f'({start_id}..{end_id-1}), {self._parse_workers} worker(s), '
                  f'shard_size={self._shard_size}...', flush=True)

        t_start = time.time()
        tasks = [(fid, files[fid]) for fid in range(start_id, end_id)]

        # Buffers de shard (mutables via dict)
        state = {
            'buffer': [],
            'start': start_id,
            'end': start_id,
            'files_in_shard': 0,
            'done': 0,
        }
        log_every = max(1, min(200, n // 100))
        last_log_time = [time.time()]

        def flush_shard():
            if not state['buffer']:
                return
            name = f"shard_{state['start']:06d}_{state['end']:06d}.pkl"
            path = os.path.join(shard_dir, name)
            tmp = path + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump(state['buffer'], f, protocol=4)
            os.replace(tmp, path)
            if verbose:
                print(f"  [shard] {name} ({len(state['buffer'])} samples, "
                      f"{state['files_in_shard']} fichiers)", flush=True)
            # Reset + log immediat
            last_log_time[0] = time.time()
            state['buffer'] = []
            state['start'] = state['end']
            state['files_in_shard'] = 0

        def process(fid, samples):
            state['done'] += 1
            if samples:
                state['buffer'].extend(samples)
            state['end'] = max(state['end'], fid + 1)
            state['files_in_shard'] += 1
            if state['files_in_shard'] >= self._shard_size:
                flush_shard()

        def log_progress():
            if not verbose:
                return
            now = time.time()
            should_log = (
                state['done'] % log_every == 0
                or (now - last_log_time[0]) >= LOG_INTERVAL_S
                or state['done'] == n
            )
            if not should_log:
                return
            last_log_time[0] = now
            elapsed = now - t_start
            rate = state['done'] / max(elapsed, 1e-6)
            eta = (n - state['done']) / max(rate, 1e-6)
            print(f"  [{state['done']}/{n}] "
                  f"elapsed={_format_duration(elapsed)}  "
                  f"ETA={_format_duration(eta)}  "
                  f"{rate:.3f} f/s  "
                  f"buffer={len(state['buffer'])}", flush=True)

        if self._parse_workers <= 1:
            for fid, path in tasks:
                _fid, samples = _parse_one_file((fid, path))
                process(_fid, samples)
                log_progress()
        else:
            import multiprocessing as mp
            with mp.Pool(processes=self._parse_workers) as pool:
                # imap (ordonne) : necessaire pour frontiere de shards propre
                for fid, samples in pool.imap(_parse_one_file, tasks, chunksize=1):
                    process(fid, samples)
                    log_progress()

        flush_shard()
        if verbose:
            print(f'[dataset/{self.head_type}] parse termine : '
                  f'{state["done"]} fichiers en '
                  f'{_format_duration(time.time()-t_start)}', flush=True)

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        obs_u8, label, _file_id = self.samples[i]
        obs_f32 = obs_u8.astype(np.float32) / 255.0
        return torch.from_numpy(obs_f32), torch.tensor(label, dtype=torch.long)

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