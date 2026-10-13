"""
Orchestre l'entrainement des 5 modeles Suphx SL.
Chaque tete est entrainee sequentiellement en lancant train_policy_sl.py
en sous-processus. Les tetes partagent le meme cache de parsing (shards
dans /_shards/), donc relancer ce script apres un premier run
complet ne re-parse pas les fichiers.

[MODIF 2026-10-13] Suppression de --game-format (detecte automatiquement).

Usage :
python -m mahjong_ai.sl.train_all_sl ^
    --data-dir data/tenhou ^
    --output-dir checkpoints/sl_v2 ^
    --cache-dir cache/sl_v2 ^
    --max-files 1000 --epochs 2 --batch-size 128 --num-workers 0 ^
    --parse-workers 5
"""
import argparse
import os
import subprocess
import sys
import time

HEADS = ['discard', 'riichi', 'chow', 'pong', 'kong']

def build_cmd(args, head, output_path):
    cmd = [
        sys.executable, '-m', 'mahjong_ai.sl.train_policy_sl',
        '--head', head,
        '--data-dir', args.data_dir,
        '--output', output_path,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--weight-decay', str(args.weight_decay),
        '--val-fraction', str(args.val_fraction),
        '--num-workers', str(args.num_workers),
        '--parse-workers', str(args.parse_workers),
        '--channels', str(args.channels),
        '--n-blocks', str(args.n_blocks),
        '--binary-hidden', str(args.binary_hidden),
    ]
    if args.cache_dir:
        cmd += ['--cache-dir', args.cache_dir]
    if args.max_files:
        cmd += ['--max-files', str(args.max_files)]
    if getattr(args, 'max_samples', None) is not None:
        cmd += ['--max-samples', str(args.max_samples)]
    # Warm-start par tete
    if args.warm_start_dir:
        candidate = os.path.join(args.warm_start_dir, f'{head}.pt')
        if os.path.isfile(candidate):
            cmd += ['--warm-start', candidate]
        else:
            print(f'[train_all_sl] {head} : pas de warm-start '
                  f'({candidate} introuvable)', flush=True)
    return cmd

def main():
    import multiprocessing as mp
    mp.freeze_support()

    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True,
                   help='dossier racine des fichiers .mjai.json.zst '
                   '(peut contenir hanchan ET tonpusen melanges)')
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--max-files', type=int, default=None)
    p.add_argument('--warm-start-dir', type=str, default=None)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--parse-workers', type=int, default=1)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--n-blocks', type=int, default=16)
    p.add_argument('--binary-hidden', type=int, default=256)
    p.add_argument('--heads', nargs='+', default=HEADS, choices=HEADS)
    p.add_argument('--skip-existing', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    for head in args.heads:
        output_path = os.path.join(args.output_dir, f'{head}.pt')
        if args.skip_existing and os.path.isfile(output_path):
            print(f'[train_all_sl] {head} : skip (existe deja)',
                  flush=True)
            continue

        print(f'\n{"="*60}')
        print(f'[train_all_sl] entrainement de : {head}')
        print(f'{"="*60}')

        cmd = build_cmd(args, head, output_path)
        print(f'[train_all_sl] cmd : {" ".join(cmd)}\n', flush=True)

        t0 = time.time()
        result = subprocess.run(cmd)
        dt = time.time() - t0
        print(f'\n[train_all_sl] {head} termine en {dt:.0f}s '
              f'(exit code: {result.returncode})', flush=True)

        if result.returncode != 0:
            print(f'[train_all_sl] ECHEC pour {head} !', flush=True)
            if not args.skip_existing:
                break

if __name__ == '__main__':
    main()