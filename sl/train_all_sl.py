"""
Orchestre l'entrainement des 5 modeles Suphx SL (discard/riichi/chow/pong/kong).

Chaque tete est entrainee sequentiellement en lancant train_policy_sl.py en
sous-processus. Les tetes partagent le meme cache de parsing (un cache par
tete, cf. train_policy_sl.py), donc relancer ce script apres un premier run
complet ne re-parse pas les fichiers.

Usage :
    python -m mahjong_ai.sl.train_all_sl ^
        --data-dir data/tenhou/2015 ^
        --output-dir checkpoints/sl_2015_run1 ^
        --cache-dir cache/sl_2015 ^
        --max-files 1000 --epochs 2 --batch-size 128 --num-workers 0

Options :
    --heads discard riichi       pour n'entrainer que certaines tetes
    --skip-existing              ne pas re-entrainer une tete dont le
                                 checkpoint existe deja
"""
import argparse
import os
import subprocess
import sys

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
        '--channels', str(args.channels),
        '--n-blocks', str(args.n_blocks),
        '--binary-hidden', str(args.binary_hidden),
    ]
    if args.cache_dir:
        cmd += ['--cache-dir', args.cache_dir]
    if args.max_files:
        cmd += ['--max-files', str(args.max_files)]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True,
                   help='dossier racine des fichiers .mjai.json.zst')
    p.add_argument('--output-dir', required=True,
                   help='dossier des checkpoints .pt par tete')
    p.add_argument('--cache-dir', default=None,
                   help='dossier des caches pickle (1 par tete) ; '
                        'fortement recommande (sinon re-parse a chaque run)')
    p.add_argument('--max-files', type=int, default=None,
                   help='limiter le nombre de fichiers a parser')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=0,
                   help='0 obligatoire sur Windows au premier run')
    # Architecture
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--n-blocks', type=int, default=16)
    p.add_argument('--binary-hidden', type=int, default=256)
    # Selection
    p.add_argument('--heads', nargs='+', default=HEADS,
                   choices=HEADS,
                   help='sous-ensemble de tetes a entrainer')
    p.add_argument('--skip-existing', action='store_true',
                   help='ne pas re-entrainer si le .pt existe deja')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    results = {}
    for head in args.heads:
        output_path = os.path.join(args.output_dir, f'{head}.pt')

        if args.skip_existing and os.path.isfile(output_path):
            print(f'\n========== {head} : checkpoint existant, skip ==========')
            results[head] = 'skipped'
            continue

        print(f'\n========== Training head: {head} ==========')
        cmd = build_cmd(args, head, output_path)
        print('CMD :', ' '.join(cmd))
        t0 = __import__('time').time()
        try:
            subprocess.run(cmd, check=True)
            results[head] = 'ok'
        except subprocess.CalledProcessError as e:
            print(f'[train_all_sl] {head} a echoue (code {e.returncode})')
            results[head] = f'failed (code {e.returncode})'
        dt = __import__('time').time() - t0
        print(f'========== {head} termine en {dt:.1f}s ==========')

    print('\n===== Resume =====')
    for head, status in results.items():
        print(f'  {head:10s} : {status}')
    print(f'\nCheckpoints dans : {args.output_dir}')


if __name__ == '__main__':
    main()