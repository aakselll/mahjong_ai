"""
Entraine UN modele Suphx SL (discard / riichi / chow / pong / kong).

Les obs produites par mjai_parser sont deja en forme (N_CHANNELS=290, 34) --
il n'y a plus de conversion flat->canaux ici.

Usage :
    python -m mahjong_ai.sl.train_policy_sl ^
        --head discard --data-dir data/tenhou/2015 ^
        --cache-dir cache/sl_2015 ^
        --output checkpoints/sl_2015_run1/discard.pt ^
        --max-files 1000 --epochs 2 --batch-size 128 --num-workers 0
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mahjong_ai.sl.dataset import PolicyDataset, discover_files
from mahjong_ai.agents.suphx_cnn import build_model, N_CHANNELS


def make_collate():
    """Empile des (obs (N_CHANNELS, 34), label) en batchs (B, C, 34), (B,)."""
    def collate(batch):
        chs = torch.stack([b[0] for b in batch], dim=0)      # (B, N_CHANNELS, 34)
        labels = torch.stack([b[1] for b in batch], dim=0)   # (B,)
        return chs, labels
    return collate


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for chs, labels in loader:
        chs = chs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(chs)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
    model.train()
    return total_loss / max(n, 1), correct / max(n, 1)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[train/{args.head}] device={device}', flush=True)

    # ---------------- Donnees ----------------
    files = discover_files(args.data_dir, max_files=args.max_files)
    cache = None
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache = os.path.join(args.cache_dir, f'samples_{args.head}.pkl')

    ds = PolicyDataset(files, args.head, cache_path=cache, verbose=True)
    if len(ds) == 0:
        raise RuntimeError(f'Aucun echantillon pour {args.head}')

    train_ds, val_ds = ds.make_subsets(val_fraction=args.val_fraction)
    print(f'[train/{args.head}] train={len(train_ds)}  val={len(val_ds)}', flush=True)

    pin = (device == 'cuda')
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=make_collate(),
        pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=make_collate(),
        pin_memory=pin,
    )

    # ---------------- Modele ----------------
    model = build_model(
        args.head,
        channels=args.channels,
        n_blocks=args.n_blocks,
        binary_hidden=args.binary_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[train/{args.head}] params : {n_params/1e6:.3f}M  '
          f'(channels={args.channels}, blocks={args.n_blocks}, '
          f'in_ch={N_CHANNELS})', flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(args.epochs * len(train_loader), 1))

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, correct, n = 0.0, 0, 0

        for step, (chs, labels) in enumerate(train_loader):
            chs = chs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(chs)
            loss = F.cross_entropy(logits, labels)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            n += labels.size(0)

            if (step + 1) % 100 == 0:
                print(f'  step {step+1:5d}/{len(train_loader)} '
                      f'loss={loss.item():.4f}', flush=True)

        tr_loss = total_loss / max(n, 1)
        tr_acc = correct / max(n, 1)
        va_loss, va_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f'[epoch {epoch+1:2d}/{args.epochs}] '
              f'train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  '
              f'val_loss={va_loss:.4f} val_acc={va_acc:.4f}  ({dt:.1f}s)',
              flush=True)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'head_type': args.head,
                'config': {
                    'channels': args.channels,
                    'n_blocks': args.n_blocks,
                    'binary_hidden': args.binary_hidden,
                    'in_channels': N_CHANNELS,
                },
                'val_acc': va_acc,
                'epoch': epoch + 1,
            }, args.output)
            print(f'   -> save ({args.output}, val_acc={va_acc:.4f})', flush=True)

    print(f'[train/{args.head}] best val_acc = {best_val_acc:.4f}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--head', required=True,
                   choices=['discard', 'riichi', 'chow', 'pong', 'kong'])
    p.add_argument('--data-dir', required=True)
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--max-files', type=int, default=None)
    p.add_argument('--output', required=True)

    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=0)

    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--n-blocks', type=int, default=16)
    p.add_argument('--binary-hidden', type=int, default=256)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()