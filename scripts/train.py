"""Train the from-scratch ViT species classifier.

Run from anywhere:  python scripts/train.py
(The script switches to the project root, because the manifest CSVs store
image paths relative to it.)

Outputs
    models/vit_species.pt        best checkpoint by validation accuracy, a dict:
                                 {model_state, config, label_map, epoch, val_acc}
    models/train_history.json    per-epoch train/val loss and accuracy (for plots)
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `from src...` works regardless of cwd
os.chdir(ROOT)  # manifest CSV paths are relative to the project root

from src.dataset import WildlifeDataset, load_label_map  # noqa: E402
from src.model import build_model, count_parameters  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", default="data/processed/train.csv")
    p.add_argument("--val-csv", default="data/processed/val.csv")
    p.add_argument("--label-map", default="data/processed/label_map.json")
    p.add_argument("--out", default="models/vit_species.pt")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=0,
                   help="Stop if val accuracy hasn't improved for N epochs (0 = off).")
    p.add_argument("--class-weights", action="store_true",
                   help="Use inverse-frequency class weights in CrossEntropyLoss.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    # Doesn't make weight init non-random, makes the *sequence* of randomness
    # (weight init, shuffling) reproducible across runs with the same seed.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                # Gradients accumulate by default, so they must be cleared
                # before each new batch or last batch's gradients leak in.
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    # Mixed precision (CUDA only): scale the loss up before
                    # backward() so small float16 gradients don't underflow
                    # to zero, then unscale before clipping/stepping so both
                    # see true gradient magnitudes.
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # Clip to guard against the large, destabilising updates
                    # a from-scratch (no pretrained prior) model is prone to.
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()  # per-batch, not per-epoch, for a smooth warmup/decay curve

            bs = labels.size(0)
            total_loss += loss.item() * bs  # un-average back to a sum so the final mean is exact, not an average-of-averages
            correct += (logits.argmax(1) == labels).sum().item()
            n += bs

    return total_loss / n, correct / n


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    label_map = load_label_map(args.label_map)
    num_classes = len(label_map)
    idx_to_species = {v: k for k, v in label_map.items()}
    print(f"Classes ({num_classes}): {label_map}")

    train_ds = WildlifeDataset(args.train_csv, label_map, args.image_size, split="train")
    val_ds = WildlifeDataset(args.val_csv, label_map, args.image_size, split="val")
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # Class distribution (useful for the report, and to decide on --class-weights).
    counts = (
        pd.Series([label_map[s] for s in train_ds.df["species"]])
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
    )
    for i, c in counts.items():
        print(f"  {idx_to_species[i]:<20} {c:>7} train images")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    model = build_model(num_classes=num_classes, image_size=args.image_size).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    weight = None
    if args.class_weights:
        # Inverse-frequency weighting: rarer species get a higher loss weight
        # so the model isn't incentivised to just favour common classes.
        w = counts.sum() / (num_classes * counts.clip(lower=1))
        weight = torch.tensor(w.values, dtype=torch.float32, device=device)
        print(f"Class weights: {np.round(w.values, 3).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Linear warmup then cosine decay, stepped per batch.
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        # Returns a multiplier on the base LR, not an absolute value.
        if step < warmup_steps:
            # Ramp 0 -> 1 linearly so the model doesn't take big steps while
            # its random weights are still essentially noise.
            return (step + 1) / max(1, warmup_steps)
        # Then cosine-decay 1 -> 0 so training settles into a minimum
        # smoothly instead of stopping abruptly.
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_acc, best_epoch, stale = -1.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, lr=optimizer.param_groups[0]["lr"]))

        # Only overwrite the checkpoint on a new best, so the saved file is
        # always the best epoch seen, not just the most recent one (a model
        # can start overfitting and get worse on val in later epochs).
        improved = va_acc > best_acc
        if improved:
            best_acc, best_epoch, stale = va_acc, epoch, 0
            torch.save(
                {"model_state": model.state_dict(), "config": model.config,
                 "label_map": label_map, "epoch": epoch, "val_acc": va_acc},
                out_path,
            )
        else:
            stale += 1

        print(f"Epoch {epoch:>3}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.0f}s"
              f"{'  * saved' if improved else ''}")

        if args.patience and stale >= args.patience:
            print(f"Early stopping: no val improvement for {args.patience} epochs.")
            break

    with open(out_path.parent / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val accuracy {best_acc:.4f} at epoch {best_epoch}. Saved to {out_path}")


if __name__ == "__main__":
    main()