"""Evaluate the trained ViT on the test split.

Run from anywhere:  python scripts/evaluate.py

Reports overall accuracy, per-class precision / recall / F1, and a confusion
matrix. Everything is printed and also written to reports/:

    reports/metrics.json              accuracy, macro / weighted averages, per-class scores
    reports/classification_report.csv per-class precision, recall, F1, support
    reports/confusion_matrix.csv      rows = true species, columns = predicted
    reports/confusion_matrix.png      figure for the slides / paper
    reports/predictions.csv           one row per test image (path, true, predicted, confidence)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # manifest CSVs store image paths relative to the project root

from src.dataset import WildlifeDataset  # noqa: E402
from src.model import build_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-csv", default="data/processed/test.csv")
    p.add_argument("--checkpoint", default="models/vit_species.pt")
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device):
    """Rebuild the model from the config stored in the checkpoint and load its weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model, loader, device):
    """Return (true labels, predicted labels, confidence of the predicted class)."""
    y_true, y_pred, conf = [], [], []
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1).cpu()
        c, pred = probs.max(dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist())
        conf.extend(c.tolist())
    return np.array(y_true), np.array(y_pred), np.array(conf)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: Path, title: str):
    """Confusion matrix as a single-hue heatmap: colour = share of each true species
    (row-normalised, so rare species stay readable), cell text = raw image count."""
    row_sums = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)

    n = len(class_names)
    size = max(5.0, 0.9 * n + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.9), dpi=200)
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", color="#222222")
    ax.set_yticklabels(class_names, color="#222222")
    ax.set_xlabel("Predicted species", color="#222222")
    ax.set_ylabel("True species", color="#222222")
    ax.set_title(title, color="#222222", pad=12)

    # Hairline gaps between cells instead of heavy gridlines.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Count in every cell; white text on dark cells, dark text on light cells.
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "#222222", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Share of true species", color="#222222")
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(args.checkpoint, device)
    label_map = ckpt["label_map"]
    idx_to_species = {v: k for k, v in label_map.items()}
    labels = list(range(len(label_map)))
    class_names = [idx_to_species[i] for i in labels]
    print(f"Device: {device}")
    print(f"Loaded {args.checkpoint} (epoch {ckpt['epoch']}, val acc {ckpt['val_acc']:.4f})")

    # Use the label map saved with the model so class order always matches training.
    test_ds = WildlifeDataset(args.test_csv, label_map, ckpt["config"]["image_size"], split="test")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    print(f"Test set: {len(test_ds)} images\n")

    y_true, y_pred, conf = predict(model, loader, device)

    # ---- Metrics -----------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"Test accuracy: {acc:.4f}  ({(y_true == y_pred).sum()}/{len(y_true)} correct)\n")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                zero_division=0, digits=3))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())

    # ---- Save outputs ------------------------------------------------------
    pd.DataFrame(report).T.rename_axis("class").to_csv(out_dir / "classification_report.csv")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / "confusion_matrix.csv")

    pred_df = test_ds.df.copy()
    pred_df["true_species"] = [idx_to_species[i] for i in y_true]
    pred_df["pred_species"] = [idx_to_species[i] for i in y_pred]
    pred_df["confidence"] = conf
    pred_df["correct"] = y_true == y_pred
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": ckpt["epoch"],
                "num_test_images": int(len(y_true)),
                "accuracy": acc,
                "macro_avg": report["macro avg"],
                "weighted_avg": report["weighted avg"],
                "per_class": {c: report[c] for c in class_names},
            },
            f,
            indent=2,
        )

    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png",
                          f"Test confusion matrix (accuracy {acc:.1%})")
    print(f"\nSaved results to {out_dir}/")


if __name__ == "__main__":
    main()