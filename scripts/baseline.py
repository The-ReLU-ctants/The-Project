"""Non-deep-learning baseline for the species classifier: a majority-class
floor plus a Logistic Regression on small flattened pixel features, trained
and evaluated on the same 5-species splits as the ViT so the numbers are
directly comparable in the "Results" section.

Run from anywhere:  python scripts/baseline.py

Reports overall accuracy, per-class precision / recall / F1, and a confusion
matrix for the Logistic Regression baseline (plus the majority-class floor
accuracy), printed and written to reports/ (prefixed baseline_ so nothing
here overwrites evaluate.py's ViT outputs):

    reports/baseline_metrics.json
    reports/baseline_classification_report.csv
    reports/baseline_confusion_matrix.csv
    reports/baseline_confusion_matrix.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # manifest CSVs store image paths relative to the project root

from src.dataset import load_label_map  # noqa: E402

# Tiny and flattened on purpose: this baseline exists to give the ViT a
# floor to beat, not to be competitive on its own.
IMAGE_SIZE = 32


def load_features(csv_path: str, label_map: dict):
    df = pd.read_csv(csv_path)
    X = np.zeros((len(df), IMAGE_SIZE * IMAGE_SIZE * 3), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.int64)
    for i, row in enumerate(df.itertuples(index=False)):
        image = Image.open(row.path.replace("\\", "/")).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        X[i] = (np.asarray(image, dtype=np.float32) / 255.0).ravel()
        y[i] = label_map[row.species]
    return X, y


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: Path, title: str):
    row_sums = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)

    n = len(class_names)
    size = max(5.0, 0.9 * n + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.9), dpi=200)
    im = ax.imshow(share, cmap="Oranges", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", color="#222222")
    ax.set_yticklabels(class_names, color="#222222")
    ax.set_xlabel("Predicted species", color="#222222")
    ax.set_ylabel("True species", color="#222222")
    ax.set_title(title, color="#222222", pad=12)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", default="data/processed/train.csv")
    p.add_argument("--test-csv", default="data/processed/test.csv")
    p.add_argument("--label-map", default="data/processed/label_map.json")
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--model-out", default="models/baseline_logreg.joblib")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(args.label_map)
    idx_to_species = {v: k for k, v in label_map.items()}
    labels = list(range(len(label_map)))
    class_names = [idx_to_species[i] for i in labels]

    print(f"Loading train features ({IMAGE_SIZE}x{IMAGE_SIZE} flattened RGB)...")
    X_train, y_train = load_features(args.train_csv, label_map)
    print("Loading test features...")
    X_test, y_test = load_features(args.test_csv, label_map)
    print(f"Train: {len(X_train)} images, Test: {len(X_test)} images\n")

    # ---- Floor: majority-class baseline ------------------------------------
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    acc_dummy = accuracy_score(y_test, dummy.predict(X_test))
    print(f"Majority-class floor accuracy: {acc_dummy:.4f}\n")

    # ---- Real baseline: Logistic Regression on raw flattened pixels -------
    print("Training Logistic Regression baseline...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=labels, target_names=class_names,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"\nLogistic Regression test accuracy: {acc:.4f}  ({(y_test == y_pred).sum()}/{len(y_test)} correct)\n")
    print(classification_report(y_test, y_pred, labels=labels, target_names=class_names,
                                zero_division=0, digits=3))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())

    # ---- Save outputs -------------------------------------------------------
    pd.DataFrame(report).T.rename_axis("class").to_csv(out_dir / "baseline_classification_report.csv")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / "baseline_confusion_matrix.csv")

    with open(out_dir / "baseline_metrics.json", "w") as f:
        json.dump(
            {
                "model": f"LogisticRegression (flattened {IMAGE_SIZE}x{IMAGE_SIZE} RGB pixels)",
                "image_size": IMAGE_SIZE,
                "num_train_images": int(len(y_train)),
                "num_test_images": int(len(y_test)),
                "majority_class_accuracy": acc_dummy,
                "accuracy": acc,
                "macro_avg": report["macro avg"],
                "weighted_avg": report["weighted avg"],
                "per_class": {c: report[c] for c in class_names},
            },
            f,
            indent=2,
        )

    plot_confusion_matrix(cm, class_names, out_dir / "baseline_confusion_matrix.png",
                          f"Baseline confusion matrix (accuracy {acc:.1%})")
    print(f"\nSaved results to {out_dir}/")

    # ---- Save the fitted model so the inference server can load it without
    # retraining on every request --------------------------------------------
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "image_size": IMAGE_SIZE,
            "label_map": label_map,
            "test_accuracy": acc,
        },
        model_out,
    )
    print(f"Saved model to {model_out}")


if __name__ == "__main__":
    main()
