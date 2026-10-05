"""Data prep step 3: freeze the species -> integer label mapping to disk, so the
training code, evaluation code, and inference server all agree on the same indices.

Usage:
    python scripts/build_label_map.py
"""
import json

import pandas as pd

TRAIN_CSV = "data/processed/train.csv"
OUT = "data/processed/label_map.json"

if __name__ == "__main__":
    species = sorted(pd.read_csv(TRAIN_CSV)["species"].unique())
    label_map = {name: idx for idx, name in enumerate(species)}
    with open(OUT, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"{label_map} -> {OUT}")
