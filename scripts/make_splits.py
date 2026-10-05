"""Data prep step 2: cap per-class image counts (polar_bear is ~138k near-duplicate
video frames vs. 0.7k-8k for the other 4 species) and produce a train/val/test split
grouped by identity, so the same individual animal never appears in more than one
split (prevents leakage from near-duplicate photos/frames of the same animal).

Usage:
    python scripts/make_splits.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

MANIFEST = "data/processed/manifest.csv"
OUT_DIR = "data/processed"
MAX_PER_CLASS = 2500
RANDOM_STATE = 42


def cap_species(df_species: pd.DataFrame, max_total: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample up to max_total images, spreading the cap across identities first
    so we don't just keep images from a handful of individuals."""
    if len(df_species) <= max_total:
        return df_species

    identities = df_species["identity"].unique()
    rng.shuffle(identities)
    per_identity_cap = max(1, max_total // len(identities))

    picked = []
    for identity in identities:
        sub = df_species[df_species["identity"] == identity]
        n = min(len(sub), per_identity_cap)
        picked.append(sub.sample(n, random_state=RANDOM_STATE))
    picked_df = pd.concat(picked)

    if len(picked_df) > max_total:
        picked_df = picked_df.sample(max_total, random_state=RANDOM_STATE)
    elif len(picked_df) < max_total:
        remaining = df_species.drop(picked_df.index)
        extra_n = min(len(remaining), max_total - len(picked_df))
        picked_df = pd.concat([picked_df, remaining.sample(extra_n, random_state=RANDOM_STATE)])
    return picked_df


def group_split(df_species: pd.DataFrame):
    """70/15/15 split grouped by identity so no individual leaks across splits."""
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=RANDOM_STATE)
    train_idx, rest_idx = next(gss.split(df_species, groups=df_species["identity"]))
    train_df = df_species.iloc[train_idx]
    rest_df = df_species.iloc[rest_idx]

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss2.split(rest_df, groups=rest_df["identity"]))
    val_df = rest_df.iloc[val_idx]
    test_df = rest_df.iloc[test_idx]
    return train_df, val_df, test_df


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_STATE)
    manifest = pd.read_csv(MANIFEST)

    capped = []
    for species, df_species in manifest.groupby("species"):
        before = len(df_species)
        df_capped = cap_species(df_species, MAX_PER_CLASS, rng)
        print(f"{species}: {before} -> {len(df_capped)} images, {df_species['identity'].nunique()} identities")
        capped.append(df_capped)
    capped_df = pd.concat(capped)

    train_parts, val_parts, test_parts = [], [], []
    for species, df_species in capped_df.groupby("species"):
        train_df, val_df, test_df = group_split(df_species)
        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)

    train_df = pd.concat(train_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print("\n=== Split sizes ===")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(df)} total")
        print(df["species"].value_counts().to_string())
        print()
