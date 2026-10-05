"""Data prep step 1: walk data/raw_subset/<species>/... and build a flat manifest
of (path, species, identity) for every image, regardless of each source
dataset's own folder layout.

Rule: species = top-level folder under data/raw_subset; identity = the image's
immediate parent folder name (holds across all 5 sources: DogFaceNet, PolarBearVidID,
SeaStarReID2023, NyalaData, LionData).

Usage:
    python scripts/build_manifest.py
"""
import os

import pandas as pd

ROOT = "data/raw_subset"
OUT = "data/processed/manifest.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPECIES = ["dog", "polar_bear", "sea_star", "nyala", "lion"]

if __name__ == "__main__":
    rows = []
    for species in SPECIES:
        species_root = os.path.join(ROOT, species)
        for dirpath, _, filenames in os.walk(species_root):
            identity = os.path.basename(dirpath)
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue
                rows.append(
                    {
                        "path": os.path.join(dirpath, filename),
                        "species": species,
                        "identity": f"{species}_{identity}",
                    }
                )
        count = sum(r["species"] == species for r in rows)
        print(f"{species}: {count} images")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nTotal: {len(df)} images -> {OUT}")
    print(df["species"].value_counts())
