"""Download a small, disk-budgeted subset of species for ViT training (~4GB total)
instead of the full ~26GB WildlifeReID-10k archive.

Each species comes from its own standalone source (via the wildlife-datasets
library), so nothing requires the full Kaggle download.

Usage:
    pip install wildlife-datasets
    python scripts/download_subset.py
"""
import wildlife_datasets.datasets as d

DEST = "data/raw_subset"

DATASETS = {
    "dog": d.DogFaceNet,
    "polar_bear": d.PolarBearVidID,
    "sea_star": d.SeaStarReID2023,
    "nyala": d.NyalaData,
    "lion": d.LionData,
}

if __name__ == "__main__":
    for species, cls in DATASETS.items():
        root = f"{DEST}/{species}"
        print(f"=== {species} ({cls.__name__}) -> {root} ===")
        cls.get_data(root)
