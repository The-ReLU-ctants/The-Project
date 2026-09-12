"""Download the full WildlifeReID-10k dataset (~26GB) for training.

Requires a Kaggle account with an API token at ~/.kaggle/kaggle.json
(https://www.kaggle.com/settings -> API -> Create New Token).

Usage:
    pip install kaggle
    python scripts/download_full_dataset.py
"""
import subprocess

DATASET = "wildlifedatasets/wildlifereid-10k"
DEST = "data/raw"

if __name__ == "__main__":
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", DEST, "--unzip"],
        check=True,
    )
