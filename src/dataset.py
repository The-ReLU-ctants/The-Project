"""Image transform + Dataset pipeline: turns the manifest CSVs produced by
scripts/build_manifest.py and scripts/make_splits.py into tensor-ready
(image, label) pairs for a DataLoader. Does not define or train any model.
"""
import json

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_label_map(path: str = "data/processed/label_map.json") -> dict:
    with open(path) as f:
        return json.load(f)


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """Augmented transforms for training; deterministic resize+normalize for
    val/test so evaluation numbers are stable and comparable across runs."""
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class WildlifeDataset(Dataset):
    """Reads a manifest CSV (path, species, identity) and yields (image_tensor, label_idx)."""

    def __init__(self, csv_path: str, label_map: dict, image_size: int = 224, split: str = "train"):
        self.df = pd.read_csv(csv_path)
        self.label_map = label_map
        self.transform = get_transforms(split, image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        ## image = Image.open(row["path"]).convert("RGB") [windows path]
        image = Image.open(row["path"].replace("\\", "/")).convert("RGB") ## [macOS path]
        image = self.transform(image)
        label = self.label_map[row["species"]]
        return image, label
