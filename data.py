"""Dataset, transforms and dataloaders.

Reads the splits.csv + pre-resized images written by prepare_data.py.

    from data import build_dataloaders
    loaders = build_dataloaders("data", scheme="driver")   # or "random"
    loaders["train"], loaders["val"], loaders["test"]
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CLASS_NAMES = [
    "safe driving",
    "texting - right",
    "phone call - right",
    "texting - left",
    "phone call - left",
    "operating radio",
    "drinking",
    "reaching behind",
    "hair and makeup",
    "talking to passenger",
]
PHONE_CLASSES = [1, 2, 3, 4]  # safety-critical: any phone use
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DriverDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.paths = [img_dir / p for p in df["relpath"]]
        self.labels = df["label"].astype(int).tolist()
        self.subjects = df["subject"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def build_transforms(train: bool, img_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if train:
        return T.Compose([
            # NO RandomHorizontalFlip: flipping turns "texting - right" (c1) into
            # what looks like "texting - left" (c3) without changing the label.
            T.RandomRotation(10),
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def load_splits(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "splits.csv")


def build_dataloaders(
    data_dir: str | Path,
    scheme: str = "driver",          # "driver" (honest) or "random" (leaky)
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
    resized_size: int = 256,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    limit: int | None = None,        # cap images per split for quick debug runs
    seed: int = 42,
) -> dict:
    if scheme not in ("driver", "random"):
        raise ValueError("scheme must be 'driver' or 'random'")
    data_dir = Path(data_dir)
    img_dir = data_dir / f"imgs_{resized_size}"
    df = load_splits(data_dir)
    col = f"{scheme}_split"

    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df[col] == split]
        if limit:
            part = part.sample(n=min(limit, len(part)), random_state=seed)
        ds = DriverDataset(part, img_dir, build_transforms(split == "train", img_size, mean, std))
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            drop_last=(split == "train"),
        )
    return loaders


if __name__ == "__main__":
    # Quick smoke test: python src/data.py data
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    for scheme in ("driver", "random"):
        loaders = build_dataloaders(data_dir, scheme=scheme, batch_size=8, num_workers=0)
        sizes = {k: len(v.dataset) for k, v in loaders.items()}
        x, y = next(iter(loaders["train"]))
        print(f"{scheme:6s} sizes={sizes} batch={tuple(x.shape)} labels={y.tolist()}")
