"""Create a tiny fake dataset with the same layout as the real State Farm data.

Lets you test every script on your Mac in seconds, before touching Kaggle.

    python scripts/make_fake_dataset.py --out fake_data
    python src/prepare_data.py --data-root fake_data --out-dir data_fake --results-dir results_fake
    python src/data.py data_fake
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="fake_data")
    p.add_argument("--drivers", type=int, default=26)
    p.add_argument("--per-class", type=int, default=3, help="images per driver per class")
    args = p.parse_args()

    out = Path(args.out)
    train = out / "imgs" / "train"
    rng = np.random.default_rng(0)
    rows = ["subject,classname,img"]
    for d in range(args.drivers):
        subject = f"p{d:03d}"
        for c in range(10):
            cls = f"c{c}"
            (train / cls).mkdir(parents=True, exist_ok=True)
            for k in range(args.per_class):
                name = f"img_{d}_{c}_{k}.jpg"
                arr = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
                Image.fromarray(arr).save(train / cls / name, quality=70)
                rows.append(f"{subject},{cls},{name}")
    (out / "driver_imgs_list.csv").write_text("\n".join(rows) + "\n")
    print(f"Wrote {len(rows) - 1} fake images for {args.drivers} drivers to {out}")


if __name__ == "__main__":
    main()
