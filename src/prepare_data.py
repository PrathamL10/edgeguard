"""Phase 1: build the splits, pre-resize the images, and save EDA plots.

What this script does
---------------------
1. Reads driver_imgs_list.csv (columns: subject, classname, img).
2. Holds out N_TEST drivers as a fixed TEST set. Every experiment is scored on
   these unseen drivers.
3. Builds two train/val schemes over the remaining drivers:
     - driver_split: N_VAL whole drivers go to validation (honest).
     - random_split: images shuffled randomly into train/val, stratified by
       class, same val size (leaky: every driver appears in both train and val).
   Same test set for both, so the leakage experiment is a fair comparison:
   the random split's val accuracy will look great, but its test accuracy on
   unseen drivers will not match it. The driver split's val accuracy will.
4. Pre-resizes every training image once (640x480 -> SIZE x SIZE) so the GPU
   is not starved by JPEG decoding during training.
5. Writes splits.csv to the data dir (gitignored) and a small JSON summary +
   plots to results/ (safe to commit: no images, only driver IDs and counts).

Usage (Kaggle):
    python src/prepare_data.py --out-dir /kaggle/working/data
Usage (local, on the fake dataset):
    python src/prepare_data.py --data-root fake_data --out-dir data
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (Kaggle / headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

CLASS_NAMES = {
    "c0": "safe driving",
    "c1": "texting - right",
    "c2": "phone call - right",
    "c3": "texting - left",
    "c4": "phone call - left",
    "c5": "operating radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "hair and makeup",
    "c9": "talking to passenger",
}


# --------------------------------------------------------------------------- #
# Locating the dataset (Kaggle mounts competition data under /kaggle/input/)
# --------------------------------------------------------------------------- #
def find_data_root(given: str | None) -> Path:
    if given:
        root = Path(given)
        if not (root / "driver_imgs_list.csv").exists():
            raise FileNotFoundError(f"driver_imgs_list.csv not found in {root}")
        return root
    # Search shallowly so we don't walk 100k image files.
    base = Path("/kaggle/input")
    for depth in range(1, 4):
        pattern = "/".join(["*"] * depth) + "/driver_imgs_list.csv"
        hits = sorted(base.glob(pattern))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(
        "Could not find driver_imgs_list.csv under /kaggle/input. "
        "Did you add the competition data as notebook input? "
        "Or pass --data-root explicitly."
    )


def find_train_dir(root: Path) -> Path:
    for candidate in (root / "imgs" / "train", root / "train"):
        if (candidate / "c0").is_dir():
            return candidate
    for depth in range(1, 4):
        pattern = "/".join(["*"] * depth) + "/train/c0"
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(f"No train/c0 folder found under {root}")


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def assign_splits(df: pd.DataFrame, n_test: int, n_val: int, seed: int) -> dict:
    drivers = np.array(sorted(df["subject"].unique()))
    if n_test + n_val >= len(drivers):
        raise ValueError(f"Only {len(drivers)} drivers; n_test+n_val too large")

    rng = np.random.default_rng(seed)
    rng.shuffle(drivers)
    test_drivers = sorted(drivers[:n_test].tolist())
    val_drivers = sorted(drivers[n_test : n_test + n_val].tolist())
    train_drivers = sorted(drivers[n_test + n_val :].tolist())

    # Honest scheme: whole drivers per split.
    df["driver_split"] = "train"
    df.loc[df["subject"].isin(val_drivers), "driver_split"] = "val"
    df.loc[df["subject"].isin(test_drivers), "driver_split"] = "test"

    # Leaky scheme: same held-out test drivers, but the remaining images are
    # shuffled randomly into train/val. Val size matched to the driver split.
    non_test = df[df["driver_split"] != "test"]
    val_frac = (df["driver_split"] == "val").sum() / len(non_test)
    tr_idx, va_idx = train_test_split(
        non_test.index,
        test_size=val_frac,
        stratify=non_test["label"],
        random_state=seed,
    )
    df["random_split"] = "test"
    df.loc[tr_idx, "random_split"] = "train"
    df.loc[va_idx, "random_split"] = "val"

    # ---- Sanity checks: these are the facts you quote in the README ----
    d_sets = {s: set(df.loc[df["driver_split"] == s, "subject"]) for s in ("train", "val", "test")}
    assert not (d_sets["train"] & d_sets["val"]), "driver split leaks train->val"
    assert not (d_sets["train"] & d_sets["test"]), "driver split leaks train->test"
    assert not (d_sets["val"] & d_sets["test"]), "driver split leaks val->test"

    r_train = set(df.loc[df["random_split"] == "train", "subject"])
    r_val = set(df.loc[df["random_split"] == "val", "subject"])
    shared = sorted(r_train & r_val)

    return {
        "seed": seed,
        "num_drivers": int(len(drivers)),
        "test_drivers": test_drivers,
        "val_drivers": val_drivers,
        "train_drivers": train_drivers,
        "random_split_drivers_in_both_train_and_val": len(shared),
    }


# --------------------------------------------------------------------------- #
# Pre-resizing
# --------------------------------------------------------------------------- #
def _resize_one(args):
    src, dst, size = args
    if os.path.exists(dst):
        return 0
    with Image.open(src) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        im.save(dst, quality=95)
    return 1


def resize_all(df: pd.DataFrame, train_dir: Path, img_dir: Path, size: int, workers: int):
    for cls in sorted(df["classname"].unique()):
        (img_dir / cls).mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(train_dir / r.classname / r.img), str(img_dir / r.classname / r.img), size)
        for r in df.itertuples()
    ]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, n in enumerate(ex.map(_resize_one, jobs, chunksize=64), 1):
            done += n
            if i % 2000 == 0 or i == len(jobs):
                print(f"  resized {i}/{len(jobs)}")
    return done


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def save_eda(df: pd.DataFrame, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)

    per_driver = df.groupby("subject").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    per_driver.plot.bar(ax=ax)
    ax.set_title("Images per driver")
    ax.set_ylabel("images")
    fig.tight_layout()
    fig.savefig(results_dir / "eda_images_per_driver.png", dpi=120)
    plt.close(fig)

    per_class = df["classname"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    per_class.plot.bar(ax=ax)
    ax.set_xticklabels([f"{c}\n{CLASS_NAMES[c]}" for c in per_class.index], rotation=45, ha="right", fontsize=8)
    ax.set_title("Images per class")
    fig.tight_layout()
    fig.savefig(results_dir / "eda_images_per_class.png", dpi=120)
    plt.close(fig)

    return per_driver, per_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None, help="folder containing driver_imgs_list.csv (auto-detected on Kaggle)")
    p.add_argument("--out-dir", default="data", help="where resized images + splits.csv go (gitignored)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--size", type=int, default=256, help="pre-resize to SIZE x SIZE; training crops 224 from this")
    p.add_argument("--n-test", type=int, default=4, help="drivers held out for test")
    p.add_argument("--n-val", type=int, default=4, help="drivers held out for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--skip-resize", action="store_true")
    args = p.parse_args()

    root = find_data_root(args.data_root)
    train_dir = find_train_dir(root)
    out_dir = Path(args.out_dir)
    img_dir = out_dir / f"imgs_{args.size}"
    results_dir = Path(args.results_dir)
    print(f"Data root : {root}\nTrain dir : {train_dir}\nOutput    : {out_dir}")

    df = pd.read_csv(root / "driver_imgs_list.csv")
    df["label"] = df["classname"].str[1:].astype(int)  # "c3" -> 3
    df["relpath"] = df["classname"] + "/" + df["img"]
    print(f"{len(df)} labelled images, {df['subject'].nunique()} drivers, {df['classname'].nunique()} classes")

    per_driver, per_class = save_eda(df, results_dir)
    print(f"Images per driver: min {per_driver.min()}, max {per_driver.max()}, median {int(per_driver.median())}")

    summary = assign_splits(df, args.n_test, args.n_val, args.seed)

    counts = {
        scheme: df.groupby(scheme).size().reindex(["train", "val", "test"]).astype(int).to_dict()
        for scheme in ("driver_split", "random_split")
    }
    class_by_split = (
        df.groupby(["driver_split", "classname"]).size().unstack(fill_value=0).reindex(["train", "val", "test"])
    )
    summary["image_counts"] = counts
    summary["driver_split_class_counts"] = class_by_split.to_dict(orient="index")
    summary["size"] = args.size

    print("\nImage counts per split:")
    for scheme, c in counts.items():
        print(f"  {scheme:13s} {c}")
    print(f"\nRandom split: {summary['random_split_drivers_in_both_train_and_val']} drivers appear in BOTH train and val "
          f"(this is the leak). Driver split: 0 by construction.")
    print("\nDriver split, images per class (check no class is starved):")
    print(class_by_split.to_string())

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "splits.csv", index=False)
    with open(results_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not args.skip_resize:
        print(f"\nPre-resizing to {args.size}x{args.size} with {args.workers} workers ...")
        n = resize_all(df, train_dir, img_dir, args.size, args.workers)
        print(f"  wrote {n} new files to {img_dir}")

    print(f"\nDone. splits.csv -> {out_dir}, summary + plots -> {results_dir}")


if __name__ == "__main__":
    main()
