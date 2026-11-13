"""Pack the small slice of data Phase 5 needs, so it can run on your Mac.

Contents of the pack (~100 MB instead of 4 GB):
  - splits.csv                       (the same split as training)
  - calib.csv                        (calibration images: drawn from TRAIN drivers only)
  - imgs_256/<class>/<img>.jpg       (all test-driver images + the calibration images)

Calibration images come from the training drivers, never from test, otherwise
the INT8 scales would be tuned on the very images you then score.

On Kaggle:
    python scripts/pack_eval_data.py --data-dir /kaggle/working/data --out /kaggle/working/eval_pack
    cd /kaggle/working && zip -qr eval_pack.zip eval_pack
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/kaggle/working/data")
    p.add_argument("--out", default="/kaggle/working/eval_pack")
    p.add_argument("--resized-size", type=int, default=256)
    p.add_argument("--calib-per-class", type=int, default=30, help="30 x 10 classes = 300 calibration images")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir, out = Path(args.data_dir), Path(args.out)
    src_imgs = data_dir / f"imgs_{args.resized_size}"
    dst_imgs = out / f"imgs_{args.resized_size}"
    df = pd.read_csv(data_dir / "splits.csv")

    test = df[df["driver_split"] == "test"]
    calib = (df[df["driver_split"] == "train"]
             .groupby("label", group_keys=False)
             .apply(lambda g: g.sample(n=min(args.calib_per_class, len(g)), random_state=args.seed)))

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "splits.csv", index=False)
    calib.to_csv(out / "calib.csv", index=False)

    rel = pd.concat([test["relpath"], calib["relpath"]]).unique()
    for i, r in enumerate(rel, 1):
        dst = dst_imgs / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_imgs / r, dst)
        if i % 1000 == 0:
            print(f"  copied {i}/{len(rel)}")
    print(f"Packed {len(test)} test images + {len(calib)} calibration images "
          f"({test['subject'].nunique()} test drivers) into {out}")


if __name__ == "__main__":
    main()
