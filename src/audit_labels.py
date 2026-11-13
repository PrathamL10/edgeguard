"""Phase 4: flag likely mislabeled images with cleanlab (confident learning).

Idea: a trained model's predicted probabilities on images it has NEVER seen are
an independent opinion about each label. Where the model is confident the label
is something else, the label is suspicious. cleanlab estimates per-class
thresholds and ranks those suspicious images.

Important: the probabilities must be out-of-sample. A driver-split model never
trained on the val and test drivers, so we audit those 8 drivers. (Auditing the
whole dataset would need cross-validated predictions: one model per fold.)

Flagged != mislabeled. The script writes a review sheet; you look at the images
and record a verdict for each. "N of the top 30 flags were truly mislabeled" is
the honest number to report.

    python src/audit_labels.py --data-dir /kaggle/working/data --ckpt checkpoints/resnet18_driver.pth
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cleanlab.filter import find_label_issues
from PIL import Image
from torch.utils.data import DataLoader

from data import CLASS_NAMES, DriverDataset, build_transforms, load_splits
from models import build_model
from train import pick_device, predict


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    model, _, _ = build_model(state["model_name"], num_classes=len(state["class_names"]), pretrained=False)
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), state


def save_montage(rows: pd.DataFrame, img_dir: Path, path: Path, cols: int = 6):
    n = len(rows)
    if n == 0:
        return
    nrows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 2.6, nrows * 2.9))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (rank, r) in zip(axes.flat, enumerate(rows.itertuples(), 1)):
        with Image.open(img_dir / r.relpath) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"#{rank} {r.subject}\nlabel: {r.given_label}\nmodel: {r.predicted_label} ({r.p_predicted:.2f})",
                     fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--ckpt", required=True, help="a driver-split checkpoint, e.g. checkpoints/resnet18_driver.pth")
    p.add_argument("--splits", nargs="+", default=["val", "test"], help="driver_split parts the model never trained on")
    p.add_argument("--resized-size", type=int, default=256)
    p.add_argument("--top", type=int, default=30, help="how many flagged images to put in the montage / review sheet")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="results/audit")
    args = p.parse_args()

    device = pick_device(args.device)
    use_amp = device.type == "cuda"
    model, state = load_checkpoint(args.ckpt, device)
    if state.get("scheme") != "driver":
        print("WARNING: this checkpoint was not trained on the driver split, so its predictions on these "
              "images are NOT out-of-sample. Use a *_driver.pth checkpoint.")

    data_dir = Path(args.data_dir)
    img_dir = data_dir / f"imgs_{args.resized_size}"
    df = load_splits(data_dir)
    df = df[df["driver_split"].isin(args.splits)].reset_index(drop=True)

    ds = DriverDataset(df, img_dir, build_transforms(False, state["img_size"], state["mean"], state["std"]))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Auditing {len(df)} images from {df['subject'].nunique()} unseen drivers with {state['model_name']} ...")
    probs, labels = predict(model, loader, device, use_amp)

    ranked = find_label_issues(labels=labels, pred_probs=probs, return_indices_ranked_by="self_confidence")
    preds = probs.argmax(1)

    out = df.loc[ranked, ["relpath", "subject", "label"]].copy()
    out["given_label"] = [CLASS_NAMES[i] for i in out["label"]]
    out["predicted_label"] = [CLASS_NAMES[i] for i in preds[ranked]]
    out["p_given"] = probs[ranked, labels[ranked]].round(3)
    out["p_predicted"] = probs[ranked, preds[ranked]].round(3)
    out["verdict"] = ""  # you fill this in: mislabeled / ambiguous / model_error
    out = out.reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "all_flagged.csv", index_label="rank")
    top = out.head(args.top)
    top.to_csv(out_dir / "review_top.csv", index_label="rank")
    save_montage(top, img_dir, out_dir / "review_top.png")

    pairs = (out.groupby(["given_label", "predicted_label"]).size()
             .sort_values(ascending=False).head(10))
    summary = {
        "checkpoint": str(args.ckpt),
        "audited_images": int(len(df)),
        "audited_drivers": sorted(df["subject"].unique().tolist()),
        "flagged": int(len(out)),
        "flagged_pct": round(100 * len(out) / len(df), 2),
        "top_confusions_among_flags": {f"{g} -> {p}": int(n) for (g, p), n in pairs.items()},
    }
    with open(out_dir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Flagged {len(out)} of {len(df)} images ({summary['flagged_pct']}%) as possible label issues.")
    print("Most common (given label -> model's label) among flags:")
    for k, v in summary["top_confusions_among_flags"].items():
        print(f"   {v:4d}  {k}")
    print(f"\nReview sheet: {out_dir / 'review_top.csv'}\nMontage     : {out_dir / 'review_top.png'}")


if __name__ == "__main__":
    main()
