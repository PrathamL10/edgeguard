"""Train one model on one split scheme, then score it on the held-out test drivers.

Hand-written loop (no Lightning / Trainer wrappers) so every step is visible:
forward -> loss -> backward -> optimizer step -> LR schedule, with mixed
precision on CUDA, best-checkpoint selection on validation accuracy, and a
final evaluation on the 4 unseen test drivers.

Leakage experiment (Phase 2):
    python src/train.py --data-dir /kaggle/working/data --model resnet18 --scheme random
    python src/train.py --data-dir /kaggle/working/data --model resnet18 --scheme driver

Edge models (Phase 3):
    python src/train.py --data-dir /kaggle/working/data --model mobilenetv3_small --scheme driver
    python src/train.py --data-dir /kaggle/working/data --model efficientnet_lite0 --scheme driver
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import CLASS_NAMES, build_dataloaders
from metrics import compute_metrics, plot_confusion_matrix
from models import MODEL_NAMES, build_model, count_params


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU, for local debugging
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup for the first epoch, then cosine decay to zero.

    Warmup avoids wrecking the pretrained weights with large updates while the
    new randomly-initialized head is still producing garbage gradients.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        # GradScaler multiplies the loss before backward so small FP16 gradients
        # don't underflow to zero, then unscales before the optimizer step.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += x.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, loader, device, use_amp):
    """Returns (probs [N, C], labels [N]) for a whole loader."""
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu())
        all_labels.append(y)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model", choices=MODEL_NAMES, default="resnet18")
    p.add_argument("--scheme", choices=["driver", "random"], default="driver")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--resized-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-pretrained", action="store_true", help="random init (only for offline debugging)")
    p.add_argument("--limit", type=int, default=None, help="cap images per split for a quick debug run")
    p.add_argument("--ckpt-dir", default="checkpoints", help="weights go here (gitignored)")
    p.add_argument("--results-dir", default="results/runs", help="metrics JSON + plots go here (commit these)")
    p.add_argument("--tag", default=None, help="run name; defaults to <model>_<scheme>")
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)
    use_amp = device.type == "cuda"
    tag = args.tag or f"{args.model}_{args.scheme}"
    ckpt_dir, results_dir = Path(args.ckpt_dir), Path(args.results_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model, mean, std = build_model(args.model, num_classes=len(CLASS_NAMES), pretrained=not args.no_pretrained)
    model = model.to(device)
    n_params = count_params(model)

    loaders = build_dataloaders(
        args.data_dir, scheme=args.scheme, batch_size=args.batch_size, num_workers=args.num_workers,
        img_size=args.img_size, resized_size=args.resized_size, mean=mean, std=std,
        limit=args.limit, seed=args.seed,
    )
    sizes = {k: len(v.dataset) for k, v in loaders.items()}
    print(f"[{tag}] device={device} amp={use_amp} params={n_params/1e6:.2f}M sizes={sizes}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(loaders["train"])
    scheduler = make_scheduler(optimizer, warmup_steps=steps_per_epoch, total_steps=steps_per_epoch * args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_val_acc, best_epoch = [], -1.0, -1
    ckpt_path = ckpt_dir / f"{tag}.pth"
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, scheduler, scaler, device, use_amp)
        val_probs, val_labels = predict(model, loaders["val"], device, use_amp)
        val_m = compute_metrics(val_labels, val_probs.argmax(1))
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_acc": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "val_phone_recall": val_m["phone_recall"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(f"[{tag}] ep {epoch:2d}/{args.epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val acc {val_m['accuracy']:.3f} f1 {val_m['macro_f1']:.3f} phone-recall {val_m['phone_recall']:.3f} | "
              f"{row['seconds']:.0f}s")

        if val_m["accuracy"] > best_val_acc:
            best_val_acc, best_epoch = val_m["accuracy"], epoch
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "mean": mean, "std": std,
                "img_size": args.img_size,
                "class_names": CLASS_NAMES,
                "scheme": args.scheme,
                "epoch": epoch,
            }, ckpt_path)

    train_minutes = (time.time() - t_start) / 60

    # ---- Final: reload the best checkpoint and score the unseen test drivers ----
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    test_probs, test_labels = predict(model, loaders["test"], device, use_amp)
    test_m = compute_metrics(test_labels, test_probs.argmax(1))
    np.savez(ckpt_dir / f"{tag}_test_preds.npz", probs=test_probs, labels=test_labels)
    plot_confusion_matrix(test_m["confusion_matrix"], results_dir / f"{tag}_test_cm.png",
                          title=f"{tag}: test (unseen drivers), row-normalized")

    result = {
        "tag": tag,
        "model": args.model,
        "scheme": args.scheme,
        "params_millions": round(n_params / 1e6, 3),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test": test_m,
        "val_minus_test_acc": best_val_acc - test_m["accuracy"],
        "train_minutes": round(train_minutes, 2),
        "config": vars(args),
        "history": history,
    }
    with open(results_dir / f"{tag}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{tag}] best epoch {best_epoch} | VAL acc {best_val_acc:.3f} | "
          f"TEST acc (unseen drivers) {test_m['accuracy']:.3f} | gap {best_val_acc - test_m['accuracy']:+.3f} | "
          f"test phone-recall {test_m['phone_recall']:.3f} | {train_minutes:.1f} min")
    print("Per-class test recall:")
    for name, r in test_m["per_class_recall"].items():
        print(f"   {name:22s} {r:.3f}")


if __name__ == "__main__":
    main()
