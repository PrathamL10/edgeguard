"""Phase 5c: accuracy of every exported model on the unseen test drivers.

Answers "what did quantization cost?": accuracy, macro-F1 and phone-use recall
for each FP32 and INT8 model, plus how often the INT8 model makes exactly the
same prediction as its FP32 parent (agreement).

    python src/evaluate_onnx.py --models models/*.onnx --data-dir data
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader

from data import DriverDataset, build_raw_transform, load_splits
from metrics import compute_metrics


def run_model(path: Path, loader) -> tuple[np.ndarray, np.ndarray]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    preds, labels = [], []
    for x, y in loader:
        logits = sess.run(None, {"input": x.numpy()})[0]
        preds.append(logits.argmax(1))
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--resized-size", type=int, default=256)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out", default="results/onnx_eval.json")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    df = load_splits(data_dir)
    test = df[df["driver_split"] == "test"]
    ds = DriverDataset(test, data_dir / f"imgs_{args.resized_size}", build_raw_transform(args.img_size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Evaluating on {len(ds)} images from {test['subject'].nunique()} unseen test drivers")

    paths = sorted(Path(m) for m in args.models if not m.endswith("_pre.onnx"))
    results, preds_by_name = {}, {}
    for path in paths:
        name = path.stem  # e.g. efficientnet_lite0_driver_int8
        preds, labels = run_model(path, loader)
        m = compute_metrics(labels, preds)
        preds_by_name[name] = preds
        results[name] = {
            "file": str(path),
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "phone_recall": m["phone_recall"],
            "phone_precision": m["phone_precision"],
            "per_class_recall": m["per_class_recall"],
        }
        print(f"{name:34s} acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
              f"phone recall {m['phone_recall']:.4f} | {results[name]['size_mb']} MB")

    # INT8 vs its FP32 parent: accuracy drop and prediction agreement.
    print()
    for name in results:
        if name.endswith("_int8"):
            parent = name.replace("_int8", "_fp32")
            if parent in results:
                agree = float((preds_by_name[name] == preds_by_name[parent]).mean())
                drop = results[parent]["accuracy"] - results[name]["accuracy"]
                results[name]["agreement_with_fp32"] = agree
                results[name]["accuracy_drop_vs_fp32"] = drop
                results[name]["phone_recall_drop_vs_fp32"] = results[parent]["phone_recall"] - results[name]["phone_recall"]
                print(f"{name:34s} vs FP32: accuracy drop {drop * 100:+.2f} pts | "
                      f"same prediction on {agree:.1%} of images")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
