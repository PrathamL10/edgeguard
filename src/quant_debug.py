"""Find WHERE quantization breaks a model, layer by layer.

Runs the same calibration images through the FP32 model and its INT8 (QDQ)
version, captures every intermediate activation, and measures the
signal-to-quantization-noise ratio (SQNR, in dB; higher is better):

  - local error  (qdq_err):    damage done by quantizing THIS tensor alone.
  - cumulative   (xmodel_err): how far the INT8 model's activation has drifted
                               from the FP32 model's activation at this point,
                               including all damage from earlier layers.

Reading it: walk down the network in order. The layer where cumulative SQNR
falls off a cliff (and local SQNR is low there too) is where it breaks.
Rough guide: > 30 dB is fine, 15-30 dB is degraded, < 10 dB is broken.
Weight SQNR is reported too, to separate "bad weights" from "bad activations".

    python src/quant_debug.py --fp32 models/mobilenetv3_small_driver_fp32.onnx \
        --int8 models/mobilenetv3_small_driver_int8.onnx --data-dir eval_pack
"""

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import onnx
import pandas as pd
from onnxruntime.quantization.qdq_loss_debug import (collect_activations, compute_activation_error,
                                                     compute_weight_error, create_activation_matching,
                                                     create_weight_matching,
                                                     modify_model_output_intermediate_tensors)
from onnxruntime.quantization.shape_inference import quant_pre_process

from quantize import ImageCalibrationReader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", required=True)
    p.add_argument("--int8", required=True)
    p.add_argument("--data-dir", default="eval_pack")
    p.add_argument("--resized-size", type=int, default=256)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--n-images", type=int, default=40, help="images to compare on (a few dozen is plenty)")
    p.add_argument("--out-dir", default="results/quant_debug")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    calib = pd.read_csv(data_dir / "calib.csv")
    paths = [data_dir / f"imgs_{args.resized_size}" / r for r in calib["relpath"]][: args.n_images]
    name = Path(args.int8).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # The INT8 model was built from the pre-processed graph, so compare against
        # the same graph (tensor names must match).
        fp32_pre = tmp / "fp32_pre.onnx"
        quant_pre_process(args.fp32, str(fp32_pre))
        fp32_aug, int8_aug = tmp / "fp32_aug.onnx", tmp / "int8_aug.onnx"
        modify_model_output_intermediate_tensors(str(fp32_pre), str(fp32_aug))
        modify_model_output_intermediate_tensors(args.int8, str(int8_aug))

        print(f"Capturing activations on {len(paths)} images ...")
        fp32_acts = collect_activations(str(fp32_aug), ImageCalibrationReader(paths, args.img_size, batch_size=10))
        int8_acts = collect_activations(str(int8_aug), ImageCalibrationReader(paths, args.img_size, batch_size=10))
        act_err = compute_activation_error(create_activation_matching(int8_acts, fp32_acts))
        w_err = compute_weight_error(create_weight_matching(str(fp32_pre), args.int8))

        # Order tensors by where they are produced in the network, and note the op type.
        graph = onnx.load(str(fp32_pre)).graph
        producer = {}
        for i, node in enumerate(graph.node):
            for o in node.output:
                producer[o] = (i, node.op_type, node.name or f"(unnamed, output {o})")

    rows = []
    for tensor, e in act_err.items():
        if tensor not in producer or "xmodel_err" not in e:
            continue
        idx, op, node = producer[tensor]
        rows.append({"order": idx, "node": node, "tensor": tensor, "op_type": op,
                     "local_sqnr_db": round(float(e.get("qdq_err", float("nan"))), 2),
                     "cumulative_sqnr_db": round(float(e["xmodel_err"]), 2)})
    df = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)
    df.to_csv(out_dir / f"{name}_activations.csv", index=False)

    wdf = pd.DataFrame([{"weight": k, "weight_sqnr_db": round(float(v), 2)} for k, v in w_err.items()])
    wdf = wdf.sort_values("weight_sqnr_db").reset_index(drop=True)
    wdf.to_csv(out_dir / f"{name}_weights.csv", index=False)

    # ---- Plot cumulative SQNR along the network ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(df)), df["cumulative_sqnr_db"], marker=".", lw=1, label="cumulative (vs FP32 model)")
    ax.plot(range(len(df)), df["local_sqnr_db"], marker=".", lw=1, alpha=0.6, label="local (this tensor only)")
    for thr, c in ((30, "green"), (10, "red")):
        ax.axhline(thr, color=c, ls="--", lw=0.8)
    ax.set_xlabel("tensor, in network order")
    ax.set_ylabel("SQNR (dB), higher = better")
    ax.set_title(f"{name}: where does INT8 diverge from FP32?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_sqnr.png", dpi=120)
    plt.close(fig)

    # ---- Console summary ----
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)
    print("\nCumulative SQNR along the network (every tensor, in order):")
    print(df[["order", "op_type", "local_sqnr_db", "cumulative_sqnr_db", "node"]].to_string(index=False))

    drops = df["cumulative_sqnr_db"].diff()
    worst_drop = drops.idxmin() if len(df) > 1 else 0
    first_broken = df.index[df["cumulative_sqnr_db"] < 10]
    print("\nSummary")
    print(f"  final output SQNR      : {df['cumulative_sqnr_db'].iloc[-1]:.1f} dB")
    if len(first_broken):
        r = df.loc[first_broken[0]]
        print(f"  first tensor < 10 dB   : #{int(r['order'])} {r['op_type']} node={r['node']} -> {r['cumulative_sqnr_db']} dB")
    r = df.loc[worst_drop]
    print(f"  biggest single drop    : #{int(r['order'])} {r['op_type']} node={r['node']} "
          f"{drops.loc[worst_drop]:+.1f} dB -> {r['cumulative_sqnr_db']} dB")
    print(f"  lowest local SQNR ops  : " + ", ".join(
        f"{t.op_type}({t.local_sqnr_db})" for t in df.nsmallest(5, "local_sqnr_db").itertuples()))
    print("  worst weight SQNR      : " + ", ".join(
        f"{w.weight}({w.weight_sqnr_db})" for w in wdf.head(5).itertuples()))
    print(f"\nSaved CSVs + plot to {out_dir}")


if __name__ == "__main__":
    main()
