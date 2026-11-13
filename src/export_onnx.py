"""Phase 5a: export trained checkpoints to ONNX (FP32) and verify them.

Design choices:
  - Normalization (mean/std) is baked into the graph, so the input is simply
    RGB float in [0, 1], shape [N, 3, 224, 224]. The C++ demo then only has to
    resize, convert BGR->RGB, and divide by 255.
  - Softmax is NOT in the graph: the output is raw logits [N, 10]. Runtimes
    take argmax / softmax themselves.
  - Batch dimension is dynamic (batch 1 for the camera, larger for evaluation).
  - Class names, input size and preprocessing are stored as ONNX metadata so
    any consumer can read them from the file.
  - Parity check: the ONNX model must match PyTorch on the same input.

    python src/export_onnx.py --ckpts checkpoints/*_driver.pth --out-dir models
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from models import build_model


class NormalizedModel(nn.Module):
    """Wraps a classifier so it accepts [0, 1] RGB input."""

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def export_one(ckpt_path: Path, out_dir: Path, opset: int) -> dict:
    state = torch.load(ckpt_path, map_location="cpu")
    model, _, _ = build_model(state["model_name"], num_classes=len(state["class_names"]), pretrained=False)
    model.load_state_dict(state["state_dict"])
    wrapped = NormalizedModel(model, state["mean"], state["std"]).eval()

    size = state["img_size"]
    name = ckpt_path.stem  # e.g. efficientnet_lite0_driver
    onnx_path = out_dir / f"{name}_fp32.onnx"
    dummy = torch.rand(1, 3, size, size)

    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # classic TorchScript exporter: stable, well-understood graphs
    )

    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    meta = {
        "class_names": json.dumps(state["class_names"]),
        "img_size": str(size),
        "input_format": "RGB float32 in [0,1], NCHW; resize to img_size x img_size (no crop)",
        "source_checkpoint": ckpt_path.name,
    }
    for k, v in meta.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(m, str(onnx_path))

    # ---- Parity check: PyTorch vs ONNX Runtime on random inputs ----
    x = torch.rand(8, 3, size, size)
    with torch.no_grad():
        ref = wrapped(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    same_argmax = float((ref.argmax(1) == got.argmax(1)).mean())

    info = {
        "name": name,
        "onnx": str(onnx_path),
        "size_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "parity_max_abs_diff": max_diff,
        "parity_argmax_agreement": same_argmax,
    }
    status = "OK" if max_diff < 1e-3 and same_argmax == 1.0 else "CHECK THIS"
    print(f"[{name}] -> {onnx_path} ({info['size_mb']} MB) | max |diff| {max_diff:.2e} | "
          f"argmax agree {same_argmax:.0%} | {status}")
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--out-dir", default="models")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    infos = [export_one(Path(c), out_dir, args.opset) for c in args.ckpts]
    Path("results").mkdir(exist_ok=True)
    with open("results/onnx_export.json", "w") as f:
        json.dump(infos, f, indent=2)


if __name__ == "__main__":
    main()
