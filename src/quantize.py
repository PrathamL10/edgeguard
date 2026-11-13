"""Phase 5b: static INT8 post-training quantization (PTQ) with ONNX Runtime.

How static PTQ works:
  1. Run ~300 calibration images (from TRAIN drivers) through the FP32 model.
  2. Record the range of every activation tensor.
  3. Pick a scale + zero-point per tensor so FP32 values map onto 8-bit ints.
  4. Weights are quantized per-channel (one scale per output channel), which
     preserves accuracy much better than one scale for the whole tensor.

The output uses the QDQ format: QuantizeLinear/DequantizeLinear pairs around
ops. ONNX Runtime fuses them into real INT8 kernels on CPU, and TensorRT can
read the same format.

    python src/quantize.py --models models/*_fp32.onnx --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

from data import build_raw_transform


class ImageCalibrationReader(CalibrationDataReader):
    """Feeds calibration images one batch at a time, in the model's input format."""

    def __init__(self, image_paths, img_size=224, batch_size=16):
        self.paths = list(image_paths)
        self.transform = build_raw_transform(img_size)
        self.batch_size = batch_size
        self.i = 0

    def get_next(self):
        if self.i >= len(self.paths):
            return None
        batch = []
        for pth in self.paths[self.i:self.i + self.batch_size]:
            with Image.open(pth) as im:
                batch.append(self.transform(im.convert("RGB")).numpy())
        self.i += self.batch_size
        return {"input": np.stack(batch).astype(np.float32)}

    def rewind(self):
        self.i = 0


def quantize_one(fp32_path: Path, calib_paths, method: str, img_size: int) -> Path:
    stem = fp32_path.name.replace("_fp32.onnx", "")
    pre_path = fp32_path.with_name(f"{stem}_fp32_pre.onnx")
    int8_path = fp32_path.with_name(f"{stem}_int8.onnx")

    # Shape inference + graph cleanup first: ORT recommends this before quantizing.
    quant_pre_process(str(fp32_path), str(pre_path))

    quantize_static(
        model_input=str(pre_path),
        model_output=str(int8_path),
        calibration_data_reader=ImageCalibrationReader(calib_paths, img_size),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,  # U8 activations / S8 weights: the fast path on x86 and fine on ARM
        weight_type=QuantType.QInt8,
        calibrate_method={"minmax": CalibrationMethod.MinMax,
                          "percentile": CalibrationMethod.Percentile,
                          "entropy": CalibrationMethod.Entropy}[method],
    )
    pre_path.unlink(missing_ok=True)
    print(f"[{stem}] INT8 ({method}) -> {int8_path} "
          f"({fp32_path.stat().st_size / 1e6:.1f} MB -> {int8_path.stat().st_size / 1e6:.1f} MB)")
    return int8_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="*_fp32.onnx files from export_onnx.py")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--resized-size", type=int, default=256)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--method", choices=["minmax", "percentile", "entropy"], default="minmax",
                   help="how activation ranges are chosen from calibration data")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    calib = pd.read_csv(data_dir / "calib.csv")
    assert (calib["driver_split"] == "train").all(), "calibration images must come from train drivers"
    img_dir = data_dir / f"imgs_{args.resized_size}"
    calib_paths = [img_dir / r for r in calib["relpath"]]
    print(f"Calibrating on {len(calib_paths)} train-driver images, method={args.method}")

    for m in args.models:
        quantize_one(Path(m), calib_paths, args.method, args.img_size)


if __name__ == "__main__":
    main()
