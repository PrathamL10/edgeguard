"""Model factory: one baseline and two edge-friendly architectures.

    model, mean, std = build_model("mobilenetv3_small", num_classes=10)

All three start from ImageNet weights (transfer learning) and get a fresh
10-class head. mean/std are returned because each model must be fed inputs
normalized the way it was pretrained; the C++ demo and export steps reuse them.
"""

import timm
import torch.nn as nn
from torchvision import models as tvm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_NAMES = ["resnet18", "mobilenetv3_small", "efficientnet_lite0"]


def build_model(name: str, num_classes: int = 10, pretrained: bool = True):
    if name == "resnet18":
        # ~11.2M params. Baseline: accurate, but heavy for an edge device.
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, IMAGENET_MEAN, IMAGENET_STD

    if name == "mobilenetv3_small":
        # ~1.5M params. Built for mobile CPUs; uses hard-swish + squeeze-excite,
        # which tend to lose more accuracy under INT8 quantization.
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model, IMAGENET_MEAN, IMAGENET_STD

    if name == "efficientnet_lite0":
        # ~3.4M params. EfficientNet with squeeze-excite removed and ReLU6
        # instead of swish, specifically so it quantizes cleanly to INT8.
        model = timm.create_model("efficientnet_lite0.ra_in1k", pretrained=pretrained, num_classes=num_classes)
        cfg = model.pretrained_cfg
        return model, tuple(cfg.get("mean", IMAGENET_MEAN)), tuple(cfg.get("std", IMAGENET_STD))

    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
