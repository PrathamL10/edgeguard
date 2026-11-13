Host: Apple M1 (arm64), Darwin 25.6.0, ONNX Runtime 1.32.0

| Model | Threads | Test acc | Phone recall | Size (MB) | p50 (ms) | p95 (ms) | Peak RAM (+MB) |
|---|---|---|---|---|---|---|---|
| efficientnet_lite0_driver_fp32 | 1 | 89.0% | 96.4% | 13.49 | 20.61 | 21.693 | 48.3 |
| efficientnet_lite0_driver_fp32 | 4 | 89.0% | 96.4% | 13.49 | 6.46 | 6.855 | 48.5 |
| efficientnet_lite0_driver_int8 | 1 | 89.1% | 96.9% | 3.82 | 4.556 | 4.888 | 35.0 |
| efficientnet_lite0_driver_int8 | 4 | 89.1% | 96.9% | 3.82 | 1.601 | 1.812 | 35.6 |
| mobilenetv3_small_driver_fp32 | 1 | 82.6% | 91.7% | 6.13 | 5.227 | 5.393 | 31.9 |
| mobilenetv3_small_driver_fp32 | 4 | 82.6% | 91.7% | 6.13 | 1.921 | 2.231 | 32.5 |
| mobilenetv3_small_driver_int8 | 1 | 20.3% | 40.9% | 1.89 | 1.664 | 1.705 | 25.2 |
| mobilenetv3_small_driver_int8 | 4 | 20.3% | 40.9% | 1.89 | 0.999 | 2.429 | 25.8 |
| resnet18_driver_fp32 | 1 | 90.6% | 97.6% | 44.72 | 47.143 | 49.21 | 89.1 |
| resnet18_driver_fp32 | 4 | 90.6% | 97.6% | 44.72 | 16.565 | 21.103 | 82.1 |
| resnet18_driver_int8 | 1 | 90.4% | 97.4% | 11.3 | 11.988 | 12.626 | 49.4 |
| resnet18_driver_int8 | 4 | 90.4% | 97.4% | 11.3 | 4.402 | 4.457 | 49.2 |
