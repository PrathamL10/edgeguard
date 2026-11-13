Host: Apple M1 (arm64), Darwin 25.6.0, ONNX Runtime 1.32.0

| Model | Threads | Test acc | Phone recall | Size (MB) | p50 (ms) | p95 (ms) | Peak RAM (+MB) |
|---|---|---|---|---|---|---|---|
| mobilenetv3_small_driver_fp32 | 1 | 82.6% | 91.7% | 6.13 | 5.258 | 5.923 | 31.9 |
| mobilenetv3_small_driver_int8 | 1 | 20.3% | 40.9% | 1.89 | 1.665 | 2.299 | 25.8 |
| mobilenetv3_small_driver_int8_convonly | 1 | 18.8% | 28.9% | 1.87 | 1.893 | 2.025 | 28.2 |
| mobilenetv3_small_driver_int8_entropy | 1 | 20.3% | 40.9% | 1.89 | 1.663 | 1.794 | 25.3 |
| mobilenetv3_small_driver_int8_pct | 1 | 16.3% | 20.0% | 1.89 | 1.665 | 2.071 | 25.4 |
