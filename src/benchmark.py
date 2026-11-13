"""Phase 5d: CPU latency (p50/p95), model size and peak memory for each ONNX model.

Measurement rules (each one is an interview talking point):
  - Batch size 1: a dashcam processes one frame at a time.
  - Warm-up runs are discarded: the first calls pay for memory allocation and
    kernel selection, which a running device has already done.
  - p95, not just the average: an alert system has to meet its frame budget
    nearly every frame, so tail latency matters.
  - Thread count is fixed and reported: 1 thread approximates a small edge CPU
    core that is shared with other work; 4 threads shows the multicore number.
  - Each model runs in its own fresh process, so peak memory is not polluted by
    models loaded earlier.

    python src/benchmark.py --models models/*.onnx --threads 1 4
"""

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3  # macOS: bytes, Linux: KB


def cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def bench_single(model_path: str, threads: int, warmup: int, runs: int, img_size: int) -> dict:
    """Runs inside a child process."""
    import numpy as np
    import onnxruntime as ort

    base_mb = peak_rss_mb()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1000

    x = np.random.default_rng(0).random((1, 3, img_size, img_size), dtype=np.float32)
    feed = {"input": x}
    for _ in range(warmup):
        sess.run(None, feed)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times)

    return {
        "model": Path(model_path).stem,
        "threads": threads,
        "size_mb": round(Path(model_path).stat().st_size / 1e6, 2),
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "mean_ms": round(float(times.mean()), 3),
        "fps_at_p50": round(1000 / float(np.percentile(times, 50)), 1),
        "load_ms": round(load_ms, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "peak_rss_delta_mb": round(peak_rss_mb() - base_mb, 1),
        "runs": runs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--tag", default=None, help="name for this machine, e.g. mac_m2 or kaggle_cpu")
    p.add_argument("--accuracy-json", default="results/onnx_eval.json")
    p.add_argument("--out-dir", default="results/bench")
    p.add_argument("--single", default=None, help=argparse.SUPPRESS)  # internal: child-process mode
    args = p.parse_args()

    if args.single:
        print(json.dumps(bench_single(args.single, args.threads[0], args.warmup, args.runs, args.img_size)))
        return

    if not args.models:
        p.error("--models is required")

    import onnxruntime as ort

    host = {
        "cpu": cpu_name(),
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
    }
    tag = args.tag or f"{platform.system().lower()}_{platform.machine()}"
    print(f"Host: {host}")

    models = sorted(m for m in args.models if not m.endswith("_pre.onnx"))
    rows = []
    for m in models:
        for th in args.threads:
            cmd = [sys.executable, __file__, "--single", m, "--threads", str(th),
                   "--warmup", str(args.warmup), "--runs", str(args.runs), "--img-size", str(args.img_size)]
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode != 0:
                print(f"FAILED {m} threads={th}\n{out.stderr}")
                continue
            r = json.loads(out.stdout.strip().splitlines()[-1])
            rows.append(r)
            print(f"{r['model']:34s} threads={th} | p50 {r['p50_ms']:7.2f} ms | p95 {r['p95_ms']:7.2f} ms | "
                  f"{r['size_mb']:6.2f} MB | peak RAM +{r['peak_rss_delta_mb']:.0f} MB")

    acc = {}
    if Path(args.accuracy_json).exists():
        acc = json.loads(Path(args.accuracy_json).read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"bench_{tag}.json", "w") as f:
        json.dump({"host": host, "results": rows}, f, indent=2)

    header = ("| Model | Threads | Test acc | Phone recall | Size (MB) | p50 (ms) | p95 (ms) | "
              "Peak RAM (+MB) |\n|---|---|---|---|---|---|---|---|")
    lines = [header]
    for r in rows:
        a = acc.get(r["model"], {})
        fmt = lambda k: f"{a[k] * 100:.1f}%" if k in a else "n/a"
        lines.append(f"| {r['model']} | {r['threads']} | {fmt('accuracy')} | {fmt('phone_recall')} | "
                     f"{r['size_mb']} | {r['p50_ms']} | {r['p95_ms']} | {r['peak_rss_delta_mb']} |")
    table = f"Host: {host['cpu']} ({host['machine']}), {host['os']}, ONNX Runtime {host['onnxruntime']}\n\n" + "\n".join(lines)
    (out_dir / f"bench_{tag}.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
