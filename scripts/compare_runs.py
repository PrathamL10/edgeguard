"""Print one table comparing every training run in results/runs/.

    python scripts/compare_runs.py                 # prints a table
    python scripts/compare_runs.py --markdown      # paste-ready for the README
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="results/runs")
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()

    rows = []
    for f in sorted(Path(args.runs_dir).glob("*.json")):
        r = json.loads(f.read_text())
        rows.append({
            "run": r["tag"],
            "params (M)": r["params_millions"],
            "val acc": round(r["best_val_acc"], 3),
            "test acc (unseen drivers)": round(r["test"]["accuracy"], 3),
            "val - test gap": round(r["val_minus_test_acc"], 3),
            "test macro-F1": round(r["test"]["macro_f1"], 3),
            "test phone recall": round(r["test"]["phone_recall"], 3),
            "train min": r["train_minutes"],
        })
    if not rows:
        print(f"No runs found in {args.runs_dir}")
        return
    df = pd.DataFrame(rows)
    if args.markdown:
        try:
            print(df.to_markdown(index=False))
        except ImportError:  # needs the 'tabulate' package
            print("pip install tabulate for --markdown; plain table instead:\n")
            print(df.to_string(index=False))
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
