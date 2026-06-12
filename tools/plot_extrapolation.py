#!/usr/bin/env python3
"""P3 extrapolation degradation curves (workflow §8 分析图):
per-model MAE vs high-wear stage interval on the protocol3 test range."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.utils.seed import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str, default="protocol3")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bin-size", type=int, default=4, help="folders per interval")
    parser.add_argument("--out", type=str, default="outputs/figures/p3_degradation.png")
    args = parser.parse_args()

    root = project_root()
    results_dir = root / "outputs" / "results" / args.protocol
    if not results_dir.is_dir():
        raise SystemExit(f"No results at {results_dir}")

    curves: dict[str, dict[int, list[float]]] = {}
    for model_dir in sorted(results_dir.iterdir()):
        pred_csv = model_dir / f"seed{args.seed}" / f"{args.split}_predictions.csv"
        if not pred_csv.exists():
            continue
        per_folder: dict[int, list[float]] = defaultdict(list)
        with pred_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                err = abs(float(row["wear_pct_true"]) - float(row["wear_pct_pred"]))
                per_folder[int(row["folder_id"])].append(err)
        curves[model_dir.name] = per_folder
    if not curves:
        raise SystemExit("No predictions found; run tools/eval.py on protocol3 first.")

    all_folders = sorted({f for c in curves.values() for f in c})
    lo, hi = min(all_folders), max(all_folders)
    bins = [(s, min(s + args.bin_size - 1, hi)) for s in range(lo, hi + 1, args.bin_size)]
    labels = [f"{a}-{b}" for a, b in bins]

    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for model, per_folder in curves.items():
        ys = []
        for a, b in bins:
            errs = [e for f in range(a, b + 1) for e in per_folder.get(f, [])]
            ys.append(float(np.mean(errs)) if errs else np.nan)
        ax.plot(labels, ys, "o-", label=model)
        for lab, y in zip(labels, ys):
            rows.append({"model": model, "stage_interval": lab,
                         "mae": round(y, 4) if not np.isnan(y) else ""})
    ax.set_xlabel("Wear stage interval (extrapolation test range)")
    ax.set_ylabel("MAE (wear %)")
    ax.set_title(f"Extrapolation degradation — {args.protocol} {args.split}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "stage_interval", "mae"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved {out_path} and {csv_path}")


if __name__ == "__main__":
    main()
