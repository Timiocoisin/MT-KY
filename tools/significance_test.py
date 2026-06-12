#!/usr/bin/env python3
"""Folder-level paired Wilcoxon signed-rank test: target model vs every baseline.

Workflow v3.2 §6.4: per-folder absolute errors (median over images of the same
folder, averaged across available seeds) are paired across models on common
folders; reports the Wilcoxon p-value and the rank-biserial effect size.

Reads {split}_predictions.csv exported by tools/eval.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.utils.seed import project_root


def load_folder_errors(pred_csv: Path) -> dict[int, float]:
    """Per-folder median absolute error (image-level errors -> folder median)."""
    per_folder: dict[int, list[float]] = defaultdict(list)
    with pred_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            err = abs(float(row["wear_pct_true"]) - float(row["wear_pct_pred"]))
            per_folder[int(row["folder_id"])].append(err)
    return {fid: float(np.median(errs)) for fid, errs in per_folder.items()}


def collect_model_errors(
    results_dir: Path, model: str, split: str, tag: str | None
) -> dict[int, float] | None:
    """Average per-folder errors across all matching seed dirs of one model."""
    model_dir = results_dir / model
    if not model_dir.is_dir():
        return None
    acc: dict[int, list[float]] = defaultdict(list)
    for seed_dir in sorted(model_dir.iterdir()):
        name = seed_dir.name
        if tag:
            if not name.endswith(f"_{tag}"):
                continue
        elif "_" in name:
            continue
        pred_csv = seed_dir / f"{split}_predictions.csv"
        if not pred_csv.exists():
            continue
        for fid, err in load_folder_errors(pred_csv).items():
            acc[fid].append(err)
    if not acc:
        return None
    return {fid: float(np.mean(v)) for fid, v in acc.items()}


def rank_biserial(diff: np.ndarray) -> float:
    """Rank-biserial correlation as the Wilcoxon effect size."""
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    pos = ranks[nz > 0].sum()
    neg = ranks[nz < 0].sum()
    total = pos + neg
    return float((neg - pos) / total) if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str, default="protocol2")
    parser.add_argument("--target", type=str, default="soformer")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--level", type=str, default="folder", choices=["folder"])
    parser.add_argument("--test", type=str, default="wilcoxon", choices=["wilcoxon"])
    parser.add_argument("--tag", type=str, default=None, help="Seed dir tag filter (e.g. smoke)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    root = project_root()
    results_dir = root / "outputs" / "results" / args.protocol
    if not results_dir.is_dir():
        print(f"No results at {results_dir}")
        return

    target_errors = collect_model_errors(results_dir, args.target, args.split, args.tag)
    if target_errors is None:
        print(
            f"No {args.split}_predictions.csv found for target '{args.target}'. "
            f"Run tools/eval.py first (it exports predictions automatically)."
        )
        return

    rows = []
    for model_dir in sorted(results_dir.iterdir()):
        model = model_dir.name
        if not model_dir.is_dir() or model == args.target:
            continue
        base_errors = collect_model_errors(results_dir, model, args.split, args.tag)
        if base_errors is None:
            continue
        common = sorted(set(target_errors) & set(base_errors))
        if len(common) < 5:
            print(f"  skip {model}: only {len(common)} common folders")
            continue
        t = np.array([target_errors[f] for f in common])
        b = np.array([base_errors[f] for f in common])
        diff = t - b  # negative -> target better
        if np.allclose(diff, 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(t, b)
        rows.append(
            {
                "protocol": args.protocol,
                "target": args.target,
                "baseline": model,
                "n_folders": len(common),
                "target_mean_err": round(float(t.mean()), 4),
                "baseline_mean_err": round(float(b.mean()), 4),
                "wilcoxon_stat": round(float(stat), 4),
                "p_value": float(f"{p:.3e}"),
                "effect_size_rb": round(rank_biserial(diff), 4),
                "target_better": bool(t.mean() < b.mean()),
                "significant_p05": bool(p < 0.05),
            }
        )
        marker = "*" if p < 0.05 else " "
        print(
            f"  {args.target} vs {model:24s} n={len(common):3d} "
            f"err {t.mean():6.3f} vs {b.mean():6.3f}  p={p:.3e}{marker} r_rb={rank_biserial(diff):+.3f}"
        )

    if not rows:
        print("No baselines with predictions found.")
        return

    out = root / (args.out or f"outputs/results/significance_{'p2' if args.protocol == 'protocol2' else args.protocol}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
