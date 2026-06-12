#!/usr/bin/env python3
"""Summarize test metrics into main results CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.utils.seed import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str, default="protocol2")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Aggregate a public benchmark instead (e.g. utkface -> results/benchmark_utkface)")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--out", type=str, default="outputs/results/main_results.csv")
    parser.add_argument("--tag", type=str, default=None,
                        help="Only include seed dirs matching seed{N}_{tag}; default: untagged dirs only")
    args = parser.parse_args()

    root = project_root()
    if args.benchmark:
        args.protocol = f"benchmark_{args.benchmark}"
        if args.out == "outputs/results/main_results.csv":
            args.out = f"outputs/results/benchmark_{args.benchmark}.csv"
    results_dir = root / args.results_dir / args.protocol
    rows = []

    if not results_dir.exists():
        print(f"No results at {results_dir}")
        return

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for seed_dir in sorted(model_dir.iterdir()):
            name = seed_dir.name
            if args.tag:
                if not name.endswith(f"_{args.tag}"):
                    continue
            elif "_" in name:
                continue  # skip tagged (smoke/fuse/scratch) dirs in the main table
            metrics_file = seed_dir / "test_metrics.json"
            if not metrics_file.exists():
                continue
            with metrics_file.open(encoding="utf-8") as f:
                m = json.load(f)
            rows.append(
                {
                    "protocol": args.protocol,
                    "model": model_dir.name,
                    "seed": seed_dir.name.replace("seed", "").split("_")[0],
                    **m,
                }
            )

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No test_metrics.json found. Run tools/eval.py first.")
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")

    # mean±std 聚合(多 seed)
    from collections import defaultdict
    import statistics
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["model"]].append(r)
    agg_rows = []
    for model, items in grouped.items():
        agg = {"protocol": args.protocol, "model": model, "n_seeds": len(items)}
        for key in ("mae", "medae", "rmse", "r2", "acc_at_5", "acc_at_10", "spearman", "qwk"):
            vals = [float(it[key]) for it in items if key in it]
            if vals:
                agg[f"{key}_mean"] = round(statistics.mean(vals), 4)
                agg[f"{key}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        agg_rows.append(agg)
    agg_path = out_path.with_name(out_path.stem + "_meanstd.csv")
    with agg_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"Wrote mean±std aggregation to {agg_path}")

    agg_sorted = sorted(agg_rows, key=lambda r: r.get("mae_mean", 1e9))
    print("\nRanking by mean MAE:")
    for r in agg_sorted:
        print(f"  {r['model']:24s} MAE={r.get('mae_mean', float('nan')):.4f}±{r.get('mae_std', 0):.4f} "
              f"QWK={r.get('qwk_mean', float('nan')):.4f} (n={r['n_seeds']})")


if __name__ == "__main__":
    main()
