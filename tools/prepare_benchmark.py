#!/usr/bin/env python3
"""Prepare a public ordinal benchmark (workflow §10): parse labels + 80/10/10 split.

Datasets must be downloaded manually first (license/terms require it):
  UTKFace ★ : https://susanqq.github.io/UTKFace/  (Aligned&Cropped, ~23k imgs)
              -> unpack all jpgs under  datasets_public/utkface/
  AFAD      : https://github.com/afad-dataset/tarball
              -> unpack age folders under datasets_public/afad/
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.benchmark import DATASET_SPECS, build_benchmark_split, write_benchmark_csv
from boilerwear.utils.seed import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--data-root", type=str, default=None,
                        help="default: datasets_public/{dataset}")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = project_root()
    data_root = root / (args.data_root or f"datasets_public/{args.dataset}")
    if not data_root.is_dir():
        raise SystemExit(f"Data root not found: {data_root}\nDownload the dataset first (see --help).")

    spec = DATASET_SPECS[args.dataset]
    samples = spec["parser"](data_root)
    if not samples:
        raise SystemExit(f"No parsable images under {data_root}")

    rows = build_benchmark_split(samples, seed=args.seed)
    out_csv = root / "outputs" / "splits" / f"benchmark_{args.dataset}.csv"
    write_benchmark_csv(rows, out_csv)

    counts = Counter(r["split"] for r in rows)
    labels = [r["label"] for r in rows]
    print(f"{args.dataset}: {len(rows)} images, label range {min(labels)}-{max(labels)}")
    print(f"  splits: {dict(counts)}  (seed={args.seed}, image-level 80/10/10)")
    print(f"  saved: {out_csv}")


if __name__ == "__main__":
    main()
