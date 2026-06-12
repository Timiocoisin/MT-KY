#!/usr/bin/env python3
"""Generate three split protocols and data statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.data.builder import generate_all_splits
from boilerwear.utils.seed import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BoilerWear-190 splits")
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--out-dir", type=str, default="outputs/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p2-offset", type=int, default=0,
                        help="Protocol-2 hold-out offset k (test=folder%%10==k, val=folder%%10==(k+5)%%10)")
    parser.add_argument("--smoke-fraction", type=float, default=None,
                        help="Folder-level subsample fraction for the smoke gate (e.g. 0.1); writes *_smoke files")
    args = parser.parse_args()

    root = project_root()
    data_root = root / args.data_root
    out_dir = root / args.out_dir

    manifest = generate_all_splits(
        data_root, out_dir, seed=args.seed,
        p2_offset=args.p2_offset, smoke_fraction=args.smoke_fraction,
    )
    print(f"Splits written to {out_dir}")
    for name, leak in manifest["leak_check"].items():
        status = "OK" if leak["leak_free"] else "LEAK"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
