#!/usr/bin/env python3
"""Plot prediction scatter y_true vs y_pred."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.utils.metrics import compute_metrics
from boilerwear.utils.seed import project_root


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(float(row["wear_pct_true"]))
            y_pred.append(float(row["wear_pct_pred"]))
    return np.array(y_true), np.array(y_pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--title", type=str, default="Prediction Scatter")
    parser.add_argument("--out", type=str, default="outputs/figures/scatter.png")
    args = parser.parse_args()

    root = project_root()
    pred_path = root / args.predictions
    y_true, y_pred = load_predictions(pred_path)
    metrics = compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, s=12, edgecolors="none")
    ax.plot([0, 100], [0, 100], "r--", lw=1.5)
    ax.set_xlabel("True wear (%)")
    ax.set_ylabel("Predicted wear (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_title(f"{args.title}\nMAE={metrics['mae']:.2f}% QWK={metrics['qwk']:.3f}")
    ax.grid(True, alpha=0.3)

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
