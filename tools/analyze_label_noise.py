#!/usr/bin/env python3
"""Analyze adjacent-folder visual similarity (label ambiguity)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.data.builder import discover_folders
from boilerwear.utils.seed import project_root


def load_folder_mean_image(data_root: Path, folder_id: int) -> np.ndarray:
    folder = data_root / str(folder_id)
    imgs = sorted(folder.glob("*.bmp"))[:5]
    arrays = [np.array(Image.open(p).convert("RGB"), dtype=np.float32) for p in imgs]
    return np.mean(arrays, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--out-dir", type=str, default="outputs/reports/label_noise")
    args = parser.parse_args()

    root = project_root()
    data_root = root / args.data_root
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    folders = discover_folders(data_root)
    rows = []
    ssim_vals = []

    for a, b in zip(folders[:-1], folders[1:]):
        img_a = load_folder_mean_image(data_root, a)
        img_b = load_folder_mean_image(data_root, b)
        mse = float(np.mean((img_a - img_b) ** 2))
        s = float(ssim(img_a, img_b, channel_axis=2, data_range=255.0))
        delta_wear = (b - a) / 190 * 100
        rows.append({"folder_a": a, "folder_b": b, "delta_wear_pct": delta_wear, "ssim": round(s, 6), "mse": round(mse, 4)})
        ssim_vals.append(s)

    csv_path = out_dir / "adjacent_folder_similarity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "adjacent_pairs": len(rows),
        "ssim_mean": round(float(np.mean(ssim_vals)), 6),
        "ssim_std": round(float(np.std(ssim_vals)), 6),
        "ssim_min": round(float(np.min(ssim_vals)), 6),
        "ssim_p95": round(float(np.percentile(ssim_vals, 95)), 6),
        "interpretation": "High adjacent SSIM supports LDL soft labels for fine-grained ordinal stages.",
    }
    import json
    with (out_dir / "label_noise_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Adjacent SSIM mean={summary['ssim_mean']:.4f} -> {csv_path}")


if __name__ == "__main__":
    main()
