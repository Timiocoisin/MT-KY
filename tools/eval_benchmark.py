#!/usr/bin/env python3
"""Evaluate a trained benchmark head on the test split (§10)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.benchmark import (
    DATASET_SPECS,
    BenchmarkDataset,
    BenchmarkNet,
    collate_benchmark,
    load_benchmark_records,
)
from boilerwear.utils.metrics import compute_metrics
from boilerwear.utils.seed import get_device, project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--head", type=str, required=True, choices=["reg", "coral", "ldl", "hod"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    root = project_root()
    spec = DATASET_SPECS[args.dataset]
    num_stages = spec["max_label"] - spec["min_label"] + 1
    device = get_device("cpu" if args.gpu < 0 else "cuda", gpu_id=None if args.gpu < 0 else args.gpu)

    model_name = f"{args.head}_{args.backbone}"
    ckpt = root / "outputs" / "checkpoints" / f"benchmark_{args.dataset}" / model_name / f"seed{args.seed}" / "best.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = BenchmarkNet(backbone=args.backbone, head=args.head, num_stages=num_stages,
                         num_bins=spec["num_bins"], pretrained=False).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    csv_path = root / "outputs" / "splits" / f"benchmark_{args.dataset}.csv"
    ds = BenchmarkDataset(
        load_benchmark_records(csv_path, args.split),
        data_root=root / "datasets_public" / args.dataset,
        min_label=spec["min_label"], max_label=spec["max_label"], img_size=args.img_size,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_benchmark)

    yt, yp, ft, fp, paths = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            yt.append(batch["wear_pct"].numpy()); yp.append(out["wear_pct"].cpu().numpy())
            ft.append(batch["folder_id"].numpy()); fp.append(out["folder_id_pred"].cpu().numpy())
            paths.extend(batch["image_path"])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    ft, fp = np.concatenate(ft), np.concatenate(fp)
    metrics = compute_metrics(yt, yp, ft, fp)
    metrics["mae_label_units"] = round(metrics["mae"] * num_stages / 100.0, 4)

    out_dir = root / "outputs" / "results" / f"benchmark_{args.dataset}" / model_name / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{args.split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / f"{args.split}_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "folder_id", "wear_pct_true", "wear_pct_pred"])
        for i in range(len(yt)):
            w.writerow([paths[i], int(ft[i]), float(yt[i]), float(yp[i])])

    print(f"{model_name} {args.dataset} {args.split}: MAE={metrics['mae']:.4f} "
          f"({metrics['mae_label_units']:.2f} label-units) QWK={metrics['qwk']:.4f}")
    print(f"Saved: {out_dir / f'{args.split}_metrics.json'}")


if __name__ == "__main__":
    main()
