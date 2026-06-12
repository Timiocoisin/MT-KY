#!/usr/bin/env python3
"""Export per-image predictions for analysis plots."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.data.dataset import BoilerWearDataset, collate_batch
from boilerwear.data.splits import load_split_records
from boilerwear.models.registry import build_model
from boilerwear.utils.config import load_config, merge_dict
from boilerwear.utils.seed import get_device, project_root


def load_cfg(model: str, protocol: str, seed: int) -> dict:
    root = project_root()
    base = load_config(root / "configs/dataset/boilerwear_190.yaml")
    train = load_config(root / "configs/train/default.yaml")
    model_cfg = load_config(root / f"configs/model/{model}.yaml")
    cfg = merge_dict(merge_dict(base, train), model_cfg)
    cfg["project_root"] = str(root)
    cfg["protocol"] = protocol
    cfg["seed"] = seed
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="soformer")
    parser.add_argument("--protocol", type=str, default="protocol2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (-1 for CPU)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Read/write seed{N}_{tag} dirs (e.g. smoke); note tools/eval.py "
                             "already exports predictions automatically — this tool is kept "
                             "for standalone re-export")
    args = parser.parse_args()

    cfg = load_cfg(args.model, args.protocol, args.seed)
    root = project_root()
    data_cfg = cfg["data"]
    data_root = root / data_cfg["data_root"]
    split_name = args.protocol + ("_smoke" if (args.tag and "smoke" in args.tag) else "")
    split_csv = root / data_cfg["splits_dir"] / f"{split_name}.csv"
    records = load_split_records(split_csv, args.split)

    device = get_device("cpu" if args.gpu < 0 else "cuda", gpu_id=None if args.gpu < 0 else args.gpu)
    cfg["model"]["pretrained"] = False  # weights come from the checkpoint
    model = build_model(cfg).to(device)
    seed_dir = f"seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    ckpt = root / cfg["output"]["checkpoints_dir"] / args.protocol / args.model / seed_dir / "best.pt"
    if not ckpt.exists():
        ckpt = root / cfg["output"]["checkpoints_dir"] / args.protocol / args.model / f"seed{args.seed}" / "best.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    ds = BoilerWearDataset(records, data_root, photometric_aug=False, label_mode="hard")
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_batch)

    out_dir = root / "outputs" / "results" / args.protocol / args.model / seed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_predictions.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "folder_id", "wear_pct_true", "wear_pct_pred"])
        with torch.no_grad():
            for batch in loader:
                pred = model(batch["strips"].to(device))["wear_pct"].cpu().numpy()
                for i in range(len(pred)):
                    writer.writerow([
                        batch["image_path"][i],
                        int(batch["folder_id"][i].item()),
                        float(batch["wear_pct"][i].item()),
                        float(pred[i]),
                    ])
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
