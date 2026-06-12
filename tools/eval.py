#!/usr/bin/env python3
"""Evaluate a trained model on val/test split."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.data.dataset import BoilerWearDataset, collate_batch
from boilerwear.data.splits import load_split_records
from boilerwear.engine.evaluator import evaluate_hog_lr, evaluate_model
from boilerwear.losses import LossBuilder
from boilerwear.models.registry import build_model
from boilerwear.utils.config import load_config, merge_dict
from boilerwear.utils.seed import get_device, project_root


def load_cfg(args) -> dict:
    root = project_root()
    if args.config:
        cfg = load_config(root / args.config, root=root / Path(args.config).parent)
    else:
        base = load_config(root / "configs/dataset/boilerwear_190.yaml")
        train = load_config(root / "configs/train/default.yaml")
        model = load_config(root / f"configs/model/{args.model}.yaml")
        cfg = merge_dict(merge_dict(base, train), model)
    cfg["project_root"] = str(root)
    cfg["protocol"] = args.protocol
    cfg["seed"] = args.seed
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--protocol", type=str, default="protocol2")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (-1 for CPU)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--p2-offset", type=int, default=0,
                        help="Protocol-2 hold-out offset k; reads protocol2_k{k}.csv and seed{N}_k{k} ckpt when k>0")
    parser.add_argument("--tag", type=str, default=None,
                        help="Result dir suffix seed{N}_{tag}; 'smoke' also switches to *_smoke split CSV. "
                             "If a tagged checkpoint exists it is used, otherwise falls back to the untagged one "
                             "(e.g. --tag fuse reuses Full weights)")
    parser.add_argument("--infer-alpha", type=float, default=None,
                        help="Override SOFormer HOD-ord fusion weight at inference (soformer_fuse)")
    parser.add_argument("--infer-beta", type=float, default=None,
                        help="Override SOFormer HOD-dist fusion weight at inference (soformer_fuse)")
    args = parser.parse_args()

    cfg = load_cfg(args)
    root = project_root()
    model_name = cfg["model"]["name"]
    data_cfg = cfg["data"]
    data_root = root / data_cfg["data_root"]

    split_name = args.protocol
    if args.protocol == "protocol2" and args.p2_offset % 10 != 0:
        split_name = f"protocol2_k{args.p2_offset % 10}"
    tag = args.tag or ""
    if args.protocol == "protocol2" and args.p2_offset % 10 != 0:
        tag = f"k{args.p2_offset % 10}" + (f"_{tag}" if tag else "")
    if "smoke" in tag:
        split_name += "_smoke"
    split_csv = root / data_cfg["splits_dir"] / f"{split_name}.csv"
    records = load_split_records(split_csv, args.split)

    seed_dir = f"seed{args.seed}" + (f"_{tag}" if tag else "")
    out_dir = root / "outputs" / "results" / args.protocol / model_name / seed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def resolve_ckpt(filename: str) -> Path:
        base = root / cfg["output"]["checkpoints_dir"] / args.protocol / model_name
        tagged = base / seed_dir / filename
        if tagged.exists():
            return tagged
        return base / f"seed{args.seed}" / filename

    if cfg["model"].get("family") == "hog_lr":
        ckpt = resolve_ckpt("hog_lr.pkl")
        with ckpt.open("rb") as f:
            model = pickle.load(f)
        metrics, predictions = evaluate_hog_lr(model, records, data_root, return_predictions=True)
    else:
        gpu_id = None if args.gpu < 0 or args.device == "cpu" else args.gpu
        device = get_device(args.device, gpu_id=gpu_id)
        cfg["model"]["pretrained"] = False  # weights come from the checkpoint; avoid hub download
        model = build_model(cfg).to(device)
        ckpt = resolve_ckpt("best.pt")
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        if args.infer_alpha is not None and hasattr(model, "infer_alpha"):
            model.infer_alpha = args.infer_alpha
        if args.infer_beta is not None and hasattr(model, "infer_beta"):
            model.infer_beta = args.infer_beta
        ds = BoilerWearDataset(
            records,
            data_root,
            num_strips=data_cfg.get("num_strips", 6),
            strip_size=data_cfg.get("strip_size", 256),
            photometric_aug=False,
            label_mode=cfg["model"].get("label_mode", data_cfg.get("label_mode", "hard")),
            ldl_sigma_folders=data_cfg.get("ldl_sigma_folders", 1.0),
        )
        loader = DataLoader(
            ds,
            batch_size=cfg.get("loader", {}).get("batch_size", 8),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )
        metrics, predictions = evaluate_model(model, loader, device, LossBuilder(cfg), return_predictions=True)

    metrics_path = out_dir / f"{args.split}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 逐样本预测(显著性检验 tools/significance_test.py 依赖此文件)
    pred_path = out_dir / f"{args.split}_predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "folder_id", "wear_pct_true", "wear_pct_pred"])
        for i in range(len(predictions["wear_pct_true"])):
            writer.writerow([
                predictions["image_path"][i] if predictions["image_path"] else "",
                predictions["folder_id"][i],
                predictions["wear_pct_true"][i],
                predictions["wear_pct_pred"][i],
            ])

    print(
        f"{model_name} {args.protocol} {args.split}: "
        f"MAE={metrics['mae']:.4f} QWK={metrics['qwk']:.4f} Acc@5%={metrics['acc_at_5']:.4f}"
    )
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
