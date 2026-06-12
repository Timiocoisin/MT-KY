#!/usr/bin/env python3
"""Train a model from experiment config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.engine.trainer import Trainer
from boilerwear.utils.config import load_config, merge_dict
from boilerwear.utils.seed import parse_gpu_ids, project_root, validate_gpu_ids


def build_experiment_cfg(model: str, protocol: str, seed: int, batch_size: int | None) -> dict:
    root = project_root()
    base = load_config(root / "configs/dataset/boilerwear_190.yaml")
    train = load_config(root / "configs/train/default.yaml")
    model_cfg = load_config(root / f"configs/model/{model}.yaml")
    cfg = merge_dict(merge_dict(base, train), model_cfg)
    cfg["project_root"] = str(root)
    cfg["protocol"] = protocol
    cfg["seed"] = seed
    if batch_size is not None:
        cfg.setdefault("loader", {})["batch_size"] = batch_size
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BoilerWear model")
    parser.add_argument("--config", type=str, default=None, help="Custom experiment YAML")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--protocol", type=str, default="protocol2", choices=["protocol1", "protocol2", "protocol3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Micro batch size per step (default: from config)")
    parser.add_argument("--accum-steps", type=int, default=1,
                        help="Gradient accumulation steps; effective batch = batch-size * accum-steps")
    parser.add_argument("--p2-offset", type=int, default=0,
                        help="Protocol-2 hold-out offset k (test=folder%%10==k); uses protocol2_k{k}.csv when k>0")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: use *_smoke split CSVs and write to seed{N}_smoke dirs")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override train.epochs (e.g. 2 for smoke)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffix outputs/checkpoints dirs with seed{N}_{tag} (e.g. scratch)")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Disable ImageNet pretrained weights (from-scratch cross-check)")
    parser.add_argument("--gpu", type=int, default=0, help="Single GPU id, e.g. 0 (-1 for CPU)")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids for DataParallel training, e.g. 0,1",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument("--progress", action="store_true", help="Force tqdm progress bar")
    progress.add_argument("--no-progress", action="store_true", help="Disable tqdm (for nohup/log files)")
    parser.add_argument(
        "--log-batch-interval",
        type=int,
        default=0,
        help="When --no-progress, log every N batches (0=epoch only)",
    )
    args = parser.parse_args()

    root = project_root()
    if args.config:
        cfg = load_config(root / args.config, root=root / Path(args.config).parent)
        cfg["project_root"] = str(root)
        model_name = cfg["model"]["name"]
    else:
        cfg = build_experiment_cfg(args.model, args.protocol, args.seed, args.batch_size)
        model_name = args.model

    cfg["protocol"] = args.protocol
    cfg["seed"] = args.seed
    if args.batch_size is not None:
        cfg.setdefault("loader", {})["batch_size"] = args.batch_size

    # split file & run tag resolution
    split_name = args.protocol
    if args.protocol == "protocol2" and args.p2_offset % 10 != 0:
        split_name = f"protocol2_k{args.p2_offset % 10}"
    if args.smoke:
        split_name += "_smoke"
    cfg["split_name"] = split_name
    run_tag = args.tag or ("smoke" if args.smoke else "")
    if args.protocol == "protocol2" and args.p2_offset % 10 != 0:
        run_tag = f"k{args.p2_offset % 10}" + (f"_{run_tag}" if run_tag else "")
    cfg["run_tag"] = run_tag

    if args.no_pretrain:
        cfg["model"]["pretrained"] = False
        if not run_tag:
            print("WARNING: --no-pretrain without --tag will overwrite the pretrained checkpoint; "
                  "consider --tag scratch")

    train_cfg = cfg.setdefault("train", {})
    if args.max_epochs is not None:
        train_cfg["epochs"] = args.max_epochs
    if args.accum_steps and args.accum_steps > 1:
        train_cfg["accum_steps"] = args.accum_steps
    if args.gpu < 0 or args.device == "cpu":
        train_cfg["device"] = "cpu"
        train_cfg.pop("gpu_id", None)
        train_cfg.pop("gpu_ids", None)
    elif args.gpus:
        gpu_ids = parse_gpu_ids(args.gpus)
        validate_gpu_ids(gpu_ids)
        train_cfg["device"] = "cuda"
        train_cfg["gpu_ids"] = gpu_ids
        train_cfg["gpu_id"] = gpu_ids[0]
    else:
        train_cfg["device"] = "cuda"
        train_cfg["gpu_ids"] = [args.gpu]
        train_cfg["gpu_id"] = args.gpu

    if args.progress:
        train_cfg["show_progress"] = True
    elif args.no_progress:
        train_cfg["show_progress"] = False
    if args.log_batch_interval > 0:
        train_cfg["log_batch_interval"] = args.log_batch_interval

    seed_dir = f"seed{cfg['seed']}" + (f"_{cfg['run_tag']}" if cfg.get("run_tag") else "")
    out_dir = root / "outputs" / "results" / cfg["protocol"] / model_name / seed_dir
    trainer = Trainer(cfg, out_dir)
    summary = trainer.train()
    best = summary.get("best_val_mae", summary.get("val_metrics", {}).get("mae", "N/A"))
    print(f"Training done. best_val_mae={best}")
    print(f"Log: {out_dir / 'train.log'}")
    print(f"Checkpoint: {summary.get('checkpoint')}")


if __name__ == "__main__":
    main()
