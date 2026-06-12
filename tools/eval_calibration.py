#!/usr/bin/env python3
"""Uncertainty calibration for heteroscedastic models (workflow §6.2):
Gaussian NLL, coverage-based ECE, reliability diagram.

The model must output log_var (SOFormer's Uncertainty Head). For each nominal
confidence p, the predicted Gaussian interval is pred ± z_{(1+p)/2}·sigma;
ECE = mean |observed coverage − p|. A well-calibrated head tracks the diagonal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.data.dataset import BoilerWearDataset, collate_batch
from boilerwear.data.splits import load_split_records
from boilerwear.models.registry import build_model
from boilerwear.utils.config import load_config, merge_dict
from boilerwear.utils.seed import get_device, project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="soformer")
    parser.add_argument("--protocol", type=str, default="protocol2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=str, default=None, help="reliability figure path")
    args = parser.parse_args()

    root = project_root()
    device = get_device("cpu" if args.gpu < 0 else "cuda", gpu_id=None if args.gpu < 0 else args.gpu)
    cfg = merge_dict(merge_dict(load_config(root / "configs/dataset/boilerwear_190.yaml"),
                                load_config(root / "configs/train/default.yaml")),
                     load_config(root / f"configs/model/{args.model}.yaml"))
    cfg["model"]["pretrained"] = False
    model = build_model(cfg).to(device)
    ckpt = root / "outputs" / "checkpoints" / args.protocol / args.model / f"seed{args.seed}" / "best.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    data_cfg = cfg["data"]
    split_csv = root / data_cfg["splits_dir"] / f"{args.protocol}.csv"
    records = load_split_records(split_csv, args.split)
    ds = BoilerWearDataset(records, root / data_cfg["data_root"], photometric_aug=False,
                           label_mode="hard")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    preds, trues, sigmas = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["strips"].to(device))
            if "log_var" not in out:
                raise SystemExit(f"Model '{args.model}' has no uncertainty head (log_var missing).")
            preds.append(out["wear_pct"].cpu().numpy())
            sigmas.append(torch.exp(0.5 * out["log_var"]).cpu().numpy())
            trues.append(batch["wear_pct"].numpy())
    pred = np.concatenate(preds); true = np.concatenate(trues)
    sigma = np.clip(np.concatenate(sigmas), 1e-4, None)
    err = pred - true

    nll = float(np.mean(0.5 * (np.log(2 * np.pi * sigma**2) + err**2 / sigma**2)))
    levels = np.arange(0.1, 1.0, 0.1)
    observed = []
    for p in levels:
        z = norm.ppf((1 + p) / 2)
        observed.append(float(np.mean(np.abs(err) <= z * sigma)))
    observed = np.array(observed)
    ece = float(np.mean(np.abs(observed - levels)))

    result = {
        "model": args.model, "protocol": args.protocol, "split": args.split, "seed": args.seed,
        "n_samples": int(len(err)), "nll": round(nll, 4), "ece": round(ece, 4),
        "mean_sigma": round(float(sigma.mean()), 4), "mean_abs_err": round(float(np.abs(err).mean()), 4),
        "coverage": {f"{p:.1f}": round(o, 4) for p, o in zip(levels, observed)},
    }
    out_dir = root / "outputs" / "results" / args.protocol / args.model / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{args.split}_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(levels, observed, "o-", label=args.model)
    ax.set_xlabel("Nominal confidence"); ax.set_ylabel("Observed coverage")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Reliability — {args.model} ({args.protocol} {args.split})\n"
                 f"ECE={ece:.4f}  NLL={nll:.3f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig_path = root / (args.out or f"outputs/figures/reliability_{args.model}.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(fig_path, dpi=150)
    print(f"ECE={ece:.4f}  NLL={nll:.4f}  -> {fig_path}")


if __name__ == "__main__":
    main()
