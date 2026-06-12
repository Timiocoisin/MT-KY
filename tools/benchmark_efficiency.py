#!/usr/bin/env python3
"""Efficiency table (workflow §6.3): Params (M), FLOPs (G), single-image
latency in ms (batch=1, warmup then median over repeated runs)."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.models.registry import build_model
from boilerwear.utils.config import load_config, merge_dict
from boilerwear.utils.seed import get_device, project_root


def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops_g(model: torch.nn.Module, x: torch.Tensor) -> float | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fcm:
            model(x)
        return fcm.get_total_flops() / 1e9
    except Exception as e:  # pragma: no cover - depends on torch version/ops
        print(f"    FLOPs counter failed ({type(e).__name__}); reporting N/A")
        return None


def measure_latency_ms(model: torch.nn.Module, x: torch.Tensor, device, warmup=10, runs=50) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def hog_lr_row(root: Path) -> dict:
    """HOG+Ridge: parameter count = Ridge weights on HOG dim; latency = feature
    extraction + predict on one 256x1536 panorama."""
    try:
        from boilerwear.models.baselines.hog_lr import HogLRModel
        m = HogLRModel()
        img = (np.random.rand(256, 1536, 3) * 255).astype(np.uint8)
        feat_dim = m._extract(img).shape[0]
        m.fit([img, img], np.array([10.0, 20.0]))
        t0 = time.perf_counter()
        for _ in range(5):
            m.predict([img])
        lat = (time.perf_counter() - t0) / 5 * 1000.0
        return {"model": "hog_lr", "params_m": round((feat_dim + 1) / 1e6, 4),
                "flops_g": "", "latency_ms": round(lat, 2), "device": "cpu"}
    except Exception as e:
        return {"model": "hog_lr", "params_m": "", "flops_g": "", "latency_ms": "",
                "device": f"error: {e}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    parser.add_argument("--out", type=str, default="outputs/results/efficiency.csv")
    args = parser.parse_args()

    root = project_root()
    device = get_device("cpu" if args.gpu < 0 else "cuda", gpu_id=None if args.gpu < 0 else args.gpu)
    base = merge_dict(load_config(root / "configs/dataset/boilerwear_190.yaml"),
                      load_config(root / "configs/train/default.yaml"))
    rows = []
    for name in args.models:
        if name == "hog_lr":
            rows.append(hog_lr_row(root))
            print(f"  hog_lr: {rows[-1]}")
            continue
        cfg = merge_dict(base, load_config(root / f"configs/model/{name}.yaml"))
        cfg["model"]["pretrained"] = False  # structure only; no hub download
        model = build_model(cfg).to(device).eval()
        strip = cfg["data"].get("strip_size", 256)
        n_strips = cfg["data"].get("num_strips", 6)
        x = torch.randn(1, n_strips, 3, strip, strip, device=device)
        params = count_params_m(model)
        flops = count_flops_g(model, x)
        lat = measure_latency_ms(model, x, device)
        rows.append({"model": name, "params_m": round(params, 2),
                     "flops_g": round(flops, 2) if flops is not None else "",
                     "latency_ms": round(lat, 2), "device": str(device)})
        print(f"  {name:24s} params={params:7.2f}M  flops={flops if flops else 'N/A'}G  latency={lat:.2f}ms")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "params_m", "flops_g", "latency_ms", "device"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
