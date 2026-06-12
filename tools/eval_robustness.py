#!/usr/bin/env python3
"""Eval-time robustness suite (workflow §8): controlled perturbations on the
P2 test split, evaluated for every trained model — no retraining.

Perturbations (8-bit semantics): brightness/contrast ±10%/±20%, Gaussian
noise sigma=5/10, slight perspective <=2% of image extent, JPEG Q=90/70.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.data.splits import load_split_records
from boilerwear.data.transforms import normalize_strips, split_into_strips
from boilerwear.engine.evaluator import evaluate_model
from boilerwear.models.registry import build_model
from boilerwear.utils.config import load_config, merge_dict
from boilerwear.utils.seed import get_device, project_root

PERTURB_LEVELS: dict[str, list] = {
    "clean": [0],
    "brightness": [-20, -10, 10, 20],     # percent
    "contrast": [-20, -10, 10, 20],       # percent
    "gaussian_noise": [5, 10],            # sigma on 8-bit scale
    "perspective": [2],                    # percent of image extent
    "jpeg": [90, 70],                      # quality
}


def perturb_image(img: Image.Image, kind: str, level, rng: np.random.Generator) -> Image.Image:
    if kind == "clean":
        return img
    if kind == "brightness":
        return TF.adjust_brightness(img, 1.0 + level / 100.0)
    if kind == "contrast":
        return TF.adjust_contrast(img, 1.0 + level / 100.0)
    if kind == "gaussian_noise":
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + rng.normal(0.0, float(level), arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind == "perspective":
        w, h = img.size
        dx, dy = w * level / 100.0, h * level / 100.0
        start = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        end = [[int(rng.uniform(0, dx)), int(rng.uniform(0, dy))] for _ in range(4)]
        end = [[s[0] + e[0] * (1 if s[0] == 0 else -1), s[1] + e[1] * (1 if s[1] == 0 else -1)]
               for s, e in zip(start, end)]
        return TF.perspective(img, start, end, fill=[0, 0, 0])
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(level))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(f"Unknown perturbation: {kind}")


class PerturbedWearDataset(Dataset):
    """Loads panorama -> perturb -> resize -> 6-strip -> normalize (deterministic per index)."""

    def __init__(self, records, data_root: Path, kind: str, level, seed: int = 0,
                 num_strips: int = 6, strip_size: int = 256) -> None:
        self.records = records
        self.data_root = Path(data_root)
        self.kind, self.level, self.seed = kind, level, seed
        self.num_strips, self.strip_size = num_strips, strip_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        image = Image.open(self.data_root / rec.image_path).convert("RGB")
        image = perturb_image(image, self.kind, self.level, rng)
        tensor = TF.to_tensor(image)
        expected = (self.strip_size, self.num_strips * self.strip_size)
        if tensor.shape[-2:] != expected:
            tensor = TF.resize(tensor, list(expected), antialias=True)
        strips = split_into_strips(tensor, self.num_strips, self.strip_size)
        strips = normalize_strips(strips, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return {
            "strips": strips,
            "wear_pct": torch.tensor(rec.wear_pct, dtype=torch.float32),
            "folder_id": torch.tensor(rec.folder_id, dtype=torch.long),
            "image_path": rec.image_path,
        }


def discover_models(root: Path, protocol: str, seed: int) -> list[str]:
    base = root / "outputs" / "checkpoints" / protocol
    names = []
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if (d / f"seed{seed}" / "best.pt").exists() and (root / f"configs/model/{d.name}.yaml").exists():
                names.append(d.name)
    return names


def main() -> None:
    from boilerwear.data.dataset import collate_batch

    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=str, default="protocol2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    parser.add_argument("--models", nargs="*", default=None,
                        help="default: every model with a seed{N} checkpoint")
    parser.add_argument("--perturb", nargs="+",
                        default=["brightness", "contrast", "gaussian_noise", "perspective", "jpeg"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=str, default="outputs/results/robustness_p2.csv")
    args = parser.parse_args()

    root = project_root()
    device = get_device("cpu" if args.gpu < 0 else "cuda", gpu_id=None if args.gpu < 0 else args.gpu)
    models = args.models or discover_models(root, args.protocol, args.seed)
    if not models:
        raise SystemExit("No trained checkpoints found; run training first.")
    print(f"Models: {models}")

    base_cfg = merge_dict(load_config(root / "configs/dataset/boilerwear_190.yaml"),
                          load_config(root / "configs/train/default.yaml"))
    split_csv = root / base_cfg["data"]["splits_dir"] / f"{args.protocol}.csv"
    records = load_split_records(split_csv, args.split)
    data_root = root / base_cfg["data"]["data_root"]

    kinds = ["clean"] + [k for k in args.perturb if k in PERTURB_LEVELS]
    rows = []
    for name in models:
        cfg = merge_dict(base_cfg, load_config(root / f"configs/model/{name}.yaml"))
        cfg["model"]["pretrained"] = False
        if cfg["model"].get("family") == "hog_lr":
            print(f"  {name}: hog_lr is sklearn-based; skipped here (perturb via its own eval if needed)")
            continue
        model = build_model(cfg).to(device)
        ckpt = root / "outputs" / "checkpoints" / args.protocol / name / f"seed{args.seed}" / "best.pt"
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        for kind in kinds:
            for level in PERTURB_LEVELS[kind]:
                ds = PerturbedWearDataset(records, data_root, kind, level, seed=args.seed)
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=collate_batch)
                m = evaluate_model(model, loader, device)
                rows.append({"model": name, "perturbation": kind, "level": level, **m})
                print(f"  {name:24s} {kind:16s} level={level:>4}  MAE={m['mae']:.4f} QWK={m['qwk']:.4f}")

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
