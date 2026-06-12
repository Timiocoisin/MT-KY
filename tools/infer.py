#!/usr/bin/env python3
"""Single-image wear inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.data.transforms import normalize_strips, split_into_strips
from boilerwear.models.registry import build_model
from boilerwear.utils.config import load_config
from boilerwear.utils.seed import get_device, project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment/p2_soformer.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    root = project_root()
    cfg = load_config(root / args.config, root=root / Path(args.config).parent)
    cfg["project_root"] = str(root)
    device = get_device("cuda")

    cfg["model"]["pretrained"] = False  # weights come from the checkpoint
    model = build_model(cfg).to(device)
    state = torch.load(root / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    tensor = TF.to_tensor(image)
    strips = split_into_strips(tensor, num_strips=6, strip_size=256)
    data_cfg = cfg["data"]
    strips = normalize_strips(
        strips,
        data_cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
        data_cfg.get("normalize_std", [0.229, 0.224, 0.225]),
    )
    strips = strips.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(strips)
    wear = float(out["wear_pct"][0].item())
    folder = int(round(wear / 100.0 * 190))
    print(f"Predicted wear: {wear:.2f}%  (stage {folder}/190)")


if __name__ == "__main__":
    main()
