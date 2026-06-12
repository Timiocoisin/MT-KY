#!/usr/bin/env python3
"""Causal AST attention visualization (workflow §5.2 必备支撑):
strip-to-strip attention maps for samples across the wear range."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.data.dataset import BoilerWearDataset, collate_batch
from boilerwear.data.splits import load_split_records
from boilerwear.models.registry import build_model
from boilerwear.utils.config import load_config, merge_dict
from boilerwear.utils.seed import get_device, project_root


class AttnCapture:
    """Force need_weights=True on each layer's self-attention and record maps."""

    def __init__(self, model) -> None:
        self.maps: list[torch.Tensor] = []
        self._orig = []
        for layer in model.ast.encoder.layers:
            attn = layer.self_attn
            orig_forward = attn.forward

            def wrapped(*a, _orig=orig_forward, **kw):
                kw["need_weights"] = True
                kw["average_attn_weights"] = True
                out, weights = _orig(*a, **kw)
                self.maps.append(weights.detach().cpu())
                return out, weights

            self._orig.append((attn, orig_forward))
            attn.forward = wrapped

    def restore(self) -> None:
        for attn, orig in self._orig:
            attn.forward = orig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="soformer")
    parser.add_argument("--protocol", type=str, default="protocol2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--out", type=str, default="outputs/figures/ast_attention.png")
    args = parser.parse_args()

    root = project_root()
    device = get_device("cpu" if args.gpu < 0 else "cuda", gpu_id=None if args.gpu < 0 else args.gpu)
    cfg = merge_dict(merge_dict(load_config(root / "configs/dataset/boilerwear_190.yaml"),
                                load_config(root / "configs/train/default.yaml")),
                     load_config(root / f"configs/model/{args.model}.yaml"))
    cfg["model"]["pretrained"] = False
    model = build_model(cfg).to(device)
    if getattr(model, "ast", None) is None:
        raise SystemExit(f"Model '{args.model}' has no AST module.")
    ckpt = root / "outputs" / "checkpoints" / args.protocol / args.model / f"seed{args.seed}" / "best.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    data_cfg = cfg["data"]
    records = load_split_records(root / data_cfg["splits_dir"] / f"{args.protocol}.csv", args.split)
    records = sorted(records, key=lambda r: r.folder_id)
    # spread samples across the wear range, one image per chosen folder
    folders = sorted({r.folder_id for r in records})
    pick = [folders[int(i)] for i in np.linspace(0, len(folders) - 1, min(args.samples, len(folders)))]
    chosen = []
    for fid in pick:
        chosen.append(next(r for r in records if r.folder_id == fid))

    ds = BoilerWearDataset(chosen, root / data_cfg["data_root"], photometric_aug=False, label_mode="hard")
    batch = collate_batch([ds[i] for i in range(len(ds))])

    cap = AttnCapture(model)
    try:
        # The fused TransformerEncoderLayer fast path bypasses self_attn.forward
        # entirely (so nothing would be captured); force the slow path.
        torch.backends.mha.set_fastpath_enabled(False)
    except AttributeError:
        pass  # older torch without the toggle uses the slow path when hooked
    with torch.no_grad():
        out = model(batch["strips"].to(device))
    try:
        torch.backends.mha.set_fastpath_enabled(True)
    except AttributeError:
        pass
    cap.restore()

    n_layers = len(model.ast.encoder.layers)
    if len(cap.maps) < n_layers:
        raise SystemExit(
            f"Captured {len(cap.maps)} attention maps but the model has {n_layers} AST layers; "
            f"this torch build may not allow disabling the MHA fast path."
        )
    n_samples = len(chosen)
    # maps captured in layer order per forward: [layer0, layer1, ...] each [B, S, S]
    layer_maps = cap.maps[:n_layers]
    fig, axes = plt.subplots(n_samples, n_layers, figsize=(3 * n_layers, 2.6 * n_samples),
                             squeeze=False)
    for i, rec in enumerate(chosen):
        for l in range(n_layers):
            ax = axes[i][l]
            im = ax.imshow(layer_maps[l][i].numpy(), cmap="viridis", vmin=0)
            ax.set_xticks(range(6)); ax.set_yticks(range(6))
            if i == 0:
                ax.set_title(f"Layer {l + 1}")
            if l == 0:
                pred = float(out["wear_pct"][i].item())
                ax.set_ylabel(f"stage {rec.folder_id}\npred {pred:.1f}%", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Causal AST attention — {args.model} ({args.protocol} {args.split})")
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
