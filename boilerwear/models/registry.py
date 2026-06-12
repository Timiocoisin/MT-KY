from __future__ import annotations

from typing import Any

import torch.nn as nn

from boilerwear.models.baselines.strip_regressor import (
    CoralStripModel,
    CornStripModel,
    LDLStripModel,
    RncStripModel,
    StripRegressor,
)
from boilerwear.models.soformer import SOFormer

MODEL_REGISTRY: dict[str, type] = {
    "strip_regressor": StripRegressor,
    "coral": CoralStripModel,
    "corn": CornStripModel,
    "rnc": RncStripModel,
    "ldl": LDLStripModel,
    "soformer": SOFormer,
}


def _parse_ablation(model_cfg: dict[str, Any]) -> dict[str, Any]:
    ab = model_cfg.get("ablation", {})
    if "use_cross_attn" in ab:
        ab.setdefault("use_ast", ab["use_cross_attn"])
    if "use_ordinal" in ab:
        ab.setdefault("use_hod", ab["use_ordinal"])
    return ab


def build_model(cfg: dict[str, Any]) -> nn.Module:
    model_cfg = cfg.get("model", cfg)
    family = model_cfg.get("family", "strip_regressor")
    backbone_kwargs = model_cfg.get("backbone_kwargs", {})

    common = {
        "backbone": model_cfg.get("backbone", "resnet50"),
        "pretrained": model_cfg.get("pretrained", True),
        "input_size": model_cfg.get("backbone_input_size", 256),
        "hidden_dim": model_cfg.get("head_hidden_dim", 512),
        "pool": model_cfg.get("pool", "mean"),
        "freeze_backbone": model_cfg.get("freeze_backbone", False),
        "backbone_kwargs": backbone_kwargs,
    }

    if family in ("strip_regressor", "regressor"):
        return StripRegressor(**common)
    if family == "coral":
        return CoralStripModel(**common, num_bins=model_cfg.get("coral_num_bins", 19))
    if family == "corn":
        return CornStripModel(**common, num_bins=model_cfg.get("coral_num_bins", 19))
    if family == "rnc":
        return RncStripModel(**common)
    if family == "ldl":
        return LDLStripModel(**common)

    if family == "soformer":
        ab = _parse_ablation(model_cfg)
        return SOFormer(
            backbone=model_cfg.get("backbone", "axial_strip_cnn"),
            pretrained=model_cfg.get("pretrained", False),
            input_size=model_cfg.get("backbone_input_size", 256),
            hidden_dim=model_cfg.get("head_hidden_dim", 512),
            backbone_kwargs=backbone_kwargs,
            num_strips=model_cfg.get("num_strips", 6),
            num_bins=model_cfg.get("coral_num_bins", 19),
            fusion_layers=model_cfg.get("fusion_layers", 2),
            fusion_heads=model_cfg.get("fusion_heads", 8),
            infer_alpha=model_cfg.get("infer_alpha", 0.0),
            infer_beta=model_cfg.get("infer_beta", 0.0),
            use_ast=ab.get("use_ast", ab.get("use_cross_attn", True)),
            use_ape=ab.get("use_ape", True),
            use_causal=ab.get("use_causal", True),
            use_adm=ab.get("use_adm", True),
            use_hod=ab.get("use_hod", ab.get("use_ordinal", True)),
            use_uncertainty=ab.get("use_uncertainty", True),
        )

    raise ValueError(f"Unknown model family: {family}")
