from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from boilerwear.data.splits import NUM_STAGES
from boilerwear.models.strip_encoder import StripEncoder, StripPoolHead


class StripRegressor(nn.Module):
    """ResNet / EfficientNet strip regression baseline."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        input_size: int = 256,
        hidden_dim: int = 512,
        pool: str = "mean",
        freeze_backbone: bool = False,
        backbone_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = StripEncoder(
            backbone=backbone,
            pretrained=pretrained,
            input_size=input_size,
            out_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=backbone_kwargs,
        )
        self.head = StripPoolHead(hidden_dim, hidden_dim, pool=pool)

    def forward(self, strips: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(strips)
        wear_pct = self.head(feats)
        return {"wear_pct": wear_pct, "strip_feats": feats}


class CoralStripModel(nn.Module):
    """CORAL ordinal regression with strip encoder."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        input_size: int = 256,
        hidden_dim: int = 512,
        num_bins: int = 19,
        pool: str = "mean",
        freeze_backbone: bool = False,
        backbone_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.encoder = StripEncoder(
            backbone,
            pretrained,
            input_size,
            hidden_dim,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=backbone_kwargs,
        )
        in_dim = hidden_dim
        self.pool = pool
        self.ordinal = nn.Linear(in_dim, num_bins - 1)

    def forward(self, strips: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(strips)
        if self.pool == "mean":
            pooled = feats.mean(dim=1)
        else:
            pooled = feats.max(dim=1).values
        logits = self.ordinal(pooled)
        wear_pct = coral_logits_to_wear(logits, self.num_bins)
        return {"wear_pct": wear_pct, "coral_logits": logits, "strip_feats": feats}


class LDLStripModel(nn.Module):
    """Label distribution learning head over 190 stages."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        input_size: int = 256,
        hidden_dim: int = 512,
        num_stages: int = NUM_STAGES,
        pool: str = "mean",
        freeze_backbone: bool = False,
        backbone_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.encoder = StripEncoder(
            backbone,
            pretrained,
            input_size,
            hidden_dim,
            freeze_backbone=freeze_backbone,
            backbone_kwargs=backbone_kwargs,
        )
        self.pool = pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_stages),
        )
        wear_levels = torch.arange(1, num_stages + 1, dtype=torch.float32) / num_stages * 100.0
        self.register_buffer("wear_levels", wear_levels)

    def forward(self, strips: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(strips)
        if self.pool == "mean":
            pooled = feats.mean(dim=1)
        else:
            pooled = feats.max(dim=1).values
        logits = self.classifier(pooled)
        dist = torch.softmax(logits, dim=-1)
        wear_pct = (dist * self.wear_levels).sum(dim=-1)
        return {"wear_pct": wear_pct, "wear_dist_pred": dist, "strip_feats": feats}


def coral_logits_to_wear(logits: torch.Tensor, num_bins: int = 19) -> torch.Tensor:
    """Map CORAL logits to wear %.

    Expected (continuous) level = 1 + sum_k sigmoid(logit_k); mapping the level
    to its bin midpoint, wear = (level - 0.5) / num_bins * 100. A perfect model
    on bin k thus predicts the bin-k midpoint instead of the previous version's
    left-edge estimate (which was biased low by a full half-bin, ~2.6%).
    """
    probs = torch.sigmoid(logits)
    expected_level = 1.0 + probs.sum(dim=1)
    wear = (expected_level - 0.5) / num_bins * 100.0
    return wear.clamp(0.0, 100.0)


def corn_logits_to_wear(logits: torch.Tensor, num_bins: int = 19) -> torch.Tensor:
    """Map CORN logits to wear %.

    P(level > k) = prod_{j<=k} sigmoid(logit_j) (conditional chain rule);
    expected level = 1 + sum_k P(level > k); midpoint mapping as in CORAL.
    """
    cond_probs = torch.sigmoid(logits)
    cum_probs = torch.cumprod(cond_probs, dim=1)
    expected_level = 1.0 + cum_probs.sum(dim=1)
    wear = (expected_level - 0.5) / num_bins * 100.0
    return wear.clamp(0.0, 100.0)


class CornStripModel(CoralStripModel):
    """CORN (Conditional Ordinal Regression, 2022) — same architecture as the
    CORAL baseline; only the loss (conditional training) and the logits->wear
    decoding differ, exactly as specified in workflow v3.2 §4 (#8)."""

    def forward(self, strips: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(strips)
        if self.pool == "mean":
            pooled = feats.mean(dim=1)
        else:
            pooled = feats.max(dim=1).values
        logits = self.ordinal(pooled)
        wear_pct = corn_logits_to_wear(logits, self.num_bins)
        return {"wear_pct": wear_pct, "corn_logits": logits, "strip_feats": feats}


class RncStripModel(StripRegressor):
    """Rank-N-Contrast baseline (NeurIPS 2023, workflow v3.2 §4 #9).

    Same ResNet50 strip encoder + regression head as the plain baseline; the
    pooled strip feature is additionally exposed so the LossBuilder can apply
    the RnC contrastive term. NOTE: the official protocol is two-stage
    (contrastive pre-training, then a frozen-encoder linear probe); under the
    unified single-loop training budget of v3.2 §4 we use the one-stage joint
    variant (L1 + lambda * RnC), which the paper also reports as an effective
    regularizer — state this in the paper's Implementation Details.
    """

    def forward(self, strips: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(strips)
        pooled = feats.mean(dim=1) if self.head.pool == "mean" else feats.max(dim=1).values
        wear_pct = self.head.regressor(pooled).squeeze(-1)
        return {"wear_pct": wear_pct, "pooled_feats": pooled, "strip_feats": feats}
