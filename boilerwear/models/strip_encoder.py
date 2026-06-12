from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import timm

from boilerwear.models.axial_strip_cnn import CUSTOM_BACKBONE_NAMES, build_axial_strip_cnn


class StripEncoder(nn.Module):
    """Shared-weight encoder for each axial strip."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        input_size: int = 256,
        out_dim: int = 512,
        freeze_backbone: bool = False,
        backbone_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.backbone_name = backbone
        extra = backbone_kwargs or {}

        if backbone in CUSTOM_BACKBONE_NAMES:
            if pretrained:
                import warnings

                warnings.warn(
                    f"Custom backbone '{backbone}' has no pretrained weights; using random init.",
                    stacklevel=2,
                )
            self.backbone = build_axial_strip_cnn(backbone, **extra)
            feat_dim = self.backbone.num_features
        else:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            feat_dim = self.backbone.num_features

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, strips: torch.Tensor) -> torch.Tensor:
        """
        strips: [B, S, 3, H, W]
        returns: [B, S, out_dim]
        """
        b, s, c, h, w = strips.shape
        x = strips.view(b * s, c, h, w)
        if (h, w) != (self.input_size, self.input_size):
            x = nn.functional.interpolate(
                x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
            )
        feats = self.backbone(x)
        feats = self.proj(feats)
        return feats.view(b, s, -1)


class StripPoolHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, pool: str = "mean") -> None:
        super().__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, strip_feats: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            pooled = strip_feats.mean(dim=1)
        elif self.pool == "max":
            pooled = strip_feats.max(dim=1).values
        else:
            raise ValueError(f"Unknown pool: {self.pool}")
        return self.regressor(pooled).squeeze(-1)
