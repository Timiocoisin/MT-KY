from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from boilerwear.data.splits import NUM_STAGES
from boilerwear.models.baselines.strip_regressor import coral_logits_to_wear
from boilerwear.models.strip_encoder import StripEncoder


class AxialProgressionEmbedding(nn.Module):
    """Learnable axial position encoding for strip tokens (wear progresses along width)."""

    def __init__(self, num_strips: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, num_strips, dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, strip_feats: torch.Tensor) -> torch.Tensor:
        return strip_feats + self.embedding


class CausalAxialStripTransformer(nn.Module):
    """
    Causal AST: axial self-attention with a unidirectional mask.
    Each strip attends only to itself and earlier strips along the wear progression axis.
    """

    def __init__(
        self,
        dim: int,
        num_strips: int = 6,
        num_layers: int = 2,
        num_heads: int = 8,
        use_ape: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.num_strips = num_strips
        self.use_ape = use_ape
        self.causal = causal
        self.ape = AxialProgressionEmbedding(num_strips, dim) if use_ape else None
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

    def forward(self, strip_feats: torch.Tensor) -> torch.Tensor:
        x = self.ape(strip_feats) if self.ape is not None else strip_feats
        mask = self._causal_mask(x.size(1), x.device) if self.causal else None
        return self.encoder(x, mask=mask)


class MultiScaleAxialDifferentialModule(nn.Module):
    """
    MS-ADM: multi-scale axial differential aggregation.
    First-order strip differences approximate dw/dx; second-order differences capture
    local acceleration of degradation along the wear axis.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.diff1_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.diff2_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, strip_feats: torch.Tensor) -> torch.Tensor:
        mean_ctx = strip_feats.mean(dim=1)

        if strip_feats.size(1) > 1:
            d1 = strip_feats[:, 1:] - strip_feats[:, :-1]
            d1_ctx = self.diff1_proj(d1.mean(dim=1))
        else:
            d1_ctx = torch.zeros_like(mean_ctx)

        if strip_feats.size(1) > 2:
            d2 = d1[:, 1:] - d1[:, :-1]
            d2_ctx = self.diff2_proj(d2.mean(dim=1))
            diff_ctx = self.fusion(torch.cat([d1_ctx, d2_ctx], dim=-1))
        else:
            diff_ctx = d1_ctx

        gate = self.gate(torch.cat([mean_ctx, diff_ctx], dim=-1))
        return mean_ctx + gate * diff_ctx


class HierarchicalOrdinalDistributionHead(nn.Module):
    """
    HOD-Head: coarse CORAL deciles + fine 190-way distribution for LDL alignment.
    """

    def __init__(self, in_dim: int, num_bins: int = 19, num_stages: int = NUM_STAGES) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.num_stages = num_stages
        self.coral = nn.Linear(in_dim, num_bins - 1)
        self.fine = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, num_stages),
        )
        wear_levels = torch.arange(1, num_stages + 1, dtype=torch.float32) / num_stages * 100.0
        self.register_buffer("wear_levels", wear_levels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        coral_logits = self.coral(x)
        fine_logits = self.fine(x)
        fine_dist = torch.softmax(fine_logits, dim=-1)
        ord_wear = coral_logits_to_wear(coral_logits, self.num_bins)
        dist_wear = (fine_dist * self.wear_levels).sum(dim=-1)
        return {
            "coral_logits": coral_logits,
            "wear_dist_pred": fine_dist,
            "wear_pct_ord": ord_wear,
            "wear_pct_dist": dist_wear,
        }


class SOFormer(nn.Module):
    """SOFormer: AxialStripCNN + Causal AST + MS-ADM + HOD-Head for panoramic wear."""

    def __init__(
        self,
        backbone: str = "axial_strip_cnn",
        pretrained: bool = False,
        input_size: int = 256,
        hidden_dim: int = 512,
        backbone_kwargs: dict[str, Any] | None = None,
        num_strips: int = 6,
        num_bins: int = 19,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        infer_alpha: float = 0.0,
        infer_beta: float = 0.0,
        use_ast: bool = True,
        use_ape: bool = True,
        use_causal: bool = True,
        use_adm: bool = True,
        use_hod: bool = True,
        use_uncertainty: bool = True,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.infer_alpha = infer_alpha
        self.infer_beta = infer_beta
        self.use_hod = use_hod
        self.encoder = StripEncoder(
            backbone,
            pretrained,
            input_size,
            hidden_dim,
            backbone_kwargs=backbone_kwargs,
        )
        self.ast = (
            CausalAxialStripTransformer(
                hidden_dim, num_strips, fusion_layers, fusion_heads, use_ape, causal=use_causal
            )
            if use_ast
            else None
        )
        self.adm = MultiScaleAxialDifferentialModule(hidden_dim) if use_adm else None
        self.hod = HierarchicalOrdinalDistributionHead(hidden_dim, num_bins) if use_hod else None
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.uncertainty_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            if use_uncertainty
            else None
        )

    def _aggregate(self, strip_feats: torch.Tensor) -> torch.Tensor:
        if self.ast is not None:
            strip_feats = self.ast(strip_feats)
        if self.adm is not None:
            return self.adm(strip_feats)
        return strip_feats.mean(dim=1)

    def forward(self, strips: torch.Tensor) -> dict[str, torch.Tensor]:
        strip_feats = self.encoder(strips)
        pooled = self._aggregate(strip_feats)
        reg_wear = self.reg_head(pooled).squeeze(-1)

        out: dict[str, torch.Tensor] = {
            "wear_pct_reg": reg_wear,
            "strip_feats": strip_feats,
        }

        if self.use_hod and self.hod is not None:
            hod = self.hod(pooled)
            out.update(hod)
            alpha, beta = self.infer_alpha, self.infer_beta
            reg_weight = max(0.0, 1.0 - alpha - beta)
            wear_pct = alpha * hod["wear_pct_ord"] + beta * hod["wear_pct_dist"] + reg_weight * reg_wear
        else:
            wear_pct = reg_wear

        out["wear_pct"] = wear_pct.clamp(0.0, 100.0)

        if self.uncertainty_head is not None:
            log_var = self.uncertainty_head(pooled).squeeze(-1)
            out["log_var"] = log_var
            out["uncertainty"] = torch.exp(0.5 * log_var)

        folder_pred = (out["wear_pct"] / 100.0 * NUM_STAGES).round().clamp(1, NUM_STAGES)
        out["folder_id_pred"] = folder_pred
        return out
