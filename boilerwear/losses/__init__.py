from __future__ import annotations

from typing import Any

import torch

from boilerwear.losses.regression import (
    coral_loss,
    corn_loss,
    heteroscedastic_loss,
    ldl_kl_loss,
    monotonic_pairwise_loss,
    rnc_loss,
    smooth_l1_loss,
)


class LossBuilder:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.model_cfg = cfg.get("model", {})
        self.family = self.model_cfg.get("family", "strip_regressor")
        self.num_bins = self.model_cfg.get("coral_num_bins", 19)
        weights = self.model_cfg.get("loss_weights", {})
        self.w_ord = weights.get("ord", 0.3)
        self.w_reg = weights.get("reg", 0.5)
        self.w_ldl = weights.get("ldl", 0.25)
        self.w_mono = weights.get("mono", 0.1)
        self.w_cal = weights.get("cal", 0.1)
        self.w_rnc = weights.get("rnc", 1.0)
        self.rnc_temperature = self.model_cfg.get("rnc_temperature", 2.0)

    def _soformer_loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        wear: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logs: dict[str, float] = {}
        reg_loss = smooth_l1_loss(outputs["wear_pct_reg"], wear)
        total = self.w_reg * reg_loss
        logs["loss_reg"] = float(reg_loss.item())

        if "coral_logits" in outputs:
            ord_loss = coral_loss(outputs["coral_logits"], wear, self.num_bins)
            total = total + self.w_ord * ord_loss
            logs["loss_ord"] = float(ord_loss.item())

        if "wear_dist_pred" in outputs and "wear_dist" in batch:
            ldl_loss = ldl_kl_loss(outputs["wear_dist_pred"], batch["wear_dist"])
            total = total + self.w_ldl * ldl_loss
            logs["loss_ldl"] = float(ldl_loss.item())

        if self.w_mono > 0:
            mono_pred = outputs.get("wear_pct_reg", outputs["wear_pct"])
            mono = monotonic_pairwise_loss(mono_pred, batch["folder_id"])
            total = total + self.w_mono * mono
            logs["loss_mono"] = float(mono.item())

        if self.w_cal > 0 and "log_var" in outputs:
            cal = heteroscedastic_loss(outputs["wear_pct_reg"], wear, outputs["log_var"])
            total = total + self.w_cal * cal
            logs["loss_cal"] = float(cal.item())

        logs["loss_total"] = float(total.item())
        return total, logs

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        wear = batch["wear_pct"]
        logs: dict[str, float] = {}

        if self.family == "ldl":
            loss = ldl_kl_loss(outputs["wear_dist_pred"], batch["wear_dist"])
            logs["loss_ldl"] = float(loss.item())
            return loss, logs

        if self.family == "coral":
            loss = coral_loss(outputs["coral_logits"], wear, self.num_bins)
            logs["loss_coral"] = float(loss.item())
            return loss, logs

        if self.family == "corn":
            loss = corn_loss(outputs["corn_logits"], wear, self.num_bins)
            logs["loss_corn"] = float(loss.item())
            return loss, logs

        if self.family == "rnc":
            import torch.nn.functional as _F
            reg = _F.l1_loss(outputs["wear_pct"], wear)  # official stage-2 uses L1
            rnc = rnc_loss(outputs["pooled_feats"], wear, self.rnc_temperature)
            loss = self.w_reg * reg + self.w_rnc * rnc
            logs["loss_reg"] = float(reg.item())
            logs["loss_rnc"] = float(rnc.item())
            logs["loss_total"] = float(loss.item())
            return loss, logs

        if self.family == "soformer":
            return self._soformer_loss(outputs, batch, wear)

        loss = smooth_l1_loss(outputs["wear_pct"], wear)
        logs["loss_reg"] = float(loss.item())
        return loss, logs
