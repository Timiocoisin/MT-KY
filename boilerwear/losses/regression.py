from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target)


def wear_pct_to_level(wear_pct: torch.Tensor, num_bins: int = 19) -> torch.Tensor:
    """Map wear% to a 1-based ordinal level in {1..num_bins}.

    Bin k covers ((k-1)/num_bins, k/num_bins] * 100, so with 190 folders and
    19 bins, folders 1-10 -> level 1, 11-20 -> level 2, ..., 181-190 -> level 19.
    """
    boundaries = torch.arange(1, num_bins + 1, device=wear_pct.device, dtype=wear_pct.dtype) / num_bins * 100.0
    # bucketize returns the 0-based bucket index for (b_{k-1}, b_k]; +1 -> 1-based level
    return (torch.bucketize(wear_pct.clamp(0, 100), boundaries) + 1).clamp(1, num_bins)


def coral_loss(logits: torch.Tensor, wear_pct: torch.Tensor, num_bins: int = 19) -> torch.Tensor:
    """CORAL loss from wear % targets.

    For true level r (1-based), the K-1 binary targets are 1[r > k] for k=1..K-1,
    i.e. exactly (r - 1) ones.
    """
    true_levels = wear_pct_to_level(wear_pct, num_bins)
    thresholds = torch.arange(1, num_bins, device=logits.device).unsqueeze(0)
    targets = (true_levels.unsqueeze(1) > thresholds).float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def corn_loss(logits: torch.Tensor, wear_pct: torch.Tensor, num_bins: int = 19) -> torch.Tensor:
    """CORN loss (Shi, Cao & Raschka, 2022) — conditional ordinal training.

    Task k (k=1..K-1) is trained only on samples with true level > k-1,
    with binary target 1[level > k]; this guarantees rank consistency
    without the shared-bias constraint of CORAL.
    """
    true_levels = wear_pct_to_level(wear_pct, num_bins)
    total = logits.new_zeros(())
    n_examples = 0
    for k in range(1, num_bins):  # threshold r_k
        mask = true_levels > (k - 1)  # conditional subset: y > r_{k-1}
        if mask.sum() == 0:
            continue
        target_k = (true_levels[mask] > k).float()
        total = total + F.binary_cross_entropy_with_logits(
            logits[mask, k - 1], target_k, reduction="sum"
        )
        n_examples += int(mask.sum().item())
    if n_examples == 0:
        return logits.new_zeros(())
    return total / n_examples


def ldl_kl_loss(pred_dist: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    pred = pred_dist.clamp(min=1e-8)
    target = target_dist.clamp(min=1e-8)
    return (target * (target.log() - pred.log())).sum(dim=-1).mean()


def heteroscedastic_loss(pred: torch.Tensor, target: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    precision = torch.exp(-log_var)
    return (0.5 * (precision * (pred - target) ** 2 + log_var)).mean()


def monotonic_pairwise_loss(wear_pred: torch.Tensor, folder_id: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Vectorized pairwise monotonicity hinge.

    For every ordered pair (i, j) with folder_id[i] < folder_id[j], penalize
    relu(pred[i] - pred[j] + margin). Equivalent to the previous O(n^2)
    Python loop but runs as a single batched op.
    """
    n = wear_pred.size(0)
    if n < 2:
        return wear_pred.new_zeros(())
    f = folder_id.view(-1)
    lower = f.unsqueeze(1) < f.unsqueeze(0)  # [i, j] True iff folder_i < folder_j
    if not bool(lower.any()):
        return wear_pred.new_zeros(())
    violation = F.relu(wear_pred.unsqueeze(1) - wear_pred.unsqueeze(0) + margin)
    count = lower.sum()
    return (violation * lower).sum() / count


def rnc_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Rank-N-Contrast loss (Zha et al., NeurIPS 2023).

    Faithful to the official implementation (kaiwenzha/Rank-N-Contrast):
    label difference = L1 |y_i - y_j|; feature similarity = negative L2
    distance / temperature. For each anchor i and positive j, the negative
    set is every k whose label distance from the anchor is >= that of j:

        L = -mean_{i,j} log( exp(s_ij) / sum_{k: d_ik >= d_ij} exp(s_ik) )

    Ranking quality depends on batch diversity, so keep the effective batch
    large (the unified batch=128 of workflow v3.2 is fine).
    """
    n = features.size(0)
    if n < 2:
        return features.new_zeros(())
    labels = labels.view(-1, 1).float()
    label_diffs = (labels - labels.T).abs()  # [n, n]
    logits = -torch.cdist(features, features, p=2) / temperature  # [n, n]
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()  # stability

    off_diag = ~torch.eye(n, dtype=torch.bool, device=features.device)
    exp_logits = torch.exp(logits) * off_diag

    loss = features.new_zeros(())
    count = 0
    for j in range(n):  # each column j as the positive for every anchor i (i != j)
        anchor_mask = off_diag[:, j]  # exclude i == j
        # negatives for (i, j): k with d_ik >= d_ij (and k != i)
        neg_mask = (label_diffs >= label_diffs[:, j].view(-1, 1)) & off_diag
        denom = (exp_logits * neg_mask).sum(dim=1).clamp_min(1e-12)
        log_prob = logits[:, j] - denom.log()
        loss = loss - log_prob[anchor_mask].sum()
        count += int(anchor_mask.sum().item())
    if count == 0:
        return features.new_zeros(())
    return loss / count
