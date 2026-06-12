"""Regression / ordinal metrics for wear % prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr


def _as_numpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def wear_pct_to_bin(wear_pct: np.ndarray, num_bins: int = 19) -> np.ndarray:
    wear_pct = np.clip(wear_pct, 0.0, 100.0)
    bin_width = 100.0 / num_bins
    bins = np.floor(wear_pct / bin_width).astype(np.int64)
    return np.clip(bins, 0, num_bins - 1)


def folder_id_to_stage(folder_id: np.ndarray) -> np.ndarray:
    return _as_numpy(folder_id).astype(np.int64)


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_bins: int = 19,
) -> float:
    yt = wear_pct_to_bin(_as_numpy(y_true), num_bins)
    yp = wear_pct_to_bin(_as_numpy(y_pred), num_bins)
    conf = np.zeros((num_bins, num_bins), dtype=np.float64)
    for t, p in zip(yt, yp, strict=True):
        conf[t, p] += 1.0
    if conf.sum() < 1e-12:
        return 0.0
    hist_true = conf.sum(axis=1)
    hist_pred = conf.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / conf.sum()
    weights = np.zeros((num_bins, num_bins), dtype=np.float64)
    for i in range(num_bins):
        for j in range(num_bins):
            weights[i, j] = ((i - j) ** 2) / max((num_bins - 1) ** 2, 1.0)
    observed = (weights * conf).sum()
    expected_score = (weights * expected).sum()
    if expected_score < 1e-12:
        return 0.0
    return float(1.0 - observed / expected_score)


def compute_metrics(
    y_true: Any,
    y_pred: Any,
    folder_true: Any | None = None,
    folder_pred: Any | None = None,
    num_bins: int = 19,
) -> dict[str, float]:
    yt = _as_numpy(y_true)
    yp = _as_numpy(y_pred)
    err = np.abs(yt - yp)

    mae = float(np.mean(err))
    medae = float(np.median(err))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    acc5 = float(np.mean(err <= 5.0))
    acc10 = float(np.mean(err <= 10.0))

    if len(yt) > 1 and np.std(yt) > 1e-12 and np.std(yp) > 1e-12:
        spearman = float(spearmanr(yt, yp).statistic)
    else:
        spearman = 0.0

    qwk = quadratic_weighted_kappa(yt, yp, num_bins=num_bins)

    metrics = {
        "mae": round(mae, 4),
        "medae": round(medae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "acc_at_5": round(acc5, 4),
        "acc_at_10": round(acc10, 4),
        "spearman": round(spearman, 4),
        "qwk": round(qwk, 4),
        "n_samples": int(len(yt)),
    }

    if folder_true is not None and folder_pred is not None:
        ft = folder_id_to_stage(folder_true)
        fp = folder_id_to_stage(folder_pred)
        stage_err = np.abs(ft - fp)
        metrics["acc_at_stage_1"] = round(float(np.mean(stage_err <= 1)), 4)
        metrics["acc_at_stage_3"] = round(float(np.mean(stage_err <= 3)), 4)
        metrics["acc_at_stage_5"] = round(float(np.mean(stage_err <= 5)), 4)

    return metrics
