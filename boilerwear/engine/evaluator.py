from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from boilerwear.losses import LossBuilder
from boilerwear.utils.metrics import compute_metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_builder: LossBuilder | None = None,
    return_predictions: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    all_true, all_pred, all_folder_true, all_folder_pred = [], [], [], []
    all_paths: list[str] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        strips = batch["strips"].to(device)
        outputs = model(strips)
        wear_pred = outputs["wear_pct"]
        all_paths.extend(batch.get("image_path", []))

        if loss_builder is not None:
            loss, _ = loss_builder(outputs, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
            total_loss += float(loss.item())
            n_batches += 1

        all_true.append(batch["wear_pct"].numpy())
        all_pred.append(wear_pred.cpu().numpy())
        all_folder_true.append(batch["folder_id"].numpy())
        folder_pred = outputs.get("folder_id_pred")
        if folder_pred is not None:
            all_folder_pred.append(folder_pred.cpu().numpy())
        else:
            all_folder_pred.append(
                (wear_pred.cpu().numpy() / 100.0 * 190).round().clip(1, 190)
            )

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    f_true = np.concatenate(all_folder_true)
    f_pred = np.concatenate(all_folder_pred)

    metrics = compute_metrics(y_true, y_pred, f_true, f_pred)
    if n_batches > 0:
        metrics["loss"] = round(total_loss / n_batches, 4)
    if return_predictions:
        predictions = {
            "image_path": all_paths,
            "folder_id": f_true.astype(int).tolist(),
            "wear_pct_true": y_true.tolist(),
            "wear_pct_pred": y_pred.tolist(),
        }
        return metrics, predictions
    return metrics


def evaluate_hog_lr(model, records, data_root, return_predictions: bool = False):
    images, y_true, f_true, paths = [], [], [], []
    for rec in records:
        img_path = data_root / rec.image_path
        images.append(np.array(Image.open(img_path).convert("RGB")))
        y_true.append(rec.wear_pct)
        f_true.append(rec.folder_id)
        paths.append(rec.image_path)

    y_pred = model.predict(images)
    y_true = np.array(y_true)
    f_true = np.array(f_true)
    f_pred = (y_pred / 100.0 * 190).round().clip(1, 190)
    metrics = compute_metrics(y_true, y_pred, f_true, f_pred)
    if return_predictions:
        predictions = {
            "image_path": paths,
            "folder_id": f_true.astype(int).tolist(),
            "wear_pct_true": y_true.tolist(),
            "wear_pct_pred": np.asarray(y_pred).tolist(),
        }
        return metrics, predictions
    return metrics
