"""Public ordinal benchmark support (workflow v3.2/v3.3 §10).

Validates the transferable ordinal components — HOD two-level head, mono loss,
LDL — on a public benchmark (UTKFace ★ / AFAD) with a shared ResNet50
backbone, against four heads: (a) reg, (b) CORAL, (c) LDL, (d) HOD(+mono).

Labels are internally normalized onto the same 0-100 "wear-like" percent
scale used by the main task so that all loss functions and metrics are
reused verbatim; reported MAE is converted back to label units (years).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from boilerwear.data.transforms import make_ldl_distribution
from boilerwear.models.baselines.strip_regressor import coral_logits_to_wear

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------- #
# Dataset parsing & splits
# --------------------------------------------------------------------------- #

def parse_utkface(data_root: Path) -> list[tuple[str, int]]:
    """UTKFace filenames: ``{age}_{gender}_{race}_{datetime}.jpg`` (age 0-116).

    Download: https://susanqq.github.io/UTKFace/ (Aligned&Cropped) or the
    Kaggle mirror; unpack all images under ``data_root``.
    """
    samples: list[tuple[str, int]] = []
    for p in sorted(data_root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            age = int(p.name.split("_")[0])
        except (ValueError, IndexError):
            continue
        if 0 <= age <= 116:
            samples.append((p.relative_to(data_root).as_posix(), age))
    return samples


def parse_afad(data_root: Path) -> list[tuple[str, int]]:
    """AFAD layout: ``{age}/{111|112}/{image}.jpg`` (age 15-40 for AFAD-Full).

    Download: https://github.com/afad-dataset/tarball
    """
    samples: list[tuple[str, int]] = []
    for age_dir in sorted(data_root.iterdir()):
        if not (age_dir.is_dir() and age_dir.name.isdigit()):
            continue
        age = int(age_dir.name)
        for p in sorted(age_dir.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                samples.append((p.relative_to(data_root).as_posix(), age))
    return samples


DATASET_SPECS: dict[str, dict[str, Any]] = {
    # min/max label define the stage range; num_bins = HOD/CORAL coarse bins
    "utkface": {"parser": parse_utkface, "min_label": 0, "max_label": 116, "num_bins": 12},
    "afad": {"parser": parse_afad, "min_label": 15, "max_label": 75, "num_bins": 12},
}


def build_benchmark_split(
    samples: list[tuple[str, int]],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> list[dict[str, Any]]:
    """Random 80/10/10 image-level split (the customary protocol for UTKFace;
    record the seed in the paper for reproducibility)."""
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    rows = []
    for rank, i in enumerate(idx):
        split = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
        path, label = samples[i]
        rows.append({"image_path": path, "label": label, "split": split})
    return sorted(rows, key=lambda r: r["image_path"])


def write_benchmark_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)


def load_benchmark_records(csv_path: Path, split: str) -> list[dict[str, Any]]:
    records = []
    with Path(csv_path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                records.append({"image_path": row["image_path"], "label": int(row["label"])})
    if not records:
        raise ValueError(f"No samples for split='{split}' in {csv_path}")
    return records


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class BenchmarkDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        data_root: Path | str,
        min_label: int,
        max_label: int,
        img_size: int = 224,
        augment: bool = False,
        ldl_sigma_stages: float = 2.0,
        need_ldl: bool = False,
    ) -> None:
        self.records = records
        self.data_root = Path(data_root)
        self.min_label = min_label
        self.num_stages = max_label - min_label + 1
        self.img_size = img_size
        self.augment = augment
        self.ldl_sigma_stages = ldl_sigma_stages
        self.need_ldl = need_ldl

    def label_to_pct(self, label: int) -> float:
        stage = label - self.min_label + 1
        return stage / self.num_stages * 100.0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = Image.open(self.data_root / rec["image_path"]).convert("RGB")
        tensor = TF.to_tensor(image)
        if tensor.shape[-2:] != (self.img_size, self.img_size):
            tensor = TF.resize(tensor, [self.img_size, self.img_size], antialias=True)
        if self.augment and random.random() < 0.5:
            tensor = TF.hflip(tensor)
        tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        stage = rec["label"] - self.min_label + 1
        sample = {
            "image": tensor,
            "wear_pct": torch.tensor(self.label_to_pct(rec["label"]), dtype=torch.float32),
            "folder_id": torch.tensor(stage, dtype=torch.long),  # stage doubles as ordinal id
            "image_path": rec["image_path"],
        }
        if self.need_ldl:
            sample["wear_dist"] = make_ldl_distribution(
                stage, num_stages=self.num_stages, sigma_folders=self.ldl_sigma_stages
            )
        return sample


def collate_benchmark(batch: list[dict]) -> dict:
    out = {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "wear_pct": torch.stack([b["wear_pct"] for b in batch], dim=0),
        "folder_id": torch.stack([b["folder_id"] for b in batch], dim=0),
        "image_path": [b["image_path"] for b in batch],
    }
    if "wear_dist" in batch[0]:
        out["wear_dist"] = torch.stack([b["wear_dist"] for b in batch], dim=0)
    return out


# --------------------------------------------------------------------------- #
# Model: shared backbone + one of four heads
# --------------------------------------------------------------------------- #

class BenchmarkNet(nn.Module):
    """ResNet50 (or any timm backbone) + head in {reg, coral, ldl, hod}.

    HOD reuses exactly the same two-level structure as the main model
    (coarse CORAL bins + fine per-stage distribution + reg primary output,
    reg-only inference) so §10 tests the *same* components.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        head: str = "reg",
        num_stages: int = 117,
        num_bins: int = 12,
        pretrained: bool = True,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        import timm

        self.head_type = head
        self.num_stages = num_stages
        self.num_bins = num_bins
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.proj = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True))

        wear_levels = torch.arange(1, num_stages + 1, dtype=torch.float32) / num_stages * 100.0
        self.register_buffer("wear_levels", wear_levels)

        if head in ("reg", "hod"):
            self.reg_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
            )
        if head in ("coral", "hod"):
            self.coral = nn.Linear(hidden_dim, num_bins - 1)
        if head in ("ldl", "hod"):
            self.fine = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_stages)
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.proj(self.backbone(x))
        out: dict[str, torch.Tensor] = {}

        if self.head_type == "reg":
            out["wear_pct_reg"] = self.reg_head(feats).squeeze(-1)
            out["wear_pct"] = out["wear_pct_reg"].clamp(0.0, 100.0)
        elif self.head_type == "coral":
            logits = self.coral(feats)
            out["coral_logits"] = logits
            out["wear_pct"] = coral_logits_to_wear(logits, self.num_bins)
        elif self.head_type == "ldl":
            logits = self.fine(feats)
            dist = torch.softmax(logits, dim=-1)
            out["wear_dist_pred"] = dist
            out["wear_pct"] = (dist * self.wear_levels).sum(dim=-1)
        elif self.head_type == "hod":
            out["wear_pct_reg"] = self.reg_head(feats).squeeze(-1)
            logits = self.coral(feats)
            out["coral_logits"] = logits
            fine = torch.softmax(self.fine(feats), dim=-1)
            out["wear_dist_pred"] = fine
            # reg-only inference, HOD as deep supervision — same as the main model
            out["wear_pct"] = out["wear_pct_reg"].clamp(0.0, 100.0)
        else:
            raise ValueError(f"Unknown head: {self.head_type}")

        folder_pred = (out["wear_pct"] / 100.0 * self.num_stages).round().clamp(1, self.num_stages)
        out["folder_id_pred"] = folder_pred
        return out


def benchmark_loss(
    head: str,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    num_bins: int,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    from boilerwear.losses.regression import (
        coral_loss,
        ldl_kl_loss,
        monotonic_pairwise_loss,
        smooth_l1_loss,
    )

    w = {"reg": 0.5, "ord": 0.25, "ldl": 0.25, "mono": 0.1, **(weights or {})}
    wear = batch["wear_pct"]
    logs: dict[str, float] = {}

    if head == "reg":
        loss = smooth_l1_loss(outputs["wear_pct_reg"], wear)
        logs["loss_reg"] = float(loss.item())
        return loss, logs
    if head == "coral":
        loss = coral_loss(outputs["coral_logits"], wear, num_bins)
        logs["loss_coral"] = float(loss.item())
        return loss, logs
    if head == "ldl":
        loss = ldl_kl_loss(outputs["wear_dist_pred"], batch["wear_dist"])
        logs["loss_ldl"] = float(loss.item())
        return loss, logs
    if head == "hod":
        reg = smooth_l1_loss(outputs["wear_pct_reg"], wear)
        ordl = coral_loss(outputs["coral_logits"], wear, num_bins)
        ldl = ldl_kl_loss(outputs["wear_dist_pred"], batch["wear_dist"])
        mono = monotonic_pairwise_loss(outputs["wear_pct_reg"], batch["folder_id"])
        loss = w["reg"] * reg + w["ord"] * ordl + w["ldl"] * ldl + w["mono"] * mono
        logs.update(
            loss_reg=float(reg.item()), loss_ord=float(ordl.item()),
            loss_ldl=float(ldl.item()), loss_mono=float(mono.item()),
            loss_total=float(loss.item()),
        )
        return loss, logs
    raise ValueError(f"Unknown head: {head}")
