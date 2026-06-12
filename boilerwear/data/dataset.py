from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from boilerwear.data.splits import SampleRecord, folder_to_wear_pct
from boilerwear.data.transforms import (
    PhotometricAugment,
    make_ldl_distribution,
    normalize_strips,
    split_into_strips,
)


class BoilerWearDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        data_root: Path | str,
        num_strips: int = 6,
        strip_size: int = 256,
        photometric_aug: bool = False,
        aug_cfg: dict | None = None,
        label_mode: str = "hard",
        ldl_sigma_folders: float = 1.0,
        normalize_mean: list[float] | None = None,
        normalize_std: list[float] | None = None,
        auto_resize: bool = True,
    ) -> None:
        self.records = records
        self.data_root = Path(data_root)
        self.num_strips = num_strips
        self.strip_size = strip_size
        self.label_mode = label_mode
        self.ldl_sigma_folders = ldl_sigma_folders
        self.auto_resize = auto_resize
        self.normalize_mean = normalize_mean or [0.485, 0.456, 0.406]
        self.normalize_std = normalize_std or [0.229, 0.224, 0.225]
        self.augment = None
        if photometric_aug:
            cfg = aug_cfg or {}
            self.augment = PhotometricAugment(**cfg)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img_path = self.data_root / rec.image_path if not Path(rec.image_path).is_absolute() else Path(rec.image_path)
        image = Image.open(img_path).convert("RGB")
        tensor = TF.to_tensor(image)
        expected_hw = (self.strip_size, self.num_strips * self.strip_size)
        if tensor.shape[-2:] != expected_hw:
            if not self.auto_resize:
                raise ValueError(
                    f"Image {img_path} has size {tuple(tensor.shape[-2:])}, expected {expected_hw}. "
                    f"Set data.auto_resize=true to resize original-resolution panoramas on the fly."
                )
            tensor = TF.resize(tensor, list(expected_hw), antialias=True)
        strips = split_into_strips(tensor, self.num_strips, self.strip_size)
        if self.augment is not None:
            strips = self.augment(strips)
        strips = normalize_strips(strips, self.normalize_mean, self.normalize_std)

        sample = {
            "strips": strips,
            "wear_pct": torch.tensor(rec.wear_pct, dtype=torch.float32),
            "folder_id": torch.tensor(rec.folder_id, dtype=torch.long),
            "image_path": str(rec.image_path),
        }
        if self.label_mode == "ldl":
            sample["wear_dist"] = make_ldl_distribution(rec.folder_id, sigma_folders=self.ldl_sigma_folders)
        return sample


def collate_batch(batch: list[dict]) -> dict:
    out: dict = {
        "strips": torch.stack([b["strips"] for b in batch], dim=0),
        "wear_pct": torch.stack([b["wear_pct"] for b in batch], dim=0),
        "folder_id": torch.stack([b["folder_id"] for b in batch], dim=0),
        "image_path": [b["image_path"] for b in batch],
    }
    if "wear_dist" in batch[0]:
        out["wear_dist"] = torch.stack([b["wear_dist"] for b in batch], dim=0)
    return out
