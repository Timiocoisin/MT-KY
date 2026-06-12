from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def make_ldl_distribution(
    folder_id: int,
    num_stages: int = 190,
    sigma_folders: float = 1.0,
) -> torch.Tensor:
    stages = np.arange(1, num_stages + 1, dtype=np.float64)
    logits = -0.5 * ((stages - folder_id) / max(sigma_folders, 1e-6)) ** 2
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return torch.from_numpy(probs.astype(np.float32))


def ldl_distribution_to_wear(dist: torch.Tensor) -> torch.Tensor:
    """Expected wear %% under a stage distribution (was elementwise product before)."""
    num_stages = dist.shape[-1]
    wear = torch.arange(1, num_stages + 1, dtype=torch.float32, device=dist.device) / num_stages * 100.0
    return (dist * wear).sum(dim=-1)


class PhotometricAugment:
    def __init__(
        self,
        brightness: float = 0.15,
        contrast: float = 0.15,
        saturation: float = 0.10,
        noise_var_limit: tuple[float, float] = (5.0, 20.0),
        noise_p: float = 0.2,
        blur_p: float = 0.1,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.noise_var_limit = noise_var_limit
        self.noise_p = noise_p
        self.blur_p = blur_p

    def __call__(self, strips: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.8:
            b = 1.0 + random.uniform(-self.brightness, self.brightness)
            c = 1.0 + random.uniform(-self.contrast, self.contrast)
            s = 1.0 + random.uniform(-self.saturation, self.saturation)
            strips = strips * b
            mean = strips.mean(dim=(1, 2), keepdim=True)
            strips = (strips - mean) * c + mean
            gray = strips.mean(dim=0, keepdim=True)
            strips = (1 - s) * gray + s * strips
        if random.random() < self.noise_p:
            var = random.uniform(*self.noise_var_limit) / 255.0**2
            strips = strips + torch.randn_like(strips) * var**0.5
        if random.random() < self.blur_p:
            strips = torch.nn.functional.avg_pool2d(strips, kernel_size=3, stride=1, padding=1)
        return strips.clamp(0.0, 1.0)


def split_into_strips(
    image_tensor: torch.Tensor,
    num_strips: int = 6,
    strip_size: int = 256,
) -> torch.Tensor:
    """image_tensor: [3, H, W] -> [num_strips, 3, strip_size, strip_size]"""
    _, h, w = image_tensor.shape
    assert h == strip_size, f"Expected height {strip_size}, got {h}"
    assert w == num_strips * strip_size, f"Expected width {num_strips * strip_size}, got {w}"
    strips = []
    for i in range(num_strips):
        x0 = i * strip_size
        strips.append(image_tensor[:, :, x0 : x0 + strip_size])
    return torch.stack(strips, dim=0)


def normalize_strips(strips: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=strips.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=strips.dtype).view(1, 3, 1, 1)
    return (strips - mean_t) / std_t
