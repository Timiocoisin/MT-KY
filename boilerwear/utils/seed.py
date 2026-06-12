from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_gpu_ids(spec: str) -> list[int]:
    """Parse comma-separated GPU ids, e.g. ``'0,1'`` -> ``[0, 1]``."""
    ids = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if not ids:
        raise ValueError(f"Empty GPU list: {spec!r}")
    return ids


def validate_gpu_ids(gpu_ids: list[int]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    n = torch.cuda.device_count()
    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= n:
            raise ValueError(f"Invalid gpu_id={gpu_id}; available GPUs: 0..{n - 1}")


def get_device(name: str = "cuda", gpu_id: int | None = None) -> torch.device:
    if name == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id is not None:
        validate_gpu_ids([gpu_id])
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cuda")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
