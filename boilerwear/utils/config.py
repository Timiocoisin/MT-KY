from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: Path | str, root: Path | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve()
    root = root or path.parent
    cfg = load_yaml(path)
    bases = cfg.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = (root / base).resolve() if not Path(base).is_absolute() else Path(base)
        merged = merge_dict(merged, load_config(base_path, root=base_path.parent))
    merged = merge_dict(merged, cfg)
    return merged


def save_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
