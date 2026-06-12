from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from boilerwear.data.splits import NUM_STAGES, SampleRecord, folder_to_wear_pct


def discover_folders(data_root: Path) -> list[int]:
    folders = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name.isdigit():
            folders.append(int(p.name))
    return sorted(folders)


def discover_images(data_root: Path) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    for folder_id in discover_folders(data_root):
        folder = data_root / str(folder_id)
        for img in sorted(folder.glob("*.bmp")):
            rel = img.relative_to(data_root).as_posix()
            samples.append((rel, folder_id))
    return samples


def build_protocol1(folders: list[int], seed: int = 42) -> dict[str, Any]:
    rng = random.Random(seed)
    shuffled = folders[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train : n_train + n_val])
    test = sorted(shuffled[n_train + n_val :])
    return {
        "protocol": "protocol1",
        "name": "Group Split 70/15/15 by folder",
        "seed": seed,
        "split_unit": "folder",
        "train_folders": train,
        "val_folders": val,
        "test_folders": test,
    }


def build_protocol2(folders: list[int], seed: int = 42, k: int = 0) -> dict[str, Any]:
    """P2 level hold-out per workflow v3.2 §3.2.

    test = folder %% 10 == k; val = folder %% 10 == (k+5) %% 10 (deterministic,
    structurally identical to test: both are *unseen stages*). The previous
    random-15%% val violated the protocol — hyperparameters selected on a
    random-folder val carry no guidance for an unseen-stage test.
    """
    k = k % 10
    val_k = (k + 5) % 10
    test = sorted([f for f in folders if f % 10 == k])
    val = sorted([f for f in folders if f % 10 == val_k])
    train = sorted([f for f in folders if f % 10 not in (k, val_k)])
    return {
        "protocol": "protocol2",
        "name": f"Level Hold-out (test: folder%10=={k}, val: folder%10=={val_k})",
        "seed": seed,
        "p2_offset": k,
        "split_unit": "folder",
        "test_rule": f"folder_id % 10 == {k}",
        "val_rule": f"folder_id % 10 == {val_k}",
        "train_folders": train,
        "val_folders": val,
        "test_folders": test,
    }


def build_protocol3(folders: list[int]) -> dict[str, Any]:
    """P3 extrapolation split per workflow v3.2 §3.2:
    train 1–152 with 140–152 held out as val (so effective train = 1–139),
    test 172–190; folders 153–171 form an excluded buffer zone."""
    return {
        "protocol": "protocol3",
        "name": "Range Split (extrapolation to high wear)",
        "seed": None,
        "split_unit": "folder",
        "train_range": "1-139",
        "val_range": "140-152",
        "buffer_range_excluded": "153-171",
        "test_range": "172-190",
        "train_folders": [f for f in folders if 1 <= f <= 139],
        "val_folders": [f for f in folders if 140 <= f <= 152],
        "test_folders": [f for f in folders if 172 <= f <= 190],
    }


def attach_counts(data_root: Path, split_data: dict[str, Any]) -> dict[str, Any]:
    split_data = dict(split_data)
    counts: dict[str, Any] = {}
    for split_name, key in [("train", "train_folders"), ("val", "val_folders"), ("test", "test_folders")]:
        folder_list = split_data[key]
        images = 0
        wear_vals = []
        for fid in folder_list:
            folder = data_root / str(fid)
            imgs = list(folder.glob("*.bmp"))
            images += len(imgs)
            wear_vals.append(folder_to_wear_pct(fid))
        counts[split_name] = {
            "folders": len(folder_list),
            "images": images,
            "wear_pct_min": min(wear_vals) if wear_vals else 0.0,
            "wear_pct_max": max(wear_vals) if wear_vals else 0.0,
        }
    split_data["counts"] = counts
    return split_data


def check_folder_leakage(split_data: dict[str, Any]) -> dict[str, Any]:
    train = set(split_data["train_folders"])
    val = set(split_data["val_folders"])
    test = set(split_data["test_folders"])
    overlap_tv = train & val
    overlap_tt = train & test
    overlap_vt = val & test
    return {
        "leak_free": not (overlap_tv or overlap_tt or overlap_vt),
        "train_val_overlap": sorted(overlap_tv),
        "train_test_overlap": sorted(overlap_tt),
        "val_test_overlap": sorted(overlap_vt),
    }


def split_data_to_records(
    data_root: Path,
    split_data: dict[str, Any],
    protocol: str,
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    mapping = {
        "train": split_data["train_folders"],
        "val": split_data["val_folders"],
        "test": split_data["test_folders"],
    }
    folder_to_split: dict[int, str] = {}
    for split_name, folder_ids in mapping.items():
        for fid in folder_ids:
            folder_to_split[fid] = split_name

    for rel, folder_id in discover_images(data_root):
        if folder_id not in folder_to_split:
            continue
        records.append(
            SampleRecord(
                image_path=rel,
                folder_id=folder_id,
                wear_pct=folder_to_wear_pct(folder_id),
                split=folder_to_split[folder_id],
                protocol=protocol,
            )
        )
    return records


def write_split_csv(records: list[SampleRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "folder_id", "wear_pct", "split", "protocol"],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "image_path": r.image_path,
                    "folder_id": r.folder_id,
                    "wear_pct": r.wear_pct,
                    "split": r.split,
                    "protocol": r.protocol,
                }
            )


def generate_all_splits(
    data_root: Path,
    out_dir: Path,
    seed: int = 42,
    p2_offset: int = 0,
    smoke_fraction: float | None = None,
) -> dict[str, Any]:
    folders = discover_folders(data_root)
    if smoke_fraction is None and len(folders) != NUM_STAGES:
        raise ValueError(f"Expected {NUM_STAGES} folders, found {len(folders)}")

    suffix = ""
    if smoke_fraction is not None:
        if not (0.0 < smoke_fraction <= 1.0):
            raise ValueError(f"smoke_fraction must be in (0, 1], got {smoke_fraction}")
        rng = random.Random(seed)
        # Stratified by folder %% 10 so every residue class survives — keeps the
        # P2 deterministic test/val rules (folder%%10==k / ==k+5) non-empty.
        kept: list[int] = []
        for r in range(10):
            cls = [f for f in folders if f % 10 == r]
            n_keep = max(1, int(round(len(cls) * smoke_fraction)))
            kept.extend(rng.sample(cls, min(n_keep, len(cls))))
        folders = sorted(kept)
        suffix = "_smoke"

    p2 = build_protocol2(folders, seed=seed, k=p2_offset)
    p2_names = ["protocol2", f"protocol2_k{p2_offset % 10}"] if p2_offset % 10 == 0 else [f"protocol2_k{p2_offset % 10}"]

    builders = {
        "protocol1": build_protocol1(folders, seed=seed),
        "protocol3": build_protocol3(folders),
    }
    for nm in p2_names:
        builders[nm] = p2

    manifest: dict[str, Any] = {"protocols": {}, "leak_check": {}}
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    for name, split_data in builders.items():
        split_data = attach_counts(data_root, split_data)
        leak = check_folder_leakage(split_data)
        manifest["protocols"][name] = split_data
        manifest["leak_check"][name] = leak

        json_path = out_dir / f"{name}{suffix}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)

        records = split_data_to_records(data_root, split_data, name)
        write_split_csv(records, out_dir / f"{name}{suffix}.csv")

        for split_name in ("train", "val", "test"):
            c = split_data["counts"][split_name]
            stats_rows.append(
                {
                    "protocol": name,
                    "split": split_name,
                    "folders": c["folders"],
                    "images": c["images"],
                    "wear_pct_min": c["wear_pct_min"],
                    "wear_pct_max": c["wear_pct_max"],
                    "leak_free": leak["leak_free"],
                }
            )

    stats_path = reports_dir / f"split_stats_table{suffix}.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    leak_path = reports_dir / f"leak_check{suffix}.json"
    with leak_path.open("w", encoding="utf-8") as f:
        json.dump(manifest["leak_check"], f, indent=2)

    return manifest
