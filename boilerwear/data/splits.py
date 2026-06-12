from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


NUM_STAGES = 190


@dataclass(frozen=True)
class SampleRecord:
    image_path: str
    folder_id: int
    wear_pct: float
    split: str
    protocol: str


def folder_to_wear_pct(folder_id: int) -> float:
    return folder_id / NUM_STAGES * 100.0


def load_split_records(
    split_csv: Path | str,
    split: str | None = None,
) -> list[SampleRecord]:
    path = Path(split_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")

    records: list[SampleRecord] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = SampleRecord(
                image_path=row["image_path"],
                folder_id=int(row["folder_id"]),
                wear_pct=float(row["wear_pct"]),
                split=row["split"],
                protocol=row.get("protocol", ""),
            )
            if split is None or record.split == split:
                records.append(record)
    if split is not None and not records:
        raise ValueError(f"No samples for split='{split}' in {path}")
    return records
