from __future__ import annotations

from pathlib import Path


DATASETS = ("CVC-ClinicDB", "kvasir-seg")
SPLITS = ("train", "validation", "test")


def safe_id(merged_id: str) -> str:
    return merged_id.replace("::", "__").replace("/", "_")


def normalize_data_record(row: dict, data_root: Path) -> dict:
    """Return the canonical fields used by the pseudo-video scripts."""
    image_path = Path(row.get("file_name") or row.get("image_path"))
    mask_path = Path(row.get("mask_file_name") or row.get("mask_path"))
    if not image_path.is_absolute():
        image_path = data_root / image_path
    if not mask_path.is_absolute():
        mask_path = data_root / mask_path
    merged_id = row["merged_id"]
    dataset = row.get("source_dataset") or merged_id.split("::", 1)[0]
    sample_id = row.get("sample_id") or merged_id.split("::", 1)[1]
    return {
        **row,
        "merged_id": merged_id,
        "sample_id": sample_id,
        "source_dataset": dataset,
        "file_name": str(image_path.resolve()),
        "mask_file_name": str(mask_path.resolve()),
        "image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()),
    }
