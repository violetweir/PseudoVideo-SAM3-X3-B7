#!/usr/bin/env python3
"""Freeze the merged CVC/Kvasir protocol and a 1%-of-total train-only support set."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

DATASETS = {
    "CVC-ClinicDB": "CVC-ClinicDB",
    "kvasir-seg": "kvasir-seg",
}
SPLITS = ("train", "validation", "test")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SUPPORT_PER_SOURCE = 8
SELECTION_SALT = "t17-autonomous-support-v1-20260724"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T02_fresh_polyp_hf_sources/raw_hf_snapshots"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T17_autonomous_target_1pct"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def find_mask(mask_root: Path, image_path: Path) -> Path:
    direct = mask_root / image_path.name
    if direct.is_file():
        return direct
    for extension in IMAGE_EXTENSIONS:
        candidate = mask_root / f"{image_path.stem}{extension}"
        if candidate.is_file():
            return candidate
    matches = sorted(path for path in mask_root.glob(f"{image_path.stem}*") if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one mask for {image_path}, found {len(matches)}"
        )
    return matches[0]


def build_manifest(input_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_dataset, directory_name in DATASETS.items():
        snapshot = input_root / directory_name / "snapshot"
        for split in SPLITS:
            image_root = snapshot / split / "images"
            mask_root = snapshot / split / "masks"
            images = sorted(
                path
                for path in image_root.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for image_path in images:
                mask_path = find_mask(mask_root, image_path)
                with Image.open(image_path) as image:
                    width, height = image.size
                rows.append(
                    {
                        "merged_dataset": "CVC-ClinicDB_plus_kvasir-seg",
                        "source_dataset": source_dataset,
                        "split": split,
                        "sample_id": image_path.stem,
                        "merged_id": f"{source_dataset}::{image_path.stem}",
                        "image_path": str(image_path.resolve()),
                        "mask_path": str(mask_path.resolve()),
                        "width": width,
                        "height": height,
                    }
                )
    return rows


def choose_support(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for source_dataset in DATASETS:
        candidates = [
            row
            for row in rows
            if row["source_dataset"] == source_dataset and row["split"] == "train"
        ]
        ranked = sorted(
            candidates,
            key=lambda row: hashlib.sha256(
                f"{SELECTION_SALT}::{row['merged_id']}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ranked[:SUPPORT_PER_SOURCE])
    return sorted(selected, key=lambda row: str(row["merged_id"]))


def write_once_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise RuntimeError(f"Frozen file already exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    protocol_root = output_root / "protocol"
    support_root = output_root / "support_frozen"

    rows = build_manifest(input_root)
    expected = {
        ("CVC-ClinicDB", "train"): 490,
        ("CVC-ClinicDB", "validation"): 61,
        ("CVC-ClinicDB", "test"): 61,
        ("kvasir-seg", "train"): 800,
        ("kvasir-seg", "validation"): 100,
        ("kvasir-seg", "test"): 100,
    }
    counts = {
        key: sum(
            row["source_dataset"] == key[0] and row["split"] == key[1] for row in rows
        )
        for key in expected
    }
    if counts != expected:
        raise RuntimeError(f"Dataset counts differ from frozen expectation: {counts}")

    support = choose_support(rows)
    if len(rows) != 1612 or len(support) != 16:
        raise RuntimeError(f"Unexpected total/support sizes: {len(rows)}/{len(support)}")
    if any(row["split"] != "train" for row in support):
        raise RuntimeError("Support set contains a non-train sample")

    manifest_payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )
    support_rows: list[dict[str, object]] = []
    for row in support:
        image_path = Path(str(row["image_path"]))
        mask_path = Path(str(row["mask_path"]))
        safe_name = f"{row['source_dataset']}__{row['sample_id']}"
        image_copy = support_root / "images" / f"{safe_name}{image_path.suffix.lower()}"
        mask_copy = support_root / "masks" / f"{safe_name}{mask_path.suffix.lower()}"
        image_copy.parent.mkdir(parents=True, exist_ok=True)
        mask_copy.parent.mkdir(parents=True, exist_ok=True)
        if not image_copy.exists():
            shutil.copy2(image_path, image_copy)
        if not mask_copy.exists():
            shutil.copy2(mask_path, mask_copy)
        support_rows.append(
            {
                **row,
                "frozen_image_path": str(image_copy),
                "frozen_mask_path": str(mask_copy),
                "image_sha256": sha256_file(image_copy),
                "mask_sha256": sha256_file(mask_copy),
                "selection_key_sha256": hashlib.sha256(
                    f"{SELECTION_SALT}::{row['merged_id']}".encode("utf-8")
                ).hexdigest(),
            }
        )

    support_payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in support_rows
    )
    write_once_or_verify(protocol_root / "merged_manifest.jsonl", manifest_payload)
    write_once_or_verify(protocol_root / "support_manifest.jsonl", support_payload)

    protocol = {
        "protocol_name": "T17 autonomous target discovery with frozen 1pct support",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "merged_dataset": "CVC-ClinicDB_plus_kvasir-seg",
        "split_policy": "merge matching source splits; never move samples across splits",
        "counts": {
            f"{dataset}/{split}": count for (dataset, split), count in counts.items()
        },
        "merged_split_counts": {
            split: sum(row["split"] == split for row in rows) for split in SPLITS
        },
        "total_images": len(rows),
        "support_images": len(support_rows),
        "support_fraction_of_total": len(support_rows) / len(rows),
        "support_selection": {
            "eligible_split": "train only",
            "strategy": "deterministic SHA256 ranking, stratified 8 per source dataset",
            "salt": SELECTION_SALT,
            "replacement": "forbidden after manifest freeze",
        },
        "merged_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "support_manifest_sha256": hashlib.sha256(support_payload).hexdigest(),
        "validation_or_test_support_count": 0,
    }
    protocol_path = protocol_root / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        for key in (
            "merged_manifest_sha256",
            "support_manifest_sha256",
            "total_images",
            "support_images",
        ):
            if old[key] != protocol[key]:
                raise RuntimeError(f"Frozen protocol mismatch for {key}")
    else:
        protocol_path.write_bytes(stable_json_bytes(protocol))

    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    print("support_ids:")
    for row in support_rows:
        print(row["merged_id"])


if __name__ == "__main__":
    main()
