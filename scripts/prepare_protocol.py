#!/usr/bin/env python3
"""Build local manifests from the fixed public split/support protocol."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pvseg.io import read_jsonl, write_jsonl
from pvseg.protocol import normalize_data_record


EXPECTED_COUNTS = {
    "train": 1290,
    "validation": 161,
    "test": 161,
}


def read_metadata(data_root: Path) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for split in EXPECTED_COUNTS:
        path = data_root / split / "metadata.jsonl"
        rows = read_jsonl(path)
        if len(rows) != EXPECTED_COUNTS[split]:
            raise RuntimeError(f"{path} expected {EXPECTED_COUNTS[split]} rows, got {len(rows)}")
        for row in rows:
            canonical = normalize_data_record({**row, "split": split}, data_root)
            if canonical["merged_id"] in output:
                raise RuntimeError(f"Duplicate merged_id: {canonical['merged_id']}")
            output[canonical["merged_id"]] = canonical
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--support-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata = read_metadata(args.data_root)
    split_protocol = read_jsonl(args.splits)
    support_ids = [row["merged_id"] for row in read_jsonl(args.support_ids)]
    missing = sorted({row["merged_id"] for row in split_protocol} - set(metadata))
    if missing:
        raise RuntimeError(f"Protocol IDs absent from data metadata: {missing[:10]}")
    if len(support_ids) != 16 or len(set(support_ids)) != 16:
        raise RuntimeError("Support protocol must contain exactly 16 unique IDs")

    manifest_rows = []
    for row in sorted(split_protocol, key=lambda item: (item["split"], item["source_dataset"], item["merged_id"])):
        local = metadata[row["merged_id"]]
        if local["split"] != row["split"]:
            raise RuntimeError(f"Split mismatch for {row['merged_id']}: {local['split']} != {row['split']}")
        manifest_rows.append(
            {
                "height": row.get("height"),
                "width": row.get("width"),
                "image_path": local["image_path"],
                "mask_path": local["mask_path"],
                "file_name": local["file_name"],
                "mask_file_name": local["mask_file_name"],
                "merged_dataset": "CVC-ClinicDB_plus_kvasir-seg",
                "merged_id": local["merged_id"],
                "sample_id": row.get("sample_id", local["sample_id"]),
                "source_dataset": local["source_dataset"],
                "split": local["split"],
            }
        )

    by_id = {row["merged_id"]: row for row in manifest_rows}
    support_rows = []
    for merged_id in support_ids:
        row = by_id[merged_id]
        if row["split"] != "train":
            raise RuntimeError(f"Support anchor is not train: {merged_id}")
        support_rows.append(
            {
                **row,
                "frozen_image_path": row["image_path"],
                "frozen_mask_path": row["mask_path"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "merged_manifest.jsonl", manifest_rows)
    write_jsonl(args.output_dir / "support_manifest.jsonl", support_rows)
    labeled = "\n".join(row["image_path"] for row in support_rows) + "\n"
    (args.output_dir / "frozen_labeled_images.txt").write_text(labeled, encoding="utf-8")
    summary = {
        "data_root": str(args.data_root.resolve()),
        "counts": dict(Counter(row["split"] for row in manifest_rows)),
        "support_count": len(support_rows),
        "support_dataset_counts": dict(Counter(row["source_dataset"] for row in support_rows)),
        "target": "S27 X3+B7 reproduction protocol",
    }
    (args.output_dir / "protocol_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
