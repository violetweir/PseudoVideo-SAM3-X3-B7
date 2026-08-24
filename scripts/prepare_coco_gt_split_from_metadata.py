#!/usr/bin/env python3
"""Create one GT COCO split from frozen baseline metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_c0_256_b7_medsam3_dataset import read_jsonl, write_coco_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.metadata)
    records = [
        {
            "image_path": row["file_name"],
            "mask_path": row["mask_file_name"],
            "sample_type": "gt",
            "target_id": row["merged_id"],
        }
        for row in rows
    ]
    write_coco_split(args.output_dir, records)
    print(
        json.dumps(
            {
                "metadata": str(args.metadata.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "count": len(records),
                "target_ids_unique": len({record["target_id"] for record in records}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
