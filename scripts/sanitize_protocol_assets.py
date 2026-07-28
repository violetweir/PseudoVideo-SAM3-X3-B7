#!/usr/bin/env python3
"""Create path-free public protocol assets from historical T21 manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pvseg.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-records", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.all_records)
    support = read_jsonl(args.support_manifest)
    split_rows = [
        {
            "merged_id": row["merged_id"],
            "sample_id": row.get("sample_id", row["merged_id"].split("::", 1)[1]),
            "source_dataset": row["source_dataset"],
            "split": row["split"],
            "height": row.get("height"),
            "width": row.get("width"),
        }
        for row in records
    ]
    support_ids = [
        {
            "merged_id": row["merged_id"],
            "sample_id": row.get("sample_id", row["merged_id"].split("::", 1)[1]),
            "source_dataset": row["source_dataset"],
            "split": row["split"],
        }
        for row in support
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "splits.jsonl", split_rows)
    write_jsonl(args.output_dir / "support_ids.jsonl", support_ids)
    (args.output_dir / "support_ids.txt").write_text(
        "".join(row["merged_id"] + "\n" for row in support_ids),
        encoding="utf-8",
    )
    print(
        {
            "split_rows": len(split_rows),
            "support_ids": len(support_ids),
            "output_dir": str(args.output_dir),
        }
    )


if __name__ == "__main__":
    main()
