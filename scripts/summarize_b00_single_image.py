#!/usr/bin/env python3
"""Summarize B00 single-image SAM3 results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows: list[dict]) -> dict:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[row["source_dataset"]].append(row)
    return {
        "n": len(rows),
        "dice": float(np.mean([row["dice"] for row in rows])),
        "iou": float(np.mean([row["iou"] for row in rows])),
        "nonempty_rate": float(np.mean([row["pred_area_ratio"] > 0 for row in rows])),
        "mean_area": float(np.mean([row["pred_area_ratio"] for row in rows])),
        "candidate_count_mean": float(np.mean([row["candidate_count"] for row in rows])),
        "dataset_counts": dict(Counter(row["source_dataset"] for row in rows)),
        "by_dataset": {
            name: {
                "n": len(items),
                "dice": float(np.mean([row["dice"] for row in items])),
                "iou": float(np.mean([row["iou"] for row in items])),
            }
            for name, items in sorted(by_dataset.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(read_jsonl(args.results))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
