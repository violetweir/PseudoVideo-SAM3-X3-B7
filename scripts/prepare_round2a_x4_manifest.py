#!/usr/bin/env python3
"""Make an e33-B7 hard-mask manifest compatible with the frozen X3 recipe."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b7-manifest", type=Path, required=True)
    parser.add_argument("--x3-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    reference = {row["target_id"]: row for row in read_jsonl(args.x3_reference)}
    output = []
    membership = Counter()
    for row in read_jsonl(args.b7_manifest):
        previous = reference.get(row["target_id"])
        sample_type = previous["sample_type"] if previous else "original"
        membership["retained_from_x3" if previous else "new_to_x4"] += 1
        converted = dict(row)
        converted.update(
            {
                "sample_type": sample_type,
                "explicit_quality_weight": float(row["b7"]),
                "q_model_mean": float(row["q_model"]),
                "q_model_var": 0.0,
            }
        )
        # e33 B7 supplies a new binary hard mask. Do not retain old X3
        # consensus or pixel-weight paths that belong to the base teacher.
        converted.pop("pseudo_consensus_path", None)
        converted.pop("pixel_weight_path", None)
        output.append(converted)

    counts = Counter(row["sample_type"] for row in output)
    if not counts["original"] or not (counts["tier_a"] + counts["tier_b"]):
        raise RuntimeError(f"Cannot reproduce X3's three streams: {dict(counts)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    summary = {
        "targets": len(output),
        "sample_types": dict(counts),
        "membership": dict(membership),
        "batch_recipe": {"gt": 3, "original": 3, "tier_a_or_b": 6},
        "mask_source": "e33 propagation + validation-recalibrated X3/B7",
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
