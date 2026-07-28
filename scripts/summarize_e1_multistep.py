#!/usr/bin/env python3
"""Build paired A/B/C and depth tables for E1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root.resolve()
    data = {
        structure: read_jsonl(root / f"{structure}_per_target.jsonl")
        for structure in ("star", "chain", "hybrid")
    }
    indexed = {
        structure: {
            (row["target_merged_id"], int(row["depth"])): row for row in rows
        }
        for structure, rows in data.items()
    }
    complete_targets = set.intersection(
        *[
            {
                target_id
                for target_id in {
                    row["target_merged_id"] for row in rows if row["matched_k5"]
                }
                if all(
                    (target_id, depth) in indexed[structure]
                    and indexed[structure][(target_id, depth)]["matched_k5"]
                    for depth in range(1, 6)
                )
            }
            for structure, rows in data.items()
        ]
    )
    table = []
    for depth in range(1, 6):
        keys = {(target_id, depth) for target_id in complete_targets}
        row: dict[str, Any] = {"depth": depth, "n_matched_k5": len(keys)}
        for structure in ("star", "chain", "hybrid"):
            values = np.asarray(
                [indexed[structure][key]["dice"] for key in sorted(keys)],
                dtype=np.float64,
            )
            row[f"{structure}_dice"] = float(values.mean()) if len(values) else None
            row[f"{structure}_success_nonempty"] = (
                float(
                    np.mean(
                        [
                            indexed[structure][key]["candidate_count"] > 0
                            for key in sorted(keys)
                        ]
                    )
                )
                if keys
                else None
            )
        for structure in ("chain", "hybrid"):
            deltas = np.asarray(
                [
                    indexed[structure][key]["dice"]
                    - indexed["star"][key]["dice"]
                    for key in sorted(keys)
                ],
                dtype=np.float64,
            )
            row[f"{structure}_minus_star"] = (
                float(deltas.mean()) if len(deltas) else None
            )
            row[f"{structure}_win_rate_vs_star"] = (
                float(np.mean(deltas > 0)) if len(deltas) else None
            )
        table.append(row)
    output = {
        "cohort": (
            "global intersection of matched_k5 target IDs present in every "
            "K=1..5 and every A/B/C structure"
        ),
        "n_global_matched_targets": len(complete_targets),
        "propagation_success_definition": "SAM3 returned at least one mask",
        "table": table,
    }
    (root / "paired_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
