#!/usr/bin/env python3
"""Summarize coverage and evaluation-only GT Dice across B7 thresholds."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {}
    for name, path_text in args.input:
        rows = read_jsonl(Path(path_text))
        table = []
        for threshold in args.thresholds:
            kept = [row for row in rows if float(row["b7"]) >= threshold]
            gt = [float(row["gt_dice_evaluation_only"]) for row in kept]
            table.append(
                {
                    "threshold": threshold,
                    "selected": len(kept),
                    "coverage": len(kept) / len(rows) if rows else 0.0,
                    "selected_gt_dice_evaluation_only": float(np.mean(gt)) if gt else None,
                    "bridge_counts": dict(sorted(Counter(int(row["bridge_count"]) for row in kept).items())),
                    "mode_counts": dict(sorted(Counter(row["route_mode"] for row in kept).items())),
                }
            )
        result[name] = table

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, table in result.items():
        print(name)
        print("threshold\tselected\tcoverage\tgt_dice_evaluation_only")
        for row in table:
            print(
                f'{row["threshold"]:.3f}\t{row["selected"]}\t'
                f'{row["coverage"]:.3f}\t{row["selected_gt_dice_evaluation_only"]}'
            )


if __name__ == "__main__":
    main()
