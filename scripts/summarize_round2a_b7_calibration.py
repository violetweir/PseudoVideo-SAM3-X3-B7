#!/usr/bin/env python3
"""Build the exact validation B7 coverage/quality frontier for Round-2A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-dice", type=float, default=0.95)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: (-float(row["b7"]), row["target_id"]))
    frontier = []
    dice_sum = 0.0
    for index, row in enumerate(rows, start=1):
        dice_sum += float(row["gt_dice_evaluation_only"])
        next_b7 = float(rows[index]["b7"]) if index < len(rows) else None
        threshold = float(row["b7"])
        # Record only complete equal-score groups so --min-b7 reproduces count.
        if next_b7 is not None and next_b7 == threshold:
            continue
        frontier.append(
            {
                "min_b7": threshold,
                "selected_targets": index,
                "coverage": index / len(rows),
                "selected_gt_dice_evaluation_only": dice_sum / index,
            }
        )

    eligible = [
        row
        for row in frontier
        if row["selected_gt_dice_evaluation_only"] >= args.target_dice
    ]
    recommended = max(eligible, key=lambda row: row["selected_targets"]) if eligible else None
    result = {
        "candidate_targets": len(rows),
        "selection_rule": (
            "maximum validation coverage among B7-prefixes whose selected GT "
            f"Dice is at least {args.target_dice}"
        ),
        "target_dice": args.target_dice,
        "recommended": recommended,
        "frontier": frontier,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_tsv.write_text(
        "min_b7\tselected_targets\tcoverage\tselected_gt_dice_evaluation_only\n"
        + "".join(
            f'{row["min_b7"]:.12f}\t{row["selected_targets"]}\t'
            f'{row["coverage"]:.6f}\t'
            f'{row["selected_gt_dice_evaluation_only"]:.12f}\n'
            for row in frontier
        ),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "frontier"}, indent=2))


if __name__ == "__main__":
    main()
