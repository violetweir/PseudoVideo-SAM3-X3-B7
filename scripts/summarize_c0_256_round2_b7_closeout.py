#!/usr/bin/env python3
"""Summarize e33 fixed-topology B7 with X3/X4 student selectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(summary: dict) -> dict:
    return {
        "selected_targets": summary["selected_targets"],
        "selected_dice": summary["selected_gt_dice_evaluation_only"],
        "oracle_dice": summary.get("oracle_gt_dice_evaluation_only"),
        "oracle_gap": summary.get("oracle_gap_evaluation_only"),
        "mean_b7_confidence_not_dice": summary["b7_mean"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparisons = {}
    for selector in ("X3_best", "X4_best", "X4_final"):
        comparisons[selector] = {
            split: compact(load(args.root / f"{selector}_{split}.summary.json"))
            for split in ("validation", "test")
        }
    result = {
        "propagation_checkpoint": "round-1 SAM3 e33",
        "knn_topology": "frozen SAM3-base@256 b0-b6",
        "round2_student_selector": "X4_best (X4 checkpoint chosen on student validation)",
        "best_b7_selector_by_validation": max(
            ("X3_best", "X4_best"),
            key=lambda selector: comparisons[selector]["validation"]["selected_dice"],
        ),
        "diagnostic_selector_not_eligible_for_selection": "X4_final",
        "selectors": comparisons,
        "X4_best_minus_X3_best": {
            split: (
                comparisons["X4_best"][split]["selected_dice"]
                - comparisons["X3_best"][split]["selected_dice"]
            )
            for split in ("validation", "test")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
