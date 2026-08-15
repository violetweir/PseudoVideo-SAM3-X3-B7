#!/usr/bin/env python3
"""Phase-1 Stage 4: B7-style route selection on the ft_1pct b3-b6 test pool.

For each test target, score every candidate with
    B7 = (q_return * q_multi^2 * q_model^2)^0.2
where q_return is the ft_1pct propagation-quality cycle consistency,
q_multi is the candidate-pool agreement, and q_model is the X3-student vs
route-mask Dice.  The top-scoring candidate mask is reported once on test.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


MODES = (
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_binary(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--student-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    args = parser.parse_args()

    quality: dict[str, list[dict]] = defaultdict(list)
    for mode in MODES:
        for row in read_jsonl(
            args.quality_root / mode / "propagation_quality_test/propagation_quality.jsonl"
        ):
            if args.min_bridge <= int(row["bridge_count"]) <= args.max_bridge:
                quality[row["target_id"]].append(row)

    predictions = {
        row["merged_id"]: row for row in read_jsonl(args.student_predictions)
    }

    selected = []
    oracle_values = []
    canvas_shape = (512, 512)
    for target, rows in sorted(quality.items()):
        if target not in predictions:
            raise RuntimeError(f"Missing student prediction for {target}")
        student = load_binary(predictions[target]["student_binary_mask"], canvas_shape)
        masks = [load_binary(row["forward_mask_path"], canvas_shape) for row in rows]
        for index, row in enumerate(rows):
            q_multi = float(
                np.mean(
                    [
                        dice(masks[index], other)
                        for j, other in enumerate(masks)
                        if j != index
                    ]
                )
            )
            row["q_return"] = float(row.get("q_cycle", 0.0))
            row["q_multi"] = q_multi
            row["q_model"] = dice(masks[index], student)
            row["b7"] = (
                max(row["q_return"], 1e-6)
                * max(row["q_multi"], 1e-6) ** 2
                * max(row["q_model"], 1e-6) ** 2
            ) ** 0.2
        chosen = max(
            rows,
            key=lambda row: (
                row["b7"],
                row["q_multi"],
                row["q_return"],
                row["route_id"],
            ),
        )
        best = max(rows, key=lambda row: float(row["gt_dice_evaluation_only"]))
        selected.append(float(chosen["gt_dice_evaluation_only"]))
        oracle_values.append(float(best["gt_dice_evaluation_only"]))

    report = {
        "n_targets": len(selected),
        "selected_dice": float(np.mean(selected)),
        "oracle_dice": float(np.mean(oracle_values)),
        "oracle_gap": float(np.mean(oracle_values) - np.mean(selected)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "b7_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
