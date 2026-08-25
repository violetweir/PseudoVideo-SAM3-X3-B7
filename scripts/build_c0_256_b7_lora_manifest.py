#!/usr/bin/env python3
"""Build a GT-free B7-selected pseudo-label manifest for SAM3 LoRA.

The selector matches the final C0-256-base X3+B7 rule.  For every target it
collects b3-b6 candidates from both route families, computes candidate-pool
agreement and agreement with the frozen X3-best student, and retains the
highest-scoring route mask.  ``--min-b7`` can then filter low-confidence
targets without consulting target ground truth.
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
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--student-predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(MODES),
        help="Route-mode directories to include (default: original target+patch modes).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--all-candidates-output",
        type=Path,
        help="Optional JSONL audit containing every scored route candidate.",
    )
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    parser.add_argument("--min-b7", type=float, default=0.0)
    parser.add_argument("--canvas", type=int, default=512)
    parser.add_argument(
        "--sample-type",
        default="c0_256_base_x3_best_b7",
        help=(
            "Manifest sample_type. Keep the historical default for LoRA data; "
            "use 'original' when the selected hard masks train an S27 student."
        ),
    )
    args = parser.parse_args()

    quality: dict[str, list[dict]] = defaultdict(list)
    for mode in args.modes:
        path = (
            args.quality_root
            / mode
            / f"propagation_quality_{args.split}"
            / "propagation_quality.jsonl"
        )
        for original in read_jsonl(path):
            bridge = int(original["bridge_count"])
            if args.min_bridge <= bridge <= args.max_bridge:
                row = dict(original)
                row["route_mode"] = mode
                quality[row["target_id"]].append(row)

    predictions = {
        row["merged_id"]: row for row in read_jsonl(args.student_predictions)
    }
    shape = (args.canvas, args.canvas)
    selected_all: list[dict] = []
    all_candidates: list[dict] = []
    oracle_by_target: dict[str, float] = {}
    missing_predictions: list[str] = []

    for target_id, rows in sorted(quality.items()):
        prediction = predictions.get(target_id)
        if prediction is None:
            missing_predictions.append(target_id)
            continue
        student = load_binary(prediction["student_binary_mask"], shape)
        masks = [load_binary(row["forward_mask_path"], shape) for row in rows]
        available_gt = [
            float(row["gt_dice_evaluation_only"])
            for row in rows
            if "gt_dice_evaluation_only" in row
        ]
        if available_gt:
            oracle_by_target[target_id] = max(available_gt)

        for index, row in enumerate(rows):
            peers = [other for j, other in enumerate(masks) if j != index]
            q_multi = float(np.mean([dice(masks[index], other) for other in peers]))
            q_return = float(row.get("q_cycle", 0.0))
            q_model = dice(masks[index], student)
            b7 = (
                max(q_return, 1e-6)
                * max(q_multi, 1e-6) ** 2
                * max(q_model, 1e-6) ** 2
            ) ** 0.2
            row["q_return"] = q_return
            row["q_multi"] = q_multi
            row["q_model"] = q_model
            row["b7"] = b7
            if args.all_candidates_output is not None:
                all_candidates.append(dict(row))

        chosen = max(
            rows,
            key=lambda row: (
                row["b7"],
                row["q_multi"],
                row["q_return"],
                row["route_id"],
            ),
        )
        output_row = {
            "target_id": target_id,
            "target_image_path": chosen["target_image_path"],
            "pseudo_mask_path": chosen["forward_mask_path"],
            "sample_type": args.sample_type,
            "split": args.split,
            "route_id": chosen["route_id"],
            "route_mode": chosen["route_mode"],
            "bridge_count": int(chosen["bridge_count"]),
            "q_return": chosen["q_return"],
            "q_multi": chosen["q_multi"],
            "q_model": chosen["q_model"],
            "b7": chosen["b7"],
            # S27 Student-X4 consumes hard pseudo masks through its
            # ``original`` stream and expects an explicit per-image weight.
            # LoRA dataset preparation safely ignores this extra audit field.
            "explicit_quality_weight": chosen["b7"],
            "student_checkpoint": prediction.get("checkpoint"),
        }
        # Evaluation-only fields are useful for validation calibration and
        # auditing, but never participate in selection or filtering.
        if "gt_dice_evaluation_only" in chosen:
            output_row["gt_dice_evaluation_only"] = float(
                chosen["gt_dice_evaluation_only"]
            )
        selected_all.append(output_row)

    selected = [row for row in selected_all if row["b7"] >= args.min_b7]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    if args.all_candidates_output is not None:
        args.all_candidates_output.parent.mkdir(parents=True, exist_ok=True)
        args.all_candidates_output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_candidates),
            encoding="utf-8",
        )

    b7_values = np.asarray([row["b7"] for row in selected_all], dtype=np.float64)
    selected_gt = [row["gt_dice_evaluation_only"] for row in selected if "gt_dice_evaluation_only" in row]
    selected_oracle = [
        oracle_by_target[row["target_id"]]
        for row in selected
        if row["target_id"] in oracle_by_target
    ]
    selected_gt_mean = float(np.mean(selected_gt)) if selected_gt else None
    oracle_mean = float(np.mean(selected_oracle)) if selected_oracle else None
    summary = {
        "split": args.split,
        "modes": args.modes,
        "min_bridge": args.min_bridge,
        "max_bridge": args.max_bridge,
        "min_b7": args.min_b7,
        "candidate_targets": len(selected_all),
        "selected_targets": len(selected),
        "coverage": len(selected) / len(selected_all) if selected_all else 0.0,
        "missing_predictions": missing_predictions,
        "b7_mean": float(b7_values.mean()) if len(b7_values) else None,
        "b7_quantiles": {
            str(q): float(np.quantile(b7_values, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        } if len(b7_values) else {},
        "selected_gt_dice_evaluation_only": selected_gt_mean,
        "oracle_gt_dice_evaluation_only": oracle_mean,
        "oracle_gap_evaluation_only": (
            oracle_mean - selected_gt_mean
            if oracle_mean is not None and selected_gt_mean is not None
            else None
        ),
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
