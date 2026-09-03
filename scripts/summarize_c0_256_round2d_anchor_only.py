#!/usr/bin/env python3
"""Summarize the frozen Round-2D anchor-only full-route validation study."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
# The frozen validation baseline is the original e33 full evaluation.  The
# similarly named Round-2A directory contains train-side quality only.
BASELINE_QUALITY_ROOT = (
    ROOT
    / "work/rerun_c0_256_sam3knn_s256_base"
    / "medsam3_lora_b0_b6_e50/e33_full_evaluation/quality_root"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "dice": float(np.mean([float(row["gt_dice_evaluation_only"]) for row in rows])),
        "q_return": float(np.mean([float(row["q_cycle"]) for row in rows])),
    }


def paired(current: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    current_by_target = {row["target_id"]: row for row in current}
    baseline_by_target = {row["target_id"]: row for row in baseline}
    target_ids = sorted(set(current_by_target) & set(baseline_by_target))
    dice_delta = np.asarray([
        float(current_by_target[target_id]["gt_dice_evaluation_only"])
        - float(baseline_by_target[target_id]["gt_dice_evaluation_only"])
        for target_id in target_ids
    ])
    cycle_delta = np.asarray([
        float(current_by_target[target_id]["q_cycle"])
        - float(baseline_by_target[target_id]["q_cycle"])
        for target_id in target_ids
    ])
    changed = np.asarray([
        current_by_target[target_id]["anchor_id"] != baseline_by_target[target_id]["anchor_id"]
        for target_id in target_ids
    ])
    def subset(values: np.ndarray, membership: np.ndarray) -> float | None:
        return float(values[membership].mean()) if membership.any() else None
    return {
        "targets": len(target_ids),
        "dice_delta": float(dice_delta.mean()),
        "q_return_delta": float(cycle_delta.mean()),
        "anchor_changed": int(changed.sum()),
        "anchor_changed_rate": float(changed.mean()),
        "dice_delta_changed_anchor": subset(dice_delta, changed),
        "dice_delta_unchanged_anchor": subset(dice_delta, ~changed),
        "q_return_delta_changed_anchor": subset(cycle_delta, changed),
        "q_return_delta_unchanged_anchor": subset(cycle_delta, ~changed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top1-quality-root", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "split": "validation",
        "feature_size": 256,
        "propagation_canvas": 256,
        "teacher": "sam3_e33",
        "anchor_rule": "forward_auc_token_topk__round1",
        "topology": "sam3_base@256 frozen",
        "modes": {},
        "combined": {},
        "test_used": False,
        "train_propagation": False,
        "new_training": False,
    }
    combined_current: dict[int, list[dict[str, Any]]] = defaultdict(list)
    combined_base: dict[int, list[dict[str, Any]]] = defaultdict(list)
    combined_top1: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for mode in args.modes:
        new_path = args.phase / "quality_root" / mode / "propagation_quality_validation/propagation_quality.jsonl"
        route_path = args.phase / "quality_root" / mode / "validation_pool0_stage1/routes.jsonl"
        base_mode = mode.removeprefix("round2d_forward_token_top3_")
        base_path = BASELINE_QUALITY_ROOT / base_mode / "propagation_quality_validation/propagation_quality.jsonl"
        new_rows = read_jsonl(new_path)
        route_rank = {
            row["route_id"]: int(row["round2d_anchor_rank"])
            for row in read_jsonl(route_path)
        }
        base_rows = read_jsonl(base_path)
        # Reused Round-2A / Round-2C quality rows predate Round-2D and do
        # not carry its route metadata.  The frozen routes file is the
        # provenance source for anchor rank in both reused and newly run rows.
        top1 = [row for row in new_rows if route_rank[row["route_id"]] == 1]
        write_jsonl(
            args.top1_quality_root / mode / "propagation_quality_validation/propagation_quality.jsonl",
            top1,
        )
        result["modes"][mode] = {}
        for bridge in range(7):
            current_b = [row for row in new_rows if int(row["bridge_count"]) == bridge]
            top1_b = [row for row in top1 if int(row["bridge_count"]) == bridge]
            base_b = [row for row in base_rows if int(row["bridge_count"]) == bridge]
            by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in current_b:
                by_target[row["target_id"]].append(row)
            oracle = [max(rows, key=lambda row: float(row["gt_dice_evaluation_only"])) for rows in by_target.values()]
            result["modes"][mode][f"b{bridge}"] = {
                "baseline": metrics(base_b),
                "forward_token_top1": metrics(top1_b),
                "forward_token_top3_candidate_oracle": metrics(oracle),
                "paired_top1_vs_baseline": paired(top1_b, base_b),
            }
            combined_current[bridge].extend(current_b)
            combined_base[bridge].extend(base_b)
            combined_top1[bridge].extend(top1_b)

    for bridge in range(7):
        by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in combined_current[bridge]:
            by_target[row["target_id"]].append(row)
        oracle = [max(rows, key=lambda row: float(row["gt_dice_evaluation_only"])) for rows in by_target.values()]
        result["combined"][f"b{bridge}"] = {
            "baseline": metrics(combined_base[bridge]),
            "forward_token_top1": metrics(combined_top1[bridge]),
            "forward_token_top3_candidate_oracle": metrics(oracle),
            "paired_top1_vs_baseline": paired(combined_top1[bridge], combined_base[bridge]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "combined": result["combined"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
