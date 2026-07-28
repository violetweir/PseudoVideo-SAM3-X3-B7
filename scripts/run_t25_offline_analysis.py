#!/usr/bin/env python3
"""T25 frozen-route/student agreement analysis and predefined selectors."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mask(path: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if shape is not None and image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    # Exact T21 binary convention.
    return np.asarray(image) > 127


def probability(path: str, shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(path)
    if image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return array / (65535.0 if array.max() > 255 else 255.0)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """T25 Q_model Dice: two empty masks agree perfectly."""
    sa, sb = int(a.sum()), int(b.sum())
    if sa == 0 and sb == 0:
        return 1.0
    if sa == 0 or sb == 0:
        return 0.0
    return float(2 * np.logical_and(a, b).sum() / (sa + sb))


def t21_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Exact T21 Dice: two empty route masks have agreement zero."""
    denominator = int(a.sum()) + int(b.sum())
    return float(2 * np.logical_and(a, b).sum() / max(denominator, 1))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def corr(x: list[float], y: list[float]) -> dict:
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "mae": float(np.mean(np.abs(np.asarray(x) - np.asarray(y)))),
        "mse": float(np.mean((np.asarray(x) - np.asarray(y)) ** 2)),
    }


SELECTORS = {
    "B0_old": lambda r: 0.5 * r["q_return"] + 0.5 * r["q_multi"],
    "B1_model": lambda r: r["q_model"],
    "B2_multi_model": lambda r: 0.5 * r["q_multi"] + 0.5 * r["q_model"],
    "B3_weak_return": lambda r: 0.2 * r["q_return"]
    + 0.4 * r["q_multi"]
    + 0.4 * r["q_model"],
    "B4_weaker_return": lambda r: 0.1 * r["q_return"]
    + 0.45 * r["q_multi"]
    + 0.45 * r["q_model"],
    "B5_student_priority": lambda r: 0.1 * r["q_return"]
    + 0.3 * r["q_multi"]
    + 0.6 * r["q_model"],
    "B6_product_gate": lambda r: r["q_model"]
    * (0.5 * r["q_return"] + 0.5 * r["q_multi"]),
    "B7_geometric": lambda r: (
        max(r["q_return"], 1e-6)
        * max(r["q_multi"], 1e-6) ** 2
        * max(r["q_model"], 1e-6) ** 2
    )
    ** 0.2,
}
ORDER = {"direct": 0, "one_bridge": 1, "two_bridges": 2}


def build_rows(route_path: Path, prediction_path: Path) -> list[dict]:
    predictions = {r["merged_id"]: r for r in read_jsonl(prediction_path)}
    groups: dict[str, list[dict]] = defaultdict(list)
    for route in read_jsonl(route_path):
        groups[route["target_id"]].append(route)
    if any(len(v) != 3 for v in groups.values()):
        raise RuntimeError("Every target must have exactly three routes")
    if set(groups) - set(predictions):
        raise RuntimeError("Student prediction coverage is incomplete")

    output = []
    for target_id, routes in sorted(groups.items()):
        routes.sort(key=lambda r: ORDER[r["route_type"]])
        pred = predictions[target_id]
        # T21 route selection and evaluation were performed on the native
        # 512x512 SAM3 masks. Preserve those masks exactly and resize only the
        # 256x256 student output to the frozen T21 canvas.
        sam_masks = [mask(r["forward_mask_path"]) for r in routes]
        canvas_shape = sam_masks[0].shape
        if any(item.shape != canvas_shape for item in sam_masks):
            raise RuntimeError(f"Inconsistent SAM mask shapes for {target_id}")
        student = mask(pred["student_binary_mask"], canvas_shape)
        student_prob = probability(pred["student_probability_map"], canvas_shape)
        gt = mask(routes[0]["target_mask_path_evaluation_only"], canvas_shape)
        pairwise = [[t21_dice(a, b) for b in sam_masks] for a in sam_masks]
        for index, (route, sam) in enumerate(zip(routes, sam_masks)):
            intersection = int(np.logical_and(sam, student).sum())
            sam_area, student_area = int(sam.sum()), int(student.sum())
            q_model = dice(sam, student)
            q_multi = float(
                sum(pairwise[index][j] for j in range(3) if j != index) / 2
            )
            eps = 1e-7
            q_area = math.exp(
                -abs(math.log((sam_area + eps) / (student_area + eps)))
            )
            # T21 stores binary route masks, not probability maps. This is explicitly
            # a binary-SAM/soft-student proxy, not the unavailable true soft Dice.
            q_model_soft_proxy = float(
                2 * np.sum(sam.astype(np.float32) * student_prob)
                / (sam_area + np.sum(student_prob) + eps)
            )
            output.append(
                {
                    **route,
                    "q_multi": q_multi,
                    "old_quality": 0.5 * route["q_return"] + 0.5 * q_multi,
                    "q_model": q_model,
                    "q_model_soft_proxy": q_model_soft_proxy,
                    "q_fg_intersection": intersection / (sam_area + eps),
                    "q_student_coverage": intersection / (student_area + eps),
                    "q_area": q_area,
                    "gt_dice_evaluation_only": t21_dice(sam, gt),
                    "gt_iou_evaluation_only": iou(sam, gt),
                    "student_gt_dice_evaluation_only": t21_dice(student, gt),
                    "both_empty": sam_area == 0 and student_area == 0,
                    "sam_empty": sam_area == 0,
                    "student_empty": student_area == 0,
                    "sam_area_ratio": float(sam.mean()),
                    "student_area_ratio": float(student.mean()),
                    "student_binary_mask": pred["student_binary_mask"],
                    "student_probability_map": pred["student_probability_map"],
                }
            )
    return output


def evaluate(rows: list[dict], score_name: str, score_fn) -> tuple[dict, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["target_id"]].append(row)
    selected, changes = [], []
    for target_id, candidates in sorted(groups.items()):
        candidates.sort(key=lambda r: ORDER[r["route_type"]])
        # Match T21's deterministic tie break exactly.
        old = max(
            candidates,
            key=lambda r: (
                r["old_quality"],
                r["q_multi"],
                r["q_return"],
                r["route_id"],
            ),
        )
        chosen = max(
            candidates,
            key=lambda r: (
                score_fn(r),
                r["q_multi"],
                r["q_return"],
                r["route_id"],
            ),
        )
        oracle_value = max(r["gt_dice_evaluation_only"] for r in candidates)
        oracle_routes = [
            r for r in candidates if abs(r["gt_dice_evaluation_only"] - oracle_value) < 1e-12
        ]
        selected.append(chosen)
        changes.append(
            {
                "target_id": target_id,
                "old_selected_route": old["route_type"],
                "new_selected_route": chosen["route_type"],
                "oracle_routes": [r["route_type"] for r in oracle_routes],
                "old_selected_dice": old["gt_dice_evaluation_only"],
                "new_selected_dice": chosen["gt_dice_evaluation_only"],
                "oracle_dice": oracle_value,
                "student_dice": chosen["student_gt_dice_evaluation_only"],
                "delta": chosen["gt_dice_evaluation_only"]
                - old["gt_dice_evaluation_only"],
            }
        )
    values = np.asarray([r["gt_dice_evaluation_only"] for r in selected])
    ious = np.asarray([r["gt_iou_evaluation_only"] for r in selected])
    oracle = np.asarray([c["oracle_dice"] for c in changes])
    deltas = np.asarray([c["delta"] for c in changes])
    by_dataset = {}
    for dataset in sorted({r["target_source_dataset"] for r in selected}):
        subset = [r for r in selected if r["target_source_dataset"] == dataset]
        by_dataset[dataset] = {
            "n": len(subset),
            "dice": float(np.mean([r["gt_dice_evaluation_only"] for r in subset])),
            "iou": float(np.mean([r["gt_iou_evaluation_only"] for r in subset])),
        }
    summary = {
        "selector": score_name,
        "n": len(selected),
        "dice": float(values.mean()),
        "iou": float(ious.mean()),
        "oracle_dice": float(oracle.mean()),
        "oracle_gap": float((oracle - values).mean()),
        "route_counts": dict(Counter(r["route_type"] for r in selected)),
        "route_accuracy": float(
            np.mean(
                [
                    c["new_selected_route"] in c["oracle_routes"]
                    for c in changes
                ]
            )
        ),
        "nonempty_rate": float(np.mean([not r["sam_empty"] for r in selected])),
        "mean_area": float(np.mean([r["sam_area_ratio"] for r in selected])),
        "by_dataset": by_dataset,
        "changes": {
            "improved": int((deltas > 1e-12).sum()),
            "unchanged": int((np.abs(deltas) <= 1e-12).sum()),
            "degraded": int((deltas < -1e-12).sum()),
            "mean": float(deltas.mean()),
            "median": float(np.median(deltas)),
            "drop_over_0_1": int((deltas < -0.1).sum()),
            "min": float(deltas.min()),
            "max": float(deltas.max()),
        },
    }
    return summary, changes


def analyze(rows: list[dict]) -> dict:
    scores = {
        "q_return": [r["q_return"] for r in rows],
        "q_multi": [r["q_multi"] for r in rows],
        "q_model": [r["q_model"] for r in rows],
        "q_model_soft_proxy": [r["q_model_soft_proxy"] for r in rows],
        "q_area": [r["q_area"] for r in rows],
        "old_quality": [r["old_quality"] for r in rows],
    }
    gt = [r["gt_dice_evaluation_only"] for r in rows]
    correlations = {"all_routes": {name: corr(v, gt) for name, v in scores.items()}}
    for route_type in ORDER:
        subset = [r for r in rows if r["route_type"] == route_type]
        correlations[route_type] = {
            name: corr([r[name] for r in subset], [r["gt_dice_evaluation_only"] for r in subset])
            for name in scores
        }
    for dataset in sorted({r["target_source_dataset"] for r in rows}):
        subset = [r for r in rows if r["target_source_dataset"] == dataset]
        correlations[dataset] = {
            name: corr([r[name] for r in subset], [r["gt_dice_evaluation_only"] for r in subset])
            for name in scores
        }
    bad = np.asarray(gt) < 0.5
    bad_detection = {}
    for name, values in scores.items():
        error_score = 1 - np.asarray(values)
        bad_detection[name] = {
            "auroc": float(roc_auc_score(bad, error_score)),
            "auprc": float(average_precision_score(bad, error_score)),
            "bad_mean_score": float(np.mean(np.asarray(values)[bad])),
            "normal_mean_score": float(np.mean(np.asarray(values)[~bad])),
        }
    self_consistent = {}
    for threshold in (0.9, 0.95):
        subset = [
            r
            for r in rows
            if r["q_return"] >= threshold
            and r["q_multi"] >= threshold
            and r["gt_dice_evaluation_only"] < 0.5
        ]
        self_consistent[str(threshold)] = {
            "count": len(subset),
            "rate": len(subset) / len(rows),
            "mean_gt_dice": float(np.mean([r["gt_dice_evaluation_only"] for r in subset]))
            if subset
            else None,
            "mean_q_model": float(np.mean([r["q_model"] for r in subset]))
            if subset
            else None,
            "route_counts": dict(Counter(r["route_type"] for r in subset)),
            "dataset_counts": dict(
                Counter(r["target_source_dataset"] for r in subset)
            ),
        }
    return {
        "correlations": correlations,
        "bad_route_detection": bad_detection,
        "self_consistent_errors": self_consistent,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.routes, args.predictions)
    with (args.output_dir / f"q_model_routes_{args.split}.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    results, changes = {}, {}
    for name, fn in SELECTORS.items():
        results[name], changes[name] = evaluate(rows, name, fn)
    payload = {
        "student": args.student,
        "split": args.split,
        "route_count": len(rows),
        "target_count": len(rows) // 3,
        "soft_metric_note": "q_model_soft_proxy uses binary SAM masks because T21 did not save SAM probability maps",
        "analysis": analyze(rows),
        "selectors": results,
    }
    (args.output_dir / f"summary_{args.split}.json").write_text(
        json.dumps(payload, indent=2)
    )
    with (args.output_dir / f"per_query_{args.split}.jsonl").open("w") as handle:
        for selector, items in changes.items():
            for item in items:
                handle.write(json.dumps({"selector": selector, **item}) + "\n")
    print(json.dumps({"student": args.student, "split": args.split, "B0": results["B0_old"]["dice"]}))


if __name__ == "__main__":
    main()
