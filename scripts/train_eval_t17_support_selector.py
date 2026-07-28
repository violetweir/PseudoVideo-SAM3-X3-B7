#!/usr/bin/env python3
"""Train only on the frozen 16 support masks, then evaluate frozen T17 selectors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

METHODS = (
    "native_pooled",
    "native_qwen_text",
    "native_qwen_box",
    "native_qwen_text_box",
    "fixed_no_gt",
    "support_grid",
    "support_ridge",
    "oracle_audit",
)
BASE_FEATURE_NAMES = (
    "sam_rank",
    "mask_qbox_iou",
    "bbox_qbox_iou",
    "mask_inside_qbox",
    "center_proximity",
)
RIDGE_FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    "log_area_ratio",
    "source_qwen_text",
    "source_qwen_box",
    "source_qwen_text_box",
)
FIXED_WEIGHTS = np.asarray([0.50, 0.20, 0.20, 0.10, 0.00], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T17_autonomous_target_1pct/sam3_candidates"
        ),
    )
    parser.add_argument(
        "--support-manifest",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T17_autonomous_target_1pct/protocol/support_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T17_autonomous_target_1pct/selector_results"
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rank01(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / (len(values) - 1)


def load_sample(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        feature_names = [str(item) for item in archive["feature_names"]]
        raw = np.asarray(archive["features"], dtype=np.float64)
        lookup = {name: index for index, name in enumerate(feature_names)}
        source_id = np.asarray(archive["source_id"], dtype=np.int64)
        sam_score = np.asarray(archive["sam_score"], dtype=np.float64)
        base = np.column_stack(
            [
                rank01(sam_score),
                raw[:, lookup["mask_qbox_iou"]],
                raw[:, lookup["bbox_qbox_iou"]],
                raw[:, lookup["mask_inside_qbox"]],
                raw[:, lookup["center_proximity"]],
            ]
        )
        area = np.log1p(1000 * np.maximum(raw[:, lookup["area_ratio"]], 0))
        source_onehot = np.column_stack(
            [(source_id == index).astype(np.float64) for index in range(3)]
        )
        ridge = np.column_stack([base, area, source_onehot])
        return {
            "path": str(path),
            "merged_id": str(archive["merged_id"]),
            "source_dataset": str(archive["source_dataset"]),
            "split": str(archive["split"]),
            "sample_id": str(archive["sample_id"]),
            "qwen_text": str(archive["qwen_text"]),
            "qwen_box": np.asarray(archive["qwen_box"], dtype=np.float64).tolist(),
            "base_features": base,
            "ridge_features": ridge,
            "source_id": source_id,
            "sam_score": sam_score,
            "dice": np.asarray(archive["dice"], dtype=np.float64),
            "iou": np.asarray(archive["iou"], dtype=np.float64),
            "intersection": np.asarray(archive["intersection"], dtype=np.int64),
            "pred_area": np.asarray(archive["pred_area"], dtype=np.int64),
            "gt_area": np.asarray(archive["gt_area"], dtype=np.int64),
        }


def load_all(candidate_root: Path) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for split in ("train", "validation", "test"):
        paths = sorted((candidate_root / split).glob("*.npz"))
        result[split] = [load_sample(path) for path in paths]
    return result


def fit_ridge(
    samples: list[dict[str, Any]],
    alpha: float,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([sample["ridge_features"] for sample in samples], axis=0)
    y = np.concatenate([sample["dice"] for sample in samples], axis=0)
    if mean is None:
        mean = x.mean(axis=0)
    if scale is None:
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), standardized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ y
    )
    return coefficients, mean, scale


def ridge_score(
    sample: dict[str, Any],
    coefficients: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    standardized = (sample["ridge_features"] - mean) / scale
    return coefficients[0] + standardized @ coefficients[1:]


def select_ridge_alpha(
    samples: list[dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    candidates = (0.01, 0.1, 1.0, 10.0, 100.0)
    scores: dict[str, float] = {}
    for alpha in candidates:
        heldout_dice = []
        for index, heldout in enumerate(samples):
            training = samples[:index] + samples[index + 1 :]
            coefficients, mean, scale = fit_ridge(training, alpha)
            selected = int(
                np.argmax(ridge_score(heldout, coefficients, mean, scale))
            )
            heldout_dice.append(float(heldout["dice"][selected]))
        scores[str(alpha)] = float(np.mean(heldout_dice))
    best = max(candidates, key=lambda alpha: (scores[str(alpha)], -alpha))
    return best, scores


def grid_weights() -> list[np.ndarray]:
    weights = []
    for integer_weights in itertools.product(range(5), repeat=5):
        if sum(integer_weights) == 4:
            weights.append(np.asarray(integer_weights, dtype=np.float64) / 4)
    return weights


def select_grid_weights(
    samples: list[dict[str, Any]],
) -> tuple[np.ndarray, float]:
    best_weights = None
    best_score = -1.0
    for weights in grid_weights():
        selected_dice = [
            float(sample["dice"][np.argmax(sample["base_features"] @ weights)])
            for sample in samples
        ]
        score = float(np.mean(selected_dice))
        if score > best_score:
            best_score = score
            best_weights = weights
    assert best_weights is not None
    return best_weights, best_score


def select_index(
    sample: dict[str, Any],
    method: str,
    grid: np.ndarray,
    ridge_parameters: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> int | None:
    if len(sample["dice"]) == 0:
        return None
    if method == "native_pooled":
        return int(np.argmax(sample["sam_score"]))
    if method.startswith("native_qwen_"):
        source = {
            "native_qwen_text": 0,
            "native_qwen_box": 1,
            "native_qwen_text_box": 2,
        }[method]
        eligible = np.where(sample["source_id"] == source)[0]
        if len(eligible) == 0:
            return int(np.argmax(sample["sam_score"]))
        return int(eligible[np.argmax(sample["sam_score"][eligible])])
    if method == "fixed_no_gt":
        return int(np.argmax(sample["base_features"] @ FIXED_WEIGHTS))
    if method == "support_grid":
        return int(np.argmax(sample["base_features"] @ grid))
    if method == "support_ridge":
        coefficients, mean, scale = ridge_parameters
        return int(np.argmax(ridge_score(sample, coefficients, mean, scale)))
    if method == "oracle_audit":
        return int(np.argmax(sample["dice"]))
    raise KeyError(method)


def evaluate(
    samples: list[dict[str, Any]],
    group: str,
    support_ids: set[str],
    grid: np.ndarray,
    ridge_parameters: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    summaries = []
    for method in METHODS:
        method_rows = []
        for sample in samples:
            selected = select_index(sample, method, grid, ridge_parameters)
            if selected is None:
                dice = iou = 0.0
                intersection = pred_area = 0
                gt_area = int(sample["gt_area"][0]) if len(sample["gt_area"]) else 0
                source_id = -1
                score = float("nan")
            else:
                dice = float(sample["dice"][selected])
                iou = float(sample["iou"][selected])
                intersection = int(sample["intersection"][selected])
                pred_area = int(sample["pred_area"][selected])
                gt_area = int(sample["gt_area"][selected])
                source_id = int(sample["source_id"][selected])
                score = float(sample["sam_score"][selected])
            row = {
                "group": group,
                "method": method,
                "merged_id": sample["merged_id"],
                "source_dataset": sample["source_dataset"],
                "split": sample["split"],
                "sample_id": sample["sample_id"],
                "is_support": sample["merged_id"] in support_ids,
                "selected_index": selected if selected is not None else -1,
                "selected_source_id": source_id,
                "sam_score": score,
                "dice": dice,
                "iou": iou,
                "intersection": intersection,
                "pred_area": pred_area,
                "gt_area": gt_area,
                "qwen_text": sample["qwen_text"],
                "qwen_box": json.dumps(sample["qwen_box"]),
            }
            rows.append(row)
            method_rows.append(row)
        dice_values = np.asarray([row["dice"] for row in method_rows])
        iou_values = np.asarray([row["iou"] for row in method_rows])
        total_intersection = sum(row["intersection"] for row in method_rows)
        total_pred = sum(row["pred_area"] for row in method_rows)
        total_gt = sum(row["gt_area"] for row in method_rows)
        summaries.append(
            {
                "group": group,
                "method": method,
                "n": len(method_rows),
                "dice_mean": float(dice_values.mean()),
                "dice_std": float(dice_values.std()),
                "dice_median": float(np.median(dice_values)),
                "iou_mean": float(iou_values.mean()),
                "zero_dice_rate": float((dice_values == 0).mean()),
                "micro_dice": float(
                    2 * total_intersection / max(total_pred + total_gt, 1)
                ),
            }
        )
    return rows, summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    support_payload = args.support_manifest.read_bytes()
    support_rows = read_jsonl(args.support_manifest.resolve())
    support_ids = {row["merged_id"] for row in support_rows}
    if len(support_ids) != 16 or any(row["split"] != "train" for row in support_rows):
        raise RuntimeError("Frozen support manifest must contain 16 train-only samples")

    samples = load_all(args.candidate_root.resolve())
    by_id = {sample["merged_id"]: sample for sample in samples["train"]}
    missing_support = sorted(support_ids - set(by_id))
    if missing_support:
        raise RuntimeError(f"Candidate files missing for support samples: {missing_support}")
    support_samples = [by_id[merged_id] for merged_id in sorted(support_ids)]

    grid, grid_support_dice = select_grid_weights(support_samples)
    alpha, alpha_cv = select_ridge_alpha(support_samples)
    ridge_parameters = fit_ridge(support_samples, alpha)
    coefficients, mean, scale = ridge_parameters
    frozen = {
        "primary_method": "support_ridge",
        "supervision": "exactly 16 frozen train masks; no validation/test labels",
        "support_count": len(support_samples),
        "support_manifest_sha256": hashlib.sha256(support_payload).hexdigest(),
        "base_feature_names": BASE_FEATURE_NAMES,
        "fixed_no_gt_weights": FIXED_WEIGHTS.tolist(),
        "support_grid_weights": grid.tolist(),
        "support_grid_training_dice": grid_support_dice,
        "ridge_feature_names": RIDGE_FEATURE_NAMES,
        "ridge_alpha_selection": "leave-one-support-image-out CV",
        "ridge_alpha_cv_dice": alpha_cv,
        "ridge_alpha": alpha,
        "ridge_intercept_and_coefficients": coefficients.tolist(),
        "ridge_standardization_mean": mean.tolist(),
        "ridge_standardization_scale": scale.tolist(),
        "freeze_order": "selector parameters written before validation/test evaluation",
    }
    frozen_path = output_root / "frozen_selector.json"
    frozen_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    groups = {
        "train_support": support_samples,
        "train_non_support": [
            sample for sample in samples["train"] if sample["merged_id"] not in support_ids
        ],
        "train_all": samples["train"],
        "validation": samples["validation"],
        "test": samples["test"],
    }
    all_rows = []
    all_summaries = []
    for group, group_samples in groups.items():
        if not group_samples:
            continue
        rows, summaries = evaluate(
            group_samples, group, support_ids, grid, ridge_parameters
        )
        all_rows.extend(rows)
        all_summaries.extend(summaries)
        for source_dataset in ("CVC-ClinicDB", "kvasir-seg"):
            source_samples = [
                sample
                for sample in group_samples
                if sample["source_dataset"] == source_dataset
            ]
            if source_samples:
                source_rows, source_summaries = evaluate(
                    source_samples,
                    f"{group}/{source_dataset}",
                    support_ids,
                    grid,
                    ridge_parameters,
                )
                all_rows.extend(source_rows)
                all_summaries.extend(source_summaries)

    write_csv(output_root / "per_sample.csv", all_rows)
    write_csv(output_root / "summary.csv", all_summaries)
    (output_root / "summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("FROZEN SELECTOR")
    print(json.dumps(frozen, ensure_ascii=False, indent=2))
    print("PRIMARY RESULTS")
    for row in all_summaries:
        if row["method"] == "support_ridge" and row["group"] in {
            "train_non_support",
            "validation",
            "test",
        }:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
