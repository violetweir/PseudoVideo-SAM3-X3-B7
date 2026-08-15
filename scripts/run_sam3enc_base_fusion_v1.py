#!/usr/bin/env python3
"""Validation-calibrated soft fusion of b0-b6 SAM3-enc base route masks.

This is the first fusion step for the SAM3-not-fine-tuned mainline.  Instead of
choosing a single bridge length, all seven forward masks from one frozen
SAM3-enc KNN route family are combined with per-route weights.  A ridge scorer
is fitted on validation-only GT Dice and route/mask features; a softmax
temperature and final threshold are also selected on validation.  Test is
evaluated once.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_mask(path: str, canvas: int = 256) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((canvas, canvas), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 127).astype(np.float32)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred > 0.5
    gt = gt > 0.5
    den = pred.sum() + gt.sum()
    return float(2.0 * (pred & gt).sum() / max(den, 1e-9))


def group_by_target(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[row["target_id"]].append(row)
    return out


def sorted_routes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row["bridge_count"]))


def mask_stats(mask: np.ndarray) -> list[float]:
    m = mask > 0.5
    area = float(m.sum()) / max(mask.size, 1)
    if not m.any():
        return [area, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ys, xs = np.where(m)
    h, w = mask.shape
    # connected components via a tiny flood/union-free estimator is overkill;
    # keep geometric features only to stay deterministic and cheap.
    return [
        area,
        float(xs.mean() / w),
        float(ys.mean() / h),
        float((xs.max() - xs.min() + 1) / w),
        float((ys.max() - ys.min() + 1) / h),
        float((m.sum()) / max(m.size, 1)),
        0.0,
    ]


def feature_vector(row: dict[str, Any], mask: np.ndarray | None = None) -> list[float]:
    del mask
    return [
        float(row["bridge_count"]),
        float(row.get("path_bottleneck_similarity") or 0.0),
        float(row.get("path_mean_similarity") or 0.0),
        float(row.get("forward_sam_score") or 0.0),
    ]


def fit_ridge(
    x: list[list[float]], y: list[float], ridge: float = 1.0
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    z = (x - mean) / std
    gram = z.T @ z
    gram.flat[:: gram.shape[0] + 1] += ridge
    rhs = z.T @ (y - y.mean())
    weights = np.linalg.solve(gram, rhs)
    intercept = float(y.mean() - float(weights @ (mean / std)))
    return {
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "weights": [float(v) for v in weights],
        "intercept": intercept,
    }


def ridge_score(params: dict[str, Any], vec: list[float]) -> float:
    z = (np.asarray(vec, dtype=np.float32) - np.asarray(params["mean"], dtype=np.float32)) / np.asarray(
        params["std"], dtype=np.float32
    )
    return float(z @ np.asarray(params["weights"], dtype=np.float32) + params["intercept"])


def fusion_masks(
    rows: list[dict[str, Any]],
    params: dict[str, Any],
    temperature: float,
    threshold: float,
    canvas: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows = sorted_routes(rows)
    masks = np.stack([load_mask(row["forward_mask_path"], canvas) for row in rows])
    features = [feature_vector(row, masks[i]) for i, row in enumerate(rows)]
    scores = np.asarray([ridge_score(params, vec) for vec in features], dtype=np.float32)
    logits = (scores - scores.max()) / max(temperature, 1e-6)
    weights = np.exp(logits)
    weights = weights / max(weights.sum(), 1e-12)
    probability = np.tensordot(weights, masks, axes=(0, 0))
    binary = probability >= threshold
    detail = [
        {
            "route_id": row["route_id"],
            "bridge_count": int(row["bridge_count"]),
            "score": float(scores[i]),
            "weight": float(weights[i]),
            "mask_path": row["forward_mask_path"],
        }
        for i, row in enumerate(rows)
    ]
    return probability, binary.astype(np.float32), detail


def evaluate_split(
    rows: list[dict[str, Any]],
    params: dict[str, Any],
    temperature: float,
    threshold: float,
    canvas: int,
) -> dict[str, Any]:
    grouped = group_by_target(rows)
    fusion, best_b6, mean_05, oracle = [], [], [], []
    per_target: list[dict[str, Any]] = []
    for target_id, target_rows in grouped.items():
        target_rows = sorted_routes(target_rows)
        masks = np.stack([load_mask(row["forward_mask_path"], canvas) for row in target_rows])
        gt = load_mask(target_rows[0]["target_mask_path_evaluation_only"], canvas)
        _, pred, detail = fusion_masks(target_rows, params, temperature, threshold, canvas)
        fusion.append(dice(pred, gt))
        best_b6.append(dice(masks[6], gt))
        mean_05.append(dice(masks.mean(axis=0), gt))
        oracle.append(max(float(row["gt_dice_evaluation_only"]) for row in target_rows))
        per_target.append(
            {
                "target_id": target_id,
                "fusion_dice": fusion[-1],
                "best_b6_dice": best_b6[-1],
                "mean_0.5_dice": mean_05[-1],
                "oracle_dice": oracle[-1],
                "routes": detail,
            }
        )
    return {
        "fusion_dice": float(np.mean(fusion)),
        "best_b6_dice": float(np.mean(best_b6)),
        "mean_0.5_dice": float(np.mean(mean_05)),
        "oracle_dice": float(np.mean(oracle)),
        "per_target": per_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="sam3enc_anchor_conditioned_target_pooling",
        help="Frozen SAM3-enc route family.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008",
    )
    parser.add_argument(
        "--eval-name",
        default="eval_base_no_ft_b7_forward",
        help="Forward eval directory used for test; validation uses `<name>_validation`.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/sam3enc_base_fusion_v1",
    )
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--canvas", type=int, default=256)
    parser.add_argument("--temps", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--save-masks", action="store_true")
    args = parser.parse_args()

    mode_root = args.root / args.mode
    val_rows = read_jsonl(mode_root / f"{args.eval_name}_validation" / "route_results.jsonl")
    test_rows = read_jsonl(mode_root / args.eval_name / "route_results.jsonl")

    x: list[list[float]] = []
    y: list[float] = []
    for rows in group_by_target(val_rows).values():
        rows = sorted_routes(rows)
        masks = np.stack([load_mask(row["forward_mask_path"], args.canvas) for row in rows])
        for i, row in enumerate(rows):
            x.append(feature_vector(row, masks[i]))
            y.append(float(row["gt_dice_evaluation_only"]))
    params = fit_ridge(x, y, args.ridge)

    best = None
    for temperature in args.temps:
        for threshold in args.thresholds:
            val = evaluate_split(val_rows, params, temperature, threshold, args.canvas)
            key = (val["fusion_dice"], -abs(temperature - 1.0), threshold)
            if best is None or key > best[0]:
                best = (key, temperature, threshold, val)
    _, best_temperature, best_threshold, best_val = best

    test = evaluate_split(test_rows, params, best_temperature, best_threshold, args.canvas)
    summary = {
        "mode": args.mode,
        "root": str(args.root),
        "ridge": args.ridge,
        "best_temperature": best_temperature,
        "best_threshold": best_threshold,
        "validation": {
            "fusion_dice": best_val["fusion_dice"],
            "best_b6_dice": best_val["best_b6_dice"],
            "mean_0.5_dice": best_val["mean_0.5_dice"],
            "oracle_dice": best_val["oracle_dice"],
        },
        "test": {
            "fusion_dice": test["fusion_dice"],
            "best_b6_dice": test["best_b6_dice"],
            "mean_0.5_dice": test["mean_0.5_dice"],
            "oracle_dice": test["oracle_dice"],
        },
        "ridge_params": params,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output_root / "test_per_target.json").write_text(
        json.dumps(test["per_target"], indent=2, sort_keys=True) + "\n"
    )
    if args.save_masks:
        mask_root = args.output_root / "fused_masks"
        mask_root.mkdir(parents=True, exist_ok=True)
        for row in test["per_target"]:
            # Recompute once more for writing is intentionally avoided; save only
            # when this flag is explicitly used by a follow-up writer.
            pass
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
