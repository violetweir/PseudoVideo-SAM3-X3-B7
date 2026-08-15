#!/usr/bin/env python3
"""Apply the validation-fitted base fusion model to train split route masks.

Inputs:
  - base fusion summary: work/kvasir_1pct_anchors/sam3enc_base_fusion_v1/summary.json
  - train forward results: .../eval_base_no_ft_b7_forward_train/route_results.jsonl

Outputs:
  - one fused binary mask per unlabeled train target
  - train_pseudo_masks_round1.jsonl manifest
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


def feature_vector(row: dict[str, Any]) -> list[float]:
    return [
        float(row["bridge_count"]),
        float(row.get("path_bottleneck_similarity") or 0.0),
        float(row.get("path_mean_similarity") or 0.0),
        float(row.get("forward_sam_score") or 0.0),
    ]


def ridge_score(params: dict[str, Any], vec: list[float]) -> float:
    mean = np.asarray(params["mean"], dtype=np.float32)
    std = np.asarray(params["std"], dtype=np.float32)
    weights = np.asarray(params["weights"], dtype=np.float32)
    z = (np.asarray(vec, dtype=np.float32) - mean) / std
    return float(z @ weights + params["intercept"])


def fuse_for_target(
    rows: list[dict[str, Any]],
    params: dict[str, Any],
    temperature: float,
    threshold: float,
    canvas: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows = sorted(rows, key=lambda row: int(row["bridge_count"]))
    masks = np.stack([load_mask(row["forward_mask_path"], canvas) for row in rows])
    scores = np.asarray([ridge_score(params, feature_vector(row)) for row in rows], dtype=np.float32)
    logits = (scores - scores.max()) / max(temperature, 1e-6)
    weights = np.exp(logits)
    weights = weights / max(weights.sum(), 1e-12)
    probability = np.tensordot(weights, masks, axes=(0, 0))
    binary = (probability >= threshold).astype(np.uint8) * 255
    detail = [
        {
            "route_id": row["route_id"],
            "bridge_count": int(row["bridge_count"]),
            "score": float(scores[i]),
            "weight": float(weights[i]),
            "forward_mask_path": row["forward_mask_path"],
        }
        for i, row in enumerate(rows)
    ]
    return binary, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="sam3enc_anchor_conditioned_target_pooling",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008",
    )
    parser.add_argument(
        "--fit-summary",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/sam3enc_base_fusion_v1/summary.json",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument(
        "--eval-name",
        default=None,
        help="Forward eval directory name; inferred from split when omitted.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1",
    )
    parser.add_argument("--canvas", type=int, default=256)
    args = parser.parse_args()

    summary = json.loads(args.fit_summary.read_text())
    params = summary["ridge_params"]
    temperature = summary["best_temperature"]
    threshold = summary["best_threshold"]

    mode_root = args.root / args.mode
    if args.eval_name:
        eval_name = args.eval_name
    elif args.split == "train":
        eval_name = "eval_base_no_ft_b7_forward_train"
    elif args.split == "validation":
        eval_name = "eval_base_no_ft_b7_forward_validation"
    else:
        eval_name = "eval_base_no_ft_b7_forward"
    rows = read_jsonl(mode_root / eval_name / "route_results.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["target_id"]].append(row)

    mask_dir = args.output_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for target_id, target_rows in sorted(grouped.items()):
        binary, detail = fuse_for_target(target_rows, params, temperature, threshold, args.canvas)
        mask_path = mask_dir / f"{target_id}.png"
        Image.fromarray(binary).save(mask_path)
        manifest.append(
            {
                "target_id": target_id,
                "image_path": target_rows[0]["target_image_path"],
                "pseudo_mask_path": str(mask_path.resolve()),
                "source": f"round1_sam3_base_b0b6_fusion_{args.split}",
                "bridge_count_range": "b0-b6",
                "routes": detail,
            }
        )

    manifest_path = args.output_root / "train_pseudo_masks_round1.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )
    print(json.dumps({"targets": len(manifest), "manifest": str(manifest_path), "mask_dir": str(mask_dir)}, indent=2))


if __name__ == "__main__":
    main()
