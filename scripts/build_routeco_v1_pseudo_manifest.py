#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
PQR_PATH = ROOT / "scripts/analyze_propagation_quality_router.py"
spec = importlib.util.spec_from_file_location("pqr", PQR_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {PQR_PATH}")
pqr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pqr
spec.loader.exec_module(pqr)

MODES = [
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_quality(root: Path, mode: str, split: str, route_root_name: str) -> list[dict[str, Any]]:
    if split == "validation":
        path = (
            root
            / "work/kvasir_1pct_anchors/stage1_feature_knn_b7_validation"
            / mode
            / "propagation_quality_validation/propagation_quality.jsonl"
        )
    elif split == "test":
        path = (
            root
            / "work/kvasir_1pct_anchors/stage1_feature_knn_b7"
            / mode
            / "propagation_quality_test/propagation_quality.jsonl"
        )
    else:
        path = (
            root
            / "work/kvasir_1pct_anchors"
            / route_root_name
            / mode
            / "propagation_quality_train/propagation_quality.jsonl"
        )
    rows = read_jsonl(path)
    for row in rows:
        row["feature_mode"] = mode
    return rows


def mask_array(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum() + b.sum())
    return 1.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


def target_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["target_id"], []).append(row)
    return out


def route_uncertainty(candidates: list[dict[str, Any]], selected: dict[str, Any]) -> dict[str, float]:
    masks = []
    for row in candidates:
        path = row.get("forward_mask_path")
        if path and Path(path).exists():
            masks.append((row, mask_array(path)))
    if len(masks) < 2:
        return {
            "route_mask_mean_variance": 0.0,
            "route_selected_mean_disagreement": 0.0,
            "route_candidate_count": float(len(masks)),
        }
    stack = np.stack([mask.astype(np.float32) for _, mask in masks], axis=0)
    selected_mask = mask_array(selected["forward_mask_path"])
    disagreements = [
        1.0 - dice(selected_mask, mask)
        for row, mask in masks
        if row["route_id"] != selected["route_id"]
    ]
    return {
        "route_mask_mean_variance": float(stack.var(axis=0).mean()),
        "route_selected_mean_disagreement": float(np.mean(disagreements)) if disagreements else 0.0,
        "route_candidate_count": float(len(masks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--route-root-name", default="routeco_sam3_v1_routes")
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    train_rows = []
    val_rows = []
    for mode in MODES:
        train_rows.extend(load_quality(args.root, mode, "train", args.route_root_name))
        val_rows.extend(load_quality(args.root, mode, "validation", args.route_root_name))
    train_rows = [
        row for row in train_rows if args.min_bridge <= int(row["bridge_count"]) <= args.max_bridge
    ]
    val_rows = [
        row for row in val_rows if args.min_bridge <= int(row["bridge_count"]) <= args.max_bridge
    ]
    scorer = pqr.Ridge.fit(val_rows, ridge=1.0, include_mode=True)

    output_root = args.output_root or (
        args.root / "work/kvasir_1pct_anchors/routeco_sam3_v1"
    )
    pseudo_mask_root = output_root / "pseudo_masks"
    pseudo_mask_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for target_id, candidates in sorted(target_groups(train_rows).items()):
        selected = max(
            candidates,
            key=lambda row: (
                scorer.score(row),
                row["feature_mode"],
                -int(row["bridge_count"]),
                row["route_id"],
            ),
        )
        safe_id = target_id.replace("::", "__").replace("/", "_")
        pseudo_path = pseudo_mask_root / f"{safe_id}.png"
        Image.open(selected["forward_mask_path"]).save(pseudo_path)
        uncertainty = route_uncertainty(candidates, selected)
        manifest.append(
            {
                **selected,
                "pseudo_mask_path": str(pseudo_path),
                "routeco_stage": "v1_pseudo_warm_start",
                "routeco_candidate_modes": MODES,
                "routeco_candidate_min_bridge": args.min_bridge,
                "routeco_candidate_max_bridge": args.max_bridge,
                "routeco_selected_score": float(scorer.score(selected)),
                "routeco_uncertainty": uncertainty,
                "student_aug_variance": None,
                "student_weak_strong_variance": None,
                "t_to_s_gate": None,
                "s_to_t_gate": None,
                "target_gt_used_for_selection": False,
            }
        )
    write_jsonl(output_root / "routeco_v1_pseudo_manifest.jsonl", manifest)
    summary = {
        "n": len(manifest),
        "candidate_modes": MODES,
        "min_bridge": args.min_bridge,
        "max_bridge": args.max_bridge,
        "selected_counts": {},
        "mean_route_mask_variance": float(
            np.mean([row["routeco_uncertainty"]["route_mask_mean_variance"] for row in manifest])
        )
        if manifest
        else 0.0,
    }
    for row in manifest:
        key = f"{row['feature_mode']}:{row['route_type']}"
        summary["selected_counts"][key] = summary["selected_counts"].get(key, 0) + 1
    (output_root / "routeco_v1_pseudo_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
