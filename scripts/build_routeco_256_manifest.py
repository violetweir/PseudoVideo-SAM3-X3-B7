#!/usr/bin/env python3
"""Build a RouteCo training manifest on the C0-256-base mainline.

Unlike the old 1008 routeco_v1 manifest, this builder:
  * reads routes from work/rerun_c0_256_base/stage1_feature_knn_b7_base
  * restricts targets to the current pseudo_manifest_x3.jsonl (or any supplied manifest)
  * uses the X3 pseudo mask (binary) as the SAM3 training target
  * keeps the full anchor/bridge route so SAM3 can be trained in propagation form
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_quality(root: Path, mode: str, split: str) -> list[dict]:
    path = root / mode / f"propagation_quality_{split}/propagation_quality.jsonl"
    rows = read_jsonl(path)
    for row in rows:
        row["feature_mode"] = mode
    return rows


def mask_array(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum() + b.sum())
    return 1.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


def route_uncertainty(candidates: list[dict], selected: dict) -> dict:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--quality-root", type=Path, required=True,
                        help="e.g. work/rerun_c0_256_base/stage1_feature_knn_b7_base")
    parser.add_argument("--pseudo-manifest", type=Path, required=True,
                        help="e.g. work/rerun_c0_256_base/pseudo_manifest_x3.jsonl")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="e.g. work/rerun_c0_256_base/routeco_256_base")
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    args = parser.parse_args()

    quality_root = args.root / args.quality_root if not args.quality_root.is_absolute() else args.quality_root
    pseudo_manifest = args.root / args.pseudo_manifest if not args.pseudo_manifest.is_absolute() else args.pseudo_manifest
    output_root = args.root / args.output_root if not args.output_root.is_absolute() else args.output_root

    x3_rows = read_jsonl(pseudo_manifest)
    x3_by_target = {row["target_id"]: row for row in x3_rows}
    print(f"pseudo manifest targets: {len(x3_by_target)}", flush=True)

    train_rows = []
    val_rows = []
    for mode in MODES:
        train_rows.extend(load_quality(quality_root, mode, "train"))
        val_rows.extend(load_quality(quality_root, mode, "validation"))

    train_rows = [
        row for row in train_rows
        if args.min_bridge <= int(row["bridge_count"]) <= args.max_bridge
    ]
    val_rows = [
        row for row in val_rows
        if args.min_bridge <= int(row["bridge_count"]) <= args.max_bridge
    ]
    print(f"train candidate rows: {len(train_rows)}, val candidate rows: {len(val_rows)}", flush=True)

    scorer = pqr.Ridge.fit(val_rows, ridge=1.0, include_mode=True)

    by_target: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows:
        by_target[row["target_id"]].append(row)

    manifest = []
    missing = []
    for target_id, x3 in sorted(x3_by_target.items()):
        candidates = by_target.get(target_id, [])
        if not candidates:
            missing.append(target_id)
            continue

        # If the X3 pseudo mask is exactly one of the SAM3 forward masks, keep that route.
        selected = None
        mask_path = str(x3.get("pseudo_mask_path") or "")
        if "/forward_masks/" in mask_path:
            route_id = Path(mask_path).stem
            for row in candidates:
                if row["route_id"] == route_id:
                    selected = row
                    break
        if selected is None:
            selected = max(
                candidates,
                key=lambda row: (
                    scorer.score(row),
                    row["feature_mode"],
                    -int(row["bridge_count"]),
                    row["route_id"],
                ),
            )

        uncertainty = route_uncertainty(candidates, selected)
        manifest.append({
            **selected,
            "pseudo_mask_path": str(x3["pseudo_mask_path"]),
            "pseudo_consensus_path": str(x3["pseudo_consensus_path"]) if x3.get("pseudo_consensus_path") else None,
            "pixel_weight_path": str(x3["pixel_weight_path"]) if x3.get("pixel_weight_path") else None,
            "sample_type": x3.get("sample_type", "unknown"),
            "x3_q_multi": x3.get("q_multi"),
            "x3_q_model_mean": x3.get("q_model_mean"),
            "x3_q_model_var": x3.get("q_model_var"),
            "explicit_quality_weight": float(x3.get("explicit_quality_weight", 1.0)),
            "routeco_stage": "c0_256_base_v1",
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
        })

    if missing:
        print(f"WARNING: missing routes for {len(missing)} targets, e.g. {missing[:5]}", flush=True)

    write_jsonl(output_root / "routeco_256_manifest.jsonl", manifest)
    summary = {
        "n_targets": len(manifest),
        "missing": len(missing),
        "sample_types": {
            typ: sum(1 for r in manifest if r["sample_type"] == typ)
            for typ in sorted({r["sample_type"] for r in manifest})
        },
        "selected_counts": {},
        "mean_route_mask_variance": float(
            np.mean([r["routeco_uncertainty"]["route_mask_mean_variance"] for r in manifest])
        ) if manifest else 0.0,
    }
    for row in manifest:
        key = f"{row['feature_mode']}:{row['route_type']}"
        summary["selected_counts"][key] = summary["selected_counts"].get(key, 0) + 1
    write_jsonl(output_root / "routeco_256_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
