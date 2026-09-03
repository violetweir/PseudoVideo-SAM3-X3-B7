#!/usr/bin/env python3
"""Extract SAM3 FPN-transport descriptors from an explicit checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
STAGE1_PATH = ROOT / "scripts/stage1_feature_knn_routes.py"
SPEC = importlib.util.spec_from_file_location("fpn_transport_stage1", STAGE1_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {STAGE1_PATH}")
stage1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage1
SPEC.loader.exec_module(stage1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/protocol",
    )
    parser.add_argument("--feature-size", type=int, default=256)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    feature_root = args.output_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    feature_path = feature_root / f"sam3_base_s{args.feature_size}_fpn_transport_features.npz"
    summary_path = feature_root / f"sam3_e33_s{args.feature_size}_fpn_transport_audit.json"

    records = stage1.read_jsonl(args.protocol_root / "merged_manifest.jsonl")
    support = stage1.read_jsonl(args.protocol_root / "support_manifest.jsonl")
    if not feature_path.exists():
        stage1.SAM3_CKPT = str(checkpoint)
        stage1.extract_sam3_fpn_transport_features(
            records,
            support,
            feature_root,
            feature_size=args.feature_size,
        )

    updated = np.load(feature_path)
    baseline = np.load(args.base_cache)
    updated_desc = updated["descriptors"].astype(np.float64)
    baseline_desc = baseline["descriptors"].astype(np.float64)
    if updated_desc.shape != baseline_desc.shape:
        raise RuntimeError(f"Descriptor shape mismatch: {updated_desc.shape} != {baseline_desc.shape}")
    row_cosine = np.einsum("and,and->an", updated_desc, baseline_desc)
    delta = updated_desc - baseline_desc
    if float(np.abs(delta).max()) < 1e-7:
        raise RuntimeError("Checkpoint-specific FPN descriptors are identical to SAM3-base")

    summary = {
        "encoder_checkpoint": str(checkpoint),
        "feature_cache": str(feature_path.resolve()),
        "feature_size": args.feature_size,
        "descriptor_shape": list(updated_desc.shape),
        "base_vs_updated_row_cosine": {
            "mean": float(row_cosine.mean()),
            "min": float(row_cosine.min()),
            "p10": float(np.quantile(row_cosine, 0.1)),
            "median": float(np.median(row_cosine)),
            "max": float(row_cosine.max()),
        },
        "mean_l2_descriptor_delta": float(np.linalg.norm(delta, axis=2).mean()),
        "max_abs_descriptor_delta": float(np.abs(delta).max()),
        "records": len(records),
        "anchors": len(support),
        "target_gt_used_for_features_or_topology": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
