#!/usr/bin/env python3
"""Extract lesion descriptors with GT masks for TEST images (analysis-only upper bound).

Same formula as extract_sam3_mask_descriptors.py (mean of SAM3-trunk patch
tokens inside the mask, L2-normalized) but computed from the cached @1008
patch tokens (extract_sam3_encoder_patches.py) with:
  - test split  -> GT mask (merged_manifest.mask_path)   [analysis ONLY, never a method]
  - train/val   -> round-1 pseudo mask (same as pseudo lesion)

Output npz: ids (1000), descriptors (1000, 1024), feature_size=1008.
Feeds stage1_feature_knn_routes.py --lesion-desc-npz for the sam3enc_lesion mode.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
STAGE1 = ROOT / "scripts/stage1_feature_knn_routes.py"
spec = importlib.util.spec_from_file_location("stage1", STAGE1)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {STAGE1}")
stage1 = importlib.util.module_from_spec(spec)
sys.modules["stage1"] = stage1
spec.loader.exec_module(stage1)

PATCHES = (
    ROOT
    / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/sam3_base_s1008_patches.npz"
)
PSEUDO_DIRS = [
    ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1",
    ROOT / "work/kvasir_1pct_anchors/validation_pseudo_masks_round1",
    ROOT / "work/kvasir_1pct_anchors/test_pseudo_masks_round1",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/protocol")
    parser.add_argument("--patches", type=Path, default=PATCHES)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_lesion_knn/features/sam3enc_gt_lesion_descriptors_s1008.npz",
    )
    parser.add_argument("--feature-size", type=int, default=1008)
    args = parser.parse_args()

    records = stage1.read_jsonl(args.protocol_root / "merged_manifest.jsonl")
    # pseudo-mask map (round1): target_id -> pseudo_mask_path
    pseudo_map: dict[str, str] = {}
    for d in PSEUDO_DIRS:
        p = d / "train_pseudo_masks_round1.jsonl"
        if p.exists():
            for row in stage1.read_jsonl(p):
                pseudo_map[row["target_id"]] = row["pseudo_mask_path"]
    anchor_manifest = ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/anchor_mask_manifest.jsonl"
    if anchor_manifest.exists():
        for row in stage1.read_jsonl(anchor_manifest):
            pseudo_map[row["target_id"]] = row["pseudo_mask_path"]

    grid = args.feature_size // 14
    data = np.load(args.patches, mmap_mode="r")
    patches = data["patches"]  # (1000, P, C) fp16
    descs, ids = [], []
    n_gt = 0
    for record in records:
        mid = record["merged_id"]
        if record["split"] == "test":
            mask = record["mask_path"]  # GT (analysis only)
            n_gt += 1
        else:
            mask = pseudo_map.get(mid)
        if not mask or not Path(mask).exists():
            raise SystemExit(f"Missing mask for {mid} (split={record['split']})")
        idx = next(i for i, r in enumerate(records) if r["merged_id"] == mid)
        tok = patches[idx].astype(np.float32)
        fg = stage1.load_mask_grid(mask, grid).reshape(-1)
        sel = tok[fg]
        if len(sel) == 0:
            sel = tok
        descs.append(stage1.l2norm(sel.mean(axis=0, keepdims=True))[0])
        ids.append(mid)
    arr = np.stack(descs).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=np.asarray(ids), descriptors=arr, feature_size=args.feature_size)
    print(json.dumps({"output": str(args.output), "rows": len(ids), "dim": int(arr.shape[1]),
                      "test_gt_masks": n_gt}, indent=2))


if __name__ == "__main__":
    main()
