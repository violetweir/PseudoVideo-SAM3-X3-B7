#!/usr/bin/env python3
"""Extract and cache SAM3-trunk patch tokens @1008 (fp16) for anchor-count ablation.

1000 images x (72*72=5184) x 1024 dims fp16 ≈ 10.6 GB. Reused by
build_sam3enc_anchor_subsets.py to recompute per-anchor pooled / cond scores
for ANY anchor subset without re-running the trunk encoder.

Output: {output}/sam3_base_s1008_patches.npz  (ids, patches fp16, feature_size)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
STAGE1 = ROOT / "scripts/stage1_feature_knn_routes.py"
spec = importlib.util.spec_from_file_location("stage1", STAGE1)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {STAGE1}")
stage1 = importlib.util.module_from_spec(spec)
sys.modules["stage1"] = stage1
spec.loader.exec_module(stage1)

SAM3_CKPT = "/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/protocol",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/sam3_base_s1008_patches.npz",
    )
    parser.add_argument("--feature-size", type=int, default=1008)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.output.exists():
        print(json.dumps({"output": str(args.output), "status": "exists"}))
        return
    records = stage1.read_jsonl(args.protocol_root / "merged_manifest.jsonl")

    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(
        checkpoint_path=SAM3_CKPT, load_from_HF=False, device="cuda", compile=False
    )
    model.eval()
    trunk = model.detector.backbone.vision_backbone.trunk
    stage1.prepare_sam3_trunk(trunk, args.feature_size)
    grid = args.feature_size // 14

    def preprocess(x: torch.Tensor) -> torch.Tensor:
        return (x - 0.5) / 0.5

    ids: list[str] = []
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            chunk = records[start : start + args.batch_size]
            x = torch.stack(
                [preprocess(stage1.load_rgb_tensor(row["image_path"], args.feature_size)) for row in chunk]
            ).cuda()
            feats = trunk(x)[0]  # (B, C, H, W)
            tokens = feats.flatten(2).permute(0, 2, 1)  # (B, P, C)
            tokens = torch.nn.functional.normalize(tokens, dim=-1)
            rows.append(tokens.half().float().cpu().numpy().astype(np.float16))
            ids.extend(row["merged_id"] for row in chunk)
            print(f"patches {min(start + args.batch_size, len(records))}/{len(records)}", flush=True)

    arr = np.concatenate(rows, axis=0).astype(np.float16)
    assert arr.shape == (len(records), grid * grid, 1024), arr.shape
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=np.asarray(ids),
        patches=arr,
        feature_size=args.feature_size,
    )
    print(json.dumps({"output": str(args.output), "rows": len(ids), "shape": list(arr.shape)}, indent=2))


if __name__ == "__main__":
    main()
