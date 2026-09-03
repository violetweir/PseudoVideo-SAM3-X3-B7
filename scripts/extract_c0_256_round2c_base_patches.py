#!/usr/bin/env python3
"""Cache SAM3-base trunk patch tokens at the frozen Round-2 256 protocol."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
BASE_CHECKPOINT = Path(
    "/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt"
)
STAGE1_PATH = ROOT / "scripts/stage1_feature_knn_routes.py"
SPEC = importlib.util.spec_from_file_location("round2c_stage1_features", STAGE1_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {STAGE1_PATH}")
stage1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage1
SPEC.loader.exec_module(stage1)


def select_records(protocol: Path, split: str, max_targets: int | None) -> list[dict]:
    records = stage1.read_jsonl(protocol / "merged_manifest.jsonl")
    anchors = stage1.read_jsonl(protocol / "support_manifest.jsonl")
    anchor_ids = {row["merged_id"] for row in anchors}
    targets = sorted(
        (row for row in records if row["split"] == split),
        key=lambda row: row["merged_id"],
    )
    if max_targets is not None:
        targets = targets[:max_targets]
    selected_ids = anchor_ids | {row["merged_id"] for row in targets}
    return [row for row in records if row["merged_id"] in selected_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/protocol")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--feature-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-targets", type=int)
    args = parser.parse_args()

    if args.feature_size != 256:
        raise SystemExit("Round-2C is strictly 256-only; --feature-size must be 256")
    records = select_records(args.protocol_root, args.split, args.max_targets)
    expected_ids = [row["merged_id"] for row in records]
    if args.output.exists():
        existing = np.load(args.output)
        if int(existing["feature_size"]) != 256:
            raise SystemExit(f"Existing cache is not @256: {args.output}")
        if existing["ids"].tolist() != expected_ids:
            raise SystemExit(f"Existing @256 patch cache covers different records: {args.output}")
        print(
            json.dumps(
                {
                    "status": "reused",
                    "output": str(args.output),
                    "feature_size": 256,
                    "rows": len(expected_ids),
                    "grid": int(existing["grid"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        device="cuda",
        compile=False,
    ).eval()
    trunk = model.detector.backbone.vision_backbone.trunk
    stage1.prepare_sam3_trunk(trunk, args.feature_size)

    chunks: list[np.ndarray] = []
    ids: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            rows = records[start : start + args.batch_size]
            batch = torch.stack(
                [(stage1.load_rgb_tensor(row["image_path"], 256) - 0.5) / 0.5 for row in rows]
            ).cuda()
            features = trunk(batch)[0]
            tokens = features.flatten(2).permute(0, 2, 1)
            tokens = torch.nn.functional.normalize(tokens, dim=-1)
            chunks.append(tokens.to(dtype=torch.float16).cpu().numpy())
            ids.extend(row["merged_id"] for row in rows)
            print(f"[round2c] base@256 patch tokens {len(ids)}/{len(records)}", flush=True)

    patches = np.concatenate(chunks, axis=0)
    grid = math.isqrt(patches.shape[1])
    if grid * grid != patches.shape[1]:
        raise RuntimeError(f"Unexpected non-square token grid: {patches.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=np.asarray(ids),
        patches=patches,
        feature_size=np.asarray(256),
        grid=np.asarray(grid),
        encoder_checkpoint=np.asarray(str(args.checkpoint.resolve())),
        split=np.asarray(args.split),
    )
    print(
        json.dumps(
            {
                "status": "created",
                "output": str(args.output),
                "shape": list(patches.shape),
                "dtype": str(patches.dtype),
                "feature_size": 256,
                "grid": grid,
                "split": args.split,
                "anchor_and_validation_only": True,
                "target_gt_used_for_search_or_inference": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
