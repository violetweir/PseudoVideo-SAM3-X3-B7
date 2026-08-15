#!/usr/bin/env python3
"""Extract SAM3-encoder mask-conditioned foreground descriptors.

For every record in the Kvasir 1% protocol, a mask (GT for anchors, round-1
pseudo mask otherwise) selects foreground SAM3 backbone patch tokens; their
mean is L2-normalized and stored as the visual mask descriptor.
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
spec = importlib.util.spec_from_file_location("stage1_features", STAGE1)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {STAGE1}")
stage1 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stage1
spec.loader.exec_module(stage1)


SAM3_CKPT = "/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/protocol")
    parser.add_argument(
        "--mask-manifest-dirs",
        nargs="+",
        type=Path,
        default=[
            ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1",
            ROOT / "work/kvasir_1pct_anchors/validation_pseudo_masks_round1",
            ROOT / "work/kvasir_1pct_anchors/test_pseudo_masks_round1",
        ],
    )
    parser.add_argument(
        "--anchor-manifest",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/anchor_mask_manifest.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/sam3enc_mask_descriptors_s1008.npz",
    )
    parser.add_argument("--feature-size", type=int, default=1008)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    records = read_jsonl(args.protocol_root / "merged_manifest.jsonl")
    mask_map: dict[str, str] = {}
    for manifest in args.mask_manifest_dirs:
        path = manifest / "train_pseudo_masks_round1.jsonl"
        if path.exists():
            for row in read_jsonl(path):
                mask_map[row["target_id"]] = row["pseudo_mask_path"]
    for row in read_jsonl(args.anchor_manifest):
        mask_map[row["target_id"]] = row["pseudo_mask_path"]

    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(checkpoint_path=SAM3_CKPT, load_from_HF=False, device="cuda", compile=False)
    model.eval()
    trunk = model.detector.backbone.vision_backbone.trunk
    stage1.prepare_sam3_trunk(trunk, args.feature_size)
    grid = args.feature_size // 14

    def preprocess(x: torch.Tensor) -> torch.Tensor:
        return (x - 0.5) / 0.5

    def trunk_tokens(x: torch.Tensor) -> torch.Tensor:
        feats = trunk(x)[0]
        tokens = feats.flatten(2).permute(0, 2, 1)
        return torch.nn.functional.normalize(tokens, dim=-1)

    descriptors = []
    ids = []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            chunk = records[start : start + args.batch_size]
            x = torch.stack(
                [preprocess(stage1.load_rgb_tensor(row["image_path"], args.feature_size)) for row in chunk]
            ).cuda()
            tokens = trunk_tokens(x)
            for local_idx, row in enumerate(chunk):
                mask_path = mask_map.get(row["merged_id"])
                if not mask_path:
                    descriptor = tokens[local_idx].mean(dim=0)
                else:
                    fg = stage1.load_mask_grid(mask_path, grid).reshape(-1)
                    selected = tokens[local_idx][fg]
                    if len(selected) == 0:
                        selected = tokens[local_idx]
                    descriptor = selected.mean(dim=0)
                descriptor = torch.nn.functional.normalize(descriptor, dim=-1)
                ids.append(row["merged_id"])
                descriptors.append(descriptor.cpu().numpy())
            print(f"{min(start + args.batch_size, len(records))}/{len(records)}", flush=True)

    arr = np.stack(descriptors).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=np.asarray(ids), descriptors=arr, feature_size=args.feature_size)
    print(json.dumps({"output": str(args.output), "rows": len(ids), "dim": int(arr.shape[1])}, indent=2))


if __name__ == "__main__":
    main()
