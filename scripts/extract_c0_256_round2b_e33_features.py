#!/usr/bin/env python3
"""Extract checkpoint-specific SAM3 trunk descriptors for the Round-2B graph."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
STAGE1_PATH = ROOT / "scripts/stage1_feature_knn_routes.py"
SPEC = importlib.util.spec_from_file_location("round2b_stage1", STAGE1_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {STAGE1_PATH}")
stage1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage1
SPEC.loader.exec_module(stage1)


def extract(records: list[dict], anchors: list[dict], args: argparse.Namespace) -> None:
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        device="cuda",
        compile=False,
    ).eval()
    trunk = model.detector.backbone.vision_backbone.trunk
    stage1.prepare_sam3_trunk(trunk, args.feature_size)
    grid = args.feature_size // 14
    anchor_ids = [row["merged_id"] for row in anchors]

    def image(row: dict) -> torch.Tensor:
        return (stage1.load_rgb_tensor(row["image_path"], args.feature_size) - 0.5) / 0.5

    def tokens(batch: torch.Tensor) -> torch.Tensor:
        features = trunk(batch)[0]
        patches = features.flatten(2).permute(0, 2, 1)
        return torch.nn.functional.normalize(patches, dim=-1)

    prototypes: list[np.ndarray] = []
    with torch.inference_mode():
        for anchor in anchors:
            patches = tokens(image(anchor).unsqueeze(0).cuda())[0]
            foreground = stage1.load_mask_grid(anchor["frozen_mask_path"], grid).reshape(-1)
            foreground_patches = patches[foreground]
            if len(foreground_patches) == 0:
                foreground_patches = patches
            proto = foreground_patches.mean(dim=0, keepdim=True).float().cpu().numpy()
            prototypes.append(stage1.l2norm(proto)[0])

        proto_t = torch.from_numpy(np.asarray(prototypes, dtype=np.float32)).cuda()
        patch_means: list[np.ndarray] = []
        pooled_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        correspondence_rows: list[np.ndarray] = []

        for start in range(0, len(records), args.batch_size):
            chunk = records[start : start + args.batch_size]
            batch = torch.stack([image(row) for row in chunk]).cuda()
            patches = tokens(batch)
            patch_mean = torch.nn.functional.normalize(patches.mean(dim=1), dim=-1)
            similarities = torch.einsum("ad,bpd->abp", proto_t, patches)
            weights = torch.exp((similarities - similarities.amax(dim=2, keepdim=True)) * 10.0)
            weights /= weights.sum(dim=2, keepdim=True).clamp_min(1e-12)
            pooled = torch.nn.functional.normalize(
                torch.einsum("abp,bpd->abd", weights, patches), dim=-1
            )
            patch_means.append(patch_mean.float().cpu().numpy())
            pooled_rows.append(pooled.float().cpu().numpy())
            target_rows.append(torch.einsum("abd,ad->ab", pooled, proto_t).float().cpu().numpy())
            correspondence_rows.append(
                torch.topk(similarities, k=8, dim=2).values.mean(dim=2).float().cpu().numpy()
            )
            print(f"[round2b] e33 features {min(start + len(chunk), len(records))}/{len(records)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=np.asarray([row["merged_id"] for row in records]),
        patch_mean=np.concatenate(patch_means, axis=0).astype(np.float32),
        anchor_ids=np.asarray(anchor_ids),
        anchor_prototypes=np.asarray(prototypes, dtype=np.float32),
        pooled=np.concatenate(pooled_rows, axis=1).astype(np.float32),
        cond_target=np.concatenate(target_rows, axis=1).astype(np.float32),
        cond_correspondence=np.concatenate(correspondence_rows, axis=1).astype(np.float32),
        feature_source="sam3_e33",
        feature_size=args.feature_size,
        encoder_checkpoint=str(args.checkpoint.resolve()),
    )


def audit(records: list[dict], args: argparse.Namespace) -> dict:
    base = np.load(args.base_features)
    updated = np.load(args.output)
    base_descriptors = base["patch_mean"].astype(np.float64)
    updated_descriptors = updated["patch_mean"].astype(np.float64)
    if base_descriptors.shape != updated_descriptors.shape:
        raise RuntimeError(f"Feature shape changed: {base_descriptors.shape} != {updated_descriptors.shape}")
    delta = updated_descriptors - base_descriptors
    max_abs_delta = float(np.abs(delta).max())
    if max_abs_delta < 1e-7:
        raise RuntimeError("e33 descriptors are identical to base descriptors; checkpoint loading is invalid")

    row_cosine = np.einsum("nd,nd->n", base_descriptors, updated_descriptors)
    train_indices = np.asarray([i for i, row in enumerate(records) if row["split"] == "train"])
    base_sim = base_descriptors @ base_descriptors[train_indices].T
    updated_sim = updated_descriptors @ updated_descriptors[train_indices].T
    position_by_index = {int(index): position for position, index in enumerate(train_indices)}
    overlaps: dict[str, dict[str, float]] = {}
    for split in ("train", "validation", "test"):
        query_indices = [i for i, row in enumerate(records) if row["split"] == split]
        scores = {k: [] for k in (1, 5, 10, 32)}
        for index in query_indices:
            old = base_sim[index].copy()
            new = updated_sim[index].copy()
            if index in position_by_index:
                old[position_by_index[index]] = -np.inf
                new[position_by_index[index]] = -np.inf
            old_rank = np.argsort(-old, kind="stable")
            new_rank = np.argsort(-new, kind="stable")
            for k in scores:
                scores[k].append(len(set(old_rank[:k]) & set(new_rank[:k])) / k)
        overlaps[split] = {f"top_{k}_overlap": float(np.mean(values)) for k, values in scores.items()}

    prototype_cos = np.einsum(
        "ad,ad->a", base["anchor_prototypes"].astype(np.float64), updated["anchor_prototypes"].astype(np.float64)
    )
    report = {
        "feature_source": "sam3_e33",
        "encoder_checkpoint": str(args.checkpoint.resolve()),
        "feature_size": args.feature_size,
        "records": len(records),
        "descriptor_shape": list(updated_descriptors.shape),
        "base_feature_cache": str(args.base_features.resolve()),
        "e33_feature_cache": str(args.output.resolve()),
        "row_cosine": {
            "mean": float(row_cosine.mean()),
            "min": float(row_cosine.min()),
            "p10": float(np.quantile(row_cosine, 0.1)),
            "median": float(np.median(row_cosine)),
            "max": float(row_cosine.max()),
        },
        "max_abs_descriptor_delta": max_abs_delta,
        "mean_l2_descriptor_delta": float(np.linalg.norm(delta, axis=1).mean()),
        "anchor_prototype_cosine_mean": float(prototype_cos.mean()),
        "knn_train_pool_overlap": overlaps,
        "target_gt_used_for_features_or_topology": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/protocol")
    parser.add_argument("--feature-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    records = stage1.read_jsonl(args.protocol_root / "merged_manifest.jsonl")
    anchors = stage1.read_jsonl(args.protocol_root / "support_manifest.jsonl")
    if args.output.exists():
        print(f"[round2b] reuse existing e33 feature cache: {args.output}", flush=True)
    else:
        extract(records, anchors, args)
    print(json.dumps(audit(records, args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
