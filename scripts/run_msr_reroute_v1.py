#!/usr/bin/env python3
"""Round-2 Mask-aware Semantic Re-routing: first version.

Round-1 appearance KNN (SAM3-enc original patch_mean) recalls a Top-L train
candidate pool; within the pool we rerank by score-level fusion of visual, ROI
(mask-conditioned), shape-geometry and Qwen-text similarities, then rebuild a
pseudo-video route with the same Round-1 anchor and evaluate SAM3 @256.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
QF = ROOT / "scripts/build_qwen_text_knn_features.py"
spec = importlib.util.spec_from_file_location("qwen_features", QF)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {QF}")
qf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qf
spec.loader.exec_module(qf)

T21 = ROOT / "scripts/run_t21_dynamic_pseudovideo.py"
spec = importlib.util.spec_from_file_location("t21_dynamic", T21)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {T21}")
t21 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t21
spec.loader.exec_module(t21)


ROUTE_TYPES = {0: "direct", **{idx: f"bridge_{idx}" for idx in range(1, 8)}}
EXPERIMENTS = {
    "B1": {"visual": 0.5, "roi": 0.5},
    "B2": {"visual": 0.5, "shape": 0.5},
    "B3": {"visual": 0.5, "text": 0.5},
    "B4": {"visual": 1 / 3, "roi": 1 / 3, "shape": 1 / 3},
    "B5": {"visual": 0.25, "roi": 0.25, "shape": 0.25, "text": 0.25},
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def shape_vector(mask_path: str) -> np.ndarray:
    m = np.asarray(Image.open(mask_path).convert("L").resize((256, 256), Image.Resampling.NEAREST)) > 127
    if not m.any():
        return np.zeros(12, dtype=np.float32)
    ys, xs = np.where(m)
    h, w = m.shape
    area = m.mean()
    bbox_w = (xs.max() - xs.min() + 1) / w
    bbox_h = (ys.max() - ys.min() + 1) / h
    aspect = bbox_w / max(bbox_h, 1e-6)
    cx = xs.mean() / w
    cy = ys.mean() / h
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours) / max(2 * (w + h), 1e-6)
    hull_mask = np.zeros_like(m, dtype=np.uint8)
    if contours:
        hull = cv2.convexHull(np.concatenate(contours))
        cv2.fillConvexPoly(hull_mask, hull, 1)
    solidity = m.sum() / max(hull_mask.sum(), 1)
    compactness = 4 * np.pi * m.sum() / max(perimeter * perimeter * m.size, 1e-6)
    n, _ = cv2.connectedComponents(m.astype(np.uint8))
    components = float(max(n - 1, 0))
    extent = m.sum() / max(bbox_w * bbox_h * m.size, 1e-6)
    edges = cv2.Canny(m.astype(np.uint8), 100, 200).sum() / max(m.sum(), 1)
    vec = np.asarray(
        [area, bbox_w, bbox_h, aspect, cx, cy, perimeter, solidity, compactness, components, extent, edges],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def load_text_map(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for row in read_jsonl(path):
        out[row["target_id"]] = qf.to_vector(row.get("qwen_features") or {})
    return out


def route_key(target: dict, bridge_count: int, anchor_id: str, bridge_ids: list[str]) -> str:
    return f"{target['merged_id']}::{ROUTE_TYPES[bridge_count]}::{anchor_id}::" + "::".join(bridge_ids)


def make_route(target: dict, bridge_count: int, anchor: dict, bridge_records: list[dict], score: float) -> dict:
    bridge_ids = [r["merged_id"] for r in bridge_records]
    return {
        "route_id": hashlib.sha256(route_key(target, bridge_count, anchor["anchor_id"], bridge_ids).encode()).hexdigest()[:24],
        "route_type": ROUTE_TYPES[bridge_count],
        "bridge_count": bridge_count,
        "target_id": target["merged_id"],
        "target_split": target["split"],
        "target_source_dataset": target["source_dataset"],
        "target_image_path": target["image_path"],
        "target_mask_path_evaluation_only": target["mask_path"],
        "anchor_id": anchor["anchor_id"],
        "anchor_generation": anchor["generation"],
        "anchor_is_human": anchor["is_human"],
        "anchor_image_path": anchor["image_path"],
        "anchor_mask_path": anchor["mask_path"],
        "anchor_mask_sha256": anchor["mask_sha256"],
        "anchor_box_xywh_normalized": anchor["box_xywh_normalized"],
        "bridge_ids": bridge_ids,
        "bridge_image_paths": [r["image_path"] for r in bridge_records],
        "path_bottleneck_similarity": score,
        "path_mean_similarity": score,
        "target_gt_used_for_search_or_inference": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/protocol")
    parser.add_argument("--feature-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008")
    parser.add_argument("--round1-mode", default="sam3enc_anchor_conditioned_target_pooling")
    parser.add_argument("--bridge-counts", type=int, nargs="+", default=[5])
    parser.add_argument("--top-l", type=int, default=20)
    parser.add_argument("--output-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/msr_reroute_v1")
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS), default=sorted(EXPERIMENTS))
    args = parser.parse_args()

    records = read_jsonl(args.protocol_root / "merged_manifest.jsonl")
    support = read_jsonl(args.protocol_root / "support_manifest.jsonl")
    anchors = t21.human_pool(support, 512)
    record_by_id = {r["merged_id"]: r for r in records}
    train_records = [r for r in records if r["split"] == "train"]
    train_ids = [r["merged_id"] for r in train_records]
    train_pos = {i: pos for pos, i in enumerate(train_ids)}

    orig = np.load(args.feature_root / "features/sam3_base_s1008_features.npz")
    orig_desc = orig["patch_mean"]
    orig_by_id = {r["merged_id"]: orig_desc[pos] for pos, r in enumerate(records)}

    mask_data = np.load(args.feature_root / "features/sam3enc_mask_descriptors_s1008.npz")
    mask_by_id = {str(i): v for i, v in zip(mask_data["ids"], mask_data["descriptors"])}

    mask_manifest = {}
    for split in ("train", "validation", "test"):
        p = args.feature_root.parent / f"{split}_pseudo_masks_round1/train_pseudo_masks_round1.jsonl"
        if p.exists():
            for row in read_jsonl(p):
                mask_manifest[row["target_id"]] = row["pseudo_mask_path"]
    anchor_manifest = {}
    ap = args.feature_root.parent / "train_pseudo_masks_round1/anchor_mask_manifest.jsonl"
    for row in read_jsonl(ap):
        anchor_manifest[row["target_id"]] = row["pseudo_mask_path"]

    shape_by_id = {}
    for i in train_ids + [r["merged_id"] for r in records if r["split"] in ("validation", "test")]:
        mask_path = mask_manifest.get(i) or anchor_manifest.get(i)
        shape_by_id[i] = shape_vector(mask_path) if mask_path else np.zeros(12, dtype=np.float32)
    for a in anchors:
        mask_path = anchor_manifest.get(a["anchor_id"]) or a["mask_path"]
        shape_by_id[a["anchor_id"]] = shape_vector(mask_path)

    text_by_id = {}
    for path in [
        args.feature_root.parent / "train_pseudo_masks_round1/qwen35_mask_descriptions.jsonl",
        args.feature_root.parent / "validation_pseudo_masks_round1/qwen35_mask_descriptions.jsonl",
        args.feature_root.parent / "test_pseudo_masks_round1/qwen35_mask_descriptions.jsonl",
        args.feature_root.parent / "train_pseudo_masks_round1/qwen35_anchor_descriptions.jsonl",
    ]:
        text_by_id.update(load_text_map(path))

    orig_train = np.stack([orig_by_id[i] for i in train_ids]).astype(np.float32)
    mask_train = np.stack([mask_by_id[i] for i in train_ids]).astype(np.float32)
    shape_train = np.stack([shape_by_id[i] for i in train_ids]).astype(np.float32)
    text_train = np.stack([text_by_id.get(i, np.zeros(len(qf.FIELDS) * 2, dtype=np.float32)) for i in train_ids]).astype(np.float32)
    text_dim = max(len(v) for v in text_by_id.values()) if text_by_id else 0
    text_train = np.stack([text_by_id.get(i, np.zeros(text_dim, dtype=np.float32)) for i in train_ids]).astype(np.float32)

    round1_routes = read_jsonl(args.feature_root / args.round1_mode / "test_pool0_stage1/routes.jsonl")
    anchor_by_target_bridge = {
        (r["target_id"], int(r["bridge_count"])): r["anchor_id"]
        for r in round1_routes
    }
    anchor_by_id = {a["anchor_id"]: a for a in anchors}

    test_records = [r for r in records if r["split"] == "test"]
    for exp in args.experiments:
        weights = EXPERIMENTS[exp]
        for bridge_count in args.bridge_counts:
            rows = []
            for target in test_records:
                tid = target["merged_id"]
                target_orig = np.asarray(orig_by_id[tid], dtype=np.float32)
                target_mask = np.asarray(mask_by_id[tid], dtype=np.float32)
                target_shape = np.asarray(shape_by_id[tid], dtype=np.float32)
                target_text = np.asarray(text_by_id.get(tid, np.zeros(text_dim, dtype=np.float32)), dtype=np.float32)
                s_v = orig_train @ target_orig
                anchor_id = anchor_by_target_bridge.get((tid, bridge_count))
                order = np.argsort(-s_v)
                cand = []
                for pos in order:
                    i = train_ids[pos]
                    if i == anchor_id:
                        continue
                    cand.append(pos)
                    if len(cand) >= args.top_l:
                        break
                s_r = mask_train[cand] @ target_mask
                s_m = shape_train[cand] @ target_shape
                s_t = text_train[cand] @ target_text
                score = np.zeros(len(cand), dtype=np.float32)
                if "visual" in weights:
                    score += weights["visual"] * s_v[cand]
                if "roi" in weights:
                    score += weights["roi"] * s_r
                if "shape" in weights:
                    score += weights["shape"] * s_m
                if "text" in weights:
                    score += weights["text"] * s_t
                top = [cand[i] for i in np.argsort(-score)[:bridge_count]]
                bridge_records = [record_by_id[train_ids[i]] for i in top]
                anchor = anchor_by_id.get(anchor_id) or anchors[0]
                rows.append(make_route(target, bridge_count, anchor, bridge_records, float(score.max()) if len(score) else 0.0))
            out_dir = args.output_root / f"msr_{exp}" / f"test_pool0_stage1"
            write_jsonl(out_dir / "routes.jsonl", rows)
            print(f"{exp} b{bridge_count}: {len(rows)} routes -> {out_dir}")


if __name__ == "__main__":
    main()
