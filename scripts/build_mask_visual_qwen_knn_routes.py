#!/usr/bin/env python3
"""Round-2 hybrid KNN: mask-conditioned SAM3 encoder visual descriptors + Qwen text."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_text_map(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for row in read_jsonl(path):
        out[row["target_id"]] = qf.to_vector(row.get("qwen_features") or {})
    return out


def load_mask_desc_map(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {str(k): v for k, v in zip(data["ids"], data["descriptors"])}


def route_key(target: dict, bridge_count: int, anchor_id: str, bridge_ids: list[str]) -> str:
    return f"{target['merged_id']}::{ROUTE_TYPES[bridge_count]}::{anchor_id}::" + "::".join(bridge_ids)


def make_route(target: dict, bridge_count: int, anchor: dict, bridge_records: list[dict], score: tuple[float, float]) -> dict:
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
        "path_bottleneck_similarity": score[0],
        "path_mean_similarity": score[1],
        "target_gt_used_for_search_or_inference": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/protocol")
    parser.add_argument("--mask-desc", type=Path, default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/sam3enc_mask_descriptors_s1008.npz")
    parser.add_argument("--text-blend", type=float, default=0.2)
    parser.add_argument("--output-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/mask_visual_qwen_hybrid_lambda02")
    parser.add_argument("--max-bridge", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=["validation", "test"])
    args = parser.parse_args()

    records = read_jsonl(args.protocol_root / "merged_manifest.jsonl")
    support = read_jsonl(args.protocol_root / "support_manifest.jsonl")
    anchors = t21.human_pool(support, 512)
    record_by_id = {r["merged_id"]: r for r in records}
    train_records = [r for r in records if r["split"] == "train"]
    train_ids = [r["merged_id"] for r in train_records]
    mask_desc = load_mask_desc_map(args.mask_desc)

    text_map: dict[str, np.ndarray] = {}
    for path in [
        ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/qwen35_mask_descriptions.jsonl",
        ROOT / "work/kvasir_1pct_anchors/validation_pseudo_masks_round1/qwen35_mask_descriptions.jsonl",
        ROOT / "work/kvasir_1pct_anchors/test_pseudo_masks_round1/qwen35_mask_descriptions.jsonl",
        ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/qwen35_anchor_descriptions.jsonl",
    ]:
        text_map.update(load_text_map(path))

    def vec(id_: str) -> np.ndarray:
        return mask_desc.get(id_, np.zeros(next(iter(mask_desc.values())).shape, dtype=np.float32))

    def tvec(id_: str) -> np.ndarray:
        return text_map.get(id_, np.zeros(len(next(iter(text_map.values()))), dtype=np.float32))

    train_visual = np.stack([vec(i) for i in train_ids]).astype(np.float32)
    train_text = np.stack([tvec(i) for i in train_ids]).astype(np.float32)

    def combined(a_v: np.ndarray, a_t: np.ndarray, b_v: np.ndarray, b_t: np.ndarray) -> float:
        v = float(a_v @ b_v)
        t = float(a_t @ b_t)
        return (1.0 - args.text_blend) * v + args.text_blend * t

    for split in args.splits:
        targets = [r for r in records if r["split"] == split]
        rows = []
        for target in targets:
            tv = vec(target["merged_id"])
            tt = tvec(target["merged_id"])
            for bridge_count in range(0, args.max_bridge + 1):
                candidates = []
                for anchor in anchors:
                    av = vec(anchor["anchor_id"])
                    at = tvec(anchor["anchor_id"])
                    beams = [([], (combined(av, at, tv, tt), combined(av, at, tv, tt)))]
                    for _ in range(bridge_count):
                        expanded = []
                        for path, _ in beams:
                            used = set(path)
                            tail_idx = path[0] if path else -1
                            if tail_idx >= 0:
                                tail_v = train_visual[tail_idx]
                                tail_t = train_text[tail_idx]
                            else:
                                tail_v = tv
                                tail_t = tt
                            scores = np.asarray([combined(train_visual[i], train_text[i], tail_v, tail_t) for i in range(len(train_ids))], dtype=np.float32)
                            order = np.argsort(-scores)
                            ranked = [int(i) for i in order if int(i) not in used][: args.beam_width]
                            for node in ranked:
                                new_path = [node, *path]
                                bridge_v = [train_visual[i] for i in new_path]
                                bridge_t = [train_text[i] for i in new_path]
                                node_vs = [av, *bridge_v, tv]
                                node_ts = [at, *bridge_t, tt]
                                vals = [combined(node_vs[i], node_ts[i], node_vs[i + 1], node_ts[i + 1]) for i in range(len(node_vs) - 1)]
                                expanded.append((new_path, (min(vals), float(np.mean(vals)))))
                        beams = sorted(expanded, key=lambda item: item[1], reverse=True)[: args.beam_width]
                    for path, score in beams:
                        if len(path) == bridge_count:
                            candidates.append((score, anchor, path))
                _, anchor, path = max(candidates, key=lambda item: (item[0], item[1]["anchor_id"]))
                bridge_records = [record_by_id[train_ids[i]] for i in path]
                av = vec(anchor["anchor_id"]); at = tvec(anchor["anchor_id"])
                bridge_v = [train_visual[i] for i in path]; bridge_t = [train_text[i] for i in path]
                node_vs = [av, *bridge_v, tv]; node_ts = [at, *bridge_t, tt]
                vals = [combined(node_vs[i], node_ts[i], node_vs[i + 1], node_ts[i + 1]) for i in range(len(node_vs) - 1)]
                rows.append(make_route(target, bridge_count, anchor, bridge_records, (min(vals), float(np.mean(vals)))))
        out_dir = args.output_root / "qwen_text_knn" / f"{split}_pool0_stage1"
        write_jsonl(out_dir / "routes.jsonl", rows)
        print(f"{split}: {len(rows)} routes -> {out_dir}")


if __name__ == "__main__":
    main()
