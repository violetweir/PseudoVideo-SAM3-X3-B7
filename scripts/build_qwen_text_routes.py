#!/usr/bin/env python3
"""Build a simple round-2 KNN route graph from Qwen3.5 text mask descriptions.

The graph uses the 8 fixed GT anchors and only the train bridge pool.  Candidate
transition score is cosine similarity between categorical text-feature vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
BUILDER = ROOT / "scripts/build_qwen_text_knn_features.py"
spec = importlib.util.spec_from_file_location("qwen_text_features", BUILDER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {BUILDER}")
qf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qf
spec.loader.exec_module(qf)

T21_PATH = ROOT / "scripts/run_t21_dynamic_pseudovideo.py"
spec = importlib.util.spec_from_file_location("t21_dynamic", T21_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {T21_PATH}")
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


def load_desc_map(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for row in read_jsonl(path):
        out[row["target_id"]] = qf.to_vector(row.get("qwen_features") or {})
    return out


def sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def route_key(target: dict, bridge_count: int, anchor_id: str, bridge_ids: list[str]) -> str:
    return (
        f"{target['merged_id']}::{ROUTE_TYPES[bridge_count]}::{anchor_id}::"
        + "::".join(bridge_ids)
    )


def make_route(
    target: dict,
    bridge_count: int,
    anchor: dict,
    bridge_records: list[dict],
    score: tuple[float, float],
) -> dict:
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


def path_scores(
    anchor_vec: np.ndarray,
    bridge_vecs: list[np.ndarray],
    target_vec: np.ndarray,
) -> tuple[float, float]:
    nodes = [anchor_vec, *bridge_vecs, target_vec]
    values = [sim(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    return min(values), float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/protocol")
    parser.add_argument("--output-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/qwen_text_knn_routes")
    parser.add_argument("--max-bridge", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=["validation", "test"])
    args = parser.parse_args()

    records = read_jsonl(args.protocol_root / "merged_manifest.jsonl")
    support = read_jsonl(args.protocol_root / "support_manifest.jsonl")
    anchors = t21.human_pool(support, 512)
    train_records = {r["merged_id"]: r for r in records if r["split"] == "train"}
    train_ids = list(train_records.keys())
    record_by_id = {r["merged_id"]: r for r in records}

    train_desc = load_desc_map(
        ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/qwen35_mask_descriptions.jsonl"
    )
    anchor_desc = load_desc_map(
        ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/qwen35_anchor_descriptions.jsonl"
    )
    target_desc = {}
    for split in args.splits:
        path = (
            ROOT / "work/kvasir_1pct_anchors/validation_pseudo_masks_round1/qwen35_mask_descriptions.jsonl"
            if split == "validation"
            else ROOT / "work/kvasir_1pct_anchors/test_pseudo_masks_round1/qwen35_mask_descriptions.jsonl"
        )
        target_desc.update(load_desc_map(path))

    train_matrix = np.stack([train_desc.get(i, np.zeros(len(next(iter(train_desc.values()))) if train_desc else 0)) for i in train_ids])
    train_norm = np.linalg.norm(train_matrix, axis=1, keepdims=True)
    train_norm[train_norm == 0] = 1
    train_matrix = train_matrix / train_norm

    for split in args.splits:
        targets = [r for r in records if r["split"] == split]
        rows = []
        for target in targets:
            target_vec = target_desc.get(target["merged_id"], np.zeros(train_matrix.shape[1]))
            target_vec = target_vec / (np.linalg.norm(target_vec) or 1)
            for bridge_count in range(0, args.max_bridge + 1):
                candidates = []
                for anchor in anchors:
                    anchor_vec = anchor_desc.get(anchor["anchor_id"], np.zeros(train_matrix.shape[1]))
                    anchor_vec = anchor_vec / (np.linalg.norm(anchor_vec) or 1)
                    beams: list[tuple[list[int], tuple[float, float]]] = [([], path_scores(anchor_vec, [], target_vec))]
                    for _ in range(bridge_count):
                        expanded = []
                        for path, _ in beams:
                            used = set(path)
                            tail_vec = train_matrix[path[0]] if path else target_vec
                            scores = train_matrix @ tail_vec
                            order = np.argsort(-scores)
                            ranked = [int(i) for i in order if int(i) not in used][: args.beam_width]
                            for node in ranked:
                                new_path = [node, *path]
                                bridge_vecs = [train_matrix[i] for i in new_path]
                                expanded.append((new_path, path_scores(anchor_vec, bridge_vecs, target_vec)))
                        beams = sorted(expanded, key=lambda item: item[1], reverse=True)[: args.beam_width]
                    for path, score in beams:
                        if len(path) == bridge_count:
                            candidates.append((score, anchor, path))
                _, anchor, path = max(candidates, key=lambda item: (item[0], item[1]["anchor_id"]))
                bridge_records = [record_by_id[train_ids[i]] for i in path]
                rows.append(make_route(target, bridge_count, anchor, bridge_records, path_scores(
                    anchor_desc.get(anchor["anchor_id"], np.zeros(train_matrix.shape[1])),
                    [train_matrix[i] for i in path],
                    target_vec,
                )))
        mode_dir = args.output_root / "qwen_text_knn" / f"{split}_pool0_stage1"
        write_jsonl(mode_dir / "routes.jsonl", rows)
        print(f"{split}: {len(rows)} routes -> {mode_dir}")


if __name__ == "__main__":
    main()
