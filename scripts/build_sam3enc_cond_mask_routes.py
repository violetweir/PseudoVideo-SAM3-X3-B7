#!/usr/bin/env python3
"""Build mask-aware routes from the SAM3-encoder target-pooling cond baseline.

Only candidate retrieval changes.  The original anchor-conditioned target
pooling condition score is rank-fused with the pseudo-mask foreground
descriptor similarity between a candidate bridge and the current route tail.
The original route score and frozen SAM3 propagation remain unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
STAGE1_PATH = ROOT / "scripts/stage1_feature_knn_routes.py"
spec = importlib.util.spec_from_file_location("stage1_feature_knn", STAGE1_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {STAGE1_PATH}")
stage1 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stage1
spec.loader.exec_module(stage1)


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic ascending ranks in [0, 1]."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    if len(values) > 1:
        ranks /= float(len(values) - 1)
    return ranks


def load_mask_descriptors(path: Path, records: list[dict[str, Any]]) -> np.ndarray:
    data = np.load(path)
    by_id = {
        str(record_id): np.asarray(descriptor, dtype=np.float32)
        for record_id, descriptor in zip(data["ids"].tolist(), data["descriptors"])
    }
    missing = [row["merged_id"] for row in records if row["merged_id"] not in by_id]
    if missing:
        raise SystemExit(f"Missing {len(missing)} mask descriptors; first={missing[0]}")
    rows = np.stack([by_id[row["merged_id"]] for row in records]).astype(np.float32)
    return stage1.l2norm(rows)


def build_rank_cache(
    state: dict[str, Any],
    records: list[dict[str, Any]],
    support: list[dict[str, Any]],
    train_indices: list[int],
    mask_desc: np.ndarray,
    mask_weight: float,
) -> dict[tuple[str, int], list[int]]:
    cache: dict[tuple[str, int], list[int]] = {}
    ids = np.asarray([records[i]["merged_id"] for i in train_indices])
    train_mask = mask_desc[train_indices]

    for anchor in stage1.t21.human_pool(support, 512):
        anchor_id = anchor["anchor_id"]
        cond = np.asarray(
            [stage1.conditioned_score(state, anchor_id, i) for i in train_indices],
            dtype=np.float32,
        )
        if mask_weight == 0.0:
            order = np.lexsort((ids, cond))[::-1]
            ranked = [train_indices[int(pos)] for pos in order]
            for tail in range(len(records)):
                cache[(anchor_id, tail)] = ranked
            continue

        cond_rank = percentile_ranks(cond)
        for tail in range(len(records)):
            mask_similarity = train_mask @ mask_desc[tail]
            mask_rank = percentile_ranks(mask_similarity)
            fused = (1.0 - mask_weight) * cond_rank + mask_weight * mask_rank
            order = np.lexsort((ids, fused))[::-1]
            cache[(anchor_id, tail)] = [train_indices[int(pos)] for pos in order]
    return cache


def freeze_routes(
    mode_key: str,
    output_root: Path,
    records: list[dict[str, Any]],
    support: list[dict[str, Any]],
    state: dict[str, Any],
    rank_cache: dict[tuple[str, int], list[int]],
    split: str,
    min_bridge: int,
    max_bridge: int,
    beam_width: int,
) -> list[dict[str, Any]]:
    phase = output_root / mode_key / f"{split}_pool0_stage1"
    routes_path = phase / "routes.jsonl"
    if routes_path.exists():
        return stage1.read_jsonl(routes_path)
    phase.mkdir(parents=True, exist_ok=True)

    id_to_index = {row["merged_id"]: i for i, row in enumerate(records)}
    state["id_to_index"] = id_to_index
    anchors = stage1.t21.human_pool(support, 512)
    support_ids = {row["merged_id"] for row in support}
    targets = [
        row
        for row in records
        if row["split"] == split and row["merged_id"] not in support_ids
    ]
    routes: list[dict[str, Any]] = []
    for target_no, target in enumerate(sorted(targets, key=lambda row: row["merged_id"]), 1):
        if target_no == 1 or target_no % 10 == 0:
            print(f"{mode_key}: target {target_no}/{len(targets)}", flush=True)
        target_idx = id_to_index[target["merged_id"]]
        for bridge_count in range(min_bridge, max_bridge + 1):
            candidates = []
            for anchor in anchors:
                anchor_id = anchor["anchor_id"]
                forbidden = {target_idx, id_to_index[anchor_id]}
                beams: list[tuple[list[int], tuple[float, float]]] = [
                    ([], stage1.route_score(state, anchor_id, [], target_idx))
                ]
                for _ in range(bridge_count):
                    expanded = []
                    for path, _ in beams:
                        used = forbidden | set(path)
                        tail = target_idx if not path else path[0]
                        ranked = stage1.top_ranked_nodes(
                            rank_cache, state, anchor_id, tail, used, beam_width
                        )
                        for node in ranked:
                            new_path = [node, *path]
                            score = stage1.route_score(state, anchor_id, new_path, target_idx)
                            expanded.append((new_path, score))
                    beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:beam_width]
                for path, score in beams:
                    candidates.append((score, anchor_id, anchor, path))
            score, _, anchor, path = max(candidates, key=lambda item: (item[0], item[1]))
            routes.append(
                stage1.make_route(target, bridge_count, anchor, path, score, records)
            )

    stage1.write_jsonl(routes_path, routes)
    return routes


def weight_key(weight: float) -> str:
    return f"w{int(round(weight * 100)):03d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-weight", type=float, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--min-bridge", type=int, default=2)
    parser.add_argument("--max-bridge", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/protocol",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features",
    )
    parser.add_argument(
        "--mask-descriptors",
        type=Path,
        default=ROOT
        / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/sam3enc_mask_descriptors_s1008.npz",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/sam3enc_cond_mask_knn_v1",
    )
    args = parser.parse_args()
    if not 0.0 <= args.mask_weight <= 1.0:
        raise SystemExit("--mask-weight must be in [0, 1]")

    records = stage1.read_jsonl(args.protocol_root / "merged_manifest.jsonl")
    support = stage1.read_jsonl(args.protocol_root / "support_manifest.jsonl")
    state = stage1.build_mode_state(
        "sam3enc_anchor_conditioned_target_pooling",
        records,
        support,
        args.feature_root,
        "sam3_base",
        "cond",
        1008,
    )
    state["id_to_index"] = {row["merged_id"]: i for i, row in enumerate(records)}
    mask_desc = load_mask_descriptors(args.mask_descriptors, records)
    train_indices = [i for i, row in enumerate(records) if row["split"] == "train"]
    rank_cache = build_rank_cache(
        state, records, support, train_indices, mask_desc, args.mask_weight
    )
    mode_key = f"sam3enc_target_pooling_cond_maskrank_{weight_key(args.mask_weight)}"
    routes = freeze_routes(
        mode_key,
        args.output_root,
        records,
        support,
        state,
        rank_cache,
        args.split,
        args.min_bridge,
        args.max_bridge,
        args.beam_width,
    )
    meta = {
        "checkpoint": "base_sam3",
        "feature_size": 1008,
        "route_mode": "sam3enc_anchor_conditioned_target_pooling",
        "baseline_knn_feature": "cond",
        "retrieval_formula": "(1-w)*rank(cond_anchor_candidate)+w*rank(mask_candidate_tail_cosine)",
        "route_score_changed": False,
        "mask_weight": args.mask_weight,
        "mask_descriptors": str(args.mask_descriptors),
        "split": args.split,
        "min_bridge": args.min_bridge,
        "max_bridge": args.max_bridge,
        "beam_width": args.beam_width,
        "routes": len(routes),
    }
    meta_path = args.output_root / mode_key / f"meta_{args.split}.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mode": mode_key, "routes": len(routes), "split": args.split}, indent=2))


if __name__ == "__main__":
    main()
