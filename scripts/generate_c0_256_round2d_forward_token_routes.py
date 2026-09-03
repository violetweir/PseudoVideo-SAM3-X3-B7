#!/usr/bin/env python3
"""Generate Round-2D validation routes with a frozen Forward-Token anchor top-3.

Only anchor admission changes.  The SAM3-base@256 topology, per-mode route
score, KNN candidate ranking and beam width are the existing Round-2A rules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
PROTOCOL = ROOT / "work/kvasir_1pct_anchors/protocol"
ROUND2C = ROOT / "work/rerun_c0_256_round2c_lesion_anchor_validation"
FEATURE_ROOT = ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/features"
STAGE1_PATH = ROOT / "scripts/stage1_feature_knn_routes.py"
SPEC = importlib.util.spec_from_file_location("round2d_stage1", STAGE1_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {STAGE1_PATH}")
stage1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage1
SPEC.loader.exec_module(stage1)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def anchor_rankings(scores_path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(scores_path):
        if row["target_mask_source"] == "round1":
            grouped[row["target_id"]].append(row)
    rankings: dict[str, list[dict[str, Any]]] = {}
    for target_id, rows in grouped.items():
        ranked = sorted(
            rows,
            key=lambda row: (float(row["forward_auc_token_topk"]), row["anchor_id"]),
            reverse=True,
        )
        if len(ranked) != 8:
            raise RuntimeError(f"Expected eight anchors for {target_id}, received {len(ranked)}")
        rankings[target_id] = ranked
    return rankings


def make_routes(args: argparse.Namespace) -> None:
    if args.feature_size != 256:
        raise SystemExit("Round-2D is frozen to SAM3-base@256 features")
    records = read_jsonl(args.protocol_root / "merged_manifest.jsonl")
    support = read_jsonl(args.protocol_root / "support_manifest.jsonl")
    anchors = {row["anchor_id"]: row for row in stage1.t21.human_pool(support, 512)}
    rankings = anchor_rankings(args.scores)
    id_to_index = {row["merged_id"]: index for index, row in enumerate(records)}
    train_indices = [index for index, row in enumerate(records) if row["split"] == "train"]
    targets = sorted((row for row in records if row["split"] == "validation"), key=lambda row: row["merged_id"])
    if args.max_targets is not None:
        targets = targets[: args.max_targets]

    state = stage1.build_mode_state(
        args.mode,
        records,
        support,
        args.feature_root,
        "sam3_base",
        "patch_mean",
        256,
        0.0,
        ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1",
        0.0,
        stage1.MASK_DESC_NPZ,
    )
    state["id_to_index"] = id_to_index
    rank_cache = stage1.build_rank_cache(state, records, support, train_indices)
    routes: list[dict[str, Any]] = []
    for target_no, target in enumerate(targets, start=1):
        target_id = target["merged_id"]
        ranked_anchors = rankings.get(target_id)
        if ranked_anchors is None:
            raise RuntimeError(f"No frozen forward-token ranking for {target_id}")
        target_idx = id_to_index[target_id]
        for anchor_rank, score_row in enumerate(ranked_anchors[: args.k_anchors], start=1):
            anchor_id = score_row["anchor_id"]
            anchor = anchors[anchor_id]
            forbidden = {target_idx, id_to_index[anchor_id]}
            for bridge_count in range(args.min_bridge, args.max_bridge + 1):
                beams: list[tuple[list[int], tuple[float, float]]] = [
                    ([], stage1.route_score(state, anchor_id, [], target_idx))
                ]
                for _ in range(bridge_count):
                    expanded = []
                    for path, _ in beams:
                        used = forbidden | set(path)
                        tail = target_idx if not path else path[0]
                        for node in stage1.top_ranked_nodes(
                            rank_cache, state, anchor_id, tail, used, args.beam_width
                        ):
                            candidate_path = [node, *path]
                            expanded.append(
                                (
                                    candidate_path,
                                    stage1.route_score(state, anchor_id, candidate_path, target_idx),
                                )
                            )
                    beams = sorted(expanded, key=lambda item: item[1], reverse=True)[: args.beam_width]
                path, route_score = max(beams, key=lambda item: item[1])
                route = stage1.make_route(target, bridge_count, anchor, path, route_score, records)
                route.update(
                    {
                        "round2d_anchor_rank": anchor_rank,
                        "round2d_anchor_score": float(score_row["forward_auc_token_topk"]),
                        "round2d_anchor_score_source": "round1",
                        "round2d_anchor_rule": "forward_token_topk",
                        "round2d_feature_size": 256,
                        "round2d_beam_width": args.beam_width,
                        "round2d_topology": "sam3_base_s256_frozen",
                    }
                )
                routes.append(route)
        if target_no == 1 or target_no % 10 == 0 or target_no == len(targets):
            print(f"[round2d] {args.mode}: target {target_no}/{len(targets)}", flush=True)

    routes.sort(key=lambda row: (row["target_id"], row["bridge_count"], row["round2d_anchor_rank"]))
    mode_key = args.mode_key or f"round2d_forward_token_top3_{args.mode}"
    route_path = args.output_root / mode_key / "validation_pool0_stage1/routes.jsonl"
    write_jsonl(route_path, routes)
    metadata = {
        "mode": args.mode,
        "mode_key": mode_key,
        "split": "validation",
        "feature_size": 256,
        "k_anchors": args.k_anchors,
        "min_bridge": args.min_bridge,
        "max_bridge": args.max_bridge,
        "beam_width": args.beam_width,
        "targets": len(targets),
        "routes": len(routes),
        "anchor_rule": "forward_auc_token_topk__round1",
        "topology": "sam3_base@256 frozen",
        "target_gt_used_for_search_or_inference": False,
    }
    (args.output_root / mode_key / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(
        "sam3enc_anchor_conditioned_target_pooling",
        "sam3enc_anchor_conditioned_patch_correspondence",
    ), required=True)
    parser.add_argument("--mode-key")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, default=PROTOCOL)
    parser.add_argument("--scores", type=Path, default=ROUND2C / "anchor_scores_validation.jsonl")
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--feature-size", type=int, default=256)
    parser.add_argument("--k-anchors", type=int, default=3)
    parser.add_argument("--min-bridge", type=int, default=0)
    parser.add_argument("--max-bridge", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-targets", type=int)
    args = parser.parse_args()
    if args.k_anchors != 3 or args.min_bridge != 0 or args.max_bridge != 6 or args.beam_width != 32:
        raise SystemExit("Round-2D is frozen to Top-3, b0-b6 and beam width 32")
    make_routes(args)


if __name__ == "__main__":
    main()
