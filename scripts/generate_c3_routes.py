#!/usr/bin/env python3
"""C3: multi-anchor independent-route generation for path-invariance experiments.
For each target in --split: rank anchors by lesion similarity (mode sam3enc_lesion),
take top --k-anchors, and for each anchor generate routes for bridge counts 0..--max-bridge
keeping the top --top-beams beams per depth (route diversity).
Writes {--output-root}/{mode_key}/{split}_pool0_stage1/routes.jsonl (same layout as
stage1_feature_knn_routes.py so stage1_eval_routes_forward_only.py can evaluate it).
Rules identical for validation and test (freeze protocol).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
RPC_PATH = ROOT / "scripts/stage1_feature_knn_routes.py"
spec = importlib.util.spec_from_file_location("rpc", RPC_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {RPC_PATH}")
rpc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rpc
spec.loader.exec_module(rpc)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def generate(args: argparse.Namespace) -> None:
    protocol = args.protocol_root
    records = read_jsonl(protocol / "merged_manifest.jsonl")
    support = read_jsonl(args.support_manifest or protocol / "support_manifest.jsonl")
    feature_root = args.output_root / "features"
    state = rpc.build_mode_state(
        args.mode,
        records,
        support,
        feature_root,
        args.feature_source,
        args.knn_feature,
        args.feature_size,
        args.text_blend,
        args.qwen_desc_root,
        args.mask_visual_fraction,
        args.lesion_desc_npz,
    )
    id_to_index = {row["merged_id"]: i for i, row in enumerate(records)}
    state["id_to_index"] = id_to_index
    train_indices = [i for i, row in enumerate(records) if row["split"] == "train"]
    anchors = rpc.t21.human_pool(support, 512)
    support_ids = {row["merged_id"] for row in support}
    targets = [
        row
        for row in records
        if row["split"] == args.split
        and not (args.exclude_support_targets and row["merged_id"] in support_ids)
    ]
    rank_cache = rpc.build_rank_cache(state, records, support, train_indices)

    mode_key = args.mode_key or f"{args.mode}_c3"
    phase = args.output_root / mode_key / f"{args.split}_pool0_stage1"
    routes_path = phase / "routes.jsonl"
    routes_path.parent.mkdir(parents=True, exist_ok=True)

    sim = state["sim"]
    routes: list[dict[str, Any]] = []
    n_skip_target = 0
    for target_no, target in enumerate(sorted(targets, key=lambda row: row["merged_id"]), start=1):
        if target_no == 1 or target_no % 10 == 0:
            print(f"{mode_key}: target {target_no}/{len(targets)}", flush=True)
        target_idx = id_to_index[target["merged_id"]]
        # rank anchors by lesion similarity to the target
        scored = []
        for anchor in anchors:
            a_idx = id_to_index[anchor["anchor_id"]]
            if a_idx == target_idx:
                continue
            scored.append((float(sim[a_idx, target_idx]), anchor))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: args.k_anchors]
        if len(top) < 1:
            n_skip_target += 1
            continue
        for rank, (anchor_sim, anchor) in enumerate(top, start=1):
            forbidden = {target_idx, id_to_index[anchor["anchor_id"]]}
            anchor_id = anchor["anchor_id"]
            for bridge_count in range(0, args.max_bridge + 1):
                beams: list[tuple[list[int], tuple[float, float]]] = [
                    ([], rpc.route_score(state, anchor_id, [], target_idx))
                ]
                for depth in range(bridge_count):
                    expanded = []
                    for path, _ in beams:
                        used = forbidden | set(path)
                        tail = target_idx if not path else path[0]
                        ranked = rpc.top_ranked_nodes(rank_cache, state, anchor_id, tail, used, args.beam_width)
                        for node in ranked:
                            new_path = [node, *path]
                            expanded.append((new_path, rpc.route_score(state, anchor_id, new_path, target_idx)))
                    beams = sorted(expanded, key=lambda item: item[1], reverse=True)[: args.beam_width]
                keep = sorted(beams, key=lambda item: item[1], reverse=True)[: args.top_beams]
                for path, score in keep:
                    if len(path) != bridge_count:
                        continue
                    route = rpc.make_route(target, bridge_count, anchor, path, score, records)
                    route["anchor_rank"] = rank
                    route["anchor_target_sim"] = round(anchor_sim, 6)
                    route["c3_top_beams"] = args.top_beams
                    routes.append(route)
    write_jsonl(routes_path, routes)
    meta = {
        "mode": args.mode,
        "mode_key": mode_key,
        "split": args.split,
        "k_anchors": args.k_anchors,
        "max_bridge": args.max_bridge,
        "top_beams": args.top_beams,
        "beam_width": args.beam_width,
        "feature_size": args.feature_size,
        "feature_source": args.feature_source,
        "routes": len(routes),
        "n_targets": len(targets),
        "n_skip_target": n_skip_target,
        "rule_note": "identical generation for validation and test (frozen)",
    }
    (args.output_root / mode_key / "meta.json").parent.mkdir(parents=True, exist_ok=True)
    (args.output_root / mode_key / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sam3enc_lesion")
    parser.add_argument("--mode-key", default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--k-anchors", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=2)
    parser.add_argument("--top-beams", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--feature-source", default="sam3_base")
    parser.add_argument("--feature-size", type=int, default=1008)
    parser.add_argument("--knn-feature", default="patch_mean")
    parser.add_argument("--text-blend", type=float, default=0.0)
    parser.add_argument("--mask-visual-fraction", type=float, default=0.0)
    parser.add_argument("--qwen-desc-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_lesion_knn")
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/protocol")
    parser.add_argument("--support-manifest", type=Path, default=None)
    parser.add_argument("--lesion-desc-npz", type=Path, default=rpc.MASK_DESC_NPZ)
    parser.add_argument("--exclude-support-targets", action="store_true")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
