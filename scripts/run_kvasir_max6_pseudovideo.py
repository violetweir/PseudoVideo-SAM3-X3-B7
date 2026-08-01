#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_video_model


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
T21_PATH = ROOT / "scripts/run_t21_dynamic_pseudovideo.py"
spec = importlib.util.spec_from_file_location("t21_dynamic", T21_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {T21_PATH}")
t21 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t21
spec.loader.exec_module(t21)


ROUTE_TYPES = {0: "direct", **{i: f"bridge_{i}" for i in range(1, 7)}}


def edge_similarity(a: int, b: int, descriptors: np.ndarray) -> float:
    return float(descriptors[a] @ descriptors[b])


def best_route_for_bridge_count(
    target: dict[str, Any],
    bridge_count: int,
    anchors: list[dict[str, Any]],
    records: list[dict[str, Any]],
    descriptors: np.ndarray,
    neighbors: list[list[int]],
    id_to_index: dict[str, int],
    query_k: int,
    beam_width: int,
) -> dict[str, Any]:
    target_index = id_to_index[target["merged_id"]]
    target_id = target["merged_id"]
    train_indices = [
        index
        for index, row in enumerate(records)
        if row["split"] == "train" and row["merged_id"] != target_id
    ]
    ranked_train = sorted(
        train_indices,
        key=lambda index: (
            edge_similarity(target_index, index, descriptors),
            records[index]["merged_id"],
        ),
        reverse=True,
    )

    if bridge_count == 0:
        choices = []
        for anchor in anchors:
            anchor_index = id_to_index[anchor["anchor_id"]]
            sim = edge_similarity(anchor_index, target_index, descriptors)
            choices.append(((sim, sim), [], anchor))
        score, bridge_indices, anchor = max(
            choices, key=lambda item: (item[0], t21.anchor_sort_id(item[2]))
        )
    else:
        terminal_candidates = ranked_train[: max(query_k, beam_width)]
        partials: list[tuple[tuple[int, ...], tuple[float, ...]]] = [
            ((bridge,), (edge_similarity(bridge, target_index, descriptors),))
            for bridge in terminal_candidates
        ]
        for _ in range(bridge_count - 1):
            expanded: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
            for path, sims in partials:
                front = path[0]
                used = set(path)
                for candidate in neighbors[front]:
                    if candidate in used or records[candidate]["merged_id"] == target_id:
                        continue
                    sim = edge_similarity(candidate, front, descriptors)
                    expanded.append(((candidate, *path), (sim, *sims)))
            if not expanded:
                break
            partials = sorted(
                expanded,
                key=lambda item: (
                    min(item[1]),
                    float(np.mean(item[1])),
                    tuple(records[index]["merged_id"] for index in item[0]),
                ),
                reverse=True,
            )[:beam_width]
        if not partials or len(partials[0][0]) != bridge_count:
            raise RuntimeError(f"No {bridge_count}-bridge path for {target_id}")

        candidates = []
        for path, path_sims in partials:
            excluded = {target_id, *(records[index]["merged_id"] for index in path)}
            anchor_choice = t21.best_anchor_for(
                path[0], anchors, id_to_index, descriptors, neighbors, excluded
            )
            if anchor_choice is None:
                continue
            anchor, anchor_sim = anchor_choice
            sims = (anchor_sim, *path_sims)
            candidates.append(
                ((min(sims), float(np.mean(sims))), list(path), anchor, True)
            )
        if not candidates:
            for path, path_sims in partials:
                excluded = {target_id, *(records[index]["merged_id"] for index in path)}
                relaxed_choices = [
                    (
                        edge_similarity(id_to_index[anchor["anchor_id"]], path[0], descriptors),
                        anchor["anchor_id"],
                        anchor,
                    )
                    for anchor in anchors
                    if anchor["anchor_id"] not in excluded
                ]
                if not relaxed_choices:
                    continue
                anchor_sim, _, anchor = max(relaxed_choices)
                sims = (anchor_sim, *path_sims)
                candidates.append(
                    ((min(sims), float(np.mean(sims))), list(path), anchor, False)
                )
        if not candidates:
            raise RuntimeError(f"No {bridge_count}-bridge path for {target_id}")
        score, bridge_indices, anchor, anchor_knn_edge = max(
            candidates,
            key=lambda item: (
                item[0],
                t21.anchor_sort_id(item[2]),
                tuple(records[index]["merged_id"] for index in item[1]),
            ),
        )
    if bridge_count == 0:
        anchor_knn_edge = False

    route_key = (
        f"{target_id}::{ROUTE_TYPES[bridge_count]}::{anchor['anchor_id']}::"
        f"{anchor['mask_sha256']}::"
        + "::".join(records[index]["merged_id"] for index in bridge_indices)
    )
    query_neighbor_rank = (
        ranked_train.index(bridge_indices[-1]) + 1 if bridge_indices else None
    )
    return {
        "route_id": hashlib.sha256(route_key.encode()).hexdigest()[:24],
        "route_type": ROUTE_TYPES[bridge_count],
        "bridge_count": bridge_count,
        "target_id": target_id,
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
        "bridge_ids": [records[index]["merged_id"] for index in bridge_indices],
        "bridge_image_paths": [records[index]["image_path"] for index in bridge_indices],
        "path_bottleneck_similarity": score[0],
        "path_mean_similarity": score[1],
        "query_attachment_neighbor_rank": query_neighbor_rank,
        "query_attachment_within_knn_k": (
            query_neighbor_rank is None or query_neighbor_rank <= query_k
        ),
        "anchor_knn_edge": anchor_knn_edge,
        "target_gt_used_for_search_or_inference": False,
    }


def freeze_routes(
    phase_root: Path,
    targets: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    records: list[dict[str, Any]],
    descriptors: np.ndarray,
    neighbors: list[list[int]],
    id_to_index: dict[str, int],
    query_k: int,
    beam_width: int,
) -> list[dict[str, Any]]:
    path = phase_root / "routes.jsonl"
    meta_path = phase_root / "routes_meta.json"
    meta = {
        "route_version": "kvasir_max6_bridge_beam_v1",
        "bridge_counts": list(range(7)),
        "query_k": query_k,
        "beam_width": beam_width,
        "target_ids": sorted(row["merged_id"] for row in targets),
        "anchor_ids": sorted(row["anchor_id"] for row in anchors),
    }
    signature = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    if path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("signature") != signature:
            raise RuntimeError("Frozen max6 routes differ; use a new output root")
        return t21.read_jsonl(path)
    routes = []
    for target in sorted(targets, key=lambda row: row["merged_id"]):
        base_routes = t21.make_three_routes(
            target,
            anchors,
            records,
            descriptors,
            neighbors,
            id_to_index,
            query_k,
        )
        for row in base_routes:
            row["route_type"] = ROUTE_TYPES[int(row["bridge_count"])]
            routes.append(row)
        for bridge_count in range(3, 7):
            routes.append(
                best_route_for_bridge_count(
                    target,
                    bridge_count,
                    anchors,
                    records,
                    descriptors,
                    neighbors,
                    id_to_index,
                    query_k,
                    beam_width,
                )
            )
    t21.write_jsonl_atomic(path, routes)
    meta_path.write_text(json.dumps({**meta, "signature": signature}, indent=2) + "\n")
    return routes


def load_mask(path: str, canvas: int) -> np.ndarray:
    return t21.load_mask(path, canvas)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return t21.dice(a, b)


def select_targets(
    phase_root: Path,
    targets: list[dict[str, Any]],
    route_results: list[dict[str, Any]],
    alpha: float,
    canvas: int,
    max_bridge: int,
) -> list[dict[str, Any]]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in route_results:
        if int(row["bridge_count"]) <= max_bridge:
            by_target.setdefault(row["target_id"], []).append(row)
    selected_root = phase_root / f"selected_masks_max{max_bridge}"
    selected_root.mkdir(parents=True, exist_ok=True)
    selections = []
    for target in sorted(targets, key=lambda row: row["merged_id"]):
        candidates = sorted(by_target[target["merged_id"]], key=lambda row: row["bridge_count"])
        masks = [load_mask(row["forward_mask_path"], canvas) for row in candidates]
        scored = []
        for index, row in enumerate(candidates):
            q_multi = float(
                np.mean([dice(masks[index], masks[j]) for j in range(len(masks)) if j != index])
            )
            quality = alpha * float(row["q_return"]) + (1 - alpha) * q_multi
            scored.append((quality, q_multi, row, masks[index]))
        quality, q_multi, row, selected_mask = max(
            scored,
            key=lambda item: (item[0], item[1], item[2]["q_return"], item[2]["route_id"]),
        )
        gt = load_mask(target["mask_path"], canvas)
        metrics = t21.metrics(selected_mask, gt)
        safe_id = target["merged_id"].replace("::", "__")
        selected_path = selected_root / f"{safe_id}.png"
        Image.fromarray(selected_mask.astype(np.uint8) * 255).save(selected_path)
        selections.append(
            {
                "target_id": target["merged_id"],
                "target_split": target["split"],
                "target_source_dataset": target["source_dataset"],
                "selected_route_id": row["route_id"],
                "selected_route_type": row["route_type"],
                "selected_bridge_count": row["bridge_count"],
                "q_return": row["q_return"],
                "q_multi": q_multi,
                "quality_score": quality,
                "alpha": alpha,
                "max_bridge": max_bridge,
                "selected_mask_path": str(selected_path),
                "target_gt_used_for_quality_or_selection": False,
                **{f"gt_{key}_evaluation_only": value for key, value in metrics.items()},
            }
        )
    t21.write_jsonl_atomic(phase_root / f"selections_max{max_bridge}.jsonl", selections)
    return selections


def write_summary(output_root: Path) -> None:
    summaries = {}
    for max_bridge in (5, 6):
        rows = t21.read_jsonl(output_root / "test_pool0_max6" / f"selections_max{max_bridge}.jsonl")
        dice_values = np.asarray([row["gt_dice_evaluation_only"] for row in rows], dtype=np.float64)
        counts = {
            ROUTE_TYPES[i]: sum(1 for row in rows if row["selected_bridge_count"] == i)
            for i in range(max_bridge + 1)
        }
        summaries[f"max_bridge_{max_bridge}"] = {
            "n": len(rows),
            "dice_mean": float(dice_values.mean()),
            "dice_std": float(dice_values.std()),
            "route_type_counts": counts,
        }
    (output_root / "summary_max5_max6.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beam-width", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = t21.read_jsonl(args.manifest)
    support = t21.read_jsonl(args.support_manifest)
    records, descriptors, neighbors = t21.build_graph(manifest, args.output_root, args.knn_k)
    id_to_index = {row["merged_id"]: i for i, row in enumerate(records)}
    targets = [row for row in records if row["split"] == "test"]
    anchors = t21.human_pool(support, args.canvas_size)
    phase_root = args.output_root / "test_pool0_max6"
    phase_root.mkdir(parents=True, exist_ok=True)
    routes = freeze_routes(
        phase_root,
        targets,
        anchors,
        records,
        descriptors,
        neighbors,
        id_to_index,
        args.knn_k,
        args.beam_width,
    )
    model = build_sam3_video_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        device="cuda",
        compile=False,
    )
    model.eval()
    results = t21.evaluate_routes(model, phase_root, routes, args.canvas_size, args.resume)
    expected = len(targets) * 7
    if len(results) != expected:
        raise RuntimeError(f"Expected {expected} route results, got {len(results)}")
    select_targets(phase_root, targets, results, args.alpha, args.canvas_size, 5)
    select_targets(phase_root, targets, results, args.alpha, args.canvas_size, 6)
    (phase_root / "PHASE_COMPLETE").touch()
    write_summary(args.output_root)


if __name__ == "__main__":
    main()
