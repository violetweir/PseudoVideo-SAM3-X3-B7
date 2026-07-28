#!/usr/bin/env python3
"""Run the leakage-safe Dynamic Pseudo-Video Propagation experiment.

The pipeline builds a train-only visual graph, evaluates three predeclared route
lengths per target (direct, one bridge, two bridges), scores them without target
GT using backward anchor return and multi-route agreement, grows two generations
of train-only pseudo anchors, freezes the pool, and evaluates each test image
independently against train-only knowledge.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_video_model
try:
    from sam3_memory_adapter import load_memory_adapter
except ImportError:
    load_memory_adapter = None


PHASES = ("round0_train", "round1_train", "test_pool0", "test_pool1", "test_pool2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t17-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--support-manifest", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tracker-adapter",
        type=Path,
        default=None,
        help=(
            "Optional T22 tracker-only checkpoint. Its state_dict is loaded into "
            "model.tracker after the base SAM3 checkpoint."
        ),
    )
    parser.add_argument(
        "--memory-adapter",
        type=Path,
        default=None,
        help="Optional T23 residual memory-read adapter checkpoint.",
    )
    parser.add_argument("--phase", choices=(*PHASES, "all"), default="all")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_fsync(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptor(image_path: str) -> np.ndarray:
    """The exact T18 199-D descriptor, including its post-resize aspect term."""
    rgb = np.asarray(
        Image.open(image_path).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]
    ).reshape(-1)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    low_frequency = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1)
    moments = np.concatenate(
        [
            rgb.reshape(-1, 3).mean(axis=0),
            rgb.reshape(-1, 3).std(axis=0),
            np.asarray([rgb.shape[1] / rgb.shape[0]], dtype=np.float64),
        ]
    )
    vector = np.concatenate(
        [
            histogram.astype(np.float64),
            low_frequency.astype(np.float64) / 255.0,
            moments.astype(np.float64) / np.asarray([255] * 6 + [1]),
        ]
    )
    return vector / max(np.linalg.norm(vector), 1e-12)


def tight_box(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    height, width = mask.shape[-2:]
    x1, x2 = xs.min() / width, (xs.max() + 1) / width
    y1, y2 = ys.min() / height, (ys.max() + 1) / height
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def load_rgb(path: str, canvas: int) -> Image.Image:
    return (
        Image.open(path)
        .convert("RGB")
        .resize((canvas, canvas), Image.Resampling.BILINEAR)
    )


def load_mask(path: str, canvas: int) -> np.ndarray:
    return (
        np.asarray(
            Image.open(path)
            .convert("L")
            .resize((canvas, canvas), Image.Resampling.NEAREST)
        )
        > 127
    )


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a, b = mask_a.astype(bool), mask_b.astype(bool)
    return float(2 * np.logical_and(a, b).sum() / max(a.sum() + b.sum(), 1))


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred, gt = prediction.astype(bool), target.astype(bool)
    intersection = int(np.logical_and(pred, gt).sum())
    pred_area, gt_area = int(pred.sum()), int(gt.sum())
    union = pred_area + gt_area - intersection
    return {
        "dice": 2 * intersection / max(pred_area + gt_area, 1),
        "iou": intersection / max(union, 1),
        "precision": intersection / max(pred_area, 1),
        "recall": intersection / max(gt_area, 1),
        "pred_area_ratio": pred_area / pred.size,
        "gt_area_ratio": gt_area / gt.size,
    }


def select_top(output: dict[str, Any], canvas: int) -> dict[str, Any]:
    masks = np.asarray(output["out_binary_masks"], dtype=bool)
    scores = np.asarray(output["out_probs"], dtype=np.float64)
    if len(masks):
        index = int(np.argmax(scores))
        return {
            "mask": masks[index],
            "candidate_count": int(len(masks)),
            "sam_score": float(scores[index]),
        }
    return {
        "mask": np.zeros((canvas, canvas), dtype=bool),
        "candidate_count": 0,
        "sam_score": 0.0,
    }


def propagate(
    model: Any,
    image_paths: list[str],
    prompt_box: list[float],
    canvas: int,
) -> dict[str, Any]:
    frames = [load_rgb(path, canvas) for path in image_paths]
    state = model.init_state(
        resource_path=frames,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    model.add_prompt(
        state,
        frame_idx=0,
        text_str=None,
        boxes_xywh=[prompt_box],
        box_labels=[1],
    )
    final_output = None
    for frame_idx, output in model.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=len(frames),
        reverse=False,
    ):
        if frame_idx == len(frames) - 1:
            final_output = output
    if final_output is None:
        raise RuntimeError(f"No output for final frame of {image_paths}")
    selected = select_top(final_output, canvas)
    del state
    torch.cuda.empty_cache()
    return selected


def propagate_return_from_predicted_mask(
    model: Any,
    forward_paths: list[str],
    predicted_query_mask: np.ndarray,
    canvas: int,
) -> dict[str, Any]:
    """Return query -> anchor in a fresh state initialized by the predicted mask."""
    predicted_box = tight_box(predicted_query_mask)
    if predicted_box is None:
        return {
            "mask": np.zeros((canvas, canvas), dtype=bool),
            "candidate_count": 0,
            "sam_score": 0.0,
            "success": False,
            "failure_reason": "empty_forward_mask",
        }
    frames = [load_rgb(path, canvas) for path in forward_paths]
    query_frame_idx = len(frames) - 1
    state = model.init_state(
        resource_path=frames,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    _, prompted = model.add_prompt(
        state,
        frame_idx=query_frame_idx,
        text_str=None,
        boxes_xywh=[predicted_box],
        box_labels=[1],
    )
    prompt_ids = np.asarray(prompted["out_obj_ids"], dtype=np.int64)
    prompt_scores = np.asarray(prompted["out_probs"], dtype=np.float64)
    if len(prompt_ids) == 0:
        del state
        torch.cuda.empty_cache()
        return {
            "mask": np.zeros((canvas, canvas), dtype=bool),
            "candidate_count": 0,
            "sam_score": 0.0,
            "success": False,
            "failure_reason": "query_box_created_no_object",
        }
    selected_index = int(np.argmax(prompt_scores))
    object_id = int(prompt_ids[selected_index])
    for other_id in prompt_ids.tolist():
        if int(other_id) != object_id:
            model.remove_object(
                state,
                int(other_id),
                is_user_action=False,
            )
    found = 0
    mask_tensor = torch.from_numpy(predicted_query_mask.astype(np.float32))
    for tracker_state in state["tracker_inference_states"]:
        tracker_ids = [int(value) for value in tracker_state["obj_ids"]]
        if object_id in tracker_ids:
            model.tracker.add_new_mask(
                inference_state=tracker_state,
                frame_idx=query_frame_idx,
                obj_id=object_id,
                mask=mask_tensor,
                add_mask_to_memory=True,
            )
            found += 1
    if found != 1:
        del state
        torch.cuda.empty_cache()
        return {
            "mask": np.zeros((canvas, canvas), dtype=bool),
            "candidate_count": 0,
            "sam_score": 0.0,
            "success": False,
            "failure_reason": f"object_tracker_state_matches={found}",
        }
    for tracker_state in state["tracker_inference_states"]:
        if len(tracker_state["obj_ids"]):
            model.tracker.propagate_in_video_preflight(
                tracker_state,
                run_mem_encoder=True,
            )
    anchor_mask = None
    anchor_return_score = 0.0
    for frame_idx, output in model.propagate_in_video(
        state,
        start_frame_idx=query_frame_idx,
        max_frame_num_to_track=query_frame_idx + 1,
        reverse=True,
    ):
        if frame_idx != 0:
            continue
        output_ids = np.asarray(output["out_obj_ids"], dtype=np.int64)
        matches = np.flatnonzero(output_ids == object_id)
        if len(matches) == 1:
            masks = np.asarray(output["out_binary_masks"], dtype=bool)
            anchor_mask = masks[int(matches[0])]
            output_scores = np.asarray(output["out_probs"], dtype=np.float64)
            anchor_return_score = float(output_scores[int(matches[0])])
    del state
    torch.cuda.empty_cache()
    if anchor_mask is None:
        return {
            "mask": np.zeros((canvas, canvas), dtype=bool),
            "candidate_count": 0,
            "sam_score": 0.0,
            "success": False,
            "failure_reason": "object_missing_at_return_anchor",
        }
    return {
        "mask": anchor_mask,
        "candidate_count": 1,
        "sam_score": anchor_return_score,
        "success": True,
        "failure_reason": None,
    }


def build_graph(
    records: list[dict[str, Any]], output_root: Path, knn_k: int
) -> tuple[list[dict[str, Any]], np.ndarray, list[list[int]]]:
    protocol = output_root / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    graph_path = protocol / "train_graph.pkl"
    features_path = protocol / "all_descriptors.npz"
    records_path = protocol / "all_records.jsonl"
    graph_meta_path = protocol / "graph_meta.json"
    ordered = sorted(records, key=lambda row: row["merged_id"])
    record_signature = hashlib.sha256(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in ordered
        ).encode("utf-8")
    ).hexdigest()
    expected_meta = {
        "graph_version": "t21_symmetric_knn_v2",
        "descriptor_version": "exact_t18_199d_v1",
        "record_signature_sha256": record_signature,
        "record_count": len(ordered),
        "knn_k": knn_k,
    }
    if graph_path.exists() and features_path.exists() and records_path.exists():
        if not graph_meta_path.exists():
            raise RuntimeError("Cached graph has no graph_meta.json")
        cached_meta = json.loads(graph_meta_path.read_text(encoding="utf-8"))
        if cached_meta != expected_meta:
            raise RuntimeError(
                f"Cached graph protocol mismatch: {cached_meta} != {expected_meta}"
            )
        stored_records = read_jsonl(records_path)
        payload = np.load(features_path)
        with graph_path.open("rb") as handle:
            graph = pickle.load(handle)
        if payload["descriptors"].shape != (len(ordered), 199):
            raise RuntimeError("Cached descriptor shape mismatch")
        if [row["merged_id"] for row in stored_records] != [
            row["merged_id"] for row in ordered
        ]:
            raise RuntimeError("Cached record order mismatch")
        return stored_records, payload["descriptors"], graph["neighbors"]

    descriptors = np.stack([descriptor(row["image_path"]) for row in ordered])
    train_indices = [i for i, row in enumerate(ordered) if row["split"] == "train"]
    train_features = descriptors[train_indices]
    similarity = train_features @ train_features.T
    directed_neighbors: list[list[int]] = [[] for _ in ordered]
    for local_index, global_index in enumerate(train_indices):
        order = np.argsort(similarity[local_index])[::-1]
        directed_neighbors[global_index] = [
            train_indices[j] for j in order if j != local_index
        ][:knn_k]
    neighbor_sets = [set(values) for values in directed_neighbors]
    for source in train_indices:
        for destination in directed_neighbors[source]:
            neighbor_sets[destination].add(source)
    neighbors = [
        sorted(
            values,
            key=lambda index: (
                float(descriptors[node] @ descriptors[index]),
                ordered[index]["merged_id"],
            ),
            reverse=True,
        )
        if values
        else []
        for node, values in enumerate(neighbor_sets)
    ]
    write_jsonl_atomic(records_path, ordered)
    np.savez_compressed(features_path, descriptors=descriptors)
    with graph_path.open("wb") as handle:
        pickle.dump(
            {
                "knn_k": knn_k,
                "train_indices": train_indices,
                "neighbors": neighbors,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    graph_meta_path.write_text(
        json.dumps(expected_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ordered, descriptors, neighbors


def human_pool(
    support: list[dict[str, Any]], canvas: int
) -> list[dict[str, Any]]:
    pool = []
    for row in support:
        mask = load_mask(row["frozen_mask_path"], canvas)
        box = tight_box(mask)
        if box is None:
            raise RuntimeError(f"Empty human anchor: {row['merged_id']}")
        pool.append(
            {
                "anchor_id": row["merged_id"],
                "generation": 0,
                "is_human": True,
                "image_path": row["frozen_image_path"],
                "mask_path": row["frozen_mask_path"],
                "mask_sha256": sha256_file(Path(row["frozen_mask_path"])),
                "box_xywh_normalized": box,
                "quality_score": 1.0,
            }
        )
    return sorted(pool, key=lambda row: row["anchor_id"])


def best_anchor_for(
    node_index: int,
    anchors: list[dict[str, Any]],
    id_to_index: dict[str, int],
    descriptors: np.ndarray,
    neighbors: list[list[int]],
    excluded: set[str],
) -> tuple[dict[str, Any], float] | None:
    adjacent = set(neighbors[node_index])
    choices = [
        (
            float(descriptors[node_index] @ descriptors[id_to_index[a["anchor_id"]]]),
            a["anchor_id"],
            a,
        )
        for a in anchors
        if a["anchor_id"] not in excluded
        and id_to_index[a["anchor_id"]] in adjacent
    ]
    if not choices:
        return None
    similarity, _, anchor = max(choices, key=lambda item: (item[0], item[1]))
    return anchor, similarity


def make_three_routes(
    target: dict[str, Any],
    anchors: list[dict[str, Any]],
    records: list[dict[str, Any]],
    descriptors: np.ndarray,
    neighbors: list[list[int]],
    id_to_index: dict[str, int],
    query_k: int,
) -> list[dict[str, Any]]:
    target_index = id_to_index[target["merged_id"]]
    target_id = target["merged_id"]
    train_indices = [
        index
        for index, row in enumerate(records)
        if row["split"] == "train" and row["merged_id"] != target_id
    ]
    ranked_train_neighbors = sorted(
        train_indices,
        key=lambda index: (
            float(descriptors[target_index] @ descriptors[index]),
            records[index]["merged_id"],
        ),
        reverse=True,
    )
    query_neighbors = ranked_train_neighbors[:query_k]

    candidates: dict[int, list[tuple[tuple[float, float], list[int], dict[str, Any]]]] = {
        0: [],
        1: [],
        2: [],
    }
    for anchor in anchors:
        if anchor["anchor_id"] == target_id:
            continue
        anchor_index = id_to_index[anchor["anchor_id"]]
        sim = float(descriptors[anchor_index] @ descriptors[target_index])
        candidates[0].append(((sim, sim), [], anchor))

    for bridge in query_neighbors:
        excluded = {target_id, records[bridge]["merged_id"]}
        anchor_choice = best_anchor_for(
            bridge, anchors, id_to_index, descriptors, neighbors, excluded
        )
        if anchor_choice is None:
            continue
        anchor, first_sim = anchor_choice
        last_sim = float(descriptors[bridge] @ descriptors[target_index])
        candidates[1].append(
            ((min(first_sim, last_sim), (first_sim + last_sim) / 2), [bridge], anchor)
        )

    for bridge2 in query_neighbors:
        for bridge1 in neighbors[bridge2]:
            ids = {
                target_id,
                records[bridge1]["merged_id"],
                records[bridge2]["merged_id"],
            }
            if len(ids) < 3:
                continue
            anchor_choice = best_anchor_for(
                bridge1, anchors, id_to_index, descriptors, neighbors, ids
            )
            if anchor_choice is None:
                continue
            anchor, sim1 = anchor_choice
            sim2 = float(descriptors[bridge1] @ descriptors[bridge2])
            sim3 = float(descriptors[bridge2] @ descriptors[target_index])
            candidates[2].append(
                (
                    (min(sim1, sim2, sim3), (sim1 + sim2 + sim3) / 3),
                    [bridge1, bridge2],
                    anchor,
                )
            )

    # Some nodes are more than two graph hops from every current anchor.  Keep
    # the anchor-side edge inside the frozen kNN graph, but expand only the
    # temporary query attachment until the requested route length is feasible.
    if not candidates[1]:
        for bridge in ranked_train_neighbors[query_k:]:
            excluded = {target_id, records[bridge]["merged_id"]}
            anchor_choice = best_anchor_for(
                bridge, anchors, id_to_index, descriptors, neighbors, excluded
            )
            if anchor_choice is None:
                continue
            anchor, first_sim = anchor_choice
            last_sim = float(descriptors[bridge] @ descriptors[target_index])
            candidates[1].append(
                (
                    (min(first_sim, last_sim), (first_sim + last_sim) / 2),
                    [bridge],
                    anchor,
                )
            )
            if len(candidates[1]) >= 16:
                break
    if not candidates[2]:
        for bridge2 in ranked_train_neighbors[query_k:]:
            for bridge1 in neighbors[bridge2]:
                ids = {
                    target_id,
                    records[bridge1]["merged_id"],
                    records[bridge2]["merged_id"],
                }
                if len(ids) < 3:
                    continue
                anchor_choice = best_anchor_for(
                    bridge1, anchors, id_to_index, descriptors, neighbors, ids
                )
                if anchor_choice is None:
                    continue
                anchor, sim1 = anchor_choice
                sim2 = float(descriptors[bridge1] @ descriptors[bridge2])
                sim3 = float(descriptors[bridge2] @ descriptors[target_index])
                candidates[2].append(
                    (
                        (min(sim1, sim2, sim3), (sim1 + sim2 + sim3) / 3),
                        [bridge1, bridge2],
                        anchor,
                    )
                )
                if len(candidates[2]) >= 16:
                    break
            if len(candidates[2]) >= 16:
                break

    names = {0: "direct", 1: "one_bridge", 2: "two_bridges"}
    ranked: dict[int, list[tuple[tuple[float, float], list[int], dict[str, Any]]]] = {}
    for bridge_count in range(3):
        if not candidates[bridge_count]:
            raise RuntimeError(
                f"No {bridge_count}-bridge route for {target['merged_id']}"
            )
        ranked[bridge_count] = sorted(
            candidates[bridge_count],
            key=lambda item: (
                item[0],
                anchor_sort_id(item[2]),
                tuple(records[index]["merged_id"] for index in item[1]),
            ),
            reverse=True,
        )[:16]

    combinations = []
    for choice in itertools.product(ranked[0], ranked[1], ranked[2]):
        anchor_ids = [item[2]["anchor_id"] for item in choice]
        bridge_sets = [set(item[1]) for item in choice]
        distinct_anchors = len(set(anchor_ids)) == 3
        disjoint_bridges = all(
            bridge_sets[i].isdisjoint(bridge_sets[j])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        key = (
            int(distinct_anchors),
            int(disjoint_bridges),
            sum(item[0][0] for item in choice),
            sum(item[0][1] for item in choice),
            tuple(anchor_ids),
        )
        combinations.append((key, choice))
    _, selected_combination = max(combinations, key=lambda item: item[0])

    routes = []
    for bridge_count, (score, bridge_indices, anchor) in enumerate(
        selected_combination
    ):
        route_key = (
            f"{target_id}::{names[bridge_count]}::{anchor['anchor_id']}::"
            + anchor["mask_sha256"]
            + "::"
            + "::".join(records[index]["merged_id"] for index in bridge_indices)
        )
        selected_anchor_ids = [item[2]["anchor_id"] for item in selected_combination]
        selected_bridge_sets = [set(item[1]) for item in selected_combination]
        anchors_distinct_count = len(set(selected_anchor_ids))
        bridges_pairwise_disjoint = all(
            selected_bridge_sets[i].isdisjoint(selected_bridge_sets[j])
            for i in range(3)
            for j in range(i + 1, 3)
        )
        query_neighbor_rank = (
            ranked_train_neighbors.index(bridge_indices[-1]) + 1
            if bridge_indices
            else None
        )
        routes.append(
            {
                "route_id": hashlib.sha256(route_key.encode()).hexdigest()[:24],
                "route_type": names[bridge_count],
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
                "bridge_ids": [
                    records[index]["merged_id"] for index in bridge_indices
                ],
                "bridge_image_paths": [
                    records[index]["image_path"] for index in bridge_indices
                ],
                "path_bottleneck_similarity": score[0],
                "path_mean_similarity": score[1],
                "query_attachment_neighbor_rank": query_neighbor_rank,
                "query_attachment_within_knn_k": (
                    query_neighbor_rank is None or query_neighbor_rank <= query_k
                ),
                "target_gt_used_for_search_or_inference": False,
                "route_set_anchor_distinct_count": anchors_distinct_count,
                "route_set_bridges_pairwise_disjoint": bridges_pairwise_disjoint,
                "route_set_diversity_fallback": bool(
                    anchors_distinct_count < 3 or not bridges_pairwise_disjoint
                ),
            }
        )
    return routes


def anchor_sort_id(anchor: dict[str, Any]) -> str:
    return anchor["anchor_id"]


def freeze_routes(
    phase_root: Path,
    targets: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    records: list[dict[str, Any]],
    descriptors: np.ndarray,
    neighbors: list[list[int]],
    id_to_index: dict[str, int],
    query_k: int,
) -> list[dict[str, Any]]:
    path = phase_root / "routes.jsonl"
    meta_path = phase_root / "routes_meta.json"
    input_meta = {
        "route_version": "three_lengths_joint_diversity_v3",
        "query_k": query_k,
        "target_ids": sorted(row["merged_id"] for row in targets),
        "anchors": sorted(
            [
                {
                    "anchor_id": row["anchor_id"],
                    "generation": row["generation"],
                    "mask_sha256": row["mask_sha256"],
                    "box": row["box_xywh_normalized"],
                }
                for row in anchors
            ],
            key=lambda row: row["anchor_id"],
        ),
    }
    input_sha = hashlib.sha256(
        json.dumps(input_meta, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    if path.exists():
        if not meta_path.exists():
            raise RuntimeError(f"Frozen routes have no metadata: {path}")
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta.get("phase_input_sha256") != input_sha:
            raise RuntimeError("Frozen route inputs differ; use a new output root")
        return read_jsonl(path)
    routes = []
    for target in sorted(targets, key=lambda row: row["merged_id"]):
        routes.extend(
            make_three_routes(
                target,
                anchors,
                records,
                descriptors,
                neighbors,
                id_to_index,
                query_k,
            )
        )
    write_jsonl_atomic(path, routes)
    meta_path.write_text(
        json.dumps(
            {
                "phase_input_sha256": input_sha,
                "target_count": len(targets),
                "anchor_count": len(anchors),
                "query_k": query_k,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return routes


def evaluate_routes(
    model: Any,
    phase_root: Path,
    routes: list[dict[str, Any]],
    canvas: int,
    resume: bool,
) -> list[dict[str, Any]]:
    result_path = phase_root / "route_results.jsonl"
    old_rows = read_jsonl(result_path) if resume else []
    existing = {
        row["route_id"]: row
        for row in old_rows
        if row.get("status") == "success"
        and Path(row.get("forward_mask_path", "")).exists()
    }
    pending = [row for row in routes if row["route_id"] not in existing]
    forward_root = phase_root / "forward_masks"
    backward_root = phase_root / "backward_anchor_masks"
    forward_root.mkdir(parents=True, exist_ok=True)
    backward_root.mkdir(parents=True, exist_ok=True)
    for position, route in enumerate(pending, 1):
        started = time.time()
        forward_paths = [
            route["anchor_image_path"],
            *route["bridge_image_paths"],
            route["target_image_path"],
        ]
        forward = propagate(
            model,
            forward_paths,
            route["anchor_box_xywh_normalized"],
            canvas,
        )
        forward_path = forward_root / f"{route['route_id']}.png"
        Image.fromarray(forward["mask"].astype(np.uint8) * 255).save(forward_path)
        anchor_mask = load_mask(route["anchor_mask_path"], canvas)
        backward = propagate_return_from_predicted_mask(
            model,
            forward_paths,
            forward["mask"],
            canvas,
        )
        backward_mask = backward["mask"]
        backward_count = backward["candidate_count"]
        backward_score = backward["sam_score"]
        return_dice = dice(backward_mask, anchor_mask)
        backward_path = backward_root / f"{route['route_id']}.png"
        Image.fromarray(backward_mask.astype(np.uint8) * 255).save(backward_path)
        result = {
            **route,
            "status": "success",
            "forward_candidate_count": forward["candidate_count"],
            "forward_sam_score": forward["sam_score"],
            "forward_mask_path": str(forward_path),
            "forward_mask_sha256": sha256_file(forward_path),
            "backward_prompt_source": (
                "fresh state at query: tight box creates tracker object, then the "
                "forward predicted query mask replaces the box mask and is encoded"
            ),
            "backward_candidate_count": backward_count,
            "backward_sam_score": backward_score,
            "backward_success": backward["success"],
            "backward_failure_reason": backward["failure_reason"],
            "backward_anchor_mask_path": str(backward_path),
            "q_return": return_dice,
            "seconds": round(time.time() - started, 3),
        }
        append_fsync(result_path, result)
        existing[route["route_id"]] = result
        print(
            f"[{position}/{len(pending)}] {route['target_id']} "
            f"{route['route_type']} return={return_dice:.4f}",
            flush=True,
        )
    final_rows = [existing[row["route_id"]] for row in routes]
    write_jsonl_atomic(result_path, final_rows)
    return final_rows


def select_targets(
    phase_root: Path,
    targets: list[dict[str, Any]],
    route_results: list[dict[str, Any]],
    alpha: float,
    tau: float,
    canvas: int,
    generation: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in route_results:
        by_target.setdefault(row["target_id"], []).append(row)
    selections = []
    new_anchors = []
    selected_mask_root = phase_root / "selected_masks"
    selected_mask_root.mkdir(parents=True, exist_ok=True)
    for target in sorted(targets, key=lambda row: row["merged_id"]):
        routes = sorted(by_target[target["merged_id"]], key=lambda row: row["route_type"])
        masks = [load_mask(row["forward_mask_path"], canvas) for row in routes]
        scored = []
        for index, route in enumerate(routes):
            other = [dice(masks[index], masks[j]) for j in range(len(masks)) if j != index]
            q_multi = float(np.mean(other)) if other else 0.0
            quality = alpha * float(route["q_return"]) + (1 - alpha) * q_multi
            scored.append((quality, q_multi, route, masks[index]))
        quality, q_multi, selected, selected_mask = max(
            scored,
            key=lambda item: (
                item[0],
                item[1],
                item[2]["q_return"],
                item[2]["route_id"],
            ),
        )
        gt = load_mask(target["mask_path"], canvas)
        gt_metrics = metrics(selected_mask, gt)
        safe_id = target["merged_id"].replace("::", "__")
        selected_path = selected_mask_root / f"{safe_id}.png"
        Image.fromarray(selected_mask.astype(np.uint8) * 255).save(selected_path)
        accepted = bool(
            generation is not None and quality >= tau and selected_mask.any()
        )
        selection = {
            "target_id": target["merged_id"],
            "target_split": target["split"],
            "target_source_dataset": target["source_dataset"],
            "selected_route_id": selected["route_id"],
            "selected_route_type": selected["route_type"],
            "selected_bridge_count": selected["bridge_count"],
            "q_return": selected["q_return"],
            "q_multi": q_multi,
            "quality_score": quality,
            "alpha": alpha,
            "tau": tau,
            "accepted_as_pseudo_anchor": accepted,
            "selected_mask_path": str(selected_path),
            "selected_mask_sha256": sha256_file(selected_path),
            "target_gt_used_for_quality_or_selection": False,
            **{f"gt_{key}_evaluation_only": value for key, value in gt_metrics.items()},
        }
        selections.append(selection)
        if accepted:
            box = tight_box(selected_mask)
            new_anchors.append(
                {
                    "anchor_id": target["merged_id"],
                    "generation": generation,
                    "is_human": False,
                    "image_path": target["image_path"],
                    "mask_path": str(selected_path),
                    "mask_sha256": sha256_file(selected_path),
                    "box_xywh_normalized": box,
                    "quality_score": quality,
                    "source_route_id": selected["route_id"],
                }
            )
    write_jsonl_atomic(phase_root / "selections.jsonl", selections)
    if generation is not None:
        write_jsonl_atomic(phase_root / "new_anchors.jsonl", new_anchors)
    return selections, new_anchors


def run_phase(
    phase: str,
    model: Any,
    output_root: Path,
    targets: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    records: list[dict[str, Any]],
    descriptors: np.ndarray,
    neighbors: list[list[int]],
    id_to_index: dict[str, int],
    args: argparse.Namespace,
    generation: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    phase_root = output_root / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    if args.limit:
        targets = sorted(targets, key=lambda row: row["merged_id"])[: args.limit]
    write_jsonl_atomic(phase_root / "anchor_pool_input.jsonl", anchors)
    routes = freeze_routes(
        phase_root,
        targets,
        anchors,
        records,
        descriptors,
        neighbors,
        id_to_index,
        args.knn_k,
    )
    results = evaluate_routes(model, phase_root, routes, args.canvas_size, args.resume)
    selections, new_anchors = select_targets(
        phase_root,
        targets,
        results,
        args.alpha,
        args.tau,
        args.canvas_size,
        generation,
    )
    (phase_root / "PHASE_COMPLETE").touch()
    return selections, new_anchors


def summarize_all(output_root: Path) -> None:
    summaries = {}
    for phase in PHASES:
        rows = read_jsonl(output_root / phase / "selections.jsonl")
        if not rows:
            continue
        dice_values = np.asarray(
            [row["gt_dice_evaluation_only"] for row in rows], dtype=np.float64
        )
        quality = np.asarray([row["quality_score"] for row in rows], dtype=np.float64)
        pearson = (
            float(np.corrcoef(quality, dice_values)[0, 1])
            if len(rows) > 1 and quality.std() > 0 and dice_values.std() > 0
            else None
        )
        def average_ranks(values: np.ndarray) -> np.ndarray:
            order = np.argsort(values, kind="mergesort")
            ranks = np.empty(len(values), dtype=np.float64)
            start = 0
            while start < len(values):
                end = start + 1
                while end < len(values) and values[order[end]] == values[order[start]]:
                    end += 1
                ranks[order[start:end]] = (start + end - 1) / 2
                start = end
            return ranks

        ranks_q = average_ranks(quality)
        ranks_d = average_ranks(dice_values)
        spearman = (
            float(np.corrcoef(ranks_q, ranks_d)[0, 1]) if len(rows) > 1 else None
        )
        summaries[phase] = {
            "n": len(rows),
            "dice_mean": float(dice_values.mean()),
            "dice_std": float(dice_values.std()),
            "quality_mean": float(quality.mean()),
            "quality_gt_dice_pearson": pearson,
            "quality_gt_dice_spearman": spearman,
            "accepted_count": int(
                sum(row["accepted_as_pseudo_anchor"] for row in rows)
            ),
            "route_type_counts": {
                route_type: sum(
                    row["selected_route_type"] == route_type for row in rows
                )
                for route_type in ("direct", "one_bridge", "two_bridges")
            },
            "by_dataset": {
                dataset: {
                    "n": len(
                        [
                            row
                            for row in rows
                            if row["target_source_dataset"] == dataset
                        ]
                    ),
                    "dice_mean": float(
                        np.mean(
                            [
                                row["gt_dice_evaluation_only"]
                                for row in rows
                                if row["target_source_dataset"] == dataset
                            ]
                        )
                    ),
                }
                for dataset in ("CVC-ClinicDB", "kvasir-seg")
                if any(row["target_source_dataset"] == dataset for row in rows)
            },
        }
    if all(phase in summaries for phase in ("test_pool0", "test_pool1", "test_pool2")):
        summaries["q3_pool_growth"] = {
            "human16": summaries["test_pool0"]["dice_mean"],
            "human16_plus_gen1": summaries["test_pool1"]["dice_mean"],
            "human16_plus_gen1_plus_gen2": summaries["test_pool2"]["dice_mean"],
        }
    (output_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.manifest and args.support_manifest:
        manifest_path = args.manifest.resolve()
        support_path = args.support_manifest.resolve()
    elif args.t17_root:
        t17_root = args.t17_root.resolve()
        manifest_path = t17_root / "protocol/merged_manifest.jsonl"
        support_path = t17_root / "protocol/support_manifest.jsonl"
    else:
        raise ValueError(
            "Pass --manifest and --support-manifest, or the legacy --t17-root."
        )
    manifest = read_jsonl(manifest_path)
    support = read_jsonl(support_path)
    manifest_ids = [row["merged_id"] for row in manifest]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise RuntimeError("merged_manifest contains duplicate merged_id values")
    if len(support) != 16:
        raise RuntimeError(f"Expected exactly 16 support images, found {len(support)}")
    support_ids_list = [row["merged_id"] for row in support]
    if len(support_ids_list) != len(set(support_ids_list)):
        raise RuntimeError("support_manifest contains duplicate merged_id values")
    if any(row["split"] != "train" for row in support):
        raise RuntimeError("Every human support anchor must come from train")
    support_dataset_counts = {
        dataset: sum(row["source_dataset"] == dataset for row in support)
        for dataset in ("CVC-ClinicDB", "kvasir-seg")
    }
    if support_dataset_counts != {"CVC-ClinicDB": 8, "kvasir-seg": 8}:
        raise RuntimeError(f"Unexpected support composition: {support_dataset_counts}")
    if not set(support_ids_list).issubset(set(manifest_ids)):
        raise RuntimeError("support_manifest contains IDs absent from merged_manifest")
    for row in support:
        if not Path(row["frozen_image_path"]).exists() or not Path(
            row["frozen_mask_path"]
        ).exists():
            raise RuntimeError(f"Missing frozen support artifact: {row['merged_id']}")
    records, descriptors, neighbors = build_graph(manifest, output_root, args.knn_k)
    id_to_index = {row["merged_id"]: i for i, row in enumerate(records)}
    record_by_id = {row["merged_id"]: row for row in records}
    support_ids = {row["merged_id"] for row in support}
    train_non_support = [
        row for row in records if row["split"] == "train" and row["merged_id"] not in support_ids
    ]
    test_targets = [row for row in records if row["split"] == "test"]
    pool0 = human_pool(support, args.canvas_size)

    protocol = {
        "name": "T21 Dynamic Pseudo-Video Propagation",
        "manifest_sha256": sha256_file(manifest_path),
        "support_manifest_sha256": sha256_file(support_path),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "tracker_adapter": (
            str(args.tracker_adapter.resolve()) if args.tracker_adapter else None
        ),
        "tracker_adapter_sha256": (
            sha256_file(args.tracker_adapter.resolve())
            if args.tracker_adapter
            else None
        ),
        "memory_adapter": (
            str(args.memory_adapter.resolve()) if args.memory_adapter else None
        ),
        "memory_adapter_sha256": (
            sha256_file(args.memory_adapter.resolve())
            if args.memory_adapter
            else None
        ),
        "descriptor": "exact T18 199-D descriptor, L2 normalized",
        "graph": f"train-only symmetric union of directed kNN, k={args.knn_k}",
        "top3_routes": [
            "best direct route",
            "best one-bridge route",
            "best two-bridge route",
        ],
        "route_diversity": (
            "jointly prefer three distinct anchors and pairwise-disjoint bridge "
            "sets before maximizing path similarity"
        ),
        "query_attachment_fallback": (
            "query first uses its train Top-k neighbors; if an exact route length "
            "cannot reach any current anchor, expand only the temporary query edge "
            "and record its full train-neighbor rank"
        ),
        "path_score": "maximize (minimum edge cosine, mean edge cosine)",
        "quality": f"{args.alpha} * return_dice + {1-args.alpha} * route_agreement",
        "backward_prompt": "tight box of forward predicted query mask",
        "pseudo_anchor_threshold": args.tau,
        "pool_rounds": 2,
        "canvas_size": args.canvas_size,
        "test_policy": "each test query independently uses frozen train-only graph/pool; no test-to-test edges or updates",
        "target_gt_used_for_search_quality_selection_or_prompting": False,
        "python": platform.python_version(),
        "executable": sys.executable,
    }
    protocol_path = output_root / "protocol/protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old != protocol:
            raise RuntimeError("Existing protocol differs; use a new output root")
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    requested = PHASES if args.phase == "all" else (args.phase,)
    model = build_sam3_video_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        device="cuda",
        compile=False,
    )
    if args.tracker_adapter:
        payload = torch.load(
            args.tracker_adapter.resolve(), map_location="cpu", weights_only=False
        )
        state_dict = payload.get("tracker_state_dict", payload)
        missing, unexpected = model.tracker.load_state_dict(state_dict, strict=False)
        trainable_names = set(payload.get("trainable_parameter_names", ()))
        relevant_missing = [name for name in missing if name in trainable_names]
        if relevant_missing or unexpected:
            raise RuntimeError(
                "Invalid tracker adapter: "
                f"missing_trainable={relevant_missing}, unexpected={unexpected}"
            )
        print(
            f"Loaded T22 tracker adapter {args.tracker_adapter} "
            f"({len(state_dict)} tensors)"
        )
    if args.memory_adapter:
        if load_memory_adapter is None:
            raise ImportError(
                "sam3_memory_adapter is required only when --memory-adapter is used"
            )
        payload = torch.load(
            args.memory_adapter.resolve(), map_location="cpu", weights_only=False
        )
        adapter = load_memory_adapter(model.tracker, payload)
        adapter.to("cuda").eval()
        print(f"Loaded T23 memory adapter {args.memory_adapter}")
    model.eval()

    if "round0_train" in requested:
        _, gen1 = run_phase(
            "round0_train",
            model,
            output_root,
            train_non_support,
            pool0,
            records,
            descriptors,
            neighbors,
            id_to_index,
            args,
            generation=1,
        )
    else:
        if any(
            phase in requested
            for phase in ("round1_train", "test_pool1", "test_pool2")
        ) and not (output_root / "round0_train/PHASE_COMPLETE").exists():
            raise RuntimeError("round0_train is not complete")
        gen1 = read_jsonl(output_root / "round0_train/new_anchors.jsonl")
    pool1 = sorted(pool0 + gen1, key=lambda row: row["anchor_id"])

    gen1_ids = {row["anchor_id"] for row in gen1}
    round1_targets = [
        row for row in train_non_support if row["merged_id"] not in gen1_ids
    ]
    if "round1_train" in requested:
        _, gen2 = run_phase(
            "round1_train",
            model,
            output_root,
            round1_targets,
            pool1,
            records,
            descriptors,
            neighbors,
            id_to_index,
            args,
            generation=2,
        )
    else:
        if "test_pool2" in requested and not (
            output_root / "round1_train/PHASE_COMPLETE"
        ).exists():
            raise RuntimeError("round1_train is not complete")
        gen2 = read_jsonl(output_root / "round1_train/new_anchors.jsonl")
    pool2 = sorted(pool1 + gen2, key=lambda row: row["anchor_id"])

    phase_pools = {
        "test_pool0": pool0,
        "test_pool1": pool1,
        "test_pool2": pool2,
    }
    for phase in ("test_pool0", "test_pool1", "test_pool2"):
        if phase in requested:
            run_phase(
                phase,
                model,
                output_root,
                test_targets,
                phase_pools[phase],
                records,
                descriptors,
                neighbors,
                id_to_index,
                args,
                generation=None,
            )
    summarize_all(output_root)
    if args.phase == "all" and args.limit == 0:
        (output_root / "ALL_COMPLETE").touch()


if __name__ == "__main__":
    main()
