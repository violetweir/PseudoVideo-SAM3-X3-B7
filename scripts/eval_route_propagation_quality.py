#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
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


def route_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (row["target_id"], int(row["bridge_count"]), row["route_id"])


def select_top(output: dict[str, Any], canvas: int) -> dict[str, Any]:
    return t21.select_top(output, canvas)


def mask_stats(mask: np.ndarray) -> dict[str, float]:
    mask = mask.astype(bool)
    height, width = mask.shape
    area = float(mask.sum()) / max(mask.size, 1)
    if not mask.any():
        return {
            "area_ratio": 0.0,
            "empty": 1.0,
            "components": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "bbox_w": 0.0,
            "bbox_h": 0.0,
        }
    ys, xs = np.where(mask)
    labels, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return {
        "area_ratio": area,
        "empty": 0.0,
        "components": float(max(labels - 1, 0)),
        "centroid_x": float(xs.mean() / width),
        "centroid_y": float(ys.mean() / height),
        "bbox_w": float((xs.max() - xs.min() + 1) / width),
        "bbox_h": float((ys.max() - ys.min() + 1) / height),
    }


def rel_delta(a: float, b: float, eps: float = 1e-6) -> float:
    return abs(b - a) / max(a, eps)


def trace_features(masks: list[np.ndarray], scores: list[float], candidate_counts: list[int]) -> dict[str, Any]:
    stats = [mask_stats(mask) for mask in masks]
    areas = [item["area_ratio"] for item in stats]
    components = [item["components"] for item in stats]
    empties = [item["empty"] for item in stats]
    adjacent_dice = [t21.dice(masks[i], masks[i + 1]) for i in range(len(masks) - 1)]
    area_deltas = [rel_delta(areas[i], areas[i + 1]) for i in range(len(areas) - 1)]
    centroid_steps = [
        float(
            np.hypot(
                stats[i + 1]["centroid_x"] - stats[i]["centroid_x"],
                stats[i + 1]["centroid_y"] - stats[i]["centroid_y"],
            )
        )
        for i in range(len(stats) - 1)
    ]
    bbox_w_deltas = [rel_delta(stats[i]["bbox_w"], stats[i + 1]["bbox_w"]) for i in range(len(stats) - 1)]
    bbox_h_deltas = [rel_delta(stats[i]["bbox_h"], stats[i + 1]["bbox_h"]) for i in range(len(stats) - 1)]
    return {
        "trace_frame_count": len(masks),
        "trace_area_min": min(areas),
        "trace_area_max": max(areas),
        "trace_area_final": areas[-1],
        "trace_area_max_rel_delta": max(area_deltas, default=0.0),
        "trace_empty_count": int(sum(empties)),
        "trace_component_max": max(components),
        "trace_component_final": components[-1],
        "trace_centroid_max_step": max(centroid_steps, default=0.0),
        "trace_bbox_w_max_rel_delta": max(bbox_w_deltas, default=0.0),
        "trace_bbox_h_max_rel_delta": max(bbox_h_deltas, default=0.0),
        "trace_adjacent_dice_min": min(adjacent_dice, default=1.0),
        "trace_adjacent_dice_mean": float(np.mean(adjacent_dice)) if adjacent_dice else 1.0,
        "trace_adjacent_dice_last": adjacent_dice[-1] if adjacent_dice else 1.0,
        "trace_sam_score_min": min(scores),
        "trace_sam_score_mean": float(np.mean(scores)),
        "trace_sam_score_final": scores[-1],
        "trace_candidate_count_max": max(candidate_counts),
        "trace_candidate_count_final": candidate_counts[-1],
    }


def propagate_with_trace(
    model: Any,
    image_paths: list[str],
    prompt_box: list[float],
    canvas: int,
) -> dict[str, Any]:
    frames = [t21.load_rgb(path, canvas) for path in image_paths]
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
    frame_outputs: dict[int, dict[str, Any]] = {}
    for frame_idx, output in model.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=len(frames),
        reverse=False,
    ):
        frame_outputs[int(frame_idx)] = select_top(output, canvas)
    del state
    torch.cuda.empty_cache()
    if len(frame_outputs) != len(frames):
        raise RuntimeError(f"Expected {len(frames)} trace frames, got {len(frame_outputs)}")
    masks = [frame_outputs[idx]["mask"] for idx in range(len(frames))]
    scores = [float(frame_outputs[idx]["sam_score"]) for idx in range(len(frames))]
    candidate_counts = [int(frame_outputs[idx]["candidate_count"]) for idx in range(len(frames))]
    return {
        "final_mask": masks[-1],
        "final_candidate_count": candidate_counts[-1],
        "final_sam_score": scores[-1],
        **trace_features(masks, scores, candidate_counts),
    }


def evaluate(
    model: Any,
    routes: list[dict[str, Any]],
    output_root: Path,
    canvas: int,
    resume: bool,
) -> list[dict[str, Any]]:
    result_path = output_root / "propagation_quality.jsonl"
    old_rows = read_jsonl(result_path) if resume else []
    existing = {
        row["route_id"]: row
        for row in old_rows
        if row.get("status") == "success"
    }
    pending = [row for row in routes if row["route_id"] not in existing]
    for position, route in enumerate(pending, 1):
        started = time.time()
        forward_paths = [
            route["anchor_image_path"],
            *route["bridge_image_paths"],
            route["target_image_path"],
        ]
        trace = propagate_with_trace(
            model,
            forward_paths,
            route["anchor_box_xywh_normalized"],
            canvas,
        )
        anchor_mask = t21.load_mask(route["anchor_mask_path"], canvas)
        cycle = t21.propagate_return_from_predicted_mask(
            model,
            forward_paths,
            trace["final_mask"],
            canvas,
        )
        q_cycle = t21.dice(cycle["mask"], anchor_mask)
        gt = t21.load_mask(route["target_mask_path_evaluation_only"], canvas)
        gt_metric = t21.metrics(trace["final_mask"], gt)
        row = {
            **route,
            "status": "success",
            "q_cycle": q_cycle,
            "cycle_success": cycle["success"],
            "cycle_failure_reason": cycle["failure_reason"],
            "cycle_candidate_count": cycle["candidate_count"],
            "cycle_sam_score": cycle["sam_score"],
            "seconds": round(time.time() - started, 3),
            **{key: value for key, value in trace.items() if key != "final_mask"},
            **{f"gt_{key}_evaluation_only": value for key, value in gt_metric.items()},
        }
        t21.append_fsync(result_path, row)
        existing[row["route_id"]] = row
        print(
            f"[{position}/{len(pending)}] {route['target_id']} {route['route_type']} "
            f"dice={gt_metric['dice']:.4f} cycle={q_cycle:.4f}",
            flush=True,
        )
    final_rows = [existing[row["route_id"]] for row in routes]
    write_jsonl(result_path, final_rows)
    return final_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--canvas", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    mode_root = args.root / args.mode
    routes = sorted(
        read_jsonl(mode_root / f"{args.split}_pool0_stage1/routes.jsonl"),
        key=route_sort_key,
    )
    output_root = mode_root / f"propagation_quality_{args.split}"
    output_root.mkdir(parents=True, exist_ok=True)
    model = build_sam3_video_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        device="cuda",
        compile=False,
    )
    model.eval()
    rows = evaluate(model, routes, output_root, args.canvas, args.resume)
    summary = {}
    for bridge_count in range(8):
        values = [row["gt_dice_evaluation_only"] for row in rows if int(row["bridge_count"]) == bridge_count]
        cycles = [row["q_cycle"] for row in rows if int(row["bridge_count"]) == bridge_count]
        if values:
            summary[f"bridge_{bridge_count}"] = {
                "n": len(values),
                "dice": float(np.mean(values)),
                "q_cycle": float(np.mean(cycles)),
            }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mode": args.mode, "split": args.split, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
