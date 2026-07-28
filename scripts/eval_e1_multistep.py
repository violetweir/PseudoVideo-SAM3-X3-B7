#!/usr/bin/env python3
"""Evaluate star, chain, or rolling anchor+previous SAM3 pseudo-video memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--structure", choices=("star", "chain", "hybrid"), required=True)
    parser.add_argument(
        "--anchor-init",
        choices=("box", "box_gt_mask"),
        default="box",
        help=(
            "box: original E1; box_gt_mask: use the same box to create the visual "
            "object, then replace frame-0 tracker conditioning memory with support GT."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/Data_8TB/lht/models/modelscope/models/facebook--sam3/"
            "snapshots/master/sam3.pt"
        ),
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument(
        "--hybrid-confidence",
        type=float,
        default=0.5,
        help="Fixed, untuned gate for retaining the previous frame in hybrid memory.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--matched-k5-only",
        action="store_true",
        help="Evaluate only targets eligible at every depth K=1..5.",
    )
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


def append_fsync(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_rgb(path: str, canvas_size: int) -> Image.Image:
    return (
        Image.open(path)
        .convert("RGB")
        .resize((canvas_size, canvas_size), Image.Resampling.BILINEAR)
    )


def load_gt(path: str, canvas_size: int) -> np.ndarray:
    return (
        np.asarray(
            Image.open(path)
            .convert("L")
            .resize((canvas_size, canvas_size), Image.Resampling.NEAREST)
        )
        > 127
    )


def segmentation_metrics(prediction: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = prediction.astype(bool)
    target = gt.astype(bool)
    intersection = int(np.logical_and(pred, target).sum())
    pred_area = int(pred.sum())
    target_area = int(target.sum())
    union = pred_area + target_area - intersection
    return {
        "dice": 2 * intersection / max(pred_area + target_area, 1),
        "iou": intersection / max(union, 1),
        "precision": intersection / max(pred_area, 1),
        "recall": intersection / max(target_area, 1),
        "pred_area_ratio": pred_area / pred.size,
        "gt_area_ratio": target_area / target.size,
    }


def select_top(output: dict[str, Any], shape: tuple[int, int]) -> dict[str, Any]:
    masks = np.asarray(output["out_binary_masks"], dtype=bool)
    scores = np.asarray(output["out_probs"], dtype=np.float64)
    if len(masks):
        index = int(np.argmax(scores))
        mask = masks[index]
        return {
            "mask": mask,
            "candidate_count": int(len(masks)),
            "score": float(scores[index]),
        }
    return {
        "mask": np.zeros(shape, dtype=bool),
        "candidate_count": 0,
        "score": 0.0,
    }


def propagate(
    model: Any,
    anchor: Image.Image,
    frames: list[Image.Image],
    anchor_box: list[float],
    anchor_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = model.init_state(
        resource_path=[anchor, *frames],
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    _, prompted = model.add_prompt(
        state,
        frame_idx=0,
        text_str=None,
        boxes_xywh=[anchor_box],
        box_labels=[1],
    )
    if anchor_mask is not None:
        corrected_objects = 0
        mask_tensor = torch.from_numpy(anchor_mask.astype(np.float32))
        for tracker_state in state["tracker_inference_states"]:
            for obj_id in list(tracker_state["obj_ids"]):
                model.tracker.add_new_mask(
                    inference_state=tracker_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=mask_tensor,
                    add_mask_to_memory=True,
                )
                corrected_objects += 1
            if tracker_state["obj_ids"]:
                model.tracker.propagate_in_video_preflight(
                    tracker_state, run_mem_encoder=True
                )
        if corrected_objects != 1:
            raise RuntimeError(
                "Expected exactly one anchor object before GT-mask correction, "
                f"found {corrected_objects}"
            )
    last_output = None
    final_index = len(frames)
    for frame_idx, output in model.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=final_index + 1,
        reverse=False,
    ):
        if frame_idx == final_index:
            last_output = output
    if last_output is None:
        raise RuntimeError(f"No propagated output for frame {final_index}")
    del state
    return prompted, last_output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows), "by_depth": {}}
    for depth in sorted({int(row["depth"]) for row in rows}):
        depth_rows = [row for row in rows if int(row["depth"]) == depth]
        matched_rows = [row for row in depth_rows if row["matched_k5"]]

        def group_stats(group: list[dict[str, Any]]) -> dict[str, Any]:
            dice = np.asarray([row["dice"] for row in group], dtype=np.float64)
            anchors = sorted({row["anchor_merged_id"] for row in group})
            anchor_means = [
                float(
                    np.mean(
                        [
                            row["dice"]
                            for row in group
                            if row["anchor_merged_id"] == anchor
                        ]
                    )
                )
                for anchor in anchors
            ]
            return {
                "n": len(group),
                "dice_mean_micro": float(dice.mean()) if len(dice) else None,
                "dice_std": float(dice.std()) if len(dice) else None,
                "dice_mean_macro_anchor": (
                    float(np.mean(anchor_means)) if anchor_means else None
                ),
                "iou_mean": (
                    float(np.mean([row["iou"] for row in group])) if group else None
                ),
                "propagation_success_rate_nonempty": (
                    float(np.mean([row["candidate_count"] > 0 for row in group]))
                    if group
                    else None
                ),
                "positive_overlap_rate": (
                    float(np.mean([row["dice"] > 0 for row in group]))
                    if group
                    else None
                ),
            }

        result["by_depth"][str(depth)] = {
            "all_eligible": group_stats(depth_rows),
            "matched_k5": group_stats(matched_rows),
        }
    return result


def main() -> None:
    args = parse_args()
    all_windows = read_jsonl(args.manifest.resolve())
    windows = [
        row
        for row in all_windows
        if row["query_split"] == args.split
        and args.min_depth <= int(row["depth"]) <= args.max_depth
        and (not args.matched_k5_only or row["matched_k5"])
    ]
    if args.limit:
        windows = windows[: args.limit]
    output_root = args.output_root.resolve()
    result_path = output_root / f"{args.structure}_per_target.jsonl"
    existing_rows = read_jsonl(result_path) if args.resume else []
    existing = {
        (row["sequence_id"], row["structure"]): row for row in existing_rows
    }
    pending = [
        row
        for row in windows
        if (row["sequence_id"], args.structure) not in existing
    ]
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(
        f"structure={args.structure} split={args.split} "
        f"windows={len(windows)} pending={len(pending)}"
    )
    print(
        f"canvas={args.canvas_size} anchor_init={args.anchor_init} "
        f"hybrid_confidence={args.hybrid_confidence}"
    )

    model = None
    if pending:
        model = build_sam3_video_model(
            checkpoint_path=str(args.checkpoint),
            load_from_HF=False,
            device="cuda",
            compile=False,
        )
        model.eval()
    mask_root = output_root / f"{args.structure}_predictions"
    mask_root.mkdir(parents=True, exist_ok=True)

    # Star is invariant to K. Reuse its first fresh 512x512 result for the same target.
    star_cache = {
        row["target_merged_id"]: row
        for row in existing_rows
        if row["structure"] == "star"
    }
    for position, window in enumerate(pending, 1):
        started = time.time()
        anchor = load_rgb(window["anchor_image_path"], args.canvas_size)
        anchor_mask = (
            load_gt(window["anchor_mask_path"], args.canvas_size)
            if args.anchor_init == "box_gt_mask"
            else None
        )
        frame_paths = window["frame_image_paths"]
        gate_trace: list[dict[str, Any]] = []
        if args.structure == "star" and window["target_merged_id"] in star_cache:
            cached = star_cache[window["target_merged_id"]]
            top = {
                "mask": np.asarray(
                    Image.open(cached["prediction_path"]).convert("L")
                )
                > 127,
                "candidate_count": cached["candidate_count"],
                "score": cached["top_score"],
            }
            prompted_count = cached["prompted_frame_candidate_count"]
        elif args.structure == "star":
            target_frame = load_rgb(frame_paths[-1], args.canvas_size)
            prompted, output = propagate(
                model,
                anchor,
                [target_frame],
                window["anchor_box_xywh_normalized"],
                anchor_mask,
            )
            top = select_top(output, (args.canvas_size, args.canvas_size))
            prompted_count = int(len(prompted["out_binary_masks"]))
        elif args.structure == "chain":
            frames = [load_rgb(path, args.canvas_size) for path in frame_paths]
            prompted, output = propagate(
                model,
                anchor,
                frames,
                window["anchor_box_xywh_normalized"],
                anchor_mask,
            )
            top = select_top(output, (args.canvas_size, args.canvas_size))
            prompted_count = int(len(prompted["out_binary_masks"]))
        else:
            previous_path = None
            previous_top = None
            prompted_count = 0
            for current_path in frame_paths:
                keep_previous = (
                    previous_path is not None
                    and previous_top is not None
                    and previous_top["candidate_count"] > 0
                    and previous_top["score"] >= args.hybrid_confidence
                )
                selected_paths = (
                    [previous_path, current_path] if keep_previous else [current_path]
                )
                prompted, output = propagate(
                    model,
                    anchor,
                    [load_rgb(path, args.canvas_size) for path in selected_paths],
                    window["anchor_box_xywh_normalized"],
                    anchor_mask,
                )
                previous_top = select_top(
                    output, (args.canvas_size, args.canvas_size)
                )
                gate_trace.append(
                    {
                        "current_image_path": current_path,
                        "previous_included": keep_previous,
                        "candidate_count": previous_top["candidate_count"],
                        "top_score": previous_top["score"],
                    }
                )
                previous_path = current_path
                prompted_count = int(len(prompted["out_binary_masks"]))
                torch.cuda.empty_cache()
            top = previous_top

        gt = load_gt(window["target_mask_path"], args.canvas_size)
        values = segmentation_metrics(top["mask"], gt)
        safe_target = window["target_merged_id"].replace("::", "__")
        prediction_path = (
            mask_root
            / f"{window['sequence_id']}_K{window['depth']}_{safe_target}.png"
        )
        Image.fromarray(top["mask"].astype(np.uint8) * 255).save(prediction_path)
        result = {
            **window,
            "structure": args.structure,
            "anchor_init": args.anchor_init,
            "canvas_size": args.canvas_size,
            "hybrid_confidence": (
                args.hybrid_confidence if args.structure == "hybrid" else None
            ),
            "hybrid_gate_trace": gate_trace,
            "candidate_count": top["candidate_count"],
            "prompted_frame_candidate_count": prompted_count,
            "top_score": top["score"],
            **values,
            "prediction_path": str(prediction_path),
            "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "seconds": round(time.time() - started, 3),
            "status": "success",
        }
        append_fsync(result_path, result)
        existing[(window["sequence_id"], args.structure)] = result
        if args.structure == "star":
            star_cache[window["target_merged_id"]] = result
        print(
            f"[{position}/{len(pending)}] K={window['depth']} "
            f"{window['target_merged_id']} dice={values['dice']:.4f} "
            f"n={top['candidate_count']} score={top['score']:.4f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    final_rows = [
        existing[(row["sequence_id"], args.structure)]
        for row in windows
        if (row["sequence_id"], args.structure) in existing
    ]
    temporary = result_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in final_rows
        ),
        encoding="utf-8",
    )
    temporary.replace(result_path)
    summary = {
        "structure": args.structure,
        "anchor_init": args.anchor_init,
        "split": args.split,
        "canvas_size": args.canvas_size,
        "hybrid_definition": (
            "at each step, fresh [anchor, previous, current] if previous SAM score "
            f">={args.hybrid_confidence}; otherwise fresh [anchor, current]; "
            "only anchor receives the frozen train-GT box"
            if args.structure == "hybrid"
            else None
        ),
        **summarize(final_rows),
    }
    (output_root / f"{args.structure}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
