#!/usr/bin/env python3
"""Compare SAM3 route propagation seeded by a first-frame MASK prompt vs BOX prompt.

For each frozen route (anchor -> bridges -> target):
  - mask prompt: seed the SAM3 tracker with the anchor GT mask via
    `Sam3TrackerPredictor.add_new_mask` and propagate forward; compute Dice vs the
    target GT at `--canvas` resolution.
  - box prompt: read the existing propagation-quality rows from the same route
    root (`propagation_quality_{split}/propagation_quality.jsonl`,
    `gt_dice_evaluation_only`).

The comparison is per-route (same route_id, same checkpoint), so the only
difference is the first-frame prompt type.

Usage:
  python scripts/compare_mask_vs_box_prompt.py \
    --root work/kvasir_1pct_anchors/stage1_feature_knn_b7_ftround2_ckpt1 \
    --checkpoint work/kvasir_1pct_anchors/video_checkpoints/round2_ckpt1_merged_video.pt \
    --split test --canvas 512 --min-bridge 3 --max-bridge 6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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

MODES = (
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
)


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def to_bool_mask(mask: Any, canvas: int) -> np.ndarray:
    """Convert a (H, W) mask tensor/array to a bool mask at canvas resolution."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    # video_res_masks may come back as (1, 1, H, W) or (1, H, W); collapse to 2D.
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim > 2:
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])
    if mask.dtype == np.bool_:
        pass
    elif mask.dtype == np.uint8 and mask.max() > 1:
        mask = mask > 127
    else:
        mask = mask > 0.5
    if mask.shape[0] != canvas or mask.shape[1] != canvas:
        mask = (
            np.asarray(
                Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).resize(
                    (canvas, canvas), Image.Resampling.NEAREST
                )
            )
            > 127
        )
    return mask


def propagate_with_mask_prompt(
    tracker: Any,
    forward_paths: list[str],
    anchor_mask_path: str,
    canvas: int,
    tmp_dir: Path,
    prompt_type: str = "mask",
    anchor_box_xywh_norm: list[float] | None = None,
) -> dict[str, Any]:
    """Seed the tracker on frame 0 and propagate forward.

    prompt_type:
      - "mask": anchor GT mask via the public add_new_mask API.
      - "mask_box": anchor GT mask + anchor box (as corner points) fed together into
        the SAM prompt encoder; the model's refined frame-0 mask is propagated.
    """
    # Refresh the numbered symlink frames in the temp dir.
    # NOTE: the SAM2-style tracker loader only accepts *.jpg/*.jpeg names; the
    # symlinks point at the original files, so there is no lossy conversion.
    for old in tmp_dir.glob("*.jpg"):
        old.unlink()
    for frame_idx, frame_path in enumerate(forward_paths):
        (tmp_dir / f"{frame_idx:04d}.jpg").symlink_to(frame_path)

    state = tracker.init_state(
        video_path=str(tmp_dir),
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    anchor_mask = torch.from_numpy(
        np.asarray(Image.open(anchor_mask_path).convert("L")) > 127
    )
    if prompt_type == "mask":
        tracker.add_new_mask(
            state,
            frame_idx=0,
            obj_id=1,
            mask=anchor_mask,
        )
    elif prompt_type == "mask_box":
        add_mask_and_box(
            tracker,
            state,
            frame_idx=0,
            obj_id=1,
            mask=anchor_mask,
            box_xywh_norm=anchor_box_xywh_norm,
        )
    else:
        raise ValueError(f"unknown prompt_type {prompt_type}")

    final_mask = None
    final_score = None
    num_frames = len(forward_paths)
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in tracker.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=num_frames,
        reverse=False,
        propagate_preflight=True,
    ):
        if frame_idx == num_frames - 1:
            final_mask = video_res_masks
            final_score = obj_scores
    del state
    torch.cuda.empty_cache()

    if final_mask is None:
        return {"success": False, "failure_reason": "no_final_frame_output"}
    mask_canvas = to_bool_mask(final_mask, canvas)
    score = (
        float(final_score[0])
        if isinstance(final_score, (torch.Tensor, np.ndarray)) and len(final_score) > 0
        else None
    )
    return {"success": True, "mask": mask_canvas, "sam_score": score}


def add_mask_and_box(
    tracker: Any,
    state: dict[str, Any],
    frame_idx: int,
    obj_id: int,
    mask: torch.Tensor,
    box_xywh_norm: list[float] | None,
) -> tuple[int, list, None, Any]:
    """Feed a dense GT mask AND a box (as corner points, labels 2/3) on one frame.

    This mirrors `Sam3TrackerPredictor.add_new_mask` but passes both sparse
    (box corners) and dense (mask) prompts to the SAM prompt encoder, and keeps
    the model's refined frame-0 mask (instead of overwriting it with the input
    mask like the video-editing `add_new_mask` does).
    """
    if box_xywh_norm is None:
        raise ValueError("box_xywh_norm is required for mask_box prompt")
    obj_idx = tracker._obj_id_to_idx(state, obj_id)
    point_inputs_per_frame = state["point_inputs_per_obj"][obj_idx]
    mask_inputs_per_frame = state["mask_inputs_per_obj"][obj_idx]

    assert mask.dim() == 2
    mask_inputs_orig = mask[None, None].float().to(state["device"])
    mask_H, mask_W = mask.shape
    if mask_H != tracker.input_mask_size or mask_W != tracker.input_mask_size:
        mask_inputs = F.interpolate(
            mask_inputs_orig,
            size=(tracker.input_mask_size, tracker.input_mask_size),
            align_corners=False,
            mode="bilinear",
            antialias=True,
        )
    else:
        mask_inputs = mask_inputs_orig

    video_H = state["video_height"]
    video_W = state["video_width"]
    if mask_H != video_H or mask_W != video_W:
        mask_inputs_video_res = F.interpolate(
            mask_inputs_orig,
            size=(video_H, video_W),
            align_corners=False,
            mode="bilinear",
            antialias=True,
        )
    else:
        mask_inputs_video_res = mask_inputs_orig
    mask_inputs_video_res = mask_inputs_video_res > 0.5

    # box xywh (normalized) -> two corner points in absolute pixels (labels 2/3),
    # consistent with how SAM 2 encodes box prompts.
    x, y, w, h = [float(value) for value in box_xywh_norm]
    box_xyxy_abs = (
        torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
        * tracker.image_size
    )
    point_inputs = {
        "point_coords": box_xyxy_abs.reshape(1, 2, 2).to(state["device"]),
        "point_labels": torch.tensor([[2, 3]], dtype=torch.int32, device=state["device"]),
    }

    mask_inputs_per_frame[frame_idx] = mask_inputs_video_res
    point_inputs_per_frame[frame_idx] = point_inputs

    is_init_cond_frame = frame_idx not in state["frames_already_tracked"]
    reverse = False if is_init_cond_frame else state["frames_already_tracked"][frame_idx]["reverse"]
    obj_output_dict = state["output_dict_per_obj"][obj_idx]
    obj_temp_output_dict = state["temp_output_dict_per_obj"][obj_idx]
    is_cond = is_init_cond_frame or tracker.add_all_frames_to_correct_as_cond
    storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

    current_out, _ = tracker._run_single_frame_inference(
        inference_state=state,
        output_dict=obj_output_dict,
        frame_idx=frame_idx,
        batch_size=1,
        is_init_cond_frame=is_init_cond_frame,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        reverse=reverse,
        run_mem_encoder=False,
    )
    obj_temp_output_dict[storage_key][frame_idx] = current_out

    obj_ids = state["obj_ids"]
    consolidated_out = tracker._consolidate_temp_output_across_obj(
        state,
        frame_idx,
        is_cond=is_cond,
        run_mem_encoder=False,
        consolidate_at_video_res=True,
    )
    _, video_res_masks = tracker._get_orig_video_res_output(
        state, consolidated_out["pred_masks_video_res"]
    )
    return frame_idx, obj_ids, None, video_res_masks


def propagate_with_detector_gtmask(
    model: Any,
    tracker: Any,
    forward_paths: list[str],
    prompt_box: list[float],
    anchor_mask_path: str,
    canvas: int,
) -> dict[str, Any]:
    """Run the standard box path (detector on frame 0) but seed the tracker on
    frame 0 with the anchor GT mask instead of the detector's frame-0 mask.

    This is the supported "mask refines detector output" combination: the
    detector still runs (bbox prompt, object ids/scores/geometry conditioning),
    while the mask the tracker propagates from is the GT mask.
    """
    anchor_mask = torch.from_numpy(
        np.asarray(Image.open(anchor_mask_path).convert("L")) > 127
    )
    original_add_new_mask = tracker.add_new_mask
    override = {"mask": anchor_mask}

    def patched_add_new_mask(inference_state, frame_idx, obj_id, mask, **kwargs):
        if frame_idx == 0 and override["mask"] is not None:
            mask = override["mask"]
        return original_add_new_mask(
            inference_state, frame_idx, obj_id, mask, **kwargs
        )

    tracker.add_new_mask = patched_add_new_mask
    try:
        frames = [t21.load_rgb(path, canvas) for path in forward_paths]
        state = model.init_state(
            resource_path=frames,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        _, prompted = model.add_prompt(
            state,
            frame_idx=0,
            text_str=None,
            boxes_xywh=[prompt_box],
            box_labels=[1],
        )
        prompt_ids = np.asarray(prompted["out_obj_ids"], dtype=np.int64)
        if len(prompt_ids) == 0:
            del state
            torch.cuda.empty_cache()
            return {"success": False, "failure_reason": "empty_detection_from_box"}
        final_mask = None
        final_score = None
        for frame_idx, output in model.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=len(frames),
            reverse=False,
        ):
            if frame_idx == len(frames) - 1:
                final_mask = t21.select_top(output, canvas)["mask"]
                final_score = output.get("out_probs")
        del state
        torch.cuda.empty_cache()
        if final_mask is None:
            return {"success": False, "failure_reason": "no_final_frame_output"}
        score = (
            float(final_score[0])
            if isinstance(final_score, (torch.Tensor, np.ndarray)) and len(final_score) > 0
            else None
        )
        return {"success": True, "mask": final_mask, "sam_score": score}
    finally:
        tracker.add_new_mask = original_add_new_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--canvas", type=int, default=512)
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--prompt",
        choices=("mask", "mask_box", "det_gtmask"),
        default="mask",
        help="first-frame prompt variant to test (mask = GT mask only; "
        "mask_box = GT mask + box corners fed together; "
        "det_gtmask = box detector runs, tracker seeded with GT mask)",
    )
    args = parser.parse_args()

    output = args.output or (ROOT / "work/kvasir_1pct_anchors/mask_prompt_experiment")
    output.mkdir(parents=True, exist_ok=True)
    tmp_dir = output / "tmp_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building SAM3 video model from {args.checkpoint} ...", flush=True)
    model = build_sam3_video_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        device="cuda",
        compile=False,
    )
    model.eval()
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    print("Tracker backbone attached.", flush=True)

    result_path = output / f"{args.prompt}_vs_box_{args.split}_b{args.min_bridge}_{args.max_bridge}.jsonl"
    done = {row["route_id"] for row in read_jsonl(result_path)}

    all_rows = []
    for mode in args.modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        mode_root = args.root / mode
        routes = read_jsonl(mode_root / f"{args.split}_pool0_stage1/routes.jsonl")
        baseline = {
            row["route_id"]: row
            for row in read_jsonl(mode_root / f"propagation_quality_{args.split}/propagation_quality.jsonl")
            if row.get("status") == "success"
        }
        selected = [
            route
            for route in routes
            if args.min_bridge <= int(route["bridge_count"]) <= args.max_bridge
            and route["route_id"] in baseline
        ]
        if args.max_targets is not None:
            seen = set()
            filtered = []
            for route in selected:
                if route["target_id"] in seen:
                    continue
                seen.add(route["target_id"])
                filtered.append(route)
                if len(seen) >= args.max_targets:
                    break
            selected = filtered

        print(
            f"\n[{mode}] routes to run: {len(selected)} "
            f"(b{args.min_bridge}-b{args.max_bridge}, {args.split})",
            flush=True,
        )
        for position, route in enumerate(selected, 1):
            if route["route_id"] in done:
                continue
            forward_paths = [
                route["anchor_image_path"],
                *route["bridge_image_paths"],
                route["target_image_path"],
            ]
            started = time.time()
            try:
                if args.prompt == "det_gtmask":
                    result = propagate_with_detector_gtmask(
                        model,
                        tracker,
                        forward_paths,
                        route["anchor_box_xywh_normalized"],
                        route["anchor_mask_path"],
                        args.canvas,
                    )
                else:
                    result = propagate_with_mask_prompt(
                        tracker,
                        forward_paths,
                        route["anchor_mask_path"],
                        args.canvas,
                        tmp_dir,
                        prompt_type=args.prompt,
                        anchor_box_xywh_norm=route["anchor_box_xywh_normalized"],
                    )
            except Exception as exc:  # noqa: BLE001 - keep going on per-route failures
                result = {"success": False, "failure_reason": f"exception: {exc}"}

            box_dice = baseline[route["route_id"]]["gt_dice_evaluation_only"]
            if result["success"]:
                gt = t21.load_mask(route["target_mask_path_evaluation_only"], args.canvas)
                mask_dice = t21.dice(result["mask"], gt)
            else:
                mask_dice = None
            row = {
                "route_id": route["route_id"],
                "mode": mode,
                "prompt": args.prompt,
                "target_id": route["target_id"],
                "anchor_id": route["anchor_id"],
                "bridge_count": int(route["bridge_count"]),
                "route_type": route["route_type"],
                "status": "success" if result["success"] else "error",
                "failure_reason": result.get("failure_reason"),
                "mask_dice_evaluation_only": mask_dice,
                "box_dice_evaluation_only": box_dice,
                "delta_mask_minus_box": (
                    None if mask_dice is None else round(mask_dice - box_dice, 6)
                ),
                "mask_sam_score": result.get("sam_score"),
                "seconds": round(time.time() - started, 3),
            }
            append_jsonl(result_path, row)
            done.add(route["route_id"])
            all_rows.append(row)
            if mask_dice is not None:
                print(
                    f"[{position}/{len(selected)}] {route['route_type']} "
                    f"mask={mask_dice:.4f} box={box_dice:.4f} "
                    f"delta={mask_dice - box_dice:+.4f}",
                    flush=True,
                )
            else:
                print(
                    f"[{position}/{len(selected)}] {route['route_type']} FAILED "
                    f"({result.get('failure_reason')})",
                    flush=True,
                )

    # Summarize per (mode, bridge).
    rows = read_jsonl(result_path)
    summary: dict[str, Any] = {"canvas": args.canvas, "checkpoint": str(args.checkpoint), "split": args.split}
    per_bridge: dict[str, dict[str, Any]] = {}
    for mode in args.modes:
        for bridge in range(args.min_bridge, args.max_bridge + 1):
            key = f"{mode}|b{bridge}"
            subset = [
                row
                for row in rows
                if row["mode"] == mode
                and row["bridge_count"] == bridge
                and row["mask_dice_evaluation_only"] is not None
            ]
            if not subset:
                continue
            mask_mean = float(np.mean([row["mask_dice_evaluation_only"] for row in subset]))
            box_mean = float(np.mean([row["box_dice_evaluation_only"] for row in subset]))
            wins = sum(row["mask_dice_evaluation_only"] > row["box_dice_evaluation_only"] for row in subset)
            per_bridge[key] = {
                "n": len(subset),
                "box_mean_dice": box_mean,
                "mask_mean_dice": mask_mean,
                "delta": mask_mean - box_mean,
                "mask_wins": wins,
            }
    summary["per_bridge"] = per_bridge
    summary_path = output / f"{args.prompt}_vs_box_{args.split}_b{args.min_bridge}_{args.max_bridge}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("\n=== Summary (mean Dice at canvas {}, {} checkpoint, prompt={}) ===".format(
        args.canvas, args.checkpoint.name, args.prompt
    ))
    print(f"{'mode':42s} {'bridge':>6s} {'N':>4s} {'box':>7s} {'mask':>7s} {'delta':>7s} {'wins':>5s}")
    for key in sorted(per_bridge):
        info = per_bridge[key]
        mode, bridge = key.rsplit("|", 1)
        print(
            f"{mode[:42]:42s} {bridge:>6s} {info['n']:>4d} "
            f"{info['box_mean_dice']:>7.4f} {info['mask_mean_dice']:>7.4f} "
            f"{info['delta']:>+7.4f} {info['mask_wins']:>5d}"
        )
    print(f"\nResults: {result_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
