#!/usr/bin/env python3
"""Replay frozen routes and export SAM3 tracker sigmoid maps without route search."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "t21", PROJECT_ROOT / "scripts/run_t21_dynamic_pseudovideo.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot import T21 implementation")
t21 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t21
spec.loader.exec_module(t21)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def append(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def install_soft_capture(model) -> None:
    original = model.build_outputs

    def wrapped(self, *args, **kwargs):
        output = original(*args, **kwargs)
        names = [
            "frame_idx",
            "num_frames",
            "reverse",
            "det_out",
            "tracker_low_res_masks_global",
            "tracker_obj_scores_global",
            "tracker_metadata_prev",
            "tracker_update_plan",
            "orig_vid_height",
            "orig_vid_width",
            "reconditioned_obj_ids",
            "det_to_matched_trk_obj_ids",
        ]
        values = dict(zip(names, args))
        values.update(kwargs)
        height = values["orig_vid_height"]
        width = values["orig_vid_width"]
        logits_by_id = {}
        previous_ids = values["tracker_metadata_prev"]["obj_ids_all_gpu"]
        previous_logits = F.interpolate(
            values["tracker_low_res_masks_global"].unsqueeze(1).float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        for obj_id, logits in zip(previous_ids, previous_logits):
            logits_by_id[int(obj_id)] = logits.squeeze(0)
        # The prompted anchor frame contains a newly detected object and has no
        # tracker propagation logits. The target frame must contain the same
        # object as an existing propagated masklet; replay() fails closed there
        # if its probability is unavailable.
        self._t26_soft_by_obj = {
            obj_id: torch.sigmoid(logits).detach().float().cpu().numpy()
            for obj_id, logits in logits_by_id.items()
        }
        return output

    model.build_outputs = types.MethodType(wrapped, model)


def replay(model, route: dict, canvas: int) -> tuple[np.ndarray, int, float]:
    paths = [
        route["anchor_image_path"],
        *route["bridge_image_paths"],
        route["target_image_path"],
    ]
    frames = [t21.load_rgb(path, canvas) for path in paths]
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
        boxes_xywh=[route["anchor_box_xywh_normalized"]],
        box_labels=[1],
    )
    final = None
    for frame_idx, output in model.propagate_in_video(
        state,
        start_frame_idx=0,
        max_frame_num_to_track=len(frames),
        reverse=False,
    ):
        if frame_idx == len(frames) - 1:
            final = output
    if final is None or len(final["out_obj_ids"]) == 0:
        probability = np.zeros((canvas, canvas), dtype=np.float32)
        return probability, 0, 0.0
    candidate = int(np.argmax(final["out_probs"]))
    obj_id = int(final["out_obj_ids"][candidate])
    probability = model._t26_soft_by_obj[obj_id]
    score = float(final["out_probs"][candidate])
    del state
    torch.cuda.empty_cache()
    return probability, len(final["out_obj_ids"]), score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-results", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--canvas", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    routes = read_jsonl(args.route_results)
    if args.limit:
        routes = routes[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"sam3_probabilities_{args.split}.jsonl"
    completed = {
        x["route_id"]: x
        for x in read_jsonl(manifest_path)
        if Path(x["sam3_probability_path"]).exists()
    } if manifest_path.exists() else {}
    model = t21.build_sam3_video_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device="cuda",
        compile=False,
    )
    model.eval()
    install_soft_capture(model)
    probability_dir = args.output_dir / args.split
    probability_dir.mkdir(parents=True, exist_ok=True)
    for index, route in enumerate(routes, 1):
        if route["route_id"] in completed:
            continue
        probability, candidate_count, score = replay(model, route, args.canvas)
        stored = t21.load_mask(route["forward_mask_path"], args.canvas)
        reproduced = probability >= 0.5
        mismatch = reproduced != stored
        corrected_pixels = int(mismatch.sum())
        if corrected_pixels:
            difference = float(mismatch.mean())
            if difference > 1e-4:
                raise RuntimeError(
                    f"Frozen binary route mismatch {route['route_id']}: {difference}"
                )
            # Replay can flip a handful of logits numerically indistinguishable
            # from zero. Keep the frozen binary mask authoritative while retaining
            # the captured probability everywhere else.
            probability[np.logical_and(stored, mismatch)] = 0.5001
            probability[np.logical_and(~stored, mismatch)] = 0.4999
            reproduced = probability >= 0.5
        if not np.array_equal(reproduced, stored):
            raise RuntimeError(
                f"Boundary correction failed for {route['route_id']}"
            )
        path = probability_dir / f"{route['route_id']}.png"
        Image.fromarray(np.rint(probability * 65535).astype(np.uint16)).save(path)
        row = {
            "split": args.split,
            "target_id": route["target_id"],
            "route_id": route["route_id"],
            "route_type": route["route_type"],
            "sam3_probability_path": str(path.resolve()),
            "binary_reproduction_exact": True,
            "boundary_corrected_pixels": corrected_pixels,
            "boundary_corrected_fraction": corrected_pixels / (args.canvas**2),
            "candidate_count": candidate_count,
            "sam_score": score,
            "frozen_route_replayed_without_search": True,
        }
        append(manifest_path, row)
        print(f"[{index}/{len(routes)}] {route['route_id']}", flush=True)
    (args.output_dir / f"{args.split.upper()}_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
