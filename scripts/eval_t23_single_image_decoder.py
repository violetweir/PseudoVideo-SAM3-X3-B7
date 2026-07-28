#!/usr/bin/env python3
"""Fixed train-only positive/negative probe for T23 single-image decoder."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from train_t22_sam3_tracker import load_base_tracker, load_frame, read_jsonl
from train_t23_memory_adapter import tight_box
from train_t23_single_image_decoder import (
    actual_soft_dice,
    background_box,
    forward_single_batch,
    jitter_box,
    unique_gt_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-video-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="label=/absolute/path.pt; use label=FROZEN for no adapter",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--box-jitter", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_adapter(tracker: torch.nn.Module, spec: str) -> str:
    label, path = spec.split("=", 1)
    if path == "FROZEN":
        return label
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("tracker_state_dict", payload)
    missing, unexpected = tracker.load_state_dict(state, strict=False)
    relevant = set(payload.get("trainable_parameter_names", ()))
    bad_missing = [name for name in missing if name in relevant]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"{label}: bad adapter missing={bad_missing}, unexpected={unexpected}"
        )
    return label


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = [
        "positive_soft_dice",
        "positive_nonempty",
        "positive_area",
        "positive_object_probability",
        "positive_iou_score",
        "negative_nonempty",
        "negative_area",
        "negative_object_probability",
        "negative_iou_score",
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    anchors = unique_gt_rows(read_jsonl(args.gt_video_manifest))
    all_rows: list[dict[str, float | str]] = []
    summaries = {}
    for spec in args.adapter:
        tracker = load_base_tracker(args.base_checkpoint, device)
        label = load_adapter(tracker, spec)
        tracker.eval()
        rows = []
        with torch.inference_mode():
            for index, anchor in enumerate(anchors):
                image, mask = load_frame(
                    anchor["image_path"],
                    anchor["mask_path"],
                    args.image_size,
                    device,
                )
                positive_box = jitter_box(
                    tight_box(mask),
                    mask.shape[-2],
                    mask.shape[-1],
                    args.box_jitter,
                    random.Random(args.seed + 10000 + index),
                )
                negative_box = background_box(
                    mask, random.Random(args.seed + index)
                )
                positive, negative = forward_single_batch(
                    tracker,
                    image.unsqueeze(0),
                    [positive_box, negative_box],
                    sample_indices=[0, 0],
                )
                positive_logits = positive["pred_masks_high_res"]
                negative_logits = negative["pred_masks_high_res"]
                row = {
                    "checkpoint": label,
                    "anchor_id": anchor["id"],
                    "positive_soft_dice": float(
                        actual_soft_dice(positive_logits, mask).mean()
                    ),
                    "positive_nonempty": float(
                        (positive_logits.sigmoid() > 0.5).any()
                    ),
                    "positive_area": float(
                        (positive_logits.sigmoid() > 0.5).float().mean()
                    ),
                    "positive_object_probability": float(
                        positive["object_score_logits"].float().sigmoid().mean()
                    ),
                    "positive_iou_score": float(
                        positive["iou_score"].float().mean()
                    ),
                    "negative_nonempty": float(
                        (negative_logits.sigmoid() > 0.5).any()
                    ),
                    "negative_area": float(
                        (negative_logits.sigmoid() > 0.5).float().mean()
                    ),
                    "negative_object_probability": float(
                        negative["object_score_logits"].float().sigmoid().mean()
                    ),
                    "negative_iou_score": float(
                        negative["iou_score"].float().mean()
                    ),
                }
                rows.append(row)
                all_rows.append(row)
        summaries[label] = summarize(rows)
        print(json.dumps({"checkpoint": label, **summaries[label]}), flush=True)
        del tracker
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summaries, "rows": all_rows}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
