#!/usr/bin/env python3
"""Category-free single-image adaptation of SAM3's tracker mask decoder."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from train_t22_sam3_tracker import (
    backbone_features,
    load_base_tracker,
    load_frame,
    read_jsonl,
)
from train_t23_memory_adapter import segmentation_loss, tight_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-video-manifest", type=Path, required=True)
    parser.add_argument("--pseudo-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-objectness", type=float, default=0.1)
    parser.add_argument("--lambda-iou-score", type=float, default=0.1)
    parser.add_argument("--pseudo-weight", type=float, default=0.2)
    parser.add_argument("--negative-weight", type=float, default=0.05)
    parser.add_argument("--box-jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-every", type=int, default=100)
    return parser.parse_args()


def unique_gt_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row["anchor_id"]
        result.setdefault(
            key,
            {
                "id": key,
                "image_path": row["image_paths"][0],
                "mask_path": row["mask_paths"][0],
            },
        )
    return [result[key] for key in sorted(result)]


def pseudo_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "id": row["target_id"],
            "image_path": row["target_image_path"],
            "mask_path": row["pseudo_mask_path"],
        }
        for row in rows
    ]


def jitter_box(
    box: torch.Tensor, height: int, width: int, fraction: float, rng: random.Random
) -> torch.Tensor:
    result = box.clone()
    x1, y1 = result[0, 0, 0].item(), result[0, 0, 1].item()
    x2, y2 = result[0, 1, 0].item(), result[0, 1, 1].item()
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    result[0, 0, 0] = max(0, x1 + rng.uniform(-fraction, fraction) * bw)
    result[0, 1, 0] = min(width - 1, x2 + rng.uniform(-fraction, fraction) * bw)
    result[0, 0, 1] = max(0, y1 + rng.uniform(-fraction, fraction) * bh)
    result[0, 1, 1] = min(height - 1, y2 + rng.uniform(-fraction, fraction) * bh)
    return result


def background_box(mask: torch.Tensor, rng: random.Random) -> torch.Tensor:
    height, width = mask.shape[-2:]
    foreground = mask[0] > 0.5
    for _ in range(200):
        bw = rng.randint(max(16, width // 10), max(17, width // 3))
        bh = rng.randint(max(16, height // 10), max(17, height // 3))
        x1 = rng.randint(0, max(0, width - bw - 1))
        y1 = rng.randint(0, max(0, height - bh - 1))
        covered = foreground[y1 : y1 + bh, x1 : x1 + bw]
        if float(covered.float().mean()) <= 0.01:
            return torch.tensor(
                [[[x1, y1], [x1 + bw, y1 + bh]]],
                dtype=torch.float32,
                device=mask.device,
            )
    # Deterministic least-overlap fallback over the four image corners.
    bw, bh = max(16, width // 5), max(16, height // 5)
    candidates = [
        (0, 0),
        (width - bw - 1, 0),
        (0, height - bh - 1),
        (width - bw - 1, height - bh - 1),
    ]
    x1, y1 = min(
        candidates,
        key=lambda xy: float(
            foreground[xy[1] : xy[1] + bh, xy[0] : xy[0] + bw]
            .float()
            .mean()
        ),
    )
    return torch.tensor(
        [[[x1, y1], [x1 + bw, y1 + bh]]],
        dtype=torch.float32,
        device=mask.device,
    )


def actual_soft_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.float().sigmoid()
    target = target.float()
    if target.ndim == 3:
        target = target.unsqueeze(0)
    intersection = (probability * target).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
    return ((2 * intersection + 1) / (denominator + 1)).detach()


def load_sample(
    row: dict[str, str],
    args: argparse.Namespace,
    device: torch.device,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image, mask = load_frame(
        row["image_path"], row["mask_path"], args.image_size, device
    )
    box = jitter_box(
        tight_box(mask), mask.shape[-2], mask.shape[-1], args.box_jitter, rng
    )
    return image, mask, box


def forward_single_batch(
    tracker: torch.nn.Module,
    images: torch.Tensor,
    boxes: list[torch.Tensor],
    sample_indices: list[int] | None = None,
) -> list[dict[str, torch.Tensor]]:
    feats, positions, sizes = backbone_features(tracker, images)
    if sample_indices is None:
        sample_indices = list(range(len(boxes)))
    outputs = []
    for index, box in zip(sample_indices, boxes):
        point_inputs = {
            "point_coords": box,
            "point_labels": torch.tensor(
                [[2, 3]], dtype=torch.int32, device=images.device
            ),
        }
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        output = tracker.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=[item[:, index : index + 1] for item in feats],
            current_vision_pos_embeds=[
                item[:, index : index + 1] for item in positions
            ],
            feat_sizes=sizes,
            image=images[index : index + 1],
            point_inputs=point_inputs,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=1,
            run_mem_encoder=False,
        )
        outputs.append(output)
    return outputs


def output_loss(
    output: dict[str, torch.Tensor],
    mask: torch.Tensor,
    args: argparse.Namespace,
    *,
    object_present: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask_loss, mask_terms = segmentation_loss(
        output["pred_masks_high_res"], mask, args.lambda_bce
    )
    objectness = F.binary_cross_entropy_with_logits(
        output["object_score_logits"].float(),
        torch.full_like(
            output["object_score_logits"],
            float(object_present),
            dtype=torch.float32,
        ),
    )
    dice_target = actual_soft_dice(output["pred_masks_high_res"], mask)
    iou_score = output["iou_score"].float().reshape(dice_target.shape)
    iou_loss = F.mse_loss(iou_score, dice_target)
    total = (
        mask_loss
        + args.lambda_objectness * objectness
        + args.lambda_iou_score * iou_loss
    )
    binary = output["pred_masks_high_res"].detach().sigmoid() > 0.5
    return total, {
        **mask_terms,
        "objectness_loss": float(objectness.detach()),
        "iou_score_loss": float(iou_loss.detach()),
        "soft_dice": float(dice_target.mean()),
        "nonempty": float(binary.any()),
        "area": float(binary.float().mean()),
    }


def save_checkpoint(
    path: Path,
    tracker: torch.nn.Module,
    names: list[str],
    args: argparse.Namespace,
    step: int,
) -> None:
    selected = set(names)
    state = {
        name: value.detach().cpu()
        for name, value in tracker.state_dict().items()
        if name in selected
    }
    torch.save(
        {
            "tracker_state_dict": state,
            "trainable_parameter_names": names,
            "step": step,
            "config": vars(args),
            "video_memory_frozen": True,
            "category_text_used": False,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    gt = unique_gt_rows(read_jsonl(args.gt_video_manifest))
    pseudo = pseudo_rows(read_jsonl(args.pseudo_manifest))
    if len(gt) != 16 or not pseudo:
        raise RuntimeError(f"Expected 16 GT and pseudo data, got {len(gt)}, {len(pseudo)}")

    tracker = load_base_tracker(args.base_checkpoint, device)
    tracker.eval()
    for parameter in tracker.parameters():
        parameter.requires_grad_(False)
    for parameter in tracker.sam_mask_decoder.parameters():
        parameter.requires_grad_(True)
    trainable_names = [
        name for name, parameter in tracker.named_parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in tracker.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        gt_count=len(gt),
        pseudo_count=len(pseudo),
        trainable_parameter_count=sum(
            parameter.numel()
            for parameter in tracker.parameters()
            if parameter.requires_grad
        ),
        trainable_prefix="sam_mask_decoder",
        memory_frozen=True,
        category_text_used=False,
        batch_composition="1 GT + 1 pseudo",
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    save_checkpoint(
        args.output_dir / "decoder_step000000.pt",
        tracker,
        trainable_names,
        args,
        0,
    )
    log_path = args.output_dir / "train.jsonl"
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        gt_row = gt[(step - 1) % len(gt)]
        pseudo_row = pseudo[(step - 1) % len(pseudo)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gt_image, gt_mask, gt_box = load_sample(
                gt_row, args, device, rng
            )
            pseudo_image, pseudo_mask, pseudo_box = load_sample(
                pseudo_row, args, device, rng
            )
            negative_prompt = background_box(gt_mask, rng)
            gt_output, pseudo_output, negative_output = forward_single_batch(
                tracker,
                torch.stack([gt_image, pseudo_image]),
                [gt_box, pseudo_box, negative_prompt],
                [0, 1, 0],
            )
            gt_loss, gt_terms = output_loss(gt_output, gt_mask, args)
            pseudo_loss, pseudo_terms = output_loss(
                pseudo_output, pseudo_mask, args
            )
            empty_mask = torch.zeros_like(gt_mask)
            negative_loss, negative_terms = output_loss(
                negative_output,
                empty_mask,
                args,
                object_present=False,
            )
            loss = (
                gt_loss
                + args.pseudo_weight * pseudo_loss
                + args.negative_weight * negative_loss
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in tracker.parameters() if parameter.requires_grad],
            0.1,
        )
        warmup = min(1.0, step / max(1, args.warmup_steps))
        decay = math.sqrt(max(1, args.warmup_steps) / max(step, args.warmup_steps))
        lr = args.lr * warmup * decay
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "lr": lr,
            "grad_norm": float(grad_norm),
            **{f"gt_{key}": value for key, value in gt_terms.items()},
            **{f"pseudo_{key}": value for key, value in pseudo_terms.items()},
            **{f"negative_{key}": value for key, value in negative_terms.items()},
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step == 1 or step % 10 == 0:
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(
                args.output_dir / f"decoder_step{step:06d}.pt",
                tracker,
                trainable_names,
                args,
                step,
            )
    (args.output_dir / "TRAINING_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
