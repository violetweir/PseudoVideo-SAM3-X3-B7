#!/usr/bin/env python3
"""Differentiable SAM3 tracker adaptation for the T22 protocol.

This entry deliberately bypasses the inference-only video API and calls the
tracker core. Pseudo labels are terminal-frame supervision only; every sequence
must start from a generation-0 human anchor.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sam3.model_builder import build_tracker


MEMORY_PREFIXES = (
    "transformer.",
    "maskmem_backbone.",
    "maskmem_tpos_enc",
    "no_mem_embed",
    "no_mem_pos_enc",
    "no_obj_ptr",
    "no_obj_embed_spatial",
    "obj_ptr_proj.",
    "obj_ptr_tpos_proj.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help="Optional earlier T22 tracker adapter used to initialize this stage.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("anchor", "pseudo"), default="anchor")
    parser.add_argument("--variant", choices=("md", "md_mem", "full"), default="md")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-pseudo", type=float, default=0.5)
    parser.add_argument("--lambda-path", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--save-step0",
        action="store_true",
        help="Save an untouched adapter immediately after base/init loading.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_base_tracker(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    tracker = build_tracker(
        apply_temporal_disambiguation=True, with_backbone=True
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model", payload)
    mapped = {}
    for name, value in state.items():
        if name.startswith("tracker."):
            mapped[name[len("tracker.") :]] = value
        elif name.startswith("detector.backbone.vision_backbone."):
            mapped[
                "backbone.vision_backbone."
                + name[len("detector.backbone.vision_backbone.") :]
            ] = value
    missing, unexpected = tracker.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Base tracker mapping mismatch: missing={missing}, unexpected={unexpected}"
        )
    return tracker.to(device)


def choose_trainable(tracker: torch.nn.Module, variant: str) -> list[str]:
    names = []
    for name, parameter in tracker.named_parameters():
        if variant == "full":
            enabled = True
        else:
            enabled = name.startswith("sam_mask_decoder.")
            if variant == "md_mem":
                enabled = enabled or name.startswith(MEMORY_PREFIXES)
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    if not names:
        raise RuntimeError("No trainable parameters selected")
    return names


def augment_pair(
    image: Image.Image, mask: Image.Image, rng: random.Random
) -> tuple[Image.Image, Image.Image]:
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if rng.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.85, 1.15))
    return image, mask


def load_frame(
    image_path: str,
    mask_path: str | None,
    size: int,
    device: torch.device,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L") if mask_path else None
    if rng is not None and mask is not None:
        image, mask = augment_pair(image, mask, rng)
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = ((tensor - 0.5) / 0.5).to(device)
    mask_tensor = None
    if mask is not None:
        mask = mask.resize((size, size), Image.Resampling.NEAREST)
        mask_tensor = torch.from_numpy(
            (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        )[None].to(device)
    return tensor, mask_tensor


def tight_box(mask: torch.Tensor, rng: random.Random) -> torch.Tensor:
    ys, xs = torch.where(mask[0] > 0.5)
    if len(xs) == 0:
        raise RuntimeError("Empty human anchor mask")
    x1, x2 = xs.min().item(), xs.max().item()
    y1, y2 = ys.min().item(), ys.max().item()
    # Small outward/inward jitter improves robustness without changing semantics.
    width, height = max(1, x2 - x1), max(1, y2 - y1)
    jitter_x, jitter_y = 0.03 * width, 0.03 * height
    x1 = max(0.0, x1 + rng.uniform(-jitter_x, jitter_x))
    x2 = min(mask.shape[-1] - 1.0, x2 + rng.uniform(-jitter_x, jitter_x))
    y1 = max(0.0, y1 + rng.uniform(-jitter_y, jitter_y))
    y2 = min(mask.shape[-2] - 1.0, y2 + rng.uniform(-jitter_y, jitter_y))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("Degenerate jittered box")
    return torch.tensor(
        [[[x1, y1], [x2, y2]]], dtype=torch.float32, device=mask.device
    )


def backbone_features(
    tracker: torch.nn.Module, images: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[int, int]]]:
    # Backbone stays frozen for md/md_mem. Decoder high-resolution projections
    # remain outside no_grad because conv_s0/conv_s1 belong to sam_mask_decoder.
    backbone_trainable = any(
        parameter.requires_grad for parameter in tracker.backbone.parameters()
    )
    with torch.set_grad_enabled(backbone_trainable):
        output = tracker.backbone.forward_image(images)["sam2_backbone_out"]
    output["backbone_fpn"][0] = tracker.sam_mask_decoder.conv_s0(
        output["backbone_fpn"][0]
    )
    output["backbone_fpn"][1] = tracker.sam_mask_decoder.conv_s1(
        output["backbone_fpn"][1]
    )
    _, feats, positions, sizes = tracker._prepare_backbone_features(output)
    return feats, positions, sizes


def forward_sequence(
    tracker: torch.nn.Module,
    images: torch.Tensor,
    anchor_box: torch.Tensor,
) -> list[dict[str, torch.Tensor]]:
    feats, positions, sizes = backbone_features(tracker, images)
    outputs: dict[str, dict[int, dict[str, torch.Tensor]]] = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    rows = []
    for index in range(images.shape[0]):
        prompt = None
        if index == 0:
            prompt = {
                "point_coords": anchor_box,
                "point_labels": torch.tensor(
                    [[2, 3]], dtype=torch.int32, device=images.device
                ),
            }
        result = tracker.track_step(
            frame_idx=index,
            is_init_cond_frame=index == 0,
            current_vision_feats=[item[:, index : index + 1] for item in feats],
            current_vision_pos_embeds=[
                item[:, index : index + 1] for item in positions
            ],
            feat_sizes=sizes,
            image=images[index : index + 1],
            point_inputs=prompt,
            mask_inputs=None,
            output_dict=outputs,
            num_frames=images.shape[0],
            run_mem_encoder=True,
        )
        bucket = "cond_frame_outputs" if index == 0 else "non_cond_frame_outputs"
        outputs[bucket][index] = result
        rows.append(result)
    return rows


def segmentation_loss(
    logits: torch.Tensor, target: torch.Tensor, lambda_bce: float
) -> tuple[torch.Tensor, dict[str, float]]:
    target = target[None] if target.ndim == 3 else target
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
    probability = logits.float().sigmoid()
    intersection = (probability * target).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    total = dice_loss + lambda_bce * bce
    return total, {"dice_loss": float(dice_loss.detach()), "bce": float(bce.detach())}


def save_adapter(
    path: Path,
    tracker: torch.nn.Module,
    names: list[str],
    metadata: dict[str, Any],
) -> None:
    name_set = set(names)
    state = {
        name: value.detach().cpu()
        for name, value in tracker.state_dict().items()
        if name in name_set
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tracker_state_dict": state,
            "trainable_parameter_names": names,
            **metadata,
        },
        path,
    )


def anchor_balanced_order(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["anchor_id"]].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered = []
    while buckets:
        for anchor_id in sorted(list(buckets)):
            bucket = buckets[anchor_id]
            ordered.append(bucket.pop())
            if not bucket:
                del buckets[anchor_id]
    return ordered


def module_grad_norms(tracker: torch.nn.Module) -> dict[str, float]:
    totals = {"md": 0.0, "memory": 0.0, "backbone": 0.0, "other": 0.0}
    for name, parameter in tracker.named_parameters():
        if parameter.grad is None:
            continue
        squared = float(parameter.grad.detach().float().pow(2).sum())
        if name.startswith("sam_mask_decoder."):
            group = "md"
        elif name.startswith(MEMORY_PREFIXES):
            group = "memory"
        elif name.startswith("backbone."):
            group = "backbone"
        else:
            group = "other"
        totals[group] += squared
    return {f"grad_{name}": value**0.5 for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda")
    rows = read_jsonl(args.manifest)
    if not rows:
        raise RuntimeError("Empty training manifest")
    if args.mode == "pseudo":
        rows = anchor_balanced_order(rows, rng)
    tracker = load_base_tracker(args.base_checkpoint, device)
    if args.init_adapter:
        initial = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
        initial_state = initial.get("tracker_state_dict", initial)
        _, unexpected = tracker.load_state_dict(initial_state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected init-adapter keys: {unexpected}")
    # build_tracker constructs the inference predictor subclass; this training-only
    # flag is normally injected by the official training wrapper.
    tracker.teacher_force_obj_scores_for_mem = False
    tracker.prob_to_dropout_spatial_mem = 0.0
    trainable_names = choose_trainable(tracker, args.variant)
    tracker.train()
    if args.variant != "full":
        tracker.backbone.eval()
    parameters = [p for p in tracker.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config["trainable_parameter_count"] = sum(p.numel() for p in parameters)
    config["trainable_parameter_names"] = trainable_names
    config["selection_rule"] = "fixed final step; no val/test GT checkpoint selection"
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.save_step0:
        save_adapter(
            args.output_dir / "tracker_step000000.pt",
            tracker,
            trainable_names,
            {
                "step": 0,
                "variant": args.variant,
                "mode": args.mode,
                "manifest": str(args.manifest.resolve()),
            },
        )
    log_path = args.output_dir / "train.jsonl"

    started = time.time()
    for step in range(1, args.steps + 1):
        row = rows[(step - 1) % len(rows)]
        optimizer.zero_grad(set_to_none=True)
        if args.mode == "anchor":
            image, target = load_frame(
                row["image_path"],
                row["mask_path"],
                args.image_size,
                device,
                rng,
            )
            box = tight_box(target, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = forward_sequence(tracker, image[None], box)[0]
                loss, terms = segmentation_loss(
                    output["pred_masks_high_res"], target, args.lambda_bce
                )
        else:
            anchor_image, anchor_target = load_frame(
                row["anchor_image_path"],
                row["anchor_mask_path"],
                args.image_size,
                device,
                rng,
            )
            bridge_images = [
                load_frame(path, None, args.image_size, device)[0]
                for path in row["bridge_image_paths"]
            ]
            query_image, pseudo_target = load_frame(
                row["target_image_path"],
                row["pseudo_mask_path"],
                args.image_size,
                device,
                rng,
            )
            images = torch.stack([anchor_image, *bridge_images, query_image])
            box = tight_box(anchor_target, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sequence = forward_sequence(tracker, images, box)
                anchor_loss, anchor_terms = segmentation_loss(
                    sequence[0]["pred_masks_high_res"],
                    anchor_target,
                    args.lambda_bce,
                )
                pseudo_loss, pseudo_terms = segmentation_loss(
                    sequence[-1]["pred_masks_high_res"],
                    pseudo_target,
                    args.lambda_bce,
                )
                # Full vs drop-last-bridge path consistency. The full prediction
                # is the supervised branch; the shorter branch is a stop-gradient target.
                path_loss = torch.zeros((), device=device)
                if bridge_images:
                    with torch.no_grad():
                        drop_images = torch.stack(
                            [anchor_image, *bridge_images[:-1], query_image]
                        )
                        drop_logits = forward_sequence(
                            tracker, drop_images, box
                        )[-1]["pred_masks_high_res"].float()
                    full_prob = sequence[-1]["pred_masks_high_res"].float().sigmoid()
                    drop_prob = drop_logits.sigmoid()
                    inter = (full_prob * drop_prob).flatten(1).sum(1)
                    denom = (
                        full_prob.flatten(1).sum(1)
                        + drop_prob.flatten(1).sum(1)
                    )
                    path_loss = 1.0 - ((2 * inter + 1) / (denom + 1)).mean()
                loss = (
                    anchor_loss
                    + args.lambda_pseudo * pseudo_loss
                    + args.lambda_path * path_loss
                )
                terms = {
                    "anchor_dice_loss": anchor_terms["dice_loss"],
                    "anchor_bce": anchor_terms["bce"],
                    "pseudo_dice_loss": pseudo_terms["dice_loss"],
                    "pseudo_bce": pseudo_terms["bce"],
                    "path_loss": float(path_loss.detach()),
                }
        scaler.scale(loss).backward()
        grouped_grad_norms = module_grad_norms(tracker)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "seconds": round(time.time() - started, 2),
            **grouped_grad_norms,
            **terms,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step == 1 or step % 10 == 0:
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_adapter(
                args.output_dir / f"tracker_step{step:06d}.pt",
                tracker,
                trainable_names,
                {
                    "step": step,
                    "variant": args.variant,
                    "mode": args.mode,
                    "manifest": str(args.manifest.resolve()),
                },
            )
    (args.output_dir / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
