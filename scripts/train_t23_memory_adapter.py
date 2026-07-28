#!/usr/bin/env python3
"""Train a zero-residual SAM3 memory-read adapter with protected supervision."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sam3_memory_adapter import attach_memory_read_adapter
from sam3_memory_write_modes import replace_output_memory
from train_t22_sam3_tracker import (
    backbone_features,
    load_base_tracker,
    load_frame,
    read_jsonl,
    segmentation_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-video-manifest", type=Path, required=True)
    parser.add_argument("--gt-probe-manifest", type=Path, required=True)
    parser.add_argument("--route-probe-manifest", type=Path, default=None)
    parser.add_argument("--pseudo-manifest", type=Path, default=None)
    parser.add_argument("--preserve-manifest", type=Path, default=None)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--memory-write-mode",
        choices=("soft", "hard_detach", "hard_ste", "gt_teacher"),
        default="hard_ste",
    )
    parser.add_argument("--adapter-reduction", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--gt-videos-per-step", type=int, default=2)
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-pseudo", type=float, default=0.0)
    parser.add_argument("--pseudo-warmup-steps", type=int, default=120)
    parser.add_argument("--lambda-mask", type=float, default=0.0)
    parser.add_argument("--lambda-mem", type=float, default=0.0)
    parser.add_argument("--lambda-ptr", type=float, default=0.0)
    parser.add_argument("--lambda-obj", type=float, default=0.0)
    parser.add_argument("--lambda-iou", type=float, default=0.0)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--probe-every", type=int, default=100)
    parser.add_argument("--disable-probes", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--save-step0", action="store_true")
    return parser.parse_args()


def tight_box(mask: torch.Tensor) -> torch.Tensor:
    ys, xs = torch.where(mask[0] > 0.5)
    if len(xs) == 0:
        raise RuntimeError("Empty prompt mask")
    return torch.tensor(
        [
            [
                [float(xs.min().item()), float(ys.min().item())],
                [float(xs.max().item()), float(ys.max().item())],
            ]
        ],
        dtype=torch.float32,
        device=mask.device,
    )


def load_video(
    row: dict[str, Any], size: int, device: torch.device
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    images, masks = [], []
    for image_path, mask_path in zip(row["image_paths"], row["mask_paths"]):
        image, mask = load_frame(image_path, mask_path, size, device)
        images.append(image)
        masks.append(mask)
    return torch.stack(images), masks


def load_pseudo_video(
    row: dict[str, Any], size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_paths = [
        row["anchor_image_path"],
        *row["bridge_image_paths"],
        row["target_image_path"],
    ]
    images = [load_frame(path, None, size, device)[0] for path in image_paths]
    _, anchor_mask = load_frame(
        row["anchor_image_path"], row["anchor_mask_path"], size, device
    )
    _, pseudo_mask = load_frame(
        row["target_image_path"], row["pseudo_mask_path"], size, device
    )
    return torch.stack(images), anchor_mask, pseudo_mask


def forward_sequence(
    tracker: torch.nn.Module,
    images: torch.Tensor,
    anchor_box: torch.Tensor | None,
    *,
    memory_write_mode: str,
    gt_masks: list[torch.Tensor] | None = None,
    initial_mask: torch.Tensor | None = None,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, float]]]:
    feats, positions, sizes = backbone_features(tracker, images)
    output_dict: dict[str, dict[int, dict[str, torch.Tensor]]] = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    outputs, stats = [], []
    for index in range(images.shape[0]):
        current_feats = [item[:, index : index + 1] for item in feats]
        current_positions = [item[:, index : index + 1] for item in positions]
        point_inputs = None
        mask_inputs = None
        if index == 0 and initial_mask is not None:
            mask_inputs = initial_mask[None] if initial_mask.ndim == 3 else initial_mask
        elif index == 0:
            point_inputs = {
                "point_coords": anchor_box,
                "point_labels": torch.tensor(
                    [[2, 3]], dtype=torch.int32, device=images.device
                ),
            }
        output = tracker.track_step(
            frame_idx=index,
            is_init_cond_frame=index == 0,
            current_vision_feats=current_feats,
            current_vision_pos_embeds=current_positions,
            feat_sizes=sizes,
            image=images[index : index + 1],
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=images.shape[0],
            run_mem_encoder=False,
        )
        gt_mask = gt_masks[index] if gt_masks is not None else None
        frame_stats = replace_output_memory(
            tracker,
            output,
            image=images[index : index + 1],
            current_vision_feats=current_feats,
            feat_sizes=sizes,
            mode=memory_write_mode,
            gt_mask=gt_mask,
        )
        bucket = "cond_frame_outputs" if index == 0 else "non_cond_frame_outputs"
        output_dict[bucket][index] = output
        outputs.append(output)
        stats.append(frame_stats)
    return outputs, stats


def soft_dice_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (probability * target).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
    return 1 - ((2 * intersection + 1) / (denominator + 1)).mean()


def distillation_loss(
    student: list[dict[str, torch.Tensor]],
    teacher: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = student[0]["pred_masks_high_res"].device
    totals = {
        "mask": torch.zeros((), device=device),
        "mem": torch.zeros((), device=device),
        "ptr": torch.zeros((), device=device),
        "obj": torch.zeros((), device=device),
        "iou": torch.zeros((), device=device),
    }
    for student_out, teacher_out in zip(student, teacher):
        teacher_probability = teacher_out["pred_masks_high_res"].float().sigmoid()
        student_logits = student_out["pred_masks_high_res"].float()
        # A soft Dice loss is not stationary when its two soft inputs are
        # identical.  At the zero adapter it exactly opposed the supervised
        # GT gradient for lambda_mask=1 and locked the adapter at zero.  Use a
        # proper teacher-student divergence whose value and gradient both
        # vanish at equality.
        eps = torch.finfo(teacher_probability.dtype).eps
        teacher_probability_safe = teacher_probability.clamp(eps, 1 - eps)
        teacher_entropy = -(
            teacher_probability
            * teacher_probability_safe.log()
            + (1 - teacher_probability)
            * (1 - teacher_probability_safe).log()
        ).mean()
        relative_bce = (
            F.binary_cross_entropy_with_logits(
                student_logits, teacher_probability
            )
            - teacher_entropy
        )
        probability_mse = F.mse_loss(
            student_logits.sigmoid(), teacher_probability
        )
        totals["mask"] += relative_bce + probability_mse
        student_mem = F.normalize(
            student_out["maskmem_features"].float(), dim=1
        )
        teacher_mem = F.normalize(
            teacher_out["maskmem_features"].float(), dim=1
        )
        totals["mem"] += F.mse_loss(student_mem, teacher_mem)
        totals["ptr"] += (
            1
            - F.cosine_similarity(
                student_out["obj_ptr"].float(),
                teacher_out["obj_ptr"].float(),
                dim=-1,
            ).mean()
        )
        totals["obj"] += F.mse_loss(
            student_out["object_score_logits"].float(),
            teacher_out["object_score_logits"].float(),
        )
        totals["iou"] += F.mse_loss(
            student_out["iou_score"].float(), teacher_out["iou_score"].float()
        )
    for name in totals:
        totals[name] /= len(student)
    loss = (
        args.lambda_mask * totals["mask"]
        + args.lambda_mem * totals["mem"]
        + args.lambda_ptr * totals["ptr"]
        + args.lambda_obj * totals["obj"]
        + args.lambda_iou * totals["iou"]
    )
    return loss, {f"distill_{name}": float(value.detach()) for name, value in totals.items()}


def detach_outputs(outputs: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    keys = (
        "pred_masks_high_res",
        "maskmem_features",
        "obj_ptr",
        "object_score_logits",
        "iou_score",
    )
    return [{key: row[key].detach() for key in keys} for row in outputs]


def dice_binary(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.bool(), b.bool()
    denominator = int(a.sum() + b.sum())
    if denominator == 0:
        return 1.0
    return float(2 * (a & b).sum() / denominator)


@torch.no_grad()
def run_probe(
    tracker: torch.nn.Module,
    adapter: torch.nn.Module,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    student_nonempty, teacher_nonempty, student_areas, teacher_areas = [], [], [], []
    teacher_dices, q_student, q_teacher = [], [], []
    pointer_cosines, memory_distances, objectness, sam_scores = [], [], [], []
    for row in rows:
        if "image_paths" in row:
            images, masks = load_video(row, args.image_size, device)
            anchor_gt = masks[0] > 0.5
        else:
            images, anchor_mask, _ = load_pseudo_video(
                row, args.image_size, device
            )
            masks = None
            anchor_gt = anchor_mask > 0.5
        box = tight_box(anchor_gt.float())
        adapter.enabled = False
        teacher, _ = forward_sequence(
            tracker,
            images,
            box,
            memory_write_mode="hard_detach",
            gt_masks=masks,
        )
        adapter.enabled = True
        student, stats = forward_sequence(
            tracker,
            images,
            box,
            memory_write_mode="hard_detach",
            gt_masks=masks,
        )
        teacher_mask = teacher[-1]["pred_masks_high_res"].sigmoid() > 0.5
        student_mask = student[-1]["pred_masks_high_res"].sigmoid() > 0.5
        student_nonempty.append(bool(student_mask.any()))
        teacher_nonempty.append(bool(teacher_mask.any()))
        student_areas.append(float(student_mask.float().mean()))
        teacher_areas.append(float(teacher_mask.float().mean()))
        teacher_dices.append(dice_binary(student_mask, teacher_mask))
        pointer_cosines.append(
            float(
                F.cosine_similarity(
                    student[-1]["obj_ptr"].float(),
                    teacher[-1]["obj_ptr"].float(),
                    dim=-1,
                ).mean()
            )
        )
        memory_distances.append(
            float(
                F.mse_loss(
                    F.normalize(student[-1]["maskmem_features"].float(), dim=1),
                    F.normalize(teacher[-1]["maskmem_features"].float(), dim=1),
                )
            )
        )
        objectness.append(stats[-1]["objectness"])
        sam_scores.append(stats[-1]["sam_score"])
        # Reverse from the predicted terminal mask and compare the returned anchor.
        reversed_images = images.flip(0)
        reversed_student, _ = forward_sequence(
            tracker,
            reversed_images,
            None,
            memory_write_mode="hard_detach",
            initial_mask=student_mask.float()[0],
        )
        adapter.enabled = False
        reversed_teacher, _ = forward_sequence(
            tracker,
            reversed_images,
            None,
            memory_write_mode="hard_detach",
            initial_mask=teacher_mask.float()[0],
        )
        adapter.enabled = True
        q_student.append(
            dice_binary(
                reversed_student[-1]["pred_masks_high_res"].sigmoid() > 0.5,
                anchor_gt,
            )
        )
        q_teacher.append(
            dice_binary(
                reversed_teacher[-1]["pred_masks_high_res"].sigmoid() > 0.5,
                anchor_gt,
            )
        )
    frozen_nonempty = float(np.mean(teacher_nonempty))
    return {
        "probe_nonempty_rate": float(np.mean(student_nonempty)),
        "frozen_nonempty_rate": frozen_nonempty,
        "probe_area_ratio": float(
            np.mean(student_areas) / max(np.mean(teacher_areas), 1e-12)
        ),
        "probe_teacher_dice": float(np.mean(teacher_dices)),
        "probe_q_return": float(np.mean(q_student)),
        "frozen_q_return": float(np.mean(q_teacher)),
        "probe_q_return_ratio": float(
            np.mean(q_student) / max(np.mean(q_teacher), 1e-12)
        ),
        "probe_objectness": float(np.mean(objectness)),
        "probe_sam_score": float(np.mean(sam_scores)),
        "probe_pointer_cosine": float(np.mean(pointer_cosines)),
        "probe_memory_feature_distance": float(np.mean(memory_distances)),
    }


def save_adapter(
    path: Path,
    adapter: torch.nn.Module,
    args: argparse.Namespace,
    step: int,
    health: dict[str, float] | None,
) -> None:
    torch.save(
        {
            "memory_adapter_state_dict": {
                name: value.detach().cpu()
                for name, value in adapter.state_dict().items()
            },
            "memory_adapter_config": {
                "reduction": args.adapter_reduction,
                "zero_init": "up projection zero, gamma one",
            },
            "step": step,
            "memory_write_mode": args.memory_write_mode,
            "lambda_pseudo": args.lambda_pseudo,
            "probe_health": health,
            "base_sam3_frozen": True,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda")
    gt_rows = read_jsonl(args.gt_video_manifest)
    gt_probe_rows = read_jsonl(args.gt_probe_manifest)
    route_probe_rows = (
        read_jsonl(args.route_probe_manifest)
        if args.route_probe_manifest
        else []
    )
    probe_rows = [*gt_probe_rows, *route_probe_rows]
    pseudo_rows = read_jsonl(args.pseudo_manifest) if args.pseudo_manifest else []
    preserve_rows = (
        read_jsonl(args.preserve_manifest) if args.preserve_manifest else pseudo_rows
    )
    tracker = load_base_tracker(args.base_checkpoint, device)
    tracker.eval()
    for parameter in tracker.parameters():
        parameter.requires_grad_(False)
    adapter = attach_memory_read_adapter(tracker, args.adapter_reduction).to(device)
    if args.init_adapter:
        payload = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
        adapter.load_state_dict(payload["memory_adapter_state_dict"], strict=True)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    use_frozen_teacher = bool(
        args.lambda_mask
        or args.lambda_mem
        or args.lambda_ptr
        or args.lambda_obj
        or args.lambda_iou
    )
    teacher_tracker = None
    if use_frozen_teacher:
        # Keep a physically separate, permanently frozen teacher. Reusing the
        # student tracker for a no-grad teacher pass can leave SAM3's internal
        # execution state detached and silently zero the following adapter
        # gradients.
        teacher_tracker = load_base_tracker(args.base_checkpoint, device)
        teacher_tracker.eval()
        for parameter in teacher_tracker.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        trainable_parameters=sum(p.numel() for p in adapter.parameters()),
        frozen_base_parameters=sum(p.numel() for p in tracker.parameters())
        - sum(p.numel() for p in adapter.parameters()),
        checkpoint_rule="fixed final step or train-only safety stop; never test-selected",
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    if args.save_step0:
        save_adapter(
            args.output_dir / "memory_adapter_step000000.pt",
            adapter,
            args,
            0,
            None,
        )
    log_path = args.output_dir / "train.jsonl"
    probe_path = args.output_dir / "probes.jsonl"
    consecutive_failures = 0
    final_health = None
    started = time.time()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        aggregate: dict[str, float] = {}
        # DataLoader-equivalent fixed 2:1:1 composition is explicit per step.
        for offset in range(args.gt_videos_per_step):
            row = gt_rows[((step - 1) * args.gt_videos_per_step + offset) % len(gt_rows)]
            images, masks = load_video(row, args.image_size, device)
            box = tight_box(masks[0])
            teacher = None
            if use_frozen_teacher:
                assert teacher_tracker is not None
                with torch.no_grad():
                    teacher_raw, _ = forward_sequence(
                        teacher_tracker,
                        images,
                        box,
                        memory_write_mode="hard_detach",
                        gt_masks=masks,
                    )
                    teacher = detach_outputs(teacher_raw)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student, frame_stats = forward_sequence(
                    tracker,
                    images,
                    box,
                    memory_write_mode=args.memory_write_mode,
                    gt_masks=masks,
                )
                gt_loss = torch.zeros((), device=device)
                for frame_index, (output, target) in enumerate(zip(student, masks)):
                    frame_loss, terms = segmentation_loss(
                        output["pred_masks_high_res"], target, args.lambda_bce
                    )
                    gt_loss += frame_loss / len(student)
                    aggregate[f"gt_frame{frame_index}_dice"] = terms["dice_loss"]
                    aggregate[f"gt_frame{frame_index}_bce"] = terms["bce"]
                    for stat_name, stat_value in frame_stats[frame_index].items():
                        aggregate[f"frame{frame_index}_{stat_name}"] = stat_value
                    aggregate[f"frame{frame_index}_nonempty"] = float(
                        frame_stats[frame_index]["hard_area"] > 0
                    )
                total_loss = total_loss + gt_loss / args.gt_videos_per_step
                aggregate["gt_video_loss"] = aggregate.get("gt_video_loss", 0) + float(
                    gt_loss.detach()
                ) / args.gt_videos_per_step
                if teacher is not None and offset == 0:
                    preserve_loss, preserve_terms = distillation_loss(
                        student, teacher, args
                    )
                    total_loss = total_loss + preserve_loss
                    aggregate.update(preserve_terms)

        if pseudo_rows:
            row = pseudo_rows[(step - 1) % len(pseudo_rows)]
            images, anchor_mask, pseudo_mask = load_pseudo_video(
                row, args.image_size, device
            )
            box = tight_box(anchor_mask)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs, _ = forward_sequence(
                    tracker,
                    images,
                    box,
                    memory_write_mode="hard_ste",
                )
                pseudo_loss, pseudo_terms = segmentation_loss(
                    outputs[-1]["pred_masks_high_res"],
                    pseudo_mask,
                    args.lambda_bce,
                )
                ramp = min(1.0, step / max(1, args.pseudo_warmup_steps))
                pseudo_weight = args.lambda_pseudo * ramp
                total_loss = total_loss + pseudo_weight * pseudo_loss
                aggregate.update(
                    pseudo_loss=float(pseudo_loss.detach()),
                    pseudo_weight=pseudo_weight,
                    pseudo_dice=pseudo_terms["dice_loss"],
                    pseudo_bce=pseudo_terms["bce"],
                )

        if preserve_rows and (
            use_frozen_teacher
        ):
            assert teacher_tracker is not None
            row = preserve_rows[-step % len(preserve_rows)]
            images, anchor_mask, _ = load_pseudo_video(row, args.image_size, device)
            box = tight_box(anchor_mask)
            with torch.no_grad():
                teacher_raw, _ = forward_sequence(
                    teacher_tracker,
                    images,
                    box,
                    memory_write_mode="hard_detach",
                )
                teacher = detach_outputs(teacher_raw)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student, _ = forward_sequence(
                    tracker,
                    images,
                    box,
                    memory_write_mode="hard_ste",
                )
                preserve_loss, preserve_terms = distillation_loss(
                    student, teacher, args
                )
                total_loss = total_loss + preserve_loss
                aggregate.update(
                    {f"route_{key}": value for key, value in preserve_terms.items()}
                )

        residual = adapter.last_residual
        if residual is not None:
            reg = residual.float().pow(2).mean()
            total_loss = total_loss + args.lambda_reg * reg
            aggregate["adapter_reg"] = float(reg.detach())
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        record = {
            "step": step,
            "loss": float(total_loss.detach()),
            "grad_norm": float(grad_norm),
            "seconds": round(time.time() - started, 2),
            **aggregate,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step == 1 or step % 10 == 0:
            print(json.dumps(record), flush=True)
        if not args.disable_probes and (
            step % args.probe_every == 0 or step == args.steps
        ):
            final_health = run_probe(tracker, adapter, probe_rows, args, device)
            probe_record = {"step": step, **final_health}
            with probe_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(probe_record, sort_keys=True) + "\n")
            print("PROBE " + json.dumps(probe_record), flush=True)
            failed = (
                final_health["probe_nonempty_rate"]
                < 0.9 * final_health["frozen_nonempty_rate"]
                or final_health["probe_area_ratio"] < 0.80
                or final_health["probe_teacher_dice"] < 0.85
                or final_health["probe_q_return_ratio"] < 0.80
            )
            consecutive_failures = consecutive_failures + 1 if failed else 0
            if consecutive_failures >= 3:
                save_adapter(
                    args.output_dir / f"memory_adapter_safety_stop_step{step:06d}.pt",
                    adapter,
                    args,
                    step,
                    final_health,
                )
                (args.output_dir / "SAFETY_STOP").write_text(
                    f"step={step}\n", encoding="utf-8"
                )
                return
        if step % args.save_every == 0 or step == args.steps:
            save_adapter(
                args.output_dir / f"memory_adapter_step{step:06d}.pt",
                adapter,
                args,
                step,
                final_health,
            )
    (args.output_dir / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
