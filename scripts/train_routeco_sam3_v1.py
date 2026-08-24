#!/usr/bin/env python3
"""RouteCo-SAM3 v1 training.

The v1 contract is intentionally narrow: frozen KNN/routes, frozen SAM3 base
weights, trainable memory-read adapter plus mask-decoder LoRA, route-uncertainty
gated pseudo supervision, frozen-prior distillation, and anchor-cycle loss.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sam3_lora import attach_mask_decoder_lora, load_lora_state_dict, lora_state_dict
from sam3_memory_adapter import attach_memory_read_adapter
from train_t22_sam3_tracker import load_base_tracker, read_jsonl, segmentation_loss
from train_t23_memory_adapter import (
    detach_outputs,
    distillation_loss,
    forward_sequence,
    load_pseudo_video,
    soft_dice_loss,
    tight_box,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-routeco", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--adapter-reduction", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--no-lora-conv1x1", action="store_true")
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-pseudo", type=float, default=0.5)
    parser.add_argument("--lambda-cycle", type=float, default=0.1)
    parser.add_argument("--lambda-prior-mask", type=float, default=0.1)
    parser.add_argument("--lambda-prior-mem", type=float, default=0.05)
    parser.add_argument("--lambda-prior-ptr", type=float, default=0.05)
    parser.add_argument("--lambda-prior-obj", type=float, default=0.0)
    parser.add_argument("--lambda-prior-iou", type=float, default=0.0)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--gate-temperature", type=float, default=0.05)
    parser.add_argument("--student-temperature", type=float, default=0.05)
    parser.add_argument("--min-gate", type=float, default=0.15)
    parser.add_argument("--pseudo-warmup-steps", type=int, default=120)
    parser.add_argument("--no-student-gate-pseudo", action="store_true",
                        help="do not multiply pseudo loss by the student gate even for consensus samples")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--save-step0", action="store_true")
    return parser.parse_args()


def gate_from_variance(value: Any, temperature: float, minimum: float) -> float:
    if value is None:
        variance = 0.0
    else:
        try:
            variance = float(value)
        except (TypeError, ValueError):
            variance = 0.0
    if not math.isfinite(variance):
        variance = 0.0
    gate = math.exp(-max(0.0, variance) / max(temperature, 1e-8))
    return float(min(1.0, max(minimum, gate)))


def row_gates(row: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    route_uncertainty = row.get("routeco_uncertainty")
    route_disagreement = 0.0
    if isinstance(route_uncertainty, dict):
        route_disagreement = float(
            route_uncertainty.get("route_selected_mean_disagreement") or 0.0
        )
        route_uncertainty = route_uncertainty.get("route_mask_mean_variance", 0.0)
    if route_uncertainty is None:
        route_uncertainty = row.get("route_mask_mean_variance", 0.0)
    student_variance = row.get("student_weak_strong_variance")
    if student_variance is None:
        student_variance = row.get("student_aug_variance", 0.0)
    return {
        "gate_t_to_s": gate_from_variance(
            route_uncertainty, args.gate_temperature, args.min_gate
        ),
        "gate_s_to_t": gate_from_variance(
            student_variance, args.student_temperature, args.min_gate
        ),
        "student_feedback": bool(row.get("pseudo_consensus_path"))
            or row.get("sample_type") in ("tier_a", "tier_b"),
        "route_uncertainty": float(route_uncertainty or 0.0),
        "route_disagreement": float(route_disagreement),
        "student_variance": float(student_variance or 0.0),
    }


def trainable_parameters(tracker: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[str]]:
    params, names = [], []
    for name, parameter in tracker.named_parameters():
        enabled = (
            name.startswith("t23_memory_read_adapter.")
            or ".lora_a." in name
            or ".lora_b." in name
        )
        parameter.requires_grad_(enabled)
        if enabled:
            params.append(parameter)
            names.append(name)
    if not params:
        raise RuntimeError("No RouteCo trainable parameters found")
    return params, names


def load_routeco_checkpoint(
    tracker: torch.nn.Module, adapter: torch.nn.Module, checkpoint: Path
) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "memory_adapter_state_dict" in payload:
        adapter.load_state_dict(payload["memory_adapter_state_dict"], strict=True)
    if "mask_decoder_lora_state_dict" in payload:
        load_lora_state_dict(tracker, payload["mask_decoder_lora_state_dict"])


def save_routeco(
    path: Path,
    tracker: torch.nn.Module,
    adapter: torch.nn.Module,
    args: argparse.Namespace,
    step: int,
    trainable_names: list[str],
    lora_modules: list[str],
) -> None:
    torch.save(
        {
            "memory_adapter_state_dict": {
                name: value.detach().cpu()
                for name, value in adapter.state_dict().items()
            },
            "memory_adapter_config": {"reduction": args.adapter_reduction},
            "mask_decoder_lora_state_dict": lora_state_dict(tracker),
            "mask_decoder_lora_config": {
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "include_conv1x1": not args.no_lora_conv1x1,
                "wrapped_modules": lora_modules,
            },
            "trainable_parameter_names": trainable_names,
            "step": step,
            "base_sam3_frozen": True,
            "routeco_v1": True,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    rows = read_jsonl(args.pseudo_manifest)
    if not rows:
        raise RuntimeError("Empty RouteCo pseudo manifest")

    tracker = load_base_tracker(args.base_checkpoint, device)
    tracker.eval()
    for parameter in tracker.parameters():
        parameter.requires_grad_(False)
    adapter = attach_memory_read_adapter(tracker, args.adapter_reduction).to(device)
    lora_modules = attach_mask_decoder_lora(
        tracker,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        include_conv1x1=not args.no_lora_conv1x1,
    )
    if args.init_routeco:
        load_routeco_checkpoint(tracker, adapter, args.init_routeco)
    parameters, trainable_names = trainable_parameters(tracker)

    teacher_tracker = load_base_tracker(args.base_checkpoint, device)
    teacher_tracker.eval()
    for parameter in teacher_tracker.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        trainable_parameter_count=sum(p.numel() for p in parameters),
        trainable_parameter_names=trainable_names,
        lora_wrapped_modules=lora_modules,
        frozen_components=[
            "KNN graph",
            "b3-b6 routes",
            "SAM3 image encoder",
            "SAM3 base memory stack",
            "SAM3 base mask decoder weights",
        ],
        deferred_components=[
            "dynamic KNN rebuild",
            "spatial memory gate",
            "complex DPO",
            "full encoder finetune",
            "same-iteration bidirectional backprop",
        ],
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    if args.save_step0:
        save_routeco(
            args.output_dir / "routeco_step000000.pt",
            tracker,
            adapter,
            args,
            0,
            trainable_names,
            lora_modules,
        )

    log_path = args.output_dir / "train.jsonl"
    started = time.time()
    for step in range(1, args.steps + 1):
        row = rows[(step - 1) % len(rows)]
        gates = row_gates(row, args)
        optimizer.zero_grad(set_to_none=True)
        images, anchor_mask, pseudo_mask = load_pseudo_video(
            row, args.image_size, device
        )
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
            prior_loss, prior_terms = distillation_loss(
                outputs,
                teacher,
                argparse.Namespace(
                    lambda_mask=args.lambda_prior_mask,
                    lambda_mem=args.lambda_prior_mem,
                    lambda_ptr=args.lambda_prior_ptr,
                    lambda_obj=args.lambda_prior_obj,
                    lambda_iou=args.lambda_prior_iou,
                ),
            )
            terminal_mask = outputs[-1]["pred_masks_high_res"].float().sigmoid()
            reverse_outputs, _ = forward_sequence(
                tracker,
                images.flip(0),
                None,
                memory_write_mode="hard_ste",
                initial_mask=terminal_mask.detach()[0],
            )
            cycle_loss = soft_dice_loss(
                reverse_outputs[-1]["pred_masks_high_res"].float().sigmoid(),
                anchor_mask,
            )
            ramp = min(1.0, step / max(1, args.pseudo_warmup_steps))
            pseudo_student_gate = (
                gates["gate_s_to_t"]
                if (not args.no_student_gate_pseudo and gates["student_feedback"])
                else 1.0
            )
            loss = (
                args.lambda_pseudo * ramp * gates["gate_t_to_s"] * pseudo_student_gate * pseudo_loss
                + gates["gate_s_to_t"] * prior_loss
                + args.lambda_cycle * gates["gate_s_to_t"] * cycle_loss
            )
            residual = adapter.last_residual
            if residual is not None:
                reg = residual.float().pow(2).mean()
                loss = loss + args.lambda_reg * reg
            else:
                reg = torch.zeros((), device=device)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
            "seconds": round(time.time() - started, 2),
            "pseudo_loss": float(pseudo_loss.detach()),
            "pseudo_weight": args.lambda_pseudo * ramp * gates["gate_t_to_s"] * pseudo_student_gate,
            "pseudo_student_gate": float(pseudo_student_gate),
            "cycle_loss": float(cycle_loss.detach()),
            "cycle_weight": args.lambda_cycle * gates["gate_s_to_t"],
            "prior_loss": float(prior_loss.detach()),
            "prior_weight": gates["gate_s_to_t"],
            "adapter_reg": float(reg.detach()),
            **gates,
            **{f"pseudo_{key}": value for key, value in pseudo_terms.items()},
            **{f"prior_{key}": value for key, value in prior_terms.items()},
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if step == 1 or step % 10 == 0:
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_routeco(
                args.output_dir / f"routeco_step{step:06d}.pt",
                tracker,
                adapter,
                args,
                step,
                trainable_names,
                lora_modules,
            )
    (args.output_dir / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
