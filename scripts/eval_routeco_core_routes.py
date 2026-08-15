#!/usr/bin/env python3
"""Evaluate RouteCo checkpoints on frozen Stage1 routes with tracker-core forward."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sam3_lora import attach_mask_decoder_lora, load_lora_state_dict
from sam3_memory_adapter import attach_memory_read_adapter
from train_t22_sam3_tracker import load_base_tracker, load_frame
from train_t23_memory_adapter import forward_sequence


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
        handle.flush()


def dice_np(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = int(a.sum() + b.sum())
    return 1.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


def iou_np(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def load_mask_np(path: str, size: int) -> np.ndarray:
    return (
        np.asarray(
            Image.open(path)
            .convert("L")
            .resize((size, size), Image.Resampling.NEAREST)
        )
        > 127
    )


def normalized_xywh_to_xyxy(box: list[float], size: int, device: torch.device) -> torch.Tensor:
    x, y, w, h = [float(v) for v in box]
    return torch.tensor(
        [[[x * size, y * size], [(x + w) * size, (y + h) * size]]],
        dtype=torch.float32,
        device=device,
    )


def load_routeco_checkpoint(
    tracker: torch.nn.Module,
    checkpoint: Path,
    *,
    adapter_reduction: int | None = None,
) -> tuple[torch.nn.Module, list[str]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("memory_adapter_config", {})
    reduction = adapter_reduction or int(config.get("reduction", 4))
    adapter = attach_memory_read_adapter(tracker, reduction=reduction).to(next(tracker.parameters()).device)
    adapter.load_state_dict(payload["memory_adapter_state_dict"], strict=True)
    lora_config = payload.get("mask_decoder_lora_config", {})
    wrapped = attach_mask_decoder_lora(
        tracker,
        rank=int(lora_config.get("rank", 4)),
        alpha=float(lora_config.get("alpha", 8.0)),
        dropout=float(lora_config.get("dropout", 0.0)),
        include_conv1x1=bool(lora_config.get("include_conv1x1", True)),
    )
    load_lora_state_dict(tracker, payload["mask_decoder_lora_state_dict"])
    return adapter, wrapped


def route_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (row["target_id"], int(row["bridge_count"]), row["route_id"])


@torch.no_grad()
def evaluate_route(
    tracker: torch.nn.Module,
    route: dict[str, Any],
    image_size: int,
    device: torch.device,
) -> dict[str, Any]:
    image_paths = [
        route["anchor_image_path"],
        *route["bridge_image_paths"],
        route["target_image_path"],
    ]
    images = torch.stack(
        [load_frame(path, None, image_size, device)[0] for path in image_paths]
    )
    box = normalized_xywh_to_xyxy(
        route["anchor_box_xywh_normalized"], image_size, device
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs, _ = forward_sequence(
            tracker,
            images,
            box,
            memory_write_mode="hard_detach",
        )
    prob = outputs[-1]["pred_masks_high_res"].float().sigmoid()[0, 0]
    mask = (prob > 0.5).detach().cpu().numpy().astype(bool)
    gt = load_mask_np(route["target_mask_path_evaluation_only"], image_size)
    return {
        "dice": dice_np(mask, gt),
        "iou": iou_np(mask, gt),
        "area": float(mask.mean()),
        "object_probability": float(outputs[-1]["object_score_logits"].float().sigmoid().mean()),
        "iou_score": float(outputs[-1]["iou_score"].float().mean()),
        "mask": mask,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_route: dict[str, list[dict[str, Any]]] = {}
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_route.setdefault(row["route_type"], []).append(row)
        by_target.setdefault(row["target_id"], []).append(row)
    route_summary = {
        key: {
            "n": len(items),
            "dice": float(np.mean([row["gt_dice_evaluation_only"] for row in items])),
            "iou": float(np.mean([row["gt_iou_evaluation_only"] for row in items])),
        }
        for key, items in sorted(by_route.items())
    }
    oracle = [
        max(items, key=lambda row: row["gt_dice_evaluation_only"])
        for items in by_target.values()
    ]
    shortest = [
        min(items, key=lambda row: (int(row["bridge_count"]), row["route_id"]))
        for items in by_target.values()
    ]
    return {
        "n_routes": len(rows),
        "n_targets": len(by_target),
        "route_summary": route_summary,
        "oracle_dice": float(np.mean([row["gt_dice_evaluation_only"] for row in oracle])),
        "shortest_dice": float(np.mean([row["gt_dice_evaluation_only"] for row in shortest])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--routeco-checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-name", default="eval_routeco_core")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    mode_root = args.root / args.mode
    routes = sorted(
        read_jsonl(mode_root / f"{args.split}_pool0_stage1/routes.jsonl"),
        key=route_sort_key,
    )
    routes = [row for row in routes if 3 <= int(row["bridge_count"]) <= 6]
    if args.limit:
        routes = routes[: args.limit]
    eval_root = mode_root / f"{args.output_name}_{args.split}"
    result_path = eval_root / "route_results.jsonl"
    old_rows = read_jsonl(result_path) if args.resume else []
    existing = {
        row["route_id"]: row
        for row in old_rows
        if row.get("status") == "success"
    }
    pending = [row for row in routes if row["route_id"] not in existing]
    mask_root = eval_root / "forward_masks"
    mask_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    tracker = load_base_tracker(args.base_checkpoint, device)
    load_routeco_checkpoint(tracker, args.routeco_checkpoint)
    tracker.eval()
    for parameter in tracker.parameters():
        parameter.requires_grad_(False)

    for position, route in enumerate(pending, 1):
        started = time.time()
        metric = evaluate_route(tracker, route, args.image_size, device)
        mask_path = mask_root / f"{route['route_id']}.png"
        Image.fromarray(metric.pop("mask").astype(np.uint8) * 255).save(mask_path)
        row = {
            **route,
            "status": "success",
            "forward_mask_path": str(mask_path),
            "seconds": round(time.time() - started, 3),
            "gt_dice_evaluation_only": metric["dice"],
            "gt_iou_evaluation_only": metric["iou"],
            "forward_area": metric["area"],
            "forward_object_probability": metric["object_probability"],
            "forward_iou_score": metric["iou_score"],
        }
        append_jsonl(result_path, row)
        existing[row["route_id"]] = row
        print(
            f"[{position}/{len(pending)}] {route['target_id']} "
            f"{route['route_type']} dice={metric['dice']:.4f}",
            flush=True,
        )

    final_rows = [existing[row["route_id"]] for row in routes if row["route_id"] in existing]
    write_jsonl(result_path, final_rows)
    summary = summarize(final_rows)
    (eval_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"mode": args.mode, "split": args.split, **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
