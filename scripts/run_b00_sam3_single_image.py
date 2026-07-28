#!/usr/bin/env python3
"""Category-free single-image SAM3 baseline.

This is the first baseline in the method ladder. It deliberately avoids
support images, pseudo-video memory, Qwen category text, validation tuning, and
student predictions. SAM3 is asked for a generic visual entity and the top
scoring candidate is evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_id(value: str) -> str:
    return value.replace("::", "__").replace("/", "_")


def load_mask(path: str, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    union = pred_area + gt_area - inter
    return {
        "dice": 2 * inter / max(pred_area + gt_area, 1),
        "iou": inter / max(union, 1),
        "precision": inter / max(pred_area, 1),
        "recall": inter / max(gt_area, 1),
        "pred_area_ratio": pred_area / pred.size,
        "gt_area_ratio": gt_area / gt.size,
    }


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt", default="selected visual entity")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.manifest) if row["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]
    output_dir = args.output_root / args.split
    mask_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.jsonl"
    done = set()
    if args.resume and result_path.exists():
        done = {row["merged_id"] for row in read_jsonl(result_path)}

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
    )
    processor = Sam3Processor(
        model, confidence_threshold=args.confidence_threshold, device="cuda"
    )
    for index, row in enumerate(rows, 1):
        if row["merged_id"] in done:
            continue
        started = time.time()
        image = Image.open(row["image_path"]).convert("RGB")
        gt = load_mask(row["mask_path"], image.size)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            output = processor.set_text_prompt(args.prompt, state)
        masks = np.asarray(output["masks"][:, 0].detach().cpu().numpy(), dtype=bool)
        scores = output["scores"].detach().float().cpu().numpy()
        if len(scores):
            chosen = int(np.argmax(scores))
            pred = masks[chosen]
            sam_score = float(scores[chosen])
        else:
            chosen = -1
            pred = np.zeros_like(gt)
            sam_score = 0.0
        mask_path = mask_dir / f"{safe_id(row['merged_id'])}.png"
        Image.fromarray(pred.astype(np.uint8) * 255).save(mask_path)
        result = {
            "merged_id": row["merged_id"],
            "split": args.split,
            "source_dataset": row["source_dataset"],
            "image_path": row["image_path"],
            "mask_path_evaluation_only": row["mask_path"],
            "prediction_path": str(mask_path.resolve()),
            "prompt": args.prompt,
            "candidate_count": int(len(scores)),
            "selected_candidate_index": chosen,
            "sam_score": sam_score,
            "seconds": round(time.time() - started, 3),
            **metrics(pred, gt),
        }
        append(result_path, result)
        print(
            f"[{index}/{len(rows)}] {row['merged_id']} "
            f"dice={result['dice']:.4f} candidates={len(scores)}",
            flush=True,
        )
    (output_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
