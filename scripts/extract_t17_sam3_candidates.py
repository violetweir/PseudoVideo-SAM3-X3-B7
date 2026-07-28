#!/usr/bin/env python3
"""Extract resumable SAM3 candidate pools from autonomous Qwen text and boxes."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

SOURCE_NAMES = ("qwen_text", "qwen_box", "qwen_text_box")
FALLBACK_TEXT = "selected visual entity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T17_autonomous_target_1pct/protocol/merged_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--qwen-prompts",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T17_autonomous_target_1pct/qwen/all_prompts.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T17_autonomous_target_1pct/sam3_candidates"
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
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--max-candidates-per-source", type=int, default=64)
    parser.add_argument("--relation-resolution", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_id(merged_id: str) -> str:
    return merged_id.replace("::", "__").replace("/", "_")


def load_gt(path: str, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def qwen_box_normalized(row: dict[str, Any]) -> np.ndarray | None:
    if (
        row.get("status") != "success"
        or not row.get("target_present")
        or not row.get("qwen_box_valid")
    ):
        return None
    box = np.asarray(row["box_xyxy_1000"], dtype=np.float32) / 1000.0
    if box.shape != (4,) or box[0] >= box[2] or box[1] >= box[3]:
        return None
    return np.clip(box, 0, 1)


def xyxy_to_cxcywh(box: np.ndarray) -> list[float]:
    return [
        float((box[0] + box[2]) / 2),
        float((box[1] + box[3]) / 2),
        float(box[2] - box[0]),
        float(box[3] - box[1]),
    ]


def collect_state(
    state: dict[str, Any],
    source_id: int,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masks = state["masks"][:, 0].detach().cpu().numpy().astype(bool)
    scores = state["scores"].detach().float().cpu().numpy()
    boxes = state["boxes"].detach().float().cpu().numpy()
    if len(scores) > max_candidates:
        keep = np.argsort(-scores, kind="stable")[:max_candidates]
        masks, scores, boxes = masks[keep], scores[keep], boxes[keep]
    sources = np.full(len(scores), source_id, dtype=np.int8)
    return masks, scores.astype(np.float32), boxes.astype(np.float32), sources


def candidate_geometry(
    masks: np.ndarray,
    boxes_px: np.ndarray,
    scores: np.ndarray,
    source_ids: np.ndarray,
    qbox: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    count, height, width = masks.shape
    columns = [
        "sam_score",
        "source_id",
        "area_ratio",
        "center_x",
        "center_y",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "mask_qbox_iou",
        "bbox_qbox_iou",
        "mask_inside_qbox",
        "center_proximity",
    ]
    features = np.zeros((count, len(columns)), dtype=np.float32)
    features[:, 0] = scores
    features[:, 1] = source_ids
    features[:, 5:9] = boxes_px / np.asarray(
        [width, height, width, height], dtype=np.float32
    )
    qmask = None
    if qbox is not None:
        x1 = int(np.floor(qbox[0] * width))
        y1 = int(np.floor(qbox[1] * height))
        x2 = int(np.ceil(qbox[2] * width))
        y2 = int(np.ceil(qbox[3] * height))
        qmask = np.zeros((height, width), dtype=bool)
        qmask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
        qarea = max(int(qmask.sum()), 1)
        qcenter = np.asarray(
            [(qbox[0] + qbox[2]) / 2, (qbox[1] + qbox[3]) / 2], dtype=np.float32
        )
    for index, mask in enumerate(masks):
        ys, xs = np.where(mask)
        area = int(mask.sum())
        features[index, 2] = area / max(height * width, 1)
        if area:
            center = np.asarray(
                [xs.mean() / max(width - 1, 1), ys.mean() / max(height - 1, 1)],
                dtype=np.float32,
            )
            features[index, 3:5] = center
        else:
            center = np.asarray([0.5, 0.5], dtype=np.float32)
        if qmask is None:
            continue
        intersection = int(np.logical_and(mask, qmask).sum())
        union = area + qarea - intersection
        features[index, 9] = intersection / max(union, 1)
        features[index, 11] = intersection / max(area, 1)
        candidate_box = features[index, 5:9]
        ix1 = max(float(candidate_box[0]), float(qbox[0]))
        iy1 = max(float(candidate_box[1]), float(qbox[1]))
        ix2 = min(float(candidate_box[2]), float(qbox[2]))
        iy2 = min(float(candidate_box[3]), float(qbox[3]))
        inter_box = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
        candidate_area = max(
            float(candidate_box[2] - candidate_box[0])
            * float(candidate_box[3] - candidate_box[1]),
            0,
        )
        qbox_area = float((qbox[2] - qbox[0]) * (qbox[3] - qbox[1]))
        features[index, 10] = inter_box / max(
            candidate_area + qbox_area - inter_box, 1e-12
        )
        distance = float(np.linalg.norm(center - qcenter) / np.sqrt(2))
        features[index, 12] = max(0.0, 1.0 - distance)
    return features, columns


def candidate_metrics(
    masks: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = masks.reshape(len(masks), -1)
    gt_flat = gt.reshape(-1)
    intersection = np.logical_and(flat, gt_flat[None]).sum(axis=1)
    pred_area = flat.sum(axis=1)
    gt_area = np.full(len(masks), int(gt_flat.sum()), dtype=np.int64)
    union = pred_area + gt_area - intersection
    dice = np.divide(
        2 * intersection,
        pred_area + gt_area,
        out=np.zeros(len(masks), dtype=np.float64),
        where=(pred_area + gt_area) > 0,
    )
    iou = np.divide(
        intersection,
        union,
        out=np.zeros(len(masks), dtype=np.float64),
        where=union > 0,
    )
    return (
        dice.astype(np.float32),
        iou.astype(np.float32),
        intersection.astype(np.int32),
        pred_area.astype(np.int32),
        gt_area.astype(np.int32),
    )


def append_index(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    manifest = [
        row for row in read_jsonl(args.manifest.resolve()) if row["split"] == args.split
    ]
    if args.limit > 0:
        manifest = manifest[: args.limit]
    qwen = {row["merged_id"]: row for row in read_jsonl(args.qwen_prompts.resolve())}
    missing_qwen = [row["merged_id"] for row in manifest if row["merged_id"] not in qwen]
    if missing_qwen:
        raise RuntimeError(f"Missing Qwen rows ({len(missing_qwen)}), first={missing_qwen[:3]}")

    split_root = args.output_root.resolve() / args.split
    split_root.mkdir(parents=True, exist_ok=True)
    pending = [
        row
        for row in manifest
        if args.overwrite or not (split_root / f"{safe_id(row['merged_id'])}.npz").exists()
    ]
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"split={args.split} manifest={len(manifest)} pending={len(pending)}")
    print(f"SAM3 checkpoint: {args.checkpoint}")
    if not pending:
        return

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
    )
    processor = Sam3Processor(
        model, confidence_threshold=args.confidence_threshold, device="cuda"
    )
    index_path = args.output_root.resolve() / f"{args.split}_index.jsonl"
    for position, sample in enumerate(pending, 1):
        torch.cuda.empty_cache()
        started = time.time()
        qrow = qwen[sample["merged_id"]]
        qbox = qwen_box_normalized(qrow)
        text_prompt = (
            str(qrow.get("target_description", "")).strip()
            if qrow.get("status") == "success" and qrow.get("target_present")
            else FALLBACK_TEXT
        )
        if not text_prompt:
            text_prompt = FALLBACK_TEXT
        image = Image.open(sample["image_path"]).convert("RGB")
        gt = load_gt(sample["mask_path"], image.size)
        pieces: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            text_state = processor.set_text_prompt(text_prompt, state)
            pieces.append(
                collect_state(text_state, 0, args.max_candidates_per_source)
            )
            processor.reset_all_prompts(state)
            if qbox is not None:
                box_state = processor.add_geometric_prompt(
                    xyxy_to_cxcywh(qbox), True, state
                )
                pieces.append(
                    collect_state(box_state, 1, args.max_candidates_per_source)
                )
                processor.reset_all_prompts(state)
                text_box_state = processor.set_text_prompt(text_prompt, state)
                text_box_state = processor.add_geometric_prompt(
                    xyxy_to_cxcywh(qbox), True, text_box_state
                )
                pieces.append(
                    collect_state(text_box_state, 2, args.max_candidates_per_source)
                )
        nonempty = [piece for piece in pieces if len(piece[1]) > 0]
        if nonempty:
            masks = np.concatenate([piece[0] for piece in nonempty], axis=0)
            scores = np.concatenate([piece[1] for piece in nonempty], axis=0)
            boxes = np.concatenate([piece[2] for piece in nonempty], axis=0)
            source_ids = np.concatenate([piece[3] for piece in nonempty], axis=0)
            features, feature_names = candidate_geometry(
                masks, boxes, scores, source_ids, qbox
            )
            dice, iou, intersection, pred_area, gt_area = candidate_metrics(masks, gt)
            lowres = np.stack(
                [
                    cv2.resize(
                        mask.astype(np.uint8),
                        (args.relation_resolution, args.relation_resolution),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    for mask in masks
                ],
                axis=0,
            )
        else:
            masks = np.zeros((0, image.height, image.width), dtype=bool)
            features = np.zeros((0, 13), dtype=np.float32)
            feature_names = [
                "sam_score", "source_id", "area_ratio", "center_x", "center_y",
                "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "mask_qbox_iou",
                "bbox_qbox_iou", "mask_inside_qbox", "center_proximity",
            ]
            scores = np.zeros(0, dtype=np.float32)
            source_ids = np.zeros(0, dtype=np.int8)
            dice = iou = np.zeros(0, dtype=np.float32)
            intersection = pred_area = gt_area = np.zeros(0, dtype=np.int32)
            lowres = np.zeros(
                (0, args.relation_resolution, args.relation_resolution), dtype=np.uint8
            )
        output_path = split_root / f"{safe_id(sample['merged_id'])}.npz"
        temporary = output_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                merged_id=np.asarray(sample["merged_id"]),
                source_dataset=np.asarray(sample["source_dataset"]),
                split=np.asarray(sample["split"]),
                sample_id=np.asarray(sample["sample_id"]),
                image_path=np.asarray(sample["image_path"]),
                mask_path=np.asarray(sample["mask_path"]),
                qwen_text=np.asarray(text_prompt),
                qwen_box=np.asarray(qbox if qbox is not None else [-1, -1, -1, -1]),
                qwen_status=np.asarray(qrow.get("status", "missing")),
                features=features,
                feature_names=np.asarray(feature_names),
                source_names=np.asarray(SOURCE_NAMES),
                source_id=source_ids,
                sam_score=scores,
                mask_lowres=lowres,
                dice=dice,
                iou=iou,
                intersection=intersection,
                pred_area=pred_area,
                gt_area=gt_area,
            )
        temporary.replace(output_path)
        summary = {
            "merged_id": sample["merged_id"],
            "split": args.split,
            "output_path": str(output_path),
            "candidates": int(len(scores)),
            "source_counts": {
                SOURCE_NAMES[source]: int((source_ids == source).sum())
                for source in range(len(SOURCE_NAMES))
            },
            "native_dice": float(dice[np.argmax(scores)]) if len(scores) else 0.0,
            "oracle_dice": float(dice.max()) if len(dice) else 0.0,
            "seconds": round(time.time() - started, 3),
        }
        append_index(index_path, summary)
        print(
            f"[{position}/{len(pending)}] {sample['merged_id']} "
            f"candidates={len(scores)} native={summary['native_dice']:.4f} "
            f"oracle={summary['oracle_dice']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
