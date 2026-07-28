#!/usr/bin/env python3
"""Evaluate SAM3 frame-0 visual-box memory on frozen train-only pseudo-video pairs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-manifest",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T18_retrieval_pseudovideo/protocol/pilot_retrieval_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T18_retrieval_pseudovideo/pilot"
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_fsync(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def metrics(prediction: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = prediction.astype(bool)
    target = gt.astype(bool)
    intersection = int(np.logical_and(pred, target).sum())
    pred_area = int(pred.sum())
    gt_area = int(target.sum())
    union = pred_area + gt_area - intersection
    return {
        "dice": 2 * intersection / max(pred_area + gt_area, 1),
        "iou": intersection / max(union, 1),
        "precision": intersection / max(pred_area, 1),
        "recall": intersection / max(gt_area, 1),
        "pred_area_ratio": pred_area / pred.size,
        "gt_area_ratio": gt_area / target.size,
    }


def load_gt(path: str, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def compare_t17(
    t17_csv: Path, queries: list[dict[str, Any]]
) -> dict[tuple[str, str], float]:
    if not t17_csv.exists():
        return {}
    query_splits = {
        row["query_merged_id"]: row["query_split"] for row in queries
    }
    expected_group = {
        "train": "train_non_support",
        "validation": "validation",
        "test": "test",
    }
    result = {}
    with t17_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = query_splits.get(row["merged_id"])
            if split is not None and row["group"] == expected_group[split] and row[
                "method"
            ] in {"fixed_no_gt", "support_grid", "support_ridge"}:
                result[(row["merged_id"], row["method"])] = float(row["dice"])
    return result


def summarize(
    rows: list[dict[str, Any]], baseline: dict[tuple[str, str], float]
) -> dict[str, Any]:
    return {
        "overall": summarize_group(rows, baseline),
        "by_split": {
            split: summarize_group(
                [row for row in rows if row["query_split"] == split], baseline
            )
            for split in ("train", "validation", "test")
            if any(row["query_split"] == split for row in rows)
        },
        "by_source_dataset": {
            source: summarize_group(
                [row for row in rows if row["query_source_dataset"] == source],
                baseline,
            )
            for source in ("CVC-ClinicDB", "kvasir-seg")
            if any(row["query_source_dataset"] == source for row in rows)
        },
    }


def summarize_group(
    rows: list[dict[str, Any]], baseline: dict[tuple[str, str], float]
) -> dict[str, Any]:
    methods = {
        "pseudovideo_top": [float(row["top_dice"]) for row in rows],
        "pseudovideo_union": [float(row["union_dice"]) for row in rows],
    }
    for name in ("fixed_no_gt", "support_grid", "support_ridge"):
        methods[f"t17_{name}"] = [
            baseline[(row["query_merged_id"], name)]
            for row in rows
            if (row["query_merged_id"], name) in baseline
        ]
    methods["pseudovideo_empty_fallback_ridge"] = [
        (
            baseline[(row["query_merged_id"], "support_ridge")]
            if row["query_candidate_count"] == 0
            and (row["query_merged_id"], "support_ridge") in baseline
            else float(row["top_dice"])
        )
        for row in rows
    ]
    summary = {
        "n": len(rows),
        "empty_query_output_rate": float(
            np.mean([row["query_candidate_count"] == 0 for row in rows])
        ),
        "methods": {},
    }
    for method, values in methods.items():
        array = np.asarray(values, dtype=np.float64)
        summary["methods"][method] = {
            "n": len(array),
            "dice_mean": float(array.mean()) if len(array) else None,
            "dice_std": float(array.std()) if len(array) else None,
            "dice_median": float(np.median(array)) if len(array) else None,
            "zero_rate": float((array == 0).mean()) if len(array) else None,
        }
    return summary


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.retrieval_manifest.resolve())
    if args.limit > 0:
        rows = rows[: args.limit]
    output_root = args.output_root.resolve()
    result_path = output_root / "per_pair.jsonl"
    existing_rows = read_jsonl(result_path) if args.resume else []
    existing = {row["query_merged_id"]: row for row in existing_rows}
    pending = [
        row
        for row in rows
        if args.overwrite or row["query_merged_id"] not in existing
    ]
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"pilot={len(rows)} pending={len(pending)}")
    print(f"checkpoint={args.checkpoint}")
    if pending:
        model = build_sam3_video_model(
            checkpoint_path=str(args.checkpoint),
            load_from_HF=False,
            device="cuda",
            compile=False,
        )
        model.eval()
    else:
        model = None

    mask_root = output_root / "predictions"
    mask_root.mkdir(parents=True, exist_ok=True)
    for index, pair in enumerate(pending, 1):
        started = time.time()
        query_image = Image.open(pair["query_image_path"]).convert("RGB")
        reference_image = (
            Image.open(pair["reference_image_path"])
            .convert("RGB")
            .resize(query_image.size, Image.Resampling.BILINEAR)
        )
        gt = load_gt(pair["query_mask_path"], query_image.size)
        inference_state = model.init_state(
            resource_path=[reference_image, query_image],
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        _, prompted_output = model.add_prompt(
            inference_state,
            frame_idx=0,
            text_str=None,
            boxes_xywh=[pair["reference_box_xywh_normalized"]],
            box_labels=[1],
        )
        query_output = None
        for frame_idx, output in model.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=2,
            reverse=False,
        ):
            if frame_idx == 1:
                query_output = output
        if query_output is None:
            raise RuntimeError(f"No propagated output for {pair['query_merged_id']}")
        masks = np.asarray(query_output["out_binary_masks"], dtype=bool)
        scores = np.asarray(query_output["out_probs"], dtype=np.float64)
        if len(masks):
            top_index = int(np.argmax(scores))
            top_mask = masks[top_index]
            union_mask = masks.any(axis=0)
            top_score = float(scores[top_index])
        else:
            top_mask = np.zeros(gt.shape, dtype=bool)
            union_mask = top_mask.copy()
            top_score = 0.0
        top_metrics = metrics(top_mask, gt)
        union_metrics = metrics(union_mask, gt)
        safe_id = pair["query_merged_id"].replace("::", "__")
        Image.fromarray(top_mask.astype(np.uint8) * 255).save(
            mask_root / f"{safe_id}_top.png"
        )
        result = {
            **pair,
            "status": "success",
            "prompted_frame_candidate_count": int(
                len(prompted_output["out_binary_masks"])
            ),
            "query_candidate_count": int(len(masks)),
            "top_score": top_score,
            **{f"top_{key}": value for key, value in top_metrics.items()},
            **{f"union_{key}": value for key, value in union_metrics.items()},
            "seconds": round(time.time() - started, 3),
        }
        append_fsync(result_path, result)
        existing[result["query_merged_id"]] = result
        print(
            f"[{index}/{len(pending)}] {result['query_merged_id']} <- "
            f"{result['reference_merged_id']} candidates={len(masks)} "
            f"top={result['top_dice']:.4f} union={result['union_dice']:.4f}",
            flush=True,
        )
        del inference_state
        torch.cuda.empty_cache()

    final_rows = [
        existing[row["query_merged_id"]]
        for row in rows
        if row["query_merged_id"] in existing
    ]
    temporary = result_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in final_rows
        ),
        encoding="utf-8",
    )
    temporary.replace(result_path)
    t17_csv = Path(
        "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
        "T17_autonomous_target_1pct/selector_results/per_sample.csv"
    )
    baseline = compare_t17(t17_csv, final_rows)
    summary = summarize(final_rows, baseline)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
