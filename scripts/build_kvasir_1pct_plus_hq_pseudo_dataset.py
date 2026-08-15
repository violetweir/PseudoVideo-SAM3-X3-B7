#!/usr/bin/env python3
"""Build a SAM3 full-finetune COCO dataset from 1% GT + HQ pseudo labels."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


EXP = Path("/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T11_sam3_lowlabel_ft_kvasir_budgets")
BASE_DATASET = EXP / "dataset/KvasirSEG_1pct_seed2026"
PSEUDO_MANIFEST = Path(
    "/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_1pct_anchors/routeco_sam3_v1/routeco_v1_pseudo_manifest.jsonl"
)
OUTPUT = EXP / "dataset/KvasirSEG_1pct_plus_hq_pseudo_seed2026"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def binary_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def coco_rle(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def bbox_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, x2 = float(xs.min()), float(xs.max() + 1)
    y1, y2 = float(ys.min()), float(ys.max() + 1)
    return [x1, y1, x2 - x1, y2 - y1]


def select_hq(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        uncertainty = row["routeco_uncertainty"]
        mask = binary_mask(row["pseudo_mask_path"])
        area = float(mask.mean())
        keep = (
            float(row["q_cycle"]) >= 0.95
            and float(uncertainty["route_mask_mean_variance"]) <= 0.01
            and float(uncertainty["route_selected_mean_disagreement"]) <= 0.05
            and 0.001 <= area <= 0.6
        )
        if keep:
            selected.append(row)
    return selected


def convert_base_train(train_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    data = json.loads((BASE_DATASET / "train/_annotations.coco.json").read_text())
    images, annotations = [], []
    image_id_map = {}
    next_image_id = 1
    next_ann_id = 1
    for image in data["images"]:
        src = BASE_DATASET / "train/images" / image["file_name"]
        dst_name = f"gt_{image['file_name']}"
        symlink_or_copy(src, train_dir / "images" / dst_name)
        new_image = dict(image)
        new_image["id"] = next_image_id
        new_image["file_name"] = dst_name
        new_image["source_type"] = "gt_1pct"
        image_id_map[image["id"]] = next_image_id
        images.append(new_image)
        next_image_id += 1
    for ann in data["annotations"]:
        new_ann = dict(ann)
        new_ann["id"] = next_ann_id
        new_ann["image_id"] = image_id_map[ann["image_id"]]
        new_ann["source_type"] = "gt_1pct"
        annotations.append(new_ann)
        next_ann_id += 1
    return images, annotations, next_image_id, next_ann_id


def add_pseudo_rows(
    rows: list[dict[str, Any]],
    train_dir: Path,
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    next_image_id: int,
    next_ann_id: int,
) -> tuple[int, int]:
    used_names = {image["file_name"] for image in images}
    for row in rows:
        src = Path(row["target_image_path"])
        stem = row["target_id"].replace("::", "__").replace("/", "_")
        dst_name = f"pseudo_{stem}{src.suffix.lower()}"
        if dst_name in used_names:
            raise RuntimeError(f"Duplicate output image name: {dst_name}")
        symlink_or_copy(src, train_dir / "images" / dst_name)
        image = Image.open(src).convert("RGB")
        width, height = image.size
        mask = binary_mask(row["pseudo_mask_path"])
        if mask.shape != (height, width):
            mask = (
                np.asarray(
                    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).resize(
                        (width, height), Image.Resampling.NEAREST
                    )
                )
                > 127
            )
        images.append(
            {
                "id": next_image_id,
                "file_name": dst_name,
                "width": width,
                "height": height,
                "source_type": "hq_pseudo",
                "source_target_id": row["target_id"],
            }
        )
        annotations.append(
            {
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": 1,
                "segmentation": coco_rle(mask),
                "area": float(mask.sum()),
                "bbox": bbox_from_mask(mask),
                "iscrowd": 0,
                "source_type": "hq_pseudo",
                "q_cycle": float(row["q_cycle"]),
                "route_mask_mean_variance": float(row["routeco_uncertainty"]["route_mask_mean_variance"]),
                "route_selected_mean_disagreement": float(row["routeco_uncertainty"]["route_selected_mean_disagreement"]),
                "route_id": row["route_id"],
                "feature_mode": row["feature_mode"],
                "route_type": row["route_type"],
            }
        )
        used_names.add(dst_name)
        next_image_id += 1
        next_ann_id += 1
    return next_image_id, next_ann_id


def copy_eval_split(split: str) -> None:
    out_split = "val" if split == "val" else split
    src = BASE_DATASET / out_split
    dst = OUTPUT / out_split
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "images").exists():
        try:
            os.symlink(src / "images", dst / "images")
        except OSError:
            shutil.copytree(src / "images", dst / "images", dirs_exist_ok=True)
    shutil.copy2(src / "_annotations.coco.json", dst / "_annotations.coco.json")


def main() -> None:
    train_dir = OUTPUT / "train"
    if train_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset: {OUTPUT}")
    (train_dir / "images").mkdir(parents=True)
    images, annotations, next_image_id, next_ann_id = convert_base_train(train_dir)
    pseudo_rows = select_hq(read_jsonl(PSEUDO_MANIFEST))
    add_pseudo_rows(pseudo_rows, train_dir, images, annotations, next_image_id, next_ann_id)
    categories = [{"id": 1, "name": "colon polyp", "supercategory": "medical lesion"}]
    (train_dir / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )
    copy_eval_split("val")
    copy_eval_split("test")
    summary = {
        "protocol": "Kvasir-SEG 1% GT plus high-quality pseudo labels, seed 2026",
        "base_dataset": str(BASE_DATASET),
        "pseudo_manifest": str(PSEUDO_MANIFEST),
        "output": str(OUTPUT),
        "gt_1pct_images": 8,
        "hq_pseudo_images": len(pseudo_rows),
        "train_images": len(images),
        "train_annotations": len(annotations),
        "thresholds": {
            "q_cycle_min": 0.95,
            "route_mask_mean_variance_max": 0.01,
            "route_selected_mean_disagreement_max": 0.05,
            "area_range": [0.001, 0.6],
        },
        "unlabeled_train_gt_masks_used": False,
        "validation_test_gt_used_for_training": False,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
