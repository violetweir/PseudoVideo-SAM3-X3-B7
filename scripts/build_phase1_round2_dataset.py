#!/usr/bin/env python3
"""Build the round-2 SAM3 fine-tune dataset: 8 GT + student-audited consensus labels."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def read_jsonl(path: Path) -> list[dict]:
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


def coco_rle(mask: np.ndarray) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=str, default="round2_student_audited")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite existing dataset: {args.output}")

    train_dir = args.output / "train"
    (train_dir / "images").mkdir(parents=True)
    base_train = args.base_dataset / "train"
    base_data = json.loads((base_train / "_annotations.coco.json").read_text())
    images, annotations = [], []
    next_image_id = 1
    next_ann_id = 1
    image_id_map = {}
    for image in base_data["images"]:
        src = base_train / "images" / image["file_name"]
        dst_name = f"gt_{image['file_name']}"
        symlink_or_copy(src, train_dir / "images" / dst_name)
        new_image = dict(image)
        new_image["id"] = next_image_id
        new_image["file_name"] = dst_name
        new_image["source_type"] = "gt_1pct"
        image_id_map[image["id"]] = next_image_id
        images.append(new_image)
        next_image_id += 1
    for ann in base_data["annotations"]:
        new_ann = dict(ann)
        new_ann["id"] = next_ann_id
        new_ann["image_id"] = image_id_map[ann["image_id"]]
        new_ann["source_type"] = "gt_1pct"
        annotations.append(new_ann)
        next_ann_id += 1

    pseudo_rows = read_jsonl(args.pool)
    used_names = {image["file_name"] for image in images}
    for row in pseudo_rows:
        target_id = row["target_id"]
        image_path = (
            Path(
                "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
                "T02_fresh_polyp_hf_sources/raw_hf_snapshots/kvasir-seg/snapshot/train/images"
            )
            / f"{target_id.rsplit('::', 1)[-1]}.png"
        )
        stem = target_id.replace("::", "__").replace("/", "_")
        dst_name = f"pseudo_{stem}{image_path.suffix.lower()}"
        if dst_name in used_names:
            raise RuntimeError(f"Duplicate output image name: {dst_name}")
        symlink_or_copy(image_path, train_dir / "images" / dst_name)
        image = Image.open(image_path).convert("RGB")
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
                "source_type": args.source,
                "source_target_id": target_id,
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
                "source_type": args.source,
                "q_multi": float(row["q_multi"]),
                "q_return": float(row["q_return"]),
                "q_model_mean": float(row["q_model_mean"]),
                "q_model_min": float(row["q_model_min"]),
                "route_id": row["route_id"],
                "feature_mode": row["feature_mode"],
                "bridge_count": int(row["bridge_count"]),
            }
        )
        used_names.add(dst_name)
        next_image_id += 1
        next_ann_id += 1

    categories = [{"id": 1, "name": "colon polyp", "supercategory": "medical lesion"}]
    (train_dir / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )
    for split in ("val", "test"):
        out_split = args.output / split
        out_split.mkdir(parents=True, exist_ok=True)
        if not (out_split / "images").exists():
            try:
                os.symlink(args.base_dataset / split / "images", out_split / "images")
            except OSError:
                shutil.copytree(args.base_dataset / split / "images", out_split / "images", dirs_exist_ok=True)
        shutil.copy2(
            args.base_dataset / split / "_annotations.coco.json",
            out_split / "_annotations.coco.json",
        )
    summary = {
        "base_dataset": str(args.base_dataset),
        "pool": str(args.pool),
        "output": str(args.output),
        "gt_images": len(base_data["images"]),
        "pseudo_images": len(pseudo_rows),
        "train_images": len(images),
        "train_annotations": len(annotations),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
