#!/usr/bin/env python3
"""Prepare the C0-256-base X3+B7 dataset for MedSAM3-style LoRA.

Train split:
  * B7-selected pseudo masks from the final C0-256-base X3-best selector
  * the eight frozen human-labelled anchors
Valid split:
  * images/masks from the real Kvasir validation metadata (for loss monitoring)

This produces the directory layout expected by scripts/train_sam3_lora_kvasir.py:
  <output_root>/train/_annotations.coco.json
  <output_root>/train/images/*.png
  <output_root>/valid/_annotations.coco.json
  <output_root>/valid/images/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils


CATEGORY_ID = 1
CATEGORY_NAME = "colon polyp"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mask_to_rle(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]


def write_coco_split(split_dir: Path, records: list[dict]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1
    for image_id, record in enumerate(records, 1):
        src_image = Path(record["image_path"])
        src_mask = Path(record["mask_path"])
        safe_name = src_image.stem
        # Use a namespaced file name to avoid collisions across datasets.
        rel_image = f"images/{safe_name}.png"
        dst_image = split_dir / rel_image
        if not dst_image.exists():
            dst_image.symlink_to(src_image.resolve())

        with Image.open(src_mask).convert("L") as im:
            mask = np.asarray(im) > 127
        if not mask.any():
            print(f"WARNING: empty mask for {record['image_path']}, skipping annotation")
            continue
        with Image.open(src_image) as im:
            width, height = im.size

        rle = mask_to_rle(mask)
        rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]

        images.append({
            "id": image_id,
            "file_name": rel_image,
            "width": width,
            "height": height,
        })
        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": CATEGORY_ID,
            "bbox": mask_to_bbox(mask),
            "segmentation": rle,
            "area": float(mask.sum()),
            "iscrowd": 0,
        })
        ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}],
    }
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps(coco, indent=2), encoding="utf-8"
    )
    print(f"{split_dir.name}: images={len(images)} annotations={len(annotations)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-manifest", type=Path, required=True,
                        help="B7-selected train manifest")
    parser.add_argument("--labeled-list", type=Path, required=True,
                        help="Eight frozen GT image paths, one per line")
    parser.add_argument("--validation-metadata", type=Path, required=True,
                        help="work/kvasir_1pct_anchors/baseline_data/validation/metadata.jsonl")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="e.g. work/rerun_c0_256_base/medsam3_lora/data")
    args = parser.parse_args()

    pseudo_rows = read_jsonl(args.pseudo_manifest)
    val_rows = read_jsonl(args.validation_metadata)

    train_records = [
        {
            "image_path": row.get("target_image_path") or row.get("image_path"),
            "mask_path": row["pseudo_mask_path"],
            "sample_type": row.get("sample_type", "b7_pseudo"),
            "target_id": row["target_id"],
        }
        for row in pseudo_rows
        if row.get("pseudo_mask_path")
    ]
    missing_image = [r for r in train_records if not r["image_path"]]
    if missing_image:
        raise SystemExit(
            f"Missing image_path for {len(missing_image)} rows, e.g. {missing_image[0]}"
        )

    labeled_paths = [
        Path(line.strip())
        for line in args.labeled_list.read_text().splitlines()
        if line.strip()
    ]
    gt_records = []
    for image_path in labeled_paths:
        mask_path = Path(str(image_path).replace("/images/", "/masks/"))
        if not image_path.exists() or not mask_path.exists():
            raise SystemExit(f"Missing GT pair: image={image_path}, mask={mask_path}")
        gt_records.append({
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "sample_type": "human_gt_anchor",
            "target_id": f"kvasir-seg::{image_path.stem}",
        })

    pseudo_ids = {record["target_id"] for record in train_records}
    overlap = sorted(record["target_id"] for record in gt_records if record["target_id"] in pseudo_ids)
    if overlap:
        raise SystemExit(f"GT anchors unexpectedly overlap pseudo targets: {overlap}")
    train_records.extend(gt_records)

    val_records = [
        {
            "image_path": row["file_name"],
            "mask_path": row["mask_file_name"],
            "sample_type": "gt",
            "target_id": row["merged_id"],
        }
        for row in val_rows
    ]

    train_dir = args.output_root / "train"
    valid_dir = args.output_root / "valid"
    write_coco_split(train_dir, train_records)
    write_coco_split(valid_dir, val_records)
    print(json.dumps({
        "train": len(train_records),
        "train_pseudo": len(pseudo_rows),
        "train_human_gt": len(gt_records),
        "valid": len(val_records),
        "category": CATEGORY_NAME,
        "output_root": str(args.output_root),
    }, indent=2))


if __name__ == "__main__":
    main()
