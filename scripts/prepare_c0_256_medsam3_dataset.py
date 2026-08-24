#!/usr/bin/env python3
"""Prepare a MedSAM3-style COCO dataset from C0-256-base pseudo labels.

Train split:
  * images/masks from pseudo_manifest_x3.jsonl (C0-256 X3 pseudo labels)
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
CATEGORY_NAME = "polyp"


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
                        help="work/rerun_c0_256_base/pseudo_manifest_x3.jsonl")
    parser.add_argument("--validation-metadata", type=Path, required=True,
                        help="work/kvasir_1pct_anchors/baseline_data/validation/metadata.jsonl")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="e.g. work/rerun_c0_256_base/medsam3_lora/data")
    args = parser.parse_args()

    pseudo_rows = read_jsonl(args.pseudo_manifest)
    val_rows = read_jsonl(args.validation_metadata)

    train_records = [
        {
            "image_path": row["target_image_path"] if "target_image_path" in row else row.get("image_path"),
            "mask_path": row["pseudo_mask_path"],
            "sample_type": row.get("sample_type", "unknown"),
            "target_id": row["target_id"],
        }
        for row in pseudo_rows
        if row.get("pseudo_mask_path")
    ]
    # The X3 manifest does not include target_image_path; join through routeco
    # manifest is done in a later step if needed. Here we require it.
    missing_image = [r for r in train_records if not r["image_path"]]
    if missing_image:
        print(f"WARNING: {len(missing_image)} X3 rows do not have target_image_path")

    # If pseudo_manifest_x3 has no image path, try to load the routeco manifest
    # which contains target_image_path for the same targets.
    if missing_image:
        routeco_manifest = args.pseudo_manifest.parent / "routeco_256_base" / "routeco_256_manifest_with_student_uncertainty.jsonl"
        if routeco_manifest.exists():
            route_rows = read_jsonl(routeco_manifest)
            by_target = {r["target_id"]: r for r in route_rows}
            for rec in train_records:
                if not rec["image_path"]:
                    rr = by_target.get(rec["target_id"])
                    if rr:
                        rec["image_path"] = rr["target_image_path"]
        missing_image = [r for r in train_records if not r["image_path"]]
        if missing_image:
            raise SystemExit(f"Still missing image_path for {len(missing_image)} rows, e.g. {missing_image[0]}")

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
        "valid": len(val_records),
        "output_root": str(args.output_root),
    }, indent=2))


if __name__ == "__main__":
    main()
