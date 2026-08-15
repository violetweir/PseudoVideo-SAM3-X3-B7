#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


SRC = Path(
    "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T11_sam3_lowlabel_ft_kvasir_budgets/"
    "dataset/KvasirSEG_1pct_plus_hq_pseudo_seed2026"
)
DST = Path(
    "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T11_sam3_lowlabel_ft_kvasir_budgets/"
    "dataset/KvasirSEG_1pct_plus_hq_pseudo_safe_v2_seed2026"
)
BAD_SOURCE_TARGET_IDS = {
    "kvasir-seg::cju1dq3x1vgx109889c7wyirg",
    "kvasir-seg::cju5c7oijaqmq09878qwgqv8n",
}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, dst.parent)
    try:
        os.symlink(rel, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing source dataset: {SRC}")
    if DST.exists():
        raise SystemExit(f"refusing to overwrite existing dataset: {DST}")

    for split in ["train", "val", "test"]:
        (DST / split / "images").mkdir(parents=True, exist_ok=True)

    ann = json.loads((SRC / "train" / "_annotations.coco.json").read_text())
    bad_image_ids = {
        im["id"]
        for im in ann["images"]
        if im.get("source_target_id") in BAD_SOURCE_TARGET_IDS
        or im.get("file_name") == "pseudo_kvasir-seg__cju1dq3x1vgx109889c7wyirg.png"
        or im.get("file_name") == "pseudo_kvasir-seg__cju5c7oijaqmq09878qwgqv8n.png"
    }
    if not bad_image_ids:
        raise SystemExit("no bad images matched")

    kept_images = [im for im in ann["images"] if im["id"] not in bad_image_ids]
    kept_image_ids = {im["id"] for im in kept_images}
    kept_annotations = [a for a in ann["annotations"] if a["image_id"] in kept_image_ids]
    ann["images"] = kept_images
    ann["annotations"] = kept_annotations

    for im in kept_images:
        link_or_copy(
            SRC / "train" / "images" / im["file_name"],
            DST / "train" / "images" / im["file_name"],
        )
    (DST / "train" / "_annotations.coco.json").write_text(json.dumps(ann, indent=2) + "\n")

    for split in ["val", "test"]:
        for item in (SRC / split).iterdir():
            if item.name == "images" and item.is_dir():
                for image_item in item.iterdir():
                    link_or_copy(image_item, DST / split / "images" / image_item.name)
            else:
                link_or_copy(item, DST / split / item.name)

    summary = {
        "source": str(SRC),
        "output": str(DST),
        "bad_source_target_ids": sorted(BAD_SOURCE_TARGET_IDS),
        "removed_image_ids": sorted(bad_image_ids),
        "train_images": len(kept_images),
        "train_annotations": len(kept_annotations),
    }
    (DST / "safe_filter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
