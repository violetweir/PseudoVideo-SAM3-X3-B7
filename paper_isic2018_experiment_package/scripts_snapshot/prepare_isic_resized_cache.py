#!/usr/bin/env python3
"""Prepare 256x256 resized ISIC image/GT-mask cache for student training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_id(value: str) -> str:
    return value.replace("::", "__").replace("/", "_")


def resolve_split_path(split_root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else split_root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
    )
    args = parser.parse_args()

    summary = {}
    for split in args.splits:
        split_root = args.data_path / split
        image_dir = args.output_root / split / "images"
        mask_dir = args.output_root / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        rows = read_jsonl(split_root / "metadata.jsonl")
        written_images = 0
        written_masks = 0
        for row in rows:
            sid = safe_id(row["merged_id"])
            image_out = image_dir / f"{sid}.png"
            mask_out = mask_dir / f"{sid}.png"
            if not image_out.exists():
                image_path = resolve_split_path(split_root, row["file_name"])
                image = Image.open(image_path).convert("RGB").resize(
                    (args.image_size, args.image_size),
                    Image.Resampling.BILINEAR,
                )
                image.save(image_out)
                written_images += 1
            if not mask_out.exists():
                mask_path = resolve_split_path(split_root, row["mask_file_name"])
                mask = Image.open(mask_path).convert("L").resize(
                    (args.image_size, args.image_size),
                    Image.Resampling.NEAREST,
                )
                mask.save(mask_out)
                written_masks += 1
        summary[split] = {
            "records": len(rows),
            "written_images": written_images,
            "written_masks": written_masks,
            "image_dir": str(image_dir.resolve()),
            "mask_dir": str(mask_dir.resolve()),
        }
        print(json.dumps({"split": split, **summary[split]}, sort_keys=True), flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "CACHE_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
