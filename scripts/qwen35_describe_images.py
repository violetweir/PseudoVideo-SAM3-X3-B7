#!/usr/bin/env python3
"""Ask Qwen3.5-4B to describe the ORIGINAL endoscopic image (not the mask).

IMR multimodal branch: the model sees the raw image (optionally with the GT
mask overlaid for the 8 anchors) and returns structured polyp + surrounding
tissue fields plus a free-text sentence. The output feeds
stage1_feature_knn_routes.load_qwen_image_text_sim for the fused KNN ranking.

Usage:
  # bridge candidates: original images only
  $PY scripts/qwen35_describe_images.py \
      --manifest work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl \
      --output work/kvasir_1pct_anchors/qwen35_image_descriptions.jsonl
  # anchors: original image + GT mask overlay
  $PY scripts/qwen35_describe_images.py \
      --manifest work/kvasir_1pct_anchors/protocol/support_manifest.jsonl \
      --overlay-gt-mask \
      --output work/kvasir_1pct_anchors/qwen35_image_descriptions_overlay.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
MODEL_PATH = Path("/Data_8TB/lht/models/Qwen3.5-4B")

PROMPT = (
    "You are given an endoscopic image containing a polyp. Describe the polyp "
    "and its surrounding tissue in one compact line, then output structured "
    "fields in exactly this format, with no explanation and no markdown:\n"
    "shape=<round|oval|irregular|elongated>; size=<small|medium|large>; "
    "position=<left|center|right|top|bottom|diffuse>; boundary=<smooth|irregular>; "
    "components=<single|multiple>; spread=<local|multi_region>; "
    "texture=<smooth|lobulated|nodular|pedunculated>; "
    "context=<normal|erythema|hemorrhage|necrotic>"
)

FIELDS = [
    "shape",
    "size",
    "position",
    "boundary",
    "components",
    "spread",
    "texture",
    "context",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_fields(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for field in FIELDS:
        match = re.search(rf"\b{field}\s*=\s*([A-Za-z_]+)", text)
        if match:
            out[field] = match.group(1).lower()
    return out


def load_overlay_image(image_path: str, mask_path: str | None) -> Image.Image:
    """Original image with the mask overlaid as semi-transparent red."""
    image = Image.open(image_path).convert("RGB")
    if not mask_path or not Path(mask_path).exists():
        return image
    mask = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
    fg = np.asarray(mask) > 127
    arr = np.asarray(image).astype(np.float32).copy()
    overlay = arr.copy()
    overlay[..., 0] = 255.0
    arr[fg] = 0.5 * arr[fg] + 0.5 * overlay[fg]
    return Image.fromarray(arr.astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-gt-mask", action="store_true",
                        help="Overlay the GT mask (frozen_mask_path / mask_path) onto the image.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model),
        torch_dtype="auto",
        device_map=args.device,
        trust_remote_code=True,
    ).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            mask_path = None
            if args.overlay_gt_mask:
                mask_path = row.get("frozen_mask_path") or row.get("mask_path")
            image = load_overlay_image(row["image_path"], mask_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            ).to(args.device)
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            parsed = parse_fields(output_text)
            record = {
                "target_id": row.get("target_id") or row["merged_id"],
                "image_path": row["image_path"],
                "qwen_text": output_text,
                "qwen_features": parsed,
            }
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
