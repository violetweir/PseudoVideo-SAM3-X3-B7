#!/usr/bin/env python3
"""Ask Qwen3.5-4B to produce a numeric mask-feature JSON for each pseudo mask.

This is the Qwen branch of round-2 KNN construction.  It is intentionally
separate from the manual mask-feature extractor so the two feature sources can
be compared offline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
MODEL_PATH = Path("/Data_8TB/lht/models/Qwen3.5-4B")


PROMPT = (
    "You are given a binary polyp mask. Describe it in one compact line using exactly "
    "this format, with no explanation and no markdown:\n"
    "shape=<round|oval|irregular|elongated>; size=<small|medium|large>; "
    "position=<left|center|right|top|bottom|diffuse>; boundary=<smooth|irregular>; "
    "components=<single|multiple>; spread=<local|multi_region>"
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_fields(text: str) -> dict[str, str]:
    fields = ["shape", "size", "position", "boundary", "components", "spread"]
    out: dict[str, str] = {}
    for field in fields:
        match = re.search(rf"\b{field}\s*=\s*([A-Za-z_]+)", text)
        if match:
            out[field] = match.group(1).lower()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/train_pseudo_masks_round1.jsonl",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/qwen35_mask_features.jsonl",
    )
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

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            image = Image.open(row["pseudo_mask_path"]).convert("RGB")
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
                "target_id": row["target_id"],
                "qwen_text": output_text,
                "qwen_features": parsed,
            }
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
