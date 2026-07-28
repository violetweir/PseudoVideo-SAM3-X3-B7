#!/usr/bin/env python3
"""Export probability maps, binary masks, and confidence for a T24 student."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tv_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCSAM_ROOT = Path(
    os.environ.get("SC_SAM_ROOT", PROJECT_ROOT / "third_party" / "SC-SAM")
).resolve()
for root in (PROJECT_ROOT, SCSAM_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from Model.model import SamUnet


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(
        (args.run_dir / "protocol.json").read_text(encoding="utf-8")
    )
    config = argparse.Namespace(**protocol)
    checkpoint = args.run_dir / "student_best.pth"
    model = SamUnet(config).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
    normalize = tv_transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        metadata = read_jsonl(Path(config.data_path) / split / "metadata.jsonl")
        probability_dir = args.output_root / split / "probability"
        binary_dir = args.output_root / split / "binary"
        probability_dir.mkdir(parents=True, exist_ok=True)
        binary_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        with torch.inference_mode():
            for row in sorted(metadata, key=lambda item: item["merged_id"]):
                image = Image.open(row["file_name"]).convert("RGB")
                image = image.resize(
                    (config.image_size, config.image_size),
                    Image.Resampling.NEAREST,
                )
                array = np.asarray(image, dtype=np.float32) / 255
                tensor = torch.from_numpy(array).permute(2, 0, 1)
                tensor = normalize(tensor).unsqueeze(0).cuda()
                _, probabilities = model(tensor)
                foreground = probabilities[0, 1].float().cpu().numpy()
                binary = foreground >= 0.5
                safe_id = row["merged_id"].replace("::", "__").replace("/", "_")
                probability_path = probability_dir / f"{safe_id}.png"
                binary_path = binary_dir / f"{safe_id}.png"
                Image.fromarray(
                    np.rint(foreground * 65535).astype(np.uint16), mode="I;16"
                ).save(probability_path)
                Image.fromarray((binary * 255).astype(np.uint8)).save(binary_path)
                confidence = np.maximum(foreground, 1 - foreground)
                rows.append(
                    {
                        "merged_id": row["merged_id"],
                        "split": split,
                        "source_dataset": row["source_dataset"],
                        "image_path": str(Path(row["file_name"]).resolve()),
                        "student_probability_map": str(probability_path.resolve()),
                        "student_binary_mask": str(binary_path.resolve()),
                        "student_confidence": float(confidence.mean()),
                        "student_nonempty": bool(binary.any()),
                        "student_area_ratio": float(binary.mean()),
                        "checkpoint": str(checkpoint.resolve()),
                    }
                )
        manifest = args.output_root / f"student_predictions_{split}.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"split": split, "count": len(rows)}), flush=True)
    (args.output_root / "EXPORT_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
