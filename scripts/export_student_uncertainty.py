#!/usr/bin/env python3
"""Export student uncertainty estimates for RouteCo-256.

For every target in the RouteCo manifest, this script runs the frozen U-Net
student under several weak and strong augmentations and records:
  * student_aug_variance: mean pixel variance over all augmented predictions
  * student_weak_strong_variance: mean squared difference between weak/strong means
  * student_confidence: mean max-probability of the average prediction
  * student_nonempty_count: how many augmented predictions are non-empty
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

from dataloader.transforms import build_weak_strong_transforms
from Model.model import SamUnet


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--weak-samples", type=int, default=3)
    parser.add_argument("--strong-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.manifest)
    print(f"manifest rows: {len(rows)}", flush=True)

    config = argparse.Namespace(
        **json.loads((args.run_dir / "protocol.json").read_text())
    )
    model = SamUnet(config).cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cuda"))
    normalize = tv_transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    transforms = build_weak_strong_transforms(
        argparse.Namespace(image_size=args.image_size)
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "student_uncertainty.jsonl"

    results = []
    for index, row in enumerate(rows, 1):
        image = np.asarray(
            Image.open(row["target_image_path"]).convert("RGB"), dtype=np.float32
        ) / 255.0

        weak_probs = []
        strong_probs = []
        with torch.inference_mode():
            for _ in range(args.weak_samples):
                aug = transforms["train_weak"](image=image)["image"]
                tensor = normalize(
                    torch.from_numpy(np.ascontiguousarray(aug)).float().permute(2, 0, 1)
                )
                _, prob = model(tensor.unsqueeze(0).cuda())
                weak_probs.append(prob[0, 1].float().cpu().numpy())
            for _ in range(args.strong_samples):
                aug = transforms["train_strong"](image=image)["image"]
                tensor = normalize(
                    torch.from_numpy(np.ascontiguousarray(aug)).float().permute(2, 0, 1)
                )
                _, prob = model(tensor.unsqueeze(0).cuda())
                strong_probs.append(prob[0, 1].float().cpu().numpy())

        all_probs = weak_probs + strong_probs
        stack = np.stack(all_probs, axis=0)
        weak_mean = np.mean(np.stack(weak_probs, axis=0), axis=0)
        strong_mean = np.mean(np.stack(strong_probs, axis=0), axis=0)
        mean_prob = stack.mean(axis=0)

        record = {
            "target_id": row["target_id"],
            "student_aug_variance": float(stack.var(axis=0).mean()),
            "student_weak_strong_variance": float(
                ((weak_mean - strong_mean) ** 2).mean()
            ),
            "student_confidence": float(
                np.mean(np.maximum(mean_prob, 1.0 - mean_prob))
            ),
            "student_nonempty_count": int(
                sum(bool((p >= 0.5).any()) for p in all_probs)
            ),
            "weak_samples": args.weak_samples,
            "strong_samples": args.strong_samples,
            "image_size": args.image_size,
        }
        results.append(record)
        if index % 50 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] {row['target_id']} "
                  f"aug_var={record['student_aug_variance']:.4f} "
                  f"ws_var={record['student_weak_strong_variance']:.4f}",
                  flush=True)

    write_jsonl(result_path, results)
    summary = {
        "n": len(results),
        "student_aug_variance": {
            "mean": float(np.mean([r["student_aug_variance"] for r in results])),
            "median": float(np.median([r["student_aug_variance"] for r in results])),
            "min": float(np.min([r["student_aug_variance"] for r in results])),
            "max": float(np.max([r["student_aug_variance"] for r in results])),
        },
        "student_weak_strong_variance": {
            "mean": float(np.mean([r["student_weak_strong_variance"] for r in results])),
            "median": float(np.median([r["student_weak_strong_variance"] for r in results])),
            "min": float(np.min([r["student_weak_strong_variance"] for r in results])),
            "max": float(np.max([r["student_weak_strong_variance"] for r in results])),
        },
    }
    write_jsonl(output_dir / "student_uncertainty_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
