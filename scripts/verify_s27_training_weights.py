#!/usr/bin/env python3
"""Fail-closed verification of S27 image/pixel weights before training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def probability(path: str | Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    return array.astype(np.float32) / (65535.0 if array.dtype == np.uint16 else 255.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    reference = read_jsonl(args.original_reference)
    q_values = np.array([float(row["q_multi"]) for row in reference])
    q_min, q_max = float(q_values.min()), float(q_values.max())
    expected_original = {
        row["image_id"]: float(
            np.clip((float(row["q_multi"]) - q_min) / max(q_max - q_min, 1e-12), 0.2, 1.0)
        )
        for row in reference
    }
    errors = []
    for row in rows:
        weight = float(row["explicit_quality_weight"])
        if not 0.0 <= weight <= 1.0:
            errors.append(f"{row['target_id']}: image weight {weight}")
        if row["sample_type"] == "original":
            expected = expected_original[row["target_id"]]
            if abs(weight - expected) > 1e-12:
                errors.append(
                    f"{row['target_id']}: original changed {weight} != {expected}"
                )
        if row["sample_type"] == "tier_b" and not row.get("pixel_weight_path"):
            errors.append(f"{row['target_id']}: Tier B lacks pixel weight")
    if errors:
        raise RuntimeError("\n".join(errors[:50]))

    rng = random.Random(args.seed)
    candidates = list(rows)
    sampled = rng.sample(candidates, min(10, len(candidates)))
    audit = []
    for row in sampled:
        pixel = (
            probability(row["pixel_weight_path"])
            if row.get("pixel_weight_path")
            else np.ones((1, 1), dtype=np.float32)
        )
        audit.append(
            {
                "target_id": row["target_id"],
                "tier": row.get("tier"),
                "sample_type": row["sample_type"],
                "image_weight": float(row["explicit_quality_weight"]),
                "pixel_weight_mean": float(pixel.mean()),
                "pixel_weight_min": float(pixel.min()),
                "pixel_weight_max": float(pixel.max()),
                "q_multi": row.get("q_multi"),
                "q_model_mean": row.get("q_model_mean"),
                "q_model_var": row.get("q_model_var"),
            }
        )
    result = {
        "manifest": str(args.manifest.resolve()),
        "row_count": len(rows),
        "original_count": sum(row["sample_type"] == "original" for row in rows),
        "tier_a_count": sum(row["sample_type"] == "tier_a" for row in rows),
        "tier_b_count": sum(row["sample_type"] == "tier_b" for row in rows),
        "fixed_original_reference_count": len(reference),
        "original_weights_unchanged": True,
        "uint16_pixel_weights_decoded_without_convert_L": True,
        "sampled_weights": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
