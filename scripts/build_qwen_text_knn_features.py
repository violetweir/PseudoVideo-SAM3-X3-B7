#!/usr/bin/env python3
"""Convert Qwen3.5 mask text descriptions into categorical KNN features.

The first pass only builds a frozen feature matrix and reports top-k text
neighbours for validation/test targets.  The output is intended as a second
round KNN candidate source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
FIELDS = {
    "shape": ["round", "oval", "irregular", "elongated"],
    "size": ["small", "medium", "large"],
    "position": ["left", "center", "right", "top", "bottom", "diffuse"],
    "boundary": ["smooth", "irregular"],
    "components": ["single", "multiple"],
    "spread": ["local", "multi_region"],
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def to_vector(features: dict) -> np.ndarray:
    out: list[float] = []
    for field, allowed in FIELDS.items():
        value = (features.get(field) or "").lower()
        bits = [1.0 if value == candidate else 0.0 for candidate in allowed]
        out.extend(bits)
    vec = np.asarray(out, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def load_table(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for row in read_jsonl(path):
        out[row["target_id"]] = to_vector(row.get("qwen_features") or {})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-desc",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/qwen35_mask_descriptions.jsonl",
    )
    parser.add_argument(
        "--validation-desc",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/validation_pseudo_masks_round1/qwen35_mask_descriptions.jsonl",
    )
    parser.add_argument(
        "--test-desc",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/test_pseudo_masks_round1/qwen35_mask_descriptions.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/qwen_text_knn_v1",
    )
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args()

    train = load_table(args.train_desc)
    validation = load_table(args.validation_desc)
    test = load_table(args.test_desc)
    train_ids = list(train.keys())
    train_matrix = np.stack([train[i] for i in train_ids])

    args.output_root.mkdir(parents=True, exist_ok=True)
    result = {"train_ids": train_ids, "neighbours": {}}
    for split, table in (("validation", validation), ("test", test)):
        for target_id, vec in sorted(table.items()):
            sims = train_matrix @ vec
            order = np.argsort(-sims)[: args.top_k]
            result["neighbours"][target_id] = [
                {"train_id": train_ids[i], "similarity": float(sims[i])}
                for i in order
            ]

    (args.output_root / "qwen_text_knn_topk.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    matrix_path = args.output_root / "train_qwen_text_features.npz"
    np.savez_compressed(matrix_path, train_ids=np.asarray(train_ids), features=train_matrix)
    print(
        json.dumps(
            {
                "train": len(train_ids),
                "validation": len(validation),
                "test": len(test),
                "top_k": args.top_k,
                "output": str(args.output_root),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
