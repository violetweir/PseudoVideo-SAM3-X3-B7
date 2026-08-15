#!/usr/bin/env python3
"""Phase-1 T24-S3 analog: pixel-consensus targets over the b3-b6 pool.

Ports prepare_t24_pseudo_consensus.py: for every original-pool target, take the
mean of the candidate binary route masks as the consensus probability and
pixel_weight = max(0.1, exp(-4 * pixel variance)).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


MODES = (
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_binary(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def save_u16(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(array, 0.0, 1.0) * 65535.0).astype(np.uint16)).save(
        path
    )


def safe_name(target_id: str) -> str:
    return target_id.replace("::", "__").replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--pixel-min-weight", type=float, default=0.1)
    args = parser.parse_args()

    original = read_jsonl(args.original_manifest)
    candidates: dict[str, list[dict]] = defaultdict(list)
    for mode in MODES:
        rows = read_jsonl(
            args.quality_root
            / mode
            / "propagation_quality_train/propagation_quality.jsonl"
        )
        for row in rows:
            if args.min_bridge <= int(row["bridge_count"]) <= args.max_bridge:
                candidates[row["target_id"]].append(row)

    probability_dir = args.output_root / "consensus_probability"
    weight_dir = args.output_root / "pixel_weight"
    output_rows = []
    for pseudo in sorted(original, key=lambda row: row["target_id"]):
        cands = candidates.get(pseudo["target_id"], [])
        if not cands:
            raise RuntimeError(f"No candidates for {pseudo['target_id']}")
        masks = [load_binary(row["forward_mask_path"]) for row in cands]
        if len({mask.shape for mask in masks}) != 1:
            raise RuntimeError(f"{pseudo['target_id']}: mask shape mismatch")
        stack = np.stack(masks).astype(np.float32)
        probability = stack.mean(axis=0)
        variance = stack.var(axis=0)
        pixel_weight = np.maximum(args.pixel_min_weight, np.exp(-args.beta * variance))
        stem = safe_name(pseudo["target_id"])
        probability_path = probability_dir / f"{stem}.png"
        weight_path = weight_dir / f"{stem}.png"
        save_u16(probability_path, probability)
        save_u16(weight_path, pixel_weight)
        output_rows.append(
            {
                **pseudo,
                "pseudo_consensus_path": str(probability_path.resolve()),
                "pixel_weight_path": str(weight_path.resolve()),
                "consensus_source": "mean of b3-b6 candidate binary masks",
                "pixel_weight_formula": f"max({args.pixel_min_weight}, exp(-{args.beta} * P_var))",
                "n_candidates": len(cands),
            }
        )
    if not output_rows:
        raise RuntimeError("No pseudo consensus rows produced")
    manifest = args.output_root / "pseudo_consensus.jsonl"
    write_jsonl(manifest, output_rows)
    summary = {
        "rows": len(output_rows),
        "beta": args.beta,
        "pixel_min_weight": args.pixel_min_weight,
        "manifest": str(manifest),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
