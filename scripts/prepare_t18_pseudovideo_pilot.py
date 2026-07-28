#!/usr/bin/env python3
"""Freeze a train-only pseudo-video pilot and image-only support retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PILOT_PER_SOURCE = 16
PILOT_SALT = "t18-pseudovideo-pilot-v1-20260724"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t17-root",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T17_autonomous_target_1pct"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T18_retrieval_pseudovideo"
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def descriptor(image_path: str) -> np.ndarray:
    rgb = np.asarray(
        Image.open(image_path).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]
    ).reshape(-1)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    low_frequency = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1)
    moments = np.concatenate(
        [
            rgb.reshape(-1, 3).mean(axis=0),
            rgb.reshape(-1, 3).std(axis=0),
            np.asarray([rgb.shape[1] / rgb.shape[0]], dtype=np.float64),
        ]
    )
    vector = np.concatenate(
        [
            histogram.astype(np.float64),
            low_frequency.astype(np.float64) / 255.0,
            moments.astype(np.float64) / np.asarray([255] * 6 + [1]),
        ]
    )
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-12)


def tight_box(mask_path: str) -> list[float]:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError(f"Empty support mask: {mask_path}")
    height, width = mask.shape
    x1 = float(xs.min() / width)
    y1 = float(ys.min() / height)
    x2 = float((xs.max() + 1) / width)
    y2 = float((ys.max() + 1) / height)
    return [x1, y1, x2 - x1, y2 - y1]


def write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    args = parse_args()
    t17_root = args.t17_root.resolve()
    output_root = args.output_root.resolve()
    manifest = read_jsonl(t17_root / "protocol/merged_manifest.jsonl")
    support = read_jsonl(t17_root / "protocol/support_manifest.jsonl")
    support_ids = {row["merged_id"] for row in support}
    if len(support_ids) != 16 or any(row["split"] != "train" for row in support):
        raise RuntimeError("Expected the frozen 16-image train-only support set")

    pilot = []
    for source_dataset in ("CVC-ClinicDB", "kvasir-seg"):
        candidates = [
            row
            for row in manifest
            if row["split"] == "train"
            and row["source_dataset"] == source_dataset
            and row["merged_id"] not in support_ids
        ]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{PILOT_SALT}::{row['merged_id']}".encode("utf-8")
            ).hexdigest()
        )
        pilot.extend(candidates[:PILOT_PER_SOURCE])
    pilot.sort(key=lambda row: str(row["merged_id"]))

    support_descriptors = {
        row["merged_id"]: descriptor(row["frozen_image_path"]) for row in support
    }
    retrieval_rows = []
    for query in pilot:
        query_descriptor = descriptor(query["image_path"])
        similarities = {
            reference["merged_id"]: float(
                query_descriptor @ support_descriptors[reference["merged_id"]]
            )
            for reference in support
        }
        reference = max(
            support,
            key=lambda row: (similarities[row["merged_id"]], row["merged_id"]),
        )
        retrieval_rows.append(
            {
                "query_merged_id": query["merged_id"],
                "query_source_dataset": query["source_dataset"],
                "query_split": query["split"],
                "query_image_path": query["image_path"],
                "query_mask_path": query["mask_path"],
                "reference_merged_id": reference["merged_id"],
                "reference_source_dataset": reference["source_dataset"],
                "reference_image_path": reference["frozen_image_path"],
                "reference_mask_path": reference["frozen_mask_path"],
                "reference_box_xywh_normalized": tight_box(
                    reference["frozen_mask_path"]
                ),
                "retrieval_similarity": similarities[reference["merged_id"]],
                "retrieval_rule": "maximum cosine of fixed HSV histogram + 8x8 grayscale descriptor",
                "query_gt_used_for_retrieval": False,
            }
        )

    payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in retrieval_rows
    )
    retrieval_path = output_root / "protocol/pilot_retrieval_manifest.jsonl"
    write_once(retrieval_path, payload)
    protocol = {
        "name": "T18 retrieval-augmented two-frame pseudo-video pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_queries": len(retrieval_rows),
        "pilot_query_split": "train non-support only",
        "pilot_queries_per_source": PILOT_PER_SOURCE,
        "pilot_selection_salt": PILOT_SALT,
        "support_count": len(support),
        "support_source": "frozen T17 train-only 1pct support",
        "support_prompt": "tight normalized box computed from frozen train GT mask",
        "retrieval_input": "raw image pixels only; no query mask, validation, or test data",
        "pseudo_video": "frame 0 = resized support image; frame 1 = query image",
        "sam3_text_prompt": None,
        "sam3_internal_visual_placeholder": True,
        "pilot_retrieval_manifest_sha256": sha256_bytes(payload),
        "decision_policy": "decide whether to scale using train-only pilot before any validation/test run",
    }
    protocol_path = output_root / "protocol/protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (
            old["pilot_retrieval_manifest_sha256"]
            != protocol["pilot_retrieval_manifest_sha256"]
        ):
            raise RuntimeError("Frozen T18 protocol mismatch")
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    for row in retrieval_rows:
        print(
            f"{row['query_merged_id']} <- {row['reference_merged_id']} "
            f"similarity={row['retrieval_similarity']:.6f}"
        )


if __name__ == "__main__":
    main()
