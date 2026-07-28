#!/usr/bin/env python3
"""Freeze fair K=1..5 pseudo-video windows for the E1 multi-step study."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from prepare_t18_pseudovideo_pilot import descriptor, read_jsonl, write_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t18-root",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "T18_retrieval_pseudovideo"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
            "E1_multistep_pseudovideo"
        ),
    )
    parser.add_argument("--max-depth", type=int, default=5)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )


def main() -> None:
    args = parse_args()
    t18_root = args.t18_root.resolve()
    output_root = args.output_root.resolve()
    source_manifest = t18_root / "protocol/full_retrieval_manifest.jsonl"
    source_rows = read_jsonl(source_manifest)
    if not source_rows:
        raise RuntimeError(f"No T18 rows found in {source_manifest}")
    if args.max_depth < 1:
        raise ValueError("--max-depth must be positive")

    # Preserve T18's frozen query -> anchor assignment.  GT paths are carried only
    # for post-inference evaluation and are never read by this preparation step.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[(row["query_split"], row["reference_merged_id"])].append(row)

    descriptor_cache: dict[str, np.ndarray] = {}

    def get_descriptor(path: str) -> np.ndarray:
        if path not in descriptor_cache:
            descriptor_cache[path] = descriptor(path)
        return descriptor_cache[path]

    mother_paths: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for (split, anchor_id), group in sorted(grouped.items()):
        exemplar = group[0]
        anchor_descriptor = get_descriptor(exemplar["reference_image_path"])
        remaining = {row["query_merged_id"]: row for row in group}
        ordered: list[dict[str, Any]] = []
        predecessor_descriptor = anchor_descriptor
        predecessor_id = anchor_id
        while remaining:
            scored = [
                (
                    float(predecessor_descriptor @ get_descriptor(row["query_image_path"])),
                    merged_id,
                    row,
                )
                for merged_id, row in remaining.items()
            ]
            similarity, selected_id, selected = max(
                scored, key=lambda item: (item[0], item[1])
            )
            ordered.append(
                {
                    **selected,
                    "path_rank": len(ordered) + 1,
                    "predecessor_merged_id": predecessor_id,
                    "predecessor_similarity": similarity,
                }
            )
            del remaining[selected_id]
            predecessor_descriptor = get_descriptor(selected["query_image_path"])
            predecessor_id = selected_id

        group_key = f"{split}::{anchor_id}"
        mother_paths.append(
            {
                "group_key": group_key,
                "query_split": split,
                "anchor_merged_id": anchor_id,
                "anchor_image_path": exemplar["reference_image_path"],
                "anchor_mask_path": exemplar["reference_mask_path"],
                "anchor_box_xywh_normalized": exemplar[
                    "reference_box_xywh_normalized"
                ],
                "group_size": len(ordered),
                "ordered_query_ids": [row["query_merged_id"] for row in ordered],
            }
        )
        for target_index, target in enumerate(ordered):
            for depth in range(1, args.max_depth + 1):
                start = target_index - depth + 1
                if start < 0:
                    continue
                context = ordered[start : target_index + 1]
                sequence_key = (
                    f"{group_key}::target={target['query_merged_id']}::K={depth}"
                )
                windows.append(
                    {
                        "sequence_id": hashlib.sha256(
                            sequence_key.encode("utf-8")
                        ).hexdigest()[:24],
                        "group_key": group_key,
                        "group_size": len(ordered),
                        "query_split": split,
                        "anchor_merged_id": anchor_id,
                        "anchor_image_path": exemplar["reference_image_path"],
                        "anchor_mask_path": exemplar["reference_mask_path"],
                        "anchor_box_xywh_normalized": exemplar[
                            "reference_box_xywh_normalized"
                        ],
                        "target_merged_id": target["query_merged_id"],
                        "target_source_dataset": target["query_source_dataset"],
                        "target_image_path": target["query_image_path"],
                        "target_mask_path": target["query_mask_path"],
                        "target_path_rank": target["path_rank"],
                        "depth": depth,
                        "frame_query_ids": [
                            row["query_merged_id"] for row in context
                        ],
                        "frame_image_paths": [
                            row["query_image_path"] for row in context
                        ],
                        "frame_mask_paths_evaluation_only": [
                            row["query_mask_path"] for row in context
                        ],
                        "matched_k5": target_index >= args.max_depth - 1,
                        "query_gt_used_for_ordering_or_prompting": False,
                    }
                )

    windows.sort(
        key=lambda row: (
            ("train", "validation", "test").index(row["query_split"]),
            row["depth"],
            row["group_key"],
            row["target_path_rank"],
        )
    )
    mother_paths_payload = dump_jsonl(mother_paths)
    windows_payload = dump_jsonl(windows)
    protocol_dir = output_root / "protocol"
    write_once(protocol_dir / "mother_paths.jsonl", mother_paths_payload)
    write_once(protocol_dir / "windows_k1_k5.jsonl", windows_payload)

    counts_by_depth = {
        str(depth): sum(row["depth"] == depth for row in windows)
        for depth in range(1, args.max_depth + 1)
    }
    matched_by_split = {
        split: len(
            {
                row["target_merged_id"]
                for row in windows
                if row["query_split"] == split and row["matched_k5"]
            }
        )
        for split in ("train", "validation", "test")
    }
    protocol = {
        "name": "E1 multi-step pseudo-video propagation",
        "source_t18_manifest": str(source_manifest),
        "source_t18_manifest_sha256": sha256_file(source_manifest),
        "support_and_anchor_assignment": "exactly frozen from T18",
        "grouping": "(query_split, T18 anchor_merged_id)",
        "mother_path_rule": (
            "x1=max cosine(anchor, remaining); xi=max cosine(x[i-1], remaining); "
            "ties choose lexicographically largest merged_id"
        ),
        "window_rule": (
            "for each target and K, use its K-length suffix on the single frozen "
            "mother path; no K-specific reordering and no wrapping/repeated frames"
        ),
        "max_depth": args.max_depth,
        "canvas": "512x512 RGB squash, all structures identical",
        "primary_cohort": "matched_k5 targets eligible at every K=1..5",
        "counts_by_depth": counts_by_depth,
        "matched_k5_unique_targets_by_split": matched_by_split,
        "query_gt_used_for_ordering_or_prompting": False,
        "mother_paths_sha256": hashlib.sha256(mother_paths_payload).hexdigest(),
        "windows_sha256": hashlib.sha256(windows_payload).hexdigest(),
    }
    protocol_path = protocol_dir / "protocol.json"
    payload = (json.dumps(protocol, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_once(protocol_path, payload)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
