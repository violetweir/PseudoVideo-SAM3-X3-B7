#!/usr/bin/env python3
"""Freeze full T18 retrieval using the already frozen pilot rule and support set."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from prepare_t18_pseudovideo_pilot import descriptor, read_jsonl, tight_box, write_once


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


def main() -> None:
    args = parse_args()
    t17_root = args.t17_root.resolve()
    output_root = args.output_root.resolve()
    manifest = read_jsonl(t17_root / "protocol/merged_manifest.jsonl")
    support = read_jsonl(t17_root / "protocol/support_manifest.jsonl")
    support_ids = {row["merged_id"] for row in support}
    pilot_protocol = json.loads(
        (output_root / "protocol/protocol.json").read_text(encoding="utf-8")
    )
    pilot_summary = json.loads(
        (output_root / "pilot/summary.json").read_text(encoding="utf-8")
    )
    support_descriptors = {
        row["merged_id"]: descriptor(row["frozen_image_path"]) for row in support
    }

    queries = [
        row
        for row in manifest
        if not (row["split"] == "train" and row["merged_id"] in support_ids)
    ]
    retrieval_rows = []
    for query in queries:
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
    retrieval_rows.sort(
        key=lambda row: (
            ("train", "validation", "test").index(row["query_split"]),
            row["query_merged_id"],
        )
    )
    payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in retrieval_rows
    )
    manifest_path = output_root / "protocol/full_retrieval_manifest.jsonl"
    write_once(manifest_path, payload)
    counts = {
        split: sum(row["query_split"] == split for row in retrieval_rows)
        for split in ("train", "validation", "test")
    }
    protocol = {
        "name": "T18 full retrieval-augmented two-frame pseudo-video",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "query_counts": counts,
        "query_total": len(retrieval_rows),
        "excluded_queries": "the 16 support train images only",
        "support_count": len(support),
        "support_manifest_sha256": hashlib.sha256(
            (t17_root / "protocol/support_manifest.jsonl").read_bytes()
        ).hexdigest(),
        "retrieval_rule": "frozen before validation/test: maximum cosine of HSV histogram + 8x8 grayscale descriptor",
        "reference_prompt": "tight box from support train GT",
        "query_gt_used_for_retrieval_or_prompting": False,
        "sam3_text_prompt": None,
        "primary_prediction": "frame-1 top propagated mask",
        "predeclared_fallback": "when frame-1 has zero masks, use frozen T17 support_ridge result",
        "no_confidence_threshold_tuning": True,
        "pilot_protocol_sha256": hashlib.sha256(
            (output_root / "protocol/protocol.json").read_bytes()
        ).hexdigest(),
        "pilot_train_only_result": pilot_summary,
        "full_retrieval_manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }
    protocol_path = output_root / "protocol/full_protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (
            old["full_retrieval_manifest_sha256"]
            != protocol["full_retrieval_manifest_sha256"]
        ):
            raise RuntimeError("Frozen full protocol mismatch")
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
