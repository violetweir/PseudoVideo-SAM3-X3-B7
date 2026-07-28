#!/usr/bin/env python3
"""Freeze leakage-safe T22 anchor and pseudo-video training manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t21-root",
        type=Path,
        required=True,
        help="T21 dynamic pseudo-video output root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where the 16-anchor and 568-pseudo manifests are frozen.",
    )
    parser.add_argument("--tau-multi", type=float, default=0.90)
    parser.add_argument("--tau-return", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase = args.t21_root / "round0_train"
    selections = read_jsonl(phase / "selections.jsonl")
    routes = {
        row["route_id"]: row for row in read_jsonl(phase / "routes.jsonl")
    }
    anchors = read_jsonl(phase / "anchor_pool_input.jsonl")
    human = [
        row
        for row in anchors
        if row.get(
            "is_human_gt", row.get("is_human", row.get("anchor_is_human", False))
        )
        and int(row.get("generation", row.get("anchor_generation", 0))) == 0
    ]
    if len(human) != 16:
        raise RuntimeError(f"Expected exactly 16 human anchors, found {len(human)}")

    anchor_rows = []
    for row in sorted(human, key=lambda item: item.get("anchor_id", "")):
        anchor_rows.append(
            {
                "anchor_id": row["anchor_id"],
                "image_path": row.get("image_path", row.get("anchor_image_path")),
                "mask_path": row.get("mask_path", row.get("anchor_mask_path")),
                "box_xywh_normalized": row.get(
                    "box_xywh_normalized", row.get("anchor_box_xywh_normalized")
                ),
                "source_dataset": row.get(
                    "source_dataset", row["anchor_id"].split("::", 1)[0]
                ),
                "split": "train",
                "is_human_gt": True,
                "generation": 0,
            }
        )

    pseudo_rows = []
    for selection in selections:
        if selection["target_split"] != "train":
            continue
        if selection["q_multi"] < args.tau_multi:
            continue
        if selection["q_return"] < args.tau_return:
            continue
        route = routes[selection["selected_route_id"]]
        if not route["anchor_is_human"] or int(route["anchor_generation"]) != 0:
            raise RuntimeError(f"Non-human route slipped through: {route['route_id']}")
        if route["target_gt_used_for_search_or_inference"]:
            raise RuntimeError(f"GT leakage flag on route {route['route_id']}")
        pseudo_rows.append(
            {
                "target_id": route["target_id"],
                "target_image_path": route["target_image_path"],
                "pseudo_mask_path": selection["selected_mask_path"],
                "pseudo_mask_sha256": selection["selected_mask_sha256"],
                "anchor_id": route["anchor_id"],
                "anchor_image_path": route["anchor_image_path"],
                "anchor_mask_path": route["anchor_mask_path"],
                "anchor_box_xywh_normalized": route[
                    "anchor_box_xywh_normalized"
                ],
                "bridge_ids": route["bridge_ids"],
                "bridge_image_paths": route["bridge_image_paths"],
                "route_id": route["route_id"],
                "route_type": route["route_type"],
                "q_multi": selection["q_multi"],
                "q_return": selection["q_return"],
                "target_split": "train",
                "anchor_is_human": True,
                "anchor_generation": 0,
                "target_gt_used": False,
            }
        )
    pseudo_rows.sort(key=lambda row: row["target_id"])
    if len(pseudo_rows) != 568:
        raise RuntimeError(f"Expected frozen 568-row gate, found {len(pseudo_rows)}")

    protocol_root = args.output_root / "protocol"
    anchor_path = protocol_root / "human16_anchors.jsonl"
    pseudo_path = protocol_root / "pseudo_train_tau_m090_r095.jsonl"
    write_jsonl(anchor_path, anchor_rows)
    write_jsonl(pseudo_path, pseudo_rows)
    meta = {
        "name": "T22 Pseudo-Video-Aware SAM3 Adaptation with Mutual Student Learning",
        "t21_root": str(args.t21_root.resolve()),
        "principle": "pseudo labels are supervision only and can never become anchors",
        "tau_multi": args.tau_multi,
        "tau_return": args.tau_return,
        "anchor_count": len(anchor_rows),
        "pseudo_count": len(pseudo_rows),
        "route_type_counts": dict(Counter(row["route_type"] for row in pseudo_rows)),
        "dataset_counts": dict(
            Counter(row["target_id"].split("::", 1)[0] for row in pseudo_rows)
        ),
        "anchor_usage_counts": dict(Counter(row["anchor_id"] for row in pseudo_rows)),
        "anchor_manifest_sha256": sha256(anchor_path),
        "pseudo_manifest_sha256": sha256(pseudo_path),
        "forbidden_sources": ["round0_train/new_anchors.jsonl", "val GT", "test GT"],
    }
    (protocol_root / "protocol.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
