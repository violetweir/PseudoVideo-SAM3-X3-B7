#!/usr/bin/env python3
"""Audit graph changes and reuse exactly identical checkpoint/route inferences."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


MODES = (
    "sam3enc_anchor_conditioned_target_pooling",
    "sam3enc_anchor_conditioned_patch_correspondence",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_pair(old_rows: list[dict], new_rows: list[dict]) -> dict:
    old_by_target = {(row["target_id"], int(row["bridge_count"])): row for row in old_rows}
    new_by_target = {(row["target_id"], int(row["bridge_count"])): row for row in new_rows}
    if set(old_by_target) != set(new_by_target):
        raise RuntimeError("base/e33 route pools do not cover the same target/bridge pairs")
    per_bridge: dict[int, list[dict]] = defaultdict(list)
    for key in sorted(old_by_target):
        previous = old_by_target[key]
        updated = new_by_target[key]
        previous_bridges = set(previous["bridge_ids"])
        updated_bridges = set(updated["bridge_ids"])
        union = previous_bridges | updated_bridges
        per_bridge[key[1]].append(
            {
                "same_route": previous["route_id"] == updated["route_id"],
                "same_anchor": previous["anchor_id"] == updated["anchor_id"],
                "same_bridge_sequence": previous["bridge_ids"] == updated["bridge_ids"],
                "bridge_jaccard": len(previous_bridges & updated_bridges) / len(union) if union else 1.0,
            }
        )

    def aggregate(rows: list[dict]) -> dict:
        return {
            "routes": len(rows),
            "same_route": sum(row["same_route"] for row in rows),
            "changed_route": sum(not row["same_route"] for row in rows),
            "same_anchor": sum(row["same_anchor"] for row in rows),
            "changed_anchor": sum(not row["same_anchor"] for row in rows),
            "same_bridge_sequence": sum(row["same_bridge_sequence"] for row in rows),
            "mean_bridge_jaccard": float(np.mean([row["bridge_jaccard"] for row in rows])),
        }

    all_rows = [row for rows in per_bridge.values() for row in rows]
    return {
        "overall": aggregate(all_rows),
        "by_bridge": {str(bridge): aggregate(rows) for bridge, rows in sorted(per_bridge.items())},
    }


def routes(args: argparse.Namespace) -> None:
    result = {
        "base_topology_root": str(args.base_root.resolve()),
        "e33_topology_root": str(args.e33_root.resolve()),
        "splits": {},
    }
    for split in args.splits:
        result["splits"][split] = {}
        for mode in MODES:
            old = read_jsonl(args.base_root / mode / f"{split}_pool0_stage1/routes.jsonl")
            updated = read_jsonl(args.e33_root / mode / f"{split}_pool0_stage1/routes.jsonl")
            if len(updated) != 700:
                raise RuntimeError(f"Expected 700 e33 routes for {mode}/{split}, got {len(updated)}")
            result["splits"][split][mode] = summarize_pair(old, updated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def seed(args: argparse.Namespace) -> None:
    destination = args.quality_root / args.mode / f"propagation_quality_{args.split}/propagation_quality.jsonl"
    requested = read_jsonl(args.route_root / args.mode / f"{args.split}_pool0_stage1/routes.jsonl")
    cache: dict[str, dict] = {}
    sources = [*args.source_roots, args.quality_root]
    for source_root in sources:
        for source_mode in MODES:
            path = source_root / source_mode / f"propagation_quality_{args.split}/propagation_quality.jsonl"
            for row in read_jsonl(path):
                if row.get("status") != "success":
                    continue
                mask = Path(row["forward_mask_path"])
                if mask.exists():
                    cache[row["route_id"]] = row
    resumed = []
    for route in requested:
        previous = cache.get(route["route_id"])
        if previous is not None:
            resumed.append({**previous, **route})
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in resumed),
        encoding="utf-8",
    )
    result = {
        "split": args.split,
        "mode": args.mode,
        "routes": len(requested),
        "reused_identical_checkpoint_and_route": len(resumed),
        "new_inferences_required": len(requested) - len(resumed),
        "destination": str(destination),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("routes")
    audit_parser.add_argument("--base-root", type=Path, required=True)
    audit_parser.add_argument("--e33-root", type=Path, required=True)
    audit_parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.set_defaults(func=routes)

    seed_parser = subparsers.add_parser("seed")
    seed_parser.add_argument("--route-root", type=Path, required=True)
    seed_parser.add_argument("--quality-root", type=Path, required=True)
    seed_parser.add_argument("--source-roots", type=Path, nargs="+", required=True)
    seed_parser.add_argument("--mode", choices=MODES, required=True)
    seed_parser.add_argument("--split", choices=("validation", "test"), required=True)
    seed_parser.set_defaults(func=seed)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
