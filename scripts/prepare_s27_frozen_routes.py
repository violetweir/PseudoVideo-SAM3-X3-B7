#!/usr/bin/env python3
"""Reuse and freeze the T21 three-route topology for the S27 remaining pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remaining", type=Path, required=True)
    parser.add_argument("--t21-route-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    remaining_rows = read_jsonl(args.remaining)
    remaining_ids = {row["image_id"] for row in remaining_rows}
    routes = [
        row
        for row in read_jsonl(args.t21_route_results)
        if row["target_id"] in remaining_ids
    ]
    by_target: dict[str, list[dict]] = defaultdict(list)
    for row in routes:
        by_target[row["target_id"]].append(row)

    required_types = {"direct", "one_bridge", "two_bridges"}
    errors = []
    route_ids = [row["route_id"] for row in routes]
    if len(route_ids) != len(set(route_ids)):
        errors.append("duplicate route_id")
    for target_id in sorted(remaining_ids):
        target_routes = by_target.get(target_id, [])
        types = [row["route_type"] for row in target_routes]
        if len(target_routes) != 3 or set(types) != required_types:
            errors.append(f"{target_id}: count={len(target_routes)} types={types}")
        for row in target_routes:
            if not row.get("anchor_is_human"):
                errors.append(f"{row['route_id']}: non-human anchor")
            if row.get("target_split") != "train":
                errors.append(f"{row['route_id']}: non-train target")
            if not Path(row["forward_mask_path"]).is_file():
                errors.append(f"{row['route_id']}: missing binary mask")
            if row.get("status") != "success":
                errors.append(f"{row['route_id']}: status={row.get('status')}")
            if len(row.get("bridge_ids", [])) != int(row["bridge_count"]):
                errors.append(f"{row['route_id']}: bridge count mismatch")
    if set(by_target) != remaining_ids:
        errors.append(
            f"target coverage mismatch: missing={len(remaining_ids-set(by_target))}, "
            f"extra={len(set(by_target)-remaining_ids)}"
        )
    if errors:
        raise RuntimeError("Frozen-route audit failed:\n" + "\n".join(errors[:50]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "route_results.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in sorted(routes, key=lambda x: (x["target_id"], x["bridge_count"])):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "source_t21_route_results": str(args.t21_route_results.resolve()),
        "source_t21_sha256": sha256(args.t21_route_results),
        "remaining_manifest": str(args.remaining.resolve()),
        "remaining_manifest_sha256": sha256(args.remaining),
        "target_count": len(remaining_ids),
        "route_count": len(routes),
        "route_type_counts": dict(Counter(row["route_type"] for row in routes)),
        "human_anchor_only": all(row["anchor_is_human"] for row in routes),
        "train_target_only": all(row["target_split"] == "train" for row in routes),
        "three_complete_routes_per_target": True,
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
    }
    (args.output_dir / "routes_reuse_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
