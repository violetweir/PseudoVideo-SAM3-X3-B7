#!/usr/bin/env python3
"""Reuse identical completed e33 validation routes for Round-2D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
ROUND2A = ROOT / "work/rerun_c0_256_round2a_fixed_knn_e33/quality_root"
ROUND2C = ROOT / "work/rerun_c0_256_round2c_lesion_anchor_validation/quality_root/round2c_all_anchors_b0"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--mode-key", required=True)
    parser.add_argument("--round2a-mode", required=True)
    args = parser.parse_args()
    routes_path = args.quality_root / args.mode_key / "validation_pool0_stage1/routes.jsonl"
    output_path = args.quality_root / args.mode_key / "propagation_quality_validation/propagation_quality.jsonl"
    routes = read(routes_path)
    wanted = {row["route_id"] for row in routes}
    existing: dict[str, dict] = {}
    source_paths = [
        ROUND2C / "propagation_quality_validation/propagation_quality.jsonl",
        ROUND2A / args.round2a_mode / "propagation_quality_validation/propagation_quality.jsonl",
        output_path,
    ]
    for path in source_paths:
        for row in read(path):
            if row.get("status") == "success" and row.get("route_id") in wanted:
                existing[row["route_id"]] = row
    seeded = [existing[row["route_id"]] for row in routes if row["route_id"] in existing]
    write(output_path, seeded)
    print(json.dumps({"routes": len(routes), "seeded": len(seeded), "pending": len(routes) - len(seeded)}))


if __name__ == "__main__":
    main()
