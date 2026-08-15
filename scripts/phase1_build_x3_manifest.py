#!/usr/bin/env python3
"""Phase-1 Stage 3: combine original HQ + Tier A + Tier B into the X3 manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--tier-a", type=Path, required=True)
    parser.add_argument("--tier-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original = read_jsonl(args.original)
    tier_a = read_jsonl(args.tier_a)
    tier_b = read_jsonl(args.tier_b)
    seen: set[str] = set()
    rows = []
    for pool, name in (
        (original, "original"),
        (tier_a, "tier_a"),
        (tier_b, "tier_b"),
    ):
        for row in pool:
            target = row["target_id"]
            if target in seen:
                raise RuntimeError(f"Duplicate target {target}")
            seen.add(target)
            out = {
                "target_id": target,
                "pseudo_mask_path": row["pseudo_mask_path"],
                "sample_type": name,
                "explicit_quality_weight": float(row.get("explicit_quality_weight", 1.0)),
            }
            if row.get("pseudo_consensus_path"):
                out["pseudo_consensus_path"] = row["pseudo_consensus_path"]
            if row.get("pixel_weight_path"):
                out["pixel_weight_path"] = row["pixel_weight_path"]
            for key in ("q_multi", "q_model_mean", "q_model_var"):
                if row.get(key) is not None:
                    out[key] = row[key]
            rows.append(out)
    write_jsonl(args.output, rows)
    summary = {
        "total": len(rows),
        "sample_types": dict(Counter(r["sample_type"] for r in rows)),
    }
    (args.output.with_name(args.output.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
