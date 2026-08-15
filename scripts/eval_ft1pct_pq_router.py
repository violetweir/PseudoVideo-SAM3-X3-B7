#!/usr/bin/env python3
"""Score ft_1pct propagation-quality features with the b3-b6 ridge router."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
ANALYSIS = ROOT / "scripts/analyze_propagation_quality_router.py"
spec = importlib.util.spec_from_file_location("pqr", ANALYSIS)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {ANALYSIS}")
pqr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pqr
spec.loader.exec_module(pqr)

MODES = [
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(root: Path, mode: str, split: str) -> list[dict]:
    path = root / mode / f"propagation_quality_{split}/propagation_quality.jsonl"
    rows = read_jsonl(path)
    for row in rows:
        row["feature_mode"] = mode
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train, test = [], []
    for mode in MODES:
        train.extend(
            row for row in load(args.root, mode, "validation")
            if args.min_bridge <= int(row["bridge_count"]) <= args.max_bridge
        )
        test.extend(
            row for row in load(args.root, mode, "test")
            if args.min_bridge <= int(row["bridge_count"]) <= args.max_bridge
        )
    scorer = pqr.Ridge.fit(train, ridge=1.0, include_mode=True)
    per_mode = []
    for mode in MODES:
        m_train = [r for r in train if r["feature_mode"] == mode]
        m_test = [r for r in test if r["feature_mode"] == mode]
        m_scorer = pqr.Ridge.fit(m_train, ridge=1.0, include_mode=False)
        per_mode.append(
            {
                "experiment": mode,
                "min_bridge": args.min_bridge,
                "max_bridge": args.max_bridge,
                **pqr.evaluate(m_test, m_scorer),
            }
        )
    out = [
        {
            "experiment": "target_pooling+patch_correspondence",
            "min_bridge": args.min_bridge,
            "max_bridge": args.max_bridge,
            **pqr.evaluate(test, scorer),
        },
        *per_mode,
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for row in out:
        print(
            row["experiment"],
            f"b{args.min_bridge}-b{args.max_bridge}",
            "selected",
            f"{row['selected_dice']:.6f}",
            "oracle",
            f"{row['oracle_dice']:.6f}",
            "gap",
            f"{row['oracle_gap']:.6f}",
            "hist",
            json.dumps(row["histogram"], sort_keys=True),
        )


if __name__ == "__main__":
    main()
