#!/usr/bin/env python3
from __future__ import annotations

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


def load(mode: str, split: str, max_bridge: int) -> list[dict]:
    rows = pqr.load_quality(ROOT, mode, split)
    return [row for row in rows if int(row["bridge_count"]) <= max_bridge]


def run(max_bridge: int) -> list[dict]:
    out = []
    for mode in MODES:
        train = load(mode, "validation", max_bridge)
        test = load(mode, "test", max_bridge)
        scorer = pqr.Ridge.fit(train, ridge=1.0, include_mode=False)
        out.append({"experiment": mode, "max_bridge": max_bridge, **pqr.evaluate(test, scorer)})
    train_union, test_union = [], []
    for mode in MODES:
        train_union.extend(load(mode, "validation", max_bridge))
        test_union.extend(load(mode, "test", max_bridge))
    scorer = pqr.Ridge.fit(train_union, ridge=1.0, include_mode=True)
    out.append({"experiment": "target_pooling+patch_correspondence", "max_bridge": max_bridge, **pqr.evaluate(test_union, scorer)})
    return out


def main() -> None:
    rows = run(6) + run(7)
    output = ROOT / "work/kvasir_1pct_anchors/propagation_quality_router_no_b7_check"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    for row in rows:
        print(
            row["experiment"],
            "max_bridge",
            row["max_bridge"],
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
