#!/usr/bin/env python3
"""Mainline-style pseudo568 selection for Phase 1 (Kvasir 1%).

Mirrors prepare_t22_training.py on the ft_1pct b3-b6 propagation-quality
router output: per train target, take the validation-trained ridge scorer's
Top-1 candidate, then keep targets with
    q_multi >= 0.90  AND  q_return >= 0.95
where q_return is the ft_1pct anchor-cycle consistency (q_cycle) and q_multi
is the mean pairwise agreement across the target's b3-b6 candidate pool.

Per-image weight follows build_s27_expansion_sets.py: normalized q_multi over
the accepted pool, clipped to [0.2, 1.0].
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PQR_PATH = ROOT / "scripts/analyze_propagation_quality_router.py"
spec = importlib.util.spec_from_file_location("pqr", PQR_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {PQR_PATH}")
pqr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pqr
spec.loader.exec_module(pqr)


MODES = (
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_binary(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau-multi", type=float, default=0.90)
    parser.add_argument("--tau-return", type=float, default=0.95)
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    args = parser.parse_args()

    # Validation-trained ridge scorer on the ft_1pct validation features
    # (same protocol as eval_ft1pct_pq_router.py).
    validation = []
    for mode in MODES:
        rows = read_jsonl(
            args.quality_root
            / mode
            / "propagation_quality_validation/propagation_quality.jsonl"
        )
        for row in rows:
            row["feature_mode"] = mode
        validation.extend(rows)
    validation = [
        row
        for row in validation
        if args.min_bridge <= int(row["bridge_count"]) <= args.max_bridge
    ]
    scorer = pqr.Ridge.fit(validation, ridge=1.0, include_mode=True)

    candidates_by_target: dict[str, list[dict]] = {}
    for mode in MODES:
        rows = read_jsonl(
            args.quality_root
            / mode
            / f"propagation_quality_train/propagation_quality.jsonl"
        )
        for row in rows:
            if not args.min_bridge <= int(row["bridge_count"]) <= args.max_bridge:
                continue
            row["feature_mode"] = mode
            candidates_by_target.setdefault(row["target_id"], []).append(row)
    print(f"train targets with candidates: {len(candidates_by_target)}", flush=True)

    selected = []
    for target, cands in sorted(candidates_by_target.items()):
        chosen = max(
            cands,
            key=lambda row: (
                scorer.score(row),
                row["feature_mode"],
                -int(row["bridge_count"]),
                row["route_id"],
            ),
        )
        masks = [load_binary(row["forward_mask_path"]) for row in cands]
        q_multi = float(
            np.mean(
                [
                    dice(a, b)
                    for i, a in enumerate(masks)
                    for b in masks[i + 1 :]
                ]
            )
            if len(masks) > 1
            else 1.0
        )
        q_return = float(chosen.get("q_cycle", 0.0))
        selected.append(
            {
                "target_id": target,
                "pseudo_mask_path": str(Path(chosen["forward_mask_path"]).resolve()),
                "q_multi": q_multi,
                "q_return": q_return,
                "route_id": chosen["route_id"],
                "feature_mode": chosen["feature_mode"],
                "bridge_count": int(chosen["bridge_count"]),
                "anchor_id": chosen["anchor_id"],
                "routeco_selected_score": float(scorer.score(chosen)),
                "n_candidates": len(cands),
            }
        )

    accepted = [
        row
        for row in selected
        if row["q_multi"] >= args.tau_multi and row["q_return"] >= args.tau_return
    ]
    q_values = np.array([row["q_multi"] for row in accepted])
    q_min, q_max = float(q_values.min()), float(q_values.max())
    output_rows = []
    for row in accepted:
        normalized = (row["q_multi"] - q_min) / max(q_max - q_min, 1e-12)
        output_rows.append(
            {
                "target_id": row["target_id"],
                "pseudo_mask_path": row["pseudo_mask_path"],
                "sample_type": "original",
                "explicit_quality_weight": float(np.clip(normalized, 0.2, 1.0)),
                "q_multi": row["q_multi"],
                "q_return": row["q_return"],
                "route_id": row["route_id"],
                "feature_mode": row["feature_mode"],
                "bridge_count": row["bridge_count"],
                "anchor_id": row["anchor_id"],
                "routeco_selected_score": row["routeco_selected_score"],
            }
        )
    write_jsonl(args.output, output_rows)
    summary = {
        "tau_multi": args.tau_multi,
        "tau_return": args.tau_return,
        "total_targets": len(selected),
        "accepted": len(accepted),
        "q_multi": {
            "min": q_min,
            "max": q_max,
            "mean": float(q_values.mean()),
            "median": float(np.median(q_values)),
        },
        "q_return": {
            "min": float(min(row["q_return"] for row in accepted)),
            "mean": float(np.mean([row["q_return"] for row in accepted])),
            "median": float(np.median([row["q_return"] for row in accepted])),
        },
        "selected_counts": dict(
            Counter(f"{row['feature_mode']}:bridge_{row['bridge_count']}" for row in accepted)
        ),
    }
    (args.output.with_name(args.output.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
