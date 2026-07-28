#!/usr/bin/env python3
"""Create deterministic, diversity-aware S27 X0-X5 training manifests."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def area(path: str | Path) -> float:
    return float((np.asarray(Image.open(path).convert("L")) > 127).mean())


def assign_area_bin(value: float, q1: float, q2: float) -> str:
    return "small" if value <= q1 else ("medium" if value <= q2 else "large")


def select_balanced(
    candidates: list[dict],
    count: int,
    initial: list[dict],
    train_domain_ratio: dict[str, float],
) -> list[dict]:
    """Greedy score with deterministic domain/area/visual-anchor balancing."""
    chosen = list(initial)
    remaining = {row["target_id"]: row for row in candidates if row not in chosen}
    domain_counts = Counter(row["dataset"] for row in chosen)
    area_counts = Counter(row["area_bin"] for row in chosen)
    anchor_counts = Counter(row["anchor_id"] for row in chosen)
    while len(chosen) < count and remaining:
        next_size = len(chosen) + 1
        best_key = None
        best_value = None
        for key, row in remaining.items():
            desired_domain = train_domain_ratio[row["dataset"]] * next_size
            domain_excess = max(
                0.0, (domain_counts[row["dataset"]] + 1 - desired_domain) / next_size
            )
            area_excess = max(
                0.0, (area_counts[row["area_bin"]] + 1) / next_size - 1.0 / 3.0
            )
            # Selected human anchor is a frozen visual-neighborhood proxy.
            anchor_excess = (anchor_counts[row["anchor_id"]] + 1) / next_size
            value = (
                float(row["q_route"])
                - 0.20 * domain_excess
                - 0.10 * area_excess
                - 0.05 * anchor_excess
            )
            tie = (value, float(row["q_route"]), key)
            if best_value is None or tie > best_value:
                best_key, best_value = key, tie
        row = remaining.pop(best_key)
        chosen.append(row)
        domain_counts[row["dataset"]] += 1
        area_counts[row["area_bin"]] += 1
        anchor_counts[row["anchor_id"]] += 1
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo568", type=Path, required=True)
    parser.add_argument("--tier-a", type=Path, required=True)
    parser.add_argument("--tier-b", type=Path, required=True)
    parser.add_argument("--tier-c", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cvc", type=int, default=490)
    parser.add_argument("--train-kvasir", type=int, default=800)
    args = parser.parse_args()

    original_source = read_jsonl(args.pseudo568)
    q_values = np.array([float(row["q_multi"]) for row in original_source])
    q_min, q_max = float(q_values.min()), float(q_values.max())
    original = []
    original_areas = []
    for row in original_source:
        row = dict(row)
        value = area(row["existing_pseudo_mask_path"])
        original_areas.append(value)
        normalized = (float(row["q_multi"]) - q_min) / max(q_max - q_min, 1e-12)
        row.update(
            {
                "target_id": row["image_id"],
                "pseudo_mask_path": row["existing_pseudo_mask_path"],
                "pseudo_consensus_path": row["existing_pseudo_mask_path"],
                "pixel_weight_path": None,
                "sample_type": "original",
                "tier": "original",
                "explicit_quality_weight": float(np.clip(normalized, 0.2, 1.0)),
                "area_ratio": value,
            }
        )
        original.append(row)
    q1, q2 = (float(x) for x in np.quantile(original_areas, [1 / 3, 2 / 3]))

    def enrich(rows: list[dict]) -> list[dict]:
        output = []
        for source in rows:
            row = dict(source)
            row["sample_type"] = "tier_a" if row["tier"] == "A" else "tier_b"
            row["area_ratio"] = float(row["selected_area_ratio"])
            row["area_bin"] = assign_area_bin(row["area_ratio"], q1, q2)
            output.append(row)
        return output

    tier_a = enrich(read_jsonl(args.tier_a))
    tier_b = enrich(read_jsonl(args.tier_b))
    tier_c = read_jsonl(args.tier_c)
    for row in original:
        row["area_bin"] = assign_area_bin(row["area_ratio"], q1, q2)

    train_total = args.train_cvc + args.train_kvasir
    domain_ratio = {
        "CVC-ClinicDB": args.train_cvc / train_total,
        "kvasir-seg": args.train_kvasir / train_total,
    }
    safe_all = sorted(tier_a + tier_b, key=lambda row: (-row["q_route"], row["target_id"]))
    a_sorted = sorted(tier_a, key=lambda row: (-row["q_route"], row["target_id"]))
    b_sorted = sorted(tier_b, key=lambda row: (-row["q_route"], row["target_id"]))

    def target_new(total_pseudo: int) -> list[dict]:
        requested = max(0, total_pseudo - len(original))
        initial = list(a_sorted)
        final_count = min(max(requested, len(initial)), len(safe_all))
        return select_balanced(b_sorted, final_count, initial, domain_ratio)

    sets = {
        "X0": [],
        "X1": a_sorted,
        "X2": target_new(700),
        "X3": target_new(900),
        "X4": select_balanced(b_sorted, len(safe_all), a_sorted, domain_ratio),
    }
    # First preregistered run uses C0, so X5 is intentionally an X4 alias.
    sets["X5"] = list(sets["X4"])
    summaries = {}
    for name, new_rows in sets.items():
        rows = original + new_rows
        output = args.output_dir / name
        write_jsonl(output / "pseudo_manifest.jsonl", rows)
        summary = {
            "experiment": name,
            "human_gt": 16,
            "original_pseudo": len(original),
            "new_tier_a": sum(row.get("tier") == "A" for row in new_rows),
            "new_tier_b": sum(row.get("tier") == "B" for row in new_rows),
            "new_pseudo": len(new_rows),
            "total_pseudo": len(rows),
            "dataset_new": dict(Counter(row["dataset"] for row in new_rows)),
            "area_bin_new": dict(Counter(row["area_bin"] for row in new_rows)),
            "anchor_new": dict(Counter(row["anchor_id"] for row in new_rows)),
            "x5_c_policy": "C0_ignore" if name == "X5" else None,
            "x5_is_x4_alias": name == "X5",
        }
        (output / "protocol.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summaries[name] = summary
    global_summary = {
        "tier_counts": {"A": len(tier_a), "B": len(tier_b), "C": len(tier_c)},
        "original568_quality_normalization": {
            "q_multi_min": q_min,
            "q_multi_max": q_max,
            "clip": [0.2, 1.0],
            "fixed_over_original568_only": True,
        },
        "original568_area_terciles": {"q33": q1, "q67": q2},
        "train_domain_ratio": domain_ratio,
        "sets": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(global_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(global_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
