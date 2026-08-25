#!/usr/bin/env python3
"""Compare base-vs-e33 propagation quality with the topology and X3 frozen."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


FIELDS = ("q_return", "q_multi", "q_model", "b7", "gt_dice_evaluation_only")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def metrics(rows: list[dict]) -> dict:
    result = {"count": len(rows)}
    for field in FIELDS:
        values = np.asarray(
            [float(row[field]) for row in rows if row.get(field) is not None],
            dtype=np.float64,
        )
        if len(values):
            result[field] = {
                "mean": float(values.mean()),
                "p10": float(np.quantile(values, 0.1)),
                "median": float(np.quantile(values, 0.5)),
                "p90": float(np.quantile(values, 0.9)),
            }
    return result


def summarize(candidates: list[dict], best: list[dict], accepted: list[dict]) -> dict:
    by_bridge: dict[int, list[dict]] = defaultdict(list)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_bridge[int(row["bridge_count"])].append(row)
        by_mode[str(row["route_mode"])].append(row)
    return {
        "all_candidates": metrics(candidates),
        "best_per_target_unfiltered": metrics(best),
        "accepted_pool": metrics(accepted),
        "by_bridge": {str(key): metrics(value) for key, value in sorted(by_bridge.items())},
        "by_mode": {key: metrics(value) for key, value in sorted(by_mode.items())},
    }


def mean_deltas(base: dict, e33: dict) -> dict:
    output = {}
    for field in FIELDS:
        if field in base and field in e33:
            output[field] = e33[field]["mean"] - base[field]["mean"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for teacher in ("base", "e33"):
        parser.add_argument(f"--{teacher}-candidates", type=Path, required=True)
        parser.add_argument(f"--{teacher}-best", type=Path, required=True)
        parser.add_argument(f"--{teacher}-accepted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = summarize(
        read_jsonl(args.base_candidates),
        read_jsonl(args.base_best),
        read_jsonl(args.base_accepted),
    )
    e33 = summarize(
        read_jsonl(args.e33_candidates),
        read_jsonl(args.e33_best),
        read_jsonl(args.e33_accepted),
    )
    result = {
        "control": "same SAM3-base KNN topology and same frozen X3 q_model; propagation teacher only",
        "base": base,
        "e33": e33,
        "e33_minus_base_mean": {
            section: mean_deltas(base[section], e33[section])
            for section in ("all_candidates", "best_per_target_unfiltered", "accepted_pool")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["e33_minus_base_mean"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
