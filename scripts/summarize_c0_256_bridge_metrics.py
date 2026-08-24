#!/usr/bin/env python3
"""Summarize per-bridge propagation metrics across route modes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--min-bridge", type=int, default=0)
    parser.add_argument("--max-bridge", type=int, default=6)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    args = parser.parse_args()

    by_mode: dict[str, dict[int, list[dict]]] = {}
    combined: dict[int, list[dict]] = defaultdict(list)
    for mode in args.modes:
        path = (
            args.quality_root
            / mode
            / f"propagation_quality_{args.split}"
            / "propagation_quality.jsonl"
        )
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in read_jsonl(path):
            bridge = int(row["bridge_count"])
            if args.min_bridge <= bridge <= args.max_bridge:
                grouped[bridge].append(row)
                combined[bridge].append(row)
        by_mode[mode] = grouped

    def metrics(rows: list[dict]) -> dict:
        dice = [float(row["gt_dice_evaluation_only"]) for row in rows]
        cycle = [float(row["q_cycle"]) for row in rows]
        return {
            "n": len(rows),
            "dice": float(np.mean(dice)),
            "dice_median": float(np.median(dice)),
            "q_cycle": float(np.mean(cycle)),
        }

    result = {
        "split": args.split,
        "min_bridge": args.min_bridge,
        "max_bridge": args.max_bridge,
        "modes": {
            mode: {
                f"bridge_{bridge}": metrics(grouped[bridge])
                for bridge in range(args.min_bridge, args.max_bridge + 1)
            }
            for mode, grouped in by_mode.items()
        },
        "combined": {
            f"bridge_{bridge}": metrics(combined[bridge])
            for bridge in range(args.min_bridge, args.max_bridge + 1)
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    columns = ["scope", *[f"bridge_{bridge}" for bridge in range(args.min_bridge, args.max_bridge + 1)]]
    scopes = [("combined", result["combined"]), *[(mode, result["modes"][mode]) for mode in args.modes]]
    args.output_tsv.write_text(
        "\t".join(columns)
        + "\n"
        + "".join(
            scope
            + "\t"
            + "\t".join(str(values[f"bridge_{bridge}"]["dice"]) for bridge in range(args.min_bridge, args.max_bridge + 1))
            + "\n"
            for scope, values in scopes
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
