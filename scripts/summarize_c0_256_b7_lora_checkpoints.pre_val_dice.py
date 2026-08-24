#!/usr/bin/env python3
"""Summarize validation B7 results and select the best LoRA epoch."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--val-stats", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--best-epoch-output", type=Path, required=True)
    args = parser.parse_args()

    val_stats = {}
    if args.val_stats.exists():
        for line in args.val_stats.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                val_stats[int(row["epoch"])] = row

    rows = []
    for path in sorted(args.eval_root.glob("e*/b7_selected_validation.summary.json")):
        match = re.fullmatch(r"e(\d+)", path.parent.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        summary = json.loads(path.read_text())
        train_stats = val_stats.get(epoch, {})
        rows.append(
            {
                "epoch": epoch,
                "b7_selected_validation_dice": summary.get(
                    "selected_gt_dice_evaluation_only"
                ),
                "selected_targets": summary.get("selected_targets"),
                "train_loss": train_stats.get("train_loss"),
                "val_loss": train_stats.get("val_loss"),
            }
        )
    rows.sort(key=lambda row: row["epoch"])
    eligible = [row for row in rows if row["b7_selected_validation_dice"] is not None]
    if not eligible:
        raise SystemExit("No completed validation B7 summaries found")
    best = max(
        eligible,
        key=lambda row: (row["b7_selected_validation_dice"], -row["epoch"]),
    )

    result = {"best": best, "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    columns = (
        "epoch",
        "b7_selected_validation_dice",
        "selected_targets",
        "train_loss",
        "val_loss",
    )
    args.output_tsv.write_text(
        "\t".join(columns)
        + "\n"
        + "".join(
            "\t".join("" if row.get(key) is None else str(row[key]) for key in columns)
            + "\n"
            for row in rows
        )
    )
    args.best_epoch_output.write_text(str(best["epoch"]) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
