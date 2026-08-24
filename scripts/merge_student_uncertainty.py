#!/usr/bin/env python3
"""Merge computed student uncertainty fields into a RouteCo manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--uncertainty", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    metrics = {row["target_id"]: row for row in read_jsonl(args.uncertainty)}
    missing = [row["target_id"] for row in rows if row["target_id"] not in metrics]
    if missing:
        raise RuntimeError(f"Missing uncertainty for {len(missing)} targets, e.g. {missing[:5]}")

    for row in rows:
        m = metrics[row["target_id"]]
        row["student_aug_variance"] = m["student_aug_variance"]
        row["student_weak_strong_variance"] = m["student_weak_strong_variance"]
        row["student_confidence"] = m["student_confidence"]
        row["student_nonempty_count"] = m["student_nonempty_count"]
        row["student_uncertainty_image_size"] = m["image_size"]
        row["student_uncertainty_weak_samples"] = m["weak_samples"]
        row["student_uncertainty_strong_samples"] = m["strong_samples"]

    write_jsonl(args.output, rows)
    print(json.dumps({"n": len(rows), "missing": len(missing)}, indent=2))


if __name__ == "__main__":
    main()
