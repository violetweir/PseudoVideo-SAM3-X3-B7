#!/usr/bin/env python3
"""Filter a route JSONL pool by bridge-count range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    selected = [
        row
        for row in rows
        if args.min_bridge <= int(row["bridge_count"]) <= args.max_bridge
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "input_rows": len(rows),
                "selected_rows": len(selected),
                "min_bridge": args.min_bridge,
                "max_bridge": args.max_bridge,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
