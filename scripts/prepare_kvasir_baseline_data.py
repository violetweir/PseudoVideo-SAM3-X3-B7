#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
PROTOCOL = ROOT / "work/kvasir_1pct_anchors/protocol"
OUT = ROOT / "work/kvasir_1pct_anchors/baseline_data"


def main() -> None:
    rows = [
        json.loads(line)
        for line in (PROTOCOL / "merged_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        split_dir = OUT / split
        split_dir.mkdir(parents=True, exist_ok=True)
        payload = []
        for row in sorted(split_rows, key=lambda item: item["sample_id"]):
            payload.append(
                {
                    "file_name": row["image_path"],
                    "mask_file_name": row["mask_path"],
                    "merged_id": row["merged_id"],
                    "source_dataset": row["source_dataset"],
                }
            )
        (split_dir / "metadata.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in payload),
            encoding="utf-8",
        )
        counts[split] = len(payload)
    summary = {
        "source_manifest": str(PROTOCOL / "merged_manifest.jsonl"),
        "output": str(OUT),
        "counts": counts,
        "note": "Metadata-only adapter for SC-SAM/SynFoC; image and mask paths remain absolute.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
