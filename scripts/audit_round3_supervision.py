#!/usr/bin/env python3
"""Audit Round3 frame supervision and object-presence semantics."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

REPO = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = read_jsonl(resolve(config["protocol"]["sequence_manifest"]))
    mask_positive_cache: dict[str, bool] = {}
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    duplicate_supervision_frames = []
    invalid_indices = []
    for row in rows:
        seen: set[int] = set()
        for supervision in row["supervision"]:
            index = int(supervision["frame_index"])
            if index in seen:
                duplicate_supervision_frames.append({"route_id": row["route_id"], "frame_index": index})
            seen.add(index)
            if not 0 <= index < len(row["frames"]):
                invalid_indices.append({"route_id": row["route_id"], "frame_index": index})
                continue
            mask_path = str(resolve(supervision["mask"]))
            if mask_path not in mask_positive_cache:
                array = np.asarray(Image.open(mask_path).convert("L"))
                mask_positive_cache[mask_path] = bool(np.any(array > 0))
            source_counts[supervision["source"]] += 1
            if mask_positive_cache[mask_path]:
                counts["num_supervised_positive_frames"] += 1
                counts["num_object_positive"] += 1
            else:
                counts["num_supervised_empty_frames"] += 1
                counts["num_object_negative"] += 1
        unsupervised = len(row["frames"]) - len(seen)
        counts["num_unsupervised_frames"] += unsupervised
        counts["num_object_ignore"] += unsupervised
    required = [
        "num_supervised_positive_frames", "num_supervised_empty_frames",
        "num_unsupervised_frames", "num_object_positive",
        "num_object_negative", "num_object_ignore",
    ]
    for key in required:
        counts[key] += 0
    bug_found = bool(duplicate_supervision_frames or invalid_indices)
    result = {
        **{key: counts[key] for key in required},
        "sequence_count": len(rows),
        "frame_occurrence_count": sum(len(row["frames"]) for row in rows),
        "supervision_source_counts": dict(sorted(source_counts.items())),
        "unique_mask_count": len(mask_positive_cache),
        "segmentation_loss_scope": "Only entries in row['supervision']; absent bridge entries are IGNORE.",
        "object_presence_loss_scope": "Only entries in row['supervision']; absent bridge entries are IGNORE.",
        "teacher_force_obj_scores_for_mem": False,
        "unsupervised_bridge_policy": {"segmentation_loss": "IGNORE", "object_presence": "IGNORE"},
        "duplicate_supervision_frames": duplicate_supervision_frames,
        "invalid_supervision_indices": invalid_indices,
        "bug_found": bug_found,
    }
    if counts["num_object_ignore"] != counts["num_unsupervised_frames"]:
        raise RuntimeError("Object-ignore and unsupervised counts differ")
    if counts["num_object_positive"] + counts["num_object_negative"] != counts["num_supervised_positive_frames"] + counts["num_supervised_empty_frames"]:
        raise RuntimeError("Object and segmentation supervision accounting differs")
    output = resolve(config["experiment"]["root"]) / "protocol" / "supervision_audit.json"
    save_json(output, result)
    if bug_found:
        raise RuntimeError(f"Round3 supervision audit failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.config.resolve()), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

