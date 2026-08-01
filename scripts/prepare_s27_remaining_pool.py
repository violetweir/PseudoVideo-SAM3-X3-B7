#!/usr/bin/env python3
"""Freeze the disjoint S27 human/original-pseudo/remaining training pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--human-list", type=Path, required=True)
    parser.add_argument("--pseudo568", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    train = read_jsonl(args.train_metadata)
    train_root = args.train_metadata.parent
    normalized_train = []
    for row in train:
        row = dict(row)
        image = Path(row["file_name"])
        mask = Path(row["mask_file_name"])
        if not image.is_absolute():
            image = train_root / image
        if not mask.is_absolute():
            mask = train_root / mask
        row["file_name"] = str(image.resolve())
        row["mask_file_name"] = str(mask.resolve())
        normalized_train.append(row)
    train = normalized_train
    train_by_id = {row["merged_id"]: row for row in train}
    train_by_path = {str(Path(row["file_name"]).resolve()): row for row in train}
    if len(train_by_id) != len(train) or len(train_by_path) != len(train):
        raise RuntimeError("Duplicate ID or image path in Train metadata")

    human_paths = {
        str(Path(line.strip()).resolve())
        for line in args.human_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not human_paths:
        raise RuntimeError("No human paths found")
    unknown_human = human_paths - set(train_by_path)
    if unknown_human:
        raise RuntimeError(f"Human paths absent from Train: {sorted(unknown_human)}")
    human_ids = {train_by_path[path]["merged_id"] for path in human_paths}

    pseudo_source = read_jsonl(args.pseudo568)
    pseudo_ids = {row["target_id"] for row in pseudo_source}
    if not pseudo_source or len(pseudo_ids) != len(pseudo_source):
        raise RuntimeError(
            f"Expected unique original pseudo records, got "
            f"{len(pseudo_source)}/{len(pseudo_ids)}"
        )
    unknown_pseudo = pseudo_ids - set(train_by_id)
    if unknown_pseudo:
        raise RuntimeError(f"Pseudo IDs absent from Train: {sorted(unknown_pseudo)}")
    if human_ids & pseudo_ids:
        raise RuntimeError(f"Human/pseudo overlap: {sorted(human_ids & pseudo_ids)}")

    remaining_ids = set(train_by_id) - human_ids - pseudo_ids
    if len(human_ids) + len(pseudo_ids) + len(remaining_ids) != len(train):
        raise RuntimeError("Pool cardinalities do not sum to Train")

    def base(row: dict, status: str, has_gt: bool) -> dict:
        return {
            "image_id": row["merged_id"],
            "image_path": row["file_name"],
            "mask_path_evaluation_only": row["mask_file_name"],
            "dataset": row["source_dataset"],
            "split": "train",
            "has_gt": has_gt,
            "pseudo_status": status,
            "existing_pseudo_mask_path": None,
        }

    human_rows = [
        base(train_by_id[image_id], "human_gt", True)
        for image_id in sorted(human_ids)
    ]
    pseudo_by_id = {row["target_id"]: row for row in pseudo_source}
    pseudo_rows = []
    for image_id in sorted(pseudo_ids):
        row = base(train_by_id[image_id], "original_pseudo", False)
        source = pseudo_by_id[image_id]
        row.update(
            {
                "existing_pseudo_mask_path": source["pseudo_mask_path"],
                "pseudo_mask_sha256": source.get("pseudo_mask_sha256"),
                "q_multi": source["q_multi"],
                "q_return": source["q_return"],
                "route_id": source["route_id"],
                "route_type": source["route_type"],
            }
        )
        pseudo_rows.append(row)
    remaining_rows = [
        base(train_by_id[image_id], "remaining_unlabeled", False)
        for image_id in sorted(remaining_ids)
    ]
    all_rows = sorted(human_rows + pseudo_rows + remaining_rows, key=lambda x: x["image_id"])

    output_files = {
        "human16": args.output_dir / "human16.jsonl",
        "pseudo568_original": args.output_dir / "pseudo568_original.jsonl",
        "unlabeled_remaining": args.output_dir / "unlabeled_remaining.jsonl",
        "train_all": args.output_dir / "train_all.jsonl",
    }
    for key, rows in (
        ("human16", human_rows),
        ("pseudo568_original", pseudo_rows),
        ("unlabeled_remaining", remaining_rows),
        ("train_all", all_rows),
    ):
        write_jsonl(output_files[key], rows)

    sets = {
        "human16": human_ids,
        "pseudo568_original": pseudo_ids,
        "unlabeled_remaining": remaining_ids,
    }
    overlaps = {
        "human_pseudo": sorted(human_ids & pseudo_ids),
        "human_remaining": sorted(human_ids & remaining_ids),
        "pseudo_remaining": sorted(pseudo_ids & remaining_ids),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "git_commit": git_commit(args.project_root),
        "source_files": {
            "train_metadata": str(args.train_metadata.resolve()),
            "human_list": str(args.human_list.resolve()),
            "pseudo568": str(args.pseudo568.resolve()),
        },
        "counts": {key: len(value) for key, value in sets.items()}
        | {"train_all": len(all_rows)},
        "overlaps": overlaps,
        "union_equals_train": set().union(*sets.values()) == set(train_by_id),
        "files": {
            key: {
                "path": str(path.resolve()),
                "rows": sum(1 for line in path.open(encoding="utf-8") if line.strip()),
                "sha256": sha256(path),
            }
            for key, path in output_files.items()
        },
    }
    if any(overlaps.values()) or not summary["union_equals_train"]:
        raise RuntimeError("Disjoint-union verification failed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hashes.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
