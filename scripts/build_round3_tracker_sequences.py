#!/usr/bin/env python3
"""Build the frozen, train-only Round3 pseudo-video sequence manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
import yaml

REPO = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def checked_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path

def supervision_entry(frame_index: int, frame_id: str, mask: str, source: str, weight: float) -> dict[str, Any]:
    return {"frame_index": frame_index, "frame_id": frame_id, "mask": mask, "source": source, "weight": float(weight)}

def build(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    merged = read_jsonl(checked_path(protocol["protocol_manifest"]))
    split_by_id = {row["merged_id"]: row["split"] for row in merged}
    train_ids = {key for key, value in split_by_id.items() if value == "train"}
    validation_ids = {key for key, value in split_by_id.items() if value == "validation"}
    test_ids = {key for key, value in split_by_id.items() if value == "test"}
    pseudo_path = checked_path(protocol["pseudo_manifest"])
    if sha256(pseudo_path) != protocol["pseudo_manifest_sha256"]:
        raise RuntimeError("Frozen Round2A pseudo manifest SHA256 mismatch")
    pseudo_rows = read_jsonl(pseudo_path)
    if len(pseudo_rows) != 524:
        raise RuntimeError(f"Expected 524 frozen pseudo labels, got {len(pseudo_rows)}")
    pseudo_by_id = {row["target_id"]: row for row in pseudo_rows}
    if set(pseudo_by_id) - train_ids:
        raise RuntimeError("Pseudo pool contains a non-train target")

    rows: list[dict[str, Any]] = []
    seen_route_ids: set[str] = set()
    route_hashes: dict[str, str] = {}
    mode_counts: Counter[str] = Counter()
    bridge_counts: Counter[int] = Counter()
    supervision_counts: Counter[str] = Counter()
    for mode, root_value in protocol["route_roots"].items():
        route_path = checked_path(root_value) / "train_pool0_stage1" / "routes.jsonl"
        actual_hash = sha256(route_path)
        expected_hash = protocol["train_route_sha256"][mode]
        if actual_hash != expected_hash:
            raise RuntimeError(f"Frozen route SHA256 mismatch for {mode}: {actual_hash} != {expected_hash}")
        route_hashes[mode] = actual_hash
        for route in read_jsonl(route_path):
            if route["target_id"] not in pseudo_by_id:
                continue
            frame_ids = [route["anchor_id"], *route["bridge_ids"], route["target_id"]]
            frames = [route["anchor_image_path"], *route["bridge_image_paths"], route["target_image_path"]]
            if len(frame_ids) != len(frames):
                raise RuntimeError(f"Frame/id mismatch in {route['route_id']}")
            if any(frame_id not in train_ids for frame_id in frame_ids):
                raise RuntimeError(f"Non-train frame in {route['route_id']}")
            if set(frame_ids) & (validation_ids | test_ids):
                raise RuntimeError(f"Validation/test leakage in {route['route_id']}")
            if route["target_split"] != "train":
                raise RuntimeError(f"Non-train route in {route['route_id']}")
            round3_route_id = f"{mode}::{route['route_id']}"
            if round3_route_id in seen_route_ids:
                raise RuntimeError(f"Duplicate mode-qualified route_id {round3_route_id}")
            seen_route_ids.add(round3_route_id)
            supervision = [supervision_entry(0, route["anchor_id"], route["anchor_mask_path"], "gt_anchor", 1.0)]
            supervision_counts["gt_anchor"] += 1
            for frame_index, frame_id in enumerate(frame_ids[1:-1], 1):
                pseudo = pseudo_by_id.get(frame_id)
                if pseudo is not None:
                    supervision.append(supervision_entry(
                        frame_index, frame_id, str(checked_path(pseudo["pseudo_mask_path"])),
                        "frozen_pseudo", float(pseudo.get("explicit_quality_weight", pseudo["b7"]))))
                    supervision_counts["frozen_pseudo_bridge"] += 1
            target_pseudo = pseudo_by_id[route["target_id"]]
            supervision.append(supervision_entry(
                len(frames) - 1, route["target_id"], str(checked_path(target_pseudo["pseudo_mask_path"])),
                "frozen_pseudo", float(target_pseudo.get("explicit_quality_weight", target_pseudo["b7"]))))
            supervision_counts["frozen_pseudo_target"] += 1
            rows.append({
                "route_id": round3_route_id, "source_route_id": route["route_id"],
                "mode": mode, "bridge": int(route["bridge_count"]),
                "anchor_id": route["anchor_id"], "anchor_box_xywh_normalized": route["anchor_box_xywh_normalized"],
                "frame_ids": frame_ids, "frames": frames, "supervision": supervision,
                "target_id": route["target_id"], "target_pseudo_manifest_sha256": protocol["pseudo_manifest_sha256"],
                "route_manifest_sha256": actual_hash,
            })
            mode_counts[mode] += 1
            bridge_counts[int(route["bridge_count"])] += 1

    rows.sort(key=lambda item: (item["mode"], item["bridge"], item["target_id"]))
    output = output_override if output_override is not None else checked_path(protocol["sequence_manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    train_sequence_ids = {frame_id for row in rows for frame_id in row["frame_ids"]}
    assert not train_sequence_ids & validation_ids
    assert not train_sequence_ids & test_ids
    assert train_sequence_ids <= train_ids
    summary = {
        "sequence_count": len(rows), "unique_frame_count": len(train_sequence_ids),
        "unique_target_count": len({row["target_id"] for row in rows}),
        "mode_counts": dict(sorted(mode_counts.items())),
        "bridge_counts": {str(key): bridge_counts[key] for key in sorted(bridge_counts)},
        "supervision_counts": dict(sorted(supervision_counts.items())),
        "pseudo_manifest": str(pseudo_path), "pseudo_manifest_sha256": sha256(pseudo_path),
        "route_sha256": route_hashes, "sequence_manifest": str(output),
        "sequence_manifest_sha256": sha256(output),
        "split_counts": {"train": len(train_ids), "validation": len(validation_ids), "test": len(test_ids)},
        "leakage": {"validation_intersection": 0, "test_intersection": 0},
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    protocol_dir = checked_path(config["experiment"]["root"]) / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    locked = {
        "seed": config["experiment"]["seed"],
        "checkpoint": str(checked_path(protocol["initialization_checkpoint"])),
        "checkpoint_sha256": protocol["initialization_sha256"],
        "pseudo_manifest": str(pseudo_path), "pseudo_manifest_sha256": summary["pseudo_manifest_sha256"],
        "routes_sha256_before": route_hashes, "routes_sha256_after": route_hashes,
        "checkpoint_selection": config["training"]["checkpoint_selection"],
        "test_policy": "Run once per validation-frozen T1/T2 best checkpoint.",
        "long_chain_loss_enabled": bool(config["long_chain_loss"]["enabled"]),
    }
    (protocol_dir / "locked_protocol.json").write_text(
        json.dumps(locked, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO / "configs/c0_256_round3_tracker_stage4.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.config.resolve(), args.output), indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
