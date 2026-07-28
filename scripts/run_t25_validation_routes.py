#!/usr/bin/env python3
"""Generate the missing frozen T21-protocol routes for Validation only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
T21_SCRIPT = PROJECT_ROOT / "scripts/run_t21_dynamic_pseudovideo.py"
spec = importlib.util.spec_from_file_location("t21_frozen", T21_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {T21_SCRIPT}")
t21 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t21
spec.loader.exec_module(t21)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Merged manifest JSONL produced by scripts/prepare_protocol.py.",
    )
    parser.add_argument(
        "--support-manifest",
        type=Path,
        required=True,
        help="Frozen 16-anchor support manifest JSONL.",
    )
    parser.add_argument(
        "--t21-root",
        type=Path,
        required=True,
        help="T21 output root containing protocol/protocol.json and descriptors.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest_path = args.manifest
    support_path = args.support_manifest
    manifest = t21.read_jsonl(manifest_path)
    support = t21.read_jsonl(support_path)
    records, descriptors, neighbors = t21.build_graph(manifest, args.t21_root, 20)
    id_to_index = {row["merged_id"]: i for i, row in enumerate(records)}
    validation = [row for row in records if row["split"] == "validation"]
    anchors = t21.human_pool(support, 512)
    if len(validation) != 161 or len(anchors) != 16:
        raise RuntimeError(
            f"Expected validation=161 and anchors=16, got {len(validation)}, {len(anchors)}"
        )

    phase_root = args.output_root / "validation_pool0"
    provenance = {
        "purpose": "T25 missing Validation routes only",
        "source_t21_protocol": str(args.t21_root / "protocol/protocol.json"),
        "source_t21_protocol_sha256": sha256(
            args.t21_root / "protocol/protocol.json"
        ),
        "manifest_sha256": sha256(manifest_path),
        "support_manifest_sha256": sha256(support_path),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "conda_environment_required": "sam3",
        "teacher": "Frozen SAM3",
        "knn_k": 20,
        "canvas_size": 512,
        "alpha": 0.5,
        "targets": "validation only",
        "target_count": len(validation),
        "routes_per_target": 3,
        "existing_t21_train_test_modified": False,
        "validation_gt_used_for_route_generation_or_selection": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output_root / "validation_protocol.json"
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text())
        if old != provenance:
            raise RuntimeError("Validation protocol changed; use a new output root")
    else:
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    model = t21.build_sam3_video_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        device="cuda",
        compile=False,
    )
    model.eval()
    run_args = argparse.Namespace(
        limit=args.limit,
        knn_k=20,
        canvas_size=512,
        resume=args.resume,
        alpha=0.5,
        tau=0.85,
    )
    t21.run_phase(
        "validation_pool0",
        model,
        args.output_root,
        validation,
        anchors,
        records,
        descriptors,
        neighbors,
        id_to_index,
        run_args,
        generation=None,
    )
    results = t21.read_jsonl(phase_root / "route_results.jsonl")
    expected_targets = min(args.limit, len(validation)) if args.limit else len(validation)
    expected_routes = expected_targets * 3
    if len(results) != expected_routes or any(
        row["status"] != "success" for row in results
    ):
        raise RuntimeError(
            f"Validation route generation did not produce {expected_routes} successes"
        )
    print(
        json.dumps(
            {
                "targets": expected_targets,
                "routes": expected_routes,
                "status": "complete",
            }
        )
    )


if __name__ == "__main__":
    main()
