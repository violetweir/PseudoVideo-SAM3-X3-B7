#!/usr/bin/env python3
"""Build leakage-safe three-route binary consensus targets for T24-S3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROUTE_TYPES = ("direct", "one_bridge", "two_bridges")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-manifest", type=Path, required=True)
    parser.add_argument("--route-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--pixel-min-weight", type=float, default=0.1)
    args = parser.parse_args()

    pseudo_rows = read_jsonl(args.pseudo_manifest)
    routes_by_target: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.route_results):
        routes_by_target[row["target_id"]].append(row)

    probability_dir = args.output_root / "consensus_probability"
    weight_dir = args.output_root / "pixel_weight"
    probability_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for pseudo in sorted(pseudo_rows, key=lambda row: row["target_id"]):
        routes = routes_by_target[pseudo["target_id"]]
        route_by_type = {row["route_type"]: row for row in routes}
        if set(route_by_type) != set(ROUTE_TYPES):
            raise RuntimeError(
                f"{pseudo['target_id']}: expected {ROUTE_TYPES}, "
                f"got {sorted(route_by_type)}"
            )
        masks = []
        route_paths = {}
        for route_type in ROUTE_TYPES:
            path = Path(route_by_type[route_type]["forward_mask_path"])
            if not path.exists():
                raise FileNotFoundError(path)
            mask = (
                np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
            ).astype(np.float32)
            masks.append(mask)
            route_paths[route_type] = str(path.resolve())
        if len({mask.shape for mask in masks}) != 1:
            raise RuntimeError(f"{pseudo['target_id']}: route mask shape mismatch")
        stack = np.stack(masks)
        probability = stack.mean(axis=0)
        variance = stack.var(axis=0)
        pixel_weight = np.maximum(
            args.pixel_min_weight, np.exp(-args.beta * variance)
        )
        safe_id = pseudo["target_id"].replace("::", "__").replace("/", "_")
        probability_path = probability_dir / f"{safe_id}.png"
        weight_path = weight_dir / f"{safe_id}.png"
        Image.fromarray(np.rint(probability * 255).astype(np.uint8)).save(
            probability_path
        )
        Image.fromarray(np.rint(pixel_weight * 255).astype(np.uint8)).save(
            weight_path
        )
        output_rows.append(
            {
                **pseudo,
                "pseudo_consensus_path": str(probability_path.resolve()),
                "pixel_weight_path": str(weight_path.resolve()),
                "route_mask_paths": route_paths,
                "q_combined": 0.5
                * (float(pseudo["q_multi"]) + float(pseudo["q_return"])),
                "q_product": float(pseudo["q_multi"])
                * float(pseudo["q_return"]),
                "consensus_source": "mean of three frozen T21 binary route masks",
                "pixel_weight_formula": (
                    f"max({args.pixel_min_weight}, exp(-{args.beta} * P_var))"
                ),
            }
        )

    if len(output_rows) != 568:
        raise RuntimeError(f"Expected 568 rows, got {len(output_rows)}")
    manifest = args.output_root / "pseudo_consensus.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    protocol = {
        "count": len(output_rows),
        "route_types": list(ROUTE_TYPES),
        "source_pseudo_manifest": str(args.pseudo_manifest.resolve()),
        "source_pseudo_manifest_sha256": sha256(args.pseudo_manifest),
        "source_route_results": str(args.route_results.resolve()),
        "source_route_results_sha256": sha256(args.route_results),
        "consensus": "mean of three binary masks; stored as 8-bit probability",
        "variance": "population variance of three binary masks",
        "beta": args.beta,
        "pixel_min_weight": args.pixel_min_weight,
        "validation_or_test_used": False,
    }
    (args.output_root / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "PREPARATION_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
