#!/usr/bin/env python3
"""Build the Phase-1 Kvasir 1% original pseudo pool for the S27 trainer.

Stage 0: take the RouteCo warm-start manifest, apply the HQ filter used by
the ft_1pct propagation-quality pipeline, and attach student-audit signals
(SynFoC MedSAM student vs route mask agreement) as per-image quality weights.

Output rows are compatible with run_s27_student.py (sample_type=original).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def binary_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def read_audit(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("target_id"):
            continue
        parts = line.split("\t")
        rows[parts[0]] = {
            "sam_route": float(parts[1]),
            "unet_route": float(parts[2]),
            "sam_gt": float(parts[3]),
            "unet_gt": float(parts[4]),
            "aug_stab": float(parts[5]),
            "pseudo_gt": float(parts[6]),
        }
    return rows


def select_hq(row: dict) -> bool:
    uncertainty = row.get("routeco_uncertainty") or {}
    mask = binary_mask(row["pseudo_mask_path"])
    area = float(mask.mean())
    return (
        float(row["q_cycle"]) >= 0.95
        and float(uncertainty.get("route_mask_mean_variance", 1.0)) <= 0.01
        and float(uncertainty.get("route_selected_mean_disagreement", 1.0)) <= 0.05
        and 0.001 <= area <= 0.6
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "work/kvasir_1pct_anchors/routeco_sam3_v1/routeco_v1_pseudo_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("work/kvasir_1pct_anchors/phase1/synfoc_audit.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "work/kvasir_1pct_anchors/phase1/pseudo_manifest_original.jsonl"
        ),
    )
    parser.add_argument("--min-agreement", type=float, default=0.0)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    audit = read_audit(args.audit)
    hq = [row for row in manifest if select_hq(row)]
    missing = sorted(t for t in (r["target_id"] for r in hq) if t not in audit)
    if missing:
        raise RuntimeError(f"Audit missing {len(missing)} HQ targets: {missing[:5]}")

    output_rows = []
    for row in hq:
        target_id = row["target_id"]
        a = audit[target_id]
        agreement = min(a["sam_route"], a["unet_route"])
        if agreement < args.min_agreement:
            continue
        weight = float(np.clip(0.25 + 0.75 * agreement, 0.05, 1.0))
        uncertainty = row.get("routeco_uncertainty") or {}
        out_row = {
            "target_id": target_id,
            "pseudo_mask_path": str(Path(row["pseudo_mask_path"]).resolve()),
            "sample_type": "original",
            "explicit_quality_weight": weight,
            "q_multi": float(
                np.clip(
                    1.0
                    - float(
                        uncertainty.get("route_selected_mean_disagreement", 0.0)
                    ),
                    0.0,
                    1.0,
                )
            ),
            "q_model_mean": float(np.mean([a["sam_route"], a["unet_route"]])),
            "q_model_var": float(abs(a["sam_route"] - a["unet_route"])),
            "student_sam_route_dice": float(a["sam_route"]),
            "student_unet_route_dice": float(a["unet_route"]),
            "student_aug_stability": float(a["aug_stab"]),
            "q_cycle": float(row["q_cycle"]),
            "route_id": row["route_id"],
            "feature_mode": row["feature_mode"],
            "bridge_count": int(row["bridge_count"]),
            "anchor_id": row["anchor_id"],
            "routeco_selected_score": float(row.get("routeco_selected_score", 0.0)),
        }
        output_rows.append(out_row)

    write_jsonl(args.output, output_rows)
    weights = np.array([r["explicit_quality_weight"] for r in output_rows])
    summary = {
        "source_manifest": str(args.manifest),
        "hq_candidates": len(hq),
        "output_rows": len(output_rows),
        "min_agreement": args.min_agreement,
        "sample_types": dict(Counter(r["sample_type"] for r in output_rows)),
        "weight_mean": float(weights.mean()),
        "weight_min": float(weights.min()),
        "weight_q10": float(np.percentile(weights, 10)),
        "weight_median": float(np.median(weights)),
    }
    (args.output.with_name(args.output.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
