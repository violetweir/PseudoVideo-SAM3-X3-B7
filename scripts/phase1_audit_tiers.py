#!/usr/bin/env python3
"""Phase-1 Stage 2: mainline-style committee audit and Tier A/B/C (Kvasir 1%).

Ports build_s27_audit_and_tiers.py to the b3-b6 candidate pool:
  - per-candidate q_route = 0.20*q_multi + 0.80*q_model_mean (student dominant)
  - Top-1 candidate per target by (q_route, q_model_mean, q_multi, -bridge)
  - pixel consensus: 0.75*selected SAM3 + 0.25*student mean,
    w_final = clip(exp(-5*route_var)*exp(-5*student_var)*cross, 0.05, 1.0)
  - Tier A/B hard thresholds identical to the mainline

SAM3 probability maps are proxied by the mean of the candidate binary masks
(the same construction used by the mainline's own S3 consensus targets).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


MODES = (
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_binary(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def load_probability(path: str | Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    if np.issubdtype(array.dtype, np.floating):
        return np.clip(array.astype(np.float32), 0.0, 1.0)
    return array.astype(np.float32) / 255.0


def resize_probability(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if array.shape == shape:
        return array
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    return np.asarray(
        image.resize((shape[1], shape[0]), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def resize_binary(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if array.shape == shape:
        return array
    image = Image.fromarray(np.where(array, 255, 0).astype(np.uint8))
    return np.asarray(
        image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    ) > 127


def save_u16(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(array, 0.0, 1.0) * 65535.0).astype(np.uint16)).save(
        path
    )


def save_binary(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(array, 255, 0).astype(np.uint8)).save(path)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


def safe_name(target_id: str) -> str:
    return target_id.replace("::", "__").replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--labeled-list", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--predictions-name", action="append", required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    parser.add_argument("--beta-route", type=float, default=5.0)
    parser.add_argument("--beta-student", type=float, default=5.0)
    args = parser.parse_args()
    if len(args.predictions) != len(args.predictions_name):
        raise RuntimeError("--predictions and --predictions-name must match")

    labeled = {
        str(Path(line.strip()).resolve())
        for line in args.labeled_list.read_text().splitlines()
        if line.strip()
    }
    original_ids = {row["target_id"] for row in read_jsonl(args.original_manifest)}
    metadata = read_jsonl(args.train_metadata)
    remaining = [
        row
        for row in metadata
        if row["merged_id"] not in original_ids
        and str(Path(row["file_name"]).resolve()) not in labeled
    ]
    remaining_ids = {row["merged_id"] for row in remaining}
    print(
        f"train={len(metadata)} labeled={len(labeled)} original={len(original_ids)} "
        f"remaining={len(remaining)}",
        flush=True,
    )

    candidates: dict[str, list[dict]] = defaultdict(list)
    for mode in MODES:
        rows = read_jsonl(
            args.quality_root
            / mode
            / "propagation_quality_train/propagation_quality.jsonl"
        )
        for row in rows:
            if args.min_bridge <= int(row["bridge_count"]) <= args.max_bridge:
                row["feature_mode"] = mode
                candidates[row["target_id"]].append(row)

    students = [(name, read_jsonl(path)) for name, path in zip(args.predictions_name, args.predictions)]
    student_predictions = {
        name: {row["merged_id"]: row for row in rows}
        for name, rows in students
    }

    original_areas = np.array(
        [
            float(load_binary(row["pseudo_mask_path"]).mean())
            for row in read_jsonl(args.original_manifest)
        ]
    )
    area_q01, area_q99 = (float(x) for x in np.quantile(original_areas, [0.01, 0.99]))

    score_rows: list[dict] = []
    selected_rows: list[dict] = []
    tier_rows: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
    map_root = args.output_dir / "pixel_consensus"
    for target_index, target_id in enumerate(sorted(remaining_ids), 1):
        cands = candidates.get(target_id, [])
        if not cands:
            raise RuntimeError(f"No candidates for {target_id}")
        sam_binary = [load_binary(row["forward_mask_path"]) for row in cands]
        sam_probability = [mask.astype(np.float32) for mask in sam_binary]
        reference_shape = sam_binary[0].shape
        student_binary = [
            resize_binary(
                load_binary(student_predictions[name][target_id]["student_binary_mask"]),
                reference_shape,
            )
            for name in args.predictions_name
        ]
        student_probability = [
            resize_probability(
                load_probability(
                    student_predictions[name][target_id]["student_probability_map"]
                ),
                reference_shape,
            )
            for name in args.predictions_name
        ]

        per_route = []
        for index, route in enumerate(cands):
            q_multi = float(
                np.mean(
                    [
                        dice(sam_binary[index], other)
                        for j, other in enumerate(sam_binary)
                        if j != index
                    ]
                )
            )
            q_models = [dice(sam_binary[index], prediction) for prediction in student_binary]
            q_model_mean = float(np.mean(q_models))
            q_model_min = float(np.min(q_models))
            q_model_var = float(np.var(q_models))
            q_route = 0.20 * q_multi + 0.80 * q_model_mean
            score_rows.append(
                {
                    "target_id": target_id,
                    "route_id": route["route_id"],
                    "feature_mode": route["feature_mode"],
                    "bridge_count": int(route["bridge_count"]),
                    "anchor_id": route["anchor_id"],
                    "q_return": float(route.get("q_cycle", 0.0)),
                    "q_multi": q_multi,
                    "q_model_mean": q_model_mean,
                    "q_model_min": q_model_min,
                    "q_model_var": q_model_var,
                    "q_route": q_route,
                    "sam3_binary_mask_path": route["forward_mask_path"],
                    "sam3_nonempty": bool(sam_binary[index].any()),
                    "sam3_area_ratio": float(sam_binary[index].mean()),
                    "target_gt_used_for_selection": False,
                }
            )
            per_route.append(score_rows[-1])

        ranked = sorted(
            per_route,
            key=lambda row: (
                row["q_route"],
                row["q_model_mean"],
                row["q_multi"],
                -row["bridge_count"],
            ),
            reverse=True,
        )
        selected = ranked[0]
        selected_index = next(
            index
            for index, route in enumerate(cands)
            if route["route_id"] == selected["route_id"]
        )

        route_stack = np.stack(sam_probability)
        student_stack = np.stack(student_probability)
        route_mean = route_stack.mean(axis=0)
        route_var = route_stack.var(axis=0)
        student_mean = student_stack.mean(axis=0)
        student_var = student_stack.var(axis=0)
        selected_probability = sam_probability[selected_index]
        w_route = np.exp(-args.beta_route * route_var)
        w_student = np.exp(-args.beta_student * student_var)
        w_cross = np.clip(1.0 - np.abs(selected_probability - student_mean), 0.0, 1.0)
        w_final = np.clip(w_route * w_student * w_cross, 0.05, 1.0)
        pseudo_probability = 0.75 * selected_probability + 0.25 * student_mean
        pseudo_binary = pseudo_probability >= 0.5

        stem = safe_name(target_id)
        paths = {
            "pseudo_probability": map_root / "pseudo_probability" / f"{stem}.png",
            "pseudo_binary": map_root / "pseudo_binary" / f"{stem}.png",
            "w_final": map_root / "w_final" / f"{stem}.png",
        }
        save_u16(paths["pseudo_probability"], pseudo_probability)
        save_binary(paths["pseudo_binary"], pseudo_binary)
        save_u16(paths["w_final"], w_final)

        selected_area = float(sam_binary[selected_index].mean())
        student_nonempty_count = sum(bool(array.any()) for array in student_binary)
        area_safe = area_q01 <= selected_area <= area_q99
        tier_a = (
            selected["q_multi"] >= 0.90
            and selected["q_model_mean"] >= 0.90
            and selected["q_model_min"] >= 0.80
            and selected["q_model_var"] <= 0.01
            and selected["q_route"] >= 0.88
            and selected["sam3_nonempty"]
            and student_nonempty_count >= 2
            and area_safe
        )
        tier_b = (
            not tier_a
            and selected["q_multi"] >= 0.75
            and selected["q_model_mean"] >= 0.75
            and selected["q_model_min"] >= 0.60
            and selected["q_model_var"] <= 0.03
        )
        tier = "A" if tier_a else ("B" if tier_b else "C")
        image_weight = (
            float(
                np.clip(
                    0.5 * selected["q_multi"] + 0.5 * selected["q_model_mean"],
                    0.3,
                    1.0,
                )
            )
            if tier == "A"
            else float(
                np.clip(
                    0.25 * selected["q_multi"] + 0.75 * selected["q_model_mean"],
                    0.0,
                    1.0,
                )
            )
        )
        selected_row = {
            **selected,
            "q_margin": float(ranked[0]["q_route"] - ranked[1]["q_route"]),
            "selected_route": f"{selected['feature_mode']}:bridge_{selected['bridge_count']}",
            "selected_mask": selected["sam3_binary_mask_path"],
            "pseudo_consensus_path": str(paths["pseudo_probability"].resolve()),
            "pseudo_mask_path": str(paths["pseudo_binary"].resolve()),
            "pixel_weight_path": str(paths["w_final"].resolve()),
            "student_nonempty_count": student_nonempty_count,
            "selected_area_ratio": selected_area,
            "area_safe_original_q01_q99": bool(area_safe),
            "tier": tier,
            "explicit_quality_weight": image_weight,
            "target_gt_used_for_tiering": False,
        }
        selected_rows.append(selected_row)
        tier_rows[tier].append(selected_row)
        if target_index % 100 == 0:
            print(f"[{target_index}/{len(remaining_ids)}] {target_id} tier={tier}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "committee_route_scores.jsonl", score_rows)
    write_jsonl(args.output_dir / "selected_candidates.jsonl", selected_rows)
    for tier, rows in tier_rows.items():
        write_jsonl(args.output_dir / f"tier_{tier}.jsonl", rows)
    summary = {
        "remaining_count": len(remaining_ids),
        "route_count": len(score_rows),
        "tier_counts": {tier: len(rows) for tier, rows in tier_rows.items()},
        "original_area_quantiles": {"q01": area_q01, "q99": area_q99},
        "score": {"q_multi": 0.20, "q_model_committee_mean": 0.80},
        "pixel_consensus": {
            "beta_route": args.beta_route,
            "beta_student": args.beta_student,
            "pseudo_sam3_weight": 0.75,
            "pseudo_student_weight": 0.25,
            "w_final_clip": [0.05, 1.0],
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
