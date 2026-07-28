#!/usr/bin/env python3
"""Build S27 committee audit scores, pixel consensus maps, and fixed tiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_binary(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def load_probability(path: str | Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    return array.astype(np.float32) / 255.0


def resize_probability(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if array.shape == shape:
        return array
    height, width = shape
    image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
    return np.asarray(
        image.resize((width, height), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def resize_binary(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if array.shape == shape:
        return array
    height, width = shape
    image = Image.fromarray(np.where(array, 255, 0).astype(np.uint8))
    return (
        np.asarray(
            image.resize((width, height), resample=Image.Resampling.NEAREST)
        )
        > 127
    )


def save_u16(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.rint(np.clip(array, 0.0, 1.0) * 65535.0).astype(np.uint16)
    ).save(path)


def save_binary(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(array, 255, 0).astype(np.uint8)).save(path)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = int(a.sum()) + int(b.sum())
    return 1.0 if denominator == 0 else 2.0 * int(np.logical_and(a, b).sum()) / denominator


def safe_name(image_id: str) -> str:
    return image_id.replace("::", "__").replace("/", "_")


def index_student(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    index = {row["merged_id"]: row for row in rows}
    if len(index) != len(rows):
        raise RuntimeError(f"Duplicate student prediction IDs in {path}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remaining", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--sam-probabilities", type=Path, required=True)
    parser.add_argument("--pseudo568", type=Path, required=True)
    parser.add_argument("--auditor", type=Path, action="append", required=True)
    parser.add_argument("--auditor-name", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beta-route", type=float, default=5.0)
    parser.add_argument("--beta-student", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if len(args.auditor) != 3 or len(args.auditor_name) != 3:
        raise RuntimeError("Exactly three auditors and names are required")

    remaining = read_jsonl(args.remaining)
    if args.limit:
        remaining = remaining[: args.limit]
    target_ids = {row["image_id"] for row in remaining}
    remaining_by_id = {row["image_id"]: row for row in remaining}

    routes = [row for row in read_jsonl(args.routes) if row["target_id"] in target_ids]
    route_by_id = {row["route_id"]: row for row in routes}
    routes_by_target: dict[str, list[dict]] = defaultdict(list)
    for row in routes:
        routes_by_target[row["target_id"]].append(row)
    if len(route_by_id) != 3 * len(target_ids):
        raise RuntimeError("Expected exactly three unique routes per target")

    sam_probability_rows = read_jsonl(args.sam_probabilities)
    sam_probability_by_route = {
        row["route_id"]: row["sam3_probability_path"] for row in sam_probability_rows
    }
    missing_probabilities = set(route_by_id) - set(sam_probability_by_route)
    if missing_probabilities:
        raise RuntimeError(
            f"Missing {len(missing_probabilities)} SAM probability maps"
        )

    auditor_indexes = [index_student(path) for path in args.auditor]
    for name, index in zip(args.auditor_name, auditor_indexes):
        missing = target_ids - set(index)
        if missing:
            raise RuntimeError(f"Auditor {name} misses {len(missing)} targets")

    original_areas = np.array(
        [
            float(load_binary(row["existing_pseudo_mask_path"]).mean())
            for row in read_jsonl(args.pseudo568)
        ],
        dtype=np.float64,
    )
    area_q01, area_q99 = (float(x) for x in np.quantile(original_areas, [0.01, 0.99]))

    score_rows: list[dict] = []
    selected_rows: list[dict] = []
    tier_rows: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
    map_root = args.output_dir / "pixel_consensus"
    for target_index, target_id in enumerate(sorted(target_ids), 1):
        target_routes = sorted(
            routes_by_target[target_id], key=lambda row: int(row["bridge_count"])
        )
        if len(target_routes) != 3:
            raise RuntimeError(f"{target_id} has {len(target_routes)} routes")
        sam_binary = [load_binary(row["forward_mask_path"]) for row in target_routes]
        sam_probability = [
            load_probability(sam_probability_by_route[row["route_id"]])
            for row in target_routes
        ]
        shapes = {array.shape for array in sam_probability + sam_binary}
        if len(shapes) != 1:
            raise RuntimeError(f"Inconsistent SAM shapes for {target_id}: {shapes}")

        student_rows = [index[target_id] for index in auditor_indexes]
        raw_student_binary = [
            load_binary(row["student_binary_mask"]) for row in student_rows
        ]
        raw_student_probability = [
            load_probability(row["student_probability_map"]) for row in student_rows
        ]
        reference_shape = sam_binary[0].shape
        student_binary = [
            resize_binary(array, reference_shape) for array in raw_student_binary
        ]
        student_probability = [
            resize_probability(array, reference_shape)
            for array in raw_student_probability
        ]

        per_route = []
        for index, route in enumerate(target_routes):
            q_multi = float(
                sum(
                    dice(sam_binary[index], sam_binary[other])
                    for other in range(3)
                    if other != index
                )
                / 2.0
            )
            q_models = [
                dice(sam_binary[index], prediction) for prediction in student_binary
            ]
            q_model_mean = float(np.mean(q_models))
            q_model_min = float(np.min(q_models))
            q_model_var = float(np.var(q_models))
            q_route = 0.20 * q_multi + 0.80 * q_model_mean
            score = {
                "target_id": target_id,
                "dataset": remaining_by_id[target_id]["dataset"],
                "route_id": route["route_id"],
                "route_type": route["route_type"],
                "anchor_id": route["anchor_id"],
                "bridge_ids": route["bridge_ids"],
                "q_return": float(route["q_return"]),
                "q_multi": q_multi,
                "q_model_by_auditor": {
                    name: value for name, value in zip(args.auditor_name, q_models)
                },
                "q_model_mean": q_model_mean,
                "q_model_min": q_model_min,
                "q_model_var": q_model_var,
                "q_route": q_route,
                "sam3_binary_mask_path": route["forward_mask_path"],
                "sam3_probability_path": sam_probability_by_route[route["route_id"]],
                "sam3_nonempty": bool(sam_binary[index].any()),
                "sam3_area_ratio": float(sam_binary[index].mean()),
                "target_gt_used_for_selection": False,
            }
            score_rows.append(score)
            per_route.append(score)
        ranked = sorted(
            per_route,
            key=lambda row: (
                row["q_route"],
                row["q_model_mean"],
                row["q_multi"],
                -int(route_by_id[row["route_id"]]["bridge_count"]),
            ),
            reverse=True,
        )
        selected = ranked[0]
        selected_index = next(
            index
            for index, route in enumerate(target_routes)
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
            "selected_probability": map_root / "selected_sam3" / f"{stem}.png",
            "route_mean": map_root / "route_mean" / f"{stem}.png",
            "route_var": map_root / "route_var" / f"{stem}.png",
            "student_mean": map_root / "student_mean" / f"{stem}.png",
            "student_var": map_root / "student_var" / f"{stem}.png",
            "pseudo_probability": map_root / "pseudo_probability" / f"{stem}.png",
            "pseudo_binary": map_root / "pseudo_binary" / f"{stem}.png",
            "w_final": map_root / "w_final" / f"{stem}.png",
        }
        for key, array in (
            ("selected_probability", selected_probability),
            ("route_mean", route_mean),
            ("route_var", route_var),
            ("student_mean", student_mean),
            ("student_var", student_var),
            ("pseudo_probability", pseudo_probability),
            ("w_final", w_final),
        ):
            save_u16(paths[key], array)
        save_binary(paths["pseudo_binary"], pseudo_binary)

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
            float(np.clip(
                0.5 * selected["q_multi"] + 0.5 * selected["q_model_mean"],
                0.3,
                1.0,
            ))
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
            "image_path": remaining_by_id[target_id]["image_path"],
            "q_margin": float(ranked[0]["q_route"] - ranked[1]["q_route"]),
            "selected_route": selected["route_type"],
            "selected_mask": selected["sam3_binary_mask_path"],
            "selected_probability": str(paths["selected_probability"].resolve()),
            "p_route_mean": str(paths["route_mean"].resolve()),
            "p_route_var": str(paths["route_var"].resolve()),
            "p_student_mean": str(paths["student_mean"].resolve()),
            "p_student_var": str(paths["student_var"].resolve()),
            "pseudo_consensus_path": str(paths["pseudo_probability"].resolve()),
            "pseudo_mask_path": str(paths["pseudo_binary"].resolve()),
            "pixel_weight_path": str(paths["w_final"].resolve()),
            "student_nonempty_count": student_nonempty_count,
            "selected_area_ratio": selected_area,
            "area_safe_original568_q01_q99": bool(area_safe),
            "tier": tier,
            "explicit_quality_weight": image_weight,
            "target_gt_used_for_tiering": False,
        }
        selected_rows.append(selected_row)
        tier_rows[tier].append(selected_row)
        print(
            f"[{target_index}/{len(target_ids)}] {target_id} "
            f"route={selected['route_type']} tier={tier}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "committee_route_scores.jsonl", score_rows)
    write_jsonl(args.output_dir / "selected_candidates.jsonl", selected_rows)
    tier_dir = args.output_dir / "tiers"
    for tier, rows in tier_rows.items():
        write_jsonl(tier_dir / f"tier{tier}.jsonl", rows)
    summary = {
        "remaining_count": len(target_ids),
        "route_count": len(score_rows),
        "auditors": {
            name: {"manifest": str(path.resolve()), "sha256": sha256(path)}
            for name, path in zip(args.auditor_name, args.auditor)
        },
        "sam_probability_manifest": str(args.sam_probabilities.resolve()),
        "sam_probability_manifest_sha256": sha256(args.sam_probabilities),
        "original568_area_quantiles": {"q01": area_q01, "q99": area_q99},
        "tier_counts": {tier: len(rows) for tier, rows in tier_rows.items()},
        "tier_dataset_counts": {
            tier: dict(Counter(row["dataset"] for row in rows))
            for tier, rows in tier_rows.items()
        },
        "route_selection_counts": dict(
            Counter(row["selected_route"] for row in selected_rows)
        ),
        "score": {"q_multi": 0.20, "q_model_committee_mean": 0.80},
        "pixel_consensus": {
            "beta_route": args.beta_route,
            "beta_student": args.beta_student,
            "pseudo_sam3_weight": 0.75,
            "pseudo_student_weight": 0.25,
            "w_final_clip": [0.05, 1.0],
        },
        "evaluation_gt_used": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
