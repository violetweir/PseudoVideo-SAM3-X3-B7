#!/usr/bin/env python3
"""Round-2 pseudo pool for the next SAM3 fine-tune (Kvasir 1%).

Applies the mainline Tier-A audit rule to ALL train targets:
  - per candidate: q_route = 0.20*q_multi + 0.80*q_model_mean (committee)
  - Top-1 by (q_route, q_model_mean, q_multi, -bridge)
  - accept if q_multi >= 0.90 AND q_return >= 0.95 AND q_model_mean >= 0.90
    AND q_model_min >= 0.80
  - label = pixel consensus 0.75*selected SAM3 + 0.25*student mean, with
    w_final = clip(exp(-5*route_var)*exp(-5*student_var)*cross, 0.05, 1.0)

The accepted pool feeds the round-2 SAM3 full fine-tune.
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
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--predictions-name", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    parser.add_argument("--beta-route", type=float, default=5.0)
    parser.add_argument("--beta-student", type=float, default=5.0)
    args = parser.parse_args()
    if len(args.predictions) != len(args.predictions_name):
        raise RuntimeError("--predictions and --predictions-name must match")

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
    students = [
        (name, {row["merged_id"]: row for row in read_jsonl(path)})
        for name, path in zip(args.predictions_name, args.predictions)
    ]
    target_ids = sorted(candidates)
    print(f"targets: {len(target_ids)}", flush=True)

    map_root = args.output_dir / "pixel_consensus"
    accepted: list[dict] = []
    score_rows: list[dict] = []
    for target_index, target_id in enumerate(target_ids, 1):
        cands = candidates[target_id]
        sam_binary = [load_binary(row["forward_mask_path"]) for row in cands]
        reference_shape = sam_binary[0].shape
        student_binary = [
            resize_binary(
                load_binary(rows[target_id]["student_binary_mask"]), reference_shape
            )
            for _, rows in students
        ]
        student_probability = [
            resize_probability(
                load_probability(rows[target_id]["student_probability_map"]),
                reference_shape,
            )
            for _, rows in students
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
            q_models = [dice(sam_binary[index], pred) for pred in student_binary]
            q_model_mean = float(np.mean(q_models))
            q_model_min = float(np.min(q_models))
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
                    "q_route": q_route,
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
        accepted_flag = (
            selected["q_multi"] >= 0.90
            and selected["q_return"] >= 0.95
            and selected["q_model_mean"] >= 0.90
            and selected["q_model_min"] >= 0.80
        )
        if not accepted_flag:
            continue
        route_stack = np.stack([m.astype(np.float32) for m in sam_binary])
        student_stack = np.stack(student_probability)
        route_var = route_stack.var(axis=0)
        student_mean = student_stack.mean(axis=0)
        student_var = student_stack.var(axis=0)
        selected_probability = sam_binary[selected_index].astype(np.float32)
        w_route = np.exp(-args.beta_route * route_var)
        w_student = np.exp(-args.beta_student * student_var)
        w_cross = np.clip(1.0 - np.abs(selected_probability - student_mean), 0.0, 1.0)
        w_final = np.clip(w_route * w_student * w_cross, 0.05, 1.0)
        pseudo_probability = 0.75 * selected_probability + 0.25 * student_mean
        pseudo_binary = pseudo_probability >= 0.5
        stem = safe_name(target_id)
        pseudo_path = map_root / "pseudo_binary" / f"{stem}.png"
        prob_path = map_root / "pseudo_probability" / f"{stem}.png"
        weight_path = map_root / "w_final" / f"{stem}.png"
        save_binary(pseudo_path, pseudo_binary)
        save_u16(prob_path, pseudo_probability)
        save_u16(weight_path, w_final)
        accepted.append(
            {
                "target_id": target_id,
                "pseudo_mask_path": str(pseudo_path.resolve()),
                "pseudo_consensus_path": str(prob_path.resolve()),
                "pixel_weight_path": str(weight_path.resolve()),
                "q_multi": selected["q_multi"],
                "q_return": selected["q_return"],
                "q_model_mean": selected["q_model_mean"],
                "q_model_min": selected["q_model_min"],
                "q_route": selected["q_route"],
                "route_id": selected["route_id"],
                "feature_mode": selected["feature_mode"],
                "bridge_count": selected["bridge_count"],
                "anchor_id": selected["anchor_id"],
            }
        )
        if target_index % 200 == 0:
            print(f"[{target_index}/{len(target_ids)}] accepted={len(accepted)}", flush=True)

    write_jsonl(args.output_dir / "candidate_scores.jsonl", score_rows)
    write_jsonl(args.output_dir / "round2_pool.jsonl", accepted)
    summary = {
        "targets": len(target_ids),
        "accepted": len(accepted),
        "accept_rate": len(accepted) / len(target_ids),
        "selected_counts": dict(
            Counter(f"{r['feature_mode']}:bridge_{r['bridge_count']}" for r in accepted)
        ),
        "q_multi_mean": float(np.mean([r["q_multi"] for r in accepted])),
        "q_return_mean": float(np.mean([r["q_return"] for r in accepted])),
        "q_model_mean_mean": float(np.mean([r["q_model_mean"] for r in accepted])),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
