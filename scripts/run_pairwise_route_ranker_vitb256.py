#!/usr/bin/env python3
"""Target-grouped nested-CV audit for a pairwise ViT-B@256 route ranker.

This script is deliberately validation-only.  It uses validation GT to form
within-target pairwise preferences, produces out-of-fold selections, and never
loads or evaluates the test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.special import expit


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
FAMILIES = (
    "t18_corrected",
    "dino_global_pooling",
    "dino_patch_average",
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_target_pooling__knn_cls",
    "anchor_conditioned_target_pooling__knn_pooled",
    "anchor_conditioned_target_pooling__knn_cond",
    "anchor_conditioned_patch_correspondence",
    "anchor_conditioned_patch_correspondence__knn_cls",
    "anchor_conditioned_patch_correspondence__knn_pooled",
    "anchor_conditioned_patch_correspondence__knn_cond",
)
FIXED_FAMILY = "anchor_conditioned_target_pooling__knn_cls"
TEACHERS = ("X3_final", "S2_final", "S3_final")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_mask(path: str | Path, size: int, root: Path) -> np.ndarray:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    image = Image.open(resolved).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) <= 1:
        return np.zeros_like(ranks)
    return ranks / (len(values) - 1)


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.5, 0.5
    h, w = mask.shape
    return float(xs.mean() / max(w - 1, 1)), float(ys.mean() / max(h - 1, 1))


@dataclass
class Candidate:
    target_id: str
    family: str
    route_id: str
    bridge: int
    dice: float
    features: np.ndarray
    q_return: float
    q_multi: float
    q_model: float
    sam_score: float


@dataclass
class TargetGroup:
    target_id: str
    candidates: list[Candidate]
    fixed_index: int

    @property
    def fixed(self) -> Candidate:
        return self.candidates[self.fixed_index]

    @property
    def oracle(self) -> Candidate:
        return max(self.candidates, key=lambda candidate: candidate.dice)


def load_student_masks(
    student_root: Path,
    size: int,
    root: Path,
    split: str = "validation",
) -> dict[tuple[str, str], np.ndarray]:
    masks: dict[tuple[str, str], np.ndarray] = {}
    for teacher in TEACHERS:
        manifest = student_root / teacher / f"student_predictions_{split}.jsonl"
        for row in read_jsonl(manifest):
            masks[(teacher, row["merged_id"])] = load_mask(row["student_binary_mask"], size, root)
    return masks


def load_groups(
    quality_root: Path,
    student_root: Path,
    tag: str,
    min_bridge: int,
    max_bridge: int,
    size: int,
    root: Path,
    split: str = "validation",
    include_student: bool = True,
    families: tuple[str, ...] | None = None,
    fixed_family: str = FIXED_FAMILY,
) -> tuple[dict[str, TargetGroup], list[str], list[int]]:
    selected_families = FAMILIES if families is None else families
    if fixed_family not in selected_families:
        raise ValueError(f"Fixed family {fixed_family} is not in selected families")
    rows_by_target: dict[str, list[dict]] = {}
    for family in selected_families:
        manifest = quality_root / family / f"propagation_quality_{split}_{tag}" / "propagation_quality.jsonl"
        for row in read_jsonl(manifest):
            bridge = int(row["bridge_count"])
            if min_bridge <= bridge <= max_bridge:
                row["family"] = family
                rows_by_target.setdefault(row["target_id"], []).append(row)

    student_masks = load_student_masks(student_root, size, root, split) if include_student else {}
    feature_names = [
        "bridge_scaled",
        "q_return",
        "q_multi",
        "q_return_rank",
        "q_multi_rank",
        "sam_score",
        "sam_score_rank",
        "cycle_sam_score",
        "path_bottleneck_similarity",
        "path_mean_similarity",
        "mask_area",
        "mask_centroid_x",
        "mask_centroid_y",
        "fixed_mask_dice",
        "log_area_ratio_to_fixed",
        "centroid_distance_to_fixed",
        "delta_q_return_to_fixed",
        "delta_q_multi_to_fixed",
        "delta_sam_score_to_fixed",
        "trace_area_min",
        "trace_area_max",
        "trace_area_final",
        "log1p_trace_area_max_rel_delta",
        "trace_empty_count",
        "trace_component_max",
        "trace_component_final",
        "trace_centroid_max_step",
        "log1p_trace_bbox_w_max_rel_delta",
        "log1p_trace_bbox_h_max_rel_delta",
        "trace_adjacent_dice_min",
        "trace_adjacent_dice_mean",
        "trace_adjacent_dice_last",
        "trace_sam_score_min",
        "trace_sam_score_mean",
        "trace_sam_score_final",
        "log1p_trace_candidate_count_max",
        "log1p_trace_candidate_count_final",
        "is_fixed_b6",
    ]
    feature_names.extend(f"family::{family}" for family in selected_families)
    student_feature_indices = []
    for name in ("q_model_mean", "q_model_rank", "delta_q_model_to_fixed"):
        student_feature_indices.append(len(feature_names))
        feature_names.append(name)

    groups: dict[str, TargetGroup] = {}
    for target_id, rows in sorted(rows_by_target.items()):
        rows.sort(key=lambda row: (row["family"], int(row["bridge_count"]), row["route_id"]))
        expected = len(selected_families) * (max_bridge - min_bridge + 1)
        if len(rows) != expected:
            raise RuntimeError(f"{target_id}: expected {expected} candidates, found {len(rows)}")

        masks = np.stack([load_mask(row["forward_mask_path"], size, root) for row in rows])
        flat = masks.reshape(len(rows), -1).astype(np.float32)
        areas_px = flat.sum(axis=1, dtype=np.float64)
        intersections = flat @ flat.T
        denominators = areas_px[:, None] + areas_px[None, :]
        pair_dice = np.divide(
            2.0 * intersections,
            denominators,
            out=np.ones_like(denominators, dtype=np.float64),
            where=denominators > 0,
        )
        q_multi = (pair_dice.sum(axis=1) - 1.0) / max(len(rows) - 1, 1)

        teacher_scores = []
        if include_student:
            for teacher in TEACHERS:
                key = (teacher, target_id)
                if key not in student_masks:
                    raise RuntimeError(f"Missing {teacher} {split} prediction for {target_id}")
                teacher_flat = student_masks[key].reshape(-1).astype(np.float32)
                teacher_area = float(teacher_flat.sum())
                inter = flat @ teacher_flat
                denom = areas_px + teacher_area
                teacher_scores.append(
                    np.divide(2.0 * inter, denom, out=np.ones_like(denom), where=denom > 0)
                )
            q_model = np.mean(np.stack(teacher_scores), axis=0)
        else:
            q_model = np.zeros(len(rows), dtype=np.float64)

        fixed_matches = [
            index
            for index, row in enumerate(rows)
            if row["family"] == fixed_family and int(row["bridge_count"]) == max_bridge
        ]
        if len(fixed_matches) != 1:
            raise RuntimeError(f"{target_id}: fixed b6 count is {len(fixed_matches)}")
        fixed_index = fixed_matches[0]
        q_return = np.asarray([safe_float(row.get("q_cycle")) for row in rows])
        sam_score = np.asarray([safe_float(row.get("final_sam_score")) for row in rows])
        centroids = np.asarray([mask_centroid(mask) for mask in masks])
        area_fraction = areas_px / float(size * size)
        fixed_area = area_fraction[fixed_index]
        fixed_centroid = centroids[fixed_index]

        q_return_rank = normalized_ranks(q_return)
        q_multi_rank = normalized_ranks(q_multi)
        sam_rank = normalized_ranks(sam_score)
        q_model_rank = normalized_ranks(q_model)
        candidates = []
        for index, row in enumerate(rows):
            values = [
                int(row["bridge_count"]) / max(max_bridge, 1),
                q_return[index],
                q_multi[index],
                q_return_rank[index],
                q_multi_rank[index],
                sam_score[index],
                sam_rank[index],
                safe_float(row.get("cycle_sam_score")),
                safe_float(row.get("path_bottleneck_similarity")),
                safe_float(row.get("path_mean_similarity")),
                area_fraction[index],
                centroids[index, 0],
                centroids[index, 1],
                pair_dice[index, fixed_index],
                math.log((area_fraction[index] + 1e-6) / (fixed_area + 1e-6)),
                float(np.linalg.norm(centroids[index] - fixed_centroid)),
                q_return[index] - q_return[fixed_index],
                q_multi[index] - q_multi[fixed_index],
                sam_score[index] - sam_score[fixed_index],
                safe_float(row.get("trace_area_min")),
                safe_float(row.get("trace_area_max")),
                safe_float(row.get("trace_area_final")),
                math.log1p(max(safe_float(row.get("trace_area_max_rel_delta")), 0.0)),
                safe_float(row.get("trace_empty_count")),
                safe_float(row.get("trace_component_max")),
                safe_float(row.get("trace_component_final")),
                safe_float(row.get("trace_centroid_max_step")),
                math.log1p(max(safe_float(row.get("trace_bbox_w_max_rel_delta")), 0.0)),
                math.log1p(max(safe_float(row.get("trace_bbox_h_max_rel_delta")), 0.0)),
                safe_float(row.get("trace_adjacent_dice_min")),
                safe_float(row.get("trace_adjacent_dice_mean")),
                safe_float(row.get("trace_adjacent_dice_last")),
                safe_float(row.get("trace_sam_score_min")),
                safe_float(row.get("trace_sam_score_mean")),
                safe_float(row.get("trace_sam_score_final")),
                math.log1p(max(safe_float(row.get("trace_candidate_count_max")), 0.0)),
                math.log1p(max(safe_float(row.get("trace_candidate_count_final")), 0.0)),
                float(index == fixed_index),
            ]
            values.extend(float(row["family"] == family) for family in selected_families)
            values.extend(
                [
                    q_model[index],
                    q_model_rank[index],
                    q_model[index] - q_model[fixed_index],
                ]
            )
            candidates.append(
                Candidate(
                    target_id=target_id,
                    family=row["family"],
                    route_id=row["route_id"],
                    bridge=int(row["bridge_count"]),
                    dice=safe_float(row["gt_dice_evaluation_only"]),
                    features=np.asarray(values, dtype=np.float64),
                    q_return=q_return[index],
                    q_multi=q_multi[index],
                    q_model=q_model[index],
                    sam_score=sam_score[index],
                )
            )
        groups[target_id] = TargetGroup(target_id, candidates, fixed_index)

    return groups, feature_names, student_feature_indices


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, groups: list[TargetGroup], indices: list[int]) -> "Standardizer":
        matrix = np.stack([candidate.features[indices] for group in groups for candidate in group.candidates])
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.mean) / self.scale, -8.0, 8.0)


@dataclass
class PairwiseRanker:
    feature_indices: list[int]
    scaler: Standardizer
    weights: np.ndarray
    l2: float
    success: bool

    @classmethod
    def fit(
        cls,
        groups: list[TargetGroup],
        feature_indices: list[int],
        l2: float,
        min_pair_gap: float,
    ) -> "PairwiseRanker":
        scaler = Standardizer.fit(groups, feature_indices)
        pair_rows = []
        pair_weights = []
        for group in groups:
            x = np.stack([scaler.transform(candidate.features[feature_indices]) for candidate in group.candidates])
            y = np.asarray([candidate.dice for candidate in group.candidates])
            target_rows = []
            target_weights = []
            for i in range(len(group.candidates)):
                for j in range(i + 1, len(group.candidates)):
                    delta = y[i] - y[j]
                    if abs(delta) < min_pair_gap:
                        continue
                    target_rows.append((x[i] - x[j]) * (1.0 if delta > 0 else -1.0))
                    target_weights.append(min(abs(delta) / 0.20, 1.0))
            if not target_rows:
                continue
            target_weights_array = np.asarray(target_weights, dtype=np.float64)
            target_weights_array /= target_weights_array.sum()
            pair_rows.extend(target_rows)
            pair_weights.extend(target_weights_array.tolist())
        z = np.stack(pair_rows)
        sample_weight = np.asarray(pair_weights, dtype=np.float64)
        n_targets = max(len(groups), 1)

        def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
            margin = z @ weights
            data_loss = float(np.dot(sample_weight, np.logaddexp(0.0, -margin)) / n_targets)
            probabilities = expit(-margin)
            gradient = -(z.T @ (sample_weight * probabilities)) / n_targets
            return data_loss + 0.5 * l2 * float(weights @ weights), gradient + l2 * weights

        result = minimize(
            objective,
            np.zeros(len(feature_indices), dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-7},
        )
        return cls(feature_indices, scaler, np.asarray(result.x), l2, bool(result.success))

    def score(self, candidate: Candidate) -> float:
        values = self.scaler.transform(candidate.features[self.feature_indices])
        return float(values @ self.weights)


def make_folds(target_ids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    ordered = np.asarray(sorted(target_ids), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    return [chunk.tolist() for chunk in np.array_split(ordered, n_folds)]


def choose_candidate(
    group: TargetGroup,
    scorer: Callable[[Candidate], float],
    fallback_probability: float | None = None,
) -> tuple[Candidate, float, bool]:
    scores = [scorer(candidate) for candidate in group.candidates]
    best_index = max(
        range(len(group.candidates)),
        key=lambda index: (scores[index], -group.candidates[index].bridge, group.candidates[index].route_id),
    )
    switched = best_index != group.fixed_index
    if fallback_probability is not None and switched:
        probability = float(expit(scores[best_index] - scores[group.fixed_index]))
        if probability <= fallback_probability:
            best_index = group.fixed_index
            switched = False
    return group.candidates[best_index], scores[best_index], switched


def evaluate_choices(choices: dict[str, Candidate], groups: dict[str, TargetGroup], seed: int) -> dict:
    target_ids = sorted(choices)
    selected = np.asarray([choices[target_id].dice for target_id in target_ids])
    fixed = np.asarray([groups[target_id].fixed.dice for target_id in target_ids])
    oracle = np.asarray([groups[target_id].oracle.dice for target_id in target_ids])
    delta = selected - fixed
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(10000, dtype=np.float64)
    for index in range(len(bootstrap)):
        sampled = rng.integers(0, len(delta), size=len(delta))
        bootstrap[index] = delta[sampled].mean()
    ordered = np.sort(delta)
    trim = int(math.floor(0.10 * len(ordered)))
    trimmed = ordered[trim : len(ordered) - trim] if trim else ordered
    positive = delta[delta > 1e-12]
    net_gain = float(delta.sum())
    return {
        "n_targets": len(target_ids),
        "selected_dice": float(selected.mean()),
        "fixed_b6_dice": float(fixed.mean()),
        "oracle_dice": float(oracle.mean()),
        "oracle_gap": float(oracle.mean() - selected.mean()),
        "mean_delta_vs_fixed": float(delta.mean()),
        "bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "median_delta_vs_fixed": float(np.median(delta)),
        "trimmed_mean_delta_10pct": float(trimmed.mean()),
        "wins": int((delta > 1e-12).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()),
        "losses": int((delta < -1e-12).sum()),
        "catastrophic_regressions_delta_lt_minus_0.05": int((delta < -0.05).sum()),
        "max_gain": float(delta.max()),
        "max_regression": float(delta.min()),
        "max_gain_share_of_positive_gains": float(delta.max() / positive.sum()) if positive.size else 0.0,
        "max_gain_over_net_gain": float(delta.max() / net_gain) if net_gain > 0 else None,
    }


def select_linear_weights(groups: list[TargetGroup]) -> tuple[float, float, float]:
    best = None
    for ir in range(21):
        for im in range(21 - ir):
            wr = ir * 0.05
            wm = im * 0.05
            ws = 1.0 - wr - wm
            values = []
            for group in groups:
                chosen = max(
                    group.candidates,
                    key=lambda candidate: (
                        wr * candidate.q_return + wm * candidate.q_multi + ws * candidate.q_model,
                        -candidate.bridge,
                        candidate.route_id,
                    ),
                )
                values.append(chosen.dice)
            item = (float(np.mean(values)), -wr, -wm, -ws, (wr, wm, ws))
            if best is None or item[:4] > best[:4]:
                best = item
    return best[4]


def tune_ranker(
    train_groups: list[TargetGroup],
    feature_indices: list[int],
    inner_folds: int,
    seed: int,
    min_pair_gap: float,
) -> tuple[float, float, dict]:
    ids = [group.target_id for group in train_groups]
    by_id = {group.target_id: group for group in train_groups}
    folds = make_folds(ids, inner_folds, seed)
    l2_grid = (0.001, 0.01, 0.1, 1.0)
    fallback_grid = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    best = None
    diagnostics = []
    for l2 in l2_grid:
        predictions: dict[str, tuple[Candidate, Candidate, float, float]] = {}
        successes = []
        for heldout_ids in folds:
            heldout = set(heldout_ids)
            fit_groups = [by_id[target_id] for target_id in ids if target_id not in heldout]
            ranker = PairwiseRanker.fit(fit_groups, feature_indices, l2, min_pair_gap)
            successes.append(ranker.success)
            for target_id in heldout_ids:
                group = by_id[target_id]
                chosen, score, _ = choose_candidate(group, ranker.score)
                fixed_score = ranker.score(group.fixed)
                predictions[target_id] = (chosen, group.fixed, score, fixed_score)
        for threshold in fallback_grid:
            selected = []
            deltas = []
            catastrophic = 0
            switches = 0
            for target_id in ids:
                chosen, fixed, score, fixed_score = predictions[target_id]
                if chosen.route_id != fixed.route_id and float(expit(score - fixed_score)) > threshold:
                    final = chosen
                    switches += 1
                else:
                    final = fixed
                selected.append(final.dice)
                deltas.append(final.dice - fixed.dice)
                catastrophic += int(final.dice - fixed.dice < -0.05)
            item = (float(np.mean(selected)), -catastrophic, float(np.mean(deltas)), -threshold, -l2)
            diagnostics.append(
                {
                    "l2": l2,
                    "fallback_probability": threshold,
                    "selected_dice": item[0],
                    "mean_delta": item[2],
                    "catastrophic_regressions": catastrophic,
                    "switches": switches,
                    "optimizer_success_all": all(successes),
                }
            )
            if best is None or item > best[0]:
                best = (item, l2, threshold)
    return best[1], best[2], {"grid": diagnostics, "best_key": list(best[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--quality-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_vitb256",
    )
    parser.add_argument(
        "--student-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/phase1/predictions",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/pairwise_ranker_vitb256",
    )
    parser.add_argument("--tag", default="lora_p491_e20")
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--min-pair-gap", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    groups, feature_names, student_indices = load_groups(
        args.quality_root,
        args.student_root,
        args.tag,
        args.min_bridge,
        args.max_bridge,
        args.size,
        args.root,
    )
    target_ids = sorted(groups)
    if len(target_ids) < args.outer_folds:
        raise RuntimeError("Not enough target groups for outer CV")
    quality_indices = [index for index in range(len(feature_names)) if index not in student_indices]
    all_indices = list(range(len(feature_names)))
    outer_folds = make_folds(target_ids, args.outer_folds, args.seed)

    methods: dict[str, dict[str, Candidate]] = {
        name: {}
        for name in (
            "fixed_b6",
            "q_multi_only",
            "q_model_mean_only",
            "b7_mean",
            "linear3_outer_train",
            "pairwise_quality",
            "pairwise_quality_fallback",
            "pairwise_qmodel",
            "pairwise_qmodel_fallback",
        )
    }
    selection_meta: dict[str, dict[str, dict]] = {name: {} for name in methods}
    fold_reports = []
    for outer_index, heldout_ids in enumerate(outer_folds):
        heldout = set(heldout_ids)
        train_groups = [groups[target_id] for target_id in target_ids if target_id not in heldout]
        validation_groups = [groups[target_id] for target_id in heldout_ids]
        print(
            f"outer fold {outer_index + 1}/{args.outer_folds}: "
            f"train={len(train_groups)} heldout={len(validation_groups)}",
            flush=True,
        )

        linear_weights = select_linear_weights(train_groups)
        fold_report = {
            "fold": outer_index,
            "heldout_targets": heldout_ids,
            "linear3_weights": {
                "q_return": linear_weights[0],
                "q_multi": linear_weights[1],
                "q_model": linear_weights[2],
            },
            "rankers": {},
        }

        fitted_rankers = {}
        for label, indices in (("quality", quality_indices), ("qmodel", all_indices)):
            l2, threshold, tuning = tune_ranker(
                train_groups,
                indices,
                args.inner_folds,
                args.seed + 1000 * (outer_index + 1) + (0 if label == "quality" else 100),
                args.min_pair_gap,
            )
            ranker = PairwiseRanker.fit(train_groups, indices, l2, args.min_pair_gap)
            fitted_rankers[label] = (ranker, threshold)
            fold_report["rankers"][label] = {
                "l2": l2,
                "fallback_probability": threshold,
                "optimizer_success": ranker.success,
                "tuning": tuning,
            }
            print(
                f"  {label}: l2={l2:g} fallback={threshold:.2f} success={ranker.success}",
                flush=True,
            )

        wr, wm, ws = linear_weights
        for group in validation_groups:
            fixed = group.fixed
            method_scorers: dict[str, Callable[[Candidate], float]] = {
                "q_multi_only": lambda candidate: candidate.q_multi,
                "q_model_mean_only": lambda candidate: candidate.q_model,
                "b7_mean": lambda candidate: (
                    max(candidate.q_return, 1e-6)
                    * max(candidate.q_multi, 1e-6) ** 2
                    * max(candidate.q_model, 1e-6) ** 2
                )
                ** 0.2,
                "linear3_outer_train": lambda candidate, a=wr, b=wm, c=ws: (
                    a * candidate.q_return + b * candidate.q_multi + c * candidate.q_model
                ),
            }
            methods["fixed_b6"][group.target_id] = fixed
            selection_meta["fixed_b6"][group.target_id] = {"score": None, "switched": False}
            for method, scorer in method_scorers.items():
                chosen, score, switched = choose_candidate(group, scorer)
                methods[method][group.target_id] = chosen
                selection_meta[method][group.target_id] = {"score": score, "switched": switched}

            for label, (ranker, threshold) in fitted_rankers.items():
                raw_name = f"pairwise_{label}"
                chosen, score, switched = choose_candidate(group, ranker.score)
                methods[raw_name][group.target_id] = chosen
                selection_meta[raw_name][group.target_id] = {"score": score, "switched": switched}
                fallback_name = f"pairwise_{label}_fallback"
                chosen, score, switched = choose_candidate(group, ranker.score, threshold)
                methods[fallback_name][group.target_id] = chosen
                selection_meta[fallback_name][group.target_id] = {
                    "score": score,
                    "switched": switched,
                    "fallback_probability": threshold,
                }
        fold_reports.append(fold_report)

    summaries = {
        method: evaluate_choices(choices, groups, args.seed + index)
        for index, (method, choices) in enumerate(methods.items())
    }
    for method, summary in summaries.items():
        summary["switches_from_fixed"] = int(
            sum(meta["switched"] for meta in selection_meta[method].values())
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = f"pairwise_ranker_validation_oof_{args.tag}_b{args.min_bridge}_b{args.max_bridge}"
    report = {
        "protocol": {
            "split": "validation_only_oof",
            "test_loaded": False,
            "n_targets": len(target_ids),
            "candidates_per_target": len(FAMILIES) * (args.max_bridge - args.min_bridge + 1),
            "families": list(FAMILIES),
            "fixed_baseline": {"family": FIXED_FAMILY, "bridge": args.max_bridge},
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "min_pair_gap": args.min_pair_gap,
            "seed": args.seed,
            "teachers": list(TEACHERS),
        },
        "features": {
            "all": feature_names,
            "quality_only": [feature_names[index] for index in quality_indices],
            "student_features": [feature_names[index] for index in student_indices],
        },
        "summary": summaries,
        "folds": fold_reports,
    }
    (args.output_root / f"{stem}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    csv_fields = ["target_id", "oracle_dice", "method", "selected_dice", "fixed_b6_dice", "delta", "family", "bridge", "route_id", "score", "switched"]
    with (args.output_root / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for target_id in target_ids:
            for method in methods:
                chosen = methods[method][target_id]
                meta = selection_meta[method][target_id]
                writer.writerow(
                    {
                        "target_id": target_id,
                        "oracle_dice": groups[target_id].oracle.dice,
                        "method": method,
                        "selected_dice": chosen.dice,
                        "fixed_b6_dice": groups[target_id].fixed.dice,
                        "delta": chosen.dice - groups[target_id].fixed.dice,
                        "family": chosen.family,
                        "bridge": chosen.bridge,
                        "route_id": chosen.route_id,
                        "score": meta["score"],
                        "switched": meta["switched"],
                    }
                )

    lines = [
        f"# Pairwise route ranker validation OOF ({args.tag}, b{args.min_bridge}-b{args.max_bridge})",
        "",
        "This is a validation-only, target-grouped nested-CV audit. The test split was not loaded.",
        "",
        "| Method | Dice | Delta vs b6 | 95% bootstrap CI | Trimmed delta | W/T/L | Cat. regressions | Switches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, summary in summaries.items():
        ci = summary["bootstrap_95_ci"]
        lines.append(
            f"| {method} | {summary['selected_dice']:.6f} | {summary['mean_delta_vs_fixed']:+.6f} "
            f"| [{ci[0]:+.6f}, {ci[1]:+.6f}] | {summary['trimmed_mean_delta_10pct']:+.6f} "
            f"| {summary['wins']}/{summary['ties']}/{summary['losses']} "
            f"| {summary['catastrophic_regressions_delta_lt_minus_0.05']} | {summary['switches_from_fixed']} |"
        )
    lines.extend(
        [
            "",
            "## Concentration diagnostics",
            "",
            "| Method | Max gain | Max regression | Largest gain / positive gains | Largest gain / net gain |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method, summary in summaries.items():
        net_share = summary["max_gain_over_net_gain"]
        net_text = "n/a" if net_share is None else f"{net_share:.3f}"
        lines.append(
            f"| {method} | {summary['max_gain']:+.6f} | {summary['max_regression']:+.6f} "
            f"| {summary['max_gain_share_of_positive_gains']:.3f} | {net_text} |"
        )
    lines.extend(
        [
            "",
            "## Fold choices",
            "",
            "| Fold | Linear weights (return/multi/model) | Quality ranker (L2/fallback) | Student ranker (L2/fallback) |",
            "|---:|---:|---:|---:|",
        ]
    )
    for fold in fold_reports:
        lw = fold["linear3_weights"]
        qr = fold["rankers"]["quality"]
        sr = fold["rankers"]["qmodel"]
        lines.append(
            f"| {fold['fold']} | {lw['q_return']:.2f}/{lw['q_multi']:.2f}/{lw['q_model']:.2f} "
            f"| {qr['l2']:g}/{qr['fallback_probability']:.2f} "
            f"| {sr['l2']:g}/{sr['fallback_probability']:.2f} |"
        )
    (args.output_root / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)
    print(f"PAIRWISE_RANKER_OOF_DONE {args.output_root / stem}", flush=True)


if __name__ == "__main__":
    main()
