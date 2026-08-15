#!/usr/bin/env python3
"""Train a validation-only harm gate and evaluate one frozen policy on two tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import run_pairwise_route_ranker_vitb256 as core


@dataclass
class RiskGate:
    feature_indices: list[int]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    l2: float
    positive_weight: float
    label_mode: str
    outcome_delta: float
    success: bool

    @classmethod
    def fit(
        cls,
        groups: list[core.TargetGroup],
        feature_indices: list[int],
        l2: float,
        positive_weight: float,
        outcome_delta: float,
        label_mode: str,
    ) -> "RiskGate":
        if label_mode not in {"harm", "rescue"}:
            raise ValueError(f"Unsupported gate label mode: {label_mode}")
        matrix = np.stack(
            [
                candidate.features[feature_indices]
                for group in groups
                for index, candidate in enumerate(group.candidates)
                if index != group.fixed_index
            ]
        )
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-8] = 1.0

        rows: list[np.ndarray] = []
        labels: list[float] = []
        sample_weights: list[float] = []
        for group in groups:
            target_rows = []
            target_labels = []
            for index, candidate in enumerate(group.candidates):
                if index == group.fixed_index:
                    continue
                values = np.clip(
                    (candidate.features[feature_indices] - mean) / scale,
                    -8.0,
                    8.0,
                )
                target_rows.append(values)
                delta = candidate.dice - group.fixed.dice
                positive = delta < outcome_delta if label_mode == "harm" else delta > outcome_delta
                target_labels.append(float(positive))
            target_weights = np.where(
                np.asarray(target_labels) > 0.5,
                positive_weight,
                1.0,
            )
            target_weights /= target_weights.sum()
            rows.extend(target_rows)
            labels.extend(target_labels)
            sample_weights.extend(target_weights.tolist())

        x = np.stack(rows)
        y = np.asarray(labels, dtype=np.float64)
        w_sample = np.asarray(sample_weights, dtype=np.float64) / max(len(groups), 1)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            weights = parameters[:-1]
            intercept = parameters[-1]
            logits = x @ weights + intercept
            losses = np.logaddexp(0.0, logits) - y * logits
            residual = expit(logits) - y
            value = float(w_sample @ losses) + 0.5 * l2 * float(weights @ weights)
            gradient = np.concatenate(
                [x.T @ (w_sample * residual) + l2 * weights, [w_sample @ residual]]
            )
            return value, gradient

        result = minimize(
            objective,
            np.zeros(x.shape[1] + 1, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-7},
        )
        return cls(
            feature_indices=feature_indices,
            mean=mean,
            scale=scale,
            weights=np.asarray(result.x[:-1]),
            intercept=float(result.x[-1]),
            l2=l2,
            positive_weight=positive_weight,
            label_mode=label_mode,
            outcome_delta=outcome_delta,
            success=bool(result.success),
        )

    def predict(self, candidate: core.Candidate) -> float:
        values = np.clip(
            (candidate.features[self.feature_indices] - self.mean) / self.scale,
            -8.0,
            8.0,
        )
        return float(expit(values @ self.weights + self.intercept))


def feature_profiles(feature_names: list[str], student_indices: list[int]) -> tuple[list[int], list[int]]:
    student_names = {feature_names[index] for index in student_indices}
    consensus_names = {"q_multi", "q_multi_rank"}
    relative_names = {
        "fixed_mask_dice",
        "log_area_ratio_to_fixed",
        "centroid_distance_to_fixed",
        "delta_q_return_to_fixed",
        "delta_q_multi_to_fixed",
        "delta_sam_score_to_fixed",
    }
    geometry_names = {"mask_area", "mask_centroid_x", "mask_centroid_y"}
    internal_names = (
        set(feature_names)
        - consensus_names
        - relative_names
        - geometry_names
        - student_names
    )
    rank_names = internal_names | consensus_names
    risk_names = set(feature_names) - student_names
    rank_indices = [index for index, name in enumerate(feature_names) if name in rank_names]
    risk_indices = [index for index, name in enumerate(feature_names) if name in risk_names]
    return rank_indices, risk_indices


def deltas_summary(deltas: np.ndarray, seed: int, bootstrap_samples: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(len(bootstrap)):
        sample = rng.integers(0, len(deltas), size=len(deltas))
        bootstrap[index] = deltas[sample].mean()
    ordered = np.sort(deltas)
    trim = int(math.floor(0.10 * len(ordered)))
    trimmed = ordered[trim : len(ordered) - trim] if trim else ordered
    worst_count = max(1, int(math.ceil(0.10 * len(ordered))))
    return {
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "trimmed_mean_delta_10pct": float(trimmed.mean()),
        "cvar_worst_10pct": float(ordered[:worst_count].mean()),
        "bootstrap_95_ci": (
            [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
            if bootstrap_samples
            else None
        ),
        "wins": int((deltas > 1e-12).sum()),
        "ties": int((np.abs(deltas) <= 1e-12).sum()),
        "losses": int((deltas < -1e-12).sum()),
        "catastrophic": int((deltas < -0.05).sum()),
        "max_gain": float(deltas.max()),
        "max_regression": float(deltas.min()),
    }


def summarize_policy(
    records: list[dict],
    risk_threshold: float,
    rescue_threshold: float,
    seed: int,
    bootstrap_samples: int = 10000,
) -> dict:
    accepted = [
        row["fixed_invalid"]
        or row["risk"] <= risk_threshold
        or row["rescue"] >= rescue_threshold
        for row in records
    ]
    deltas = np.asarray(
        [row["raw_delta"] if keep else 0.0 for row, keep in zip(records, accepted)],
        dtype=np.float64,
    )
    summary = deltas_summary(deltas, seed, bootstrap_samples)
    summary["risk_threshold"] = risk_threshold
    summary["rescue_threshold"] = rescue_threshold
    summary["switches"] = int(sum(accepted))
    return summary


def choose_joint_policy(
    audits: list[dict], seed: int, enable_learned_rescue: bool
) -> tuple[float, float, list[dict]]:
    risk_thresholds = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
                       0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    rescue_thresholds = (
        [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.01]
        if enable_learned_rescue
        else [1.01]
    )
    sweep = []
    best = None
    for risk_threshold in risk_thresholds:
        for rescue_threshold in rescue_thresholds:
            per_seed = [
                summarize_policy(
                    audit["records"],
                    risk_threshold,
                    rescue_threshold,
                    seed + 1000 * audit_index,
                    0,
                )
                for audit_index, audit in enumerate(audits)
            ]
            max_catastrophic = max(item["catastrophic"] for item in per_seed)
            mean_delta = float(np.mean([item["mean_delta"] for item in per_seed]))
            min_delta = min(item["mean_delta"] for item in per_seed)
            switches = int(sum(item["switches"] for item in per_seed))
            item = {
                "risk_threshold": risk_threshold,
                "rescue_threshold": rescue_threshold,
                "max_catastrophic_across_seeds": max_catastrophic,
                "mean_delta_across_seeds": mean_delta,
                "min_delta_across_seeds": min_delta,
                "switches_across_seeds": switches,
                "per_seed": per_seed,
            }
            sweep.append(item)
            key = (-max_catastrophic, mean_delta, min_delta, switches)
            if best is None or key > best[0]:
                best = (key, risk_threshold, rescue_threshold)
    return best[1], best[2], sweep


def crossfit_validation(
    groups: dict[str, core.TargetGroup],
    rank_indices: list[int],
    risk_indices: list[int],
    seed: int,
    rank_l2: float,
    risk_l2: float,
    positive_weight: float,
    harm_delta: float,
    rescue_l2: float,
    rescue_positive_weight: float,
    rescue_delta: float,
    min_pair_gap: float,
    mask_area_index: int,
    fixed_area_floor: float,
) -> dict:
    target_ids = sorted(groups)
    folds = core.make_folds(target_ids, 5, seed)
    records = []
    for fold_index, heldout_ids in enumerate(folds):
        heldout = set(heldout_ids)
        train_groups = [groups[target_id] for target_id in target_ids if target_id not in heldout]
        ranker = core.PairwiseRanker.fit(train_groups, rank_indices, rank_l2, min_pair_gap)
        gate = RiskGate.fit(
            train_groups, risk_indices, risk_l2, positive_weight, harm_delta, "harm"
        )
        rescue_gate = RiskGate.fit(
            train_groups,
            risk_indices,
            rescue_l2,
            rescue_positive_weight,
            rescue_delta,
            "rescue",
        )
        for target_id in heldout_ids:
            group = groups[target_id]
            candidate, score, _ = core.choose_candidate(group, ranker.score)
            records.append(
                {
                    "target_id": target_id,
                    "fold": fold_index,
                    "raw_delta": candidate.dice - group.fixed.dice,
                    "risk": gate.predict(candidate),
                    "rescue": rescue_gate.predict(candidate),
                    "fixed_invalid": float(group.fixed.features[mask_area_index]) <= fixed_area_floor,
                    "family": candidate.family,
                    "bridge": candidate.bridge,
                    "score_margin": score - ranker.score(group.fixed),
                }
            )
    raw = deltas_summary(np.asarray([row["raw_delta"] for row in records]), seed + 1)
    return {"seed": seed, "raw": raw, "records": records}


def load_frozen_ranker(path: Path, feature_names: list[str]) -> Callable[[core.Candidate], float]:
    model = json.loads(path.read_text(encoding="utf-8"))
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    indices = np.asarray([name_to_index[name] for name in model["feature_names"]], dtype=np.int64)
    mean = np.asarray(model["standardizer_mean"], dtype=np.float64)
    scale = np.asarray(model["standardizer_scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)

    def score(candidate: core.Candidate) -> float:
        values = np.clip((candidate.features[indices] - mean) / scale, -8.0, 8.0)
        return float(values @ weights)

    return score


def evaluate_dataset(
    label: str,
    groups: dict[str, core.TargetGroup],
    rank_score: Callable[[core.Candidate], float],
    gate: RiskGate,
    rescue_gate: RiskGate,
    risk_threshold: float,
    rescue_threshold: float,
    seed: int,
    mask_area_index: int,
    fixed_area_floor: float,
) -> tuple[dict, list[dict]]:
    rows = []
    raw_deltas = []
    gated_deltas = []
    raw_dice = []
    gated_dice = []
    fixed_dice = []
    for target_id, group in sorted(groups.items()):
        candidate, score, _ = core.choose_candidate(group, rank_score)
        risk = gate.predict(candidate)
        rescue = rescue_gate.predict(candidate)
        fixed_mask_area = float(group.fixed.features[mask_area_index])
        fixed_invalid = fixed_mask_area <= fixed_area_floor
        accepted = (
            candidate.route_id == group.fixed.route_id
            or fixed_invalid
            or risk <= risk_threshold
            or rescue >= rescue_threshold
        )
        final = candidate if accepted else group.fixed
        raw_delta = candidate.dice - group.fixed.dice
        gated_delta = final.dice - group.fixed.dice
        raw_deltas.append(raw_delta)
        gated_deltas.append(gated_delta)
        raw_dice.append(candidate.dice)
        gated_dice.append(final.dice)
        fixed_dice.append(group.fixed.dice)
        rows.append(
            {
                "dataset": label,
                "target_id": target_id,
                "raw_family": candidate.family,
                "raw_bridge": candidate.bridge,
                "raw_dice": candidate.dice,
                "fixed_dice": group.fixed.dice,
                "fixed_mask_area": fixed_mask_area,
                "fixed_invalid": fixed_invalid,
                "raw_delta": raw_delta,
                "risk": risk,
                "rescue": rescue,
                "accepted": accepted,
                "gated_dice": final.dice,
                "gated_delta": gated_delta,
                "score_margin": score - rank_score(group.fixed),
            }
        )
    summary = {
        "n_targets": len(rows),
        "fixed_dice": float(np.mean(fixed_dice)),
        "raw_selected_dice": float(np.mean(raw_dice)),
        "gated_selected_dice": float(np.mean(gated_dice)),
        "raw": deltas_summary(np.asarray(raw_deltas), seed),
        "gated": deltas_summary(np.asarray(gated_deltas), seed + 1),
        "accepted_switches": int(sum(row["accepted"] for row in rows)),
        "fallbacks": int(sum(not row["accepted"] for row in rows)),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=core.ROOT)
    parser.add_argument(
        "--kvasir-quality-root",
        type=Path,
        default=core.ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_vitb256",
    )
    parser.add_argument(
        "--clinicdb-quality-root",
        type=Path,
        default=core.ROOT / "work/clinicdb_external_kvasir8/stage1_feature_knn_vitb256",
    )
    parser.add_argument(
        "--student-root",
        type=Path,
        default=core.ROOT / "work/kvasir_1pct_anchors/phase1/predictions",
    )
    parser.add_argument(
        "--ranker-model",
        type=Path,
        default=core.ROOT / "work/kvasir_1pct_anchors/pairwise_ranker_vitb256/frozen_internal_consensus/frozen_internal_plus_consensus_lora_p491_e20.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=core.ROOT / "work/kvasir_1pct_anchors/pairwise_ranker_vitb256/risk_gate_dual_dataset",
    )
    parser.add_argument("--tag", default="lora_p491_e20")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260810, 20260811, 20260812])
    parser.add_argument("--rank-l2", type=float, default=0.001)
    parser.add_argument("--risk-l2", type=float, default=0.1)
    parser.add_argument("--positive-weight", type=float, default=5.0)
    parser.add_argument("--harm-delta", type=float, default=-0.05)
    parser.add_argument("--rescue-l2", type=float, default=0.1)
    parser.add_argument("--rescue-positive-weight", type=float, default=3.0)
    parser.add_argument("--rescue-delta", type=float, default=0.05)
    parser.add_argument("--min-pair-gap", type=float, default=0.02)
    parser.add_argument("--fixed-area-floor", type=float, default=1e-6)
    parser.add_argument("--enable-learned-rescue", action="store_true")
    args = parser.parse_args()

    validation, feature_names, student_indices = core.load_groups(
        args.kvasir_quality_root,
        args.student_root,
        args.tag,
        min_bridge=3,
        max_bridge=6,
        size=256,
        root=args.root,
        split="validation",
        include_student=False,
    )
    rank_indices, risk_indices = feature_profiles(feature_names, student_indices)
    mask_area_index = feature_names.index("mask_area")
    audits = []
    for seed in args.seeds:
        print(f"validation crossfit seed={seed}", flush=True)
        audit = crossfit_validation(
            validation,
            rank_indices,
            risk_indices,
            seed,
            args.rank_l2,
            args.risk_l2,
            args.positive_weight,
            args.harm_delta,
            args.rescue_l2,
            args.rescue_positive_weight,
            args.rescue_delta,
            args.min_pair_gap,
            mask_area_index,
            args.fixed_area_floor,
        )
        audits.append(audit)
        print(
            f"  raw={audit['raw']['mean_delta']:+.6f} "
            f"cat={audit['raw']['catastrophic']}",
            flush=True,
        )

    risk_threshold, rescue_threshold, policy_sweep = choose_joint_policy(
        audits, args.seeds[0] + 50000, args.enable_learned_rescue
    )
    for audit_index, audit in enumerate(audits):
        audit["gated"] = summarize_policy(
            audit["records"],
            risk_threshold,
            rescue_threshold,
            args.seeds[0] + 60000 + audit_index,
        )
        print(
            f"  seed={audit['seed']} gated={audit['gated']['mean_delta']:+.6f} "
            f"cat={audit['raw']['catastrophic']}->{audit['gated']['catastrophic']}",
            flush=True,
        )
    all_validation_groups = [validation[target_id] for target_id in sorted(validation)]
    gate = RiskGate.fit(
        all_validation_groups,
        risk_indices,
        args.risk_l2,
        args.positive_weight,
        args.harm_delta,
        "harm",
    )
    rescue_gate = RiskGate.fit(
        all_validation_groups,
        risk_indices,
        args.rescue_l2,
        args.rescue_positive_weight,
        args.rescue_delta,
        "rescue",
    )
    frozen_gate = {
        "protocol": "validation_only_grouped_crossfit_then_fit_all",
        "test_loaded_during_fit": False,
        "threshold_policy": "joint_three-seed_minimax_catastrophic_then_mean_delta",
        "risk_threshold": risk_threshold,
        "rescue_threshold": rescue_threshold,
        "learned_rescue_enabled": args.enable_learned_rescue,
        "harm_delta": args.harm_delta,
        "rescue_delta": args.rescue_delta,
        "feature_names": [feature_names[index] for index in risk_indices],
        "harm_gate": {
            "l2": args.risk_l2,
            "positive_weight": args.positive_weight,
            "optimizer_success": gate.success,
            "standardizer_mean": gate.mean.tolist(),
            "standardizer_scale": gate.scale.tolist(),
            "weights": gate.weights.tolist(),
            "intercept": gate.intercept,
        },
        "rescue_gate": {
            "l2": args.rescue_l2,
            "positive_weight": args.rescue_positive_weight,
            "optimizer_success": rescue_gate.success,
            "standardizer_mean": rescue_gate.mean.tolist(),
            "standardizer_scale": rescue_gate.scale.tolist(),
            "weights": rescue_gate.weights.tolist(),
            "intercept": rescue_gate.intercept,
        },
    }

    kvasir_test, test_names, _ = core.load_groups(
        args.kvasir_quality_root,
        args.student_root,
        args.tag,
        min_bridge=3,
        max_bridge=6,
        size=256,
        root=args.root,
        split="test",
        include_student=False,
    )
    if test_names != feature_names:
        raise RuntimeError("Kvasir test feature order differs from validation")
    clinicdb_test, clinic_names, _ = core.load_groups(
        args.clinicdb_quality_root,
        args.student_root,
        args.tag,
        min_bridge=3,
        max_bridge=6,
        size=256,
        root=args.root,
        split="test",
        include_student=False,
    )
    if clinic_names != feature_names:
        raise RuntimeError("ClinicDB feature order differs from validation")
    rank_score = load_frozen_ranker(args.ranker_model, feature_names)
    kvasir_summary, kvasir_rows = evaluate_dataset(
        "Kvasir-SEG",
        kvasir_test,
        rank_score,
        gate,
        rescue_gate,
        risk_threshold,
        rescue_threshold,
        args.seeds[0] + 100,
        mask_area_index,
        args.fixed_area_floor,
    )
    clinic_summary, clinic_rows = evaluate_dataset(
        "CVC-ClinicDB",
        clinicdb_test,
        rank_score,
        gate,
        rescue_gate,
        risk_threshold,
        rescue_threshold,
        args.seeds[0] + 200,
        mask_area_index,
        args.fixed_area_floor,
    )

    report = {
        "protocol": {
            "ranker_model": str(args.ranker_model),
            "risk_gate_training": "Kvasir validation only",
            "student_loaded": False,
            "risk_threshold": risk_threshold,
            "rescue_threshold": rescue_threshold,
            "harm_delta": args.harm_delta,
            "rescue_delta": args.rescue_delta,
            "learned_rescue_enabled": args.enable_learned_rescue,
            "fixed_area_floor": args.fixed_area_floor,
            "dual_dataset_acceptance": ["Kvasir-SEG test", "CVC-ClinicDB test"],
        },
        "validation_oof": audits,
        "validation_policy_sweep": policy_sweep,
        "frozen_gate": frozen_gate,
        "test": {"Kvasir-SEG": kvasir_summary, "CVC-ClinicDB": clinic_summary},
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = args.output_root / f"risk_gate_dual_dataset_{args.tag}"
    (stem.with_suffix(".json")).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / f"frozen_risk_gate_{args.tag}.json").write_text(
        json.dumps(frozen_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(stem.with_suffix(".csv"), kvasir_rows + clinic_rows)

    lines = [
        "# Baseline-aware risk gate: dual-dataset frozen evaluation",
        "",
        f"Frozen thresholds: harm risk <= `{risk_threshold:.2f}` or rescue probability >= "
        f"`{rescue_threshold:.2f}`; labels: harm `delta < {args.harm_delta:.2f}`, "
        f"rescue `delta > +{args.rescue_delta:.2f}`.",
        "The gate and threshold use Kvasir validation only; no student predictions are loaded.",
        "",
        "| Dataset | fixed b6 | raw ranker | gated ranker | raw delta | gated delta | catastrophic raw->gated | CVaR10 raw->gated | fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in (("Kvasir-SEG", kvasir_summary), ("CVC-ClinicDB", clinic_summary)):
        lines.append(
            f"| {label} | {summary['fixed_dice']:.6f} | {summary['raw_selected_dice']:.6f} | "
            f"{summary['gated_selected_dice']:.6f} | {summary['raw']['mean_delta']:+.6f} | "
            f"{summary['gated']['mean_delta']:+.6f} | {summary['raw']['catastrophic']}->{summary['gated']['catastrophic']} | "
            f"{summary['raw']['cvar_worst_10pct']:+.6f}->{summary['gated']['cvar_worst_10pct']:+.6f} | "
            f"{summary['fallbacks']} |"
        )
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["test"], indent=2), flush=True)
    print(f"RISK_GATE_DUAL_DATASET_DONE {stem}", flush=True)


if __name__ == "__main__":
    main()
