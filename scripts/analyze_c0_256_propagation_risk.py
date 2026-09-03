#!/usr/bin/env python3
"""Validation-only diagnosis of confidently self-consistent route failures.

All candidate features are available without target GT.  Ground-truth Dice is
restricted to retrospective risk labels and post-hoc diagnostic evaluation.
No test files, SAM3 checkpoints, propagation code, or GPU libraries are read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from analyze_c0_256_propagation_aware_routing import (
    GT_FIELD,
    artifact_record,
    candidate_key,
    read_jsonl,
    selection_key,
    write_json,
    write_jsonl,
)


ROUTE_GROUPS = (
    "beneficial",
    "approximately_safe",
    "moderate_degradation",
    "severe_degradation",
)
GROUP_LABELS = {
    "beneficial": "明确改善 ΔDice > 0.01",
    "approximately_safe": "基本安全 -0.01 ≤ ΔDice ≤ 0.01",
    "moderate_degradation": "中度退化 -0.10 < ΔDice < -0.01",
    "severe_degradation": "严重退化 ΔDice ≤ -0.10",
}
RAW_NUMERIC_FIELDS = (
    "path_bottleneck_similarity",
    "path_mean_similarity",
    "q_return",
    "q_cycle",
    "q_multi",
    "q_model",
    "b7",
    "forward_sam_score",
    "trace_adjacent_dice_mean",
    "trace_adjacent_dice_min",
    "trace_adjacent_dice_last",
    "trace_sam_score_mean",
    "trace_sam_score_min",
    "trace_sam_score_final",
    "trace_area_final",
    "trace_area_min",
    "trace_area_max",
    "trace_area_max_rel_delta",
    "trace_empty_count",
    "trace_component_final",
    "trace_component_max",
    "trace_centroid_max_step",
    "trace_bbox_w_max_rel_delta",
    "trace_bbox_h_max_rel_delta",
    "trace_candidate_count_final",
    "trace_candidate_count_max",
    "cycle_candidate_count",
    "cycle_sam_score",
    "final_candidate_count",
    "bridge_count",
)
REQUESTED_ABSENT_OR_DERIVED_FIELDS = (
    "mask_area",
    "mask_area_ratio",
    "mask_area_change",
    "returned_mask_area",
    "forward_mask_area",
)
CORE_FEATURES = (
    "q_return",
    "q_multi",
    "q_model",
    "b7",
    "path_mean_similarity",
    "path_bottleneck_similarity",
    "forward_sam_score",
    "trace_adjacent_dice_min",
    "trace_adjacent_dice_mean",
    "trace_adjacent_dice_last",
    "trace_sam_score_min",
    "trace_sam_score_mean",
    "trace_area_final",
    "trace_area_max_rel_delta",
    "trace_centroid_max_step",
    "trace_bbox_w_max_rel_delta",
    "trace_bbox_h_max_rel_delta",
    "trace_empty_count",
    "bridge_count",
    "route_mode_patch_correspondence",
)
DERIVED_DISAGREEMENTS = (
    "q_return_minus_q_multi",
    "abs_q_return_minus_q_multi",
    "q_return_minus_q_model",
    "abs_q_return_minus_q_model",
    "q_return_minus_trace_adjacent_dice_min",
    "q_return_minus_trace_adjacent_dice_mean",
    "peer_disagreement",
    "student_disagreement",
    "trace_area_span",
)
GROUP_SUMMARY_FIELDS = (
    "q_return",
    "q_multi",
    "q_model",
    "b7",
    "path_mean_similarity",
    "path_bottleneck_similarity",
    "trace_adjacent_dice_min",
    "trace_adjacent_dice_mean",
    "trace_sam_score_min",
    "trace_area_final",
    "trace_area_max_rel_delta",
    "trace_centroid_max_step",
    "q_return_minus_q_multi",
    "q_return_minus_q_model",
    "abs_q_return_minus_q_multi",
    "abs_q_return_minus_q_model",
    "delta_q_return",
    "delta_q_multi",
    "delta_q_model",
    "delta_b7",
    "delta_path_mean_similarity",
    "delta_path_bottleneck_similarity",
    "delta_trace_adjacent_dice_min",
    "delta_trace_adjacent_dice_mean",
)


def scalar(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite unsupervised field {field}: {value!r}")
    return result


def values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def read_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_frozen_validation(
    candidates_path: Path,
    per_target_path: Path,
    summary_path: Path,
    raw_audit_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    candidates = read_jsonl(candidates_path)
    previous_per_target_rows = read_jsonl(per_target_path)
    previous_per_target = {row["target_id"]: row for row in previous_per_target_rows}
    summary = read_summary(summary_path)
    raw_rows = read_jsonl(raw_audit_path)
    raw_lookup = {candidate_key(row): row for row in raw_rows}
    if len(candidates) != 1400 or len(raw_rows) != 1400:
        raise ValueError(f"Frozen candidate count mismatch: normalized={len(candidates)}, raw={len(raw_rows)}")
    if len(previous_per_target_rows) != 100 or len(previous_per_target) != 100:
        raise ValueError("Frozen per-target artifact must contain exactly 100 distinct targets")
    if summary.get("strongest_knn_baseline") != "knn_mean":
        raise ValueError("Previous frozen baseline is not the required KNN mean selector")
    if summary["candidate_audit"]["candidate_count"] != 1400:
        raise ValueError("Previous candidate-count audit disagrees with frozen artifact")
    if len(raw_lookup) != len(raw_rows):
        raise ValueError("Raw historical candidate audit contains duplicate candidate identities")

    field_presence = Counter()
    for row in candidates:
        key = candidate_key(row)
        if key not in raw_lookup:
            raise ValueError(f"Raw historical candidate is missing: {key}")
        raw = raw_lookup[key]
        if row.get("target_split") != "validation" or raw.get("target_split") != "validation":
            raise ValueError(f"Non-validation candidate is prohibited: {key}")
        if row.get("target_gt_used_for_search_or_inference") is not False:
            raise ValueError(f"Candidate does not certify GT-free search/inference: {key}")
        for field in (
            "target_id",
            "route_id",
            "route_mode",
            "bridge_count",
            "q_return",
            "q_cycle",
            "q_multi",
            "q_model",
            "b7",
            "path_mean_similarity",
            "path_bottleneck_similarity",
            GT_FIELD,
        ):
            if field not in raw or not values_equal(row[field], raw[field]):
                raise ValueError(f"Frozen/raw mismatch for {field}: {key}")
        for field in RAW_NUMERIC_FIELDS:
            if field == "forward_sam_score":
                value = row.get(field, raw.get("final_sam_score"))
            elif field in raw:
                value = raw[field]
            else:
                value = row.get(field)
            if value is not None:
                row[field] = scalar(value, field)
                field_presence[field] += 1
        row["bridge_count"] = int(row["bridge_count"])
        row["route_mode_patch_correspondence"] = float(
            row["route_mode"] == "sam3enc_anchor_conditioned_patch_correspondence"
        )
    available_fields = sorted(field for field, count in field_presence.items() if count == len(candidates))
    partial_fields = {
        field: count for field, count in sorted(field_presence.items()) if count != len(candidates)
    }
    raw_keys = set().union(*(row.keys() for row in raw_rows))
    requested_status = {
        field: (
            "available"
            if field in raw_keys
            else "derived_without_gt_from_trace_area_final"
            if field == "mask_area_ratio" and "trace_area_final" in raw_keys
            else "derived_without_gt_from_trace_area_max_rel_delta"
            if field == "mask_area_change" and "trace_area_max_rel_delta" in raw_keys
            else "missing_not_recomputed"
        )
        for field in REQUESTED_ABSENT_OR_DERIVED_FIELDS
    }
    audit = {
        "candidate_count": len(candidates),
        "target_count": len(previous_per_target),
        "fully_available_numeric_fields": available_fields,
        "partially_available_numeric_fields": partial_fields,
        "requested_optional_field_status": requested_status,
        "raw_gt_prefixed_fields_excluded_from_features": sorted(
            field for field in raw_keys if field.startswith("gt_")
        ),
        "input_artifacts": [
            artifact_record(candidates_path, "prior_validation_candidates"),
            artifact_record(per_target_path, "prior_validation_per_target"),
            artifact_record(summary_path, "prior_validation_summary"),
            artifact_record(raw_audit_path, "historical_validation_b7_full_trace_audit"),
        ],
    }
    return candidates, previous_per_target, summary, audit


def risk_group(delta_dice: float) -> str:
    if delta_dice > 0.01:
        return "beneficial"
    if -0.01 <= delta_dice <= 0.01:
        return "approximately_safe"
    if -0.10 < delta_dice < -0.01:
        return "moderate_degradation"
    if delta_dice <= -0.10:
        return "severe_degradation"
    raise ValueError(f"Unable to assign retrospective risk group: {delta_dice}")


def feature_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    values = {
        field: scalar(row[field], field)
        for field in RAW_NUMERIC_FIELDS
        if field in row and row[field] is not None
    }
    values["route_mode_patch_correspondence"] = scalar(
        row["route_mode_patch_correspondence"], "route_mode_patch_correspondence"
    )
    if "trace_area_final" in values:
        values["mask_area_ratio"] = values["trace_area_final"]
    if "trace_area_max_rel_delta" in values:
        values["mask_area_change"] = values["trace_area_max_rel_delta"]
    if "trace_area_min" in values and "trace_area_max" in values:
        values["trace_area_span"] = values["trace_area_max"] - values["trace_area_min"]
    if "q_return" in values and "q_multi" in values:
        values["q_return_minus_q_multi"] = values["q_return"] - values["q_multi"]
        values["abs_q_return_minus_q_multi"] = abs(values["q_return"] - values["q_multi"])
        values["peer_disagreement"] = 1.0 - values["q_multi"]
    if "q_return" in values and "q_model" in values:
        values["q_return_minus_q_model"] = values["q_return"] - values["q_model"]
        values["abs_q_return_minus_q_model"] = abs(values["q_return"] - values["q_model"])
        values["student_disagreement"] = 1.0 - values["q_model"]
    for trace in ("trace_adjacent_dice_min", "trace_adjacent_dice_mean"):
        if "q_return" in values and trace in values:
            values[f"q_return_minus_{trace}"] = values["q_return"] - values[trace]
    return dict(sorted(values.items()))


def diagnostic_features(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    result = dict(candidate)
    for field, value in baseline.items():
        result[f"baseline_{field}"] = value
        if field in candidate:
            result[f"delta_{field}"] = candidate[field] - value
    result["route_mode_changed"] = float(
        candidate["route_mode_patch_correspondence"]
        != baseline["route_mode_patch_correspondence"]
    )
    for name in result:
        if name.startswith("gt_") or "evaluation_only" in name or name == "delta_dice":
            raise ValueError(f"GT-leaking diagnostic feature is forbidden: {name}")
    return dict(sorted(result.items()))


def build_candidate_table(
    candidates: list[dict[str, Any]],
    previous_per_target: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["target_id"]].append(candidate)
    if len(grouped) != 100 or any(len(rows) != 14 for rows in grouped.values()):
        raise ValueError("Expected exactly 100 targets with exactly 14 frozen candidates each")

    all_rows: list[dict[str, Any]] = []
    qreturn_rows: list[dict[str, Any]] = []
    for target_id, rows in sorted(grouped.items()):
        # Both route choices are exclusively unsupervised, using the exact
        # frozen tie-break from the previous mechanism-validation script.
        baseline = max(rows, key=selection_key("path_mean_similarity"))
        qreturn = max(rows, key=selection_key("q_return"))
        prior = previous_per_target[target_id]["selections"]
        for method, selected in (("knn_mean", baseline), ("q_return", qreturn)):
            expected = prior[method]
            if expected["route_id"] != selected["route_id"] or expected["route_mode"] != selected["route_mode"]:
                raise ValueError(f"Frozen {method} selection reproduction failed: {target_id}")
            if not values_equal(expected[GT_FIELD], selected[GT_FIELD]):
                raise ValueError(f"Frozen evaluation-only Dice reproduction failed: {target_id}/{method}")
        baseline_features = feature_snapshot(baseline)
        for candidate in sorted(rows, key=lambda item: (item["route_mode"], item["bridge_count"], item["route_id"])):
            candidate_features = feature_snapshot(candidate)
            delta = scalar(candidate[GT_FIELD], GT_FIELD) - scalar(baseline[GT_FIELD], GT_FIELD)
            is_selected = (
                candidate["route_id"] == qreturn["route_id"]
                and candidate["route_mode"] == qreturn["route_mode"]
            )
            is_physical_switch = candidate["route_id"] != baseline["route_id"]
            record = {
                "target_id": target_id,
                "baseline_route_id": baseline["route_id"],
                "baseline_route_mode": baseline["route_mode"],
                "baseline_bridge_count": baseline["bridge_count"],
                "candidate_route_id": candidate["route_id"],
                "candidate_route_mode": candidate["route_mode"],
                "candidate_bridge_count": candidate["bridge_count"],
                "baseline_gt_dice_evaluation_only": scalar(baseline[GT_FIELD], GT_FIELD),
                "candidate_gt_dice_evaluation_only": scalar(candidate[GT_FIELD], GT_FIELD),
                "delta_dice_evaluation_only": delta,
                "risk_group_evaluation_only": risk_group(delta),
                "severe_risk_evaluation_only": delta <= -0.10,
                "catastrophic_risk_evaluation_only": delta <= -0.20,
                "is_baseline_candidate": (
                    candidate["route_id"] == baseline["route_id"]
                    and candidate["route_mode"] == baseline["route_mode"]
                ),
                "is_physical_switch": is_physical_switch,
                "is_qreturn_selected": is_selected,
                "candidate_unsupervised_features": candidate_features,
                "baseline_unsupervised_features": baseline_features,
                "candidate_minus_baseline_features": {
                    field: candidate_features[field] - value
                    for field, value in baseline_features.items()
                    if field in candidate_features
                },
                "diagnostic_features": diagnostic_features(candidate_features, baseline_features),
            }
            all_rows.append(record)
            if is_selected:
                qreturn_rows.append(
                    {
                        **record,
                        "qreturn_route_id": record["candidate_route_id"],
                        "qreturn_route_mode": record["candidate_route_mode"],
                        "qreturn_bridge_count": record["candidate_bridge_count"],
                        "qreturn_gt_dice": record["candidate_gt_dice_evaluation_only"],
                        "baseline_gt_dice": record["baseline_gt_dice_evaluation_only"],
                        "delta_dice": record["delta_dice_evaluation_only"],
                    }
                )
    if len(all_rows) != 1400 or len(qreturn_rows) != 100:
        raise ValueError(f"Candidate reconstruction mismatch: all={len(all_rows)}, q_return={len(qreturn_rows)}")
    deltas = np.asarray([row["delta_dice"] for row in qreturn_rows], dtype=np.float64)
    tied = np.isclose(deltas, 0.0, rtol=0.0, atol=1e-12)
    reproduction = {
        "target_count": len(qreturn_rows),
        "improved_target_count": int(np.sum((deltas > 0.0) & ~tied)),
        "tied_target_count": int(np.sum(tied)),
        "degraded_target_count": int(np.sum((deltas < 0.0) & ~tied)),
        "mean_delta_dice": float(np.mean(deltas)),
        "median_delta_dice": float(np.median(deltas)),
        "qreturn_selected_dice": float(np.mean([row["qreturn_gt_dice"] for row in qreturn_rows])),
        "knn_mean_selected_dice": float(np.mean([row["baseline_gt_dice"] for row in qreturn_rows])),
        "physical_switch_target_count": sum(row["is_physical_switch"] for row in qreturn_rows),
        "same_route_target_count": sum(not row["is_physical_switch"] for row in qreturn_rows),
    }
    if (
        reproduction["improved_target_count"] != 54
        or reproduction["tied_target_count"] != 18
        or reproduction["degraded_target_count"] != 28
        or not math.isclose(reproduction["mean_delta_dice"], -0.0045359266358780456, abs_tol=1e-12)
    ):
        raise ValueError(f"Frozen q_return failure-set reproduction failed: {reproduction}")
    return all_rows, qreturn_rows, reproduction


def descriptive_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "q25": None, "q75": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def scoped_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for group in ROUTE_GROUPS:
        subset = [row for row in rows if row["risk_group_evaluation_only"] == group]
        result[group] = {
            "label": GROUP_LABELS[group],
            "candidate_count": len(subset),
            "unique_target_count": len({row["target_id"] for row in subset}),
            "delta_dice": descriptive_stats([row["delta_dice_evaluation_only"] for row in subset]),
            "feature_statistics": {
                feature: descriptive_stats(
                    [row["diagnostic_features"][feature] for row in subset if feature in row["diagnostic_features"]]
                )
                for feature in GROUP_SUMMARY_FIELDS
            },
        }
    return {
        "candidate_count": len(rows),
        "unique_target_count": len({row["target_id"] for row in rows}),
        "severe_candidate_count": sum(row["severe_risk_evaluation_only"] for row in rows),
        "severe_unique_target_count": len(
            {row["target_id"] for row in rows if row["severe_risk_evaluation_only"]}
        ),
        "catastrophic_candidate_count": sum(row["catastrophic_risk_evaluation_only"] for row in rows),
        "catastrophic_unique_target_count": len(
            {row["target_id"] for row in rows if row["catastrophic_risk_evaluation_only"]}
        ),
        "groups": result,
    }


def quartile_interaction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray([row["diagnostic_features"]["q_return"] for row in rows])
    multis = np.asarray([row["diagnostic_features"]["q_multi"] for row in rows])
    return_high = float(np.quantile(returns, 0.75))
    multi_low = float(np.quantile(multis, 0.25))
    multi_high = float(np.quantile(multis, 0.75))
    result = {}
    for name, predicate in (
        ("high_return_low_peer_agreement", lambda row: row["diagnostic_features"]["q_return"] >= return_high and row["diagnostic_features"]["q_multi"] <= multi_low),
        ("high_return_high_peer_agreement", lambda row: row["diagnostic_features"]["q_return"] >= return_high and row["diagnostic_features"]["q_multi"] >= multi_high),
    ):
        selected = [row for row in rows if predicate(row)]
        severe = sum(row["severe_risk_evaluation_only"] for row in selected)
        result[name] = {
            "candidate_count": len(selected),
            "severe_candidate_count": severe,
            "severe_fraction": severe / len(selected) if selected else None,
            "mean_delta_dice": float(np.mean([row["delta_dice_evaluation_only"] for row in selected])) if selected else None,
        }
    return {
        "descriptive_only_no_deployment_threshold": True,
        "q_return_upper_quartile": return_high,
        "q_multi_lower_quartile": multi_low,
        "q_multi_upper_quartile": multi_high,
        "regions": result,
    }


def continuous_effect(values_a: np.ndarray, values_b: np.ndarray) -> float:
    differences = values_a[:, None] - values_b[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    common = set(rows[0]["diagnostic_features"])
    for row in rows[1:]:
        common.intersection_update(row["diagnostic_features"])
    return sorted(common)


def feature_effects(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    beneficial = [row for row in rows if row["risk_group_evaluation_only"] == "beneficial"]
    dangerous = [row for row in rows if row["severe_risk_evaluation_only"]]
    output = []
    if not beneficial or not dangerous:
        return output
    for field in feature_names(rows):
        a = np.asarray([row["diagnostic_features"][field] for row in beneficial])
        b = np.asarray([row["diagnostic_features"][field] for row in dangerous])
        try:
            _, p_value = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            p_value = 1.0
        output.append(
            {
                "scope": scope,
                "feature": field,
                "beneficial_n": len(a),
                "dangerous_n": len(b),
                "beneficial_mean": float(np.mean(a)),
                "dangerous_mean": float(np.mean(b)),
                "difference_dangerous_minus_beneficial": float(np.mean(b) - np.mean(a)),
                "mann_whitney_u_p_value": float(p_value),
                "cliffs_delta_beneficial_minus_dangerous": continuous_effect(a, b),
            }
        )
    output.sort(key=lambda item: (item["mann_whitney_u_p_value"], -abs(item["cliffs_delta_beneficial_minus_dangerous"]), item["feature"]))
    return output


def score_auc(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if len(np.unique(labels)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def single_feature_auc(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    labels = np.asarray([int(row["severe_risk_evaluation_only"]) for row in rows])
    results = []
    if len(np.unique(labels)) < 2:
        return results
    for field in feature_names(rows):
        values = np.asarray([row["diagnostic_features"][field] for row in rows], dtype=np.float64)
        higher = score_auc(labels, values)
        lower = score_auc(labels, -values)
        if higher["roc_auc"] >= lower["roc_auc"]:
            orientation, chosen = "higher_values_indicate_risk", higher
        else:
            orientation, chosen = "lower_values_indicate_risk", lower
        results.append(
            {
                "scope": scope,
                "feature": field,
                "n": len(rows),
                "positive_severe_count": int(np.sum(labels)),
                "risk_prevalence": float(np.mean(labels)),
                "risk_direction_descriptive_only": orientation,
                "roc_auc": chosen["roc_auc"],
                "pr_auc": chosen["pr_auc"],
                "roc_auc_higher_values_risk": higher["roc_auc"],
                "pr_auc_higher_values_risk": higher["pr_auc"],
                "roc_auc_lower_values_risk": lower["roc_auc"],
                "pr_auc_lower_values_risk": lower["pr_auc"],
            }
        )
    results.sort(key=lambda item: (-item["roc_auc"], -item["pr_auc"], item["feature"]))
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot produce empty CSV diagnostic: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def risk_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = np.asarray([row["delta_dice_evaluation_only"] for row in rows], dtype=np.float64)
    tied = np.isclose(deltas, 0.0, rtol=0.0, atol=1e-12)
    return {
        "candidate_count": len(rows),
        "unique_target_count": len({row["target_id"] for row in rows}),
        "mean_delta_dice": float(np.mean(deltas)),
        "median_delta_dice": float(np.median(deltas)),
        "worst_delta_dice": float(np.min(deltas)),
        "improved_count": int(np.sum((deltas > 0) & ~tied)),
        "tied_count": int(np.sum(tied)),
        "degraded_count": int(np.sum((deltas < 0) & ~tied)),
        "severe_count": int(np.sum(deltas <= -0.10)),
        "severe_rate": float(np.mean(deltas <= -0.10)),
        "catastrophic_count": int(np.sum(deltas <= -0.20)),
        "catastrophic_rate": float(np.mean(deltas <= -0.20)),
        "feature_means": {
            field: float(np.mean([row["diagnostic_features"][field] for row in rows]))
            for field in (
                "q_return",
                "q_multi",
                "q_model",
                "b7",
                "path_mean_similarity",
                "trace_adjacent_dice_min",
                "trace_adjacent_dice_mean",
                "q_return_minus_q_model",
                "q_return_minus_q_multi",
                "delta_q_multi",
                "delta_q_model",
            )
            if all(field in row["diagnostic_features"] for row in rows)
        },
    }


def group_risk(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {key: risk_bucket(value) for key, value in sorted(grouped.items())}


def model_feature_list(rows: list[dict[str, Any]]) -> list[str]:
    available = set(feature_names(rows))
    ordered: list[str] = []
    for field in CORE_FEATURES:
        if field in available:
            ordered.append(field)
        delta = f"delta_{field}"
        if delta in available and field != "route_mode_patch_correspondence":
            ordered.append(delta)
    for field in DERIVED_DISAGREEMENTS:
        if field in available:
            ordered.append(field)
        delta = f"delta_{field}"
        if delta in available:
            ordered.append(delta)
    if "route_mode_changed" in available:
        ordered.append("route_mode_changed")
    return list(dict.fromkeys(ordered))


def row_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["target_id"], row["candidate_route_mode"], row["candidate_route_id"]


def grouped_cv(
    rows: list[dict[str, Any]],
    method: str,
    seed: int,
) -> tuple[dict[str, Any], dict[tuple[str, str, str], float]]:
    fields = model_feature_list(rows)
    features = np.asarray(
        [[row["diagnostic_features"][field] for field in fields] for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([int(row["severe_risk_evaluation_only"]) for row in rows], dtype=np.int64)
    groups = np.asarray([row["target_id"] for row in rows], dtype=object)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    out_of_fold = np.full(len(rows), np.nan, dtype=np.float64)
    fold_results = []

    def make_model() -> Any:
        if method == "logistic":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                ),
            )
        if method == "tree":
            return DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=seed,
            )
        raise ValueError(f"Unknown diagnostic-only classifier: {method}")

    for fold, (train, validation) in enumerate(splitter.split(features, labels, groups), 1):
        train_targets = set(groups[train])
        validation_targets = set(groups[validation])
        overlap = train_targets & validation_targets
        if overlap:
            raise RuntimeError(f"Target-level CV leakage in fold {fold}: {sorted(overlap)}")
        if len(np.unique(labels[train])) < 2:
            raise RuntimeError(f"Fold {fold} training has only one risk class")
        model = make_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(features[train], labels[train])
        probabilities = model.predict_proba(features[validation])[:, 1]
        out_of_fold[validation] = probabilities
        fold_results.append(
            {
                "fold": fold,
                "train_candidate_count": int(len(train)),
                "validation_candidate_count": int(len(validation)),
                "train_target_count": len(train_targets),
                "validation_target_count": len(validation_targets),
                "target_overlap_count": 0,
                "validation_severe_count": int(np.sum(labels[validation])),
                "validation_severe_target_count": len(
                    {groups[index] for index in validation if labels[index]}
                ),
                **score_auc(labels[validation], probabilities),
            }
        )
    if np.isnan(out_of_fold).any():
        raise RuntimeError("Some switch candidates lack target-disjoint out-of-fold predictions")
    selected = np.asarray([row["is_qreturn_selected"] for row in rows], dtype=bool)
    selected_labels = labels[selected]
    selected_scores = out_of_fold[selected]
    fitted = make_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.fit(features, labels)
    if method == "logistic":
        coefficients = fitted.named_steps["logisticregression"].coef_[0]
        interpretation = {
            "standardized_feature_coefficients_full_validation_diagnostic_only": dict(
                sorted(
                    ((field, float(value)) for field, value in zip(fields, coefficients)),
                    key=lambda item: (-abs(item[1]), item[0]),
                )
            ),
            "intercept_full_validation_diagnostic_only": float(
                fitted.named_steps["logisticregression"].intercept_[0]
            ),
        }
    else:
        interpretation = {
            "full_validation_tree_rules_diagnostic_only": export_text(
                fitted, feature_names=fields, decimals=5
            ),
            "feature_importances_full_validation_diagnostic_only": dict(
                sorted(
                    ((field, float(value)) for field, value in zip(fields, fitted.feature_importances_) if value > 0),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        }
    result = {
        "method": method,
        "purpose": "validation-only diagnostic separability; never deployed or evaluated on test",
        "cross_validation": "target-disjoint StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)",
        "feature_count": len(fields),
        "features": fields,
        "candidate_count": len(rows),
        "target_count": len(set(groups)),
        "severe_candidate_count": int(np.sum(labels)),
        "severe_target_count": len({group for group, label in zip(groups, labels) if label}),
        "severe_prevalence": float(np.mean(labels)),
        "out_of_fold_all_switch_candidates": score_auc(labels, out_of_fold),
        "out_of_fold_qreturn_switched_candidates": {
            "candidate_count": int(np.sum(selected)),
            "severe_candidate_count": int(np.sum(selected_labels)),
            "severe_prevalence": float(np.mean(selected_labels)) if len(selected_labels) else None,
            **score_auc(selected_labels, selected_scores),
        },
        "folds": fold_results,
        "hyperparameters_frozen_no_search": (
            {"C": 1.0, "class_weight": "balanced", "max_iter": 3000}
            if method == "logistic"
            else {"max_depth": 3, "min_samples_leaf": 20, "class_weight": "balanced"}
        ),
        **interpretation,
    }
    predictions = {row_identity(row): float(value) for row, value in zip(rows, out_of_fold)}
    return result, predictions


def compare_feature_auc(rows: list[dict[str, Any]], scope: str, field: str) -> dict[str, Any] | None:
    return next((row for row in rows if row["scope"] == scope and row["feature"] == field), None)


def diagnose_case(
    row: dict[str, Any],
    logistic_predictions: dict[tuple[str, str, str], float],
    tree_predictions: dict[tuple[str, str, str], float],
    competitive_logistic_predictions: dict[tuple[str, str, str], float] | None = None,
    competitive_tree_predictions: dict[tuple[str, str, str], float] | None = None,
) -> dict[str, Any]:
    features = row["diagnostic_features"]
    differences = row["candidate_minus_baseline_features"]
    observations = []
    for field, text in (
        ("q_multi", "候选间一致性 q_multi"),
        ("q_model", "Student 一致性 q_model"),
        ("b7", "历史 B7"),
        ("trace_adjacent_dice_min", "局部相邻帧最小 Dice"),
        ("trace_adjacent_dice_mean", "局部相邻帧平均 Dice"),
        ("forward_sam_score", "前向 SAM score"),
    ):
        if field not in differences:
            continue
        delta = differences[field]
        if delta < -0.01:
            observations.append(f"{text} 相对安全基准下降 {delta:.6f}")
        elif delta > 0.01 and field in ("q_multi", "q_model", "b7"):
            observations.append(f"{text} 反而相对安全基准提高 {delta:.6f}")
    if "trace_area_max_rel_delta" in differences and differences["trace_area_max_rel_delta"] > 0.10:
        observations.append(
            f"传播面积最大相对跳变增加 {differences['trace_area_max_rel_delta']:.6f}"
        )
    if "trace_centroid_max_step" in differences and differences["trace_centroid_max_step"] > 0.05:
        observations.append(
            f"传播质心最大位移增加 {differences['trace_centroid_max_step']:.6f}"
        )
    if not observations:
        observations.append("现有主要无监督字段没有相对基准呈现明确异常，存在共同失效可能")
    key = row_identity(row)
    return {
        **row,
        "diagnostic_observations": observations,
        "target_disjoint_oof_logistic_risk_probability": logistic_predictions.get(key),
        "target_disjoint_oof_tree_risk_probability": tree_predictions.get(key),
        "competitive_target_disjoint_oof_logistic_risk_probability": (
            competitive_logistic_predictions.get(key)
            if competitive_logistic_predictions is not None
            else None
        ),
        "competitive_target_disjoint_oof_tree_risk_probability": (
            competitive_tree_predictions.get(key)
            if competitive_tree_predictions is not None
            else None
        ),
        "high_return_self_consistency": features["q_return"],
    }


def n(value: Any, digits: int = 6) -> str:
    return "未定义" if value is None else f"{float(value):.{digits}f}"


def feature_auc_table(aucs: list[dict[str, Any]], scope: str, limit: int = 15) -> list[str]:
    rows = [row for row in aucs if row["scope"] == scope][:limit]
    lines = [
        "| 风险特征 | 风险方向（仅描述） | ROC-AUC | PR-AUC | 严重样本 / N |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['feature']}` | `{row['risk_direction_descriptive_only']}` | "
            f"{n(row['roc_auc'])} | {n(row['pr_auc'])} | {row['positive_severe_count']} / {row['n']} |"
        )
    return lines


def group_feature_table(summary: dict[str, Any], features: tuple[str, ...]) -> list[str]:
    lines = ["| 组别 | N | " + " | ".join(f"{field} mean / median" for field in features) + " |"]
    lines.append("|---|---:|" + "---:|" * len(features))
    for group in ROUTE_GROUPS:
        entry = summary["groups"][group]
        cells = []
        for field in features:
            feature = entry["feature_statistics"][field]
            cells.append(f"{n(feature['mean'])} / {n(feature['median'])}")
        lines.append(f"| {GROUP_LABELS[group]} | {entry['candidate_count']} | " + " | ".join(cells) + " |")
    return lines


def depth_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| bridge | N | 平均 Δ | 中位 Δ | 严重退化 | 灾难退化 | 最差 Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bridge, entry in summary.items():
        lines.append(
            f"| b{bridge} | {entry['candidate_count']} | {n(entry['mean_delta_dice'])} | "
            f"{n(entry['median_delta_dice'])} | {entry['severe_count']} "
            f"({n(entry['severe_rate'], 3)}) | {entry['catastrophic_count']} "
            f"({n(entry['catastrophic_rate'], 3)}) | {n(entry['worst_delta_dice'])} |"
        )
    return lines


def mode_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| mode | switches | 改善 / 持平 / 退化 | 严重 | 平均 Δ | 最差 Δ | q_multi | q_model | 局部最小 Dice |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, entry in summary.items():
        means = entry["feature_means"]
        lines.append(
            f"| `{mode}` | {entry['candidate_count']} | {entry['improved_count']} / "
            f"{entry['tied_count']} / {entry['degraded_count']} | {entry['severe_count']} | "
            f"{n(entry['mean_delta_dice'])} | {n(entry['worst_delta_dice'])} | "
            f"{n(means.get('q_multi'))} | {n(means.get('q_model'))} | "
            f"{n(means.get('trace_adjacent_dice_min'))} |"
        )
    return lines


def severe_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| target | ΔDice | q_return | q_multi | q_model | B7 | mode | bridge |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["delta_dice_evaluation_only"]):
        values = row["diagnostic_features"]
        lines.append(
            f"| `{row['target_id']}` | {n(row['delta_dice_evaluation_only'])} | "
            f"{n(values['q_return'])} | {n(values['q_multi'])} | {n(values['q_model'])} | "
            f"{n(values['b7'])} | `{row['candidate_route_mode']}` | {row['candidate_bridge_count']} |"
        )
    return lines


def mechanism_findings(
    qreturn_summary: dict[str, Any],
    aucs: list[dict[str, Any]],
    logistic: dict[str, Any],
    tree: dict[str, Any],
    competitive_logistic: dict[str, Any],
    competitive_tree: dict[str, Any],
    interaction: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    scope = "all_physical_switch_candidates"
    qmulti = compare_feature_auc(aucs, scope, "q_multi")
    qmodel = compare_feature_auc(aucs, scope, "q_model")
    trajectory = compare_feature_auc(aucs, scope, "trace_adjacent_dice_min")
    delta_multi = compare_feature_auc(aucs, scope, "delta_q_multi")
    delta_model = compare_feature_auc(aucs, scope, "delta_q_model")
    return_multi = compare_feature_auc(aucs, scope, "q_return_minus_q_multi")
    return_model = compare_feature_auc(aucs, scope, "q_return_minus_q_model")
    best_single = next(row for row in aucs if row["scope"] == scope)
    competitive_scope = "qreturn_competitive_physical_switch_candidates"
    competitive_qmulti = compare_feature_auc(aucs, competitive_scope, "q_multi")
    competitive_qmodel = compare_feature_auc(aucs, competitive_scope, "q_model")
    competitive_trajectory = compare_feature_auc(
        aucs, competitive_scope, "trace_adjacent_dice_min"
    )
    competitive_best_single = next(row for row in aucs if row["scope"] == competitive_scope)
    logistic_auc = logistic["out_of_fold_all_switch_candidates"]["roc_auc"]
    tree_auc = tree["out_of_fold_all_switch_candidates"]["roc_auc"]
    recommendation = (
        "mechanistic_signal_justifies_one_preregistered_validation_only_counterfactual; "
        "current_models_are_not_validated_safe_deployment_gates"
    )
    case_signal_counts = {
        "qmulti_lower_than_baseline": sum(case["candidate_minus_baseline_features"].get("q_multi", 0.0) < -0.01 for case in cases),
        "qmodel_lower_than_baseline": sum(case["candidate_minus_baseline_features"].get("q_model", 0.0) < -0.01 for case in cases),
        "local_trace_min_lower_than_baseline": sum(case["candidate_minus_baseline_features"].get("trace_adjacent_dice_min", 0.0) < -0.01 for case in cases),
    }
    return {
        "qreturn_selected_severe_count": qreturn_summary["severe_candidate_count"],
        "qreturn_selected_catastrophic_count": qreturn_summary["catastrophic_candidate_count"],
        "all_switch_q_multi_auc": qmulti,
        "all_switch_q_model_auc": qmodel,
        "all_switch_trace_adjacent_min_auc": trajectory,
        "all_switch_delta_q_multi_auc": delta_multi,
        "all_switch_delta_q_model_auc": delta_model,
        "all_switch_return_minus_multi_auc": return_multi,
        "all_switch_return_minus_model_auc": return_model,
        "best_single_feature_descriptive": best_single,
        "competitive_q_multi_auc": competitive_qmulti,
        "competitive_q_model_auc": competitive_qmodel,
        "competitive_trace_adjacent_min_auc": competitive_trajectory,
        "competitive_best_single_feature_descriptive": competitive_best_single,
        "qreturn_qmulti_interaction": interaction,
        "catastrophic_selected_signal_counts": case_signal_counts,
        "logistic_grouped_cv_roc_auc": logistic_auc,
        "tree_grouped_cv_roc_auc": tree_auc,
        "competitive_logistic_grouped_cv": competitive_logistic[
            "out_of_fold_all_switch_candidates"
        ],
        "competitive_tree_grouped_cv": competitive_tree[
            "out_of_fold_all_switch_candidates"
        ],
        "competitive_severe_target_count": competitive_logistic["severe_target_count"],
        "actual_qreturn_catastrophic_target_count": len(cases),
        "recommendation": recommendation,
        "no_cv_score_or_gt_threshold_used_as_deployment_rule": True,
    }


def case_detail(case: dict[str, Any], index: int) -> list[str]:
    lines = [
        f"### 灾难案例 {index}：`{case['target_id']}`",
        "",
        f"- 基准：`{case['baseline_route_mode']}` / b{case['baseline_bridge_count']} / "
        f"`{case['baseline_route_id']}`；Dice={n(case['baseline_gt_dice_evaluation_only'])}。",
        f"- q_return 候选：`{case['candidate_route_mode']}` / b{case['candidate_bridge_count']} / "
        f"`{case['candidate_route_id']}`；Dice={n(case['candidate_gt_dice_evaluation_only'])}，"
        f"ΔDice={n(case['delta_dice_evaluation_only'])}。",
        "",
        "| 无监督字段 | 安全基准 | q_return 候选 | 候选 - 基准 |",
        "|---|---:|---:|---:|",
    ]
    for field in (
        "q_return",
        "q_multi",
        "q_model",
        "b7",
        "path_mean_similarity",
        "path_bottleneck_similarity",
        "forward_sam_score",
        "trace_adjacent_dice_min",
        "trace_adjacent_dice_mean",
        "trace_adjacent_dice_last",
        "trace_sam_score_min",
        "trace_sam_score_mean",
        "trace_area_final",
        "trace_area_max_rel_delta",
        "trace_centroid_max_step",
        "q_return_minus_q_multi",
        "q_return_minus_q_model",
    ):
        baseline = case["baseline_unsupervised_features"].get(field)
        candidate = case["candidate_unsupervised_features"].get(field)
        delta = case["candidate_minus_baseline_features"].get(field)
        if baseline is not None and candidate is not None:
            lines.append(f"| `{field}` | {n(baseline)} | {n(candidate)} | {n(delta)} |")
    lines.extend(
        [
            "",
            f"- 五折 target-disjoint OOF logistic 风险概率："
            f"{n(case['target_disjoint_oof_logistic_risk_probability'])}；"
            f"OOF shallow tree 风险概率：{n(case['target_disjoint_oof_tree_risk_probability'])}。",
            f"- 决策相关 competitive cohort 的 OOF logistic 风险概率："
            f"{n(case['competitive_target_disjoint_oof_logistic_risk_probability'])}；"
            f"OOF shallow tree 风险概率："
            f"{n(case['competitive_target_disjoint_oof_tree_risk_probability'])}。",
            "- 逐例机制诊断：" + "；".join(case["diagnostic_observations"]) + "。",
            "",
        ]
    )
    return lines


def report(
    path: Path,
    repo_root: Path,
    audit: dict[str, Any],
    reproduction: dict[str, Any],
    risk_summary: dict[str, Any],
    effects: list[dict[str, Any]],
    aucs: list[dict[str, Any]],
    bridge: dict[str, Any],
    mode: dict[str, Any],
    logistic: dict[str, Any],
    tree: dict[str, Any],
    competitive_logistic: dict[str, Any],
    competitive_tree: dict[str, Any],
    cases: list[dict[str, Any]],
    findings: dict[str, Any],
) -> None:
    all_summary = risk_summary["all_physical_switch_candidates"]
    selected_summary = risk_summary["qreturn_selected_all_targets"]
    switched_summary = risk_summary["qreturn_selected_physical_switches"]
    competitive_summary = risk_summary["qreturn_competitive_physical_switch_candidates"]
    high_return_summary = risk_summary["high_return_physical_switch_candidates"]
    interaction = risk_summary["q_return_q_multi_quartile_interaction"]
    qmulti = findings["all_switch_q_multi_auc"]
    qmodel = findings["all_switch_q_model_auc"]
    trajectory = findings["all_switch_trace_adjacent_min_auc"]
    best = findings["best_single_feature_descriptive"]
    lines = [
        "# C0-256 传播风险诊断：定位“高一致性但错误传播”的失败机制",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        f"> 仓库：`{repo_root}`  ",
        "> 范围：仅 frozen validation、仅 CPU 离线诊断；不读取 test、不运行 propagation、不修改 B7。",
        "",
        "## 1. 背景与上一轮冻结结论",
        "",
        "此前 q_return 的 candidate-level Spearman 0.216292 高于 KNN mean 的 0.043077，"
        "但 q_return Top-1 Dice 0.883055 低于冻结 KNN mean 基准 0.887591。"
        "因此本轮不再测试 q_return 直接替代 KNN，而诊断高一致性错误传播能否被其他无监督信号识别。",
        "",
        "## 2. 数据、输入 SHA256 与冻结协议",
        "",
        "| 输入角色 | validation 文件 | SHA256 |",
        "|---|---|---|",
    ]
    for item in audit["input_artifacts"]:
        lines.append(f"| `{item['role']}` | `{item['path']}` | `{item['sha256']}` |")
    lines.extend(
        [
            "",
            f"100 targets × 14 冻结候选 = {audit['candidate_count']} candidates。"
            "安全基准固定为上一轮相同 tie-break 的 `argmax(path_mean_similarity)`，绝不重新选择 KNN baseline。",
            "",
            f"可用数值字段：`{json.dumps(audit['fully_available_numeric_fields'], ensure_ascii=False)}`。",
            f"可选面积字段状态：`{json.dumps(audit['requested_optional_field_status'], ensure_ascii=False, sort_keys=True)}`。",
            "`mask_area_ratio` 仅由已有 `trace_area_final` 派生；`mask_area_change` 仅由已有 "
            "`trace_area_max_rel_delta` 派生；返回 mask 面积未保存，因此不臆造也不重跑 GPU。",
            f"所有 `gt_*` 原始字段都排除在模型/风险特征之外：`{json.dumps(audit['raw_gt_prefixed_fields_excluded_from_features'], ensure_ascii=False)}`。",
            "",
            "## 3. 评价性风险定义",
            "",
            "- 明确改善：`ΔDice > 0.01`。",
            "- 基本安全：`-0.01 ≤ ΔDice ≤ 0.01`。",
            "- 中度退化：`-0.10 < ΔDice < -0.01`。",
            "- 严重退化：`ΔDice ≤ -0.10`；灾难性退化：`ΔDice ≤ -0.20`。",
            "- 这些 GT 阈值仅用于回顾性标签与机制评价，不是部署阈值，也不会访问 test。",
            "",
            "## 4. q_return 失败集合严格复现",
            "",
            f"- 改善 / 持平 / 退化：{reproduction['improved_target_count']} / "
            f"{reproduction['tied_target_count']} / {reproduction['degraded_target_count']}。",
            f"- 平均 ΔDice：{n(reproduction['mean_delta_dice'])}；"
            f"中位 ΔDice：{n(reproduction['median_delta_dice'])}。",
            f"- 真实 physical route switch：{reproduction['physical_switch_target_count']}；"
            f"相同 route：{reproduction['same_route_target_count']}。",
            f"- q_return 所选 target 中严重退化：{selected_summary['severe_candidate_count']}；"
            f"灾难性退化：{selected_summary['catastrophic_candidate_count']}。",
            f"- 全部非基准 physical switch candidates：{all_summary['candidate_count']}；"
            f"严重 {all_summary['severe_candidate_count']}，灾难 {all_summary['catastrophic_candidate_count']}。",
            f"- 决策相关 competitive 子集 `q_return(candidate) >= q_return(baseline)`："
            f"{competitive_summary['candidate_count']} 候选 / {competitive_summary['unique_target_count']} targets；"
            f"严重 {competitive_summary['severe_candidate_count']} 条，分布在 "
            f"{competitive_summary['severe_unique_target_count']} 个独立 target；"
            f"灾难 {competitive_summary['catastrophic_candidate_count']} 条。",
            f"- 高 q_return 上四分位 physical switches：{high_return_summary['candidate_count']}；"
            f"严重 {high_return_summary['severe_candidate_count']}。",
            "",
            "## 5. 四组 q_multi / q_model / B7 / q_return 分布",
            "",
            "### 所有候选换路",
            "",
            *group_feature_table(all_summary, ("q_return", "q_multi", "q_model", "b7")),
            "",
            "### q_return 实际切换",
            "",
            *group_feature_table(switched_summary, ("q_return", "q_multi", "q_model", "b7")),
            "",
            "### 决策相关 competitive 候选",
            "",
            *group_feature_table(competitive_summary, ("q_return", "q_multi", "q_model", "b7")),
            "",
            "完整 mean / median / std / Q25 / Q75 / min / max 和相对 baseline 变化见 `risk_group_summary.json`。",
            "",
            "## 6. q_return 高但 q_multi 低：无监督四分位二维诊断",
            "",
            f"所有阈值只是 candidate 特征自身的无监督四分位，不看 GT，也不会用作部署规则："
            f"q_return Q75={n(interaction['q_return_upper_quartile'])}，"
            f"q_multi Q25={n(interaction['q_multi_lower_quartile'])}，"
            f"q_multi Q75={n(interaction['q_multi_upper_quartile'])}。",
            "",
            "| 描述区域 | N | 严重退化 | 严重比例 | 平均 ΔDice |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, region in interaction["regions"].items():
        lines.append(
            f"| `{name}` | {region['candidate_count']} | {region['severe_candidate_count']} | "
            f"{n(region['severe_fraction'])} | {n(region['mean_delta_dice'])} |"
        )
    lines.extend(
        [
            "",
            "## 7. Student disagreement 与历史 B7 的互补性",
            "",
            f"- `q_multi` 单变量描述 ROC-AUC={n(qmulti['roc_auc'])}，PR-AUC={n(qmulti['pr_auc'])}。",
            f"- `q_model` 单变量描述 ROC-AUC={n(qmodel['roc_auc'])}，PR-AUC={n(qmodel['pr_auc'])}。",
            f"- `q_return-q_multi` ROC-AUC="
            f"{n(findings['all_switch_return_minus_multi_auc']['roc_auc'])}；"
            f"`q_return-q_model` ROC-AUC="
            f"{n(findings['all_switch_return_minus_model_auc']['roc_auc'])}。",
            f"- `Δq_multi` ROC-AUC={n(findings['all_switch_delta_q_multi_auc']['roc_auc'])}；"
            f"`Δq_model` ROC-AUC={n(findings['all_switch_delta_q_model_auc']['roc_auc'])}。",
            f"- 去除对 q_return 直接无竞争力的候选后：competitive `q_multi` ROC-AUC="
            f"{n(findings['competitive_q_multi_auc']['roc_auc'])}，competitive `q_model` ROC-AUC="
            f"{n(findings['competitive_q_model_auc']['roc_auc'])}。",
            "B7 全部来自上一轮冻结 audit，不重算、调指数或选择新权重；其失败和成功均只做事后机制解释。",
            "",
            "## 8. 局部传播轨迹与形态漂移",
            "",
            f"`trace_adjacent_dice_min` 描述 ROC-AUC={n(trajectory['roc_auc'])}，"
            f"PR-AUC={n(trajectory['pr_auc'])}。同时审计 `trace_adjacent_dice_mean`、"
            "面积跳变、bbox 跳变、质心位移、空 mask 和 SAM score；不存在的 returned-mask 面积直接标为 missing。",
            "",
            "## 9. bridge depth 风险：q_return 实际选择",
            "",
            *depth_table(bridge["qreturn_selected_all_targets"]),
            "",
            "不依据任何 depth 结果删除 b0/b6 或缩小候选空间。",
            "",
            "## 10. route mode 风险：q_return 实际 physical switch",
            "",
            *mode_table(mode["qreturn_selected_physical_switches"]),
            "",
            "## 11. 单一无监督特征风险判别",
            "",
            f"全部 physical switch candidates 的严重退化先验比例="
            f"{n(all_summary['severe_candidate_count'] / all_summary['candidate_count'])}。"
            "方向按 validation 描述性比较展示，不构成冻结阈值或 test 方法。",
            "",
            *feature_auc_table(aucs, "all_physical_switch_candidates"),
            "",
            "### q_return 至少不低于基准：决策相关候选",
            "",
            *feature_auc_table(aucs, "qreturn_competitive_physical_switch_candidates", limit=12),
            "",
            "该子集排除了显然无法赢过 q_return 基准的低一致性候选，避免大量空 mask / "
            "q_return≈0 的易分类失败人为抬高对真正错误自洽问题的判断。",
            "",
            "### 仅 q_return 实际切换的单变量诊断",
            "",
            *feature_auc_table(aucs, "qreturn_selected_physical_switches", limit=10),
            "",
            "实际 q_return severe 样本极少，上述子集 AUC 仅用于案例定位，不能作为稳定泛化证据。",
            "",
            "## 12. 改善 vs 严重退化：效应量",
            "",
            "| 特征 | beneficial mean | dangerous mean | dangerous - beneficial | Mann-Whitney p | Cliff's δ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [item for item in effects if item["scope"] == "all_physical_switch_candidates"][:15]:
        lines.append(
            f"| `{row['feature']}` | {n(row['beneficial_mean'])} | {n(row['dangerous_mean'])} | "
            f"{n(row['difference_dangerous_minus_beneficial'])} | "
            f"{row['mann_whitney_u_p_value']:.3e} | "
            f"{n(row['cliffs_delta_beneficial_minus_dangerous'])} |"
        )
    lines.extend(
        [
            "",
            "## 13. 联合信号：target-level 五折交叉验证",
            "",
            "同一 target 的全部候选始终留在同一个 fold；特征只包含 candidate/base "
            "无监督分数、轨迹、bridge、mode 及相对变化。固定超参数，不训练神经网络，不接触 test。",
            "",
            "| 训练/评估 cohort 与 CPU 诊断模型 | cohort ROC-AUC | cohort PR-AUC | q_return 实际 switch ROC-AUC | q_return 实际 switch PR-AUC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cohort_name, result in (
        ("all switches", logistic),
        ("all switches", tree),
        ("q_return-competitive", competitive_logistic),
        ("q_return-competitive", competitive_tree),
    ):
        overall = result["out_of_fold_all_switch_candidates"]
        selected = result["out_of_fold_qreturn_switched_candidates"]
        lines.append(
            f"| `{cohort_name} / {result['method']}` | {n(overall['roc_auc'])} | {n(overall['pr_auc'])} | "
            f"{n(selected['roc_auc'])} | {n(selected['pr_auc'])} |"
        )
    lines.extend(
        [
            "",
            f"competitive cohort 仅有 {competitive_logistic['severe_target_count']} 个独立严重 target；"
            f"q_return OOF 子集只有 {logistic['out_of_fold_qreturn_switched_candidates']['severe_candidate_count']} "
            "个严重 target。candidate 数量不能冒充独立 target 样本量；任何 OOF 结论均需谨慎。"
            "完整系数、树规则、fold target overlap=0 与全部 OOF 指标保存在相应 `risk_cv_*.json`。",
            "",
            "## 14. q_return 所选严重失败清单",
            "",
            *severe_table([row for row in cases if row["severe_risk_evaluation_only"]]),
            "",
            "## 15. 灾难性失败逐例机制诊断",
            "",
        ]
    )
    for index, case in enumerate(cases, 1):
        lines.extend(case_detail(case, index))
    interaction_low = interaction["regions"]["high_return_low_peer_agreement"]
    interaction_high = interaction["regions"]["high_return_high_peer_agreement"]
    lines.extend(
        [
            "## 16. 六个核心问题与主要机制发现",
            "",
            f"1. 高 q_return + 低 q_multi：低-peer 区严重率 "
            f"{n(interaction_low['severe_fraction'])}，高-peer 区 "
            f"{n(interaction_high['severe_fraction'])}；q_multi ROC-AUC="
            f"{n(qmulti['roc_auc'])}。",
            f"2. Student disagreement：q_model ROC-AUC={n(qmodel['roc_auc'])}，"
            f"Δq_model ROC-AUC={n(findings['all_switch_delta_q_model_auc']['roc_auc'])}；"
            f"灾难案例中 q_model 相对基准明显下降 "
            f"{findings['catastrophic_selected_signal_counts']['qmodel_lower_than_baseline']}/{len(cases)}。",
            f"3. 局部轨迹：trace 最小 adjacent Dice ROC-AUC={n(trajectory['roc_auc'])}；"
            f"灾难案例中局部最小值相对基准明显下降 "
            f"{findings['catastrophic_selected_signal_counts']['local_trace_min_lower_than_baseline']}/{len(cases)}。",
            "4. bridge/mode 集中情况见前述固定空间分层表；任何观察都不删除 mode/depth。",
            f"5. 现有信号联合：target-disjoint logistic ROC-AUC="
            f"{n(logistic['out_of_fold_all_switch_candidates']['roc_auc'])}，"
            f"tree ROC-AUC={n(tree['out_of_fold_all_switch_candidates']['roc_auc'])}；"
            f"competitive cohort logistic/tree ROC-AUC="
            f"{n(competitive_logistic['out_of_fold_all_switch_candidates']['roc_auc'])}/"
            f"{n(competitive_tree['out_of_fold_all_switch_candidates']['roc_auc'])}。",
            f"6. 描述性最佳单指标：`{best['feature']}`，ROC-AUC={n(best['roc_auc'])}，"
            f"PR-AUC={n(best['pr_auc'])}；这个选择存在 validation 多重比较乐观偏差，"
            "不是可部署规则。",
            "",
            "## 17. 是否值得进入风险感知路线切换阶段",
            "",
            f"诊断结论：`{findings['recommendation']}`。"
            "可以论证开展一次严格预注册的 validation-only 反事实实验，但不能宣称当前 "
            "logistic/tree 已能安全拦截实际 q_return 灾难案例；必须同时检查 competitive cohort "
            "OOF 与逐例失败，而且实际 catastrophic 只有两个独立 target。",
            "",
            "## 18. 下一步唯一建议与停止边界",
            "",
            "唯一建议：如另行授权，预注册一次 **冻结 KNN mean 安全基准 + 无监督多信号风险门控的 "
            "target-disjoint validation-only 路线切换反事实评估**，明确比较 selected Dice、负尾部、"
            "paired bootstrap 与灾难率；在协议冻结前不查看 test。",
            "",
            "本轮不部署任何阈值/模型，不修改 B7，不运行 GPU/SAM3 propagation，不训练 SAM3/Student/learned router，"
            "不读取 test，不自动进入下一轮。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    previous = root / "work/rerun_c0_256_propagation_aware_routing"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--candidates", type=Path, default=previous / "validation_candidates.jsonl")
    parser.add_argument("--per-target", type=Path, default=previous / "validation_per_target.jsonl")
    parser.add_argument("--summary", type=Path, default=previous / "validation_summary.json")
    parser.add_argument(
        "--raw-validation-audit",
        type=Path,
        default=root / "work/rerun_c0_256_round2a_fixed_knn_e33/b7_calibration/validation_all_candidates.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "work/rerun_c0_256_propagation_risk_analysis",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reproduction_reports/C0_256_propagation_risk_diagnosis.md",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates, previous_per_target, previous_summary, audit = load_frozen_validation(
        args.candidates,
        args.per_target,
        args.summary,
        args.raw_validation_audit,
    )
    all_rows, qreturn_rows, reproduction = build_candidate_table(candidates, previous_per_target)
    if not math.isclose(
        reproduction["qreturn_selected_dice"],
        previous_summary["selection_results"]["q_return"]["selected_dice"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Previous q_return selected Dice could not be reproduced exactly")
    physical_switch_rows = [row for row in all_rows if row["is_physical_switch"]]
    qreturn_switch_rows = [row for row in qreturn_rows if row["is_physical_switch"]]
    interaction = quartile_interaction(physical_switch_rows)
    competitive_rows = [
        row
        for row in physical_switch_rows
        if row["diagnostic_features"]["delta_q_return"] >= 0.0
    ]
    high_return_rows = [
        row
        for row in physical_switch_rows
        if row["diagnostic_features"]["q_return"] >= interaction["q_return_upper_quartile"]
    ]
    scope_rows = {
        "all_candidates_including_frozen_baseline": all_rows,
        "all_physical_switch_candidates": physical_switch_rows,
        "qreturn_competitive_physical_switch_candidates": competitive_rows,
        "high_return_physical_switch_candidates": high_return_rows,
        "qreturn_selected_all_targets": qreturn_rows,
        "qreturn_selected_physical_switches": qreturn_switch_rows,
    }
    risk_summary = {
        scope: scoped_group_summary(rows)
        for scope, rows in scope_rows.items()
    }
    risk_summary["q_return_q_multi_quartile_interaction"] = interaction
    risk_summary["frozen_reproduction"] = reproduction
    risk_summary["field_availability"] = audit

    effects = feature_effects(physical_switch_rows, "all_physical_switch_candidates")
    effects.extend(
        feature_effects(competitive_rows, "qreturn_competitive_physical_switch_candidates")
    )
    effects.extend(feature_effects(qreturn_switch_rows, "qreturn_selected_physical_switches"))
    aucs = single_feature_auc(physical_switch_rows, "all_physical_switch_candidates")
    aucs.extend(
        single_feature_auc(competitive_rows, "qreturn_competitive_physical_switch_candidates")
    )
    aucs.extend(single_feature_auc(high_return_rows, "high_return_physical_switch_candidates"))
    aucs.extend(single_feature_auc(qreturn_switch_rows, "qreturn_selected_physical_switches"))

    bridge = {
        "all_physical_switch_candidates": group_risk(physical_switch_rows, "candidate_bridge_count"),
        "qreturn_selected_all_targets": group_risk(qreturn_rows, "candidate_bridge_count"),
        "qreturn_selected_physical_switches": group_risk(qreturn_switch_rows, "candidate_bridge_count"),
    }
    mode = {
        "all_physical_switch_candidates": group_risk(physical_switch_rows, "candidate_route_mode"),
        "qreturn_selected_all_targets": group_risk(qreturn_rows, "candidate_route_mode"),
        "qreturn_selected_physical_switches": group_risk(qreturn_switch_rows, "candidate_route_mode"),
    }
    logistic, logistic_predictions = grouped_cv(physical_switch_rows, "logistic", args.seed)
    tree, tree_predictions = grouped_cv(physical_switch_rows, "tree", args.seed)
    competitive_logistic, competitive_logistic_predictions = grouped_cv(
        competitive_rows, "logistic", args.seed
    )
    competitive_tree, competitive_tree_predictions = grouped_cv(
        competitive_rows, "tree", args.seed
    )
    catastrophic = [
        diagnose_case(
            row,
            logistic_predictions,
            tree_predictions,
            competitive_logistic_predictions,
            competitive_tree_predictions,
        )
        for row in sorted(qreturn_rows, key=lambda item: item["delta_dice_evaluation_only"])
        if row["catastrophic_risk_evaluation_only"]
    ]
    all_catastrophic = [
        diagnose_case(
            row,
            logistic_predictions,
            tree_predictions,
            competitive_logistic_predictions,
            competitive_tree_predictions,
        )
        for row in sorted(physical_switch_rows, key=lambda item: item["delta_dice_evaluation_only"])
        if row["catastrophic_risk_evaluation_only"]
    ]
    findings = mechanism_findings(
        risk_summary["qreturn_selected_all_targets"],
        aucs,
        logistic,
        tree,
        competitive_logistic,
        competitive_tree,
        risk_summary["q_return_q_multi_quartile_interaction"],
        catastrophic,
    )
    risk_summary["mechanism_findings"] = findings

    write_jsonl(args.output_root / "all_switch_candidates.jsonl", all_rows)
    write_jsonl(args.output_root / "qreturn_switch_analysis.jsonl", qreturn_rows)
    write_json(args.output_root / "risk_group_summary.json", risk_summary)
    write_csv(args.output_root / "feature_effect_table.csv", effects)
    write_csv(args.output_root / "single_feature_risk_auc.csv", aucs)
    write_jsonl(args.output_root / "catastrophic_failures.jsonl", all_catastrophic)
    write_jsonl(args.output_root / "qreturn_catastrophic_failures.jsonl", catastrophic)
    write_json(args.output_root / "bridge_risk_summary.json", bridge)
    write_json(args.output_root / "route_mode_risk_summary.json", mode)
    write_json(args.output_root / "risk_cv_logistic.json", logistic)
    write_json(args.output_root / "risk_cv_tree.json", tree)
    write_json(args.output_root / "risk_cv_logistic_competitive.json", competitive_logistic)
    write_json(args.output_root / "risk_cv_tree_competitive.json", competitive_tree)
    report(
        args.report,
        args.repo_root,
        audit,
        reproduction,
        risk_summary,
        effects,
        aucs,
        bridge,
        mode,
        logistic,
        tree,
        competitive_logistic,
        competitive_tree,
        catastrophic,
        findings,
    )
    print(
        json.dumps(
            {
                "frozen_reproduction": reproduction,
                "all_switch_risk_counts": {
                    "candidate_count": risk_summary["all_physical_switch_candidates"]["candidate_count"],
                    "severe_count": risk_summary["all_physical_switch_candidates"]["severe_candidate_count"],
                    "catastrophic_count": risk_summary["all_physical_switch_candidates"]["catastrophic_candidate_count"],
                },
                "qreturn_risk_counts": {
                    "target_count": len(qreturn_rows),
                    "physical_switch_count": len(qreturn_switch_rows),
                    "severe_count": risk_summary["qreturn_selected_all_targets"]["severe_candidate_count"],
                    "catastrophic_count": risk_summary["qreturn_selected_all_targets"]["catastrophic_candidate_count"],
                },
                "competitive_risk_counts": {
                    "candidate_count": len(competitive_rows),
                    "target_count": risk_summary[
                        "qreturn_competitive_physical_switch_candidates"
                    ]["unique_target_count"],
                    "severe_count": risk_summary[
                        "qreturn_competitive_physical_switch_candidates"
                    ]["severe_candidate_count"],
                    "severe_target_count": risk_summary[
                        "qreturn_competitive_physical_switch_candidates"
                    ]["severe_unique_target_count"],
                    "catastrophic_count": risk_summary[
                        "qreturn_competitive_physical_switch_candidates"
                    ]["catastrophic_candidate_count"],
                },
                "mechanism_findings": findings,
                "logistic_cv": logistic["out_of_fold_all_switch_candidates"],
                "tree_cv": tree["out_of_fold_all_switch_candidates"],
                "logistic_qreturn_oof": logistic["out_of_fold_qreturn_switched_candidates"],
                "tree_qreturn_oof": tree["out_of_fold_qreturn_switched_candidates"],
                "competitive_logistic_cv": competitive_logistic[
                    "out_of_fold_all_switch_candidates"
                ],
                "competitive_tree_cv": competitive_tree[
                    "out_of_fold_all_switch_candidates"
                ],
                "competitive_logistic_qreturn_oof": competitive_logistic[
                    "out_of_fold_qreturn_switched_candidates"
                ],
                "competitive_tree_qreturn_oof": competitive_tree[
                    "out_of_fold_qreturn_switched_candidates"
                ],
                "catastrophic_qreturn_cases": [
                    {
                        "target_id": row["target_id"],
                        "delta_dice": row["delta_dice_evaluation_only"],
                        "q_return": row["diagnostic_features"]["q_return"],
                        "q_multi": row["diagnostic_features"]["q_multi"],
                        "q_model": row["diagnostic_features"]["q_model"],
                        "b7": row["diagnostic_features"]["b7"],
                        "delta_q_multi": row["diagnostic_features"]["delta_q_multi"],
                        "delta_q_model": row["diagnostic_features"]["delta_q_model"],
                        "delta_trace_adjacent_dice_min": row["diagnostic_features"]["delta_trace_adjacent_dice_min"],
                        "oof_logistic_probability": row["target_disjoint_oof_logistic_risk_probability"],
                        "oof_tree_probability": row["target_disjoint_oof_tree_risk_probability"],
                        "competitive_oof_logistic_probability": row[
                            "competitive_target_disjoint_oof_logistic_risk_probability"
                        ],
                        "competitive_oof_tree_probability": row[
                            "competitive_target_disjoint_oof_tree_risk_probability"
                        ],
                        "diagnosis": row["diagnostic_observations"],
                    }
                    for row in catastrophic
                ],
                "output_root": str(args.output_root),
                "report": str(args.report),
                "gpu_used": False,
                "test_read": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
