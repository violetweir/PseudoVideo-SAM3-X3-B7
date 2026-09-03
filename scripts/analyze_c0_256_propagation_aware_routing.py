#!/usr/bin/env python3
"""Audit frozen C0-256 route proxies without SAM3 inference or GT leakage.

The validation split is always loaded, audited, evaluated, and written before
the code is even allowed to open a test candidate or selection artifact.  GT
Dice may only enter post-selection evaluation and the explicitly named oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import kendalltau, spearmanr


ROUTE_MODES = (
    "sam3enc_anchor_conditioned_target_pooling",
    "sam3enc_anchor_conditioned_patch_correspondence",
)
EXPECTED_BRIDGES = tuple(range(7))
GT_FIELD = "gt_dice_evaluation_only"
PROXY_FIELDS = {
    "knn_bottleneck": "path_bottleneck_similarity",
    "knn_mean": "path_mean_similarity",
    "q_return": "q_return",
}
METHOD_LABELS = {
    "knn_bottleneck": "KNN 瓶颈相似度",
    "knn_mean": "KNN 平均相似度",
    "q_return": "传播返回一致性 q_return",
    "b7": "既有 B7（冻结 X3）",
    "gt_oracle": "GT Oracle（仅评价）",
}
REQUIRED_FIELDS = (
    "target_id",
    "route_id",
    "bridge_count",
    "anchor_id",
    "bridge_ids",
    "path_bottleneck_similarity",
    "path_mean_similarity",
    "q_cycle",
    GT_FIELD,
)
OPTIONAL_FIELDS = (
    "q_multi",
    "q_model",
    "b7",
    "forward_sam_score",
    "trace_adjacent_dice_mean",
    "trace_adjacent_dice_min",
    "trace_sam_score_mean",
    "trace_sam_score_min",
    "forward_mask_path",
    "cycle_success",
    "cycle_failure_reason",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path.absolute()),
        "resolved_path": str(path.resolve()),
        "role": role,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def finite_float(value: Any, description: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {description}: {value!r}")
    return result


def candidate_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["target_id"]), str(row["route_mode"]), str(row["route_id"])


def normalize_candidate(row: dict[str, Any], route_mode: str, split: str) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Missing fields {missing} in {route_mode}/{row.get('route_id')}")
    if row.get("status") != "success":
        raise ValueError(f"Non-success propagation candidate: {row.get('route_id')}")
    if row.get("target_split") != split:
        raise ValueError(f"Unexpected split for {row.get('route_id')}: {row.get('target_split')}")
    if row.get("target_gt_used_for_search_or_inference") is not False:
        raise ValueError(f"GT-free search/inference audit failed for {row.get('route_id')}")

    q_cycle = finite_float(row["q_cycle"], "q_cycle")
    normalized = {
        "target_id": str(row["target_id"]),
        "route_id": str(row["route_id"]),
        "route_mode": route_mode,
        "bridge_count": int(row["bridge_count"]),
        "anchor_id": str(row["anchor_id"]),
        "bridge_ids": [str(item) for item in row["bridge_ids"]],
        "path_bottleneck_similarity": finite_float(
            row["path_bottleneck_similarity"], "path_bottleneck_similarity"
        ),
        "path_mean_similarity": finite_float(row["path_mean_similarity"], "path_mean_similarity"),
        "q_cycle": q_cycle,
        "q_return": q_cycle,
        GT_FIELD: finite_float(row[GT_FIELD], GT_FIELD),
        "target_gt_used_for_search_or_inference": False,
        "target_split": split,
    }
    if len(normalized["bridge_ids"]) != normalized["bridge_count"]:
        raise ValueError(f"bridge_ids/count mismatch for {row['route_id']}")
    for field in OPTIONAL_FIELDS:
        if field in row:
            normalized[field] = row[field]
    if "forward_sam_score" not in normalized and "final_sam_score" in row:
        normalized["forward_sam_score"] = finite_float(row["final_sam_score"], "final_sam_score")
    return normalized


def apply_b7_overlay(
    candidates: list[dict[str, Any]], overlay_path: Path, split: str
) -> dict[str, Any]:
    overlay_rows = read_jsonl(overlay_path)
    overlay: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in overlay_rows:
        key = candidate_key(row)
        if key in overlay:
            raise ValueError(f"Duplicate historical B7 overlay candidate: {key}")
        overlay[key] = row
    if len(overlay) != len(candidates):
        raise ValueError(f"B7 audit count mismatch: {len(overlay)} != {len(candidates)}")
    for candidate in candidates:
        key = candidate_key(candidate)
        if key not in overlay:
            raise ValueError(f"Historical B7 audit is missing candidate: {key}")
        original = overlay[key]
        if original.get("target_split") != split:
            raise ValueError(f"B7 audit split mismatch: {key}")
        for field in ("bridge_count", "anchor_id"):
            if original[field] != candidate[field]:
                raise ValueError(f"B7 audit field mismatch for {field}: {key}")
        for field in (
            "path_bottleneck_similarity",
            "path_mean_similarity",
            "q_cycle",
            GT_FIELD,
        ):
            if not math.isclose(float(original[field]), candidate[field], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"B7 audit numeric mismatch for {field}: {key}")
        if not math.isclose(float(original["q_return"]), candidate["q_cycle"], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Historical q_return != q_cycle: {key}")
        for field in ("q_multi", "q_model", "b7"):
            candidate[field] = finite_float(original[field], field)
    return artifact_record(overlay_path, "historical_b7_all_candidates")


def audit_candidates(
    rows: list[dict[str, Any]], split: str, expected_targets: int
) -> dict[str, Any]:
    target_counts = Counter(row["target_id"] for row in rows)
    mode_counts = Counter(row["route_mode"] for row in rows)
    bridge_counts = Counter(row["bridge_count"] for row in rows)
    mode_bridge = Counter((row["route_mode"], row["bridge_count"]) for row in rows)
    target_mode_bridge = Counter(
        (row["target_id"], row["route_mode"], row["bridge_count"]) for row in rows
    )
    keys = Counter(candidate_key(row) for row in rows)
    expected_candidates = expected_targets * len(ROUTE_MODES) * len(EXPECTED_BRIDGES)
    if len(target_counts) != expected_targets:
        raise ValueError(f"{split}: expected {expected_targets} targets, got {len(target_counts)}")
    if len(rows) != expected_candidates:
        raise ValueError(f"{split}: expected {expected_candidates} candidates, got {len(rows)}")
    if set(mode_counts) != set(ROUTE_MODES):
        raise ValueError(f"{split}: route mode mismatch: {sorted(mode_counts)}")
    if set(bridge_counts) != set(EXPECTED_BRIDGES):
        raise ValueError(f"{split}: bridge_count mismatch: {sorted(bridge_counts)}")
    bad_targets = {target: count for target, count in target_counts.items() if count != 14}
    if bad_targets:
        raise ValueError(f"{split}: target candidate-count mismatch: {bad_targets}")
    bad_cells = {str(key): count for key, count in target_mode_bridge.items() if count != 1}
    if bad_cells or len(target_mode_bridge) != expected_candidates:
        raise ValueError(f"{split}: target/mode/bridge lattice mismatch: {bad_cells}")
    duplicate_keys = {str(key): count for key, count in keys.items() if count != 1}
    if duplicate_keys:
        raise ValueError(f"{split}: duplicate candidate keys: {duplicate_keys}")
    return {
        "split": split,
        "target_count": len(target_counts),
        "candidate_count": len(rows),
        "expected_target_count": expected_targets,
        "expected_candidate_count": expected_candidates,
        "candidates_per_target": dict(sorted(Counter(target_counts.values()).items())),
        "bridge_counts": {str(key): bridge_counts[key] for key in sorted(bridge_counts)},
        "route_mode_counts": dict(sorted(mode_counts.items())),
        "mode_bridge_counts": {
            f"{mode}/b{bridge}": mode_bridge[(mode, bridge)]
            for mode in ROUTE_MODES
            for bridge in EXPECTED_BRIDGES
        },
        "distinct_route_ids": len({row["route_id"] for row in rows}),
        "gt_free_search_and_inference_confirmed": True,
    }


def load_candidates(
    quality_root: Path,
    split: str,
    expected_targets: int,
    b7_overlay_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for mode in ROUTE_MODES:
        path = quality_root / mode / f"propagation_quality_{split}" / "propagation_quality.jsonl"
        inputs.append(artifact_record(path, f"{split}_{mode}_frozen_propagation"))
        candidates.extend(normalize_candidate(row, mode, split) for row in read_jsonl(path))
    if b7_overlay_path is not None:
        inputs.append(apply_b7_overlay(candidates, b7_overlay_path, split))
    candidates.sort(key=lambda row: (row["target_id"], row["route_mode"], row["bridge_count"], row["route_id"]))
    return candidates, inputs, audit_candidates(candidates, split, expected_targets)


def correlation(proxy: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    if len(proxy) != len(gt):
        raise ValueError("Proxy/GT array length mismatch")
    if len(proxy) < 2 or np.all(proxy == proxy[0]) or np.all(gt == gt[0]):
        return {
            "n": int(len(proxy)),
            "defined": False,
            "spearman_rho": None,
            "spearman_p_value": None,
            "kendall_tau": None,
            "kendall_p_value": None,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, rho_p = spearmanr(proxy, gt)
        tau, tau_p = kendalltau(proxy, gt)
    return {
        "n": int(len(proxy)),
        "defined": True,
        "spearman_rho": float(rho),
        "spearman_p_value": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p_value": float(tau_p),
    }


def selection_key(field: str) -> Callable[[dict[str, Any]], tuple[Any, ...]]:
    if field == "b7":
        # Exact historical B7 tie-break from build_c0_256_b7_lora_manifest.py.
        return lambda row: (row["b7"], row["q_multi"], row["q_return"], row["route_id"])
    # The deployable selectors never inspect GT: fixed proxy, then route IDs.
    return lambda row: (row[field], row["route_id"], row["route_mode"])


def historical_b7_selections(
    path: Path,
    grouped: dict[str, list[dict[str, Any]]],
    split: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(path)
    historical: dict[str, dict[str, Any]] = {}
    for original in rows:
        target_id = str(original["target_id"])
        if target_id in historical:
            raise ValueError(f"Duplicate historical B7 selected target: {target_id}")
        if original.get("split") != split:
            raise ValueError(f"Historical B7 selected split mismatch: {target_id}")
        matches = [
            candidate
            for candidate in grouped.get(target_id, [])
            if candidate["route_id"] == original["route_id"]
            and candidate["route_mode"] == original["route_mode"]
        ]
        if len(matches) != 1:
            raise ValueError(f"Historical B7 choice does not map uniquely to frozen pool: {target_id}")
        chosen = matches[0]
        if int(original["bridge_count"]) != chosen["bridge_count"]:
            raise ValueError(f"Historical B7 bridge mismatch: {target_id}")
        if GT_FIELD in original and not math.isclose(
            float(original[GT_FIELD]), chosen[GT_FIELD], rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"Historical B7 evaluation audit mismatch: {target_id}")
        for field in ("q_return", "q_multi", "q_model", "b7"):
            if field in original:
                value = finite_float(original[field], field)
                if field == "q_return" and not math.isclose(value, chosen["q_cycle"], rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"Historical B7 return mismatch: {target_id}")
                chosen[field] = value
        historical[target_id] = chosen
    if set(historical) != set(grouped):
        missing = sorted(set(grouped) - set(historical))
        extra = sorted(set(historical) - set(grouped))
        raise ValueError(f"Historical B7 coverage mismatch: missing={missing}, extra={extra}")
    return historical, artifact_record(path, "historical_frozen_b7_selected_routes")


def selected_snapshot(row: dict[str, Any], method: str) -> dict[str, Any]:
    result = {
        "route_id": row["route_id"],
        "route_mode": row["route_mode"],
        "bridge_count": row["bridge_count"],
        "path_bottleneck_similarity": row["path_bottleneck_similarity"],
        "path_mean_similarity": row["path_mean_similarity"],
        "q_return": row["q_return"],
        GT_FIELD: row[GT_FIELD],
    }
    if method == "b7" and "b7" in row:
        result["b7"] = row["b7"]
    return result


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row[field]) for row in rows)
    return dict(sorted(counts.items()))


def within_target_summary(
    correlations: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in PROXY_FIELDS:
        values = [
            payload[method]["spearman_rho"]
            for payload in correlations.values()
            if payload[method]["defined"]
        ]
        array = np.asarray(values, dtype=np.float64)
        result[method] = {
            "defined_target_count": int(len(array)),
            "undefined_target_count": int(len(correlations) - len(array)),
            "mean_spearman_rho": float(np.mean(array)) if len(array) else None,
            "median_spearman_rho": float(np.median(array)) if len(array) else None,
            "positive_target_fraction": float(np.mean(array > 0)) if len(array) else None,
            "positive_target_count": int(np.sum(array > 0)),
        }
    result["q_return_vs_knn"] = {}
    for baseline in ("knn_bottleneck", "knn_mean"):
        paired = [
            (
                payload["q_return"]["spearman_rho"],
                payload[baseline]["spearman_rho"],
            )
            for payload in correlations.values()
            if payload["q_return"]["defined"] and payload[baseline]["defined"]
        ]
        wins = sum(q > knn for q, knn in paired)
        ties = sum(math.isclose(q, knn, rel_tol=0.0, abs_tol=1e-12) for q, knn in paired)
        result["q_return_vs_knn"][baseline] = {
            "paired_defined_target_count": len(paired),
            "q_return_higher_target_count": wins,
            "q_return_higher_target_fraction": wins / len(paired) if paired else None,
            "tied_target_count": ties,
        }
    return result


def paired_comparison(
    per_target: list[dict[str, Any]], baseline: str, iterations: int, seed: int
) -> dict[str, Any]:
    differences = np.asarray(
        [
            row["selections"]["q_return"][GT_FIELD]
            - row["selections"][baseline][GT_FIELD]
            for row in per_target
        ],
        dtype=np.float64,
    )
    tied = np.isclose(differences, 0.0, rtol=0.0, atol=1e-12)
    improved = (differences > 0.0) & ~tied
    degraded = (differences < 0.0) & ~tied
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(iterations, len(differences)))
    bootstrap = np.mean(differences[indices], axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    negative_differences = np.sort(differences[degraded])
    positive_differences = differences[improved]
    return {
        "baseline_method": baseline,
        "target_count": len(per_target),
        "improved_target_count": int(np.sum(improved)),
        "tied_target_count": int(np.sum(tied)),
        "degraded_target_count": int(np.sum(degraded)),
        "mean_delta": float(np.mean(differences)),
        "median_delta": float(np.median(differences)),
        "bootstrap_unit": "paired_target",
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "bootstrap_95_ci": [float(low), float(high)],
        "failure_tail_diagnostics": {
            "largest_degradation": float(np.min(differences)),
            "largest_improvement": float(np.max(differences)),
            "two_largest_degradations_total": float(np.sum(negative_differences[:2])),
            "all_degradations_total": float(np.sum(negative_differences)),
            "all_improvements_total": float(np.sum(positive_differences)),
        },
    }


def switch_analysis(per_target: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    baseline_rows = [row["selections"][baseline] for row in per_target]
    q_rows = [row["selections"]["q_return"] for row in per_target]
    same_route = [
        row
        for row in per_target
        if row["selections"][baseline]["route_id"] == row["selections"]["q_return"]["route_id"]
    ]
    switched = [
        row
        for row in per_target
        if row["selections"][baseline]["route_id"] != row["selections"]["q_return"]["route_id"]
    ]
    buckets: dict[str, dict[str, dict[str, Any]]] = {"bridge_count": {}, "route_mode": {}}
    for field in buckets:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in switched:
            q = row["selections"]["q_return"]
            knn = row["selections"][baseline]
            grouped[str(q[field])].append(q[GT_FIELD] - knn[GT_FIELD])
        for label, values in sorted(grouped.items()):
            array = np.asarray(values, dtype=np.float64)
            ties = np.isclose(array, 0.0, rtol=0.0, atol=1e-12)
            buckets[field][label] = {
                "target_count": len(values),
                "improved_target_count": int(np.sum((array > 0) & ~ties)),
                "tied_target_count": int(np.sum(ties)),
                "degraded_target_count": int(np.sum((array < 0) & ~ties)),
                "mean_delta": float(np.mean(array)),
            }
    deltas = [
        row["selections"]["q_return"][GT_FIELD] - row["selections"][baseline][GT_FIELD]
        for row in switched
    ]
    switched_array = np.asarray(deltas, dtype=np.float64)
    switched_ties = np.isclose(switched_array, 0.0, rtol=0.0, atol=1e-12)
    return {
        "baseline_method": baseline,
        "same_route_target_count": len(same_route),
        "different_route_target_count": len(switched),
        "same_route_identity": "route_id; identical IDs across route modes are the same physical route",
        "switched_improved_target_count": int(np.sum((switched_array > 0) & ~switched_ties)),
        "switched_tied_target_count": int(np.sum(switched_ties)),
        "switched_degraded_target_count": int(np.sum((switched_array < 0) & ~switched_ties)),
        "selected_bridge_distributions": {
            baseline: distribution(baseline_rows, "bridge_count"),
            "q_return": distribution(q_rows, "bridge_count"),
        },
        "selected_route_mode_distributions": {
            baseline: distribution(baseline_rows, "route_mode"),
            "q_return": distribution(q_rows, "route_mode"),
        },
        "switches_by_q_return_bridge_count": buckets["bridge_count"],
        "switches_by_q_return_route_mode": buckets["route_mode"],
    }


def extreme_cases(per_target: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    cases = []
    for row in per_target:
        knn = row["selections"][baseline]
        q = row["selections"]["q_return"]
        cases.append(
            {
                "target_id": row["target_id"],
                "baseline_method": baseline,
                "knn_selection": dict(knn),
                "q_return_selection": dict(q),
                "dice_delta": q[GT_FIELD] - knn[GT_FIELD],
            }
        )
    gains = sorted((case for case in cases if case["dice_delta"] > 1e-12), key=lambda case: case["dice_delta"], reverse=True)
    losses = sorted((case for case in cases if case["dice_delta"] < -1e-12), key=lambda case: case["dice_delta"])
    return {
        "largest_improvements": gains[:10],
        "largest_degradations": losses[:10],
    }


def analyze_split(
    candidates: list[dict[str, Any]],
    split: str,
    input_artifacts: list[dict[str, Any]],
    audit: dict[str, Any],
    iterations: int,
    seed: int,
    frozen_knn_baseline: str | None = None,
    historical_b7_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["target_id"]].append(candidate)

    b7_historical: dict[str, dict[str, Any]] | None = None
    if historical_b7_path is not None:
        b7_historical, artifact = historical_b7_selections(historical_b7_path, grouped, split)
        input_artifacts.append(artifact)

    correlations: dict[str, Any] = {}
    gt = np.asarray([row[GT_FIELD] for row in candidates], dtype=np.float64)
    for method, field in PROXY_FIELDS.items():
        proxy = np.asarray([row[field] for row in candidates], dtype=np.float64)
        correlations[method] = {"field": field, **correlation(proxy, gt)}
    full_b7_available = all("b7" in row for row in candidates)
    if full_b7_available:
        b7_proxy = np.asarray([row["b7"] for row in candidates], dtype=np.float64)
        correlations["b7"] = {"field": "b7", **correlation(b7_proxy, gt)}

    within_target: dict[str, dict[str, dict[str, Any]]] = {}
    per_target: list[dict[str, Any]] = []
    for target_id, rows in sorted(grouped.items()):
        # GT remains absent from every deployable selection key.
        selected = {
            method: max(rows, key=selection_key(field))
            for method, field in PROXY_FIELDS.items()
        }
        if full_b7_available:
            selected["b7"] = max(rows, key=selection_key("b7"))
        elif b7_historical is not None:
            selected["b7"] = b7_historical[target_id]

        # This is the sole GT-based route choice, explicitly evaluation-only.
        selected["gt_oracle"] = max(
            rows,
            key=lambda row: (row[GT_FIELD], row["route_id"], row["route_mode"]),
        )
        target_gt = np.asarray([row[GT_FIELD] for row in rows], dtype=np.float64)
        target_correlations = {
            method: correlation(
                np.asarray([row[field] for row in rows], dtype=np.float64),
                target_gt,
            )
            for method, field in PROXY_FIELDS.items()
        }
        within_target[target_id] = target_correlations
        per_target.append(
            {
                "target_id": target_id,
                "candidate_count": len(rows),
                "selections": {
                    method: selected_snapshot(row, method)
                    for method, row in selected.items()
                },
                "within_target_correlations": target_correlations,
            }
        )

    method_names = list(PROXY_FIELDS)
    if full_b7_available or b7_historical is not None:
        method_names.append("b7")
    method_names.append("gt_oracle")
    oracle_mean = float(np.mean([row["selections"]["gt_oracle"][GT_FIELD] for row in per_target]))
    selection_summary = {}
    for method in method_names:
        chosen = [row["selections"][method] for row in per_target]
        mean_dice = float(np.mean([row[GT_FIELD] for row in chosen]))
        selection_summary[method] = {
            "target_count": len(chosen),
            "selected_dice": mean_dice,
            "gt_oracle": oracle_mean,
            "oracle_gap": oracle_mean - mean_dice,
            "bridge_count_distribution": distribution(chosen, "bridge_count"),
            "route_mode_distribution": distribution(chosen, "route_mode"),
        }

    if frozen_knn_baseline is None:
        # This labels the strongest *evaluated baseline*; it does not alter any
        # candidate choice, proxy, hyperparameter, or route-selection formula.
        frozen_knn_baseline = max(
            ("knn_bottleneck", "knn_mean"),
            key=lambda method: (selection_summary[method]["selected_dice"], method),
        )
        baseline_source = "validation_evaluation_strongest_knn_baseline"
    else:
        if frozen_knn_baseline not in ("knn_bottleneck", "knn_mean"):
            raise ValueError(f"Invalid frozen validation KNN baseline: {frozen_knn_baseline}")
        baseline_source = "frozen_on_validation_before_test_read"

    paired = {
        baseline: paired_comparison(per_target, baseline, iterations, seed)
        for baseline in ("knn_bottleneck", "knn_mean")
    }
    summary = {
        "split": split,
        "candidate_audit": audit,
        "input_artifacts": input_artifacts,
        "candidate_level_correlations": correlations,
        "within_target_ranking": within_target_summary(within_target),
        "selection_results": selection_summary,
        "strongest_knn_baseline": frozen_knn_baseline,
        "strongest_knn_baseline_source": baseline_source,
        "paired_comparisons": paired,
        "primary_paired_comparison": paired[frozen_knn_baseline],
        "route_switch": switch_analysis(per_target, frozen_knn_baseline),
        "failure_case_analysis": extreme_cases(per_target, frozen_knn_baseline),
        "b7_context": {
            "included": "b7" in selection_summary,
            "candidate_scores_available_for_all_routes": full_b7_available,
            "source": (
                "historical_all_candidate_b7_scores"
                if full_b7_available
                else "historical_frozen_selected_b7_routes"
                if b7_historical is not None
                else "not_available"
            ),
            "formula_recomputed": False,
        },
        "protocol": {
            "route_modes": list(ROUTE_MODES),
            "bridge_counts": list(EXPECTED_BRIDGES),
            "q_return_definition": "q_return = existing q_cycle = Dice(anchor_mask, returned_anchor_mask)",
            "knn_tie_break": "max(proxy, route_id, route_mode); no GT fields",
            "b7_tie_break": "historical max(b7, q_multi, q_return, route_id)",
            "gt_usage": "post-selection metrics, explicitly named GT oracle, correlations, and paired target bootstrap only",
            "gpu_inference_executed": False,
            "propagation_recomputed": False,
        },
    }
    return summary, per_target


def validation_gate(summary: dict[str, Any]) -> dict[str, Any]:
    correlations = summary["candidate_level_correlations"]
    return_rho = correlations["q_return"]["spearman_rho"]
    knn_rhos = {
        method: correlations[method]["spearman_rho"]
        for method in ("knn_bottleneck", "knn_mean")
    }
    correlation_pass = all(return_rho > rho for rho in knn_rhos.values())
    selection = summary["selection_results"]
    best_knn_dice = max(selection[method]["selected_dice"] for method in knn_rhos)
    top1_pass = selection["q_return"]["selected_dice"] > best_knn_dice
    confidence_pass = summary["primary_paired_comparison"]["bootstrap_95_ci"][0] > 0.0
    return {
        "passed": bool(correlation_pass and top1_pass),
        "q_return_spearman_greater_than_both_knn": bool(correlation_pass),
        "q_return_selected_dice_greater_than_both_knn": bool(top1_pass),
        "paired_bootstrap_ci_lower_above_zero": bool(confidence_pass),
        "q_return_spearman_rho": return_rho,
        "knn_spearman_rhos": knn_rhos,
        "q_return_selected_dice": selection["q_return"]["selected_dice"],
        "strongest_knn_selected_dice": best_knn_dice,
        "frozen_knn_baseline": summary["strongest_knn_baseline"],
    }


def number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "未定义"
    return f"{float(value):.{digits}f}"


def scientific(value: Any) -> str:
    if value is None:
        return "未定义"
    return f"{float(value):.3e}"


def correlation_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| 指标 | Spearman ρ | p-value | Kendall τ | Kendall p-value | N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, payload in summary["candidate_level_correlations"].items():
        lines.append(
            f"| {METHOD_LABELS[method]} | {number(payload['spearman_rho'])} | "
            f"{scientific(payload['spearman_p_value'])} | {number(payload['kendall_tau'])} | "
            f"{scientific(payload['kendall_p_value'])} | {payload['n']} |"
        )
    return lines


def ranking_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| 指标 | target 内平均 ρ | 中位数 ρ | 正相关 target | 有定义 target |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in PROXY_FIELDS:
        payload = summary["within_target_ranking"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {number(payload['mean_spearman_rho'])} | "
            f"{number(payload['median_spearman_rho'])} | "
            f"{payload['positive_target_count']}/{payload['defined_target_count']} "
            f"({number(payload['positive_target_fraction'], 3)}) | "
            f"{payload['defined_target_count']} |"
        )
    return lines


def selection_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| 选择方法 | Selected Dice | GT Oracle | Oracle Gap |",
        "|---|---:|---:|---:|",
    ]
    for method, payload in summary["selection_results"].items():
        lines.append(
            f"| {METHOD_LABELS[method]} | {number(payload['selected_dice'])} | "
            f"{number(payload['gt_oracle'])} | {number(payload['oracle_gap'])} |"
        )
    return lines


def case_lines(cases: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| target | KNN mode/b/route | KNN sim / q_return / Dice | "
        "q_return mode/b/route | q_return sim / q_return / Dice | Δ Dice |",
        "|---|---|---|---|---|---:|",
    ]
    for case in cases:
        knn = case["knn_selection"]
        q = case["q_return_selection"]
        similarity_field = PROXY_FIELDS[case["baseline_method"]]
        knn_mode = knn["route_mode"].removeprefix("sam3enc_anchor_conditioned_")
        q_mode = q["route_mode"].removeprefix("sam3enc_anchor_conditioned_")
        lines.append(
            f"| `{case['target_id']}` | {knn_mode} / b{knn['bridge_count']} / "
            f"`{knn['route_id']}` | {number(knn[similarity_field])} / "
            f"{number(knn['q_return'])} / {number(knn[GT_FIELD])} | "
            f"{q_mode} / b{q['bridge_count']} / `{q['route_id']}` | "
            f"{number(q[similarity_field])} / {number(q['q_return'])} / "
            f"{number(q[GT_FIELD])} | {number(case['dice_delta'])} |"
        )
    if not cases:
        lines.append("| 无符合条件的 target | — | — | — | — | — |")
    return lines


def split_report(summary: dict[str, Any], heading: str) -> list[str]:
    audit = summary["candidate_audit"]
    comparison = summary["primary_paired_comparison"]
    switches = summary["route_switch"]
    baseline = summary["strongest_knn_baseline"]
    lines = [
        f"## {heading}：候选数量审计",
        "",
        f"- targets：{audit['target_count']}；candidates：{audit['candidate_count']}。",
        f"- 每 target 候选数：`{json.dumps(audit['candidates_per_target'], ensure_ascii=False)}`。",
        f"- bridge 分布：`{json.dumps(audit['bridge_counts'], ensure_ascii=False, sort_keys=True)}`。",
        f"- route mode 分布：`{json.dumps(audit['route_mode_counts'], ensure_ascii=False, sort_keys=True)}`。",
        "- 每个 target × route mode × bridge_count 均恰好一条；未静默过滤失败、缺失或额外候选。",
        "",
        f"## {heading}：candidate-level 相关性",
        "",
        *correlation_table(summary),
        "",
        "全局相关性只作为辅助证据，不能替代同一 target 候选池中的排序和 Top-1 评价。",
        "",
        f"## {heading}：target 内部排序相关性",
        "",
        *ranking_table(summary),
        "",
    ]
    for other, payload in summary["within_target_ranking"]["q_return_vs_knn"].items():
        lines.append(
            f"- q_return target 内 ρ 高于 {METHOD_LABELS[other]}："
            f"{payload['q_return_higher_target_count']}/{payload['paired_defined_target_count']} "
            f"({number(payload['q_return_higher_target_fraction'], 3)})。"
        )
    lines.extend(
        [
            "",
            f"## {heading}：逐 target Top-1 与 oracle gap",
            "",
            *selection_table(summary),
            "",
            f"主要 KNN 对照：**{METHOD_LABELS[baseline]}**；来源："
            f"`{summary['strongest_knn_baseline_source']}`。",
            "",
            f"## {heading}：paired target bootstrap",
            "",
            f"- 改善 / 持平 / 退化：{comparison['improved_target_count']} / "
            f"{comparison['tied_target_count']} / {comparison['degraded_target_count']}。",
            f"- 平均 Δ Dice：{number(comparison['mean_delta'])}；"
            f"中位数 Δ Dice：{number(comparison['median_delta'])}。",
            f"- 95% paired-target bootstrap CI："
            f"[{number(comparison['bootstrap_95_ci'][0])}, "
            f"{number(comparison['bootstrap_95_ci'][1])}]；"
            f"resamples={comparison['bootstrap_iterations']}，seed={comparison['bootstrap_seed']}。",
            "- bootstrap 的独立重采样单位是同一个 target 的配对 Δ，不是 1400 条候选。",
            f"- 尾部风险审计：最大单例退化 "
            f"{number(comparison['failure_tail_diagnostics']['largest_degradation'])}，"
            f"最大单例改善 {number(comparison['failure_tail_diagnostics']['largest_improvement'])}，"
            f"两例最大退化合计 "
            f"{number(comparison['failure_tail_diagnostics']['two_largest_degradations_total'])}；"
            f"全部改善合计 "
            f"{number(comparison['failure_tail_diagnostics']['all_improvements_total'])}，"
            f"全部退化合计 "
            f"{number(comparison['failure_tail_diagnostics']['all_degradations_total'])}。"
            "因此胜场多于败场不保证平均 selected Dice 提升；此处不剔除异常 target，也不据此调 selector。",
            "",
            f"## {heading}：route switch 与获益分布",
            "",
            f"- 相同 route_id：{switches['same_route_target_count']}；"
            f"不同 route_id：{switches['different_route_target_count']}。",
            f"- switch targets 中改善 / 持平 / 退化："
            f"{switches['switched_improved_target_count']} / "
            f"{switches['switched_tied_target_count']} / "
            f"{switches['switched_degraded_target_count']}。",
            f"- bridge 选择分布：`{json.dumps(switches['selected_bridge_distributions'], ensure_ascii=False, sort_keys=True)}`。",
            f"- mode 选择分布：`{json.dumps(switches['selected_route_mode_distributions'], ensure_ascii=False, sort_keys=True)}`。",
            f"- switch 按 q_return bridge 分组：`{json.dumps(switches['switches_by_q_return_bridge_count'], ensure_ascii=False, sort_keys=True)}`。",
            f"- switch 按 q_return mode 分组：`{json.dumps(switches['switches_by_q_return_route_mode'], ensure_ascii=False, sort_keys=True)}`。",
            "",
            f"## {heading}：q_return 改善最大的最多 10 个 target",
            "",
            *case_lines(summary["failure_case_analysis"]["largest_improvements"]),
            "",
            f"## {heading}：q_return 退化最大的最多 10 个 target",
            "",
            *case_lines(summary["failure_case_analysis"]["largest_degradations"]),
            "",
        ]
    )
    return lines


def conclusion(validation: dict[str, Any], test: dict[str, Any] | None) -> tuple[str, str]:
    gate = validation["validation_gate"]
    if gate["q_return_spearman_greater_than_both_knn"] and gate["q_return_selected_dice_greater_than_both_knn"]:
        text = (
            "当前 validation 证据支持：传统特征相似性不能充分表示伪视频传播质量，"
            "SAM3 自身的传播返回一致性具有更强的路线质量判别能力。"
        )
        if test is not None:
            test_result = test["selection_results"]
            baseline = validation["strongest_knn_baseline"]
            if test_result["q_return"]["selected_dice"] > test_result[baseline]["selected_dice"]:
                text += " 冻结 test 比较也延续了 q_return 相对于 validation 冻结 KNN 基线的 Top-1 优势。"
            else:
                text += " 但冻结 test 未复现 q_return 相对于 validation 冻结 KNN 基线的 Top-1 优势，外部泛化证据不足。"
        recommendation = "在确认冻结 test 结果后，再考虑设计传播感知候选图或传播兼容性研究；本任务不自动训练任何模型。"
    elif gate["q_return_spearman_greater_than_both_knn"]:
        text = "q_return 能反映候选的平均质量，但未提高同一 target 的 Top-1 selected Dice，不能独立承担最终路线排序。"
        recommendation = "停止单独用 q_return 替代 KNN 的方案；如后续另行授权，可研究 q_return、q_multi、传播轨迹和学生信息的联合排序。"
    else:
        text = "当前“单独使用返回一致性替代相似度选路”的假设未获得 validation 支持，不能宣称图像相似性与传播兼容性的机制差异已被该 selector 证明。"
        recommendation = "停止这条简单替代方案，不自动调整 q_return、bridge 范围、B7 权重或训练复杂 selector。"
    return text, recommendation


def write_report(
    path: Path,
    repo_root: Path,
    quality_root: Path,
    validation: dict[str, Any],
    test: dict[str, Any] | None,
    round2b_status: dict[str, Any],
) -> None:
    gate = validation["validation_gate"]
    final_conclusion, recommendation = conclusion(validation, test)
    artifacts = list(validation["input_artifacts"])
    if test is not None:
        artifacts.extend(test["input_artifacts"])
    lines = [
        "# C0-256 传播感知选路机制验证：KNN 图像相似性 vs SAM3 返回一致性",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        f"> 仓库：`{repo_root}`  ",
        "> 实验类型：冻结历史候选的离线 CPU 统计；无新 GPU propagation、无训练、无参数搜索。",
        "",
        "## 1. 研究问题与实验动机",
        "",
        "对于同一个目标图像的 14 条冻结伪视频候选路线，比较现成 KNN 路径相似度与 "
        "SAM3 自身传播返回一致性，判断哪一种分数更能排序真实传播 Dice，并选出更好的 Top-1 路线。",
        "",
        "## 2. 与 Round2A / Round2B 的关系",
        "",
        "主实验固定 Round2A：SAM3-base@256 KNN topology + 已冻结的 SAM3-e33 propagation teacher。"
        "Route 图、anchors、两种 mode、b0-b6 和传播结果均不改变。",
        "",
        f"Round2B 辅助检查：{round2b_status['message']}。",
        "",
        "## 3. 数据协议与候选路线来源",
        "",
        f"Round2A 入口：`{quality_root}`。其中 validation/test propagation 目录是历史 "
        "e33 全量评估结果的符号链接；不跟随符号链接的 `find -type d` 会误以为这两个 split 缺失。",
        "",
        "每个 split 固定为 100 targets × 2 route modes × b0-b6 = 1400 candidates。"
        "每个 target 的 14 个候选始终共同参与每一种 selector。",
        "",
        "## 4. 所有实际读取输入文件及 SHA256",
        "",
        "| 角色 | 输入文件 | 实际解析路径 | SHA256 |",
        "|---|---|---|---|",
    ]
    for artifact in artifacts:
        lines.append(
            f"| `{artifact['role']}` | `{artifact['path']}` | "
            f"`{artifact['resolved_path']}` | `{artifact['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## 5. KNN 相似度、q_return 与既有 B7 定义",
            "",
            "- `path_bottleneck_similarity`：冻结路线中最弱边的现有 KNN 特征相似度。",
            "- `path_mean_similarity`：冻结路线各边的现有 KNN 平均特征相似度。",
            "- `q_return = q_cycle = Dice(anchor mask, returned anchor mask)`；直接读取已有传播结果，不重新运行 SAM3。",
            "- B7 仅作为上下文，读取历史冻结结果；既有公式 `B7=(q_return*q_multi²*q_model²)^0.2` 不被重新计算或调权。",
            "",
            "## 6. GT 使用边界与可审计 tie-break",
            "",
            "A/B/C 路线选择只调用 `max(proxy, route_id, route_mode)`；B7 使用历史固定 "
            "`max(b7, q_multi, q_return, route_id)` 或直接读取完整的冻结历史选择。"
            "`gt_dice_evaluation_only` 不进入这些 key、公式、阈值或候选筛选。",
            "",
            "GT 只用于选择后的 Dice 评价、candidate/target 排序相关性、GT Oracle 和 paired-target bootstrap。"
            "validation 上表现更好的 KNN 方法仅作为事后比较基线，并在读取 test 之前冻结；"
            "这一步不会改变任何路线选择方法或参数。",
            "",
            *split_report(validation, "Validation"),
            "## Validation gate 与 test 冻结协议",
            "",
            f"- q_return Spearman 高于两个 KNN 指标：**{gate['q_return_spearman_greater_than_both_knn']}**。",
            f"- q_return Top-1 selected Dice 高于两个 KNN selector：**{gate['q_return_selected_dice_greater_than_both_knn']}**。",
            f"- paired bootstrap 95% CI 下界大于 0：**{gate['paired_bootstrap_ci_lower_above_zero']}**。",
            f"- gate 最终通过：**{gate['passed']}**。",
            f"- validation 冻结 KNN 主对照：`{gate['frozen_knn_baseline']}`。",
            "- 只有前两条同时为真，程序才会打开 test propagation / 历史 B7 选择文件；"
            "不会根据 test Dice 调分数、换 baseline、删除 mode 或缩小 bridge 范围。",
            "",
        ]
    )
    if test is not None:
        lines.extend(split_report(test, "Test（完全冻结）"))
    else:
        lines.extend(
            [
                "## Test 状态",
                "",
                "Validation gate 未通过，因此未读取任何 test 候选内容或 test selector 结果，"
                "也没有生成 test_candidates.jsonl、test_per_target.jsonl 或 test_summary.json。",
                "",
            ]
        )
    lines.extend(
        [
            "## 最终结论：是否支持“图像相似性 ≠ 传播兼容性”",
            "",
            final_conclusion,
            "",
            "## 下一步建议与明确停止边界",
            "",
            recommendation,
            "",
            "本轮没有运行 GPU 推理，没有重算 q_cycle，没有修改历史 Round2A/Round2B artifact，"
            "没有调整 B7、bridge 范围或训练 SAM3、Student、X5、learned router。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def round2b_availability(repo_root: Path) -> dict[str, Any]:
    root = repo_root / "work/rerun_c0_256_round2b_e33_topology_factorial/teachers/e33/quality_root"
    validation_paths = [
        root / mode / "propagation_quality_validation/propagation_quality.jsonl"
        for mode in ROUTE_MODES
    ]
    missing = [str(path) for path in validation_paths if not path.is_file()]
    if missing:
        return {
            "executed": False,
            "missing_validation_inputs": missing,
            "message": "e33 topology + e33 teacher 仅有历史 test 传播，缺少完整 validation q_cycle/GT，因此按协议跳过且不启动 GPU",
        }
    return {
        "executed": False,
        "missing_validation_inputs": [],
        "message": "发现 Round2B validation 输入，但主任务仅交付 Round2A；未在缺少预冻结辅助方案时追加分析",
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    round2a_root = repo_root / "work/rerun_c0_256_round2a_fixed_knn_e33"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--quality-root", type=Path, default=round2a_root / "quality_root")
    parser.add_argument(
        "--validation-b7-candidates",
        type=Path,
        default=round2a_root / "b7_calibration/validation_all_candidates.jsonl",
    )
    parser.add_argument(
        "--test-b7-selected",
        type=Path,
        default=round2a_root / "x4_b7_closeout/X3_best_test.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "work/rerun_c0_256_propagation_aware_routing",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reproduction_reports/C0_256_propagation_aware_routing.md",
    )
    parser.add_argument("--expected-targets", type=int, default=100)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_rows, validation_inputs, validation_audit = load_candidates(
        args.quality_root,
        "validation",
        args.expected_targets,
        args.validation_b7_candidates,
    )
    validation, validation_per_target = analyze_split(
        validation_rows,
        "validation",
        validation_inputs,
        validation_audit,
        args.bootstrap_iterations,
        args.seed,
    )
    validation["validation_gate"] = validation_gate(validation)
    write_jsonl(args.output_root / "validation_candidates.jsonl", validation_rows)
    write_jsonl(args.output_root / "validation_per_target.jsonl", validation_per_target)
    write_json(args.output_root / "validation_summary.json", validation)

    # No test artifact may be opened above this explicit, persisted gate.
    test: dict[str, Any] | None = None
    if validation["validation_gate"]["passed"]:
        test_rows, test_inputs, test_audit = load_candidates(
            args.quality_root,
            "test",
            args.expected_targets,
        )
        b7_path = args.test_b7_selected if args.test_b7_selected.is_file() else None
        test, test_per_target = analyze_split(
            test_rows,
            "test",
            test_inputs,
            test_audit,
            args.bootstrap_iterations,
            args.seed,
            frozen_knn_baseline=validation["strongest_knn_baseline"],
            historical_b7_path=b7_path,
        )
        write_jsonl(args.output_root / "test_candidates.jsonl", test_rows)
        write_jsonl(args.output_root / "test_per_target.jsonl", test_per_target)
        write_json(args.output_root / "test_summary.json", test)

    round2b = round2b_availability(args.repo_root)
    write_report(args.report, args.repo_root, args.quality_root, validation, test, round2b)
    print(
        json.dumps(
            {
                "validation_gate": validation["validation_gate"],
                "validation_selection_results": validation["selection_results"],
                "validation_primary_paired_comparison": validation["primary_paired_comparison"],
                "test_executed": test is not None,
                "test_selection_results": test["selection_results"] if test is not None else None,
                "test_primary_paired_comparison": test["primary_paired_comparison"] if test is not None else None,
                "round2b": round2b,
                "output_root": str(args.output_root),
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
