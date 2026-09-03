#!/usr/bin/env python3
"""Frozen, validation-only, target-level risk-gated route counterfactual.

Each target contributes exactly one decision: keep its pre-existing KNN-mean
route or accept its pre-existing q_return-max route. Target GT provides only
the explicitly authorized validation training labels and final evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from analyze_c0_256_propagation_aware_routing import (
    artifact_record,
    read_jsonl,
    write_json,
    write_jsonl,
)


SEED = 2026
BOOTSTRAP_RESAMPLES = 10000
RISK_THRESHOLD = 0.5
KNOWN_CATASTROPHES = (
    "kvasir-seg::cju7b1ygu1msd0801hywhy0mc",
    "kvasir-seg::cju1ewnoh5z030855vpex9uzt",
)
# This tuple is frozen before any gate is fitted or OOF outcomes are examined.
FEATURE_SPEC = (
    ("q_return_candidate", "candidate", "q_return"),
    ("q_multi_candidate", "candidate", "q_multi"),
    ("q_model_candidate", "candidate", "q_model"),
    ("delta_q_return", "delta", "q_return"),
    ("delta_q_multi", "delta", "q_multi"),
    ("delta_q_model", "delta", "q_model"),
    ("trace_area_max_rel_delta_candidate", "candidate", "trace_area_max_rel_delta"),
    ("delta_trace_area_max_rel_delta", "delta", "trace_area_max_rel_delta"),
    ("trace_adjacent_dice_mean_candidate", "candidate", "trace_adjacent_dice_mean"),
    ("trace_adjacent_dice_last_candidate", "candidate", "trace_adjacent_dice_last"),
    ("path_mean_similarity_candidate", "candidate", "path_mean_similarity"),
    ("delta_path_mean_similarity", "delta", "path_mean_similarity"),
)
METHOD_LABELS = {
    "baseline": "KNN mean 安全基准",
    "qreturn": "q_return 永远接受",
    "logistic": "风险门控 Logistic",
    "tree": "风险门控浅层树",
    "oracle": "GT Oracle Gate（仅评价）",
}


def finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite frozen feature {name}: {value!r}")
    return result


def load_inputs(decisions_path: Path, previous_summary_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(decisions_path)
    with previous_summary_path.open("r", encoding="utf-8") as handle:
        previous = json.load(handle)
    rows.sort(key=lambda row: row["target_id"])
    if len(rows) != 100 or len({row["target_id"] for row in rows}) != 100:
        raise ValueError("Protocol requires exactly 100 distinct target-level decisions")
    if previous.get("strongest_knn_baseline") != "knn_mean":
        raise ValueError("KNN-mean baseline is not frozen in the previous validation summary")
    baseline_dice = np.mean([finite(row["baseline_gt_dice"], "baseline_gt_dice") for row in rows])
    qreturn_dice = np.mean([finite(row["qreturn_gt_dice"], "qreturn_gt_dice") for row in rows])
    if not math.isclose(
        baseline_dice,
        previous["selection_results"]["knn_mean"]["selected_dice"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Frozen baseline Dice could not be reproduced")
    if not math.isclose(
        qreturn_dice,
        previous["selection_results"]["q_return"]["selected_dice"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Frozen q_return Dice could not be reproduced")
    for row in rows:
        if not row.get("is_qreturn_selected"):
            raise ValueError(f"Input is not the frozen q_return-max candidate: {row['target_id']}")
        expected_delta = row["qreturn_gt_dice"] - row["baseline_gt_dice"]
        if not math.isclose(float(row["delta_dice"]), expected_delta, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Frozen per-target Dice delta mismatch: {row['target_id']}")
        same_route = row["baseline_route_id"] == row["qreturn_route_id"]
        if bool(row["is_physical_switch"]) == same_route:
            raise ValueError(f"Physical-switch identity mismatch: {row['target_id']}")
    labels = np.asarray([int(row["delta_dice"] <= 0.0) for row in rows])
    if int(np.sum(labels)) != 46 or len(rows) - int(np.sum(labels)) != 54:
        raise ValueError("Frozen beneficial/non-beneficial label counts must reproduce 54/46")
    return rows, previous


def feature_vector(row: dict[str, Any]) -> dict[str, float]:
    candidate = row["candidate_unsupervised_features"]
    difference = row["candidate_minus_baseline_features"]
    result = {}
    for name, source, field in FEATURE_SPEC:
        if name.startswith("gt_") or "evaluation_only" in name:
            raise RuntimeError(f"GT-leaking feature is forbidden: {name}")
        values = candidate if source == "candidate" else difference
        if field not in values:
            raise ValueError(f"Frozen feature missing for {row['target_id']}: {field}")
        result[name] = finite(values[field], name)
    if len(result) != 12:
        raise RuntimeError("The pre-registered feature set must contain exactly 12 variables")
    return result


def model_factory(method: str) -> Any:
    if method == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=SEED,
            ),
        )
    if method == "tree":
        return DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=SEED,
        )
    raise ValueError(f"Only pre-registered logistic/tree gates are permitted: {method}")


def auc_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    if len(np.unique(labels)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(labels, predictions)),
        "pr_auc": float(average_precision_score(labels, predictions)),
    }


def target_disjoint_oof(
    rows: list[dict[str, Any]],
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    groups = np.asarray([row["target_id"] for row in rows], dtype=object)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    risks = np.full(len(rows), np.nan, dtype=np.float64)
    folds = np.zeros(len(rows), dtype=np.int64)
    fold_details = []
    for fold, (train, validation) in enumerate(splitter.split(feature_matrix, labels, groups), 1):
        overlap = set(groups[train]) & set(groups[validation])
        if overlap:
            raise RuntimeError(f"Target-disjoint protocol violated in fold {fold}: {sorted(overlap)}")
        if len(np.unique(labels[train])) < 2:
            raise RuntimeError(f"Fold {fold} lacks both frozen risk-label classes")
        model = model_factory(method)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(feature_matrix[train], labels[train])
        probabilities = model.predict_proba(feature_matrix[validation])[:, 1]
        risks[validation] = probabilities
        folds[validation] = fold
        fold_details.append(
            {
                "fold": fold,
                "training_target_count": len(train),
                "validation_target_count": len(validation),
                "target_overlap_count": 0,
                "training_nonbeneficial_count": int(np.sum(labels[train])),
                "training_beneficial_count": int(len(train) - np.sum(labels[train])),
                "validation_nonbeneficial_count": int(np.sum(labels[validation])),
                "validation_beneficial_count": int(len(validation) - np.sum(labels[validation])),
                "validation_severe_count": sum(
                    rows[index]["delta_dice"] <= -0.10 for index in validation
                ),
                **auc_metrics(labels[validation], probabilities),
            }
        )
    if np.isnan(risks).any() or np.any(folds == 0):
        raise RuntimeError(f"Missing target-level OOF gate predictions: {method}")

    fitted = model_factory(method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.fit(feature_matrix, labels)
    fields = [item[0] for item in FEATURE_SPEC]
    if method == "logistic":
        coefficients = fitted.named_steps["logisticregression"].coef_[0]
        interpretation = {
            "full_validation_standardized_coefficients_diagnostic_only": dict(
                sorted(
                    ((field, float(value)) for field, value in zip(fields, coefficients)),
                    key=lambda item: (-abs(item[1]), item[0]),
                )
            ),
            "full_validation_intercept_diagnostic_only": float(
                fitted.named_steps["logisticregression"].intercept_[0]
            ),
        }
    else:
        interpretation = {
            "full_validation_tree_rules_diagnostic_only": export_text(
                fitted,
                feature_names=fields,
                decimals=6,
            ),
            "full_validation_feature_importance_diagnostic_only": {
                name: float(value)
                for name, value in sorted(
                    zip(fields, fitted.feature_importances_),
                    key=lambda item: (-item[1], item[0]),
                )
                if value > 0
            },
        }
    diagnostic = {
        "model": method,
        "target_count": len(rows),
        "fold_count": 5,
        "seed": SEED,
        "risk_label": "nonbeneficial_switch = 1[delta_dice <= 0]",
        "risk_probability_threshold": RISK_THRESHOLD,
        "features_frozen_before_oof": fields,
        "folds": fold_details,
        "oof_nonbeneficial_risk_auc_secondary_only": auc_metrics(labels, risks),
        "hyperparameters_frozen_before_oof": (
            {"C": 1.0, "class_weight": "balanced", "max_iter": 3000}
            if method == "logistic"
            else {"max_depth": 3, "min_samples_leaf": 10, "class_weight": "balanced"}
        ),
        **interpretation,
    }
    return risks, folds, diagnostic


def bootstrap_target_delta(values: np.ndarray) -> dict[str, Any]:
    generator = np.random.default_rng(SEED)
    indices = generator.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    sampled = np.mean(values[indices], axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    tied = np.isclose(values, 0.0, rtol=0.0, atol=1e-12)
    return {
        "unit": "paired_target",
        "target_count": int(len(values)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": SEED,
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "ci_95": [float(low), float(high)],
        "win_count": int(np.sum((values > 0.0) & ~tied)),
        "tie_count": int(np.sum(tied)),
        "loss_count": int(np.sum((values < 0.0) & ~tied)),
    }


def summarize_method(
    rows: list[dict[str, Any]],
    method: str,
    accept: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    baseline = np.asarray([row["baseline_gt_dice"] for row in rows], dtype=np.float64)
    qreturn = np.asarray([row["qreturn_gt_dice"] for row in rows], dtype=np.float64)
    physical = np.asarray([bool(row["is_physical_switch"]) for row in rows], dtype=bool)
    accepted = np.asarray(accept, dtype=bool) & physical
    selected = np.where(accepted, qreturn, baseline)
    deltas = selected - baseline
    positive = deltas[deltas > 0.0]
    negative = deltas[deltas < 0.0]
    bootstrap = bootstrap_target_delta(deltas)
    summary = {
        "method": method,
        "label": METHOD_LABELS[method],
        "target_count": len(rows),
        "selected_dice": float(np.mean(selected)),
        "baseline_dice": float(np.mean(baseline)),
        "delta_vs_baseline": float(np.mean(deltas)),
        "allowed_switch_target_count": int(np.sum(accepted)),
        "rejected_or_same_route_target_count": int(len(rows) - np.sum(accepted)),
        "correct_switch_count": int(np.sum(accepted & ((qreturn - baseline) > 0.0))),
        "incorrect_switch_count": int(np.sum(accepted & ((qreturn - baseline) <= 0.0))),
        "switch_tie_count": int(
            np.sum(accepted & np.isclose(qreturn - baseline, 0.0, rtol=0.0, atol=1e-12))
        ),
        "severe_degradation_count": int(np.sum(deltas <= -0.10)),
        "catastrophic_degradation_count": int(np.sum(deltas <= -0.20)),
        "total_positive_dice_gain": float(np.sum(positive)),
        "total_negative_dice_loss": float(np.sum(negative)),
        "maximum_single_improvement": float(np.max(positive)) if len(positive) else 0.0,
        "maximum_single_degradation": float(np.min(negative)) if len(negative) else 0.0,
        "win_count": bootstrap["win_count"],
        "tie_count": bootstrap["tie_count"],
        "loss_count": bootstrap["loss_count"],
        "paired_bootstrap_vs_baseline": bootstrap,
    }
    return summary, accepted


def decision_rows(
    rows: list[dict[str, Any]],
    vectors: list[dict[str, float]],
    risks: dict[str, np.ndarray],
    folds: dict[str, np.ndarray],
    accepted: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    results = []
    for index, row in enumerate(rows):
        methods = {}
        for method in ("baseline", "qreturn", "logistic", "tree", "oracle"):
            allow = bool(accepted[method][index])
            selected = "qreturn" if allow else "baseline"
            methods[method] = {
                "allows_physical_switch": allow,
                "selected_route": selected,
                "selected_route_id": row["qreturn_route_id"] if allow else row["baseline_route_id"],
                "selected_gt_dice_evaluation_only": float(
                    row["qreturn_gt_dice"] if allow else row["baseline_gt_dice"]
                ),
                "delta_vs_baseline_evaluation_only": float(row["delta_dice"] if allow else 0.0),
            }
            if method in risks:
                methods[method]["oof_predicted_nonbeneficial_risk"] = float(risks[method][index])
                methods[method]["oof_fold"] = int(folds[method][index])
                methods[method]["risk_threshold_frozen"] = RISK_THRESHOLD
        results.append(
            {
                "target_id": row["target_id"],
                "baseline_route_id": row["baseline_route_id"],
                "baseline_route_mode": row["baseline_route_mode"],
                "baseline_bridge_count": row["baseline_bridge_count"],
                "qreturn_route_id": row["qreturn_route_id"],
                "qreturn_route_mode": row["qreturn_route_mode"],
                "qreturn_bridge_count": row["qreturn_bridge_count"],
                "is_physical_switch": bool(row["is_physical_switch"]),
                "baseline_gt_dice_evaluation_only": float(row["baseline_gt_dice"]),
                "qreturn_gt_dice_evaluation_only": float(row["qreturn_gt_dice"]),
                "switch_delta_dice_evaluation_only": float(row["delta_dice"]),
                "nonbeneficial_training_label_evaluation_only": int(row["delta_dice"] <= 0.0),
                "severe_outcome_evaluation_only": bool(row["delta_dice"] <= -0.10),
                "catastrophic_outcome_evaluation_only": bool(row["delta_dice"] <= -0.20),
                "frozen_unsupervised_features": vectors[index],
                "decisions": methods,
            }
        )
    return results


def catastrophic_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_target = {row["target_id"]: row for row in rows}
    cases = []
    for target in KNOWN_CATASTROPHES:
        if target not in by_target:
            raise ValueError(f"Previously identified catastrophic target missing: {target}")
        row = by_target[target]
        if not row["catastrophic_outcome_evaluation_only"]:
            raise ValueError(f"Previously identified target is no longer catastrophic: {target}")
        cases.append(
            {
                "target_id": target,
                "baseline_dice": row["baseline_gt_dice_evaluation_only"],
                "qreturn_dice": row["qreturn_gt_dice_evaluation_only"],
                "qreturn_delta_dice": row["switch_delta_dice_evaluation_only"],
                "frozen_unsupervised_features": row["frozen_unsupervised_features"],
                "logistic": {
                    "oof_predicted_risk": row["decisions"]["logistic"]["oof_predicted_nonbeneficial_risk"],
                    "oof_fold": row["decisions"]["logistic"]["oof_fold"],
                    "allows_switch": row["decisions"]["logistic"]["allows_physical_switch"],
                    "selected_route": row["decisions"]["logistic"]["selected_route"],
                    "catastrophe_prevented": not row["decisions"]["logistic"]["allows_physical_switch"],
                },
                "tree": {
                    "oof_predicted_risk": row["decisions"]["tree"]["oof_predicted_nonbeneficial_risk"],
                    "oof_fold": row["decisions"]["tree"]["oof_fold"],
                    "allows_switch": row["decisions"]["tree"]["allows_physical_switch"],
                    "selected_route": row["decisions"]["tree"]["selected_route"],
                    "catastrophe_prevented": not row["decisions"]["tree"]["allows_physical_switch"],
                },
            }
        )
    return {
        "case_count": len(cases),
        "cases": cases,
        "logistic_catastrophes_prevented": sum(case["logistic"]["catastrophe_prevented"] for case in cases),
        "tree_catastrophes_prevented": sum(case["tree"]["catastrophe_prevented"] for case in cases),
        "target_disjoint_oof_required": True,
    }


def protocol_record(input_files: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scope": "validation-only; exactly one frozen baseline vs q_return decision per target",
        "target_count": 100,
        "candidate_choice_frozen": {
            "baseline": "previous argmax path_mean_similarity",
            "proposal": "previous argmax q_return",
        },
        "training_label": {
            "definition": "risk = 1[Dice(q_return) - Dice(KNN_mean) <= 0]",
            "beneficial_count": 54,
            "nonbeneficial_count": 46,
            "justification": "the requested severe-only label has only 2 positives and cannot support reliable 5-fold target-level learning",
            "gt_use": "authorized validation-only training label; never an inference feature",
        },
        "features_frozen_before_training": [item[0] for item in FEATURE_SPEC],
        "feature_count": len(FEATURE_SPEC),
        "models_frozen_before_training": {
            "logistic": {
                "standard_scaler": True,
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 3000,
            },
            "tree": {
                "max_depth": 3,
                "min_samples_leaf": 10,
                "class_weight": "balanced",
            },
        },
        "risk_probability_threshold": RISK_THRESHOLD,
        "decision_rule": "accept frozen q_return route only when OOF risk < 0.5 and it is a distinct physical route",
        "cross_validation": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026), grouped by target_id",
        "paired_bootstrap": {"unit": "target", "resamples": BOOTSTRAP_RESAMPLES, "seed": SEED},
        "severe_evaluation_only": "selected_delta_dice <= -0.10",
        "catastrophic_evaluation_only": "selected_delta_dice <= -0.20",
        "input_artifacts": input_files,
        "test_read": False,
        "gpu_used": False,
        "sam3_propagation_rerun": False,
        "threshold_tuned": False,
        "hyperparameter_search": False,
        "candidate_pool_researched": False,
    }


def result_conclusion(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = summaries["baseline"]
    proposal = summaries["qreturn"]
    gates = {method: summaries[method] for method in ("logistic", "tree")}
    successful = [
        method
        for method, result in gates.items()
        if result["selected_dice"] > baseline["selected_dice"]
        and result["catastrophic_degradation_count"] < proposal["catastrophic_degradation_count"]
    ]
    fully_safe = [
        method for method, result in gates.items() if result["catastrophic_degradation_count"] == 0
    ]
    if successful:
        category = "A_direction_supported_validation_only"
        assessment = "至少一个预注册 OOF 风险门控同时提升 KNN baseline Dice 并降低灾难性错误；仍不能宣称 test 泛化。"
        next_experiment = "人工审阅并冻结本轮两路线、12 特征、OOF 风险门控协议后，再决定是否授权一次独立正式评估；本轮不接触 test。"
    elif fully_safe:
        category = "B_safety_without_selected_dice_improvement"
        assessment = "至少一个预注册 OOF 风险门控消除灾难性切换，但未带来稳定的 KNN baseline Dice 提升；传播 proposal 收益仍不足。"
        next_experiment = "保持 KNN 安全基准与现有门控冻结，仅研究更有价值的 validation-only propagation proposal；不读取 test。"
    else:
        category = "C_catastrophic_failures_not_reliably_prevented"
        assessment = "预注册 target-level OOF gate 未能可靠拦截全部已知灾难切换；现有 q_return/q_multi/q_model/轨迹/形态信息不支持安全门控。"
        next_experiment = "暂停继续堆叠风险门控模型，改为研究新的 validation-only 语义漂移检测信号；不读取 test。"
    return {
        "category": category,
        "assessment": assessment,
        "successful_gate_methods": successful,
        "catastrophe_free_gate_methods": fully_safe,
        "next_single_recommended_experiment": next_experiment,
        "validation_only_no_generalization_claim": True,
    }


def number(value: Any, digits: int = 6) -> str:
    return "未定义" if value is None else f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    repo_root: Path,
    protocol: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    model_diagnostics: dict[str, dict[str, Any]],
    cases: dict[str, Any],
    conclusion: dict[str, Any],
) -> None:
    lines = [
        "# C0-256 风险门控换路：validation target-disjoint OOF 反事实实验",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}  ",
        f"> 仓库：`{repo_root}`  ",
        "> 只使用 frozen validation，100 targets 各一条 baseline vs q_return 决策；无 GPU、无 test、无 SAM3 重传播。",
        "",
        "## 1. 研究问题与先验冻结边界",
        "",
        "固定上一轮 `R_base=argmax(path_mean_similarity)` 与 `R_return=argmax(q_return)`；"
        "不重新进行 14 选 1，不修改候选、KNN、B7 或 teacher。模型唯一决策是：接受现有 q_return 换路，"
        "还是回退到现有 KNN mean baseline。",
        "",
        "## 2. 实际输入文件与 SHA256",
        "",
        "| 输入角色 | validation 文件 | SHA256 |",
        "|---|---|---|",
    ]
    for item in protocol["input_artifacts"]:
        lines.append(f"| `{item['role']}` | `{item['path']}` | `{item['sha256']}` |")
    lines.extend(
        [
            "",
            "## 3. 预注册训练标签、12 个特征与固定参数",
            "",
            "目标只提供一个决策样本；有益换路 `ΔDice>0` 共 54 个，无益换路 `ΔDice≤0` 共 46 个。"
            "由于 `ΔDice≤-0.10` 只有两个 target，不能对五折独立划分稳定训练严重风险标签，"
            "因此预先固定训练标签为 **无益换路 `risk=1[ΔDice≤0]`**；严重/灾难错误只做最终评价。",
            "",
            "固定的 12 个不含 target GT 的部署侧输入：",
            "",
        ]
    )
    lines.extend(f"- `{feature}`" for feature in protocol["features_frozen_before_training"])
    lines.extend(
        [
            "",
            "固定模型 A：standard scaler + logistic regression，C=1，class_weight=balanced。",
            "固定模型 B：decision tree，max_depth=3，min_samples_leaf=10，class_weight=balanced。",
            "冻结规则：OOF predicted risk < 0.5 才允许真实路线切换；0.5 为固定默认概率门槛，"
            "没有使用 validation GT 扫阈值，没有手写 q_multi / 面积规则。",
            "",
            "## 4. target-disjoint 五折协议",
            "",
            "`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)`；每 target 只有一条 row，"
            "每个 OOF target 在拟合其预测所用模型时均不参与训练。",
            "",
            "| 模型 | fold | train targets | OOF targets | target overlap | OOF severe targets |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("logistic", "tree"):
        for fold in model_diagnostics[method]["folds"]:
            lines.append(
                f"| `{method}` | {fold['fold']} | {fold['training_target_count']} | "
                f"{fold['validation_target_count']} | {fold['target_overlap_count']} | "
                f"{fold['validation_severe_count']} |"
            )
    lines.extend(
        [
            "",
            "## 5. 主要终点评价：最终 Selected Dice 与负尾部",
            "",
            "| 方法 | Selected Dice | 相对 KNN Δ | 允许换路 | win / tie / loss | 严重退化 | 灾难退化 |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for method in ("baseline", "qreturn", "logistic", "tree", "oracle"):
        item = summaries[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {number(item['selected_dice'])} | "
            f"{number(item['delta_vs_baseline'])} | {item['allowed_switch_target_count']} | "
            f"{item['win_count']} / {item['tie_count']} / {item['loss_count']} | "
            f"{item['severe_degradation_count']} | {item['catastrophic_degradation_count']} |"
        )
    lines.extend(
        [
            "",
            "GT Oracle Gate 只用于理论上限：当且仅当 target GT 显示 `ΔDice>0` 才换路，"
            "不会作为实际门控输入或训练中的候选选择规则。",
            "",
            "## 6. 换路收益与负尾部代价",
            "",
            "| 方法 | 正确换路 | 错误换路 | 总正收益 | 总负损失 | 最大单次改善 | 最大单次退化 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("baseline", "qreturn", "logistic", "tree", "oracle"):
        item = summaries[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {item['correct_switch_count']} | "
            f"{item['incorrect_switch_count']} | {number(item['total_positive_dice_gain'])} | "
            f"{number(item['total_negative_dice_loss'])} | "
            f"{number(item['maximum_single_improvement'])} | "
            f"{number(item['maximum_single_degradation'])} |"
        )
    lines.extend(
        [
            "",
            "## 7. paired target bootstrap：与冻结 KNN baseline 比较",
            "",
            "| 方法 | mean Δ | median Δ | 95% CI | win / tie / loss |",
            "|---|---:|---:|---|---|",
        ]
    )
    for method in ("qreturn", "logistic", "tree", "oracle"):
        item = summaries[method]["paired_bootstrap_vs_baseline"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {number(item['mean_delta'])} | "
            f"{number(item['median_delta'])} | [{number(item['ci_95'][0])}, "
            f"{number(item['ci_95'][1])}] | {item['win_count']} / "
            f"{item['tie_count']} / {item['loss_count']} |"
        )
    lines.extend(
        [
            "",
            "bootstrap 每次只对 100 个 target 配对重采样，10000 次，seed=2026；从未将 1300 条候选作为独立样本。",
            "",
            "## 8. 两个历史灾难 target 的真实 OOF 审计",
            "",
            "| target | q_return ΔDice | logistic OOF risk | logistic 换路 | tree OOF risk | tree 换路 |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for case in cases["cases"]:
        lines.append(
            f"| `{case['target_id']}` | {number(case['qreturn_delta_dice'])} | "
            f"{number(case['logistic']['oof_predicted_risk'])} | "
            f"{case['logistic']['allows_switch']} | "
            f"{number(case['tree']['oof_predicted_risk'])} | "
            f"{case['tree']['allows_switch']} |"
        )
    lines.extend(
        [
            "",
            f"- logistic 实际拦截灾难：{cases['logistic_catastrophes_prevented']}/{cases['case_count']}。",
            f"- shallow tree 实际拦截灾难：{cases['tree_catastrophes_prevented']}/{cases['case_count']}。",
            "- 两个 target 的 OOF 预测均来自不含该 target 的训练 folds；没有任何手写特例。",
            "",
            "## 9. 次要诊断：无益换路标签 AUC",
            "",
            "| 模型 | OOF ROC-AUC | OOF PR-AUC |",
            "|---|---:|---:|",
        ]
    )
    for method in ("logistic", "tree"):
        metric = model_diagnostics[method]["oof_nonbeneficial_risk_auc_secondary_only"]
        lines.append(f"| `{method}` | {number(metric['roc_auc'])} | {number(metric['pr_auc'])} |")
    lines.extend(
        [
            "",
            "上述 AUC 只是次要诊断；是否继续以 Selected Dice、catastrophic cases 和 paired bootstrap 为准。",
            "",
            "## 10. 结论与唯一下一步",
            "",
            f"实验分类：`{conclusion['category']}`。",
            "",
            conclusion["assessment"],
            "",
            f"唯一建议：{conclusion['next_single_recommended_experiment']}",
            "",
            "本轮没有读取或运行 test，没有 GPU/SAM3 propagation，没有重新生成 KNN/路线，没有修改 B7、"
            "训练 SAM3/Student 或训练神经网络；到 validation OOF 实验结束即停止。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--decision-rows",
        type=Path,
        default=root / "work/rerun_c0_256_propagation_risk_analysis/qreturn_switch_analysis.jsonl",
    )
    parser.add_argument(
        "--previous-summary",
        type=Path,
        default=root / "work/rerun_c0_256_propagation_aware_routing/validation_summary.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "work/rerun_c0_256_risk_gated_routing",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reproduction_reports/C0_256_risk_gated_routing_validation.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, _previous = load_inputs(args.decision_rows, args.previous_summary)
    vectors = [feature_vector(row) for row in rows]
    matrix = np.asarray([[vector[name] for name, _, _ in FEATURE_SPEC] for vector in vectors])
    # GT enters here solely as the user-authorized retrospective training label.
    labels = np.asarray([int(row["delta_dice"] <= 0.0) for row in rows], dtype=np.int64)

    risks: dict[str, np.ndarray] = {}
    folds: dict[str, np.ndarray] = {}
    model_diagnostics: dict[str, dict[str, Any]] = {}
    for method in ("logistic", "tree"):
        risks[method], folds[method], model_diagnostics[method] = target_disjoint_oof(
            rows,
            matrix,
            labels,
            method,
        )

    proposals = {
        "baseline": np.zeros(len(rows), dtype=bool),
        "qreturn": np.ones(len(rows), dtype=bool),
        "logistic": risks["logistic"] < RISK_THRESHOLD,
        "tree": risks["tree"] < RISK_THRESHOLD,
        # The only GT-based acceptance rule, explicitly evaluation-only.
        "oracle": np.asarray([row["delta_dice"] > 0.0 for row in rows], dtype=bool),
    }
    summaries: dict[str, dict[str, Any]] = {}
    accepted: dict[str, np.ndarray] = {}
    for method, proposal in proposals.items():
        summaries[method], accepted[method] = summarize_method(rows, method, proposal)
        if method in model_diagnostics:
            summaries[method]["model_diagnostic"] = model_diagnostics[method]

    if summaries["qreturn"]["severe_degradation_count"] != 2:
        raise ValueError("Frozen q_return must reproduce exactly two severe validation failures")
    if summaries["qreturn"]["catastrophic_degradation_count"] != 2:
        raise ValueError("Frozen q_return must reproduce exactly two catastrophic failures")
    oof = decision_rows(rows, vectors, risks, folds, accepted)
    cases = catastrophic_audit(oof)
    conclusion = result_conclusion(summaries)
    protocol = protocol_record(
        [
            artifact_record(args.decision_rows, "frozen_validation_target_decisions"),
            artifact_record(args.previous_summary, "frozen_validation_routing_summary"),
        ]
    )
    paired = {
        method: summary["paired_bootstrap_vs_baseline"]
        for method, summary in summaries.items()
    }
    paired["protocol"] = protocol["paired_bootstrap"]
    write_jsonl(args.output_root / "oof_target_decisions.jsonl", oof)
    write_json(args.output_root / "baseline_summary.json", summaries["baseline"])
    write_json(args.output_root / "qreturn_summary.json", summaries["qreturn"])
    write_json(args.output_root / "logistic_gate_summary.json", summaries["logistic"])
    write_json(args.output_root / "tree_gate_summary.json", summaries["tree"])
    write_json(args.output_root / "oracle_gate_summary.json", summaries["oracle"])
    write_json(args.output_root / "paired_bootstrap.json", paired)
    write_json(args.output_root / "catastrophic_case_audit.json", cases)
    write_json(args.output_root / "validation_protocol.json", protocol)
    write_json(args.output_root / "validation_conclusion.json", conclusion)
    write_report(
        args.report,
        args.repo_root,
        protocol,
        summaries,
        model_diagnostics,
        cases,
        conclusion,
    )
    concise_summaries = {
        method: {
            field: item[field]
            for field in (
                "selected_dice",
                "delta_vs_baseline",
                "allowed_switch_target_count",
                "correct_switch_count",
                "incorrect_switch_count",
                "win_count",
                "tie_count",
                "loss_count",
                "severe_degradation_count",
                "catastrophic_degradation_count",
                "total_positive_dice_gain",
                "total_negative_dice_loss",
            )
        }
        for method, item in summaries.items()
    }
    print(
        json.dumps(
            {
                "method_summaries": concise_summaries,
                "paired_bootstrap": paired,
                "catastrophic_case_audit": cases,
                "conclusion": conclusion,
                "feature_count": len(FEATURE_SPEC),
                "training_labels": {"beneficial": 54, "nonbeneficial": 46},
                "gpu_used": False,
                "test_read": False,
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
