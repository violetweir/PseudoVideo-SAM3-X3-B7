#!/usr/bin/env python3
"""Apply a frozen pairwise route ranker without fitting or calibration."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

import run_pairwise_route_ranker_vitb256 as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=core.ROOT)
    parser.add_argument(
        "--quality-root",
        type=Path,
        default=core.ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_vitb256",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=core.ROOT
        / "work/kvasir_1pct_anchors/pairwise_ranker_vitb256/frozen_internal_consensus"
        / "frozen_internal_plus_consensus_lora_p491_e20.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=core.ROOT / "work/kvasir_1pct_anchors/pairwise_ranker_vitb256/frozen_test_diagnostic",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--tag", default="lora_p491_e20")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--use-model-fallback", action="store_true")
    parser.add_argument("--evaluation-role", default=None)
    args = parser.parse_args()

    model = json.loads(args.model.read_text())
    if model["checkpoint_tag"] != args.tag:
        raise RuntimeError(f"Model tag {model['checkpoint_tag']} != requested tag {args.tag}")
    pool = model["candidate_pool"]
    single_prefix = "single_family_"
    if pool == "P2_b3_b6_11_families":
        families = None
        fixed_family = core.FIXED_FAMILY
        min_bridge, max_bridge = 3, 6
    elif pool.startswith(single_prefix):
        body = pool[len(single_prefix):]
        family, min_text, max_text = body.rsplit("_b", 2)
        families = (family,)
        fixed_family = family
        min_bridge, max_bridge = int(min_text), int(max_text)
    else:
        raise RuntimeError(f"Unsupported candidate pool: {model['candidate_pool']}")
    if model.get("fallback_enabled"):
        raise RuntimeError("This evaluator is locked to the no-fallback frozen configuration")
    fallback_threshold = float(model["fallback_probability_reference_only"])

    groups, all_feature_names, _ = core.load_groups(
        args.quality_root,
        args.root / "work/kvasir_1pct_anchors/phase1/predictions",
        args.tag,
        min_bridge=min_bridge,
        max_bridge=max_bridge,
        size=256,
        root=args.root,
        split=args.split,
        include_student=False,
        families=families,
        fixed_family=fixed_family,
    )
    index_by_name = {name: index for index, name in enumerate(all_feature_names)}
    missing = [name for name in model["feature_names"] if name not in index_by_name]
    if missing:
        raise RuntimeError(f"Frozen model features missing from evaluator: {missing}")
    feature_indices = [index_by_name[name] for name in model["feature_names"]]
    mean = np.asarray(model["standardizer_mean"], dtype=np.float64)
    scale = np.asarray(model["standardizer_scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if not (len(feature_indices) == len(mean) == len(scale) == len(weights)):
        raise RuntimeError("Frozen model vector lengths do not match")
    if not np.isfinite(np.concatenate([mean, scale, weights])).all() or np.any(scale <= 0):
        raise RuntimeError("Frozen model contains invalid numeric parameters")

    def score(candidate: core.Candidate) -> float:
        values = candidate.features[feature_indices]
        standardized = np.clip((values - mean) / scale, -8.0, 8.0)
        return float(standardized @ weights)

    selected = {}
    scores = {}
    raw_scores = {}
    raw_selected = {}
    fixed_scores = {}
    switch_probabilities = {}
    route_histogram = Counter()
    for target_id, group in sorted(groups.items()):
        chosen, chosen_score, _ = core.choose_candidate(group, score)
        raw_selected[target_id] = chosen
        raw_scores[target_id] = chosen_score
        fixed_score = score(group.fixed)
        fixed_scores[target_id] = fixed_score
        probability = float(core.expit(chosen_score - fixed_score))
        switch_probabilities[target_id] = probability
        if (
            args.use_model_fallback
            and chosen.route_id != group.fixed.route_id
            and probability <= fallback_threshold
        ):
            chosen = group.fixed
            chosen_score = fixed_score
        selected[target_id] = chosen
        scores[target_id] = chosen_score
        route_histogram[f"{chosen.family}:b{chosen.bridge}"] += 1

    summary = core.evaluate_choices(selected, groups, args.seed)
    summary["switches_from_fixed"] = int(
        sum(selected[target_id].route_id != groups[target_id].fixed.route_id for target_id in groups)
    )
    summary["route_histogram"] = dict(sorted(route_histogram.items()))
    report = {
        "protocol": {
            "split": args.split,
            "role": args.evaluation_role
            or ("historical_test_diagnostic" if args.split == "test" else "frozen_external_evaluation"),
            "fit_or_calibration_performed": False,
            "student_loaded": False,
            "fallback_enabled": args.use_model_fallback,
            "fallback_probability": fallback_threshold if args.use_model_fallback else None,
            "checkpoint_tag": args.tag,
            "candidate_pool": pool,
            "model_path": str(args.model),
        },
        "summary": summary,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    suffix = "_validation_fallback" if args.use_model_fallback else ""
    stem = f"frozen_pairwise_{args.tag}_{args.split}{suffix}"
    (args.output_root / f"{stem}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_root / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "target_id",
            "selected_dice",
            "fixed_b6_dice",
            "delta",
            "oracle_dice",
            "family",
            "bridge",
            "route_id",
            "score",
            "raw_selected_family",
            "raw_selected_bridge",
            "fixed_score",
            "score_margin_over_fixed",
            "switch_probability_over_fixed",
            "fallback_applied",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target_id in sorted(groups):
            chosen = selected[target_id]
            fixed = groups[target_id].fixed
            writer.writerow(
                {
                    "target_id": target_id,
                    "selected_dice": chosen.dice,
                    "fixed_b6_dice": fixed.dice,
                    "delta": chosen.dice - fixed.dice,
                    "oracle_dice": groups[target_id].oracle.dice,
                    "family": chosen.family,
                    "bridge": chosen.bridge,
                    "route_id": chosen.route_id,
                    "score": scores[target_id],
                    "raw_selected_family": raw_selected[target_id].family,
                    "raw_selected_bridge": raw_selected[target_id].bridge,
                    "fixed_score": fixed_scores[target_id],
                    "score_margin_over_fixed": raw_scores[target_id] - fixed_scores[target_id],
                    "switch_probability_over_fixed": switch_probabilities[target_id],
                    "fallback_applied": chosen.route_id != raw_selected[target_id].route_id,
                }
            )
    ci = summary["bootstrap_95_ci"]
    lines = [
        f"# Frozen pairwise ranker {args.split} diagnostic ({args.tag})",
        "",
        "The frozen validation-trained model was applied without fitting, calibration, or student predictions.",
        "",
        f"- selected Dice: **{summary['selected_dice']:.6f}**",
        f"- fixed b6 Dice: **{summary['fixed_b6_dice']:.6f}**",
        f"- delta: **{summary['mean_delta_vs_fixed']:+.6f}**; bootstrap 95% CI [{ci[0]:+.6f}, {ci[1]:+.6f}]",
        f"- oracle: {summary['oracle_dice']:.6f}; gap {summary['oracle_gap']:.6f}",
        f"- wins/ties/losses: {summary['wins']}/{summary['ties']}/{summary['losses']}",
        f"- catastrophic regressions: {summary['catastrophic_regressions_delta_lt_minus_0.05']}",
        f"- trimmed mean delta (10%): {summary['trimmed_mean_delta_10pct']:+.6f}",
        f"- largest gain/net gain: {summary['max_gain_over_net_gain']}",
        f"- validation-derived fallback: {args.use_model_fallback}"
        + (f" at probability {fallback_threshold:.2f}" if args.use_model_fallback else ""),
        "",
        "For Kvasir test this remains a historical diagnostic because earlier selector experiments already inspected the split.",
    ]
    (args.output_root / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"FROZEN_PAIRWISE_EVAL_DONE {args.output_root / stem}", flush=True)


if __name__ == "__main__":
    main()
