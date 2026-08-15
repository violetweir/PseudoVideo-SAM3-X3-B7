#!/usr/bin/env python3
"""Feature-block ablation for the validation-only pairwise route ranker."""

from __future__ import annotations

import argparse
import csv
import json
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
        "--student-root",
        type=Path,
        default=core.ROOT / "work/kvasir_1pct_anchors/phase1/predictions",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=core.ROOT / "work/kvasir_1pct_anchors/pairwise_ranker_vitb256/feature_ablation",
    )
    parser.add_argument("--tag", default="lora_p491_e20")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--min-pair-gap", type=float, default=0.02)
    parser.add_argument("--profiles", nargs="+", default=None)
    parser.add_argument("--fit-final", action="store_true")
    parser.add_argument("--single-family", default=None)
    parser.add_argument("--min-bridge", type=int, default=3)
    parser.add_argument("--max-bridge", type=int, default=6)
    args = parser.parse_args()

    groups, feature_names, student_indices = core.load_groups(
        args.quality_root,
        args.student_root,
        args.tag,
        min_bridge=args.min_bridge,
        max_bridge=args.max_bridge,
        size=256,
        root=args.root,
        families=(args.single_family,) if args.single_family else None,
        fixed_family=args.single_family or core.FIXED_FAMILY,
    )
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    target_ids = sorted(groups)
    folds = core.make_folds(target_ids, 5, args.seed)

    route_prior_names = {
        "bridge_scaled",
        "is_fixed_b6",
        *(name for name in feature_names if name.startswith("family::")),
    }
    consensus_names = {"q_multi", "q_multi_rank"}
    relative_names = {
        "fixed_mask_dice",
        "log_area_ratio_to_fixed",
        "centroid_distance_to_fixed",
        "delta_q_return_to_fixed",
        "delta_q_multi_to_fixed",
        "delta_sam_score_to_fixed",
    }
    mask_geometry_names = {"mask_area", "mask_centroid_x", "mask_centroid_y"}
    student_names = {feature_names[index] for index in student_indices}
    internal_names = set(feature_names) - consensus_names - relative_names - mask_geometry_names - student_names

    feature_sets = {
        "route_prior": route_prior_names,
        "internal_quality": internal_names,
        "internal_plus_consensus": internal_names | consensus_names,
        "quality_plus_relative": set(feature_names) - student_names,
        "quality_plus_qmodel": set(feature_names),
    }
    if args.profiles:
        unknown = sorted(set(args.profiles) - set(feature_sets))
        if unknown:
            raise ValueError(f"Unknown profiles: {unknown}")
        feature_sets = {name: feature_sets[name] for name in args.profiles}
    feature_indices = {
        label: [name_to_index[name] for name in feature_names if name in names]
        for label, names in feature_sets.items()
    }

    choices = {label: {} for label in feature_sets}
    choices.update({f"{label}_fallback": {} for label in feature_sets})
    switches = {label: {} for label in choices}
    fold_reports = []
    for fold_index, heldout_ids in enumerate(folds):
        heldout = set(heldout_ids)
        train_groups = [groups[target_id] for target_id in target_ids if target_id not in heldout]
        fold_report = {"fold": fold_index, "heldout_targets": heldout_ids, "models": {}}
        print(f"outer fold {fold_index + 1}/5", flush=True)
        for label, indices in feature_indices.items():
            l2, threshold, tuning = core.tune_ranker(
                train_groups,
                indices,
                inner_folds=4,
                seed=args.seed + 1000 * (fold_index + 1),
                min_pair_gap=args.min_pair_gap,
            )
            ranker = core.PairwiseRanker.fit(train_groups, indices, l2, args.min_pair_gap)
            weight_order = np.argsort(np.abs(ranker.weights))[::-1][:10]
            fold_report["models"][label] = {
                "l2": l2,
                "fallback_probability": threshold,
                "optimizer_success": ranker.success,
                "top_standardized_weights": [
                    {
                        "feature": feature_names[indices[index]],
                        "weight": float(ranker.weights[index]),
                    }
                    for index in weight_order
                ],
                "tuning_best_key": tuning["best_key"],
            }
            for target_id in heldout_ids:
                group = groups[target_id]
                chosen, _, switched = core.choose_candidate(group, ranker.score)
                choices[label][target_id] = chosen
                switches[label][target_id] = switched
                fallback_label = f"{label}_fallback"
                chosen, _, switched = core.choose_candidate(group, ranker.score, threshold)
                choices[fallback_label][target_id] = chosen
                switches[fallback_label][target_id] = switched
            print(
                f"  {label}: features={len(indices)} l2={l2:g} fallback={threshold:.2f}",
                flush=True,
            )
        fold_reports.append(fold_report)

    summary = {}
    for index, (label, selected) in enumerate(choices.items()):
        summary[label] = core.evaluate_choices(selected, groups, args.seed + index)
        summary[label]["switches_from_fixed"] = int(sum(switches[label].values()))

    args.output_root.mkdir(parents=True, exist_ok=True)
    final_models = {}
    if args.fit_final:
        all_groups = [groups[target_id] for target_id in target_ids]
        for profile_index, (label, indices) in enumerate(feature_indices.items()):
            l2, threshold, tuning = core.tune_ranker(
                all_groups,
                indices,
                inner_folds=5,
                seed=args.seed + 9000 + 100 * profile_index,
                min_pair_gap=args.min_pair_gap,
            )
            ranker = core.PairwiseRanker.fit(all_groups, indices, l2, args.min_pair_gap)
            final_models[label] = {
                "checkpoint_tag": args.tag,
                "candidate_pool": (
                    f"single_family_{args.single_family}_b{args.min_bridge}_b{args.max_bridge}"
                    if args.single_family
                    else "P2_b3_b6_11_families"
                ),
                "train_split": "validation_all_100_after_nested_oof_audit",
                "test_loaded": False,
                "seed": args.seed,
                "min_pair_gap": args.min_pair_gap,
                "l2": l2,
                "fallback_probability_reference_only": threshold,
                "fallback_enabled": False,
                "optimizer_success": ranker.success,
                "feature_names": [feature_names[index] for index in indices],
                "standardizer_mean": ranker.scaler.mean.tolist(),
                "standardizer_scale": ranker.scaler.scale.tolist(),
                "weights": ranker.weights.tolist(),
                "tuning_best_key": tuning["best_key"],
            }
            model_path = args.output_root / f"frozen_{label}_{args.tag}.json"
            model_path.write_text(
                json.dumps(final_models[label], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"saved frozen model: {model_path}", flush=True)

    report = {
        "protocol": {
            "split": "validation_only_oof",
            "test_loaded": False,
            "pool": "P2_b3_b6",
            "seed": args.seed,
            "outer_folds": 5,
            "inner_folds": 4,
            "min_pair_gap": args.min_pair_gap,
        },
        "feature_sets": {
            label: [feature_names[index] for index in indices]
            for label, indices in feature_indices.items()
        },
        "summary": summary,
        "folds": fold_reports,
        "final_models": final_models,
    }
    json_path = args.output_root / f"pairwise_feature_ablation_{args.tag}.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = args.output_root / f"pairwise_feature_ablation_{args.tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["target_id", "config", "selected_dice", "fixed_b6_dice", "delta", "family", "bridge", "route_id", "switched"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target_id in target_ids:
            for label, selected in choices.items():
                candidate = selected[target_id]
                fixed = groups[target_id].fixed
                writer.writerow(
                    {
                        "target_id": target_id,
                        "config": label,
                        "selected_dice": candidate.dice,
                        "fixed_b6_dice": fixed.dice,
                        "delta": candidate.dice - fixed.dice,
                        "family": candidate.family,
                        "bridge": candidate.bridge,
                        "route_id": candidate.route_id,
                        "switched": switches[label][target_id],
                    }
                )

    lines = [
        f"# Pairwise ranker feature ablation ({args.tag}, validation OOF)",
        "",
        (
            f"Single fixed family {args.single_family} at b{args.min_bridge}-b{args.max_bridge} "
            f"({args.max_bridge - args.min_bridge + 1} candidates per target). Test was not loaded."
            if args.single_family
            else "P2 contains all 11 route families at b3-b6 (44 candidates per target). Test was not loaded."
        ),
        "",
        "| Feature block | Dice | Delta vs fixed b6 | 95% CI | Trimmed delta | W/T/L | Cat. regressions | Switches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, values in summary.items():
        ci = values["bootstrap_95_ci"]
        lines.append(
            f"| {label} | {values['selected_dice']:.6f} | {values['mean_delta_vs_fixed']:+.6f} "
            f"| [{ci[0]:+.6f}, {ci[1]:+.6f}] | {values['trimmed_mean_delta_10pct']:+.6f} "
            f"| {values['wins']}/{values['ties']}/{values['losses']} "
            f"| {values['catastrophic_regressions_delta_lt_minus_0.05']} "
            f"| {values['switches_from_fixed']} |"
        )
    lines.extend(["", "Feature progression: route identity/bridge -> internal propagation quality -> candidate consensus -> mask-relative-to-b6 -> student agreement."])
    md_path = args.output_root / f"pairwise_feature_ablation_{args.tag}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"PAIRWISE_FEATURE_ABLATION_DONE {md_path}", flush=True)


if __name__ == "__main__":
    main()
