#!/usr/bin/env python3
"""Build and audit the validation-only, strictly @256 Round-2C anchor study."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata, spearmanr


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
PROTOCOL = ROOT / "work/kvasir_1pct_anchors/protocol"
ROUND2A = ROOT / "work/rerun_c0_256_round2a_fixed_knn_e33"
BASE = ROOT / "work/rerun_c0_256_sam3knn_s256_base"
BASE_FEATURES = (
    ROOT
    / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/features"
    / "sam3_base_s256_features.npz"
)
BASE_MODES = (
    "sam3enc_anchor_conditioned_target_pooling",
    "sam3enc_anchor_conditioned_patch_correspondence",
)
ALL_ANCHORS_MODE = "round2c_all_anchors_b0"
STAGE1_PATH = ROOT / "scripts/stage1_feature_knn_routes.py"
SPEC = importlib.util.spec_from_file_location("round2c_stage1_routes", STAGE1_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {STAGE1_PATH}")
stage1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage1
SPEC.loader.exec_module(stage1)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_mask(path: str | Path, grid: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize((grid, grid), Image.Resampling.NEAREST)
    return (np.asarray(image) > 127).reshape(-1)


def auc(heatmap: torch.Tensor, foreground: np.ndarray) -> float | None:
    foreground_count = int(foreground.sum())
    background_count = int(foreground.size - foreground_count)
    if foreground_count == 0 or background_count == 0:
        return None
    values = heatmap.detach().float().cpu().numpy()
    if not np.isfinite(values).all():
        return None
    ranks = rankdata(values, method="average")
    positive_rank_sum = float(ranks[foreground].sum())
    statistic = positive_rank_sum - foreground_count * (foreground_count + 1) / 2
    return float(statistic / (foreground_count * background_count))


def unit_mean(tokens: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(tokens.mean(dim=0), dim=0)


def normalized_score(value: float | None, fallback: float) -> tuple[float, bool]:
    if value is None or not math.isfinite(value):
        return float((fallback + 1.0) / 2.0), True
    return float(value), False


def variant_definitions() -> list[dict[str, str]]:
    variants = [
        {
            "name": "V0_target_pooling",
            "source": "x3_best",
            "score_key": "baseline_target_pooling",
            "family": "baseline",
        },
        {
            "name": "V0_patch_correspondence",
            "source": "x3_best",
            "score_key": "baseline_patch_correspondence",
            "family": "baseline",
        },
    ]
    for source in ("round1", "x3_best"):
        variants.append(
            {
                "name": f"V1_lesion_mean__{source}",
                "source": source,
                "score_key": "lesion_mean_cosine",
                "family": "lesion_mean",
            }
        )
        for aggregate in ("prototype", "token_topk"):
            variants.extend(
                [
                    {
                        "name": f"V2_forward_{aggregate}__{source}",
                        "source": source,
                        "score_key": f"forward_auc_{aggregate}",
                        "family": "forward",
                    },
                    {
                        "name": f"V3_reverse_{aggregate}__{source}",
                        "source": source,
                        "score_key": f"reverse_auc_{aggregate}",
                        "family": "reverse",
                    },
                    {
                        "name": f"V4_bidirectional_{aggregate}__{source}",
                        "source": source,
                        "score_key": f"bidirectional_auc_{aggregate}",
                        "family": "bidirectional",
                    },
                ]
            )
    return variants


def validate_protocol(args: argparse.Namespace) -> None:
    if args.split != "validation":
        raise SystemExit("Round-2C currently permits validation only")
    if args.feature_size != 256 or args.canvas != 256:
        raise SystemExit("Round-2C requires both --feature-size 256 and --canvas 256")


def load_target_masks(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    round1_rows = read_jsonl(args.round1_masks)
    prediction_rows = read_jsonl(args.student_predictions)
    return {
        "round1": {row["target_id"]: row["pseudo_mask_path"] for row in round1_rows},
        "x3_best": {
            row["merged_id"]: row["student_binary_mask"] for row in prediction_rows
        },
    }


def prepare(args: argparse.Namespace) -> None:
    validate_protocol(args)
    records = read_jsonl(args.protocol_root / "merged_manifest.jsonl")
    support = read_jsonl(args.protocol_root / "support_manifest.jsonl")
    anchors = stage1.t21.human_pool(support, 512)
    if len(anchors) != 8:
        raise RuntimeError(f"Expected the frozen eight human anchors, found {len(anchors)}")
    record_by_id = {row["merged_id"]: row for row in records}
    manifest_index = {row["merged_id"]: index for index, row in enumerate(records)}
    target_masks = load_target_masks(args)

    cache = np.load(args.patch_cache)
    if int(cache["feature_size"]) != 256:
        raise RuntimeError(f"Patch cache is not @256: {args.patch_cache}")
    grid = int(cache["grid"])
    cache_ids = cache["ids"].tolist()
    patches = cache["patches"]
    patch_index = {merged_id: position for position, merged_id in enumerate(cache_ids)}
    targets = sorted(
        (
            row
            for row in records
            if row["split"] == "validation" and row["merged_id"] in patch_index
        ),
        key=lambda row: row["merged_id"],
    )
    if args.max_targets is not None:
        targets = targets[: args.max_targets]
    if not targets:
        raise RuntimeError("No validation targets found in the @256 patch cache")

    baseline_features = np.load(args.base_features)
    if int(baseline_features["feature_size"]) != 256:
        raise RuntimeError(f"Baseline topology feature cache is not @256: {args.base_features}")
    baseline_anchor_index = {
        merged_id: index for index, merged_id in enumerate(baseline_features["anchor_ids"].tolist())
    }
    baseline_target_scores = baseline_features["cond_target"]
    baseline_patch_scores = baseline_features["cond_correspondence"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anchor_cache: dict[str, tuple[torch.Tensor, np.ndarray, torch.Tensor]] = {}
    for anchor in anchors:
        anchor_id = anchor["anchor_id"]
        foreground = load_mask(anchor["mask_path"], grid)
        if not foreground.any():
            raise RuntimeError(f"Anchor GT disappears at the @256 token grid: {anchor_id}")
        tokens = torch.from_numpy(patches[patch_index[anchor_id]].astype(np.float32)).to(device)
        anchor_cache[anchor_id] = (tokens, foreground, unit_mean(tokens[foreground]))

    score_rows: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = []
    with torch.inference_mode():
        for target_no, target in enumerate(targets, start=1):
            target_id = target["merged_id"]
            target_tokens = torch.from_numpy(
                patches[patch_index[target_id]].astype(np.float32)
            ).to(device)
            masks = {}
            for source, mapping in target_masks.items():
                if target_id not in mapping:
                    raise RuntimeError(f"Missing {source} pseudo-mask for {target_id}")
                masks[source] = load_mask(mapping[target_id], grid)

            for anchor in anchors:
                anchor_id = anchor["anchor_id"]
                anchor_tokens, anchor_mask, anchor_prototype = anchor_cache[anchor_id]
                pairwise = anchor_tokens @ target_tokens.T
                anchor_fg = torch.as_tensor(anchor_mask, device=device)
                pair_fg = pairwise[anchor_fg]
                forward_prototype_heatmap = target_tokens @ anchor_prototype
                forward_topk_heatmap = torch.topk(
                    pair_fg, k=min(args.match_topk, pair_fg.shape[0]), dim=0
                ).values.mean(dim=0)

                anchor_pos = baseline_anchor_index[anchor_id]
                target_pos = manifest_index[target_id]
                base_target_score = float(baseline_target_scores[anchor_pos, target_pos])
                base_patch_score = float(baseline_patch_scores[anchor_pos, target_pos])
                route = stage1.make_route(
                    target,
                    0,
                    anchor,
                    [],
                    (base_target_score, base_target_score),
                    records,
                )
                route["round2c_feature_size"] = 256
                route["round2c_anchor_candidates"] = len(anchors)
                routes.append(route)

                for source, target_mask in masks.items():
                    target_fg_count = int(target_mask.sum())
                    if target_fg_count:
                        target_fg = torch.as_tensor(target_mask, device=device)
                        target_prototype = unit_mean(target_tokens[target_fg])
                        reverse_prototype_heatmap = anchor_tokens @ target_prototype
                        pair_target_fg = pairwise[:, target_fg]
                        reverse_topk_heatmap = torch.topk(
                            pair_target_fg,
                            k=min(args.match_topk, pair_target_fg.shape[1]),
                            dim=1,
                        ).values.mean(dim=1)
                        lesion_cosine = float(torch.dot(anchor_prototype, target_prototype).item())
                    else:
                        target_prototype = unit_mean(target_tokens)
                        reverse_prototype_heatmap = anchor_tokens @ target_prototype
                        reverse_topk_heatmap = reverse_prototype_heatmap
                        lesion_cosine = float(torch.dot(anchor_prototype, target_prototype).item())

                    fallback_count = 0
                    row: dict[str, Any] = {
                        "target_id": target_id,
                        "anchor_id": anchor_id,
                        "route_id": route["route_id"],
                        "target_mask_source": source,
                        "target_mask_path": target_masks[source][target_id],
                        "feature_source": "sam3_base",
                        "feature_size": 256,
                        "propagation_canvas": 256,
                        "patch_grid": grid,
                        "match_topk": args.match_topk,
                        "anchor_fg_patch_count": int(anchor_mask.sum()),
                        "target_fg_patch_count": target_fg_count,
                        "baseline_target_pooling": base_target_score,
                        "baseline_patch_correspondence": base_patch_score,
                        "lesion_mean_cosine": lesion_cosine,
                        "target_gt_used_for_search_or_inference": False,
                    }
                    for aggregate, forward_heatmap, reverse_heatmap in (
                        ("prototype", forward_prototype_heatmap, reverse_prototype_heatmap),
                        ("token_topk", forward_topk_heatmap, reverse_topk_heatmap),
                    ):
                        forward_score, forward_fallback = normalized_score(
                            auc(forward_heatmap, target_mask), lesion_cosine
                        )
                        reverse_score, reverse_fallback = normalized_score(
                            auc(reverse_heatmap, anchor_mask), lesion_cosine
                        )
                        fallback_count += int(forward_fallback) + int(reverse_fallback)
                        row[f"forward_auc_{aggregate}"] = forward_score
                        row[f"reverse_auc_{aggregate}"] = reverse_score
                        row[f"bidirectional_auc_{aggregate}"] = math.sqrt(
                            max(forward_score, 0.0) * max(reverse_score, 0.0)
                        )
                    row["auc_fallback_count"] = fallback_count
                    score_rows.append(row)

            if target_no == 1 or target_no % 10 == 0 or target_no == len(targets):
                print(f"[round2c] @256 bidirectional anchor scoring {target_no}/{len(targets)}", flush=True)

    routes.sort(key=lambda row: (row["target_id"], row["anchor_id"]))
    score_rows.sort(
        key=lambda row: (row["target_id"], row["anchor_id"], row["target_mask_source"])
    )
    quality_root = args.phase / "quality_root"
    route_path = quality_root / ALL_ANCHORS_MODE / "validation_pool0_stage1/routes.jsonl"
    write_jsonl(route_path, routes)
    score_path = args.phase / "anchor_scores_validation.jsonl"
    write_jsonl(score_path, score_rows)

    existing: dict[str, dict[str, Any]] = {}
    desired_ids = {row["route_id"] for row in routes}
    for mode in BASE_MODES:
        path = ROUND2A / "quality_root" / mode / "propagation_quality_validation/propagation_quality.jsonl"
        for row in read_jsonl(path):
            if (
                int(row.get("bridge_count", -1)) == 0
                and row.get("status") == "success"
                and row["route_id"] in desired_ids
            ):
                existing[row["route_id"]] = row

    result_path = (
        quality_root
        / ALL_ANCHORS_MODE
        / "propagation_quality_validation/propagation_quality.jsonl"
    )
    for row in read_jsonl(result_path):
        if row.get("status") == "success" and row["route_id"] in desired_ids:
            existing[row["route_id"]] = row
    seeded = [existing[route["route_id"]] for route in routes if route["route_id"] in existing]
    write_jsonl(result_path, seeded)

    metadata = {
        "status": "prepared",
        "feature_size": 256,
        "propagation_canvas": 256,
        "encoder": "sam3_base",
        "teacher": "sam3_e33",
        "split": "validation",
        "targets": len(targets),
        "anchors": len(anchors),
        "routes": len(routes),
        "score_rows": len(score_rows),
        "pseudo_mask_sources": sorted(target_masks),
        "seeded_e33_routes": len(seeded),
        "pending_e33_routes": len(routes) - len(seeded),
        "patch_grid": grid,
        "match_topk": args.match_topk,
        "route_path": str(route_path),
        "scores_path": str(score_path),
        "variants": variant_definitions(),
        "test_used": False,
        "train_propagation": False,
        "new_training": False,
    }
    (args.phase / "prepare_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


def rank_candidates(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (float(row[score_key]), row["anchor_id"]), reverse=True)


def bootstrap_delta(values: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = values - reference
    random = np.random.default_rng(2026)
    sampled = random.integers(0, len(delta), size=(2000, len(delta)))
    means = delta[sampled].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def summarize(args: argparse.Namespace) -> None:
    validate_protocol(args)
    score_rows = read_jsonl(args.phase / "anchor_scores_validation.jsonl")
    quality_path = (
        args.phase
        / "quality_root"
        / ALL_ANCHORS_MODE
        / "propagation_quality_validation/propagation_quality.jsonl"
    )
    quality = {
        row["route_id"]: row for row in read_jsonl(quality_path) if row.get("status") == "success"
    }
    if not quality:
        raise RuntimeError(f"No completed SAM3-e33 validation routes: {quality_path}")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        result = quality.get(row["route_id"])
        if result is not None:
            joined = {**row, "gt_dice_evaluation_only": float(result["gt_dice_evaluation_only"])}
            groups[(row["target_mask_source"], row["target_id"])].append(joined)

    target_ids = sorted({target_id for _, target_id in groups})
    all_anchors = [
        max(row["gt_dice_evaluation_only"] for row in groups[("x3_best", target_id)])
        for target_id in target_ids
    ]
    selections: dict[str, dict[str, Any]] = {}
    top3_root = args.phase / "top3_quality_root"
    variants = variant_definitions()
    for variant in variants:
        chosen_rows: list[dict[str, Any]] = []
        top3_rows: list[dict[str, Any]] = []
        chosen_dice: list[float] = []
        top3_oracle: list[float] = []
        all_scores: list[float] = []
        all_dice: list[float] = []
        target_correlations: list[float] = []
        top1_oracle_hits = 0
        top3_oracle_hits = 0

        for target_id in target_ids:
            candidates = groups[(variant["source"], target_id)]
            if len(candidates) != 8:
                raise RuntimeError(
                    f"Incomplete anchor audit for {target_id}: {len(candidates)} successful routes"
                )
            ranked = rank_candidates(candidates, variant["score_key"])
            best_gt = max(row["gt_dice_evaluation_only"] for row in candidates)
            selected = ranked[0]
            chosen_rows.append(selected)
            chosen_dice.append(selected["gt_dice_evaluation_only"])
            best_top3 = max(row["gt_dice_evaluation_only"] for row in ranked[:3])
            top3_oracle.append(best_top3)
            top1_oracle_hits += int(math.isclose(selected["gt_dice_evaluation_only"], best_gt, abs_tol=1e-10))
            top3_oracle_hits += int(math.isclose(best_top3, best_gt, abs_tol=1e-10))
            local_scores = [float(row[variant["score_key"]]) for row in candidates]
            local_dice = [float(row["gt_dice_evaluation_only"]) for row in candidates]
            all_scores.extend(local_scores)
            all_dice.extend(local_dice)
            correlation = spearmanr(local_scores, local_dice).statistic
            if np.isfinite(correlation):
                target_correlations.append(float(correlation))
            top3_rows.extend(quality[row["route_id"]] for row in ranked[:3])

        mode = f"round2c_{variant['name']}_top3"
        write_jsonl(
            top3_root / mode / "propagation_quality_validation/propagation_quality.jsonl",
            top3_rows,
        )
        global_correlation = spearmanr(all_scores, all_dice).statistic
        selections[variant["name"]] = {
            **variant,
            "mode": mode,
            "targets": len(target_ids),
            "top1_b0_dice": float(np.mean(chosen_dice)),
            "top3_b0_oracle": float(np.mean(top3_oracle)),
            "all8_b0_oracle": float(np.mean(all_anchors)),
            "top1_oracle_hit_rate": top1_oracle_hits / len(target_ids),
            "top3_oracle_hit_rate": top3_oracle_hits / len(target_ids),
            "global_spearman": float(global_correlation) if np.isfinite(global_correlation) else None,
            "mean_target_spearman": float(np.mean(target_correlations)) if target_correlations else None,
            "selected_route_ids": [row["route_id"] for row in chosen_rows],
            "selected_anchor_ids": [row["anchor_id"] for row in chosen_rows],
            "selected_dice": chosen_dice,
        }

    target_baseline = selections["V0_target_pooling"]
    for result in selections.values():
        result["anchor_change_vs_target_pooling"] = float(
            np.mean(
                [
                    selected != baseline
                    for selected, baseline in zip(
                        result["selected_anchor_ids"], target_baseline["selected_anchor_ids"]
                    )
                ]
            )
        )
        result["paired_delta_vs_target_pooling"] = bootstrap_delta(
            np.asarray(result["selected_dice"]), np.asarray(target_baseline["selected_dice"])
        )
        if result["family"] != "baseline":
            lesion_baseline = selections[f"V1_lesion_mean__{result['source']}"]
            result["paired_delta_vs_lesion_mean"] = bootstrap_delta(
                np.asarray(result["selected_dice"]), np.asarray(lesion_baseline["selected_dice"])
            )

    result = {
        "status": "complete",
        "feature_size": 256,
        "propagation_canvas": 256,
        "teacher": "sam3_e33",
        "split": "validation",
        "targets": len(target_ids),
        "all8_b0_oracle": float(np.mean(all_anchors)),
        "variants": selections,
        "top3_quality_root": str(top3_root),
        "test_used": False,
        "train_propagation": False,
        "new_training": False,
    }
    summary_path = args.phase / "validation_anchor_summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    condensed = {
        name: {
            "top1_b0_dice": round(item["top1_b0_dice"], 6),
            "top3_b0_oracle": round(item["top3_b0_oracle"], 6),
            "mean_target_spearman": item["mean_target_spearman"],
        }
        for name, item in selections.items()
    }
    print(json.dumps({"summary": str(summary_path), "variants": condensed}, indent=2), flush=True)


def self_test() -> None:
    positives = np.asarray([True, True, False, False])
    if not math.isclose(auc(torch.tensor([0.9, 0.8, 0.2, 0.1]), positives), 1.0):
        raise AssertionError("Perfect foreground AUC should be 1")
    if not math.isclose(auc(torch.tensor([1.0, 1.0, 1.0, 1.0]), positives), 0.5):
        raise AssertionError("Tied foreground AUC should be 0.5")
    if auc(torch.tensor([0.0, 1.0]), np.asarray([False, False])) is not None:
        raise AssertionError("Missing foreground must trigger fallback")
    print(json.dumps({"status": "ok", "feature_size": 256, "variants": len(variant_definitions())}))


def render_report(args: argparse.Namespace) -> None:
    validate_protocol(args)
    prepare_summary = json.loads((args.phase / "prepare_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((args.phase / "validation_anchor_summary.json").read_text(encoding="utf-8"))
    baseline_path = args.phase / "b7/baseline_round2a_b0_b6.summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    lines = [
        "# C0-256 Round-2C：双向病灶对应关系 Anchor Re-ranking Validation",
        "",
        "> 状态：validation 实验已完成。  ",
        "> 协议：SAM3-base trunk@256、SAM3-e33 video propagation canvas=256、固定 X3-best。  ",
        "> 范围：仅 validation，不运行 test、train propagation 或任何训练。",
        "",
        "## 1. 冻结协议",
        "",
        "| 项目 | 配置 |",
        "|---|---|",
        "| 数据 | validation 100，固定 8 个 train-side GT anchors |",
        "| 特征 encoder | 未微调 SAM3-base vision trunk |",
        "| 特征输入 | 256 × 256 |",
        f"| patch grid | {prepare_summary['patch_grid']} × {prepare_summary['patch_grid']} |",
        "| KNN topology | 原 SAM3-base@256，未整体重建 |",
        "| propagation teacher | 已冻结 SAM3-e33 merged video checkpoint |",
        "| propagation canvas | 256 × 256 |",
        "| route | b0，即 [anchor, target] 两帧 video propagation |",
        "| target mask | round1 pseudo-mask / 冻结 X3-best mask |",
        "| GT 用途 | anchor GT 可用于 support；validation target GT 只用于事后评估 |",
        "",
        "注意：b0 是两帧 video propagation，不是带文本提示的单图 Direct segmentation。",
        "",
        "## 2. 病灶对应分数",
        "",
        "```text",
        "forward = AUC(anchor lesion → target heatmap, target pseudo-mask)",
        "reverse = AUC(target pseudo-lesion → anchor heatmap, anchor GT mask)",
        "bidirectional = sqrt(forward * reverse)",
        "```",
        "",
        "同时比较 foreground prototype 与 token-to-token top-k；全部特征均来自 base@256。",
        "",
        "## 3. 执行规模",
        "",
        f"- validation targets：{prepare_summary['targets']}。",
        f"- GT anchors：{prepare_summary['anchors']}。",
        f"- 全部 b0 anchor-target routes：{prepare_summary['routes']}。",
        f"- 复用旧 e33 b0 routes：{prepare_summary['seeded_e33_routes']}。",
        f"- 新运行 e33 b0 routes：{prepare_summary['pending_e33_routes']}。",
        f"- 全 8 anchor 事后 b0 oracle：{audit['all8_b0_oracle']:.6f}。",
        "",
        "## 4. Anchor top-1 / top-3 validation 对照",
        "",
        "| Variant | Mask | Top-1 b0 Dice | Top-3 oracle | Mean target Spearman | Δ vs lesion mean | Top-3 b0 B7 Dice |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for definition in variant_definitions():
        name = definition["name"]
        item = audit["variants"][name]
        b7_path = args.phase / "b7" / f"{name}.summary.json"
        b7 = json.loads(b7_path.read_text(encoding="utf-8"))
        lesion_delta = item.get("paired_delta_vs_lesion_mean")
        delta = f"{lesion_delta['mean']:+.6f}" if lesion_delta else "—"
        spearman = (
            f"{item['mean_target_spearman']:.4f}"
            if item["mean_target_spearman"] is not None
            else "—"
        )
        lines.append(
            f"| {name} | {item['source']} | {item['top1_b0_dice']:.6f} | "
            f"{item['top3_b0_oracle']:.6f} | {spearman} | {delta} | "
            f"{b7['selected_gt_dice_evaluation_only']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 5. B7 解释边界",
            "",
            "各 variant 的 Top-3 b0 B7 使用完全相同的旧 B7 公式、固定 X3-best、",
            "每 target 恰好 3 个 b0 anchor candidates，因此不同 variant 之间可公平比较。",
            "",
            "```text",
            "B7 = (q_return * q_multi^2 * q_model^2)^0.2",
            "```",
            "",
            f"历史 Round-2A 双 mode、b0-b6 完整 validation B7 = "
            f"{baseline['selected_gt_dice_evaluation_only']:.6f}。",
            "该完整候选池与本轮 3 个 b0 候选的池规模不同，只作主线参考，不能直接归因。",
            "",
            "## 6. 输出",
            "",
            f"- Experiment root：`{args.phase}`。",
            "- `anchor_scores_validation.jsonl`：每个 anchor-target 的双向对应分数。",
            "- `validation_anchor_summary.json`：完整 ranking、paired delta 和 oracle 审计。",
            "- `quality_root/round2c_all_anchors_b0/`：800 条 e33 b0 propagation。",
            "- `b7/*.summary.json`：固定 3-candidate 的 B7 validation 对照。",
            "",
            "本轮没有读取 test target、没有传播 train target、没有生成新的训练 pool、没有训练模型。",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "status": "complete"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")

    for command in ("prepare", "summarize", "report"):
        child = subparsers.add_parser(command)
        child.add_argument("--phase", type=Path, required=True)
        child.add_argument("--protocol-root", type=Path, default=PROTOCOL)
        child.add_argument("--patch-cache", type=Path)
        child.add_argument("--base-features", type=Path, default=BASE_FEATURES)
        child.add_argument(
            "--round1-masks",
            type=Path,
            default=ROOT
            / "work/kvasir_1pct_anchors/validation_pseudo_masks_round1/train_pseudo_masks_round1.jsonl",
        )
        child.add_argument(
            "--student-predictions",
            type=Path,
            default=BASE / "predictions/X3_best/student_predictions_validation.jsonl",
        )
        child.add_argument("--split", choices=("validation",), default="validation")
        child.add_argument("--feature-size", type=int, default=256)
        child.add_argument("--canvas", type=int, default=256)
        child.add_argument("--match-topk", type=int, default=8)
        child.add_argument("--max-targets", type=int)
        if command == "report":
            child.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "prepare":
        if args.patch_cache is None:
            raise SystemExit("prepare requires --patch-cache")
        prepare(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        render_report(args)


if __name__ == "__main__":
    main()
