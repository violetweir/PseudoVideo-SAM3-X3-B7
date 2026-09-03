#!/usr/bin/env python3
"""Offline Round-2C asymmetric three-anchor candidate-pool analysis.

Uses only completed validation b0 propagation results.  It does not invoke a
model, create routes, inspect test targets, or use validation GT for retrieval
or B7 selection.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
PHASE = ROOT / "work/rerun_c0_256_round2c_lesion_anchor_validation"
PREDICTIONS = (
    ROOT
    / "work/rerun_c0_256_sam3knn_s256_base/predictions/X3_best/student_predictions_validation.jsonl"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_binary(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = int(a.sum()) + int(b.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(a, b).sum() / denominator)


# These four pools are fixed before this analysis.  Rankers are intentionally
# not fused: forward serves precision, reverse serves recall/diversity.
POOLS: dict[str, list[tuple[str, str, str]]] = {
    "C0_forward_proto_top3": [
        ("forward_proto_round1", "round1", "forward_auc_prototype"),
        ("forward_proto_round1", "round1", "forward_auc_prototype"),
        ("forward_proto_round1", "round1", "forward_auc_prototype"),
    ],
    "C1_forward_forward_reverse": [
        ("forward_proto_round1", "round1", "forward_auc_prototype"),
        ("forward_token_round1", "round1", "forward_auc_token_topk"),
        ("reverse_token_x3", "x3_best", "reverse_auc_token_topk"),
    ],
    "C2_forward_reverse_reverse": [
        ("forward_proto_round1", "round1", "forward_auc_prototype"),
        ("reverse_token_x3", "x3_best", "reverse_auc_token_topk"),
        ("reverse_token_x3", "x3_best", "reverse_auc_token_topk"),
    ],
    "C3_forward_forward_reverse_alt": [
        ("forward_proto_round1", "round1", "forward_auc_prototype"),
        ("forward_proto_round1", "round1", "forward_auc_prototype"),
        ("reverse_token_x3", "x3_best", "reverse_auc_token_topk"),
    ],
}


def ranked_rows(
    rows: list[dict[str, Any]], source: str, score_key: str
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["target_mask_source"] == source]
    return sorted(selected, key=lambda row: (float(row[score_key]), row["anchor_id"]), reverse=True)


def choose_distinct(
    scores: list[dict[str, Any]], specifications: list[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    rankings = {
        (source, score_key): ranked_rows(scores, source, score_key)
        for _, source, score_key in specifications
    }
    for role, source, score_key in specifications:
        ranking = rankings[(source, score_key)]
        row = next((candidate for candidate in ranking if candidate["anchor_id"] not in used), None)
        if row is None:
            raise RuntimeError(f"Cannot construct three unique anchors for {role}")
        item = dict(row)
        item["candidate_role"] = role
        item["candidate_rank"] = ranking.index(row) + 1
        chosen.append(item)
        used.add(row["anchor_id"])
    return chosen


def score_pool(
    rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], canvas: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_id = rows[0]["target_id"]
    student = load_binary(predictions[target_id]["student_binary_mask"], (canvas, canvas))
    masks = [load_binary(row["forward_mask_path"], (canvas, canvas)) for row in rows]
    candidates: list[dict[str, Any]] = []
    for index, original in enumerate(rows):
        row = dict(original)
        peers = [mask for other_index, mask in enumerate(masks) if other_index != index]
        row["q_multi"] = float(np.mean([dice(masks[index], peer) for peer in peers]))
        row["q_return"] = float(row.get("q_cycle", 0.0))
        row["q_model"] = dice(masks[index], student)
        row["b7"] = (
            max(row["q_return"], 1e-6)
            * max(row["q_multi"], 1e-6) ** 2
            * max(row["q_model"], 1e-6) ** 2
        ) ** 0.2
        candidates.append(row)
    selected = max(
        candidates,
        key=lambda row: (row["b7"], row["q_multi"], row["q_return"], row["route_id"]),
    )
    return selected, candidates


def render_report(summary: dict[str, Any], report: Path) -> None:
    lines = [
        "# C0-256 Round-2C：Asymmetric Correspondence Anchor Pool",
        "",
        "> 状态：validation 离线组合分析已完成。  ",
        "> 未运行任何新模型：只重组已有的 800 条 e33 b0 propagation，并按原 B7 公式复算。",
        "",
        "## 1. 预定义候选池",
        "",
        "| Pool | Candidate 1 | Candidate 2 | Candidate 3 |",
        "|---|---|---|---|",
        "| C0 | Forward prototype #1 | Forward prototype #2 | Forward prototype #3 |",
        "| C1 | Forward prototype #1 | Forward token-topk #1 | Reverse token-topk (X3) #1 |",
        "| C2 | Forward prototype #1 | Reverse token-topk (X3) #1 | Reverse token-topk (X3) #2 |",
        "| C3 | Forward prototype #1 | Forward prototype #2 | Reverse token-topk (X3) #1 |",
        "",
        "若候选重复，则从该 ranker 的下一名补齐；不会混合成单一乘法或加法分数。",
        "",
        "## 2. Validation 结果",
        "",
        "| Pool | Candidate oracle | B7 selected Dice | Oracle gap | Mean B7 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in summary["pools"].items():
        lines.append(
            f"| {name} | {item['candidate_oracle']:.6f} | {item['b7_selected_dice']:.6f} | "
            f"{item['oracle_gap']:.6f} | {item['b7_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 3. 解释边界",
            "",
            "Candidate oracle 使用 validation GT 仅作事后审计；候选构造和 B7 选择均未读取 target GT。",
            "B7 保持原定义：`(q_return × q_multi² × q_model²)^0.2`，其中 X3-best 固定。",
            "",
            "这些结果只检验 b0 anchor candidate pool。它们不能直接外推为 bridge reranking 或 test 结论。",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, default=PHASE)
    parser.add_argument("--student-predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--canvas", type=int, default=256)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reproduction_reports/C0_256_round2c_asymmetric_candidate_pool.md",
    )
    args = parser.parse_args()
    if args.canvas != 256:
        raise SystemExit("This Round-2C analysis is frozen to canvas 256")

    score_rows = read_jsonl(args.phase / "anchor_scores_validation.jsonl")
    quality_rows = read_jsonl(
        args.phase
        / "quality_root/round2c_all_anchors_b0/propagation_quality_validation/propagation_quality.jsonl"
    )
    quality = {row["route_id"]: row for row in quality_rows if row.get("status") == "success"}
    predictions = {row["merged_id"]: row for row in read_jsonl(args.student_predictions)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        if row["route_id"] not in quality:
            continue
        grouped[row["target_id"]].append(row)
    if len(grouped) != 100:
        raise RuntimeError(f"Expected 100 validation targets, received {len(grouped)}")

    all_outputs: list[dict[str, Any]] = []
    pool_summaries: dict[str, dict[str, Any]] = {}
    for pool_name, specifications in POOLS.items():
        selected_rows: list[dict[str, Any]] = []
        candidate_oracles: list[float] = []
        b7_selected: list[float] = []
        b7_values: list[float] = []
        for target_id in sorted(grouped):
            selected_scores = choose_distinct(grouped[target_id], specifications)
            route_rows = []
            for score in selected_scores:
                route = dict(quality[score["route_id"]])
                route.update(
                    {
                        "round2c_pool": pool_name,
                        "candidate_role": score["candidate_role"],
                        "candidate_rank": score["candidate_rank"],
                        "candidate_score_source": score["target_mask_source"],
                        "candidate_score": float(
                            score[
                                next(
                                    score_key
                                    for role, _, score_key in specifications
                                    if role == score["candidate_role"]
                                )
                            ]
                        ),
                    }
                )
                route_rows.append(route)
            selected, candidates = score_pool(route_rows, predictions, args.canvas)
            candidate_oracles.append(max(float(row["gt_dice_evaluation_only"]) for row in candidates))
            b7_selected.append(float(selected["gt_dice_evaluation_only"]))
            b7_values.append(float(selected["b7"]))
            for row in candidates:
                row["selected_by_b7"] = row["route_id"] == selected["route_id"]
                all_outputs.append(row)

        candidate_oracle = float(np.mean(candidate_oracles))
        b7_dice = float(np.mean(b7_selected))
        pool_summaries[pool_name] = {
            "targets": len(candidate_oracles),
            "candidate_oracle": candidate_oracle,
            "b7_selected_dice": b7_dice,
            "oracle_gap": candidate_oracle - b7_dice,
            "b7_mean": float(np.mean(b7_values)),
        }

    outputs = args.phase / "asymmetric_candidate_pools_validation.jsonl"
    write_jsonl(outputs, all_outputs)
    summary = {
        "status": "complete",
        "split": "validation",
        "feature_size": 256,
        "propagation_canvas": 256,
        "new_propagation": False,
        "new_training": False,
        "test_used": False,
        "pools": pool_summaries,
        "candidate_output": str(outputs),
    }
    summary_path = args.phase / "asymmetric_candidate_pool_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_report(summary, args.report)
    print(json.dumps({"summary": str(summary_path), "pools": pool_summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
