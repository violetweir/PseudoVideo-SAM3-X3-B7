#!/usr/bin/env python3
"""Summarize the frozen-X3 topology-by-teacher 2x2 factorial experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
MODES = (
    "sam3enc_anchor_conditioned_target_pooling",
    "sam3enc_anchor_conditioned_patch_correspondence",
)
CELL_NAMES = ("G0_Tbase", "G0_Te33", "G1_Tbase", "G1_Te33")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def effects(values: dict[str, float]) -> dict:
    g0_base = values["G0_Tbase"]
    g0_e33 = values["G0_Te33"]
    g1_base = values["G1_Tbase"]
    g1_e33 = values["G1_Te33"]
    return {
        "teacher_effect_on_base_topology": g0_e33 - g0_base,
        "teacher_effect_on_e33_topology": g1_e33 - g1_base,
        "topology_effect_with_base_teacher": g1_base - g0_base,
        "topology_effect_with_e33_teacher": g1_e33 - g0_e33,
        "interaction": g1_e33 - g1_base - g0_e33 + g0_base,
    }


def cell_paths(phase: Path, base_phase: Path, round2a: Path, split: str) -> dict[str, dict]:
    base_metrics = (
        base_phase / "validation_bridge_b0_b6.json"
        if split == "validation"
        else base_phase / "current_base_test/bridge_b0_b6_test_metrics.json"
    )
    return {
        "G0_Tbase": {
            "metrics": base_metrics,
            "b7": phase / f"baseline_controls/G0_Tbase_{split}.summary.json",
        },
        "G0_Te33": {
            "metrics": round2a / f"two_mode_b0_b6_{split}.json",
            "b7": phase / f"baseline_controls/G0_Te33_{split}.summary.json",
        },
        "G1_Tbase": {
            "metrics": phase / f"teachers/base/two_mode_b0_b6_{split}.json",
            "b7": phase / f"teachers/base/b7/X3_best_{split}.summary.json",
        },
        "G1_Te33": {
            "metrics": phase / f"teachers/e33/two_mode_b0_b6_{split}.json",
            "b7": phase / f"teachers/e33/b7/X3_best_{split}.summary.json",
        },
    }


def summarize(args: argparse.Namespace) -> dict:
    result = {
        "design": "2x2 factorial: SAM3-base/e33 KNN topology x SAM3-base/e33 propagation teacher",
        "feature_size": 256,
        "canvas": 256,
        "bridges": list(range(7)),
        "modes": list(MODES),
        "frozen_b7_selector": "X3 student validation-best",
        "train_propagation_executed": False,
        "new_pseudo_labels_generated": False,
        "new_student_or_sam3_training_executed": False,
        "feature_audit": read(args.phase / "e33_feature_audit.json"),
        "topology_audit": read(args.phase / "topology_route_audit.json"),
        "splits": {},
    }
    for split in ("validation", "test"):
        split_result = {"cells": {}, "combined_bridge_effects": {}}
        paths = cell_paths(args.phase, args.base_phase, args.round2a, split)
        for name, sources in paths.items():
            propagation = read(sources["metrics"])
            b7 = read(sources["b7"])
            split_result["cells"][name] = {
                "topology": "SAM3-base@256" if name.startswith("G0") else "SAM3-e33@256",
                "teacher": "SAM3-base" if name.endswith("Tbase") else "SAM3-e33",
                "b7_selected_dice": b7["selected_gt_dice_evaluation_only"],
                "b7_oracle_dice": b7.get("oracle_gt_dice_evaluation_only"),
                "b7_oracle_gap": b7.get("oracle_gap_evaluation_only"),
                "b7_mean_confidence_not_dice": b7["b7_mean"],
                "selected_targets": b7["selected_targets"],
                "combined": propagation["combined"],
                "modes": propagation["modes"],
            }
        split_result["b7_effects"] = effects(
            {name: split_result["cells"][name]["b7_selected_dice"] for name in CELL_NAMES}
        )
        for bridge in range(7):
            key = f"bridge_{bridge}"
            split_result["combined_bridge_effects"][key] = effects(
                {name: split_result["cells"][name]["combined"][key]["dice"] for name in CELL_NAMES}
            )
        result["splits"][split] = split_result
    return result


def fmt(number: float | None) -> str:
    return "—" if number is None else f"{number:.6f}"


def signed(number: float) -> str:
    return f"{number:+.6f}"


def markdown(result: dict, args: argparse.Namespace) -> str:
    audit = result["feature_audit"]
    lines = [
        "# C0-256 Round-2B：SAM3-e33 KNN Topology Refresh 2×2 因子实验",
        "",
        f"> 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（服务器本地时间）  ",
        f"> 实验根目录：`{args.phase}`  ",
        "> 范围：只比较 topology 与 propagation teacher；不做 train propagation、不生成新伪标签、不训练 X4 或 SAM3。",
        "",
        "## 1. 实验设计与固定条件",
        "",
        "```text",
        "G0 = KNN(F_SAM3-base @ 256x256)",
        "G1 = KNN(F_SAM3-e33  @ 256x256)",
        "T0 = SAM3-base propagation @ 256x256",
        "T1 = SAM3-e33  propagation @ 256x256",
        "selector = frozen X3 validation-best; modes = target-pooling + patch-correspondence; bridges = b0-b6",
        "```",
        "",
        "| 因子组合 | KNN topology | Propagation teacher | 解释 |",
        "|---|---|---|---|",
        "| G0_Tbase | SAM3-base | SAM3-base | 原始 SAM3-KNN 对照 |",
        "| G0_Te33 | SAM3-base | SAM3-e33 | Round-2A teacher-only |",
        "| G1_Tbase | SAM3-e33 | SAM3-base | topology-only |",
        "| G1_Te33 | SAM3-e33 | SAM3-e33 | Round-2B full |",
        "",
        "同一套 8 个 anchors、1000 张图的 embedding、beam width=32、patch-mean descriptor、validation/test 各 100 个 target。训练图像只作为 KNN bridge 候选池参与特征提取，不执行任何 train split propagation。",
        "",
        "## 2. e33 特征与 KNN 邻居变化",
        "",
        f"- e33 checkpoint：`{audit['encoder_checkpoint']}`",
        f"- descriptor shape：`{audit['descriptor_shape']}`",
        f"- base/e33 同图 descriptor cosine：mean={fmt(audit['row_cosine']['mean'])}，min={fmt(audit['row_cosine']['min'])}，median={fmt(audit['row_cosine']['median'])}。",
        f"- descriptor mean L2 delta：{fmt(audit['mean_l2_descriptor_delta'])}；max absolute delta：{fmt(audit['max_abs_descriptor_delta'])}。",
        f"- 8 个 anchor prototype 平均 cosine：{fmt(audit['anchor_prototype_cosine_mean'])}。",
        "",
        "| Query split | Top-1 overlap | Top-5 overlap | Top-10 overlap | Top-32 overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, overlap in audit["knn_train_pool_overlap"].items():
        lines.append(
            f"| {split} | {fmt(overlap['top_1_overlap'])} | {fmt(overlap['top_5_overlap'])} | "
            f"{fmt(overlap['top_10_overlap'])} | {fmt(overlap['top_32_overlap'])} |"
        )
    lines.extend(["", "## 3. 实际 route topology 变化", "", "| Split | Mode | Routes | Changed route | Changed anchor | Bridge Jaccard |", "|---|---|---:|---:|---:|---:|"])
    for split, modes in result["topology_audit"]["splits"].items():
        for mode, stats in modes.items():
            overall = stats["overall"]
            lines.append(
                f"| {split} | {mode.replace('sam3enc_anchor_conditioned_', '')} | {overall['routes']} | "
                f"{overall['changed_route']} | {overall['changed_anchor']} | {fmt(overall['mean_bridge_jaccard'])} |"
            )

    for section, split in enumerate(("validation", "test"), start=4):
        block = result["splits"][split]
        lines.extend(["", f"## {section}. {split.title()}：2×2 B7 与固定桥传播", "", "| Cell | Topology | Teacher | B7 Dice | Oracle Dice | Oracle gap | Targets |", "|---|---|---|---:|---:|---:|---:|"])
        for name in CELL_NAMES:
            cell = block["cells"][name]
            lines.append(
                f"| {name} | {cell['topology']} | {cell['teacher']} | {fmt(cell['b7_selected_dice'])} | "
                f"{fmt(cell['b7_oracle_dice'])} | {fmt(cell['b7_oracle_gap'])} | {cell['selected_targets']} |"
            )
        lines.extend(["", "B7 因子效应：", ""])
        for name, value in block["b7_effects"].items():
            lines.append(f"- `{name}` = {signed(value)}")
        lines.extend(["", "两模式 combined 的 b0-b6 固定桥 Dice：", "", "| Bridge | G0_Tbase | G0_Te33 | G1_Tbase | G1_Te33 | Δ topology @base | Δ topology @e33 | interaction |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for bridge in range(7):
            key = f"bridge_{bridge}"
            values = [block["cells"][name]["combined"][key]["dice"] for name in CELL_NAMES]
            change = block["combined_bridge_effects"][key]
            lines.append(
                f"| b{bridge} | {' | '.join(fmt(value) for value in values)} | "
                f"{signed(change['topology_effect_with_base_teacher'])} | "
                f"{signed(change['topology_effect_with_e33_teacher'])} | {signed(change['interaction'])} |"
            )
        for mode in MODES:
            short_mode = mode.replace("sam3enc_anchor_conditioned_", "")
            lines.extend(["", f"{short_mode} 模式明细：", "", "| Bridge | G0_Tbase | G0_Te33 | G1_Tbase | G1_Te33 |", "|---|---:|---:|---:|---:|"])
            for bridge in range(7):
                key = f"bridge_{bridge}"
                values = [block["cells"][name]["modes"][mode][key]["dice"] for name in CELL_NAMES]
                lines.append(f"| b{bridge} | {' | '.join(fmt(value) for value in values)} |")

    lines.extend(
        [
            "",
            "## 6. 复现命令与产物",
            "",
            "```bash",
            "cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7",
            "DEVICE=0 bash scripts/run_c0_256_round2b_topology_refresh_factorial.sh",
            "```",
            "",
            "流水线：e33 trunk 特征提取 → base/e33 embedding 与 KNN overlap audit → validation/test 两模式 b0-b6 重新构图 → 固定 X3 的 G0 历史对照 → G1+base 与 G1+e33 validation-first propagation → B7 → 2×2 因子汇总。",
            "",
            "```text",
            f"{args.phase}/",
            "├── features/sam3_e33_s256_features.npz",
            "├── e33_feature_audit.json",
            "├── topology_route_audit.json",
            "├── stage1_feature_knn_e33_b0_b6/",
            "├── baseline_controls/",
            "├── teachers/base/",
            "├── teachers/e33/",
            "├── factorial_2x2_summary.json",
            "├── pipeline.log",
            "└── ROUND2B_COMPLETE",
            "```",
            "",
            "`b7_mean` 只表示 selector 置信度，不是 Dice；正式 B7 结果统一使用 `selected_gt_dice_evaluation_only`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--base-phase", type=Path, required=True)
    parser.add_argument("--round2a", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(markdown(result, args), encoding="utf-8")
    print(json.dumps({split: result["splits"][split]["b7_effects"] for split in ("validation", "test")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
