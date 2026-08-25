#!/usr/bin/env python3
"""Compare old/new KNN topology with the same e33 teacher on test b0-b6."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


MODES = (
    "sam3enc_anchor_conditioned_target_pooling",
    "sam3enc_anchor_conditioned_patch_correspondence",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_section(previous: dict, updated: dict) -> dict:
    return {
        f"bridge_{bridge}": {
            "base_knn_e33_teacher": previous[f"bridge_{bridge}"]["dice"],
            "e33_knn_e33_teacher": updated[f"bridge_{bridge}"]["dice"],
            "delta": updated[f"bridge_{bridge}"]["dice"] - previous[f"bridge_{bridge}"]["dice"],
            "base_targets": previous[f"bridge_{bridge}"]["n"],
            "e33_targets": updated[f"bridge_{bridge}"]["n"],
        }
        for bridge in range(7)
    }


def make_markdown(result: dict, args: argparse.Namespace) -> str:
    lines = [
        "# Round-2B：SAM3-e33 KNN 与 SAM3-e33 teacher 的 Test b0-b6 对照",
        "",
        f"> 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "> 固定传播 checkpoint = SAM3-e33；唯一变化 = KNN topology 从 SAM3-base@256 切换到 SAM3-e33@256。  ",
        "> 只评估 test b0-b6，不运行 B7、validation、base teacher、train propagation 或任何训练。",
        "",
        "## target 与 patch 完整对照",
        "",
        "| Bridge | base KNN target | e33 KNN target | target 变化 | base KNN patch | e33 KNN patch | patch 变化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    target_rows = result["modes"]["sam3enc_anchor_conditioned_target_pooling"]
    patch_rows = result["modes"]["sam3enc_anchor_conditioned_patch_correspondence"]
    for bridge in range(7):
        key = f"bridge_{bridge}"
        target = target_rows[key]
        patch = patch_rows[key]
        lines.append(
            f"| b{bridge} | {target['base_knn_e33_teacher']:.6f} | "
            f"{target['e33_knn_e33_teacher']:.6f} | {target['delta']:+.6f} | "
            f"{patch['base_knn_e33_teacher']:.6f} | "
            f"{patch['e33_knn_e33_teacher']:.6f} | {patch['delta']:+.6f} |"
        )
    lines.append("")
    sections = [("两模式 combined", result["combined"])]
    sections.extend((mode.replace("sam3enc_anchor_conditioned_", ""), result["modes"][mode]) for mode in MODES)
    for title, values in sections:
        lines.extend(
            [
                f"## {title}",
                "",
                "| Bridge | base KNN + e33 teacher | e33 KNN + e33 teacher | Delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for bridge, row in values.items():
            lines.append(
                f"| {bridge.replace('bridge_', 'b')} | {row['base_knn_e33_teacher']:.6f} | "
                f"{row['e33_knn_e33_teacher']:.6f} | {row['delta']:+.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 复现命令",
            "",
            "```bash",
            "cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7",
            "DEVICE=0 bash scripts/run_c0_256_round2b_e33_test_bridges_only.sh",
            "```",
            "",
            f"- 旧 topology 指标：`{args.baseline}`",
            f"- e33 topology 指标：`{args.updated}`",
            f"- 对照 JSON：`{args.output_json}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--updated", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    updated = load(args.updated)
    result = {
        "split": "test",
        "propagation_teacher": "SAM3-e33",
        "knn_feature_size": 256,
        "propagation_canvas": 256,
        "b7_evaluated": False,
        "validation_evaluated": False,
        "base_teacher_evaluated": False,
        "combined": compare_section(baseline["combined"], updated["combined"]),
        "modes": {
            mode: compare_section(baseline["modes"][mode], updated["modes"][mode])
            for mode in MODES
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = ["section\tbridge\tbase_knn_e33_teacher\te33_knn_e33_teacher\tdelta"]
    for section, values in [("combined", result["combined"]), *result["modes"].items()]:
        for bridge, row in values.items():
            rows.append(
                f"{section}\t{bridge}\t{row['base_knn_e33_teacher']:.10f}\t"
                f"{row['e33_knn_e33_teacher']:.10f}\t{row['delta']:.10f}"
            )
    args.output_tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(make_markdown(result, args), encoding="utf-8")
    print(json.dumps(result["combined"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
