#!/usr/bin/env python3
"""Formal report for the ViT-B @256 route-selector experiment (lora checkpoint).

Reads selector_report_{tag}.json (M1-M4) and b7_student_audit_{tag}.json
(student-audited methods + per-target selections) and writes a full markdown
report, a compact JSON, and a per-target CSV for the best configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
OUT = ROOT / "work/kvasir_1pct_anchors/route_selector_vitb256"


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="lora_p491_e20")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--fixed-reference", type=float, default=0.897779682728018,
                        help="fixed best bridge (target_pooling knn=cls b6) mean Dice")
    args = parser.parse_args()

    sel = json.loads((args.output_root / f"selector_report_{args.tag}.json").read_text())
    audit = json.loads((args.output_root / f"b7_student_audit_{args.tag}.json").read_text())
    pools = ("P1_b0_b6", "P2_b3_b6")

    # collect all method summaries: name -> {pool -> {selected, oracle, gap}}
    methods: dict[str, dict[str, dict]] = {}
    for pool in pools:
        for name, item in sel[pool].items():
            if isinstance(item, dict) and "selected" in item:
                methods.setdefault(name, {})[pool] = item
    for pool in pools:
        for item in audit[pool]:
            methods.setdefault(item["name"], {})[pool] = item

    def fixed_mean(pool: str) -> float:
        pt = audit[f"{pool}_per_target"]
        vals = [r["dice"] for r in pt["fixed_b6"].values()]
        return sum(vals) / max(len(vals), 1)

    oracle_p1 = audit["P1_b0_b6"][0]["oracle"]
    lines: list[str] = []
    lines.append(f"# 路线选择/路由器正式报告（checkpoint = {args.tag}）")
    lines.append("")
    lines.append("协议：Kvasir test 100 targets；路线 = 11 变体 × b0–b6（ViT-B @256，"
                 "beam 32）；SAM3 @256；传播质量（正向 trace + 回传 q_return）"
                 "由同一 checkpoint 计算；学生审核 q_model 复用 256-px "
                 "X3/S2/S3 预测。校准/拟合只用 validation，test 一次性报告。")
    lines.append("")
    lines.append(f"对照：固定 b6（anchor_conditioned_target_pooling__knn_cls）= "
                 f"{fmt(args.fixed_reference)}；P1 全池 oracle = {fmt(oracle_p1)}。")

    lines.append("")
    lines.append("## 1. 方法对比（test selected Dice）")
    lines.append("")
    lines.append("| 方法 | 拟合 | P1 (b0–b6) | gap | P2 (b3–b6) | gap |")
    lines.append("|---|---|---:|---:|---:|---:|")
    order = [
        ("fixed_b6", "固定 b6（不选择）", "无"),
        ("M1_0.5_0.5_test", "M1 固定 0.5/0.5（旧默认移植，未校准）", "无"),
        ("M2_calibrated_test", "M2 两信号校准线性", "validation"),
        ("M3_ridge_test", "M3 ridge 路由器", "validation"),
        ("M4_q_return_test", "M4 纯 q_return", "无"),
        ("M4_q_multi_test", "M4 纯 q_multi（T21 空掩码约定）", "无"),
        ("M4_sam_score_test", "M4 纯 SAM score", "无"),
        ("q_multi_only", "纯 q_multi（B7 空掩码约定）", "无"),
        ("q_model_X3_only", "纯 q_model（X3）", "无"),
        ("q_model_mean_only", "纯 q_model（三学生均值）", "无"),
        ("B7_X3", "B7 几何（X3 审核）", "无"),
        ("B7_mean", "B7 几何（三学生均值审核）", "无"),
        ("lin_cal_X3", "三信号校准线性（X3）", "validation"),
        ("lin_cal_mean", "三信号校准线性（三学生均值）", "validation"),
    ]
    for name, label, fit in order:
        cells = []
        for pool in pools:
            if name == "fixed_b6":
                m = fixed_mean(pool)
                pool_oracle = audit[pool][0]["oracle"]
                cells.append(f"{fmt(m)} / {fmt(pool_oracle - m)}")
                continue
            item = methods.get(name, {}).get(pool)
            if item:
                cells.append(f"{fmt(item['selected'])} / {fmt(item['gap'])}")
            else:
                cells.append("—")
        lines.append(f"| {label} | {fit} | {cells[0]} | {cells[1]} |")

    # ---- best configuration ----
    best = None
    for pool in pools:
        for item in audit[pool]:
            if best is None or item["selected"] > best["selected"]:
                best = {**item, "pool": pool}
    lines.append("")
    lines.append("## 2. 最优配置")
    lines.append("")
    lines.append(f"- **{best['name']}** @ {best['pool']}：selected "
                 f"**{fmt(best['selected'])}**（oracle {fmt(best['oracle'])}，"
                 f"gap {fmt(best['gap'])}）")
    if "weights" in best:
        w = best["weights"]
        lines.append(f"- validation 校准权重：q_return {w['q_return']:.2f} / "
                     f"q_multi {w['q_multi']:.2f} / q_model {w['q_model']:.2f}"
                     f"（validation selected {fmt(best['validation_dice'])}）")

    # ---- per-target vs fixed b6 ----
    pt = audit[f"{best['pool']}_per_target"]
    best_name = best["name"]
    fixed = pt["fixed_b6"]
    oracle = pt["oracle"]
    selrows = pt[best_name]
    gains = []
    for t in selrows:
        d_sel = selrows[t]["dice"]
        d_fix = fixed.get(t, {}).get("dice", float("nan"))
        d_ora = oracle[t]["dice"]
        gains.append((t, d_sel, d_fix, d_ora, d_sel - d_fix))
    wins = sum(1 for *_, g in gains if g > 1e-9)
    losses = sum(1 for *_, g in gains if g < -1e-9)
    ties = len(gains) - wins - losses
    mean_gain = sum(g for *_, g in gains) / max(len(gains), 1)
    lines.append("")
    lines.append(f"## 3. 最优配置 vs 固定 b6（{best['pool']}，test）")
    lines.append("")
    lines.append(f"- 胜 {wins} / 平 {ties} / 负 {losses}；平均增益 {fmt(mean_gain)}")
    lines.append("")
    lines.append("### 最大提升（top 10）")
    lines.append("")
    lines.append("| target | 选择 | 固定 b6 | oracle | 增益 |")
    lines.append("|---|---:|---:|---:|---:|")
    for t, d_sel, d_fix, d_ora, g in sorted(gains, key=lambda x: -x[4])[:10]:
        lines.append(f"| {t.split('::')[-1]} | {fmt(d_sel)} | {fmt(d_fix)} | {fmt(d_ora)} | {fmt(g)} |")
    lines.append("")
    lines.append("### 最大回退（top 10）")
    lines.append("")
    lines.append("| target | 选择 | 固定 b6 | oracle | 增益 |")
    lines.append("|---|---:|---:|---:|---:|")
    for t, d_sel, d_fix, d_ora, g in sorted(gains, key=lambda x: x[4])[:10]:
        lines.append(f"| {t.split('::')[-1]} | {fmt(d_sel)} | {fmt(d_fix)} | {fmt(d_ora)} | {fmt(g)} |")

    # ---- histogram ----
    lines.append("")
    lines.append(f"## 4. 最优配置选中的路线分布（{best['pool']}）")
    lines.append("")
    hist = Counter((r["family"], r["bridge_count"]) for r in selrows.values())
    lines.append("| family | b0 | b1 | b2 | b3 | b4 | b5 | b6 | 合计 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fam in sorted({k[0] for k in hist}):
        counts = [hist.get((fam, b), 0) for b in range(7)]
        lines.append(f"| {fam} | " + " | ".join(str(c) for c in counts) + f" | {sum(counts)} |")

    # ---- signals ----
    lines.append("")
    lines.append("## 5. 信号判别力（spearman vs test GT Dice）")
    lines.append("")
    lines.append("| 信号 | P1 | P2 |")
    lines.append("|---|---:|---:|")
    for f in ("q_cycle", "q_multi", "q_model_x3", "q_model_mean"):
        lines.append(f"| {f} | {fmt(audit['spearman'].get(f'P1_b0_b6_{f}', float('nan')))} | "
                     f"{fmt(audit['spearman'].get(f'P2_b3_b6_{f}', float('nan')))} |")

    # ---- validation stability ----
    lines.append("")
    lines.append("## 6. 固定公式方法在 validation 上的表现（稳定性参考）")
    lines.append("")
    lines.append("| 方法 | P1 validation | P2 validation |")
    lines.append("|---|---:|---:|")
    for name in ("B7_X3", "B7_mean", "q_multi_only"):
        v1 = audit["validation_reference"].get("P1_b0_b6", {}).get(name, float("nan"))
        v2 = audit["validation_reference"].get("P2_b3_b6", {}).get(name, float("nan"))
        lines.append(f"| {name} | {fmt(v1)} | {fmt(v2)} |")

    # ---- per-target csv ----
    csv_path = args.output_root / f"route_selector_best_{args.tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "selected_family", "selected_bridge", "selected_dice",
                         "fixed_b6_dice", "oracle_dice", "q_return", "q_multi", "q_model_mean"])
        for t in selrows:
            r = selrows[t]
            writer.writerow([t, r["family"], r["bridge_count"], f"{r['dice']:.6f}",
                             f"{fixed.get(t, {}).get('dice', float('nan')):.6f}",
                             f"{oracle[t]['dice']:.6f}",
                             f"{r['q_return']:.6f}", f"{r['q_multi']:.6f}", f"{r['q_model_mean']:.6f}"])

    text = "\n".join(lines) + "\n"
    (args.output_root / f"route_selector_report_{args.tag}.md").write_text(text, encoding="utf-8")
    summary = {
        "tag": args.tag,
        "best": best,
        "per_target_wins_losses": {"wins": wins, "ties": ties, "losses": losses, "mean_gain": mean_gain},
    }
    (args.output_root / f"route_selector_report_{args.tag}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(text)


if __name__ == "__main__":
    main()
