# C3:Path-Invariance 核心实验(交接文档)

> 版本: 2026-08-16(骨架版,评估完成后填充数字)
> 状态: validation 1500 条路线评估中;本文件为下一轮接手/继续分析用
> 前置: 见 10_lesion_knn_mechanism_handoff.md(轮次1-3 结论)

## 0. 一句话动机

边级 transportability 代理全灭(ρ≈0.1),但"最终预测之间的一致性"是强质量信号
(route 级 ρ≈0.55-0.64,零推理成本)。C3 检验核心科学问题:

> **Consensus reliability 是否随 route independence 增加而增强?**

## 1. 协议(严格遵守)

- 主池 = validation 100 target;规则(selector + τ)**只在 validation 冻结**,test 只跑一次汇报。
- 特征/检索口径与轮次1-3 完全一致:sam3enc_lesion 病灶描述子 @1008,8 个 GT anchor。
- 评估口径:冻结 base sam3.pt、canvas 256、forward-only Dice(与 10 号文档一致)。
- 路线生成参数(k_anchors=3, max_bridge=2, top_beams=2, beam_width=32)对 validation/test 完全相同(冻结)。

## 2. C3 路线池

- 生成器: scripts/generate_c3_routes.py
- 每 target:按 lesion 相似度取 top-3 anchor;每 anchor: direct + bridge_1 + bridge_2,每深度保留 top-2 beam。
- validation: 100 target × 3 anchor × 5 = 1500 条,输出 {root}/sam3enc_lesion_c3/validation_pool0_stage1/routes.jsonl
- test: 同样式(规则冻结后生成)

## 3. 四组对比设计

| 组 | 候选构成 | 独立性问题 |
|---|---|---|
| C3-A | 同 anchor 多 depth(b0-b2,同一 anchor) | 最低(共享 anchor/bridge/memory) |
| C3-B | 多 anchor,仅 direct(3×b0) | 只变初始监督来源 |
| C3-C | 3 anchor × {b0,b1,b2} 最佳 beam(9 条) | 中等 |
| C3-D | 全部候选含 beam diversity(15 条) | 最高 |

## 4. 三种 selector

1. medoid: Q_i = mean_{j≠i} Dice(M_i, M_j),取 argmax
2. anchor-balanced: 每个 anchor 先平均,再跨 anchor 平均(防同 anchor 重复投票)
3. cross-anchor support: support(M_i) = #{anchor a: max_{j∈a} Dice(M_i,M_j) > τ},取最大 support

## 5. 指标(每组)

- route 级 Spearman ρ(Q_cons, GT Dice)
- selector Dice / oracle Dice / **regret = oracle − selected**
- mean pair overlap(帧共享 Jaccard)与 ρ(overlap, |ΔDice|)(独立性→共识有用性)
- 与既有代理对比:path_bottleneck 0.138 / q_multi 0.130 / F_lesion 相似度 0.277

## 6. 结果(validation 100 target,τ=0.85)

| 组 | 候选数/靶 | route级ρ(Q,GT) | 固定/均值 | 最优 selector | oracle | regret |
|---|---|---|---|---|---|---|
| C3-A(同anchor多depth) | ~5 | — | 0.857(anchor medoid 均值) | 0.860 | 0.910 | 0.050 |
| C3-B(3 anchor×direct) | 3 | **0.729** | 0.809 | 0.828 | 0.856 | 0.028 |
| C3-C(3 anchor×b0-2) | 9 | **0.715** | 0.829 | 0.860(support) | 0.901 | 0.041 |
| C3-D(全含diversity) | 15 | **0.704** | 0.836 | 0.860(medoid) | 0.910 | 0.050 |

关键数字:
- 多 anchor 池的 route 级 ρ(consensus,GT) = **0.70–0.73**,远高于:
  既有 per-depth-best 池(test 0.55 / validation 0.64)、lesion 相似度 0.277、path_bottleneck 0.138。
- 跨组对比:anchor 变化(C3-B,直接路线)带来最高的 ρ(0.729);
  depth/beam 多样化(C3-C/D)候选更多但 ρ 略降 → **独立性提升主要来自换 anchor(初始监督来源),而非换 bridge**。
- selector 收益温和:相对 per-target 均值 +0.019~+0.031;oracle 随候选数上升(0.856→0.910),
  但 selector regret 也上升(0.028→0.050)→ 候选越多,选中最优越难。
- mean pair overlap(帧共享 Jaccard):B 0.333 / C 0.313 / D 0.304;ρ(overlap,|ΔDice|)≈−0.1~−0.14(弱负,共享帧越多质量差越小)。

## 6.5 Test 一次性汇报(100 target,冻结规则,未再调参)

| 组 | 候选数/靶 | route级ρ(Q,GT) | 均值 | 冻结 selector | oracle | regret |
|---|---|---|---|---|---|---|
| C3-B(3 anchor×direct) | 3 | **0.648** | 0.832 | 0.871(support) | 0.913 | 0.042 |
| C3-C(3 anchor×b0-2) | 9 | **0.651** | 0.863 | 0.895(support) | 0.930 | 0.035 |
| C3-D(全含diversity) | 15 | **0.653** | 0.870 | 0.895(medoid) | 0.931 | 0.037 |

对照(test 既有 per-depth-best 池):均值 0.882、固定最优 b1 0.8921、medoid 0.896、oracle 0.930。
C3-C/D selector ≈ 0.895,与既有池 medoid 0.896 基本持平,但候选池本身更"弱"(只含 3 anchor×b0-2,
均值 0.86-0.87)——selector 吃回了 0.03~0.04 增益,且**在完全独立的池构造下复现**。
ρ 在 test 上 0.65(validation 0.70-0.73),均远超 lesion 相似度 0.277 / path_bottleneck 0.138。

## 7. 冻结规则(已冻结,见 work/kvasir_1pct_anchors/c3_frozen_rules.json)

- τ = 0.85
- B_direct → cross_anchor_support;C_top1 → cross_anchor_support;D_all → medoid
- 说明:三种 selector 在 validation 上非常接近(差 <0.002),选型是防御性决定;
  test 只跑一次(见 6.5),冻结后不再调整。

## 7.5 结论(机制层)

1. **Consensus 是强质量信号**:多 anchor 池 route 级 ρ(consensus,GT) = 0.65(test)/0.70-0.73(validation),
   比边级代理(path_bottleneck 0.138 / q_multi 0.130 / cycle −0.090)高一个量级。
2. **独立性来源 = 换 anchor,而非换 bridge**:C3-B(仅 direct,纯 anchor 变化)ρ 最高(validation 0.729);
   同 anchor 内 depth/beam 多样化(C3-C/D)ρ 略降/持平 → 初始监督来源的多样化是关键。
3. **Consensus-as-selector 温和**:+0.03~0.04 over 候选均值,regret 0.035~0.042;
   oracle gap(≈0.93)主要由"候选池本身质量"决定,selector 无法创造新信息。
4. 方法学含义:论文叙事应为 "route-invariant consensus as uncertainty estimate"(质量信号),
   而非 "consensus selection 带来 Dice 增益"。

## 8. 产物路径

| 内容 | 路径 |
|---|---|
| 生成器 | scripts/generate_c3_routes.py |
| 分析器 | scripts/analyze_c3.py |
| 冻结器 | scripts/analyze_c3_freeze.py |
| validation 路线 | work/.../sam3enc_lesion_c3/validation_pool0_stage1/routes.jsonl |
| validation 评估 | work/.../sam3enc_lesion_c3/eval_c3_forward_validation/ |
| 配对 bootstrap | work/kvasir_1pct_anchors/pseudo_vs_gt_lesion_paired_bootstrap.json |
| 既有池共识分析 | work/kvasir_1pct_anchors/consensus_medoid_analysis.json |

## 9. 注意点

- 评估脚本对 --split validation 会自动在 eval-name 后加 _validation 后缀。
- C3-C 的"最佳 beam"按 route path_mean_similarity 选取(确定性),不依赖写入顺序。
- paired bootstrap 93/100 target 的 Δ=0(两检索选同一 top anchor),结论"统计等价"。
