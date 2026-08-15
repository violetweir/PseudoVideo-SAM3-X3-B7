# 路线选择/路由器实验（继承 KNN ViT-B @256 设置）

> 创建:2026-08-10。仿照 `docs/knn_experiment_vitb256.md` 的格式，
> 在 KNN 路线实验（ViT-B @256 / SAM3 @256 / Kvasir 协议）之上，固定
> "如何从候选路线里选一个"这一环节的协议，并把几种选择方法都实验一遍。
> 实验名暂定：**route selector / router**。

> **状态更新（2026-08-12）**：原始 M1-M5、base/LoRA 正式报告、pairwise
> nested-OOF、特征消融、冻结 Kvasir 诊断、CVC-ClinicDB 外部迁移，以及
> baseline-aware risk gate 的双数据集评估均已完成。本文按时间保留探索过程，
> 第 7 节给出总结果，第 13 节给出最终结论与完整产物索引。

> **重要口径纠正（2026-08-12）**：KNN 路径构造方法只能固定一种。此前第 10-12
> 节将 11 个 route family 混入同一候选池的 `P2=44 candidates` 违反该约束，相关
> 数字只保留为越界探索，不能作为主实验或论文结果。合法主协议固定
> `anchor_conditioned_target_pooling__knn_cls`，只在同一 KNN 路径的 b1-b6 六个
> 伪视频桥长中选择；b0 是 direct prediction，不经过 KNN，仅作额外对照。

## 0. 目标与动机

- KNN 阶段（`stage1_feature_knn_vitb256`）已给出 11 个变体 × b0–b6 的
  test 逐桥长 forward Dice（base 与 lora_p491_e20 两个 checkpoint）；
- 初始目标是看能否把 test Dice 推得更高，后续因线性分数组合依赖学生且说服力
  不足，实验方向进一步收敛为**真正的 pairwise route ranking 与风险控制**：
  固定桥长（如 b6 = 0.8978）是"不选择"的对照，全候选池 oracle ≈ 0.9357
  是天花板；本实验尝试各种加权/路由方法，看**实际选出来的 Dice 能接近
  oracle 多少**；
- 特别地，老主线的 `0.5×q_return + 0.5×q_multi` 在现有记录里是默认值而非
  校准值（T25/S27 的三信号线性权重才是 validation 网格校准的），所以本期
  **把"0.5/0.5"当作一个参考方法，并额外做 validation 校准版本**，给出
  数据拟合出来的权重，而不是拍脑袋。

> 早期探索曾只看 test selected Dice；自第 10 节起改为 validation-only grouped
> nested CV 决定模型与超参数，测试集只做冻结评估，并同时报告 bootstrap CI、
> catastrophic regression、trimmed delta 与 CVaR10。后者是当前有效口径。

## 1. 统一协议（继承 KNN，新增内容用 ★ 标注）

| 项 | 固定值 |
|---|---|
| 数据集 / 划分 | 与 KNN 实验一致：Kvasir train 800 / val 100 / test 100，8 固定 anchor，792 无标注桥池 |
| 路线 | 11 个变体（与 KNN 实验完全一致，beam 32，b0–b6） |
| DINOv3 特征 | ViT-B @256（grid 16），只用于路由构造，本阶段不再变化 |
| SAM3 评估分辨率 | **256×256** |
| checkpoint | **两个都跑**：`lora_p491_e20_merged_video.pt`（tag=`lora_p491_e20`）与 base `sam3.pt`（tag=`base`），传播质量目录按 tag 隔离 |
| ★ validation 路线 | 需要先生成 11 变体 × 700（`--split validation`），KNN 阶段只有 test 路线 |
| ★ 传播质量特征 | `eval_route_propagation_quality.py`：正向 trace（逐帧掩码/面积/质心/相邻 Dice/SAM score）+ 回传 q_return，validation + test |
| ★ 质量目录隔离 | 按 checkpoint 加 tag（如 `propagation_quality_test_lora_p491_e20`），避免 base/lora 互相覆盖 |
| 校准纪律 | 只允许用 validation 拟合/校准；test 一次性报告 |
| 输出根目录 | `work/kvasir_1pct_anchors/route_selector_vitb256/` |

## 2. 候选池

| 池 | 构成 | 每 target 候选数 |
|---|---|---:|
| S1（正式） | 固定 `target_pooling__knn_cls` × b1-b6 | 6 |
| S0（额外诊断） | 同一固定 family × b0-b6；b0 为 direct | 7 |
| P1（历史越界探索） | 11 变体 × b0-b6 | 77 |
| P2（历史越界探索） | 11 变体 × b3-b6 | 44 |

正式实验先固定 KNN 构造方法，不允许 ranker 在 family 或 KNN feature 之间选择；
`q_multi` 也只能在同一 family 的六个桥长候选中计算。跨 family 的旧 P1/P2 结果
不满足该约束。

## 3. 选择方法（几种方法都实验）

| 方法 | 打分/机制 | 拟合 | 说明 |
|---|---|---|---|
| M0 固定桥长 | 无选择，固定 b6（或报告最优桥） | 无 | 已有数字（lora target_pooling knn=cls b6 = 0.8978），作为对照下限 |
| M1 固定 0.5/0.5 | `0.5×q_return + 0.5×q_multi`，Top-1 | 无 | **旧默认公式移植（T21 `--alpha 0.5`），未在 256 新协议下校准，仅作参考基线，不参与结论** |
| M2 两信号线性校准 | `α×q_return + (1−α)×q_multi` | validation 0.05 网格 | 仿 T25 校准：按 validation 选择 Dice 优先，spearman/确定性破并列；报告校准出的 α |
| M3 ridge 路由器 | 绝对打分（candidate-invariant）：21 传播质量特征 + bridge/bottleneck/path-mean 几何 + 变体 one-hot | validation 拟合 ridge λ=1.0 | 新一代路由器；分别跑 P1(b0–b6) 与 P2(b3–b6) |
| M4 单信号消融 | 只用 q_return / 只用 q_multi / 只用 SAM score 各自 Top-1 | 无 | 信号强度参照（下界） |
| M5 学生辅助 B7（几何加权） | `(q_return × q_multi² × q_model²)^0.2` | 复用 256 协议 X3/S2/S3 | **已完成**；历史最好 0.9035，但最终 pairwise 方法不需要学生 |

候选排序纪律：同分时按 `(-bridge_count, route_id)` 确定性打破，与既有实现一致。

## 4. 执行步骤与成本估计

1. ★ 生成 validation 路线：11 变体，CPU 并行；**已完成**；
2. ★ 传播质量特征；**base 与 lora 均已完成**：
   - lora_p491_e20：11 变体 × (validation+test) × 700 ≈ 15,400 条，
     canvas 256 估计 ~1s/条 ≈ **4–5 小时 GPU**；
   - base：同量级；顺序在 lora 之后跑（总约 9–10 小时）；
   - 两 checkpoint 的 quality 目录用 tag 隔离（`propagation_quality_{split}_{tag}`）；
3. 选择/校准/分析：M1-M5、P1/P2、base/lora；**已完成**；
4. pairwise ranker、特征消融、冻结双数据集评估与 risk gate；**已完成**。

## 5. 防遗漏检查清单

- [x] validation 路线先生成（KNN 阶段没有）
- [x] 传播质量脚本输出目录按 checkpoint 加 tag，防止 base/lora 覆盖
- [x] M2 校准只在 validation 上做，test 一次性应用（防泄漏）
- [x] q_multi 用 (family, route_id) 区分掩码
- [x] M3 的 ridge 特征与旧脚本一致（21 传播质量特征 + 几何 + 变体 one-hot）
- [x] 新脚本已泛化到新根目录，不再依赖旧脚本的硬编码路径
- [x] 汇总表区分 checkpoint（base / lora_p491_e20）与候选池（P1/P2/P3）

## 6. 已知注意点

- `q_return`（= q_cycle）是回传一致性，占 GPU 大头（每条路线一次正向 trace +
  一次反向传播）；
- 传播质量特征依赖 checkpoint（lora 与 base 的 q_return/trace 不同），
  所以两个 checkpoint 必须各自算一遍，不能共用；
- **M1 的 0.5/0.5 是旧协议移植的固定权重（原始记录里没有两信号校准证据），
  在新协议下未校准、不保证最优**；结论只看 M2（validation 重新校准 α）
  与 M3（ridge，权重在 validation 拟合）；
- 已确认 phase-1 X3/S2/S3 学生就是 256 协议并完成 M5；但后续 pairwise 消融显示
  q_model 会降低 OOF 结果，所以学生不进入当前冻结方法。

## 7. 全部实验结果总览（截至 2026-08-12）

| 阶段 / 方法 | 训练或选择口径 | Kvasir-SEG | CVC-ClinicDB | 状态 |
|---|---|---:|---:|---|
| base 固定 b6 | 无选择 | 0.8126 | — | 完成，历史基线 |
| base 最优 `q_model_mean_only` P1 | 学生审核 | 0.8945 | — | 完成，依赖三学生 |
| LoRA 固定 b6 | 无选择 | 0.897780 | 0.815820 | 完成，统一对照 |
| LoRA M1 0.5/0.5 P1 | 固定公式 | 0.8953 | — | 完成，未超过 b6 |
| LoRA M3 ridge P2 | validation 拟合 | 0.8943 | — | 完成，未超过 b6 |
| LoRA B7_X3 P2 | 单学生审核 | 0.9022 | — | 完成，历史方法 |
| LoRA `lin_cal_mean` P1 | 三学生 + validation 校准 | 0.9035 | — | 完成，历史学生最优 |
| 合法单路线 pairwise S1 | validation 三 seed OOF 均值 | **0.859607**（val） | — | +0.015542 vs val b6，无学生 |
| 冻结单路线 pairwise S1 | Kvasir 历史 test 诊断 | **0.884591** | — | -0.013189，不如固定 b6 |
| 冻结单路线 pairwise S1 | ClinicDB 外部冻结测试 | — | **0.854612** | +0.038792，CI 下界 +0.000295 |
| 跨 family P2 ranker | 11 family × b3-b6 | 0.906307 | 0.857032 | **违反单 KNN 路线约束，仅保留为越界探索** |
| 跨 family risk gate | 基于上述越界候选池 | 0.900460 | 0.856862 | **同样不进入主结果** |

在合法约束下，单路线 pairwise 的 validation 和 ClinicDB 为正，但 Kvasir test 为负，
尚未形成跨数据集一致优势。因此当前最可靠的部署配置仍是固定
`anchor_conditioned_target_pooling__knn_cls+b6`；单路线 pairwise 只能作为待改进
方向，不能称为当前最好方法。

## 8. 中期诊断（lora_p491_e20，2026-08-10）

以下保留 2026-08-10 当时的中期节点；随后 base 与 LoRA 的正式报告均已完成。

### 8.1 lora 选择器结果（test Dice）

| 方法 | P1 (b0–b6) | P2 (b3–b6) |
|---|---:|---:|
| M1 固定 0.5/0.5 | **0.8953** | 0.8906 |
| M2 校准线性（α=1.0） | 0.8665 | 0.8704 |
| M3 ridge | 0.8933 | **0.8943** |
| M4 q_return | 0.8665 | 0.8704 |
| M4 q_multi | 0.8952 | 0.8885 |
| M4 sam_score | 0.8803 | 0.8897 |
| oracle | 0.9357 | 0.9347 |
| 对照：固定 b6（target_pooling knn=cls） | **0.8978** | — |

### 8.2 为什么选择器吃不到 oracle

- 信号判别力（spearman vs 真实 Dice）：q_multi 0.680，q_return 0.073，
  sam_score 0.108——LoRA 后 q_return 饱和（普遍 0.95–0.99），几乎不带信息；
  M2 校准因此选出 α=1.0（纯 q_return），test 反而最差；
- M1 vs 固定 b6：44 赢 / 56 输；
- 最大失误模式："自信的错误"——选择器挑的路线 q_return/q_multi 很高
  （0.98/0.9+）但真实 Dice 只有 0.3–0.7；oracle 赢家常是**低共识孤例**
  （q_multi 0.37–0.4，甚至 q_return=0），共识类信号系统性地排低它们；
- 尝试过的替代信号均未超过 0.8978：
  - 跨 checkpoint 一致（lora vs base 掩码 Dice）：spearman 0.55，混合后
    最高仍 0.8952（纯 q_multi）；
  - 共识 Top-2/Top-3 掩码多数投票平均：0.8945–0.8953。

### 8.3 结论与下一步

- 当前 lora 最高仍是固定 b6 = 0.8978，选择器/集成全部卡在 0.89–0.895；
- 差距来源：缺一个能识别"低共识但正确"孤例路线的信号；现有无 GT 信号
  （共识、回传、SAM score、跨 checkpoint）都识别不了；
- 最可能有效的下一步：**训练 256 协议的单图学生，用 B7 的 q_model
  审核信号**（老主线里最强的选择信号，对应文档 M5），再试能否突破 0.8978。

### 8.4 学生审核（M5/B7）——已突破固定 b6（2026-08-10）

发现现有 phase-1 学生（X3/S2/S3）本身就是 256×256 训练，且 test/validation
预测正好覆盖本协议的 100 个 Kvasir target，因此无需重训，直接复用：

| 方法 | P1 (b0–b6) | P2 (b3–b6) |
|---|---:|---:|
| B7_X3（几何，X3 审核） | 0.8983 | **0.9022** |
| B7_mean（几何，三学生均值） | 0.9014 | 0.8921 |
| lin_cal_X3（validation 校准） | 0.8954 | 0.8982 |
| **lin_cal_mean（validation 校准，三学生均值）** | **0.9035** | 0.8704 |
| q_model_X3 only | 0.8927 | 0.8940 |
| q_model_mean only | 0.8948 | 0.8949 |
| q_multi only（B7 空掩码约定） | 0.8886 | 0.8885 |
| oracle | 0.9357 | 0.9347 |
| 对照：固定 b6 | **0.8978** | — |

要点：

- **lin_cal_mean P1 = 0.9035 为当前最高**（validation 校准权重：
  q_return 0.80 / q_multi 0.05 / q_model 0.15），B7_mean P1 = 0.9014，
  B7_X3 P2 = 0.9022，三者均超过固定 b6（+0.004~+0.006）；
- q_model（学生 vs 路线掩码 Dice）spearman ≈ 0.69，与 q_multi 相当，
  组合后能修正一部分"自信的错误"；
- 空掩码约定：老 B7 主线为"两个空掩码一致=1.0"（本报告采用），
  早期 M1–M4 分析用 T21 约定"空=0"；test 中空掩码占 1.94%，
  这是 q_multi-only 数值差异（0.8886 vs 0.8952）的来源，非 bug；
- 结论：**以 lora + 256 学生审核为准，新的主配置候选为
  lin_cal_mean P1（0.9035）/ B7_mean P1（0.9014）**。

## 9. 正式报告产物（lora_p491_e20，2026-08-10）

报告文件（`work/kvasir_1pct_anchors/route_selector_vitb256/`）：

- `route_selector_report_lora_p491_e20.md` —— 完整正式报告
  （协议、方法对比、最优配置、逐 target 胜负、路线分布、信号判别力、
  validation 稳定性）；
- `route_selector_report_lora_p491_e20.json` —— 最优配置摘要；
- `route_selector_best_lora_p491_e20.csv` —— 最优配置（lin_cal_mean P1）
  的 100 target 逐条明细（所选 family/桥长/Dice、固定 b6、oracle、
  q_return/q_multi/q_model_mean）；
- 原始数据：`b7_student_audit_lora_p491_e20.json`（所有方法 + 每 target
  选择明细）、`selector_report_lora_p491_e20.json`（M1–M4）。

### 9.1 报告要点（与 8.4 一致，补充逐 target 细节）

| 方法 | P1 (b0–b6) | P2 (b3–b6) |
|---|---:|---:|
| 固定 b6（对照） | 0.8978 | 0.8978 |
| M1 固定 0.5/0.5 | 0.8953 | 0.8906 |
| M2 两信号校准 | 0.8665 | 0.8704 |
| M3 ridge | 0.8933 | 0.8943 |
| B7_X3 | 0.8983 | **0.9022** |
| B7_mean | 0.9014 | 0.8921 |
| **lin_cal_mean** | **0.9035** | 0.8704 |
| oracle | 0.9357 | 0.9347 |

- 最优配置：`lin_cal_mean` @ P1 = **0.9035**（gap 0.0323），
  validation 校准权重 q_return 0.80 / q_multi 0.05 / q_model 0.15；
- vs 固定 b6：43 胜 / 57 负，平均 +0.0057；最大提升 +0.47，最大回退 -0.19；
- 选中路线分布：t18 系 45/100、dino_patch_average 15、dino_global_pooling 12，
  桥长覆盖 b0–b6（不再依赖单一 b6）；
- 稳定性参考（validation）：B7_X3 0.861 / B7_mean 0.853 / q_multi 0.833，
  相对提升方向与 test 一致；
- base checkpoint 的传播质量与正式报告已经完成，结果补于 9.2。

### 9.2 Base checkpoint 正式结果（已完成）

base 使用与 LoRA 相同的 Kvasir test 100 targets、11 family、P1/P2 和 256 协议。
固定 `anchor_conditioned_target_pooling__knn_cls+b6` 为 `0.8126`，P1 oracle 为
`0.9291`。主要结果如下：

| 方法 | P1 (b0-b6) | P2 (b3-b6) |
|---|---:|---:|
| 固定 b6 | 0.8126 | 0.8126 |
| M1 固定 0.5/0.5 | 0.8432 | 0.8422 |
| M2 两信号校准 | 0.8417 | 0.8415 |
| M3 ridge | 0.8816 | 0.8850 |
| q_return only | 0.8285 | 0.8336 |
| q_multi only（T21 空掩码约定） | 0.8462 | 0.8459 |
| SAM score only | 0.8540 | 0.8663 |
| q_multi only（B7 空掩码约定） | 0.8309 | 0.8267 |
| q_model X3 only | 0.8892 | 0.8870 |
| **q_model mean only** | **0.8945** | **0.8932** |
| B7_X3 | 0.8889 | 0.8880 |
| B7_mean | 0.8909 | 0.8833 |
| lin_cal_X3 | 0.8892 | 0.8879 |
| lin_cal_mean | 0.8899 | 0.8679 |
| oracle | 0.9291 | 0.9210 |

base 最优为 `q_model_mean_only @ P1 = 0.894492`，相对 base 固定 b6 平均
`+0.081871`，72 胜 / 1 平 / 27 负，gap 为 `0.034614`。该结果说明学生审核能够
大幅修复较弱 base checkpoint，但它依赖三个学生，且不是后续论文主方法。
`route_selector_report_base.md` 首段的 `0.8573` 是旧文本残留；方法表、JSON 与
`route_selector_best_base.csv` 的 100 个 `fixed_b6_dice` 复算均为 `0.8126205`，
本文统一以后者为准。

产物位于 `work/kvasir_1pct_anchors/route_selector_vitb256/`：
`route_selector_report_base.{md,json}`、`route_selector_best_base.csv`、
`b7_student_audit_base.json` 和 `selector_report_base.json`。

## 10. Pairwise route ranker 第一关（validation-only OOF，2026-08-10）

> **作废说明**：本节 10.1-10.6 的原始 P2 实验把 11 个 family 当成 44 个候选，
> 违反“只能固定一种 KNN 路径”的约束。为保证审计透明，数字保留但全部视为越界
> 探索，不进入论文主表。纠正后的单路线实验见第 14 节。

为避免继续在 test 上试选择公式，新增
`scripts/run_pairwise_route_ranker_vitb256.py`，只读取 validation，按 target 做
5-fold outer / 4-fold inner nested CV。GT Dice 只用于训练 fold 内构造候选对标签
和 held-out fold 的最终统计；candidate 不允许跨 target 随机拆分，test split 未加载。

### 10.1 口径

- checkpoint：`lora_p491_e20`；候选池固定为 P2，即 11 family × b3-b6 =
  44 candidates/target；
- 11 family 包括 `t18_corrected`、两个 DINO family，以及 target-pooling / 
  patch-correspondence 各自的 `patch_mean`（默认）、`knn_cls`、`knn_pooled`、
  `knn_cond`；ranker 在全部 44 个候选中选择，不是只选某一种 KNN 路线；
- 固定对照明确为 `anchor_conditioned_target_pooling__knn_cls + b6`：
  validation Dice = **0.844065**。同一路线此前的 **0.8978** 是 test Dice，
  两者属于不同 split，不可直接横比；
- pairwise 标签只保留同一 target 内 `|Dice_i-Dice_j| >= 0.02` 的候选对，
  每个 target 的 pair loss 归一化后等权；
- quality-only 输入：传播 trace、q_return、q_multi、SAM score、mask 几何、
  相对固定 b6 的差异、family/bridge 身份；qmodel 版本额外加入
  X3/S2/S3 mean student agreement；
- inner CV 只选择 L2 与 b6 fallback threshold；所有汇报值均为 outer-fold
  held-out target 拼接后的 OOF 结果。

### 10.2 三个 outer-fold 随机种子的稳定性

| seed | fixed b6 | pairwise quality-only | delta | 95% bootstrap CI | W/T/L | catastrophic (`delta < -0.05`) |
|---:|---:|---:|---:|---:|---:|---:|
| 20260810 | 0.844065 | **0.880443** | +0.036378 | [+0.011465, +0.063539] | 61/4/35 | 3 |
| 20260811 | 0.844065 | **0.885300** | +0.041235 | [+0.018957, +0.068132] | 66/2/32 | 0 |
| 20260812 | 0.844065 | **0.877041** | +0.032976 | [+0.007191, +0.060993] | 66/3/31 | 4 |
| 三 seed 均值 | 0.844065 | **0.880928** | **+0.036863** | — | — | — |

首个 seed 的五个 held-out fold delta 分别为
`+0.0056 / +0.0564 / +0.0604 / +0.0546 / +0.0050`，五折均为正；最大单样本
提升占全部正增益约 15%，占净增益约 19%，不再由单一样本决定整体符号。

### 10.3 学生信号与线性基线

| 方法 | seed 20260810 | seed 20260811 | seed 20260812 |
|---|---:|---:|---:|
| 三信号线性（outer-train 校准） | 0.851824 | 0.857150 | 0.854734 |
| pairwise + quality-only | **0.880443** | **0.885300** | **0.877041** |
| pairwise + q_model | 0.877096 | 0.880536 | 0.875222 |
| pairwise + quality-only + fallback | 0.879641 | 0.885080 | **0.881702** |

当前结论与原先预期不同：**第一关支持 pairwise 排序目标，但不支持“学生是必要
增益来源”**。三个 seed 中加入 q_model 都低于 quality-only，因此暂不进入学生蒸馏；
下一步应先做 quality-only 的特征组消融与新数据集冻结评估。Kvasir test 已被历史实验
反复查看，只能作为工程诊断，不能用它继续选择 ranker 配置。

### 10.4 产物

- 脚本：`scripts/run_pairwise_route_ranker_vitb256.py`；
- 主 seed 报告：
  `work/kvasir_1pct_anchors/pairwise_ranker_vitb256/`
  `pairwise_ranker_validation_oof_lora_p491_e20_b3_b6.{md,json,csv}`；
- 稳定性复跑：同目录下 `seed20260811/`、`seed20260812/`；
- CSV 包含每个 target、每个方法的 OOF 所选 family/bridge、selected Dice、
  fixed b6 Dice、delta 与是否触发 fallback，可用于后续失败案例和路线分布分析。

### 10.5 特征块消融：精简方法优于学生版本

使用同一个 seed=20260810、相同 outer/inner folds，逐块增加特征。首次消融因
不同特征块误用了不同 inner-fold seed 已作废；下表为修正后、所有配置共享相同
inner folds 的结果，`quality_plus_relative=0.880443` 精确复现 10.2 主实验。

| Pairwise 输入 | OOF Dice | delta vs b6 | 95% bootstrap CI | W/T/L | catastrophic |
|---|---:|---:|---:|---:|---:|
| route family + bridge 先验 | 0.854522 | +0.010457 | [-0.012094, +0.034434] | 54/0/46 | 7 |
| 内部传播质量 | 0.879969 | +0.035904 | [+0.013941, +0.060389] | 59/0/41 | 3 |
| **内部传播质量 + q_multi** | **0.891082** | **+0.047017** | **[+0.023583, +0.074093]** | **63/1/36** | **1** |
| 上项 + 相对 b6/mask 几何 | 0.880443 | +0.036378 | [+0.011928, +0.064072] | 61/4/35 | 3 |
| 上项 + q_model | 0.874183 | +0.030118 | [+0.004583, +0.057567] | 62/2/36 | 5 |

最佳精简配置再换两个 outer-fold seed：

| seed | OOF Dice | delta vs b6 | 95% bootstrap CI | W/T/L | catastrophic |
|---:|---:|---:|---:|---:|---:|
| 20260810 | 0.891082 | +0.047017 | [+0.023583, +0.074093] | 63/1/36 | 1 |
| 20260811 | 0.892173 | +0.048108 | [+0.024858, +0.075778] | 62/3/35 | 0 |
| 20260812 | 0.891281 | +0.047216 | [+0.023398, +0.074452] | 59/5/36 | 2 |
| 三 seed 均值 | **0.891512** | **+0.047447** | — | — | — |

选择没有退化为单一 KNN 路线：seed 20260810 中，family 分布为
`dino_global_pooling=29`、`dino_patch_average=25`、`t18_corrected=23`、
`patch_correspondence__knn_cond=12`、`patch_correspondence__knn_pooled=6`，
其余 5；桥长分布为 b3/b4/b5/b6 = `22/11/24/43`。route/bridge 先验单独
不显著，说明 held-out 增益主要来自逐候选传播 trace 与 q_multi，而不是固定选择
某个 family。

当时冻结的越界探索配置是：**P2 44 candidates + pairwise logistic ranking +
内部传播质量 + q_multi，不使用学生、不使用相对 b6 特征、不使用 fallback**。
学生蒸馏只保留为后续备选，不再是当前方法成立的前提。完整消融产物位于
`work/kvasir_1pct_anchors/pairwise_ranker_vitb256/feature_ablation/`，两个稳定性
复跑位于同级 `feature_ablation_seed20260811/` 与
`feature_ablation_seed20260812/`。

已按上述冻结配置在全部 100 个 validation target 上完成最终拟合（只用 inner-CV
选 L2，不加载 test）：L2=`0.001`、40 个输入特征、fallback 关闭、优化器成功，
所有权重/标准化参数均为有限值。可部署模型保存在
`work/kvasir_1pct_anchors/pairwise_ranker_vitb256/frozen_internal_consensus/`
`frozen_internal_plus_consensus_lora_p491_e20.json`，其中包含完整 feature order、
mean/scale 与 weights，后续新数据评估不再重新选配置。

### 10.6 冻结模型 Kvasir test 诊断

冻结后使用 `eval_frozen_pairwise_route_ranker_vitb256.py` 一次性加载 P2 test
候选；没有重新拟合、校准、加载学生或改变特征。主配置（无 fallback）结果：

| 方法 | Test Dice | delta vs fixed b6 | 95% bootstrap CI | W/T/L | trimmed delta | catastrophic |
|---|---:|---:|---:|---:|---:|---:|
| fixed b6 | 0.897780 | — | — | — | — | — |
| 旧 lin_cal_mean P1 | 0.9035 | +0.0057 | — | 43/0/57 | — | — |
| **冻结 pairwise P2** | **0.906307** | **+0.008527** | **[-0.014469, +0.031172]** | **53/6/41** | **+0.000965** | **4** |
| 冻结 pairwise + validation 阈值 0.65（次要诊断） | 0.906634 | +0.008854 | [-0.014215, +0.031496] | 50/12/38 | +0.001169 | 4 |

点估计刷新当前结果，但证据仍弱：最大单样本增益 `+0.6313` 占净增益约 74%，
最大回退 `-0.7392`；validation 预先保存的 margin 阈值也无法拦截该回退，因为
错误切换概率高达 `0.9926`。因此本结果只能表述为“冻结模型能迁移并提高点估计”，
不能表述为统计显著或已解决稳定选择问题。并且该 Kvasir test 已被早期方法反复
查看，继续保留 `historical_test_diagnostic` 标记。

结果位于
`work/kvasir_1pct_anchors/pairwise_ranker_vitb256/frozen_test_diagnostic/`。

## 11. ClinicDB 外部迁移（冻结评估完成，2026-08-11）

> **作废说明**：本节冻结 ranker 仍使用跨 11 family 的 P2 候选池，因此不属于合法
> 单路线结果。正确的 ClinicDB 单路线冻结结果见第 14.3 节。

已冻结独立协议 `work/clinicdb_external_kvasir8/protocol/`：

- bridge pool：Kvasir-SEG train 800；
- human support：原 Kvasir 8 anchors；
- target：CVC-ClinicDB test 61；
- ClinicDB train/validation 不进入图、特征、桥节点或调参；GT 仅作最终评估；
- 候选与主实验完全同口径：LoRA `lora_p491_e20`、ViT-B@256、11 family、
  P2 b3-b6、44 candidates/target；最终直接加载 10.5 冻结的 40 维 ranker。

新增脚本：

- `prepare_clinicdb_external_kvasir8_protocol.py`：构建并校验外部协议；
- `run_clinicdb_external_pairwise_vitb256.sh`：生成路线、传播质量并评估；
- `queue_clinicdb_external_pairwise_vitb256.sh`：等待当前
  `run_route_selector_vitb256.sh` base 任务退出后自动启动，避免争抢单张 RTX 3090。

用户确认直接并发后，等待队列 PID=`372149` 已停止；完整流水线 PID=`391900`
于 2026-08-11 01:44:53 正常结束，无 OOM。11 个 family 共生成 2684 条路线；
冻结评估全程没有拟合、校准、加载学生或根据 ClinicDB GT 修改模型、L2、特征、
候选池和 fallback。

### 11.1 冻结外部测试结果

| 方法 | ClinicDB Dice | delta vs fixed b6 | 95% bootstrap CI | W/T/L | median delta | trimmed delta | catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed `target_pooling__knn_cls+b6` | 0.815820 | - | - | - | - | - | - |
| **冻结 pairwise P2** | **0.857032** | **+0.041213** | **[-0.005220, +0.096811]** | **36/3/22** | **+0.000978** | **+0.002896** | **2** |
| oracle | 0.912831 | +0.097011 | - | - | - | - | - |

外部集点估计明显为正，且 36 胜、22 负，说明冻结 ranker 不只在 Kvasir 上可运行；
但 61 个 target 的置信区间仍跨 0，最大增益 `+0.9495`、最大回退 `-0.4962`，
10% trimmed delta 仅 `+0.0029`，平均增益仍受少数大幅改善样本影响。因此当前最准确
的论文表述是：**独立语义一致性监督学到的轻量 pairwise ranker 在未见 ClinicDB 上
取得 +4.12 Dice 点的正向外部点估计，但有限样本下尚未达到统计显著，尾部稳定性仍
需改进。** 不应把它写成已经稳定显著优于固定 b6。

结果位于 `work/clinicdb_external_kvasir8/pairwise_ranker_vitb256/`
`frozen_pairwise_lora_p491_e20_test.{md,json,csv}`，运行日志为
`work/clinicdb_external_kvasir8/direct_run.log`。

## 12. Baseline-aware risk gate：Kvasir-SEG 与 ClinicDB 联合验收（2026-08-11）

> **降级说明**：本节 risk gate 建立在越界的 44-candidate ranker 上，不能作为当前
> 主方法；仅作为跨 family 探索的风险诊断保留。

为避免只针对 ClinicDB 的两个坏例调 fallback，新增
`scripts/run_pairwise_risk_gate_vitb256.py`。训练与阈值选择仍只使用 Kvasir
validation 的 5-fold target-grouped OOF，并重复 seed `20260810/11/12`；同一 target
的 44 个候选不会跨 fold。冻结后，同一份 gate 原样评估 Kvasir-SEG test 和
CVC-ClinicDB，两个测试集均不参与拟合。推理期不加载学生。

方法由三个步骤组成：

1. 冻结 pairwise ranker 先从 P2 的 44 个候选中选择候选 A；
2. 线性 harm critic 估计 `Dice(A)-Dice(b6)<-0.05` 的风险，输入包括内部传播质量、
   q_multi、候选与 b6 的 mask/质量差异及路线先验；
3. 若 b6 最终 mask 为空，则禁止回退到无效 baseline；否则仅在 harm probability
   大于 validation 冻结阈值 `0.10` 时回退 b6。

| Dataset | fixed b6 | raw ranker | risk-controlled | raw delta | controlled delta | catastrophic | CVaR10 | fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kvasir-SEG | 0.897780 | 0.906307 | **0.900460** | +0.008527 | **+0.002680** | 4→1 | -0.123855→**-0.022701** | 16 |
| CVC-ClinicDB | 0.815820 | 0.857032 | **0.856862** | +0.041213 | **+0.041042** | 2→0 | -0.107111→**-0.013229** | 12 |

三个 validation OOF seed 的 controlled delta 分别为 `+0.004376/+0.002834/+0.006388`，
catastrophic 均为 0。联合结果说明 gate 不是只修 ClinicDB：两个数据集的最坏尾部都
明显收缩，且平均 Dice 仍高于固定 b6。不过代价在 Kvasir 上较明显，只保留原 ranker
约 31% 的平均增益；因此应将其报告为 **risk-controlled operating point**，与无 gate
的 accuracy operating point 并列，而不是宣称它无代价地替代原 ranker。

额外尝试了 learned rescue critic（`delta>+0.05`），它在 validation OOF 中能恢复
大增益，但跨数据集把坏例误判为救援，Kvasir 仅 `4→3`、ClinicDB 仍为 `2→2`，故判定
失败并不进入主配置。失败消融保存在同一目录的
`risk_gate_dual_dataset_learned_rescue_failed_lora_p491_e20.{md,json,csv}`。

主结果位于
`work/kvasir_1pct_anchors/pairwise_ranker_vitb256/risk_gate_dual_dataset/`
`risk_gate_dual_dataset_lora_p491_e20.{md,json,csv}`，冻结 gate 为
`frozen_risk_gate_lora_p491_e20.json`。

## 13. 跨 family 探索的历史结论与产物索引（已被第 14 节纠正）

### 13.1 当时观察（以下五点均受跨 family 口径限制）

1. 线性混合 `q_return/q_multi/q_model` 能在 LoRA Kvasir test 上超过固定 b6，
   但最好结果 `0.9035` 依赖三名学生，方法叙事较弱；
2. validation-only pairwise ranker 直接学习候选偏序，最佳精简输入仅为内部传播
   质量 + q_multi，不使用学生。三个 OOF seed 均稳定提升约 `+0.047`；
3. 同一冻结 ranker 在 Kvasir 历史 test 和未见 ClinicDB 上都提高平均点估计，分别
   为 `+0.0085` 与 `+0.0412`，但原始置信区间均跨 0，且存在少数灾难性回退；
4. baseline-aware harm gate 在不加载学生的前提下，同时将 Kvasir/ClinicDB 的
   catastrophic 数量从 `4/2` 降至 `1/0`，并保持两边平均 Dice 高于固定 b6；
5. risk gate 在 Kvasir 上牺牲较多平均增益，因此最终应并列报告 accuracy mode 与
   risk-controlled mode，不能声称已经无代价地解决稳定路由。

本节原先形成的论文主线建立在跨 family 候选池上。由于该候选池不满足“只能使用
一种 KNN 路径”，现在只能作为后续方法灵感，不能直接写入论文主实验。

### 13.2 脚本

- `scripts/run_pairwise_route_ranker_vitb256.py`：validation grouped nested-OOF
  pairwise ranker；
- `scripts/run_pairwise_route_ranker_ablation_vitb256.py`：特征块消融与冻结模型导出；
- `scripts/eval_frozen_pairwise_route_ranker_vitb256.py`：冻结 ranker 测试评估；
- `scripts/prepare_clinicdb_external_kvasir8_protocol.py`：冻结 ClinicDB 外部协议；
- `scripts/run_clinicdb_external_pairwise_vitb256.sh`：ClinicDB 路线、质量与评估；
- `scripts/queue_clinicdb_external_pairwise_vitb256.sh`：GPU 队列启动器；
- `scripts/run_pairwise_risk_gate_vitb256.py`：validation-only harm gate 与
  Kvasir/ClinicDB 双数据集冻结验收；
- `scripts/stage1_feature_knn_routes.py`：增加可选 `--protocol-root`，默认行为不变。

### 13.3 结果目录

- 旧选择器 base/LoRA 正式报告：
  `work/kvasir_1pct_anchors/route_selector_vitb256/`；
- pairwise 三 seed OOF：
  `work/kvasir_1pct_anchors/pairwise_ranker_vitb256/`、`seed20260811/`、
  `seed20260812/`；
- 三 seed 特征消融：`feature_ablation/`、`feature_ablation_seed20260811/`、
  `feature_ablation_seed20260812/`；
- 冻结 40 维 ranker：`frozen_internal_consensus/`
  `frozen_internal_plus_consensus_lora_p491_e20.json`；
- Kvasir 冻结 test 诊断：`frozen_test_diagnostic/`；
- ClinicDB 外部协议与测试：`work/clinicdb_external_kvasir8/protocol/` 和
  `work/clinicdb_external_kvasir8/pairwise_ranker_vitb256/`；
- 双数据集 risk gate、冻结 gate 与 learned-rescue 失败消融：
  `work/kvasir_1pct_anchors/pairwise_ranker_vitb256/risk_gate_dual_dataset/`。

所有表格中的 Dice、delta、CI、W/T/L、catastrophic、trimmed mean 与 CVaR10 均可
由上述 JSON/CSV 逐样本产物复核。

## 14. 口径纠正：单一 KNN 路线的 pairwise ranker（2026-08-12）

### 14.1 合法协议

- 唯一路径构造方法固定为
  `anchor_conditioned_target_pooling__knn_cls`，即 target pooling + ViT-B CLS
  KNN；ranker 不允许选择其他 family、mode 或 KNN feature；
- 主候选池 S1 为 b1-b6 共 6 个桥长。b0 是 anchor→target direct prediction，
  没有 KNN bridge，只作为 S0=b0-b6 的额外诊断；
- `q_multi` 在每个 target 的这 6 个同 family 掩码之间重新计算，不使用其他 family；
- 固定对照为同 family 的 b6；validation Dice `0.844065`，Kvasir test
  `0.897780`，ClinicDB test `0.815820`；
- 训练仍为 target-grouped 5-fold outer / 4-fold inner nested CV；测试评估使用
  validation 全量拟合后冻结的同一模型，不加载学生。

### 14.2 Validation OOF

固定 seed 20260810 的特征消融：

| 输入 | S0 b0-b6 Dice / delta | S1 b1-b6 Dice / delta | S1 95% CI | S1 catastrophic |
|---|---:|---:|---:|---:|
| route/bridge prior | 0.844065 / +0.000000 | 0.844065 / +0.000000 | [0,0] | 0 |
| 内部传播质量 | 0.847394 / +0.003329 | 0.849416 / +0.005351 | [-0.001162,+0.014105] | 2 |
| **内部传播质量 + q_multi** | 0.839977 / -0.004088 | **0.860942 / +0.016877** | **[+0.001761,+0.039717]** | **1** |
| 上项 + relative-b6/mask | 0.851608 / +0.007543 | 0.857367 / +0.013302 | [-0.002896,+0.036693] | 3 |

b0 混入后最佳配置发生退化，进一步说明 direct 不应放入 KNN route ranker 主池。
S1 的最佳 `internal_plus_consensus` 再复跑两个 outer-fold seed：

| seed | fixed b6 | 单路线 pairwise S1 | delta | 95% CI | catastrophic |
|---:|---:|---:|---:|---:|---:|
| 20260810 | 0.844065 | **0.860942** | +0.016877 | [+0.001761,+0.039717] | 1 |
| 20260811 | 0.844065 | **0.857833** | +0.013768 | [-0.001432,+0.037623] | 3 |
| 20260812 | 0.844065 | **0.860045** | +0.015980 | [+0.000758,+0.039090] | 1 |
| 三 seed 均值 | 0.844065 | **0.859607** | **+0.015542** | — | — |

三个 seed 的点估计方向一致，但增益受少数样本影响，trimmed delta 仅约
`+0.0003` 至 `+0.0009`，不能只凭均值宣称稳定提升。

### 14.3 冻结双数据集结果

在全部 Kvasir validation target 上冻结 30 维 `internal_plus_consensus` ranker，
candidate pool 固定为同一 family 的 b1-b6，L2=`0.001`，无学生、无 fallback。

| Dataset | fixed b6 | frozen single-route ranker | delta | 95% CI | W/T/L | catastrophic | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kvasir-SEG test（历史诊断） | **0.897780** | 0.884591 | **-0.013189** | [-0.037232,+0.006498] | 37/28/35 | 6 | 0.913355 |
| CVC-ClinicDB test（外部） | 0.815820 | **0.854612** | **+0.038792** | [+0.000295,+0.087529] | 26/18/17 | 3 | 0.866798 |

合法单路线 ranker 在 ClinicDB 上有显著为正的冻结点估计，但在 Kvasir test 上下降，
说明当前模型存在数据集依赖，尚未取得跨数据集一致优势。**当前最好且最可靠的配置
仍是固定 `target_pooling__knn_cls+b6`；单路线 pairwise 不能作为已成立的主方法。**

### 14.4 合法实验产物

- S0 b0-b6 消融：`pairwise_ranker_vitb256/single_knn_cls_b0_b6_all/`；
- S1 b1-b6 主消融：`pairwise_ranker_vitb256/single_knn_cls_b1_b6_all/`；
- S1 另外两个 seed：`single_knn_cls_b1_b6_seed20260811/`、
  `single_knn_cls_b1_b6_seed20260812/`；
- 冻结模型：`single_knn_cls_b1_b6_frozen/`
  `frozen_internal_plus_consensus_lora_p491_e20.json`；
- Kvasir 冻结诊断：`single_knn_cls_b1_b6_frozen_test/`；
- ClinicDB 冻结评估：`work/clinicdb_external_kvasir8/pairwise_ranker_vitb256/`
  `single_knn_cls_b1_b6/`；
- ClinicDB 已扩展为该 family 的完整 b0-b6 共 427 条质量记录，其中正式评估仅加载
  b1-b6。

本 Markdown 是截至 2026-08-12、经过单一 KNN 路线约束纠正后的完整实验总账。
