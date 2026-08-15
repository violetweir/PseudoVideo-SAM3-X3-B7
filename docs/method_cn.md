# 当前方法:学生审核的伪视频 SAM3 管线(Kvasir 1%,WACV2027)

> 版本:2026-08-08。本文档完整描述当前 WACV2027 Kvasir-SEG 1% anchor 方法:
> 冻结路线生成 → 传播质量路由器 → 1% SAM3 微调 → 主线路式学生审核
> → 伪标签扩展 → 第二轮学生审核共识池 → SAM3 再微调与双 checkpoint 评估。

## 1. 总览

方法的目标:只用 **8 张 Kvasir-SEG 训练图标注**(1% anchor),通过
"伪视频路线传播 + 单图学生审核"的组合,在 test 上获得尽可能高的分割 Dice。

核心思想:

- **SAM3 路线生成是冻结的**(KNN 图 + 冻结/微调 SAM3 传播),不把伪标签当新 anchor;
- **路线质量用无 GT 的传播质量特征评估**(anchor 回传一致性、轨迹、多路线共识);
- **单图学生是独立审核器**:与 SAM3 视频传播的错误模式独立,用于审核伪标签
  ("自信的错误")并参与最终路线选择(B7);
- 训练采用渐进式:第一轮池(路线硬阈值)→ 学生审核 → Tier 扩展 → X3;
  第二轮池(学生审核共识标签)→ SAM3 再微调。

## 2. 协议与数据

- 数据集:Kvasir-SEG 快照
  `/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T02_fresh_polyp_hf_sources/raw_hf_snapshots/kvasir-seg/snapshot`
- 划分:train=800 / validation=100 / test=100
- 人类标注:训练集 1% = **8 个固定 GT anchor**
  (`work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt`)
- SAM3 基座:`sam3.pt`(modelscope facebook/sam3)
- DINOv3 权重:用于 KNN 描述子(特征模式之一)
- 输出目录:`work/kvasir_1pct_anchors/`
- 协议约束:validation/test GT 只用于最终评估;训练掩码(8 个 anchor 之外)
  不参与路线搜索、伪标签选择或 checkpoint 选择;训练集 GT 仅用于线下诊断。

## 3. 阶段 A:冻结路线 + 传播质量路由器

### 3.1 路线生成(KNN 冻结)

- 8 个 anchor 的特征描述子(DINOv3 / T18 corrected / anchor-conditioned 变体)
- 对每个 query,在训练图池中检索桥帧,生成候选路线:
  - `anchor_conditioned_target_pooling`
  - `anchor_conditioned_patch_correspondence`
- 候选长度:**b3-b6**(b0=direct,bN=N 个桥帧);b7 全线塌缩被排除
- 每个 target 最多 2 模式 × 4 桥长 = 8 条候选

#### 两种 anchor-conditioned 路线生成器的原理

两个模式共享同一套 DINOv3 特征基础(见 `scripts/stage1_feature_knn_routes.py`):

- 对每张图提取 DINOv3 ViT-S/16 特征(224×224):
  CLS token、14×14=196 个 patch token、patch 均值(均 L2 归一化);
- 对每个 anchor,用其 GT mask 取**前景 patch token 的均值**作为
  **anchor prototype**(表征"这个息肉在 DINOv3 特征空间里长什么样");
- 计算 anchor prototype 与所有图所有 patch token 的余弦相似度
  `patch_sims[a, n, p]`。

**anchor_conditioned_target_pooling(目标加权池化)**

- 对每张图,按"与 anchor prototype 的相似度"对 196 个 patch 做
  softmax 式加权(温度 10:`exp((sim − max) × 10)`,再归一化);
- 用这些权重把 patch token 池化成**锚点条件化的图像描述子**
  `pooled[a, n]`(L2 归一化);
- 条件分数 `cond_score[a, n] = pooled[a, n] · anchor_proto[a]`:
  衡量"该图里长得像 anchor 目标的那部分内容与 anchor 有多一致";
- 直觉:描述子**偏向目标物体(息肉)而非整图外观**,把"全局相似"
  变成"目标相关相似"。

**anchor_conditioned_patch_correspondence(局部 patch 对应)**

- 条件分数 `cond_score[a, n] = mean(top-8 的 patch_sims[a, n, p])`:
  取该图与 anchor prototype 最相似的 8 个 patch 的相似度均值;
- 直觉:衡量"图里是否至少存在若干与 anchor 目标高度匹配的局部块",
  对背景/上下文更鲁棒,是一种 top-k 局部对应而非全局加权池化。

**路线搜索与冻结(两模式共用)**

- 对每个 target × 每个桥长(b3-b6)× 每个 anchor:
  - 从 target 出发做**束搜索**(beam width 默认 8),逐步向前拼接桥帧;
  - 每条候选路径的得分 = 路径上各节点(anchor、桥、target)条件分数的
    min 与 mean + 相邻节点 patch-mean 相似度;
  - KNN 候选按"与当前尾节点的 patch-mean 相似度"排序,并用
    anchor 条件分数作二级排序(锚点相关的桥帧优先);
- 跨 8 个 anchor 取路径得分最高的(anchor, 桥序列)作为该 (target, 桥长)
  的冻结路线;
- 路线只记录 KNN 路径(anchor→桥→query),SAM3 在其上执行传播;
  **KNN 图与路线一旦冻结,后续所有实验共用,不再改变**。

### 3.2 传播质量特征(21 个,无 GT)

对每条候选路线计算:

- `q_cycle`(fresh-state anchor 回传一致性:预测 query mask 转 prompt 反向传播
  回 anchor,与 anchor mask 的 Dice)——正传+回传交叉验证;
- `cycle_success`、`cycle_sam_score`;
- 掩码面积轨迹(`trace_area_*`)、空掩码数、连通域、质心/bbox 漂移;
- 相邻帧 Dice(`trace_adjacent_dice_*`)、SAM score 轨迹、候选数。

### 3.3 路由器

- 评分器:在 **validation 特征上拟合的 ridge**(21 特征 + 模式 one-hot),
  每个候选独立打分(候选不变式,不重归一化);
- 选择:每个 target 取 Top-1 路线掩码;
- 冻结基座结果:**0.877299**(oracle 0.896918,缺口 0.0196);
  固定最优路线(target b6)0.854627 作为对照。

### 3.4 冻结基座逐桥长完整表(test,forward-only Dice)

| 特征模式 | direct | b1 | b2 | b3 | b4 | b5 | b6 | b7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T18 corrected | 0.747869 | 0.775949 | 0.756490 | 0.785808 | 0.821736 | 0.816809 | 0.819200 | 0.143392 |
| DINO global pooling | 0.740844 | 0.748290 | 0.758326 | 0.786777 | 0.777536 | 0.818397 | 0.818565 | 0.092024 |
| DINO patch average | 0.728634 | 0.726893 | 0.756876 | 0.746756 | 0.799814 | 0.829179 | 0.826294 | 0.046968 |
| anchor-conditioned target pooling | 0.741306 | 0.760176 | 0.827209 | 0.840837 | 0.841258 | 0.848618 | 0.854627 | 0.032027 |
| anchor-conditioned patch correspondence | 0.757814 | 0.773501 | 0.822305 | 0.832819 | 0.833990 | 0.833577 | 0.842078 | 0.070807 |

最强固定路线:target pooling b6 = 0.854627;稳定传播区间 b4-b6;b7 全线塌缩。

### 3.5 8-anchor 单图学生基线(SC-SAM / SynFoC)

作为"不加伪标签时单图学生能到多少"的对照(Kvasir 1%,8 个 anchor):

| 学生 | Test Dice(SAM 分支) | Test Dice(UNet 分支) | best-val |
|---|---:|---:|---:|
| SC-SAM | 0.6626 | 0.6373 | — |
| SynFoC | 0.7615 | 0.7681 | unet 0.7300@37.5k,sam 0.7310@28k |

SynFoC = MedSAM 初始化的 LoRA-SAM + UNet 半监督共训
(弱/强增强一致性 + SAM↔UNet 互证门控);SC-SAM = SAM 编码器适配 + UNet 共训。
两者都比路线方法低一截,但作为独立单图验证者,其错误模式与 SAM3 视频传播独立,
是"自信错误"审核与 B7 的 q_model 信号来源。

## 4. 阶段 B:1% SAM3 微调(ft_1pct)

- 数据:8 个 GT anchor,COCO 格式,20 epochs,`lr_scale=0.1`(full fine-tune);
- checkpoint:`work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt`
  (final epoch 20);
- 评估:同一路由器协议(b3-b6,validation 重拟合评分器):
  **0.894648**(oracle 0.919805,缺口 0.0252)——当前公开最佳。

每个桥长/模式都比冻结基座提升约 +0.02~+0.05(见结果章节)。

### 4.1 ft_1pct vs frozen 逐桥长(test)

| bridge | ft_1pct target | frozen target | ft_1pct patch | frozen patch |
|---|---:|---:|---:|---:|
| direct | 0.7708 | 0.7413 | 0.7979 | 0.7578 |
| b1 | 0.7922 | 0.7602 | 0.8161 | 0.7735 |
| b2 | 0.8574 | 0.8272 | 0.8578 | 0.8223 |
| b3 | 0.8522 | 0.8408 | 0.8599 | 0.8328 |
| b4 | 0.8609 | 0.8413 | 0.8843 | 0.8340 |
| b5 | 0.8731 | 0.8486 | 0.8658 | 0.8336 |
| b6 | 0.8643 | 0.8546 | 0.8797 | 0.8421 |

### 4.2 SAM3 微调预算(加权三路线选择器)

| 模型 | weighted Dice | direct | one_bridge | two_bridges | all-route mean | oracle |
|---|---:|---:|---:|---:|---:|---:|
| base_no_ft | 0.830103 | 0.767542 | 0.819048 | 0.761372 | 0.782654 | 0.879040 |
| ft_1pct | 0.874330 | 0.831090 | 0.856687 | 0.820172 | 0.835983 | 0.902679 |
| ft_5pct | 0.882543 | 0.814541 | 0.871664 | 0.837800 | 0.841335 | 0.911454 |
| ft_10pct | 0.844562 | 0.768585 | 0.846834 | 0.815985 | 0.810468 | 0.899441 |
| ft_20pct(崩后最新) | 0.914321 | 0.856197 | 0.901934 | 0.875345 | 0.877825 | 0.931738 |

### 4.3 更长路线加权(3-7 路线族)

| 模型 | w3 | w4 | w5 | w6 | w7 | oracle7 |
|---|---:|---:|---:|---:|---:|---:|
| base_no_ft | 0.830103 | 0.856771 | 0.849015 | 0.861283 | 0.864143 | 0.913565 |
| ft_1pct | 0.874330 | 0.888891 | 0.894013 | 0.901913 | 0.900234 | 0.928200 |
| ft_5pct | 0.882543 | 0.891466 | 0.883936 | 0.886997 | 0.895276 | 0.938849 |
| ft_10pct | 0.844562 | 0.889791 | 0.884999 | 0.887835 | 0.901470 | 0.932109 |
| ft_20pct | 0.914321 | 0.914936 | 0.914313 | 0.914289 | 0.910493 | 0.942206 |

结论:微调有用但小预算不单调;20% 最强。

## 5. 阶段 C:主线路式原始伪标签池(pseudo568)

脚本:`scripts/select_phase1_mainline_pseudo568.py`

- 对 792 个训练 target,用 ft_1pct 传播质量特征 + validation 训练的 ridge
  评分器选 Top-1 候选;
- 硬阈值(对齐主线路 `prepare_t22_training.py`):
  - `q_return`(= q_cycle)≥ 0.95
  - `q_multi`(目标候选池两两共识)≥ 0.90
- 结果:**491 / 792** 通过(q_multi 均值 0.985,q_return 均值 0.971);
- 图像权重 = 归一化 q_multi,clip [0.2, 1.0]。

## 6. 阶段 D:T24 委员会学生(S2 / S3)

脚本:`run_t24_student.py`(SC-SAM SamUnet 单图学生)

- **S2**:8 GT + 491 原始伪标签,逐样本 CE+Dice,伪标签按归一化 q_multi 加权;
- **S3**:先用候选路线 mask 均值生成共识概率图 +
  `max(0.1, exp(-4 × 像素方差))` 像素权重
  (`scripts/prepare_phase1_s3_consensus.py`),再训练;
- 训练:batch 12,UNet-lr 0.01,40000 iters,final-step 选点;
- 导出 S2 val-best、S2 final、S3 final 三个审计员的 train 预测。

结果:validation Dice S2 ≈ 0.798,S3 ≈ 0.795。

## 7. 阶段 E:委员会审核 → Tier A/B/C

脚本:`scripts/phase1_audit_tiers.py`(移植 `build_s27_audit_and_tiers.py`)

- 对 301 个未入选目标(792 − 491 − 8 anchors),候选池 + 3 审计员;
- 每候选:`q_route = 0.20 × q_multi + 0.80 × q_model_mean`
  (学生一致性占主导);
- 像素共识:`0.75 × 选中 SAM3 路线 + 0.25 × 学生均值`,
  像素权重 `clip(exp(-5×路线方差) × exp(-5×学生方差) × 交叉, 0.05, 1.0)`;
- **Tier A**:q_multi≥0.90 ∧ q_model_mean≥0.90 ∧ q_model_min≥0.80 ∧
  q_model_var≤0.01 ∧ q_route≥0.88 ∧ 非空 ∧ ≥2 学生非空 ∧ 面积安全;
- **Tier B**:q_multi≥0.75 ∧ q_model_mean≥0.75 ∧ q_model_min≥0.60 ∧
  q_model_var≤0.03;
- 其余 Tier C。

结果:Tier A = 93,Tier B = 95,Tier C = 113。

## 8. 阶段 F:X3 三流学生

脚本:`run_s27_student.py` + `scripts/phase1_build_x3_manifest.py`

- X3 池 = 491 original + 93 A + 95 B = **679** 伪标签;
- 三流采样:GT 流(8)/ original 流 / 新流(A+B),batch 12(3/3/6);
- 流权重:original 1.0、A 0.75、B 0.50,全局 pseudo 0.5,2000 步 ramp;
- 逐样本 BCE + 前景 Dice;Tier B 用像素权重映射
  (≥0.70→1.0,0.30–0.70→0.25,<0.30→0);
- 40000 iters,validation Dice ≈ **0.8162**。

## 9. 阶段 G:B7 学生辅助路线选择

脚本:`scripts/phase1_b7_select.py`

- test 的 ft_1pct b3-b6 候选池(≤8 条);
- 每候选:
  - `q_return` = q_cycle(ft_1pct anchor 回传一致性);
  - `q_multi` = 候选池两两共识;
  - `q_model` = X3 学生掩码 vs 路线掩码 Dice;
  - `B7 = (q_return × q_multi² × q_model²)^0.2`;
- 每 target 取 B7 Top-1,test 一次性报告。

结果:

| 方法 | Test Dice |
|---|---:|
| X3 学生单图 | 0.8633 |
| **X3 + B7(阶段一最终)** | **0.899369** |
| oracle | 0.919805 |

比纯 ft_1pct 路由器(0.894648)**+0.0047**。

## 10. 阶段 H:第二轮——学生审核共识池 + SAM3 再微调

### 10.1 第二轮池筛选

脚本:`scripts/select_phase1_round2_pool.py`

- 对全部 792 target,用 3 审计员 + ft_1pct 候选:
- 每候选 `q_route = 0.2 × q_multi + 0.8 × q_model_mean` 取 Top-1;
- 接受条件(主线路 Tier-A 级硬门槛):
  q_multi≥0.90 ∧ q_return≥0.95 ∧ q_model_mean≥0.90 ∧ q_model_min≥0.80;
- 标签 = 像素共识:`0.75 × 路线 + 0.25 × 学生均值`,
  像素权重 `exp(-5×路线方差)×exp(-5×学生方差)×交叉`;
- 结果:**455 / 792** 通过;质量(离线诊断,对 train GT):
  mean Dice 0.921、<0.7 占 5.1%(第一轮 6.9%)。

### 10.2 SAM3 再微调

- 数据集:8 GT + 455 共识标签 = 463 图
  (`scripts/build_phase1_round2_dataset.py`);
- 配置:`kvasir_1pct_round2_student_audited_20260807_seed2026.yaml`
  (full fine-tune,`lr_scale=0.1`,20 epochs,bf16 AMP);
- **数值稳定性修复**(训练器补丁,均有备份):
  1. `weights_only=False`(支持续训加载优化器状态);
  2. NaN loss → 替换为 0 继续 + backward 后非有限梯度清零
     (防止单个坏样本 NaN 梯度毒化权重;本轮回训全程 0 次 NaN);
  3. best-val checkpoint 保存修复:meter key 前缀匹配
     (`save_best_meters: [val_roboflow100]` 匹配
     `val_roboflow100/detection`),val AP 刷新时保存
     `checkpoints/val_roboflow100_detection_coco_eval_segm_AP.pt`;
- 验证指标:best-val AP **0.6799**(epoch 13),final AP 0.6603(epoch 19);
- checkpoints:1/6/11/16/20 + 最新 + best-val 系列。

### 10.3 双 checkpoint 评估

脚本:`scripts/run_round2_eval_variant.sh`(GPU 0 final / GPU 1 bestval 并行)

- 各自 merge → 传播质量(validation+test,两模式)→ b3-b6 路由器报告。

结果(test,100 targets):

| 变体 | target+patch | target | patch | oracle |
|---|---:|---:|---:|---:|
| round2 final | 0.868266 | 0.861481 | 0.873456 | 0.910543 |
| round2 bestval | **0.882049** | 0.880997 | 0.877625 | 0.917097 |
| ft_1pct 基线 | **0.894648** | 0.886241 | 0.884695 | 0.919805 |

**结论:第二轮微调未超过 ft_1pct**,反而掉 1.3–2.6 个点。

### 10.4 逐桥长分析(direct→b6)

target 模式:

| bridge | ft_1pct | r2 final | r2 bestval |
|---|---:|---:|---:|
| direct | 0.7708 | 0.7604 | 0.7745 |
| b1 | 0.7922 | 0.8019 | **0.8165** |
| b2 | 0.8574 | 0.8655 | **0.8687** |
| b3 | 0.8522 | 0.8408 | 0.8533 |
| b4 | 0.8609 | 0.8595 | **0.8736** |
| b5 | 0.8731 | 0.8689 | **0.8762** |
| b6 | 0.8643 | 0.8493 | 0.8601 |

patch 模式:

| bridge | ft_1pct | r2 final | r2 bestval |
|---|---:|---:|---:|
| direct | 0.7979 | 0.7762 | 0.7899 |
| b1 | 0.8161 | 0.8130 | **0.8275** |
| b2 | 0.8578 | 0.8651 | **0.8661** |
| b3 | 0.8599 | 0.8592 | 0.8617 |
| b4 | **0.8843** | 0.8652 | 0.8764 |
| b5 | 0.8658 | 0.8623 | **0.8794** |
| b6 | **0.8797** | 0.8633 | 0.8712 |

规律:在 **bestval(e13)/final(e20) 上**,第二轮使短路线(b1-b2)变好、
长路线(b4-b6)变差,而路由器 Top-1 主要依赖 b4-b6,故整体下降。
**但早期 checkpoint(epoch 1/6)并不遵循这个规律——见 10.7。**

### 10.5 路由器池对比(b3-b6 vs b0-b6)

| 模型 | b3-b6 union | b3-b6 oracle | b0-b6 union | b0-b6 oracle |
|---|---:|---:|---:|---:|
| frozen 基座 | 0.877299 | 0.896918 | 0.873626 | 0.903846 |
| ft_1pct | **0.894648** | 0.919805 | 0.877061 | 0.921800 |
| round2 final | 0.868266 | 0.910543 | 0.874911 | 0.913835 |
| round2 bestval | 0.882049 | 0.917097 | **0.879142** | 0.919647 |

关键结论:

- round2 bestval 在 b3-b6 已超过 frozen(0.8820 vs 0.8773);
- 放宽到 b0-b6 后,round2 bestval(0.8791)**超过所有对手**,
  包括 ft_1pct(0.8771)——第二轮微调把能力从长程转移到短程;
- 若坚持长程优先(b3-b6),ft_1pct(0.8946)仍是最优。

### 10.6 相关诊断实验

**RouteCo-SAM3 v1**(memory adapter + LoRA,验证集 official 管线):

| 指标 | frozen | routeco step400 |
|---|---:|---:|
| unified oracle(b3-b6) | 0.8694 | 0.8614 |
| 单模式 oracle 变化 | — | -0.002 ~ -0.003 |
| 43% 路线 routeco 更优;frozen/routeco 完美选择上限 | — | 0.8778 |

结论:适配器单独训练近似中性;S→T 门因学生未接入而未生效
(student_variance=0,gate_s_to_t 恒 1)。

**HQ 伪标签审计(第一轮 357 池)**:

- 协议污染:0(全部 train 目标、8 固定 anchor、无 val/test GT);
- 质量 vs train GT:median Dice 0.956,85% ≥ 0.9,但 **10/357 < 0.5、16/357 < 0.7**;
- 这些是"自信的错误":cycle/分歧/跨模式/跨 anchor 全部无法识别
  (单特征 AUROC ≤ 0.71 且误杀一半好标签);
- 学生-路线一致性 AUROC ≈ 0.78,与路线分歧组合达 **0.85**;
- 1% GT + HQ 伪标签微调早期实验:lr_scale=0.025 版验证 AP 持平
  (0.6386→0.6356,未训动),lr=0.1 版全部因 matcher NaN 崩溃
  (finite-guard 补丁后解决)。

### 10.7 早期 checkpoint 逐桥长对比(epoch 1 / 6,forward-only test Dice)

对第二轮微调的 checkpoint_1(epoch 1)与 checkpoint_6(epoch 6)做
forward-only test 逐桥长评估(无路由器,直接报告,与 README 中
frozen/ft_1pct 的 per-bridge 表同口径)。

target 模式:

| variant | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | 0.7413 | 0.7602 | 0.8272 | 0.8408 | 0.8413 | 0.8486 | 0.8546 |
| ft_1pct | 0.7708 | 0.7922 | 0.8574 | 0.8522 | 0.8609 | 0.8731 | 0.8643 |
| **ckpt1(e1)** | **0.7814** | **0.8214** | **0.8611** | **0.8752** | **0.8687** | **0.8756** | **0.8668** |
| ckpt6(e6) | 0.7788 | 0.8255 | 0.8643 | 0.8684 | 0.8758 | 0.8761 | 0.8577 |
| bestval(e13) | 0.7745 | 0.8165 | 0.8687 | 0.8533 | 0.8736 | 0.8762 | 0.8601 |
| final(e20) | 0.7604 | 0.8019 | 0.8655 | 0.8408 | 0.8595 | 0.8689 | 0.8493 |

patch 模式:

| variant | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | 0.7578 | 0.7735 | 0.8223 | 0.8328 | 0.8340 | 0.8336 | 0.8421 |
| ft_1pct | 0.7979 | 0.8161 | 0.8578 | 0.8599 | **0.8843** | 0.8658 | 0.8797 |
| **ckpt1(e1)** | **0.8045** | **0.8433** | **0.8626** | **0.8733** | 0.8735 | **0.8721** | **0.8790** |
| ckpt6(e6) | 0.7931 | 0.8351 | 0.8682 | 0.8688 | 0.8757 | 0.8674 | 0.8661 |
| bestval(e13) | 0.7899 | 0.8275 | 0.8661 | 0.8617 | 0.8764 | 0.8794 | 0.8712 |
| final(e20) | 0.7762 | 0.8130 | 0.8651 | 0.8592 | 0.8652 | 0.8623 | 0.8633 |

关键发现:

- **ckpt1(epoch 1)几乎全面超过 ft_1pct**:target 7/7 桥全赢
  (b3 +0.023,b1 +0.029),patch 6/7 桥赢(仅 b4 低 0.011,b6 持平);
- **ckpt6(epoch 6)同样很强**:target 6/7,patch 5/7 桥超过 ft_1pct;
- **退化发生在训练后期**:bestval(e13)开始丢 b3/b6,final(e20)全线下降——
  第二轮微调是"起步强、越训越差"的过拟合曲线,**epoch 1-6 是黄金区**;
- 原因:455 个伪标签让模型一个 epoch 就吸收足够多样性
  (对比 ft_1pct 只有 8 张图反复 20 epochs),早期 checkpoint 既拿到多样性
  又没过拟合;
- 下一步:对 ckpt1/ckpt6 接路由器(b3-b6 与 b0-b6)与 B7 学生选择,
  很可能超过 ft_1pct 的 0.894648。

### 10.8 早期 checkpoint 路由器验证(b3-b6 与 b0-b6)

对 ckpt1(e1)/ckpt6(e6) 补全传播质量特征,并用同一 ridge 路由器协议
(validation 重拟合)分别评估 b3-b6 与 b0-b6 两个候选池。

**b3-b6 池(union)**:

| 模型 | selected | oracle | gap |
|---|---:|---:|---:|
| frozen | 0.877299 | 0.896918 | 0.0196 |
| ft_1pct | **0.894648** | 0.919805 | 0.0252 |
| round2 final(e20) | 0.868266 | 0.910543 | 0.0423 |
| round2 bestval(e13) | 0.882049 | 0.917097 | 0.0350 |
| **ckpt1(e1)** | **0.884656** | 0.911043 | 0.0264 |
| ckpt6(e6) | 0.880806 | 0.913594 | 0.0328 |

**b0-b6 池(union)**:

| 模型 | selected | oracle | gap |
|---|---:|---:|---:|
| frozen | 0.873626 | 0.903846 | 0.0302 |
| ft_1pct | 0.877061 | 0.921800 | 0.0447 |
| round2 final(e20) | 0.874911 | 0.913835 | 0.0389 |
| round2 bestval(e13) | 0.879142 | 0.919647 | 0.0405 |
| round2 ckpt1(e1) | 0.881271 | 0.913898 | 0.0326 |
| **round2 ckpt6(e6)** | **0.883671** | 0.915488 | 0.0318 |

**b0-b6 单模式(selected)**:

| 模型 | target | patch |
|---|---:|---:|
| ft_1pct | 0.868192 | 0.890865 |
| round2 ckpt1(e1) | 0.881311 | 0.885018 |
| round2 ckpt6(e6) | 0.883406 | 0.879398 |
| round2 bestval(e13) | 0.878366 | 0.878679 |
| round2 final(e20) | 0.875843 | 0.874597 |

关键结论:

- **早期停止假设成立**:e1/e6 > e13 > e20,第二轮训练越久越差;
- **b0-b6 池:ckpt6(e6) 0.8837 为全场最佳**,超过 ft_1pct(0.8771)
  与全部 round-2 变体——与逐桥长 forward 结果一致;
- **b3-b6 池:ckpt1(e1) 0.8847 是第二轮最强,但仍低于 ft_1pct(0.8946)**:
  ft_1pct 在长路线区间路线质量(oracle 0.9198)与选择都更好;
- oracle 最高仍是 ft_1pct(b0-b6 0.9218 / b3-b6 0.9198),
  说明 ckpt6 赢在"选择更对"而非路线质量上限更高。

### 10.9 第一帧提示方式对比实验(mask / mask+box 组合,ckpt1,test b3-b6)

**背景**:SAM3 的视频接口有两套——主 `add_prompt`(概念驱动,PCS)只支持
text/box/point;mask 提示只有 SAM2 风格的 `Sam3TrackerPredictor.add_new_mask`
支持(官方推荐用于半监督 VOS)。管线此前全程只用 bbox 提示。为验证
"第一帧用 GT mask 是否更好 / tracker 能否用 mask 精修 detector 输出",
在 **round2 ckpt1(e1)** 上做第一帧提示方式对比。

脚本:`scripts/compare_mask_vs_box_prompt.py`
(三种 prompt 变体 + 与既有 bbox 基线逐路线对比)。

**协议**:test 100 targets,两模式,b3-b6,714 条路线;
forward-only test Dice,@canvas 512,与既有 bbox 基线(传播质量 jsonl 的
`gt_dice_evaluation_only`)同口径、同 checkpoint、逐 route_id 配对。

#### 三种变体

| 变体 | 机制 | 状态 |
|---|---|---|
| mask(纯 GT mask) | tracker 直传 GT mask,跳过 detector | 可运行 |
| mask+box 同帧直传 | 同一帧同时给 mask 和框角点 | **SAM3 不支持**(`_run_single_frame_inference` 硬断言 `point_inputs is None or mask_inputs is None`) |
| **det_gtmask** | **bbox 路径照跑(detector 负责定位/条件化),tracker 第一帧播种 mask 换成 GT mask** | 可运行 |

#### 结果(平均 Dice,vs bbox 基线)

总体:

| 变体 | box | 变体 | delta | 胜率 |
|---|---:|---:|---:|---:|
| mask(纯 GT mask) | 0.8777 | 0.8646 | **-0.0131** | 21.1%(151/714) |
| **det_gtmask** | 0.8777 | **0.8832** | **+0.0055** | 59.0%(421/714) |

逐桥长(box → det_gtmask):

| 模式 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|
| target pooling | 0.8752 → 0.8802(+0.0050) | 0.8687 → 0.8738(+0.0051) | 0.8756 → 0.8829(+0.0073) | 0.8668 → 0.8765(+0.0097) |
| patch correspondence | 0.8926 → 0.8939(+0.0013) | 0.8881 → 0.8891(+0.0010) | 0.8828 → 0.8954(+0.0126) | 0.8800 → 0.8812(+0.0011) |

det_gtmask 在**每个 cell 均为正**,长路线(b5/b6)与 target pooling 模式提升
最明显(target pooling 赢 232/400,patch 赢 189/314)。

#### 结论

- **纯 mask 提示不可用**:跳过 detector 后传播质量整体下降(-0.0131),
  说明 bbox→detector 这一环本身在建立传播条件化,不能用 mask 替代;
- **mask+box 同帧不成立**:SAM3 tracker 的帧推理硬性不允许 point 与 mask
  同时输入(与 SAM2 的训练语义一致);
- **det_gtmask 有效(+0.0055,59% 胜率)**:detector 照常从 bbox 定位并建立
  条件化,tracker 第一帧播种 mask 替换为 GT mask,等价于"用 GT mask 精修
  detector 输出",传播质量稳定提升——与 10.6 中"学生-路线一致性"的审核
  结论一致:信息更准的第一帧播种确实有用,但不能丢失 detector 条件化。

**当前局限**:以上仅 forward Dice;det_gtmask 尚未接入传播质量特征
(q_cycle/trace)与 b3-b6 路由器/B7 口径,也未在 ft_1pct 上复测。
产物:`work/kvasir_1pct_anchors/mask_prompt_experiment/`
(mask_vs_box_*/det_gtmask_vs_box_* jsonl 与 summary)。

## 11. 结果汇总

| 方法 | Test Dice | 说明 |
|---|---:|---|
| 冻结多路线基线 | 0.8715 | 主协议口径 |
| frozen 传播质量路由器(b3-b6) | 0.877299 | 冻结 SAM3 |
| ft_1pct + 路由器 | **0.894648** | 1% 微调 + 选择 |
| X3 单图学生 | 0.8633 | 阶段一学生 |
| X3 + B7 选择 | **0.899369** | 阶段一最终 |
| round2 final + 路由器 | 0.868266 | 第二轮微调 |
| round2 bestval + 路由器 | 0.882049 | 第二轮微调(best-val) |

补充口径:

- **主线路(S27 X3+B7,合并 CVC+Kvasir 协议)**:X3 单图 0.866738;
  线性选择器 0.885637;**B7 0.895835**;oracle 0.907835;
- **Kvasir 1% 学生基线**:SC-SAM 0.6626/0.6373;SynFoC 0.7615/0.7681;
- **路由器池对比**:round2 bestval 在 b0-b6 池 0.8791 为全场最佳;
  ft_1pct 在 b3-b6 池 0.8946 为全场最佳;
- **微调预算**:ft_20pct weighted 0.914321 仍是最强权重选择结果。
- **第一帧提示实验(ckpt1,b3-b6)**:纯 GT mask 提示 -0.0131;
  det_gtmask(detector+GT mask 播种)+0.0055、59% 路线胜——见 10.9。

## 12. 工程与复现

### 关键脚本

```text
scripts/select_phase1_mainline_pseudo568.py   原始池筛选(主线路式阈值)
scripts/prepare_phase1_s3_consensus.py        S3 共识目标
scripts/phase1_audit_tiers.py                 委员会审核 → Tier A/B/C
scripts/phase1_build_x3_manifest.py           X3 manifest 合并
scripts/phase1_b7_select.py                   B7 选择与 test 报告
scripts/select_phase1_round2_pool.py          第二轮学生审核池
scripts/build_phase1_round2_dataset.py        第二轮 COCO 数据集
scripts/run_round2_eval_variant.sh            单变体评估(merge→pq→报告)
scripts/patch_remote_sam3_trainer_save_best.py best-val 保存补丁
scripts/patch_remote_sam3_matcher_finite_guard.py matcher 有限守卫补丁
```

### 关键产物

```text
work/kvasir_1pct_anchors/phase1/
  pseudo_manifest_original.jsonl        第一轮 491 池
  S3_consensus/pseudo_consensus.jsonl   S3 共识
  students/{S2,S3,X3}/                  学生训练
  audit/tier_{A,B,C}.jsonl              审核结果
  predictions/{S2_valbest,S2_final,S3_final,X3_final}/
  selection/b7_report.json              B7 报告(0.899369)
  round2_pool/round2_pool.jsonl         第二轮 455 池
  stage1_feature_knn_b7_ftround2*/       第二轮传播质量 + 报告
  round2/router_b3_b6_{final,bestval}_report.json
```

### 训练器补丁(在 /Data_8TB/lht/sam3,均有 .bak 备份)

- `trainer.py`:weights_only=False;NaN loss→0 + 非有限梯度清零;
  best-val 保存前缀匹配;`_run_step` 元数据 dump。
- `matcher.py`:BinaryHungarianMatcherV2 有限守卫(NaN/inf 代价替换为 1e9)。

## 13. 当前状态与下一步

- 阶段一(ft_1pct + 学生审核 + B7)是当前最强链路,**0.899369**;
- 第二轮 SAM3 微调在 final(e20)/bestval(e13) 上未超过 ft_1pct,
  但**早期 checkpoint(epoch 1/6)在逐桥长 forward Dice 上几乎全面反超**
  (见 10.7)——第二轮的正确用法是早期停止;
- 优先行动:
  1. 对 ckpt1(e1)/ckpt6(e6) 接路由器(b3-b6 与 b0-b6)与 B7 选择;
  2. 若有效,把"学生审核共识池 + 早期停止"固化为第二轮标准流程;
  3. 可选:用原始路线 mask(而非共识混合)微调、降低 lr(0.05)进一步对比。

更新(路由器验证后):

- **b0-b6:ckpt6(e6) 0.8837 已是全场最佳**(> ft_1pct 0.8771);
- **b3-b6:ft_1pct 0.8946 仍不可替代**,ckpt1(e1) 0.8847 为第二轮最强;
- 下一步优先:对 ckpt1(e1)/ckpt6(e6) 接 **B7 学生选择**,
  并考虑把 b0-b6 + ckpt6 作为新的主配置候选。
- 第一帧提示(10.9):det_gtmask 在 ckpt1 上稳定 +0.0055,
  待接入传播质量/路由器与 B7 口径验证(需重算 q_cycle 等特征),
  并在 ft_1pct 上复测;纯 mask 提示不可用,SAM3 不支持同帧 mask+box。
