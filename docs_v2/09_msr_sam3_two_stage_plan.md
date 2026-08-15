# MSR-SAM3：Mask-aware Semantic Re-routing for Pseudo-Video SAM3

> 版本: 2026-08-15
> 状态: 方法设计 + 第一版实验计划

## 1. 核心思想

把当前方向从“增加 mask / 文字特征”升级为两阶段语义重路由：

> Round-1 回答“这张图视觉上像谁？”
> -> SAM3 得到粗伪 mask M¹
> -> 从 M¹ 提取 lesion-centric 信息
> -> Round-2 重排邻居并重构伪视频传播链
> -> SAM3 重新预测得到 M²

关键约束：**M¹ 不直接作为第二轮 target 的 mask prompt**，只用于暴露目标区域、
提取区域特征、几何形态和语义描述，然后重新找邻居、重新传播。

## 2. 两阶段流程

```text
Target Image
  -> Round-1 Appearance KNN
  -> Pseudo-video Route #1
  -> SAM3
  -> Initial Mask M¹
       |
       +-- Mask Geometry z_shape
       +-- Masked ROI Feature z_roi
       +-- Mask2Text (Qwen3.5) z_text
       v
  Semantic Re-ranking / Top-L KNN
  -> Pseudo-video Route #2
  -> SAM3
  -> Refined Mask M²
```

## 3. 特征定义

| 特征 | 符号 | 来源 | 回答的问题 |
|---|---|---|---|
| 原图视觉 | z_visual | SAM3-enc 原图 patch_mean | 整张图像视觉上像谁 |
| Masked ROI | z_roi | mask bbox 外扩 ROI + SAM3-enc | 目标区域看起来像什么 |
| Mask Geometry | z_shape | 确定性 mask 几何特征 | 目标形态像谁 |
| Mask2Text | z_text | Qwen3.5 结构化描述 | 目标语义形态像谁 |

所有相似度独立计算，不做特征 concat：

```text
S(q,i) = α·s_visual + β·s_roi + γ·s_shape + δ·s_text
```

第一版用固定权重；后续用 validation 冻结权重。

## 4. B0-B5 实验矩阵

| 实验 | Round-1 | ROI | Shape | Text | Round-2 |
|---|---:|---:|---:|---:|---:|
| B0 | ✓ | | | | × |
| B1 | ✓ | ✓ | | | ✓ |
| B2 | ✓ | | ✓ | | ✓ |
| B3 | ✓ | | | ✓ | ✓ |
| B4 | ✓ | ✓ | ✓ | | ✓ |
| B5 | ✓ | ✓ | ✓ | ✓ | ✓ |

B0 是当前最好 Round-1 系统；B1-B5 用于分离各信号贡献。

## 5. Round-1 冻结配置

```text
SAM3-enc @1008
  + anchor_conditioned_target_pooling
  + knn_feature = patch_mean
  + b0-b6
  + SAM3 base @256
```

M¹ 由 Round-1 b0-b6 fusion 生成：

```text
work/kvasir_1pct_anchors/train_pseudo_masks_round1/
work/kvasir_1pct_anchors/validation_pseudo_masks_round1/
work/kvasir_1pct_anchors/test_pseudo_masks_round1/
```

## 6. 已有可复用资产

- SAM3-enc 原图描述子：
  `stage1_feature_knn_sam3enc_s1008/features/sam3_base_s1008_features.npz`
- SAM3-enc mask 前景描述子：
  `stage1_feature_knn_sam3enc_s1008/features/sam3enc_mask_descriptors_s1008.npz`
- Qwen3.5 结构化 mask 文本描述：
  `train/validation/test_pseudo_masks_round1/qwen35_mask_descriptions.jsonl`

## 7. 第一版实现路径

1. 用 z_visual 在 train 池取 Top-L 候选（L=20）；
2. 在候选集内分别计算 s_visual / s_roi / s_shape / s_text；
3. score-level 融合后重排；
4. 保留 Round-1 anchor，替换 bridge 为 rerank 后的 Top-K；
5. SAM3 base @256 二次传播得到 M²；
6. 对比 B0-B5。

## 8. 成功判据

- B5 > B4 > B0，且 ROI / Shape / Text 各自有非零增量；
- 更重要的是：M² 与 M¹ 的循环一致性 Dice 高，说明两条信息路径一致；
- 若语言无增益（B4 ≈ B5），则放弃语言叙事，保留 mask+geometry。
