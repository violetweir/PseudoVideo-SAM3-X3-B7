# C0-256 Round-2C：双向病灶对应关系 Anchor Re-ranking Validation

> 状态：validation 实验已完成。  
> 协议：SAM3-base trunk@256、SAM3-e33 video propagation canvas=256、固定 X3-best。  
> 范围：仅 validation，不运行 test、train propagation 或任何训练。

## 1. 冻结协议

| 项目 | 配置 |
|---|---|
| 数据 | validation 100，固定 8 个 train-side GT anchors |
| 特征 encoder | 未微调 SAM3-base vision trunk |
| 特征输入 | 256 × 256 |
| patch grid | 18 × 18 |
| KNN topology | 原 SAM3-base@256，未整体重建 |
| propagation teacher | 已冻结 SAM3-e33 merged video checkpoint |
| propagation canvas | 256 × 256 |
| route | b0，即 [anchor, target] 两帧 video propagation |
| target mask | round1 pseudo-mask / 冻结 X3-best mask |
| GT 用途 | anchor GT 可用于 support；validation target GT 只用于事后评估 |

注意：b0 是两帧 video propagation，不是带文本提示的单图 Direct segmentation。

## 2. 病灶对应分数

```text
forward = AUC(anchor lesion → target heatmap, target pseudo-mask)
reverse = AUC(target pseudo-lesion → anchor heatmap, anchor GT mask)
bidirectional = sqrt(forward * reverse)
```

同时比较 foreground prototype 与 token-to-token top-k；全部特征均来自 base@256。

## 3. 执行规模

- validation targets：100。
- GT anchors：8。
- 全部 b0 anchor-target routes：800。
- 复用旧 e33 b0 routes：157。
- 新运行 e33 b0 routes：643。
- 全 8 anchor 事后 b0 oracle：0.886557。

## 4. Anchor top-1 / top-3 validation 对照

| Variant | Mask | Top-1 b0 Dice | Top-3 oracle | Mean target Spearman | Δ vs lesion mean | Top-3 b0 B7 Dice |
|---|---|---:|---:|---:|---:|---:|
| V0_target_pooling | x3_best | 0.716743 | 0.871520 | -0.1202 | — | 0.863736 |
| V0_patch_correspondence | x3_best | 0.757399 | 0.867900 | -0.0266 | — | 0.857033 |
| V1_lesion_mean__round1 | round1 | 0.788142 | 0.871501 | -0.1986 | +0.000000 | 0.865314 |
| V2_forward_prototype__round1 | round1 | 0.841111 | 0.874527 | -0.2040 | +0.052969 | 0.866638 |
| V3_reverse_prototype__round1 | round1 | 0.784279 | 0.868514 | 0.0405 | -0.003863 | 0.859779 |
| V4_bidirectional_prototype__round1 | round1 | 0.814134 | 0.873087 | -0.1290 | +0.025991 | 0.866010 |
| V2_forward_token_topk__round1 | round1 | 0.820911 | 0.878916 | -0.1475 | +0.032768 | 0.867872 |
| V3_reverse_token_topk__round1 | round1 | 0.793409 | 0.860919 | -0.0928 | +0.005266 | 0.855089 |
| V4_bidirectional_token_topk__round1 | round1 | 0.831955 | 0.871384 | -0.1675 | +0.043812 | 0.863841 |
| V1_lesion_mean__x3_best | x3_best | 0.769382 | 0.869638 | -0.1912 | +0.000000 | 0.863889 |
| V2_forward_prototype__x3_best | x3_best | 0.836765 | 0.876452 | -0.2031 | +0.067383 | 0.866832 |
| V3_reverse_prototype__x3_best | x3_best | 0.784161 | 0.871870 | 0.0437 | +0.014779 | 0.860608 |
| V4_bidirectional_prototype__x3_best | x3_best | 0.819685 | 0.871069 | -0.1142 | +0.050303 | 0.864368 |
| V2_forward_token_topk__x3_best | x3_best | 0.827418 | 0.878614 | -0.1500 | +0.058036 | 0.865623 |
| V3_reverse_token_topk__x3_best | x3_best | 0.784603 | 0.881457 | -0.0755 | +0.015221 | 0.866710 |
| V4_bidirectional_token_topk__x3_best | x3_best | 0.839407 | 0.875469 | -0.1605 | +0.070025 | 0.864243 |

## 5. B7 解释边界

各 variant 的 Top-3 b0 B7 使用完全相同的旧 B7 公式、固定 X3-best、
每 target 恰好 3 个 b0 anchor candidates，因此不同 variant 之间可公平比较。

```text
B7 = (q_return * q_multi^2 * q_model^2)^0.2
```

历史 Round-2A 双 mode、b0-b6 完整 validation B7 = 0.883854。
该完整候选池与本轮 3 个 b0 候选的池规模不同，只作主线参考，不能直接归因。

## 6. 输出

- Experiment root：`work/rerun_c0_256_round2c_lesion_anchor_validation`。
- `anchor_scores_validation.jsonl`：每个 anchor-target 的双向对应分数。
- `validation_anchor_summary.json`：完整 ranking、paired delta 和 oracle 审计。
- `quality_root/round2c_all_anchors_b0/`：800 条 e33 b0 propagation。
- `b7/*.summary.json`：固定 3-candidate 的 B7 validation 对照。

本轮没有读取 test target、没有传播 train target、没有生成新的训练 pool、没有训练模型。
