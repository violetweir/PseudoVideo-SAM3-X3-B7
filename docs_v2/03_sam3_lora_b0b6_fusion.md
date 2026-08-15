# SAM3 LoRA：b0-b6 路线掩码融合（第一版测试）

> 版本: 2026-08-13
> 状态: 已完成 `lora_p491_e20` 的首次 fusion 测试

## 1. 目标

把未微调 SAM3 的 b0-b6 soft fusion 直接迁移到 LoRA checkpoint，检查融合是否
仍能比单个桥长候选更强。

## 2. 固定配置

| 项 | 值 |
|---|---|
| 路线族 | `sam3enc_anchor_conditioned_target_pooling` |
| 特征分辨率 | SAM3 encoder @1008 |
| KNN 检索特征 | `patch_mean` |
| checkpoint | `lora_p491_e20_merged_video.pt` |
| 评估分辨率 | 256×256 |
| 候选掩码 | b0-b6，每个 target 7 个 forward mask |

## 3. 方法

与 base fusion 相同：validation 拟合 ridge scorer，输入 4 个 GT-free 特征，
再选择 softmax 温度 `T` 和阈值 `tau`，test 只报告一次。

```text
特征:
  bridge_count
  path_bottleneck_similarity
  path_mean_similarity
  forward_sam_score
```

## 4. 结果

| 口径 | fixed b6 | mean b0-b6 @0.5 | fusion | best single bridge | oracle |
|---|---:|---:|---:|---:|---:|
| validation | 0.856773 | 0.857475 | **0.874009** | — | 0.888770 |
| test | 0.893381 | 0.893440 | **0.897877** | **0.910832 (b1)** | 0.930744 |

最终选择参数：

```text
T   = 0.1
tau = 0.3
```

## 5. 结论

- LoRA 下 fusion 仍高于固定 b6（0.8979 vs 0.8934），也略高于简单平均。
- 但当前 4 特征 fusion **没有超过该模式的最强单桥长 b1 = 0.9108**。
- 这说明 LoRA 的路线之间互补性没有 base 那么强，或者需要更高质量的无 GT
  权重信号（例如 q_cycle、传播轨迹特征）才能发挥融合收益。

## 6. 产物

```text
work/kvasir_1pct_anchors/sam3enc_base_fusion_v1_lora/
  summary.json
  test_per_target.json

LoRA validation forward:
  work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/
    sam3enc_anchor_conditioned_target_pooling/
      eval_lora_p491_e20_validation/
```

## 7. 下一步

- 为 LoRA 路线补 propagation quality 特征；
- 在 validation 上做更严格的 OOF，避免 T/tau 选择过拟合；
- 若仍不能超过 b1=0.9108，则考虑只对低共识/高风险样本启用 fusion，而不是
  对所有样本无条件融合。
