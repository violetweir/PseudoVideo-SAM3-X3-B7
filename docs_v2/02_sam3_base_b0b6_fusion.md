# SAM3 未微调：b0-b6 路线掩码融合（第一版）

> 版本: 2026-08-13
> 状态: 已完成第一版 validation-calibrated soft fusion

## 1. 目标

不再从 b0-b6 里选一个桥长，而是把同一个 SAM3-enc KNN 路线族的 7 个
forward mask 综合成一个更强输出。

本轮只使用未微调 SAM3 base checkpoint，SAM3 评估固定在 256×256。

## 2. 固定配置

| 项 | 值 |
|---|---|
| 路线族 | `sam3enc_anchor_conditioned_target_pooling` |
| 特征分辨率 | SAM3 encoder @1008 |
| KNN 检索特征 | `patch_mean` |
| checkpoint | `sam3.pt` base |
| 评估分辨率 | 256×256 |
| 候选掩码 | b0-b6，每个 target 7 个 forward mask |

协议数据与上一阶段相同：Kvasir-SEG train=800 / val=100 / test=100，8 个
固定 GT anchor。

## 3. 融合方法

对每个 target：

1. 取 b0-b6 的 7 个 SAM3 forward binary mask；
2. 对每条路线提取 4 个 GT-free 特征：
   - `bridge_count`
   - `path_bottleneck_similarity`
   - `path_mean_similarity`
   - `forward_sam_score`
3. 在 validation 上拟合 ridge scorer，目标是逐路线的验证 Dice；
4. 用 softmax 温度 `T` 把 route score 转成权重：

```text
w_i = softmax((score_i - max(score)) / T)
```

5. 计算软概率图：

```text
P = sum_i w_i * M_i
```

6. 在 validation 上选择 `T` 和最终阈值 `tau`；
7. 在 test 上一次报告 `P >= tau` 的 Dice。

实现脚本：

```text
scripts/run_sam3enc_base_fusion_v1.py
```

## 4. 本轮结果

| 口径 | best b6 | mean b0-b6 @0.5 | fusion | oracle |
|---|---:|---:|---:|---:|
| validation | 0.825979 | 0.830309 | **0.851524** | 0.897386 |
| test | 0.877104 | 0.890437 | **0.897882** | 0.927779 |

最终选择参数：

```text
T   = 1.0
tau = 0.2
```

结论：在未微调 SAM3 下，简单 soft fusion 已经比固定 b6 和普通平均更强；
test 从 b6 的 0.8771 提升到 0.8979，说明 b0-b6 的 mask 之间存在可融合的
互补信息。

## 5. 产物

```text
work/kvasir_1pct_anchors/sam3enc_base_fusion_v1/
  summary.json
  test_per_target.json

validation forward:
  work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/
    sam3enc_anchor_conditioned_target_pooling/
      eval_base_no_ft_b7_forward_validation/

test forward:
  work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/
    sam3enc_anchor_conditioned_target_pooling/
      eval_base_no_ft_b7_forward/
```

## 6. 下一步

- 把 `q_cycle`、mask 面积轨迹、相邻帧 Dice、SAM score 轨迹等传播质量特征
  加入 fusion scorer，预期比当前 4 特征更稳；
- 在同一路线族上做 validation-only nested/OOF，避免阈值选择的少量过拟合；
- 若稳定，再把它固定为“未微调 SAM3 + b0-b6 fusion”主线第一结果。
