# SAM3 编码器 KNN 路线构造 + SAM3 256×256 forward Dice 评估

> 版本: 2026-08-13
> 状态: SAM3 base（未微调）已在 256×256 下重新执行完成，结果与已有产物一致

## 1. 目标

这是 Kvasir-SEG 1% 新主线的前置实验：**用 SAM3 自身的 encoder 特征做 KNN
路线构造，但 SAM3 传播评估统一固定在 256×256**，避免不同评估分辨率带来的
不可比。

本次只报告 forward-only 逐桥长 Dice，不含传播质量路由器、学生审核、B7 或
额外微调。

## 2. 协议

| 项 | 固定值 |
|---|---|
| 数据集 | Kvasir-SEG 快照，train=800 / validation=100 / test=100 |
| 标注 | 8 个固定 GT anchor，其余 train mask 不参与搜索与选择 |
| 路线构造特征 | SAM3 backbone encoder，32 层 ViT，patch=14，embed=1024，无 CLS |
| 特征输入分辨率 | 1008（原生）为主，256（协议对齐）作对照 |
| SAM3 评估分辨率 | **256×256** |
| 桥长 | b0–b6，b0=direct |
| beam width | 32 |
| checkpoint | `sam3.pt` base + `lora_p491_e20_merged_video.pt` |
| 评估口径 | test 100 targets，每变体 700 条路线，逐桥长 mean Dice |

关键原则：**路线可以用 1008 原生特征构造，但最终 Dice 必须在 SAM3 @256 下
测试**，这样不同路线构造方案才可公平比较。

## 3. 数据与 checkpoint

```text
协议:
  work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl
  work/kvasir_1pct_anchors/protocol/support_manifest.jsonl

checkpoint:
  base:  /Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
  LoRA:  work/kvasir_1pct_anchors/video_checkpoints/lora_p491_e20_merged_video.pt

输出根目录:
  work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/
  work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/
```

## 4. 复现命令

有 GPU 的机器上执行：

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/Data_8TB/lht/sam3:/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/scripts
bash scripts/run_stage1_knn_sam3enc.sh
```

该脚本内部完成：

1. 分别用 1008 和 256 特征生成 7 个 SAM3-enc KNN 变体；
2. 每个变体分别用 base 与 `lora_p491_e20` 在 SAM3 @256 下评估；
3. 输出 `comparison_summary_*.md/json`。

对应单步命令示例：

```bash
PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT_LORA=work/kvasir_1pct_anchors/video_checkpoints/lora_p491_e20_merged_video.pt
ROOT1008=work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008

$PY scripts/stage1_feature_knn_routes.py \
  --mode sam3enc_anchor_conditioned_target_pooling \
  --feature-source sam3_base \
  --feature-size 1008 \
  --knn-feature cond \
  --max-bridge 6 \
  --beam-width 32 \
  --split test \
  --output-root "$ROOT1008"

$PY scripts/stage1_eval_routes_forward_only.py \
  --checkpoint "$CKPT_LORA" \
  --mode sam3enc_anchor_conditioned_target_pooling__knn_cond \
  --root "$ROOT1008" \
  --canvas 256 \
  --eval-name eval_lora_p491_e20 \
  --resume
```

## 5. 结果：SAM3-enc @1008 构造路线，SAM3 @256 评估

### 5.1 base `sam3.pt`

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_patch_correspondence | patch_mean | 0.8202 | 0.8321 | 0.8302 | 0.8035 | 0.8173 | 0.8247 | 0.8388 |
| sam3enc_patch_correspondence | cond | 0.8202 | 0.8396 | 0.8018 | 0.7944 | 0.8028 | 0.8120 | 0.8084 |
| sam3enc_patch_correspondence | pooled | 0.8202 | 0.8499 | 0.8311 | 0.8033 | 0.8451 | 0.8154 | 0.8408 |
| sam3enc_target_pooling | patch_mean | 0.8205 | 0.8762 | 0.8729 | 0.8701 | 0.8655 | **0.8874** | 0.8771 |
| sam3enc_target_pooling | cond | 0.8205 | **0.8811** | **0.8823** | 0.8617 | 0.8615 | 0.8822 | 0.8639 |
| sam3enc_target_pooling | pooled | 0.8205 | 0.8735 | 0.8758 | 0.8714 | 0.8563 | 0.8643 | 0.8769 |
| sam3enc_patch_average | patch_mean | 0.7698 | 0.8020 | 0.8452 | 0.8672 | 0.8500 | 0.8546 | 0.8550 |

### 5.2 LoRA `lora_p491_e20`

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_patch_correspondence | patch_mean | 0.8992 | 0.9129 | 0.9098 | 0.9029 | 0.9047 | **0.9149** | **0.9150** |
| sam3enc_patch_correspondence | cond | 0.8992 | 0.8808 | 0.8794 | 0.8830 | 0.8704 | 0.8735 | 0.8949 |
| sam3enc_patch_correspondence | pooled | 0.8992 | 0.9000 | **0.9135** | 0.8994 | 0.9061 | 0.9066 | 0.9021 |
| sam3enc_target_pooling | patch_mean | **0.8995** | 0.9108 | 0.8991 | 0.8927 | 0.8939 | 0.9082 | 0.8934 |
| sam3enc_target_pooling | cond | **0.8995** | **0.9154** | 0.9032 | **0.9131** | **0.9154** | 0.9136 | 0.9131 |
| sam3enc_target_pooling | pooled | **0.8995** | 0.9005 | 0.8921 | 0.8989 | 0.8970 | 0.9002 | 0.9026 |
| sam3enc_patch_average | patch_mean | 0.8261 | 0.8648 | 0.8728 | 0.8995 | 0.8981 | 0.8958 | 0.8993 |

最强固定桥：`sam3enc_target_pooling__knn_cond` 的 b1/b4 均达到 **0.9154**。

## 6. 对照：SAM3-enc @256 构造路线，SAM3 @256 评估

### 6.1 base `sam3.pt`

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_patch_correspondence | patch_mean | 0.7066 | 0.7549 | 0.7885 | 0.8594 | 0.8576 | 0.8553 | 0.8536 |
| sam3enc_patch_correspondence | cond | 0.7066 | 0.7355 | 0.8357 | 0.8362 | 0.8499 | 0.8484 | 0.8318 |
| sam3enc_patch_correspondence | pooled | 0.7066 | 0.7458 | 0.8030 | 0.8652 | 0.8618 | 0.8505 | 0.8603 |
| sam3enc_target_pooling | patch_mean | 0.7981 | 0.8466 | 0.8700 | 0.8623 | 0.8739 | 0.8696 | 0.8740 |
| sam3enc_target_pooling | cond | 0.7981 | 0.8534 | 0.8647 | **0.8841** | 0.8461 | 0.8698 | 0.8462 |
| sam3enc_target_pooling | pooled | 0.7981 | 0.8391 | 0.8466 | 0.8493 | 0.8765 | 0.8657 | 0.8529 |
| sam3enc_patch_average | patch_mean | 0.7502 | 0.7889 | 0.8187 | 0.8204 | 0.8094 | 0.7795 | 0.7680 |

### 6.2 LoRA `lora_p491_e20`

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_patch_correspondence | patch_mean | 0.7647 | 0.8680 | 0.8722 | 0.8886 | 0.8997 | 0.9036 | 0.8994 |
| sam3enc_patch_correspondence | cond | 0.7647 | 0.8679 | 0.8739 | 0.8983 | **0.9082** | 0.8809 | **0.9012** |
| sam3enc_patch_correspondence | pooled | 0.7647 | 0.8683 | 0.8719 | 0.8849 | 0.8913 | 0.8985 | 0.8939 |
| sam3enc_target_pooling | patch_mean | 0.8482 | 0.8528 | 0.8659 | 0.8730 | 0.8927 | 0.8879 | 0.8960 |
| sam3enc_target_pooling | cond | 0.8482 | 0.8640 | 0.8277 | 0.8779 | 0.8871 | 0.8946 | 0.8988 |
| sam3enc_target_pooling | pooled | 0.8482 | 0.8597 | 0.8616 | 0.8620 | 0.8944 | 0.8901 | 0.8985 |
| sam3enc_patch_average | patch_mean | 0.8222 | 0.8505 | 0.8861 | 0.8689 | 0.8858 | 0.8930 | 0.8876 |

## 7. 结论

- **SAM3 编码器特征构造路线明显优于 DINOv3**：
  - base 下 0.8874 vs DINOv3 最高 0.8573；
  - LoRA 下 0.9154 vs DINOv3 最高 0.8978。
- **1008 原生特征优于 256 协议对齐**：
  - LoRA 0.9154 vs 0.9082；
  - base 0.8874 vs 0.8841。
- direct 也显著提升，说明 SAM3 编码器特征让 anchor 选择本身更准。
- SAM3 encoder 没有 CLS token，因此最优 KNN 检索特征从 DINOv3 的 `cls`
  变成 `patch_mean` / `cond`。

## 8. 产物索引

```text
work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/
  features/sam3_base_s1008_features.npz
  comparison_summary_sam3enc_s1008_base.{md,json}
  comparison_summary_sam3enc_s1008_lora.{md,json}
  sam3enc_*/test_pool0_stage1/routes.jsonl
  sam3enc_*/eval_base_no_ft_b7_forward/route_family_summary.json
  sam3enc_*/eval_lora_p491_e20/route_family_summary.json

work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/
  features/sam3_base_s256_features.npz
  comparison_summary_sam3enc_s256_base.{md,json}
  comparison_summary_sam3enc_s256_lora.{md,json}
```

## 9. 执行说明

本轮已按“只做未微调 SAM3”的口径重新执行 base 部分：

```text
GPU:        NVIDIA GeForce RTX 3090, CUDA_VISIBLE_DEVICES=0
checkpoint: /Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
canvas:     256
范围:       s1008 + s256，共 7 个 SAM3-enc KNN 变体
完成时间:   2026-08-13 21:04:32
```

执行时使用 `--resume`，已存在的 700 条成功记录被跳过；汇总文件
`comparison_summary_sam3enc_s1008_base.md` 与
`comparison_summary_sam3enc_s256_base.md` 已重新生成，数字与本文第 5.1、
第 6.1 节一致。LoRA 结果本轮未重跑，仍沿用已有
`lora_p491_e20` 产物。
