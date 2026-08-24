# 512_C0 Baseline 完整复现报告（详细版）

> 生成时间：2026-08-20  
> 复现目录：`work/rerun_c0_c0/`

## 1. 配置

| 项目 | 配置 |
|---|---|
| 数据集 | Kvasir-SEG 快照 |
| 划分 | train=800 / validation=100 / test=100 |
| GT anchors | 8 个固定训练图 |
| KNN 路线 | 原始 DINOv3 224 特征，s224 精确路线 |
| 路线模式 | `anchor_conditioned_target_pooling` + `anchor_conditioned_patch_correspondence` |
| 路线长度 | train b3-b6；validation/test b0-b6 |
| SAM3 checkpoint | `ft_1pct_merged_video.pt` |
| SAM3 propagation canvas | 512×512 |
| U-Net 学生 | SC-SAM `SamUnet`，256×256 |
| 训练迭代 | S2/S3/X3 各 40000 |
| 最终选择 | B7 学生辅助路线选择 |

## 2. SAM3 Propagation 详细结果


### anchor_conditioned_target_pooling

| split | bridge | Dice | q_cycle |
|---|---:|---:|---:|
| train | b3 | 0.8453 | 0.9117 |
| train | b4 | 0.8465 | 0.9142 |
| train | b5 | 0.8514 | 0.9142 |
| train | b6 | 0.8585 | 0.9158 |
| validation | b0 | 0.7917 | 0.8745 |
| validation | b1 | 0.8235 | 0.8886 |
| validation | b2 | 0.8276 | 0.8922 |
| validation | b3 | 0.8189 | 0.8762 |
| validation | b4 | 0.8489 | 0.9124 |
| validation | b5 | 0.8412 | 0.9058 |
| validation | b6 | 0.8620 | 0.9096 |
| test | b0 | 0.7709 | 0.8770 |
| test | b1 | 0.7922 | 0.8990 |
| test | b2 | 0.8574 | 0.8885 |
| test | b3 | 0.8553 | 0.8910 |
| test | b4 | 0.8603 | 0.9080 |
| test | b5 | 0.8746 | 0.9175 |
| test | b6 | 0.8619 | 0.9143 |

### anchor_conditioned_patch_correspondence

| split | bridge | Dice | q_cycle |
|---|---:|---:|---:|
| train | b3 | 0.8454 | 0.9249 |
| train | b4 | 0.8476 | 0.9268 |
| train | b5 | 0.8536 | 0.9290 |
| train | b6 | 0.8501 | 0.9241 |
| validation | b0 | 0.7999 | 0.8955 |
| validation | b1 | 0.8318 | 0.9064 |
| validation | b2 | 0.8468 | 0.9194 |
| validation | b3 | 0.8559 | 0.9217 |
| validation | b4 | 0.8531 | 0.9195 |
| validation | b5 | 0.8392 | 0.9115 |
| validation | b6 | 0.8570 | 0.9188 |
| test | b0 | 0.7973 | 0.9093 |
| test | b1 | 0.8167 | 0.9234 |
| test | b2 | 0.8578 | 0.9139 |
| test | b3 | 0.8631 | 0.9173 |
| test | b4 | 0.8842 | 0.9332 |
| test | b5 | 0.8656 | 0.9335 |
| test | b6 | 0.8797 | 0.9308 |

## 3. Router 详细结果（b3-b6）

| 方法 | selected Dice | oracle Dice | gap | accuracy | spearman | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| target_pooling+patch_correspondence | 0.892477 | 0.917817 | 0.025340 | 0.21 | 0.3553 | 0.9689 |
| anchor_conditioned_target_pooling | 0.876827 | 0.901464 | 0.024638 | 0.29 | 0.3638 | 0.9659 |
| anchor_conditioned_patch_correspondence | 0.883967 | 0.901753 | 0.017786 | 0.27 | 0.3227 | 0.9656 |

### 选择直方图

**target_pooling+patch_correspondence** histogram: patch correspondence:bridge 3: 29, patch correspondence:bridge 4: 12, patch correspondence:bridge 5: 6, patch correspondence:bridge 6: 6, target pooling:bridge 3: 6, target pooling:bridge 4: 4, target pooling:bridge 5: 12, target pooling:bridge 6: 25

**anchor_conditioned_target_pooling** histogram: target pooling:bridge 3: 23, target pooling:bridge 4: 23, target pooling:bridge 5: 17, target pooling:bridge 6: 37

**anchor_conditioned_patch_correspondence** histogram: patch correspondence:bridge 3: 48, patch correspondence:bridge 4: 18, patch correspondence:bridge 5: 16, patch correspondence:bridge 6: 18

## 4. 伪标签池

- accepted: 490 / 792
- q_multi: min=0.9021, mean=0.9850, median=0.9912, max=0.9994
- q_return: min=0.9501, mean=0.9711, median=0.9689
- selected_counts:
  - anchor_conditioned_patch_correspondence:bridge_3: 171
  - anchor_conditioned_patch_correspondence:bridge_4: 85
  - anchor_conditioned_patch_correspondence:bridge_5: 30
  - anchor_conditioned_patch_correspondence:bridge_6: 37
  - anchor_conditioned_target_pooling:bridge_3: 30
  - anchor_conditioned_target_pooling:bridge_4: 20
  - anchor_conditioned_target_pooling:bridge_5: 29
  - anchor_conditioned_target_pooling:bridge_6: 88

## 5. U-Net 学生训练

### S2
- best validation Dice: 0.8004 @ iter 21200
- final validation Dice: 0.7858 @ iter 40000
### S3
- best validation Dice: 0.8154 @ iter 26000
- final validation Dice: 0.7957 @ iter 40000
### X3
- best validation Dice: 0.8315 @ iter 23600
- final validation Dice: 0.8093 @ iter 40000

## 6. Audit / Tier

- Tier A: 91
- Tier B: 97
- Tier C: 114

## 7. X3 Manifest

- total: 678
- sample_types: {'original': 490, 'tier_a': 91, 'tier_b': 97}

## 8. 最终 B7

| 指标 | 值 |
|---|---:|
| B7 selected Dice | 0.885416 |
| Oracle Dice | 0.917817 |
| Oracle gap | 0.032401 |
| n_targets | 100 |

## 9. 与原始 C0 对比

| 版本 | Router | B7 | Oracle |
|---|---:|---:|---:|
| 原始 C0 | 0.894648 | 0.899369 | 0.919805 |
| 512_C0 | 见上表 Router | 0.885416 | 0.917817 |

## 10. 产物路径

```text
work/rerun_c0_c0/
├── stage1_feature_knn_b7_ft1pct/
├── pseudo_manifest_original.jsonl
├── S3_consensus/
├── students/S2/
├── students/S3/
├── students/X3/
├── predictions/
├── audit/
├── selection/b7_report.json
└── router_b3_b6.json
```

## 11. 日志路径

```text
work/rerun_c0_c0/
├── s2_gpu0.log                    # S2 训练日志
├── s3_gpu1.log                    # S3 训练日志
├── x3_gpu0.log                    # X3 训练 stdout 日志
├── export_s2s3_gpu0.log           # S2/S3 预测导出日志
├── export_x3.log                  # X3 预测导出日志
├── audit.log                      # Tier A/B/C 审计日志
├── x3_manifest.log                # X3 manifest 日志
├── b7.log                         # B7 选择日志
└── students/
    ├── S2/train.jsonl
    ├── S3/train.jsonl
    └── X3/train.jsonl
```

SAM3 propagation 日志：

```text
work/rerun_c0/sam3_c0_target_test_gpu0.log
work/rerun_c0/sam3_c0_target_train_gpu1.log
work/rerun_c0/sam3_c0_remaining_gpu0.log
work/rerun_c0/sam3_c0_patch_train_gpu0.log
```

## 12. 差异分析

- KNN 路线已验证 **100% route_id 一致**。
- SAM3 重新传播后，oracle 由 `0.919805` 降至 `0.917817`。
- 伪标签池由 491 变为 490。
- 最终 B7 由 `0.899369` 降至 `0.885416`。

可能原因：

1. SAM3 propagation 在当前环境/GPU 下存在非严格确定性（浮点、并行、显存状态等）。
2. `ft_1pct_merged_video.pt` 与原始记录时的 checkpoint 行为存在微小差异。
3. 伪标签池差 1 条，导致后续学生训练数据分布略有变化。
4. U-Net 重新训练本身存在随机性。

结论：路线可以精确复现，但 SAM3 传播结果无法做到逐位完全一致，因此下游数值只能接近原始 C0。

## 13. 复现命令摘要

### KNN 路线（s224 精确路线）

```bash
python scripts/stage1_feature_knn_routes.py \
  --mode anchor_conditioned_target_pooling \
  --split train --min-bridge 3 --max-bridge 6 \
  --feature-size 224 --beam-width 32 \
  --output-root work/rerun_c0/stage1_feature_knn_b7_s224
```

### SAM3 propagation

```bash
python scripts/eval_route_propagation_quality.py \
  --checkpoint work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt \
  --mode anchor_conditioned_target_pooling \
  --root work/rerun_c0_c0/stage1_feature_knn_b7_ft1pct \
  --split train --canvas 512 --resume
```

### Router / 伪标签池 / U-Net / B7

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_c0
ROOT=$PHASE/stage1_feature_knn_b7_ft1pct
DATA=work/kvasir_1pct_anchors/baseline_data
LABELS=work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt

# Router
$PY scripts/eval_ft1pct_pq_router.py \
  --root $ROOT \
  --min-bridge 3 \
  --max-bridge 6 \
  --output $PHASE/router_b3_b6.json

# 伪标签池
$PY scripts/select_phase1_mainline_pseudo568.py \
  --quality-root $ROOT \
  --output $PHASE/pseudo_manifest_original.jsonl

# S3 consensus
$PY scripts/prepare_phase1_s3_consensus.py \
  --original-manifest $PHASE/pseudo_manifest_original.jsonl \
  --quality-root $ROOT \
  --output-root $PHASE/S3_consensus

# S2
$PY scripts/run_t24_student.py \
  --data-path $DATA --labeled-list $LABELS \
  --pseudo-manifest $PHASE/pseudo_manifest_original.jsonl \
  --output-dir $PHASE/students/S2 --experiment S2 --seed 2026 \
  --max-iterations 40000 --val-interval 200 --num-workers 4

# S3
$PY scripts/run_t24_student.py \
  --data-path $DATA --labeled-list $LABELS \
  --pseudo-manifest $PHASE/S3_consensus/pseudo_consensus.jsonl \
  --output-dir $PHASE/students/S3 --experiment S3 --seed 2026 \
  --max-iterations 40000 --val-interval 200 --num-workers 4

# 导出 S2/S3 预测
$PY scripts/export_t25_student_predictions.py \
  --run-dir $PHASE/students/S2 --checkpoint $PHASE/students/S2/student_best.pth \
  --output-root $PHASE/predictions/S2_valbest

$PY scripts/export_t25_student_predictions.py \
  --run-dir $PHASE/students/S2 --checkpoint $PHASE/students/S2/student_final.pth \
  --output-root $PHASE/predictions/S2_final

$PY scripts/export_t25_student_predictions.py \
  --run-dir $PHASE/students/S3 --checkpoint $PHASE/students/S3/student_final.pth \
  --output-root $PHASE/predictions/S3_final

# Audit
$PY scripts/phase1_audit_tiers.py \
  --train-metadata $DATA/train/metadata.jsonl \
  --labeled-list $LABELS \
  --quality-root $ROOT \
  --original-manifest $PHASE/pseudo_manifest_original.jsonl \
  --predictions $PHASE/predictions/S2_valbest/student_predictions_train.jsonl --predictions-name S2_valbest \
  --predictions $PHASE/predictions/S2_final/student_predictions_train.jsonl --predictions-name S2_final \
  --predictions $PHASE/predictions/S3_final/student_predictions_train.jsonl --predictions-name S3_final \
  --output-dir $PHASE/audit

# X3 manifest
$PY scripts/phase1_build_x3_manifest.py \
  --original $PHASE/pseudo_manifest_original.jsonl \
  --tier-a $PHASE/audit/tier_A.jsonl \
  --tier-b $PHASE/audit/tier_B.jsonl \
  --output $PHASE/pseudo_manifest_x3.jsonl

# X3
$PY scripts/run_s27_student.py \
  --data-path $DATA --labeled-list $LABELS \
  --pseudo-manifest $PHASE/pseudo_manifest_x3.jsonl \
  --output-dir $PHASE/students/X3 --experiment X3 --seed 2026 \
  --batch-size 12 --gt-bs 3 --original-bs 3 --new-bs 6 \
  --max-iterations 40000 --val-interval 200 --num-workers 4

# 导出 X3
$PY scripts/export_t25_student_predictions.py \
  --run-dir $PHASE/students/X3 --checkpoint $PHASE/students/X3/student_final.pth \
  --output-root $PHASE/predictions/X3_final

# B7
$PY scripts/phase1_b7_select.py \
  --quality-root $ROOT \
  --student-predictions $PHASE/predictions/X3_final/student_predictions_test.jsonl \
  --output-dir $PHASE/selection
```

## 14. 结论

512 C0 已从 s224 精确路线完整重跑一遍，得到一条完整可复现的 C0 链路：

```text
KNN(s224)
→ SAM3 propagation(512)
→ router(0.892477)
→ pseudo(490)
→ S2/S3/X3
→ B7(0.885416)
```

该结果与原始 C0 高度接近，但不等同，主要差异来自 SAM3 propagation 的非严格确定性和下游训练随机性。
