# C0_256 Baseline 完整复现报告（详细版）

> 生成时间：2026-08-20  
> 复现目录：`work/rerun_c0_256/`

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
| SAM3 propagation canvas | 256×256 |
| U-Net 学生 | SC-SAM `SamUnet`，256×256 |
| 训练迭代 | S2/S3/X3 各 40000 |
| 最终选择 | B7 学生辅助路线选择 |

## 2. SAM3 Propagation 详细结果


### anchor_conditioned_target_pooling

| split | bridge | Dice | q_cycle |
|---|---:|---:|---:|
| train | b3 | 0.8457 | 0.9129 |
| train | b4 | 0.8475 | 0.9146 |
| train | b5 | 0.8570 | 0.9154 |
| train | b6 | 0.8622 | 0.9209 |
| validation | b0 | 0.8084 | 0.8732 |
| validation | b1 | 0.8220 | 0.8989 |
| validation | b2 | 0.8483 | 0.9029 |
| validation | b3 | 0.8478 | 0.8955 |
| validation | b4 | 0.8636 | 0.9131 |
| validation | b5 | 0.8609 | 0.9077 |
| validation | b6 | 0.8701 | 0.9162 |
| test | b0 | 0.7914 | 0.8894 |
| test | b1 | 0.8177 | 0.8919 |
| test | b2 | 0.8656 | 0.8868 |
| test | b3 | 0.8751 | 0.8992 |
| test | b4 | 0.8748 | 0.9146 |
| test | b5 | 0.8732 | 0.9163 |
| test | b6 | 0.8675 | 0.9164 |

### anchor_conditioned_patch_correspondence

| split | bridge | Dice | q_cycle |
|---|---:|---:|---:|
| train | b3 | 0.8456 | 0.9242 |
| train | b4 | 0.8544 | 0.9267 |
| train | b5 | 0.8614 | 0.9286 |
| train | b6 | 0.8580 | 0.9266 |
| validation | b0 | 0.8295 | 0.8962 |
| validation | b1 | 0.8241 | 0.9066 |
| validation | b2 | 0.8584 | 0.9198 |
| validation | b3 | 0.8610 | 0.9227 |
| validation | b4 | 0.8700 | 0.9283 |
| validation | b5 | 0.8538 | 0.9016 |
| validation | b6 | 0.8683 | 0.9203 |
| test | b0 | 0.8178 | 0.9223 |
| test | b1 | 0.8333 | 0.9158 |
| test | b2 | 0.8600 | 0.9028 |
| test | b3 | 0.8845 | 0.9249 |
| test | b4 | 0.8820 | 0.9317 |
| test | b5 | 0.8801 | 0.9339 |
| test | b6 | 0.8793 | 0.9344 |

## 3. Router 详细结果（b3-b6）

| 方法 | selected Dice | oracle Dice | gap | accuracy | spearman | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| target_pooling+patch_correspondence | 0.891288 | 0.918109 | 0.026821 | 0.19 | 0.3158 | 0.8755 |
| anchor_conditioned_target_pooling | 0.875413 | 0.904479 | 0.029066 | 0.25 | 0.3256 | 0.8704 |
| anchor_conditioned_patch_correspondence | 0.891065 | 0.902411 | 0.011347 | 0.33 | 0.2471 | 0.0000 |

### 选择直方图

**target_pooling+patch_correspondence** histogram: patch correspondence:bridge 3: 9, patch correspondence:bridge 4: 15, patch correspondence:bridge 5: 6, patch correspondence:bridge 6: 16, target pooling:bridge 3: 11, target pooling:bridge 4: 15, target pooling:bridge 5: 15, target pooling:bridge 6: 13

**anchor_conditioned_target_pooling** histogram: target pooling:bridge 3: 24, target pooling:bridge 4: 26, target pooling:bridge 5: 25, target pooling:bridge 6: 25

**anchor_conditioned_patch_correspondence** histogram: patch correspondence:bridge 3: 25, patch correspondence:bridge 4: 16, patch correspondence:bridge 5: 28, patch correspondence:bridge 6: 31

## 4. 伪标签池

- accepted: 470 / 792
- q_multi: min=0.9001, mean=0.9844, median=0.9917, max=0.9992
- q_return: min=0.9504, mean=0.9721, median=0.9705
- selected_counts:
  - anchor_conditioned_patch_correspondence:bridge_3: 86
  - anchor_conditioned_patch_correspondence:bridge_4: 66
  - anchor_conditioned_patch_correspondence:bridge_5: 54
  - anchor_conditioned_patch_correspondence:bridge_6: 88
  - anchor_conditioned_target_pooling:bridge_3: 38
  - anchor_conditioned_target_pooling:bridge_4: 36
  - anchor_conditioned_target_pooling:bridge_5: 28
  - anchor_conditioned_target_pooling:bridge_6: 74

## 5. U-Net 学生训练

### S2
- best validation Dice: 0.8092 @ iter 20200
- final validation Dice: 0.7973 @ iter 40000
### S3
- best validation Dice: 0.8131 @ iter 36600
- final validation Dice: 0.8081 @ iter 40000
### X3
- best validation Dice: 0.8208 @ iter 29400
- final validation Dice: 0.8076 @ iter 40000

## 6. Audit / Tier

- Tier A: 105
- Tier B: 102
- Tier C: 115

## 7. X3 Manifest

- total: 677
- sample_types: {'original': 470, 'tier_a': 105, 'tier_b': 102}

## 8. 最终 B7

| 指标 | 值 |
|---|---:|
| B7 selected Dice | 0.897146 |
| Oracle Dice | 0.918109 |
| Oracle gap | 0.020963 |
| n_targets | 100 |

## 9. 与原始 C0 对比

| 版本 | Router | B7 | Oracle |
|---|---:|---:|---:|
| 原始 C0 | 0.894648 | 0.899369 | 0.919805 |
| C0_256 | 见上表 Router | 0.897146 | 0.918109 |

## 10. 产物路径

```text
work/rerun_c0_256/
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
work/rerun_c0_256/
├── select.log
├── s3.log
├── router.log
├── audit.log
├── x3_manifest.log
├── export_x3.log
├── b7.log
├── x3_gpu0.log
└── students/
    ├── S2/train.jsonl
    ├── S3/train.jsonl
    └── X3/train.jsonl
```

SAM3 propagation 日志：

```text
work/rerun_c0/sam3_c0_256_all_gpu0.log
```

## 12. 差异分析

- KNN 路线与原始 C0 100% 一致。
- SAM3 propagation 改为 256×256 后，oracle 为 `0.918109`，非常接近原始 C0 的 `0.919805`。
- 伪标签池为 470，比原始 C0 少 21 条。
- 最终 B7 为 `0.897146`，与原始 C0 的 `0.899369` 只差约 `0.0022`。

结论：256×256 的 SAM3 侧与 U-Net 侧分辨率统一后，最终成绩仍接近原始 C0，且比本次 512 复现结果更接近原始成绩。后续做 SAM3↔U-Net 协同训练时，可以省去大量 resize 对齐。

## 13. 复现命令摘要

### KNN 路线（s224 精确路线）

```bash
python scripts/stage1_feature_knn_routes.py \
  --mode anchor_conditioned_target_pooling \
  --split train --min-bridge 3 --max-bridge 6 \
  --feature-size 224 --beam-width 32 \
  --output-root work/rerun_c0/stage1_feature_knn_b7_s224
```

### SAM3 propagation（256）

```bash
python scripts/eval_route_propagation_quality.py \
  --checkpoint work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt \
  --mode anchor_conditioned_target_pooling \
  --root work/rerun_c0_256/stage1_feature_knn_b7_ft1pct \
  --split train --canvas 256 --resume
```

### Router / 伪标签池 / U-Net / B7

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256
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

C0-256 完整跑通，最终：

```text
B7 selected Dice = 0.897146
Oracle Dice      = 0.918109
```

256×256 统一分辨率版本可以作为后续 SAM3↔U-Net 协同训练的基础配置。
