# Round-2A：固定 SAM3-base KNN topology，仅增强 propagation teacher，并验证 X3→X4 反哺

> 报告日期：2026-08-25（Asia/Shanghai）  
> 服务器：`violet@222.31.141.50`  
> 项目根目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7`  
> 实验根目录：`work/rerun_c0_256_round2a_fixed_knn_e33/`  
> 实验完成：2026-08-25 05:31:13  
> 原始结果：`x3_vs_x4.json`、`teacher_quality_comparison.json`、`students/X4/summary.json`、`manifests/*.summary.json`。

---

## 1. 实验问题与结论

本实验回答：

> 在完全固定 `SAM3-base@256` KNN topology、固定 anchor、固定旧 X3 的情况下，仅把 propagation teacher 从未微调 SAM3-base 替换为 LoRA e33，重新计算传播质量和 B7 伪标签，是否能让下一代 Student-X4 优于原 Student-X3？

控制实验的答案是：**可以，validation-best 和 test-best 均有提升。**

| 指标 | X3 | X4 | X4 − X3 |
|---|---:|---:|---:|
| best validation Dice | 0.826032 | **0.839122** | **+0.013090** |
| best validation iteration | 28800 | 26000 | -2800 |
| best test Dice | 0.853920 | **0.860007** | **+0.006087** |
| best test IoU | 0.771766 | **0.781950** | **+0.010184** |
| final validation Dice | 0.814216 | **0.828673** | **+0.014457** |
| final test Dice | **0.858591** | 0.855655 | -0.002936 |
| final test IoU | 0.778503 | **0.779323** | +0.000820 |

正式比较应采用各自由 validation 决定的 best checkpoint，因此本实验的主要结果是：

```text
X3 validation-best Test Dice = 0.8539203507540273
X4 validation-best Test Dice = 0.8600071077128872
absolute improvement          = 0.006086756958859962
```

X4 final 的 test Dice 略低于 X3 final，说明不能宣称“所有 checkpoint 均提升”。能够成立的结论是：**教师增强后，validation 选择出的下一代学生更好。**

---

## 2. 控制变量与唯一上游变化

### 2.1 固定 KNN 图

```text
G0 = KNN(F_SAM3-base)
```

以下项目不变：

- KNN 特征网络：frozen SAM3-base image trunk；
- KNN 特征输入分辨率：256×256；
- feature cache：base SAM3 产生的同一份 1024 维特征；
- route topology：同一批 target-pooling / patch-correspondence `routes.jsonl`；
- route bridge：b0-b6；
- 固定人工 anchors：8；
- train/validation/test 划分：800/100/100；
- propagation canvas：256×256；
- B7 中的 `q_model`：仍用已有 X3 validation-best prediction；
- 学生网络：`SamUnet`；
- 学生训练 seed、batch 配方、iteration 数与 validation 间隔：与 X3 一致。

唯一替换的上游模型为：

```text
SAM3 propagation checkpoint:

sam3.pt
    ↓
e33_merged_video.pt
```

e33 来自此前 `SAM3-KNN@256 b0-b6` 全模块 LoRA 的 Direct validation-best：

```text
work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/
└── e33_full_evaluation/e33_merged_video.pt
```

### 2.2 可核查的 topology 身份

KNN feature cache：

```text
work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/features/
sam3_base_s256_features.npz

SHA256:
ab8ca194f9c789ad9f6b641d382ed848751f29b23ac0bdb29a9c6722fd599907
```

固定 train routes：

```text
target-pooling routes SHA256:
4ee121f46bffda6a5c9eae529451dd966f0ba532db36987b72a63afab72933a2

patch-correspondence routes SHA256:
eefe9de51015840b756e1bc8d4eb63d20709490f17281032eb8eb0a2fe3410fb

e33 merged video checkpoint SHA256:
2eb33d5be28baf8d579eddfec2a75d2ee7b8a0c26a600603983da2db99a52280
```

Round-2A 通过符号链接复用原 `routes.jsonl`，而不是用 e33 重提 KNN 特征或重生成路线。因此它测的是“教师变强”，不是“graph 变好”。

---

## 3. 整体流程

```text
冻结 SAM3-base KNN graph G0
    ↓
复用 e33 validation/test propagation
    ↓
补跑 e33 train propagation：2 modes × 792 targets × b0-b6
    ↓
重新计算 q_return、q_multi
    ↓
使用冻结 X3-best 计算 q_model
    ↓
validation B7 threshold recalibration
    ↓
按冻结 threshold 选择 train pseudo pool
    ↓
保留 X3 的 original / tier_a / tier_b stream 配方
    ↓
训练 Student-X4，40000 iterations
    ↓
分别用 validation 选 X3 / X4 best checkpoint
    ↓
在冻结后比较 X3 / X4 test
```

B7 定义保持：

```text
B7 = (q_return × q_multi² × q_model²)^0.2
```

其中 `b7_mean` 仅表示候选选择置信度；真正分割性能看 `gt_dice_evaluation_only` 或最终 student `dice`。

---

## 4. Step 1：e33 propagation

### 4.1 复用既有 validation/test

此前 e33 已在**完全相同的 topology**上完成：

```text
validation: 2 modes × 7 bridges × 100 targets = 1400 routes
test:       2 modes × 7 bridges × 100 targets = 1400 routes
```

Round-2A 直接链接这些既有结果，不重复计算。

### 4.2 补做 train propagation

```text
target-pooling:       792 × 7 = 5544 routes
patch-correspondence: 792 × 7 = 5544 routes
total:                         11088 routes
```

启动时间：2026-08-24 19:18:38。  
完成时间：2026-08-25 04:03:25。

运行脚本：

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

nohup env DEVICE=0 \
  bash scripts/run_c0_256_round2a_e33_train_propagation.sh \
  > work/rerun_c0_256_round2a_fixed_knn_e33/nohup_train_propagation.log \
  2>&1 < /dev/null &
```

核心传播调用：

```bash
/home/violet/anaconda3/envs/sam3/bin/python \
  scripts/eval_route_propagation_quality.py \
  --checkpoint \
    work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/e33_full_evaluation/e33_merged_video.pt \
  --mode sam3enc_anchor_conditioned_target_pooling \
  --root work/rerun_c0_256_round2a_fixed_knn_e33/quality_root \
  --split train --canvas 256 --resume
```

patch 模式仅替换：

```text
--mode sam3enc_anchor_conditioned_patch_correspondence
```

### 4.3 train b0-b6 Dice

train GT 仅用于事后质量审计，不参与 B7 选择或训练 pool 筛选。

| bridge | target-pooling | patch-correspondence | combined |
|---|---:|---:|---:|
| b0 | 0.751324 | 0.778864 | 0.765094 |
| b1 | 0.817083 | 0.822922 | 0.820002 |
| b2 | 0.848573 | 0.860374 | 0.854474 |
| b3 | 0.865696 | 0.854607 | 0.860151 |
| b4 | 0.871333 | 0.861464 | 0.866398 |
| b5 | 0.873509 | 0.866301 | 0.869905 |
| b6 | **0.874573** | **0.872191** | **0.873382** |

### 4.4 validation b0-b6 Dice

| bridge | target-pooling | patch-correspondence | combined |
|---|---:|---:|---:|
| b0 | 0.716743 | 0.757399 | 0.737071 |
| b1 | 0.818560 | 0.828144 | 0.823352 |
| b2 | 0.846189 | 0.851841 | 0.849015 |
| b3 | 0.881616 | 0.868521 | 0.875069 |
| b4 | 0.869286 | 0.867411 | 0.868348 |
| b5 | 0.882076 | **0.883561** | 0.882818 |
| b6 | **0.887607** | 0.881546 | **0.884577** |

### 4.5 test b0-b6 Dice

| bridge | target-pooling | patch-correspondence | combined |
|---|---:|---:|---:|
| b0 | 0.796551 | 0.828143 | 0.812347 |
| b1 | 0.809615 | 0.840549 | 0.825082 |
| b2 | 0.875972 | 0.856388 | 0.866180 |
| b3 | 0.891441 | 0.878680 | 0.885061 |
| b4 | **0.904009** | 0.885240 | 0.894624 |
| b5 | 0.897443 | 0.893036 | 0.895240 |
| b6 | 0.899980 | **0.908151** | **0.904065** |

这些数字是固定 bridge / mode 的传播 Dice，并不是 X4 student Dice，也不是 B7 逐 target 选择 Dice。

---

## 5. Step 2：固定 X3，比较 base teacher 与 e33 teacher

比较对象使用：

- 同一批 validation targets；
- 同一批 1400 条候选 routes；
- 同一 KNN graph；
- 同一个 X3-best 预测作为 `q_model`；
- 唯一变化：propagation checkpoint。

### 5.1 全部 1400 candidates

| 指标 | SAM3-base | SAM3-e33 | delta |
|---|---:|---:|---:|
| `q_return` | 0.807626 | **0.915586** | **+0.107959** |
| `q_multi` | 0.869469 | **0.904258** | **+0.034789** |
| `q_model` | 0.813009 | **0.833374** | **+0.020365** |
| B7 | 0.756561 | **0.856623** | **+0.100062** |
| GT Dice，仅审计 | 0.796126 | **0.845750** | **+0.049625** |

传播教师增强后，提升主要首先体现在 cycle/self-consistency 和候选间一致性；`q_model` 虽然也提升，但幅度更小。

### 5.2 每个 target 的最高 B7 候选，不设 threshold

| 指标 | SAM3-base | SAM3-e33 | delta |
|---|---:|---:|---:|
| `q_return` | 0.942507 | **0.961108** | +0.018601 |
| `q_multi` | 0.874592 | **0.927519** | +0.052928 |
| `q_model` | **0.881545** | 0.875445 | -0.006100 |
| B7 | 0.875782 | **0.894803** | +0.019021 |
| selected Val Dice | 0.833866 | **0.883854** | **+0.049988** |

值得注意：最高 B7 路线对 X3 的 `q_model` 略微下降，但 `q_return` 与 `q_multi` 提升足以把真实 selected Dice 提高接近 0.05。这说明 teacher 提供了不同于原 X3 prediction 的有效纠正信号，而不是单纯复述学生预测。

### 5.3 冻结阈值后接受的 validation 候选

| 指标 | base accepted | e33 accepted | delta |
|---|---:|---:|---:|
| accepted targets | 54 | 56 | +2 |
| `q_return` | 0.970773 | 0.972920 | +0.002146 |
| `q_multi` | 0.981542 | 0.988687 | +0.007145 |
| `q_model` | 0.967463 | 0.967594 | +0.000132 |
| B7 | 0.973627 | 0.976947 | +0.003320 |
| GT Dice，仅审计 | **0.950447** | 0.950100 | -0.000347 |

两组 accepted 伪标签的 validation 质量被对齐在 0.95 左右，因此 X4 实验不是靠明显降低质量阈值换更大 pool。

---

## 6. Step 3：validation B7 recalibration

### 6.1 threshold sweep

| B7 threshold | kept | coverage | selected Val Dice |
|---:|---:|---:|---:|
| 0.00 | 100/100 | 100% | 0.883854 |
| 0.90 | 72/100 | 72% | 0.936410 |
| 0.92 | 69/100 | 69% | 0.939500 |
| 0.94 | 60/100 | 60% | 0.943289 |
| 0.96 | 49/100 | 49% | 0.952435 |
| 0.97 | 41/100 | 41% | 0.953298 |
| 0.98 | 35/100 | 35% | 0.954414 |

规则预先固定为：

```text
在 selected validation Dice >= 0.95 的所有 B7 前缀中，
选择 coverage 最大的一组。
```

由完整 validation frontier 得到：

```text
frozen B7 threshold = 0.9460352822729875
validation selected = 56 / 100
validation coverage = 56%
selected Val Dice   = 0.9500997548980131
```

对照 base teacher 原阈值：

```text
base B7 threshold = 0.94
selected          = 54 / 100
selected Val Dice = 0.9504472117345039
```

validation 仅用于阈值和 checkpoint 选择；没有把 validation mask 加入训练。train/test GT 不参与 threshold 决策。

### 6.2 校准命令

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
bash scripts/run_c0_256_round2a_b7_calibration.sh
```

核心命令：

```bash
/home/violet/anaconda3/envs/sam3/bin/python \
  scripts/build_c0_256_b7_lora_manifest.py \
  --quality-root work/rerun_c0_256_round2a_fixed_knn_e33/quality_root \
  --student-predictions \
    work/rerun_c0_256_sam3knn_s256_base/predictions/X3_best/student_predictions_validation.jsonl \
  --split validation \
  --modes \
    sam3enc_anchor_conditioned_target_pooling \
    sam3enc_anchor_conditioned_patch_correspondence \
  --min-bridge 0 --max-bridge 6 \
  --min-b7 0.9460352822729875 \
  --canvas 256 \
  --output \
    work/rerun_c0_256_round2a_fixed_knn_e33/b7_calibration/validation_frozen.jsonl
```

---

## 7. Step 4：新的 train pseudo pool

冻结阈值后，对 792 个非 anchor train targets 应用 B7，不利用 train GT 做筛选：

```text
candidate targets = 792
selected targets  = 524
coverage          = 66.1616%
minimum B7        = 0.9460352822729875
```

selected train GT Dice=0.9262967124174806，仅为筛选完成后的审计，不参与选择。

与上一轮 base teacher 的高置信 B7 pool 对比：

```text
base teacher B7 pool = 486 / 792
e33 teacher B7 pool  = 524 / 792
increase             = 38 targets
```

与旧 X3 manifest 的目标关系：

| 项目 | 数量 |
|---|---:|
| e33 新 pool 总数 | **524** |
| 与 X3 manifest 重叠 | 504 |
| X4 新增 target | 20 |
| 原 X3 manifest 总数 | 632 |
| 原 X3 targets 中未保留 | 128 |

注意：即便 target 已存在于旧 X3 manifest，其训练 mask 也会替换为由 **e33 propagation + X3/B7** 新选择的 mask；不是简单复用旧 X3 pseudo mask。

### 7.1 S27 stream 类型

为了保持 X3 的三流训练配置，旧 X3 中已存在的 target 继承其原 `sample_type`；新加入的 target 进入 `original` hard-label stream。

| sample type | targets |
|---|---:|
| original | 394 |
| tier_a | 72 |
| tier_b | 58 |
| total | **524** |

per-image `explicit_quality_weight` 使用新的 B7；旧 teacher 的 `pseudo_consensus_path` 与 `pixel_weight_path` 不再沿用，避免混入 base teacher 的 mask。

---

## 8. Step 5：Student-X4 训练

### 8.1 固定训练配置

| 项目 | X3 | X4 |
|---|---|---|
| 网络 | `SamUnet` | `SamUnet` |
| input | 256×256 | 256×256 |
| seed | 2026 | 2026 |
| optimizer learning rate | 0.01 | 0.01 |
| max iterations | 40000 | 40000 |
| validation interval | 200 | 200 |
| batch size | 12 | 12 |
| GT / original / tier A+B | 3 / 3 / 6 | 3 / 3 / 6 |
| pseudo coefficient | 0.5 | 0.5 |
| original / tier A / tier B weight | 1.0 / 0.75 / 0.50 | 1.0 / 0.75 / 0.50 |
| pseudo ramp | 2000 iterations | 2000 iterations |

X4 运行时间：2026-08-25 04:04:16 至 05:31:04。

启动：

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

DEVICE=0 B7_THRESHOLD=0.9460352822729875 \
  bash scripts/run_c0_256_round2a_build_pool_and_x4.sh
```

底层训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/violet/anaconda3/envs/mkunet_mamba/bin/python \
  scripts/run_s27_student.py \
  --data-path work/kvasir_1pct_anchors/baseline_data \
  --labeled-list work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt \
  --pseudo-manifest \
    work/rerun_c0_256_round2a_fixed_knn_e33/manifests/x4_train_b7.jsonl \
  --output-dir \
    work/rerun_c0_256_round2a_fixed_knn_e33/students/X4 \
  --experiment X4_round2a --seed 2026 \
  --batch-size 12 --gt-bs 3 --original-bs 3 --new-bs 6 \
  --max-iterations 40000 --val-interval 200 --num-workers 4 \
  --resume
```

### 8.2 与 X3 在同 iteration 的 validation 对比

| iteration | X3 Val Dice | X4 Val Dice | X4 − X3 |
|---:|---:|---:|---:|
| 2000 | 0.689412 | 0.710362 | +0.020950 |
| 4000 | 0.744245 | 0.765052 | +0.020808 |
| 8000 | 0.789594 | 0.783918 | -0.005676 |
| 12000 | 0.804632 | 0.811639 | +0.007007 |
| 16000 | 0.805306 | 0.823318 | +0.018012 |
| 20000 | 0.798668 | 0.801790 | +0.003122 |
| 24000 | 0.817594 | 0.826731 | +0.009137 |
| **26000** | 0.808162 | **0.839122** | **+0.030960** |
| 28000 | 0.812269 | 0.828106 | +0.015837 |
| **28800** | **0.826032** | 0.824922 | -0.001110 |
| 32000 | 0.809111 | 0.827340 | +0.018229 |
| 36000 | 0.820626 | 0.825201 | +0.004575 |
| 40000 | 0.814216 | 0.828673 | +0.014457 |

两位学生各记录 200 个 validation checkpoint。X3 best=28800，X4 best=26000，选择依据均为各自 validation Dice，而不是 test。

---

## 9. Step 6：冻结后 X3 vs X4 test

### 9.1 overall

| student/checkpoint | Val Dice | test Dice | test IoU | nonempty |
|---|---:|---:|---:|---:|
| X3 best | 0.826032 | 0.853920 | 0.771766 | 1.00 |
| **X4 best** | **0.839122** | **0.860007** | **0.781950** | 1.00 |
| X3 final | 0.814216 | **0.858591** | 0.778503 | 1.00 |
| X4 final | **0.828673** | 0.855655 | **0.779323** | 1.00 |

best-vs-best 结论：

```text
validation Dice +0.013089973237996544
test Dice       +0.006086756958859962
test IoU        +0.0101842030081990
```

final-vs-final 结论：

```text
validation Dice +0.014456904664201464
test Dice       -0.0029360926439946367
test IoU        +0.0008199539395884
```

### 9.2 best checkpoint 按目标大小拆分

| test size | count | X3 best Dice | X4 best Dice | delta |
|---|---:|---:|---:|---:|
| small | 24 | 0.853231 | **0.866598** | +0.013367 |
| medium | 21 | **0.899043** | 0.879945 | -0.019098 |
| large | 55 | 0.836993 | **0.849519** | +0.012526 |

收益主要来自 small 与 large；medium 是当前明显短板，后续若继续迭代应单独检查中等目标的伪标签 mask 与 route mode 分布。

### 9.3 final checkpoint 按目标大小拆分

| test size | count | X3 final Dice | X4 final Dice | delta |
|---|---:|---:|---:|---:|
| small | 24 | 0.849371 | **0.887087** | +0.037717 |
| medium | 21 | **0.909392** | 0.887889 | -0.021503 |
| large | 55 | **0.843217** | 0.829631 | -0.013586 |

因此 final 的整体 Dice 未提升，主要由 medium/large 抵消了 small 的收益。

### 9.4 评测命令

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/violet/anaconda3/envs/mkunet_mamba/bin/python \
  scripts/eval_c0_256_round2a_x3_x4.py \
  --x3-run work/rerun_c0_256_sam3knn_s256_base/students/X3 \
  --x4-run work/rerun_c0_256_round2a_fixed_knn_e33/students/X4 \
  --output work/rerun_c0_256_round2a_fixed_knn_e33/x3_vs_x4.json
```

checkpoint SHA256：

```text
X3 best:  52345e22283fa9aa6af48124fee60b3c24e1f389615ebe35dd42e120ec09e03c
X3 final: 572c693e0a533da2cb9104fea68dcc1b02c2ad3c88c45910437f12d1a298d6c2
X4 best:  08fdbe2b706d6e43ef0176ac6cb23bb9f4c3257cc9354ee7a813a53377e0b7f0
X4 final: d9cf966b4081074526b8d3870c7c2b0b587e381f4bdd85f955ac93627eeaff6b
```

---

## 10. 自动流水线与文件索引

### 10.1 自动串联命令

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

nohup bash scripts/supervise_c0_256_round2a_pipeline.sh \
  > work/rerun_c0_256_round2a_fixed_knn_e33/nohup_pipeline_supervisor.log \
  2>&1 < /dev/null &
```

supervisor 顺序：

```text
等待两种 train propagation 完成
→ 读取 validation frontier 冻结阈值
→ 构建 e33+B7 train manifest
→ 转换为 X3 同配方的 original/tier_a/tier_b pools
→ 训练 X4
→ 跑 X3/X4 best + final test
→ 生成 ROUND2A_COMPLETE
```

### 10.2 目录结构

```text
work/rerun_c0_256_round2a_fixed_knn_e33/
├── quality_root/
│   ├── sam3enc_anchor_conditioned_target_pooling/
│   │   ├── train_pool0_stage1 -> frozen base KNN routes
│   │   ├── propagation_quality_train/
│   │   ├── propagation_quality_validation -> existing e33 result
│   │   └── propagation_quality_test -> existing e33 result
│   └── sam3enc_anchor_conditioned_patch_correspondence/
├── base_control/
│   ├── validation_all_candidates.jsonl
│   ├── validation_min000.jsonl
│   └── validation_min094.summary.json
├── b7_calibration/
│   ├── validation_all_candidates.jsonl
│   ├── calibration_frontier.json
│   ├── calibration_frontier.tsv
│   ├── validation_frozen.jsonl
│   └── validation_frozen.summary.json
├── manifests/
│   ├── x4_train_b7_raw.jsonl
│   ├── x4_train_b7_raw.summary.json
│   ├── x4_train_b7.jsonl
│   └── x4_train_b7.summary.json
├── students/X4/
│   ├── protocol.json
│   ├── validation.jsonl
│   ├── student_best.pth
│   ├── student_final.pth
│   └── summary.json
├── teacher_quality_comparison.json
├── two_mode_b0_b6_train.{json,tsv}
├── two_mode_b0_b6_validation.{json,tsv}
├── two_mode_b0_b6_test.{json,tsv}
├── x3_vs_x4.json
├── train_propagation.log
├── x4_train.log
├── pipeline_supervisor.log
└── ROUND2A_COMPLETE
```

### 10.3 新增或扩展脚本

```text
scripts/run_c0_256_round2a_e33_train_propagation.sh
scripts/run_c0_256_round2a_b7_calibration.sh
scripts/summarize_round2a_b7_calibration.py
scripts/summarize_round2a_teacher_quality.py
scripts/build_c0_256_b7_lora_manifest.py
scripts/prepare_round2a_x4_manifest.py
scripts/run_c0_256_round2a_build_pool_and_x4.sh
scripts/eval_c0_256_round2a_x3_x4.py
scripts/supervise_c0_256_round2a_pipeline.sh
```

---

## 11. 结论边界

本报告能够支持：

1. 在相同 SAM3-base KNN topology 上，e33 propagation 的候选平均 Val Dice 比 base 增加 0.049625。
2. 固定 X3 `q_model` 后，未阈值化的 B7 selected Val Dice 增加 0.049988。
3. 在约 0.95 的 validation 伪标签质量水平上，e33 的 coverage 达 56%，base 为 54%。
4. 新 teacher 最终生成 524 个 train pseudo labels，比上一轮 base teacher 的 486 个多 38 个。
5. 使用同架构、同 seed、同训练 schedule 和同三流 batch 配方，X4 best validation/test 均高于 X3 best。
6. e33+X3-best+B7 test 为 0.906380，e33+X4-best+B7 test 为 0.905375，均高于 base+X3+B7 的 0.895432。

不能据此支持：

1. “所有 checkpoint 都更好”：X4 final Test Dice 低于 X3 final。
2. “所有目标大小都更好”：medium test target 出现退化。
3. “KNN graph 更新带来收益”：本实验根本没有更新 graph。
4. “X4 的单图提升必然带来 B7 提升”：X4+B7 test 0.905375 略低于 X3+B7 test 0.906380。

最准确的一句话结论：

> 固定 KNN topology 后，SAM3-e33 teacher 可以把 X4 单图 student Test Dice 从 0.853920 提升到 0.860007；但在相同 e33 candidates 上，B7 validation-selected 最优 selector 仍为 X3-best，最终 B7 Test Dice 0.906380，X4-best+B7 为 0.905375。

---

## 12. Round-2A 最终闭环：e33 + X3/X4 + B7

> 完成时间：2026-08-25 12:12:50（Asia/Shanghai）。本阶段结束于 B7，不启动下一轮 SAM3 微调。

### 12.1 控制变量与 checkpoint 选择

本阶段的问题是：同一批 e33 propagation candidates 不变，只替换参与 `q_model` 的 student prediction，X4 是否能提高 B7 最终选路结果？

- KNN topology 固定为第一轮 `SAM3-base@256` 生成的 topology，不重新构图。
- propagation checkpoint 固定为上一轮 Direct validation-best 的 SAM3-e33。
- 同时保留 `sam3enc_anchor_conditioned_target_pooling` 与 `sam3enc_anchor_conditioned_patch_correspondence`。
- 每个 target 同时比较 b0-b6，共 7 个 bridge；两模式最多提供 14 条候选。
- validation/test 均使用全部 100 个 target；评估时 `--min-b7 0`，不筛掉难例。
- `q_return`、`q_multi` 和 propagation masks 对所有 selector 完全相同；唯一变化是 student prediction，从而变化的是 `q_model` 以及最终 B7 排序。
- `X3_best` 和 `X4_best` 分别来自各自 student validation-best checkpoint；`X4_final` 仅用于诊断，不能根据 test 反选为正式 checkpoint。
- `b7_mean` 是 B7 内部置信分数的平均值，不是分割 Dice；以下只用 `selected_gt_dice_evaluation_only` 报告真实 Dice。

### 12.2 B7 validation/test 完整结果

| propagation | student selector | Validation Dice | Test Dice | Val oracle gap | Test oracle gap | 用途 |
|---|---|---:|---:|---:|---:|---|
| SAM3-e33 | X3-best | **0.8838539036343016** | **0.9063803067687748** | 0.0212681391592575 | 0.0138256213496096 | 同 candidates 的历史 student 对照；B7 validation 最优 |
| SAM3-e33 | X4-best | 0.8800700315716979 | 0.9053750856794797 | 0.0250520112218612 | 0.0148308424389048 | Round-2 新 student 的正式 checkpoint |
| SAM3-e33 | X4-final | 0.8804670161809023 | 0.9053849114608847 | 0.0246550266126568 | 0.0148210166574997 | 仅诊断，不参与模型选择 |

同一 e33 candidate pool 的 oracle 对三种 selector 一致：

```text
validation oracle Dice = 0.9051220427935591
test oracle Dice       = 0.9202059281183844
```

因此 oracle gap 的差异完全来自 B7 选路，而不是候选本身变化。`X4_best - X3_best` 为：

```text
validation: -0.0037838720626037
test:       -0.0010052210892951
```

### 12.3 与未微调 SAM3-base 教师对照

| KNN topology | propagation teacher | student selector | B7 Test Dice | 相对 base teacher | Test oracle Dice | Test oracle gap |
|---|---|---|---:|---:|---:|---:|
| 固定 SAM3-base@256 | SAM3-base | X3-best | 0.8954317676013283 | baseline | 0.9214706098083678 | 0.0260388422070396 |
| 固定 SAM3-base@256 | SAM3-e33 | X3-best | **0.9063803067687748** | **+0.0109485391674465** | 0.9202059281183844 | 0.0138256213496096 |
| 固定 SAM3-base@256 | SAM3-e33 | X4-best | 0.9053750856794797 | +0.0099433180781514 | 0.9202059281183844 | 0.0148308424389048 |

e33 的 candidate oracle 略低于 base，但最终 selected Dice 更高，因为 e33+X3 的 selector-to-oracle gap 从 0.026039 缩小到 0.013826。故本轮主要收益不是 oracle 上界提高，而是实际候选排序/选择更有效。

### 12.4 为什么 X4 单图更强，但 X4+B7 略差

student 单图测试结果：

```text
X3-best student direct Test Dice = 0.8539203507540273
X4-best student direct Test Dice = 0.8600071077128872
X4 - X3                         = +0.0060867569588600
```

但 B7 使用 student mask 去衡量它与 e33 propagation candidate 的一致性。单图 Dice 提高，不保证这种一致性分数对候选优劣的排序也同步提高。

- validation：100 个 target 中 64 个选中同一路线，36 个更换；其中 X4 改善 15 个、退化 21 个。
- test：100 个 target 中 56 个选中同一路线，44 个更换；其中 X4 改善 21 个、退化 23 个。
- 仅看 test 中改选的 44 个 target，X3 所选 mask 平均 Dice 为 0.927092，X4 所选为 0.924808。

| Test selected bridge | X3-best | X4-best |
|---|---:|---:|
| b0 | 13 | 10 |
| b1 | 5 | 7 |
| b2 | 9 | 9 |
| b3 | 12 | 16 |
| b4 | 17 | 12 |
| b5 | 21 | 24 |
| b6 | 23 | 22 |

两模式选择数量分别为：X3-best 的 patch-correspondence/target-pooling = 33/67；X4-best = 34/66。

结论：本轮证明了“更强 teacher 能反哺 student”，但没有证明“更强 student 一定反哺 B7 selector”。若依据 B7 validation 选择最终方案，应选择 `e33 + X3-best + B7`；若报告 Round-2 新 student 完整闭环，则报告 `e33 + X4-best + B7`。

### 12.5 实验指令与复现

从项目根目录执行完整闭环；脚本会在 GPU0/GPU1 分别导出 X4-best/X4-final，然后依次评估 X3-best、X4-best 和 X4-final 的 validation/test：

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
bash scripts/run_c0_256_round2_x4_b7_closeout.sh
```

导出单个 X4-best checkpoint 的等价命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/violet/anaconda3/envs/mkunet_mamba/bin/python \
  scripts/export_t25_student_predictions.py \
  --run-dir work/rerun_c0_256_round2a_fixed_knn_e33/students/X4 \
  --checkpoint work/rerun_c0_256_round2a_fixed_knn_e33/students/X4/student_best.pth \
  --output-root work/rerun_c0_256_round2a_fixed_knn_e33/predictions/X4_best \
  --splits train validation test
```

手动重算 X4-best 的 test B7：

```bash
/home/violet/anaconda3/envs/sam3/bin/python \
  scripts/build_c0_256_b7_lora_manifest.py \
  --quality-root work/rerun_c0_256_round2a_fixed_knn_e33/quality_root \
  --student-predictions work/rerun_c0_256_round2a_fixed_knn_e33/predictions/X4_best/student_predictions_test.jsonl \
  --split test \
  --modes sam3enc_anchor_conditioned_target_pooling sam3enc_anchor_conditioned_patch_correspondence \
  --min-bridge 0 --max-bridge 6 --min-b7 0 --canvas 256 \
  --output work/rerun_c0_256_round2a_fixed_knn_e33/x4_b7_closeout/X4_best_test.jsonl \
  --summary work/rerun_c0_256_round2a_fixed_knn_e33/x4_b7_closeout/X4_best_test.summary.json
```

汇总六组结果：

```bash
/home/violet/anaconda3/envs/sam3/bin/python \
  scripts/summarize_c0_256_round2_b7_closeout.py \
  --root work/rerun_c0_256_round2a_fixed_knn_e33/x4_b7_closeout \
  --output work/rerun_c0_256_round2a_fixed_knn_e33/x4_b7_closeout/x3_x4_e33_b7_comparison.json
```

### 12.6 产物索引与停止边界

```text
work/rerun_c0_256_round2a_fixed_knn_e33/
├── predictions/X4_best/
├── predictions/X4_final/
├── x4_b7_closeout/
│   ├── X3_best_validation.jsonl
│   ├── X3_best_validation.summary.json
│   ├── X3_best_test.jsonl
│   ├── X3_best_test.summary.json
│   ├── X4_best_validation.jsonl
│   ├── X4_best_validation.summary.json
│   ├── X4_best_test.jsonl
│   ├── X4_best_test.summary.json
│   ├── X4_final_validation.jsonl
│   ├── X4_final_validation.summary.json
│   ├── X4_final_test.jsonl
│   ├── X4_final_test.summary.json
│   └── x3_x4_e33_b7_comparison.json
├── x4_b7_closeout.log
└── X4_B7_COMPLETE
```

本轮到此结束：**不构建新一轮 SAM3 训练数据集，不启动新的 SAM3/LoRA 训练**。
