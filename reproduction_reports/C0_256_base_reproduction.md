# C0-256-base（未微调 SAM3 + 256×256）完整复现报告

> 生成时间：2026-08-20  
> 复现目录：`work/rerun_c0_256_base/`  
> 目标：使用**未微调 SAM3**（`sam3.pt`）在 256×256 分辨率下，完整跑通 C0 下游，作为后续 SAM3↔U-Net 协同训练的 baseline。

---

## 1. 最终成绩

| X3 checkpoint | B7 selected Dice | Oracle | gap |
|---|---:|---:|---:|
| **X3_best（best val）** | **0.886130** | 0.903589 | 0.017459 |
| X3_final（40000） | 0.875302 | 0.903589 | 0.028287 |

---

## 2. 配置总览

| 项目 | 配置 |
|---|---|
| 数据集 | Kvasir-SEG 快照 |
| 划分 | train=800 / validation=100 / test=100 |
| GT anchors | 8 个固定训练图 |
| KNN 路线 | 原始 DINOv3 224 特征，s224 精确路线 |
| 路线模式 | `anchor_conditioned_target_pooling` + `anchor_conditioned_patch_correspondence` |
| 路线长度 | train b3-b6；validation/test b0-b6 |
| SAM3 checkpoint | `/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt`（**未微调**） |
| SAM3 propagation canvas | **256×256** |
| U-Net 学生 | SC-SAM `SamUnet`，256×256 |
| 训练迭代 | S2/S3/X3 各 40000 |
| 最终选择 | B7 学生辅助路线选择 |

---

## 3. 每步指令与结果

### Step 0：KNN 路线（s224 精确路线）

指令：

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
ROOT=work/rerun_c0/stage1_feature_knn_b7_s224
PROTO=work/kvasir_1pct_anchors/protocol

for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  for split in train validation test; do
    if [ "$split" = train ]; then minb=3; maxb=6; else minb=0; maxb=6; fi
    $PY scripts/stage1_feature_knn_routes.py \
      --mode "$mode" --split "$split" \
      --min-bridge "$minb" --max-bridge "$maxb" \
      --beam-width 32 --feature-size 224 \
      --protocol-root "$PROTO" --output-root "$ROOT"
  done
done
```

结果：

```text
anchor_conditioned_target_pooling:
  train 3168 / validation 700 / test 700

anchor_conditioned_patch_correspondence:
  train 3168 / validation 700 / test 700
```

路线与原始 C0 test route 验证为 **100% route_id 一致**。

---

### Step 1：SAM3 propagation（未微调 sam3.pt，canvas 256）

指令：

```bash
PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
ROOT=work/rerun_c0_256_base/stage1_feature_knn_b7_base
SRC=work/rerun_c0/stage1_feature_knn_b7_s224

for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  for split in train validation test; do
    ln -sfn "$(pwd)/$SRC/$mode/${split}_pool0_stage1" "$ROOT/$mode/${split}_pool0_stage1"
    $PY scripts/eval_route_propagation_quality.py \
      --checkpoint "$CKPT" --mode "$mode" --root "$ROOT" \
      --split "$split" --canvas 256 --resume
  done
done
```

结果：全部完成。

```text
target train/validation/test ✅
patch train/validation/test ✅
```

---

### Step 2：Router（b3-b6 test）

指令：

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
ROOT=work/rerun_c0_256_base/stage1_feature_knn_b7_base

$PY scripts/eval_ft1pct_pq_router.py \
  --root "$ROOT" \
  --min-bridge 3 --max-bridge 6 \
  --output work/rerun_c0_256_base/router_b3_b6.json
```

结果：

| 方法 | selected Dice | oracle Dice | gap |
|---|---:|---:|---:|
| target + patch | **0.871822** | 0.903589 | 0.031767 |
| target pooling only | 0.865961 | 0.892442 | 0.026481 |
| patch correspondence only | 0.853490 | 0.872373 | 0.018883 |

---

### Step 3：伪标签池

指令：

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
ROOT=work/rerun_c0_256_base/stage1_feature_knn_b7_base
PHASE=work/rerun_c0_256_base

$PY scripts/select_phase1_mainline_pseudo568.py \
  --quality-root "$ROOT" \
  --output $PHASE/pseudo_manifest_original.jsonl

$PY scripts/prepare_phase1_s3_consensus.py \
  --original-manifest $PHASE/pseudo_manifest_original.jsonl \
  --quality-root "$ROOT" \
  --output-root $PHASE/S3_consensus
```

结果：

```text
入选伪标签：386 / 792
S3 consensus：386
```

q_multi：mean 0.9832，median 0.9903  
q_return：mean 0.9674，median 0.9686

---

### Step 4：S2 / S3 训练

指令：

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256_base
DATA=work/kvasir_1pct_anchors/baseline_data
LABELS=work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt

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
```

结果：

| 学生 | best val iter | best val Dice | final val Dice | test Dice (best) | test Dice (final) |
|---|---:|---:|---:|---:|---:|
| S2 | 22200 | 0.8006 | 0.7786 | 0.8310 | 0.8245 |
| S3 | 15600 | 0.8076 | 0.7870 | 0.8395 | 0.8264 |

---

### Step 5：导出 S2/S3 预测（best + final）

指令：

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256_base

for entry in \
  "S2 student_best.pth predictions/S2_valbest" \
  "S2 student_final.pth predictions/S2_final" \
  "S3 student_best.pth predictions/S3_valbest" \
  "S3 student_final.pth predictions/S3_final"; do
  set -- $entry
  run=$1; ckpt=$2; out=$3
  $PY scripts/export_t25_student_predictions.py \
    --run-dir $PHASE/students/$run \
    --checkpoint $PHASE/students/$run/$ckpt \
    --output-root $PHASE/$out
done
```

结果：

```text
S2_valbest test Dice = 0.8310
S2_final  test Dice = 0.8245
S3_valbest test Dice = 0.8395
S3_final  test Dice = 0.8264
```

---

### Step 6：Committee Audit

指令：

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256_base
ROOT=$PHASE/stage1_feature_knn_b7_base
DATA=work/kvasir_1pct_anchors/baseline_data
LABELS=work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt

$PY scripts/phase1_audit_tiers.py \
  --train-metadata $DATA/train/metadata.jsonl \
  --labeled-list $LABELS \
  --quality-root "$ROOT" \
  --original-manifest $PHASE/pseudo_manifest_original.jsonl \
  --predictions $PHASE/predictions/S2_valbest/student_predictions_train.jsonl --predictions-name S2_valbest \
  --predictions $PHASE/predictions/S2_final/student_predictions_train.jsonl --predictions-name S2_final \
  --predictions $PHASE/predictions/S3_valbest/student_predictions_train.jsonl --predictions-name S3_valbest \
  --predictions $PHASE/predictions/S3_final/student_predictions_train.jsonl --predictions-name S3_final \
  --output-dir $PHASE/audit
```

结果：

```text
Tier A = 134
Tier B = 103
Tier C = 169
remaining = 406
```

---

### Step 7：X3 Manifest

指令：

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256_base

$PY scripts/phase1_build_x3_manifest.py \
  --original $PHASE/pseudo_manifest_original.jsonl \
  --tier-a $PHASE/audit/tier_A.jsonl \
  --tier-b $PHASE/audit/tier_B.jsonl \
  --output $PHASE/pseudo_manifest_x3.jsonl
```

结果：

```text
original = 386
tier_a   = 134
tier_b   = 103
total    = 623
```

---

### Step 8：X3 训练

指令：

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256_base
DATA=work/kvasir_1pct_anchors/baseline_data
LABELS=work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt

$PY scripts/run_s27_student.py \
  --data-path $DATA --labeled-list $LABELS \
  --pseudo-manifest $PHASE/pseudo_manifest_x3.jsonl \
  --output-dir $PHASE/students/X3 --experiment X3 --seed 2026 \
  --batch-size 12 --gt-bs 3 --original-bs 3 --new-bs 6 \
  --max-iterations 40000 --val-interval 200 --num-workers 4
```

结果：

```text
X3 best val Dice = 0.8206 @ iter 16600
X3 final val Dice = 0.8031 @ iter 40000
```

---

### Step 9：导出 X3 预测（best + final）

指令：

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256_base

$PY scripts/export_t25_student_predictions.py \
  --run-dir $PHASE/students/X3 --checkpoint $PHASE/students/X3/student_best.pth \
  --output-root $PHASE/predictions/X3_best

$PY scripts/export_t25_student_predictions.py \
  --run-dir $PHASE/students/X3 --checkpoint $PHASE/students/X3/student_final.pth \
  --output-root $PHASE/predictions/X3_final
```

结果：

```text
X3_best / X3_final 预测均已导出
```

---

### Step 10：B7 最终选择（best + final）

指令：

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256_base
ROOT=$PHASE/stage1_feature_knn_b7_base

$PY scripts/phase1_b7_select.py \
  --quality-root "$ROOT" \
  --student-predictions $PHASE/predictions/X3_best/student_predictions_test.jsonl \
  --output-dir $PHASE/selection_best

$PY scripts/phase1_b7_select.py \
  --quality-root "$ROOT" \
  --student-predictions $PHASE/predictions/X3_final/student_predictions_test.jsonl \
  --output-dir $PHASE/selection_final
```

结果：

| X3 checkpoint | B7 selected Dice | Oracle | gap |
|---|---:|---:|---:|
| X3_best | **0.886130** | 0.903589 | 0.017459 |
| X3_final | 0.875302 | 0.903589 | 0.028287 |

---

## 4. 与其它版本对比

| 版本 | SAM3 | 分辨率 | Router | B7 best | Oracle |
|---|---:|---:|---:|---:|---:|
| 原始 C0 参考 | ft_1pct | 512 | 0.894648 | 0.899369 | 0.919805 |
| C0-256 | ft_1pct | 256 | 0.891288 | 0.897146 | 0.918109 |
| **C0-256-base** | **sam3.pt 未微调** | 256 | 0.871822 | **0.886130** | 0.903589 |

结论：未微调 SAM3 在 256 分辨率下完整跑通，成绩低于 ft_1pct 版本，适合作为“无微调 baseline”。

---

## 5. 产物与日志路径

```text
work/rerun_c0_256_base/
├── stage1_feature_knn_b7_base/        # SAM3 base 256 传播
├── pseudo_manifest_original.jsonl     # 386 伪标签池
├── S3_consensus/                      # consensus
├── router_b3_b6.json                  # router
├── students/S2/
├── students/S3/
├── students/X3/
├── predictions/
│   ├── S2_valbest/
│   ├── S2_final/
│   ├── S3_valbest/
│   ├── S3_final/
│   ├── X3_best/
│   └── X3_final/
├── audit/
├── selection_best/b7_report.json
└── selection_final/b7_report.json
```

日志：

```text
work/rerun_c0/sam3_base_256_gpu0.log
work/rerun_c0_256_base/s2_gpu0.log
work/rerun_c0_256_base/s3_gpu1.log
work/rerun_c0_256_base/x3_gpu0.log
work/rerun_c0_256_base/audit.log
work/rerun_c0_256_base/b7_best.log
work/rerun_c0_256_base/b7_final.log
```

---

## 6. 给后续 Agent 的接力说明

1. 本报告是 **未微调 SAM3 baseline**，不是 ft_1pct 版本。
2. 后续如果要继续做 `SAM3↔U-Net` 协同训练，建议基于本报告中的 `C0-256-base` 作为“无微调起点”。
3. 报告里所有命令均为实际执行过的完整命令，可直接复用。
4. 重要 checkpoint：
   - SAM3：`sam3.pt`
   - U-Net best：`students/S2/student_best.pth`、`students/S3/student_best.pth`
   - X3 best：`students/X3/student_best.pth`
   - X3 final：`students/X3/student_final.pth`
5. 最终成绩以 `X3_best + B7` 为准：**0.886130**。
