# C0-256 系列完整实验记录：Base、全模块 LoRA、SAM3-KNN@256 与 b0-b6

> 整理时间：2026-08-24（Asia/Shanghai）  
> 服务器：`violet@222.31.141.50`  
> 项目根目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7`  
> 本报告目标：把本轮对话中实际完成、补测和讨论过的 C0-256 实验统一整理成一份可复查、可续跑的实验账本。  
> 数字来源：服务器上的原始 JSON、TSV、训练日志、checkpoint 和已有复现报告；聊天中的临时口头数字不作为最终依据。

---

## 1. 一页结论

### 1.1 主要结果

| 阶段 | 模型/线路 | checkpoint 选择依据 | 主要测试结果 |
|---|---|---|---:|
| C0-256-base | 未微调 `sam3.pt`，DINOv3 ViT-S@224 KNN，b3-b6，X3+B7 | X3 validation-best | **B7 Test Dice 0.886130** |
| 第一次全模块 LoRA | 494 个 B7 伪标签 + 8 anchors，50 epoch | b3-b6 下游 Val B7，best=e35 | **B7 Test Dice 0.898734** |
| SAM3-KNN@256 base | 未微调 `sam3.pt`，SAM3 trunk@256 KNN，b0-b6，新 S2/S3/X3 | X3 validation-best@28800 | **B7 Test Dice 0.895432** |
| 第二次全模块 LoRA | 486 个 B7 伪标签 + 8 anchors，50 epoch | 普通 Direct Val Dice，best=e33 | **Direct Test Dice 0.885042**（1008+文本） |
| 第二次 LoRA e33 传播 | e33 合并为 video checkpoint，SAM3-KNN b0-b6 | e33 由 Direct Val 冻结 | 固定桥中 **b6 combined Test Dice 0.904065** |

注意：e33 目前完成的是 Direct test 和两种模式的 b0-b6 传播明细，**尚不能把固定 b6 的 0.904065 写成 B7 结果**；它不是 X3+B7 逐目标选择结果。

### 1.2 最重要的实验判断

1. 第一次 LoRA 的 validation-best 是 e35，而不是 e50。e50 的测试 B7 略高，但它是用户授权的 test audit，不能倒过来用 test 选择 checkpoint。
2. 把 KNN 特征从 DINOv3 ViT-S@224 换成 SAM3 trunk@256 后，base SAM3 的固定桥平均值并非处处提高，但新候选池的 oracle 仍有潜力；重训 S2/S3/X3 后，base B7 test 从 0.886130 提高到 0.895432。
3. 第二次 LoRA 加入了每个 epoch 的普通 Direct Val Dice。e33 是 50 个 epoch 中最高的 Direct Val checkpoint：0.872377。
4. e33 在 test 上显著优于未微调 SAM3：同一 Direct 1008+文本协议从 0.461427 提高到 0.885042。
5. 多帧传播中，e33 在 b3-b6 的提升明显；test combined b6 达 0.904065。但 b0 validation 比 base 低，说明 LoRA 对“只给 anchor box、零中间桥”的传播并非一致改善。

---

## 2. 统一术语与评测口径

### 2.1 Direct、B0 和 B1-B6 不是一回事

**Direct image segmentation**：

- 输入一张 target RGB 图像；
- 输入文本类别提示，正常实验为 `colon polyp`；
- 不输入 anchor 图像，不输入桥接帧；
- 模型按图像分割接口输出目标 mask；
- trainer 内普通 Val Dice 使用 class probability `>=0.5`、mask probability `>=0.5`，最后对 foreground queries 做 union。

**B0 / bridge_0**：

- 仍然是多帧 video propagation；
- 帧序列是 `[anchor, target]`，只是中间桥接帧数量为 0；
- frame 0 输入归一化 anchor bounding box 作为 spatial prompt；
- `text_str=None`，后续 target 帧没有额外 prompt；
- 所以 B0 不是 Direct，也不能与单图文本分割结果混写。

**Bn（n=1…6）**：

- 帧序列为 `[anchor, bridge_1, ..., bridge_n, target]`；
- 只有 anchor 帧给 box prompt；
- 中间帧和 target 帧依赖 SAM3 的多帧传播记忆；
- target GT 只用于最终 Dice 评估，不参与推理。

当前 `eval_route_propagation_quality.py` 中，anchor GT mask 不作为前向传播 prompt；它只在 target→anchor 的 return-cycle 质量计算中使用。

### 2.2 target pooling 与 patch correspondence

两种模式改变的是 **KNN 路线构造/候选帧选择方式**，不是两套不同的 SAM3 推理模型：

- `anchor_conditioned_target_pooling` / `sam3enc_anchor_conditioned_target_pooling`
- `anchor_conditioned_patch_correspondence` / `sam3enc_anchor_conditioned_patch_correspondence`

报告中的 `combined` 是同一 bridge 下两种模式各 100 条结果的算术平均（共 200 条），不是 mask ensemble，也不是 B7 选择。

### 2.3 B7 的含义

本轮线路中的 B7 评分为：

```text
B7 = (q_return * q_multi^2 * q_model^2)^0.2
```

- `q_return`：return-cycle/self-consistency；
- `q_multi`：候选之间的一致性；
- `q_model`：与 X3 student prediction 的一致性；
- `b7_mean`：100 个目标上所选候选的平均 B7 置信度，**不是 Dice**；
- `selected_gt_dice_evaluation_only` 或 `selected_dice`：冻结选择后，用 GT 计算的真正分割 Dice；
- `oracle_dice`：每个 target 在候选池中用 GT 事后挑最好候选得到的上限，只用于分析。

### 2.4 1008 与“effective 256” Direct

SAM3 image model 的固定输入分辨率是 1008。直接把模型原生输入改成 256 会触发 RoPE/形状断言。因此本轮补做的 256 Direct 使用：

```text
原图下采样到 256 → 再送入固定 1008 的 image model → 输出与 GT 在 256 上评估
```

报告统一称为 **effective 256**，不能写成“原生 256 SAM3 image input”。传播线路的 `--canvas 256` 则确实是在 256×256 video canvas 上运行。

空文本对照使用 `query_text=""`。它表示空字符串提示；完全不给 query 不会产生同样的语义 query 输出，因此不能把空文本对照解释成绝对无 query 模型。

### 2.5 数据划分与 checkpoint 规则

- Kvasir-SEG：train=800、validation=100、test=100；
- 固定人工 GT anchors：8 个，属于训练侧；
- train GT 不参与伪标签路线选择和阈值选择；
- validation 用于定阈值、选 checkpoint 和定协议；
- test 只在规则/checkpoint 冻结后评估；
- 用户额外要求的 e45/e50、S2/S3 final 等 test 属于诊断性 audit，不应改写 official validation-selection 逻辑。

---

## 3. 实验 A：C0-256-base 完整基线

### 3.1 目的与固定配置

目标是使用未微调的 `sam3.pt`、256×256 propagation canvas，完整复现 C0 下游，作为后续 LoRA 与 KNN 替换实验的共同起点。

| 项目 | 配置 |
|---|---|
| KNN backbone | DINOv3 ViT-S/16 |
| KNN feature input | 224×224 |
| KNN beam width | 32 |
| propagation checkpoint | base `sam3.pt` |
| propagation canvas | 256×256 |
| route modes | target pooling + patch correspondence |
| route range | train b3-b6；validation/test b0-b6 |
| student | SC-SAM `SamUnet`，256×256 |
| S2/S3/X3 horizon | 40000 iterations |

### 3.2 Step A0：生成 DINOv3 KNN 路线

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

每种模式的结果：train 3168、validation 700、test 700；与原 C0 test route 的 `route_id` 100% 一致。

### 3.3 Step A1：base SAM3 传播

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

两种模式的 train/validation/test 均完成。

### 3.4 Step A2：Router 与原始伪标签池

Router（b3-b6 test）结果：

| 候选 | selected Dice | oracle Dice | gap |
|---|---:|---:|---:|
| target + patch | **0.871822** | 0.903589 | 0.031767 |
| target only | 0.865961 | 0.892442 | 0.026481 |
| patch only | 0.853490 | 0.872373 | 0.018883 |

关键命令：

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
ROOT=work/rerun_c0_256_base/stage1_feature_knn_b7_base
PHASE=work/rerun_c0_256_base

$PY scripts/eval_ft1pct_pq_router.py \
  --root "$ROOT" --min-bridge 3 --max-bridge 6 \
  --output $PHASE/router_b3_b6.json

$PY scripts/select_phase1_mainline_pseudo568.py \
  --quality-root "$ROOT" \
  --output $PHASE/pseudo_manifest_original.jsonl

$PY scripts/prepare_phase1_s3_consensus.py \
  --original-manifest $PHASE/pseudo_manifest_original.jsonl \
  --quality-root "$ROOT" --output-root $PHASE/S3_consensus
```

结果：伪标签 386/792，S3 consensus 386；`q_multi` mean/median=0.9832/0.9903，`q_return` mean/median=0.9674/0.9686。

### 3.5 Step A3：S2、S3、Committee、X3

S2/S3 训练命令骨架：

```bash
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256_base
DATA=work/kvasir_1pct_anchors/baseline_data
LABELS=work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt

$PY scripts/run_t24_student.py \
  --data-path $DATA --labeled-list $LABELS \
  --pseudo-manifest $PHASE/pseudo_manifest_original.jsonl \
  --output-dir $PHASE/students/S2 --experiment S2 --seed 2026 \
  --max-iterations 40000 --val-interval 200 --num-workers 4

$PY scripts/run_t24_student.py \
  --data-path $DATA --labeled-list $LABELS \
  --pseudo-manifest $PHASE/S3_consensus/pseudo_consensus.jsonl \
  --output-dir $PHASE/students/S3 --experiment S3 --seed 2026 \
  --max-iterations 40000 --val-interval 200 --num-workers 4
```

| student | best val | best iter | best test | final test |
|---|---:|---:|---:|---:|
| S2 | 0.8006 | 22200 | 0.8310 | 0.8245 |
| S3 | 0.8076 | 15600 | 0.8395 | 0.8264 |

Committee audit：Tier A/B/C=134/103/169。X3 manifest=386 original +134 A +103 B =623。

```bash
$PY scripts/phase1_build_x3_manifest.py \
  --original $PHASE/pseudo_manifest_original.jsonl \
  --tier-a $PHASE/audit/tier_A.jsonl \
  --tier-b $PHASE/audit/tier_B.jsonl \
  --output $PHASE/pseudo_manifest_x3.jsonl

$PY scripts/run_s27_student.py \
  --data-path $DATA --labeled-list $LABELS \
  --pseudo-manifest $PHASE/pseudo_manifest_x3.jsonl \
  --output-dir $PHASE/students/X3 --experiment X3 --seed 2026 \
  --batch-size 12 --gt-bs 3 --original-bs 3 --new-bs 6 \
  --max-iterations 40000 --val-interval 200 --num-workers 4
```

X3 best Val Dice=0.8206@16600；final Val Dice=0.8031@40000。

### 3.6 Step A4：X3+B7 最终基线

```bash
$PY scripts/phase1_b7_select.py \
  --quality-root "$ROOT" \
  --student-predictions $PHASE/predictions/X3_best/student_predictions_test.jsonl \
  --output-dir $PHASE/selection_best
```

| X3 checkpoint | B7 selected Test Dice | oracle | gap |
|---|---:|---:|---:|
| X3 best | **0.886130** | 0.903589 | 0.017459 |
| X3 final | 0.875302 | 0.903589 | 0.028287 |

这就是后续两条实验线共同使用的 C0-256-base 起点。

---

## 4. 实验 B：基于原 C0-256-base 的第一次 SAM3 全模块 LoRA

### 4.1 目的

用户要求不再只训练 20 epoch，而是做 40–50 epoch 的 SAM3“全模块 LoRA”，并以 C0-256-base 最后一步的 X3+B7 规则重新筛选训练集。该实验保持原 DINOv3 路线与 b3-b6 候选不变，只更新传播用 SAM3。

实验根目录：

```text
work/rerun_c0_256_base/medsam3_lora_b7_e50/
```

### 4.2 Step B0：在 validation 上冻结 B7 阈值

固定规则：

- X3 checkpoint：`students/X3/student_best.pth`；
- 路线：target pooling + patch correspondence；
- 候选：b3-b6；
- 每个 target 选最高 B7 的路线 mask；
- 不使用 train GT 决定路线或阈值。

| minimum B7 | validation kept | coverage | selected Val Dice |
|---:|---:|---:|---:|
| 0.90 | 67/100 | 67% | 0.9294 |
| 0.92 | 61/100 | 61% | 0.9464 |
| **0.94** | **48/100** | **48%** | **0.9544** |
| 0.96 | 35/100 | 35% | 0.9577 |

冻结 `B7>=0.94` 后应用到 train，保留 494/792 个伪标签。train GT 只在事后 audit 中得到平均 Dice 0.924585，不参与筛选。

### 4.3 Step B1：构建 LoRA 数据集

| 数据来源 | 数量 |
|---|---:|
| X3-best+B7 伪标签 | 494 |
| 固定人工 anchors | 8 |
| train total | **502** |
| real validation | 100 |

COCO 类别文本为 `colon polyp`；trainer 的 image 分支使用固定 1008×1008，后续 video propagation 仍为 canvas 256。

相关脚本与产物：

```text
scripts/build_c0_256_b7_lora_manifest.py
scripts/prepare_c0_256_b7_medsam3_dataset.py
configs/c0_256_base_b7_medsam3_lora_e50.yaml
work/rerun_c0_256_base/medsam3_lora_b7_e50/manifests/
work/rerun_c0_256_base/medsam3_lora_b7_e50/data/
```

### 4.4 Step B2：全模块 LoRA 配置与启动

| 项目 | 值 |
|---|---|
| base | untouched `sam3.pt` |
| LoRA modules | 383 |
| trainable params | 17,883,264（2.08%） |
| rank / alpha / dropout | 16 / 32 / 0.1 |
| 覆盖模块 | vision/text/geometry encoder、DETR encoder/decoder、mask decoder |
| optimizer | AdamW，lr=5e-5，weight decay=0.01 |
| batch / precision | 1 / FP32 |
| seed | 2026 |
| horizon | 50 epochs |
| save | 每 epoch、best、last |

启动命令：

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
nohup env DEVICE=0 bash scripts/supervise_c0_256_b7_lora_e50.sh \
  > work/rerun_c0_256_base/medsam3_lora_b7_e50/nohup_supervisor.log \
  2>&1 < /dev/null &
```

supervisor 会从 `epoch_N_lora_weights.pt` 自动续跑，直到 epoch 50。

### 4.5 Step B3：第一次验证失败与修复

最初 epoch 1 训练完成后，在第一张 validation 图上报错：

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

原因：`torch.no_grad()` 下 SAM3 fused ViT MLP 的 `fc1` activation 进入 BF16，而 LoRA 包装后的 `fc2` base Linear 保持 FP32。历史训练没有 validation split，因此此前没有触发这条 inference-only 路径。

修复：

1. validation forward 放入 CUDA BF16 autocast；
2. 每 epoch 先保存 `last` 和 `epoch_N`，再进入 validation，防止验证失败丢失训练成果；
3. 完成一张图的 forward+matcher+loss smoke test：`VALIDATION_SMOKE_OK loss=96.692200 dtype=torch.float32`；
4. 保留 pre-fix 日志，干净重启时间为 2026-08-22 00:10。

### 4.6 Step B4：GPU1 做下游 validation checkpoint 选择

检查节点：e1-e12、e15、e20、e25、e30、e35、e40、e45、e50。每个节点执行：

1. 把 LoRA delta 合并为完整 SAM3 video checkpoint；
2. 用原两种 DINO 路线跑 validation b3-b6、canvas 256；
3. 用冻结的 X3-best+B7 计算 100 个 validation target 的 selected Dice；
4. 只根据 validation 排名 checkpoint。

运行入口：

```bash
nohup env DEVICE=1 bash scripts/supervise_c0_256_b7_lora_checkpoint_queue.sh \
  > work/rerun_c0_256_base/medsam3_lora_b7_e50/checkpoint_eval_supervisor.log \
  2>&1 < /dev/null &
```

完整节点结果：

| epoch | downstream B7 Val Dice | train loss | val loss |
|---:|---:|---:|---:|
| 1 | 0.771965 | 69.0699 | 8.9098 |
| 2 | 0.759370 | 57.8204 | 10.5085 |
| 3 | 0.846514 | 49.5724 | 7.7545 |
| 4 | 0.865217 | 47.7707 | 7.6430 |
| 5 | 0.859723 | 44.4413 | 8.2535 |
| 6 | 0.851725 | 42.9907 | 9.8856 |
| 7 | 0.871766 | 40.0015 | 9.9334 |
| 8 | 0.877442 | 38.4268 | 10.2657 |
| 9 | 0.869653 | 36.8551 | 8.6597 |
| 10 | 0.872716 | 35.5243 | 10.1779 |
| 11 | 0.866840 | 32.9592 | 10.6352 |
| 12 | 0.869509 | 31.3968 | 10.1480 |
| 15 | 0.859942 | 29.5458 | 9.9518 |
| 20 | 0.878630 | 25.6974 | 11.7627 |
| 25 | 0.880898 | 23.3764 | 13.1840 |
| 30 | 0.867376 | 22.2597 | 13.0949 |
| **35** | **0.881798** | 21.4855 | 19.4937 |
| 40 | 0.868232 | 20.9568 | 16.9408 |
| 45 | 0.834851 | 20.0486 | 20.7540 |
| 50 | 0.860540 | 17.5839 | 20.3148 |

训练 loss 持续下降不等于下游 Dice 持续提高；e35 是 validation-best，e45 明显退化，e50 有回升但仍不如 e35。

### 4.7 Step B5：冻结后 test 与 e45/e50 audit

| checkpoint | 选择身份 | B7 Test Dice | oracle | gap |
|---|---|---:|---:|---:|
| **e35** | **official validation-best** | **0.898734** | 0.904141 | 0.005407 |
| e45 | test diagnostic | 0.897579 | 0.902172 | 0.004593 |
| e50 | test diagnostic | **0.900173** | 0.906657 | 0.006485 |

e50 的测试数字比 e35 高 0.001439，但不能根据这个差异把 official checkpoint 改成 e50。正式结论仍是 e35=0.898734；e45/e50 保留为用户要求的补测。

---

## 5. 实验 C：把 KNN 特征网络替换为 SAM3 trunk@256，并保存 b0-b6

### 5.1 目的与控制变量

用户要求以 C0-256-base 为基础，只把 KNN 特征提取网络从 DINOv3 ViT-S@224 改成 frozen base SAM3 image trunk@256，并把 train/validation/test 全部扩成 b0-b6。

| component | C0-256-base | SAM3-KNN@256 |
|---|---|---|
| KNN backbone | DINOv3 ViT-S/16 | base SAM3 image trunk |
| feature input | 224×224 | 256×256 |
| descriptor | L2 patch mean | L2 patch mean |
| descriptor width | 384 | 1024 |
| anchors | 8 | 同一组 8 anchors |
| beam width | 32 | 32 |
| propagation checkpoint | base `sam3.pt` | 不变 |
| propagation canvas | 256 | 256 |
| saved range | train b3-b6；val/test b0-b6 | **all splits b0-b6** |

SAM3 feature cache：

```text
work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/features/sam3_base_s256_features.npz
SHA256=ab8ca194f9c789ad9f6b641d382ed848751f29b23ac0bdb29a9c6722fd599907
```

每种模式预期路线数：train 5544、validation 700、test 700。

### 5.2 Step C0：生成路线和 validation-first 传播

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

bash scripts/run_c0_256_sam3knn_s256_routes.sh
bash scripts/supervise_c0_256_sam3knn_s256_validation.sh
```

隔离实验根目录：

```text
work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6/
```

为了兼容冻结的下游脚本，只在该隔离目录内部建立旧 mode name alias，不覆盖 C0-256-base 原产物。

### 5.3 Step C1：base SAM3 validation 固定桥结果

以下是两种模式 combined 的平均 Dice：

| bridge | DINOv3@224 | SAM3 trunk@256 | delta |
|---|---:|---:|---:|
| b0 | 0.745905 | 0.762231 | +0.016326 |
| b1 | 0.778472 | 0.795550 | +0.017078 |
| b2 | 0.817718 | 0.802799 | -0.014919 |
| b3 | 0.811044 | 0.793372 | -0.017673 |
| b4 | 0.833340 | 0.788871 | -0.044469 |
| b5 | 0.825969 | 0.813166 | -0.012803 |
| b6 | 0.828536 | 0.816890 | -0.011646 |

SAM3 target-pooling 在 b6 为 0.842460，而 patch-correspondence b6 为 0.791321。虽然固定桥均值并不占优，但两种模式联合 oracle 为 0.889432，略高于 DINO 的 0.889047，说明主要瓶颈是候选选择/校准，而不是候选池完全失效。

五折 target-level Router：

| range | DINO selected | SAM3 selected | DINO oracle | SAM3 oracle |
|---|---:|---:|---:|---:|
| b0-b6 | 0.847886 | 0.841517 | 0.889047 | 0.889432 |
| b3-b6 | 0.831619 | 0.836823 | 0.868303 | 0.878693 |

### 5.4 Step C2：train 传播与新伪标签池

```bash
bash scripts/supervise_c0_256_sam3knn_s256_train_and_pseudo.sh
```

结果：

- 两种模式各完成 5544 条 train 路线，共 11088；
- b0-b6 manifest 接受 410/792；
- b3-b6 对照 manifest 接受 389/792；
- b0-b6 S3 consensus 包含 410 个 target。

扩成 b0-b6 后，`q_multi` 从对 7 个其它候选求一致性变成对 13 个候选求一致性；Router、S3 consensus、committee variance 和 B7 分布都必须重算，不能直接沿用旧 X3/旧阈值作为 official selector。

### 5.5 Step C3：重训 S2/S3

```bash
bash scripts/run_c0_256_sam3knn_s256_students.sh
bash scripts/run_c0_256_sam3knn_s256_student_exports.sh
```

每 200 iter 做普通 validation，训练 40000 iterations。用户授权对 best 与 final 四个 checkpoint 做 test audit：

| checkpoint | Val Dice | iteration | Test Dice | Test IoU |
|---|---:|---:|---:|---:|
| S2 best | **0.820152** | 32400 | **0.855042** | 0.771485 |
| S2 final | 0.809828 | 40000 | 0.852245 | 0.767905 |
| S3 best | **0.816272** | 20200 | 0.845245 | 0.763529 |
| S3 final | 0.812164 | 40000 | **0.862539** | **0.786166** |

这里同时报告 best/final 是 audit；正式 checkpoint policy 仍按 validation-best。此前“只有 final”是阶段性导出/测试未齐，并不是实验设计取消了 best。

### 5.6 Step C4：Committee 与 X3

```bash
bash scripts/run_c0_256_sam3knn_s256_audit_x3.sh
```

- Tier A/B/C=94/128/160；
- X3 manifest=410 original +94 A +128 B =632；
- X3 best Val Dice=0.826032@28800；
- X3 final Val Dice=0.814216@40000。

### 5.7 Step C5：validation 上冻结新的 B7 伪标签规则

| selector | min B7 | kept | coverage | selected Val Dice |
|---|---:|---:|---:|---:|
| X3-best, b0-b6 | 0.90 | 64/100 | 64% | 0.946575 |
| X3-best, b0-b6 | **0.94** | **54/100** | **54%** | **0.950447** |
| X3-best, b0-b6 | 0.96 | 43/100 | 43% | 0.953728 |
| X3-final, b0-b6 | 0.94 | 52/100 | 52% | 0.950145 |
| X3-best, b3-b6 control | 0.94 | 58/100 | 58% | 0.939349 |

“validation 保留 54/100，selected Dice 0.950447”的含义是：在 100 张 validation 图上，用 B7>=0.94 做**阈值校准/覆盖率评估**，54 张达到阈值，它们冻结选择后的 GT Dice 平均为 0.950447。validation 图像没有被改动，也没有加入训练集；它只用于确定阈值。

冻结规则：X3 validation-best + 两种 SAM3-KNN 模式 + b0-b6 + B7>=0.94。应用到 train 后保留 486/792；加 8 anchors，第二轮 LoRA train=494，validation=100。

同一普通 Direct 协议下，未微调 `sam3.pt` 在 validation 的 Dice=0.332041、median=0、IoU=0.309719、nonempty rate=0.46。相比之下，base propagation 的 b0 validation 为 target 0.773972、patch 0.750490、combined 0.762231。B0 明显高于 Direct 并不反常：B0 额外获得了一张有 box prompt 的 anchor 图像和 video memory，而 Direct 只有 target RGB 与类别文本。

### 5.8 Step C6：未微调 SAM3 当前线路的完整 test

用户要求 GPU1 暂停复杂 checkpoint queue，先把当前 base SAM3 的 B7 和每一步 test 明细跑全。候选数：2 modes × 7 bridges × 100 targets =1400，无缺失。

| result | Test Dice | IoU / oracle | gap |
|---|---:|---:|---:|
| X3 best direct student prediction | 0.853920 | IoU 0.771766 | - |
| **X3 best + B7, b0-b6** | **0.895432** | oracle 0.921471 | 0.026039 |
| X3 final direct student prediction | 0.858591 | IoU 0.778503 | - |
| X3 final + B7（audit） | 0.897684 | oracle 0.921471 | 0.023786 |

与原 C0-256-base B7 对比：X3 best 从 0.886130 提高到 0.895432（+0.009302）；X3 final 从 0.875302 提高到 0.897684（+0.022382）。

base SAM3 每一步 test 结果：

| bridge | target Dice | patch Dice | combined Dice |
|---|---:|---:|---:|
| b0 | 0.798144 | 0.706645 | 0.752395 |
| b1 | 0.846623 | 0.754897 | 0.800760 |
| b2 | 0.870032 | 0.788470 | 0.829251 |
| b3 | 0.862253 | **0.859413** | 0.860833 |
| b4 | 0.873930 | 0.857625 | **0.865778** |
| b5 | 0.869552 | 0.855339 | 0.862446 |
| b6 | **0.874004** | 0.853581 | 0.863793 |

B7 best 的 bridge 选择分布为 b0/b1/b2/b3/b4/b5/b6 =6/10/13/16/17/21/17；mode 分布为 target 68、patch 32。patch 被选候选的平均 B7 更高，但真实 Dice 更低，说明 patch 候选仍存在过度自信的校准问题。

完整明细见：

```text
work/reproduction_reports/C0_256_sam3knn_s256_b0_b6_test_details.md
work/rerun_c0_256_sam3knn_s256_base/current_base_test/
```

---

## 6. 实验 D：SAM3-KNN@256 b0-b6 的第二次全模块 LoRA（50 epochs）

### 6.1 数据与配置

实验根目录：

```text
work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/
```

| 项目 | 值 |
|---|---|
| pseudo labels | 486（X3-best+B7，b0-b6，B7>=0.94） |
| human anchors | 8 |
| train / validation | 494 / 100 |
| base | `sam3.pt` |
| LoRA | r16 / alpha32 / dropout0.1，全模块 |
| optimizer | AdamW，lr5e-5，wd0.01 |
| batch / workers | 1 / 2 |
| seed / epochs | 2026 / 50 |
| image training resolution | 1008 |
| video propagation canvas | 256 |

LoRA targets：`q_proj,k_proj,v_proj,out_proj,qkv,proj,fc1,fc2,c_fc,c_proj,linear1,linear2`，并应用到 vision、text、geometry、DETR encoder/decoder 和 mask decoder。

### 6.2 本轮相对第一次 LoRA 的关键修改

1. trainer 每个 epoch 都报告 `train_loss`、`val_loss` 和普通 Direct `val_dice`；
2. 普通 Val Dice 协议明确为 class>=0.5、mask>=0.5、foreground query union；
3. 每 epoch 保存 LoRA，另存 loss-best 与 direct-Dice-best；
4. checkpoint 先依据 validation 选择，再做 test；
5. 后续传播保留完整 b0-b6 和两种 mode 明细，不只报告 B7 一个数。

启动：

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
nohup env DEVICE=0 bash scripts/supervise_c0_256_sam3knn_s256_b0_b6_lora_e50.sh \
  > work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/nohup_supervisor.log \
  2>&1 < /dev/null &
```

训练已完成 e1-e50，`TRAIN_COMPLETE` 和 `epoch_50_lora_weights.pt` 均存在。

### 6.3 50 个 epoch 的完整普通 validation 记录

| ep | train loss | val loss | Direct Val Dice |
|---:|---:|---:|---:|
| 1 | 70.1571 | 8.4483 | 0.000000 |
| 2 | 60.2944 | 8.7172 | 0.000000 |
| 3 | 51.7448 | 8.9175 | 0.165960 |
| 4 | 48.7477 | 8.3145 | 0.728956 |
| 5 | 46.7318 | 7.0293 | 0.838772 |
| 6 | 44.0726 | 7.2390 | 0.836588 |
| 7 | 40.2390 | 6.8416 | 0.852736 |
| 8 | 37.3492 | 7.4678 | 0.845742 |
| 9 | 35.9089 | 7.5125 | 0.848325 |
| 10 | 34.6913 | 12.4047 | 0.839023 |
| 11 | 33.1688 | 10.8412 | 0.862697 |
| 12 | 32.3088 | 13.4496 | 0.835038 |
| 13 | 39.6198 | 7.2198 | 0.851145 |
| 14 | 31.4167 | 7.1011 | 0.847847 |
| 15 | 28.8357 | 7.9298 | 0.843929 |
| 16 | 28.0116 | 8.1124 | 0.839699 |
| 17 | 27.3873 | 7.0280 | 0.851271 |
| 18 | 26.8721 | 6.8327 | 0.851102 |
| 19 | 26.4657 | 7.2276 | 0.802214 |
| **20** | 28.0391 | **6.7681** | 0.835678 |
| 21 | 27.2330 | 6.7751 | 0.847744 |
| 22 | 25.9282 | 6.9614 | 0.812447 |
| 23 | 25.9459 | 7.3348 | 0.832929 |
| 24 | 25.5841 | 8.3774 | 0.837793 |
| 25 | 24.9683 | 8.9012 | 0.852208 |
| 26 | 24.3857 | 8.2377 | 0.836157 |
| 27 | 25.2601 | 8.4031 | 0.854600 |
| 28 | 23.9032 | 11.7636 | 0.841289 |
| 29 | 23.2597 | 8.6196 | 0.846814 |
| 30 | 21.9555 | 9.0984 | 0.858964 |
| 31 | 23.2797 | 9.8478 | 0.863854 |
| 32 | 22.2383 | 10.2217 | 0.863046 |
| **33** | 22.4667 | 10.9355 | **0.872377** |
| 34 | 22.4273 | 12.9370 | 0.834451 |
| 35 | 22.2238 | 11.1181 | 0.838269 |
| 36 | 22.1279 | 11.2962 | 0.863752 |
| 37 | 21.5911 | 9.9230 | 0.849289 |
| 38 | 21.2845 | 15.8075 | 0.854739 |
| 39 | 22.0876 | 9.9394 | 0.865638 |
| 40 | 20.3062 | 8.6438 | 0.856901 |
| 41 | 20.0605 | 9.2783 | 0.853330 |
| 42 | 19.8193 | 11.7178 | 0.842744 |
| 43 | 20.0179 | 13.4335 | 0.845954 |
| 44 | 19.9871 | 12.1815 | 0.830792 |
| 45 | 19.5768 | 13.4534 | 0.860808 |
| 46 | 19.0673 | 12.9153 | 0.848785 |
| 47 | 18.7502 | 13.9727 | 0.862174 |
| 48 | 19.2594 | 18.2109 | 0.848988 |
| 49 | 18.4234 | 14.1652 | 0.842374 |
| 50 | 18.4642 | 18.3787 | 0.866548 |

结论：loss-best=e20，但 Direct Dice-best=e33；两者不一致。因此后续语义分割/传播主评测使用 e33，更符合用户要求的“每 epoch 报普通 val Dice 并据此选 checkpoint”。

---

## 7. 实验 E：第二次 LoRA 的 e7、e28、e33 补测

### 7.1 Direct evaluator 的统一调用

```bash
PY=/home/violet/anaconda3/envs/sam3/bin/python

$PY scripts/eval_sam3_lora_direct_split.py \
  --config configs/c0_256_sam3knn_s256_b0_b6_medsam3_lora_e50.yaml \
  --lora <epoch_N_lora_weights.pt> --split test \
  --effective-resolution 1008 --prompt-mode category \
  --output <result.json>
```

effective256 或空文本只替换：

```text
--effective-resolution 256
--prompt-mode empty
```

### 7.2 e7：早期 checkpoint

Direct test：

| protocol | mean Dice | median Dice | mean IoU | nonempty |
|---|---:|---:|---:|---:|
| 1008 + text | 0.879077 | 0.949714 | 0.817216 | 1.00 |
| effective256 + text | **0.889406** | 0.949183 | **0.827612** | 1.00 |
| effective256 + empty text | 0.838203 | 0.943853 | 0.776492 | 0.95 |

e7 validation propagation：

| bridge | patch Dice | target Dice | combined Dice |
|---|---:|---:|---:|
| b0 | 0.836778 | 0.788416 | 0.812597 |
| b1 | 0.841059 | 0.830642 | 0.835850 |
| b2 | 0.847849 | 0.814195 | 0.831022 |
| b3 | 0.858265 | 0.850019 | 0.854142 |
| b4 | 0.857392 | 0.831267 | 0.844330 |
| b5 | 0.856883 | 0.855364 | 0.856124 |
| b6 | 0.858711 | **0.864537** | **0.861624** |

执行脚本：

```bash
bash scripts/run_c0_256_sam3knn_s256_lora_e07_val_combined_gpu1.sh
```

e7 没有跑完整 test b0-b6，只跑了 Direct test 与 validation propagation，报告时不能把二者混成同一 split。

### 7.3 e28：中后期 Direct 补测

| protocol | Test Dice | median | IoU | nonempty |
|---|---:|---:|---:|---:|
| 1008 + text | **0.879747** | 0.950777 | 0.820203 | 0.99 |
| effective256 + text | 0.877286 | 0.949109 | 0.816306 | 0.99 |

e28 普通 Direct Val Dice=0.841289。它的 test 与 e7 接近，但 validation 明显低于 e33，所以没有被选为 official best。

### 7.4 e33：validation-best 的完整 Direct test

| protocol | e33 Test Dice | median | IoU | nonempty |
|---|---:|---:|---:|---:|
| **1008 + text** | **0.885042** | 0.950500 | 0.825374 | 1.00 |
| effective256 + text | 0.882267 | 0.949426 | 0.822065 | 0.99 |
| effective256 + empty text | 0.866378 | 0.950402 | 0.809694 | 0.97 |

文本提示在 effective256 上带来 +0.015889 Dice，说明类别文本总体有帮助，但不是全部性能来源。

### 7.5 e33 validation/test 完整 b0-b6 传播

执行：

```bash
bash scripts/run_c0_256_sam3knn_s256_lora_e33_full_eval_gpu0.sh
```

脚本依次执行：三种 Direct test、LoRA→video checkpoint 合并、validation/test 两种 mode 的 b0-b6 传播，以及 combined 汇总。每个 split 每种 mode 700 条，共完成 2800 条传播，无 OOM、无 traceback。

**Validation：**

| bridge | patch Dice | target Dice | combined Dice |
|---|---:|---:|---:|
| b0 | 0.757399 | 0.716743 | 0.737071 |
| b1 | 0.828144 | 0.818560 | 0.823352 |
| b2 | 0.851841 | 0.846189 | 0.849015 |
| b3 | 0.868521 | 0.881616 | 0.875069 |
| b4 | 0.867411 | 0.869286 | 0.868348 |
| b5 | **0.883561** | 0.882076 | 0.882818 |
| b6 | 0.881546 | **0.887607** | **0.884577** |
| b0-b6 mean | 0.848346 | 0.843154 | 0.845750 |

**Test：**

| bridge | patch Dice | target Dice | combined Dice |
|---|---:|---:|---:|
| b0 | 0.828143 | 0.796551 | 0.812347 |
| b1 | 0.840549 | 0.809615 | 0.825082 |
| b2 | 0.856388 | 0.875972 | 0.866180 |
| b3 | 0.878680 | 0.891441 | 0.885061 |
| b4 | 0.885240 | **0.904009** | 0.894624 |
| b5 | 0.893036 | 0.897443 | 0.895240 |
| b6 | **0.908151** | 0.899980 | **0.904065** |
| b0-b6 mean | 0.870027 | 0.867859 | 0.868943 |

### 7.6 e33 与未微调 SAM3 的 Direct 对比

同一 evaluator、同一 test split：

| protocol | base SAM3 | e33 LoRA | delta |
|---|---:|---:|---:|
| 1008 + text | 0.461427 | **0.885042** | **+0.423614** |
| effective256 + text | 0.380221 | **0.882267** | **+0.502047** |
| effective256 + empty text | 0.311159 | **0.866378** | **+0.555219** |

base 1008+text 的 median=0.263805、IoU=0.432159、nonempty=0.61；这与早先 base validation Direct Dice=0.332041 是不同 split，不矛盾。

### 7.7 e33 与 base SAM3 的传播对比

**Validation combined：**

| bridge | base | e33 | delta |
|---|---:|---:|---:|
| b0 | 0.762231 | 0.737071 | -0.025160 |
| b1 | 0.795550 | 0.823352 | +0.027802 |
| b2 | 0.802799 | 0.849015 | +0.046216 |
| b3 | 0.793372 | 0.875069 | +0.081697 |
| b4 | 0.788871 | 0.868348 | +0.079477 |
| b5 | 0.813166 | 0.882818 | +0.069653 |
| b6 | 0.816890 | 0.884577 | +0.067686 |
| mean | 0.796126 | 0.845750 | +0.049625 |

**Test combined：**

| bridge | base | e33 | delta |
|---|---:|---:|---:|
| b0 | 0.752395 | 0.812347 | +0.059952 |
| b1 | 0.800760 | 0.825082 | +0.024322 |
| b2 | 0.829251 | 0.866180 | +0.036929 |
| b3 | 0.860833 | 0.885061 | +0.024228 |
| b4 | 0.865778 | 0.894624 | +0.028846 |
| b5 | 0.862446 | 0.895240 | +0.032794 |
| b6 | 0.863793 | 0.904065 | +0.040272 |
| mean | 0.833608 | 0.868943 | +0.035335 |

e33 的核心收益集中在带桥传播（特别是 b3-b6）；validation b0 是唯一明确下降项。它提示后续 selector 不应默认“桥越少越稳”，仍需在 validation 上重新校准各 bridge/mode 的 B7 置信度。

---

## 8. 运行过程中的 GPU 调度与异常记录

1. 第一次 LoRA 在 GPU0 训练，GPU1 并行做 checkpoint validation propagation；这是为了不阻塞 50 epoch 训练。
2. 用户提出“训练时为什么不报普通 val Dice”后，第二轮 trainer 已加入每 epoch 普通 Direct Val Dice；第一次 LoRA 的旧 checkpoint 表只能报告 downstream B7 Val Dice 和 loss，不能事后伪造普通 Direct Val Dice。
3. GPU1 的复杂 validation queue 曾按用户要求停止，改为优先完成 base 当前线路的 test B7 和 b0-b6 详细结果。
4. 后期 GPU1 被 `/Data_8TB/zhuangxinjian/RT-DETR-main` 的其它任务占用，因此 e33 完整评测在 GPU0 训练结束后执行；不影响结果协议。
5. 多次看到 GPU 功率为 0，实际原因包括等待 checkpoint、前一阶段结束后 supervisor 没有下一任务、或 GPU1 被其它用户进程占用。后续应同时检查 `nvidia-smi`、进程、日志尾部和 COMPLETE/FAILED 标记，不能只根据瞬时功率判断任务是否卡死。

---

## 9. 结果使用边界与容易误报的点

1. `b7_mean≈0.90` 不是模型 Dice；真正 Dice 看 `selected_dice`/`selected_gt_dice_evaluation_only`。
2. B0 是 anchor→target 两帧传播，不是文本 Direct。
3. `combined` 是两种 route mode 的结果均值，不是逐 target 选择，也不是融合 mask。
4. base SAM3 Direct 很低，不代表 base SAM3 propagation 也低；后者有 anchor box 和跨帧记忆，输入信息更多。
5. effective256 不是原生 256 image model，只是受控分辨率消融。
6. e50 第一次 LoRA 的 test B7 最高，但 official 仍是 validation-best e35。
7. S3 final test 高于 S3 best，不代表应用 test 选择 final；官方 policy 仍按 validation best。
8. e33 固定 b6 test combined=0.904065 不能写成 e33 B7=0.904065。要得到 e33 B7，必须重新生成 e33 candidates 的 B7 输入并运行冻结/重新校准后的 selector。
9. validation 的 54/100 是阈值 coverage，不是删掉 validation 数据，更不是把 validation mask 当训练伪标签。
10. train propagation 的目的只是为 792 个非 anchor train target 生成路线 mask，进而构建伪标签；它不是为了报告 train 性能，也不是用 train GT 调参。

---

## 10. 关键文件与产物索引

### 10.1 已有报告

```text
work/reproduction_reports/C0_256_base_reproduction.md
work/reproduction_reports/C0_256_base_B7_LoRA_e50_run.md
work/reproduction_reports/C0_256_sam3knn_s256_b0_b6_run.md
work/reproduction_reports/C0_256_sam3knn_s256_b0_b6_test_details.md
work/reproduction_reports/C0_256_full_experiment_history.md   # 本报告
```

### 10.2 C0-256-base

```text
work/rerun_c0_256_base/stage1_feature_knn_b7_base/
work/rerun_c0_256_base/pseudo_manifest_original.jsonl
work/rerun_c0_256_base/S3_consensus/
work/rerun_c0_256_base/students/S2/
work/rerun_c0_256_base/students/S3/
work/rerun_c0_256_base/students/X3/
work/rerun_c0_256_base/predictions/
work/rerun_c0_256_base/audit/
work/rerun_c0_256_base/selection_best/b7_report.json
work/rerun_c0_256_base/selection_final/b7_report.json
```

### 10.3 第一次 LoRA

```text
work/rerun_c0_256_base/medsam3_lora_b7_e50/
├── manifests/
├── data/
├── lora_weights/epoch_N_lora_weights.pt
├── lora_weights/val_stats.json
├── checkpoint_eval/validation_checkpoint_summary.{json,tsv}
├── checkpoint_eval/best_validation_epoch.txt
├── checkpoint_eval/test_validation_best_b7/b7_report.json
├── checkpoint_eval/test_e45_b7/b7_report.json
├── checkpoint_eval/test_e50_b7/b7_report.json
├── train.log
└── supervisor.log
```

### 10.4 SAM3-KNN@256 和第二次 LoRA

```text
work/rerun_c0_256_sam3knn_s256_base/
├── stage1_feature_knn_b0_b6/
├── current_base_test/
├── students/S2/
├── students/S3/
├── students/X3/
└── medsam3_lora_b0_b6_e50/
    ├── data/
    ├── lora_weights/epoch_N_lora_weights.pt
    ├── lora_weights/val_stats.json
    ├── direct_test_epoch_7*.json
    ├── direct_test_epoch_28*.json
    ├── current_best_val_sam3knn_e07/
    └── e33_full_evaluation/
        ├── direct_test_r1008_text.json
        ├── direct_test_effective_r256_text.json
        ├── direct_test_effective_r256_empty_text.json
        ├── base_direct_test_*.json
        ├── e33_merged_video.pt
        ├── two_mode_b0_b6_validation.{json,tsv}
        ├── two_mode_b0_b6_test.{json,tsv}
        ├── quality_root/
        ├── COMPLETE
        └── run.log
```

### 10.5 核心配置与执行脚本

```text
configs/c0_256_base_b7_medsam3_lora_e50.yaml
configs/c0_256_sam3knn_s256_b0_b6_medsam3_lora_e50.yaml
scripts/train_sam3_lora_kvasir_e50.py
scripts/eval_sam3_lora_direct_split.py
scripts/eval_route_propagation_quality.py
scripts/merge_sam3_lora_video_checkpoint.py
scripts/summarize_c0_256_bridge_metrics.py
scripts/run_c0_256_sam3knn_s256_routes.sh
scripts/supervise_c0_256_sam3knn_s256_validation.sh
scripts/supervise_c0_256_sam3knn_s256_train_and_pseudo.sh
scripts/run_c0_256_sam3knn_s256_students.sh
scripts/run_c0_256_sam3knn_s256_audit_x3.sh
scripts/supervise_c0_256_sam3knn_s256_b0_b6_lora_e50.sh
scripts/run_c0_256_sam3knn_s256_lora_e07_val_combined_gpu1.sh
scripts/run_c0_256_sam3knn_s256_lora_e33_full_eval_gpu0.sh
```

---

## 11. 如何复查当前最重要的数字

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

# 第一次 LoRA 的 validation checkpoint 排名
cat work/rerun_c0_256_base/medsam3_lora_b7_e50/checkpoint_eval/validation_checkpoint_summary.tsv

# 第一次 LoRA official e35 test B7
cat work/rerun_c0_256_base/medsam3_lora_b7_e50/checkpoint_eval/test_validation_best_b7/b7_report.json

# 第二次 LoRA 50 epoch 的普通 validation 日志
cat work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/lora_weights/val_stats.json

# e33 Direct test
cat work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/e33_full_evaluation/direct_test_r1008_text.json

# e33 validation/test b0-b6
cat work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/e33_full_evaluation/two_mode_b0_b6_validation.json
cat work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/e33_full_evaluation/two_mode_b0_b6_test.json
```

---

## 12. 当前实验状态与合理的下一步

已经完成：

- C0-256-base 全流程；
- 原线路第一次全模块 LoRA e1-e50 和下游 checkpoint 选择；
- SAM3-KNN@256 b0-b6 路线、base propagation、S2/S3/X3、B7 test；
- 第二次全模块 LoRA e1-e50，每 epoch 普通 Val Dice；
- e7/e28 Direct 补测；
- e33 Direct 三协议与 validation/test 两模式 b0-b6 全传播；
- base SAM3 与 e33 的同协议 Direct 对比。

尚未完成、不能提前写结果：

1. e33 在新传播候选上的最终 X3+B7 selected validation/test；
2. 如果 B7 分布因 LoRA 改变，需要先在 validation 上检查/冻结阈值或校准，再做一次 official test；
3. e50 第二轮 LoRA 的同协议完整传播没有跑，因此当前不能比较 e33 与 e50 的 b0-b6；
4. 若要把第二轮 LoRA 定为最终线路，建议至少补 e33 的 B7 selection，并保留固定桥明细和 oracle gap。

建议下一条标准命令链应遵循：

```text
e33 merged candidates（validation）
→ 用 validation 检查 B7/q_return/q_multi/q_model 分布
→ 冻结 selector/阈值
→ 输出 validation selected Dice、coverage、bridge/mode 分布
→ 只按冻结规则运行 test B7
→ 与 base SAM3-KNN B7=0.895432、第一次 LoRA e35 B7=0.898734 对比
```

最终对外汇报时，应并列给出：Direct Dice、b0-b6 target/patch/combined、B7 selected Dice、oracle 和 gap；这样既能看单图语义能力，也能看传播能力与 selector 校准能力。
