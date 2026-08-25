# C0-256 全链路完整实验报告：Base、两轮全模块 LoRA、SAM3-KNN@256、Round-2A 与 Student-X4

> 整理时间：2026-08-25（Asia/Shanghai）  
> 服务器：`violet@222.31.141.50`  
> 项目根目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7`  
> 本报告目标：把上一轮 C0-256 / SAM3-KNN / 两次 LoRA 与本轮 Round-2A / Student-X4 连成一份完整、可复查、可续跑的实验账本。  
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
| Round-2A / Student-X4 | 固定 SAM3-base KNN topology，e33 传播，B7 重校准，524 个新伪标签 | X4 validation-best@26000 | **Student Test Dice 0.860007**，X3 为 0.853920 |
| Round-2A / e33 + X3 + B7 | 固定 e33 两模式 b0-b6 candidates，用 X3-best 重算 `q_model` | X3-best，且 B7 validation 优于 X4-best | **B7 Test Dice 0.906380** |
| Round-2A / e33 + X4 + B7 | 固定同一批 e33 candidates，用 X4-best 重算 `q_model` | X4 student validation-best@26000 | **B7 Test Dice 0.905375** |

注意：固定 b6 的 0.904065 是 propagation combined Dice，**不是 B7 结果**。Round-2A 已完成 e33 validation B7 重校准、524 条 train pseudo pool、X3/X4 student 对比，以及 X3-best/X4-best/X4-final 的完整 B7 validation/test 闭环；X4-final 仅作为诊断，不参与 checkpoint 选择。

### 1.2 最重要的实验判断

1. 第一次 LoRA 的 validation-best 是 e35，而不是 e50。e50 的测试 B7 略高，但它是用户授权的 test audit，不能倒过来用 test 选择 checkpoint。
2. 把 KNN 特征从 DINOv3 ViT-S@224 换成 SAM3 trunk@256 后，base SAM3 的固定桥平均值并非处处提高，但新候选池的 oracle 仍有潜力；重训 S2/S3/X3 后，base B7 test 从 0.886130 提高到 0.895432。
3. 第二次 LoRA 加入了每个 epoch 的普通 Direct Val Dice。e33 是 50 个 epoch 中最高的 Direct Val checkpoint：0.872377。
4. e33 在 test 上显著优于未微调 SAM3：同一 Direct 1008+文本协议从 0.461427 提高到 0.885042。
5. 多帧传播中，e33 在 b3-b6 的提升明显；test combined b6 达 0.904065。但 b0 validation 比 base 低，说明 LoRA 对“只给 anchor box、零中间桥”的传播并非一致改善。
6. 在固定 KNN topology 的 Round-2A 中，只增强 propagation teacher 并重校准 B7 后，Student-X4 best Val Dice 从 0.826032 提高到 0.839122，best Test Dice 从 0.853920 提高到 0.860007。
7. 接回同一批 e33 candidates 后，X4-best+B7 test 为 0.905375，略低于 X3-best+B7 的 0.906380；但两者都高于 base teacher+X3+B7 的 0.895432。因此“student direct 变强”不能直接等价为“B7 route ranking 变强”。

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
work/reproduction_reports/C0_256_full_experiment_history.md
work/reproduction_reports/C0_256_round2a_fixed_knn_e33_x4.md
work/reproduction_reports/C0_256_Round1_Round2A_complete_reproduction.md   # 本合并报告
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

## 12. 截至 Round-2A 的实验状态与后续边界

已经完成：

- C0-256-base 全流程；
- 原 DINO 路线第一次全模块 LoRA e1-e50 和下游 checkpoint 选择；
- SAM3-KNN@256 b0-b6 路线、base propagation、S2/S3/X3 与 B7 test；
- 第二次全模块 LoRA e1-e50，并记录每 epoch 普通 Direct Val Dice；
- e7/e28/e33 Direct 与分辨率/文本补测；
- e33 validation/test 两模式 b0-b6 全传播；
- Round-2A 固定 base KNN topology 的 e33 train propagation：5544 + 5544 条；
- 固定 X3 的 q_return / q_multi / q_model 与 B7 validation 重校准；
- 新阈值 0.9460352822729875、524 个 train pseudo labels；
- Student-X4 40000 iterations，以及 X3/X4 best + final 的 test 对比；
- X3-best、X4-best、X4-final 接回同一批 e33 candidates 的 B7 validation/test 完整评估。

当前结果：

```text
X3 best Val  = 0.826032
X4 best Val  = 0.839122
X3 best Test = 0.853920
X4 best Test = 0.860007
e33 + X3 best + B7 Val/Test = 0.883854 / 0.906380
e33 + X4 best + B7 Val/Test = 0.880070 / 0.905375
e33 + X4 final + B7 Val/Test = 0.880467 / 0.905385（仅诊断）
```

本轮到 B7 为止；以下项目未执行，也不应当提前写结果：

1. 基于 X4+B7 pseudo pool 再训练新一轮 SAM3：按照本轮要求，不再开展；
2. 第二次 LoRA e50 的同协议完整 b0-b6 propagation；
3. 更新 KNN feature extractor / topology 的 Round-2B 类实验；
4. Round-2A 对 medium-size target 退化的专项分析。

下一章完整记录 Round-2A 的控制变量、三套 propagation 明细、B7 frontier、伪标签组成、X4 训练和 X3/X4 逐项比较。

---

# 第二部分：Round-2A 固定 topology 与 Student-X4 完整实验

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
