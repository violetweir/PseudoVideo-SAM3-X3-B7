# B2 方法说明:SAM3 全量微调(1% + 第一次扩展伪标签池)

> 用途:供组内讨论"如何优化长链(b4-b6)传播效果"时使用。
> 版本:2026-08-08。数据与设置均与项目协议一致,可复现。

## 1. 一句话概括

用 **8 张人类标注(1%)+ 491 张第一轮伪标签** 对 SAM3 做**全量微调**,
优化器采用 MedSAM3 风格(AdamW,统一 lr=5e-5,wd=0.01),
epoch 数从 3 开始(正在扩展 5/10),评估时直接在 test 上跑
b0-b6 逐桥长正向传播 Dice(无路由器)。

## 2. 训练数据

| 组成 | 数量 | 来源 |
|---|---:|---|
| GT anchor | 8 | 固定 1% 人类标注(`KvasirSEG_1pct_seed2026`) |
| 伪标签 | 491 | 第一轮扩展池(`pseudo_manifest_original.jsonl`,ft_1pct 传播质量硬阈值筛选:q_return≥0.95 ∧ q_multi≥0.90) |
| 合计 | 499 | COCO 格式,类别 "colon polyp" |

伪标签 = ft_1pct 沿冻结路线的正向传播 mask(未做学生共识混合、未二值化加权)。

## 3. 模型与训练设置

- 基座:`sam3.pt`(facebook/sam3,视频/图像统一模型),**840M 参数全量微调**
  (detector 部分,tracker 权重保持基座不变);
- 分辨率 1008×1008,bf16 混合精度,batch size 1;
- 优化器:**AdamW,lr=5e-5(transformer / vision backbone / language backbone
  统一),weight_decay=0.01,无层间衰减**——参考 MedSAM3;
- 调度:官方 trainer 的 InverseSquareRoot,warmup/cooldown=20;
- 损失:SAM3 原生 `Sam3LossWrapper`(Hungarian 匹配 + one-to-many):
  - Boxes:loss_bbox=5.0,loss_giou=2.0
  - IABCEMdetr:loss_ce=20.0,presence_loss=20.0,pos_weight=10.0
  - Masks:loss_mask=200.0,loss_dice=10.0
  - o2m_weight=2.0
- epochs:3(已完成),5 / 10(进行中);**每个 epoch 的 checkpoint 都保存**;
- 数值稳定性补丁:NaN loss→0 继续、非有限梯度清零、matcher 有限守卫。

## 4. 评估方式

- test 100 targets,两种路线模式(target pooling / patch correspondence),
  桥长 b0-b6(direct~6 桥),每条路线正向传播一次;
- 指标:逐桥长 mean Dice(@canvas 512,与项目内所有 forward 表同口径);
- **不用路由器、不做选择**,直接报告每个桥长的传播质量上限。

## 5. 当前结果(3 epochs)

### 5.1 target pooling 模式(每桥长 mean Dice)

| variant | direct | b1 | b2 | b3 | b4 | b5 | b6 | 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen 基座 | 0.7413 | 0.7602 | 0.8272 | 0.8408 | 0.8413 | 0.8486 | 0.8546 | 0.8163 |
| ft_1pct(8图,20ep,原配方) | 0.7708 | 0.7922 | 0.8574 | 0.8522 | 0.8609 | 0.8731 | 0.8643 | 0.8387 |
| **B2 ft3ep e3(1%+491)** | **0.8071** | 0.8303 | **0.8646** | 0.8294 | 0.8464 | 0.8524 | 0.8496 | **0.8400** |

### 5.2 patch correspondence 模式

| variant | direct | b1 | b2 | b3 | b4 | b5 | b6 | 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen 基座 | 0.7578 | 0.7735 | 0.8223 | 0.8328 | 0.8340 | 0.8336 | 0.8421 | 0.8137 |
| ft_1pct(8图,20ep,原配方) | 0.7979 | 0.8161 | 0.8578 | 0.8599 | 0.8843 | 0.8658 | 0.8797 | 0.8516 |
| B2 ft3ep e2(1%+491) | 0.7817 | 0.8220 | 0.8425 | 0.8539 | 0.8588 | 0.8639 | 0.8787 | **0.8431** |

### 5.3 观察

1. **短链(direct~b2)明显变强**:B2 的 direct/b1/b2 普遍超过 ft_1pct
   (e3 direct 0.8071 为全场最高);
2. **长链(b3-b6)弱于 ft_1pct**:target 模式 b3-b6 约 0.83-0.85,
   ft_1pct 为 0.85-0.87;patch 模式 b4/b5 差距更明显(0.86 vs 0.88);
3. **3 epochs 内单调上升、未见拐点**:e1→e3 全涨,说明还没到过拟合区;
4. 与 round2(455 共识标签,20ep)的结论一致:**伪标签数据强化短链、
   对长程传播帮助有限**,长链是当前主要瓶颈。

## 6. 待讨论:如何提升长链(b4-b6)传播效果

已知事实/观察(供讨论参考,非结论):

- ft_1pct(只训 8 张 GT)反而是长链最强的模型,说明伪标签数据对长链
  可能有"稀释"效应;
- 第二轮全量微调(20ep)的早期 checkpoint(e1/e6)短链更强、长链随训练
  退化为过拟合曲线;
- 纯 mask 提示(跳过 detector)会伤传播,但 det_gtmask(detector 照跑 +
  tracker 第一帧用 GT mask 播种)稳定 +0.0055;
- LoRA(全模型,MedSAM3 配置)实验正在跑,alpha 缩放可连续控制适应强度。

候选方向:

1. **路线/桥帧选择**:长链质量取决于桥帧与 anchor/target 的相似度,
   是否可按桥帧特征重筛 b4-b6 候选池;
2. **训练数据配方**:伪标签按桥长/质量加权,或长链伪标签降权、
   只保留短链伪标签参与微调;
3. **保真训练**:微调时加入"基座蒸馏"或 LoRA(保留基座长程传播能力),
   而不是全量覆盖;
4. **评估口径**:长链是否需要路由器/B7 选择才能体现价值(单看 forward
   Dice 可能低估)。

## 7. 复现信息

- 配置:`/Data_8TB/lht/sam3/sam3/train/configs/kvasir_budgets/
  kvasir_1pct_plus_pseudo491_ft{3,5,10}ep_medsam3opt_seed2026.yaml`;
- 数据集:`.../dataset/KvasirSEG_1pct_plus_pseudo491_seed2026`(499 张);
- 评估:`scripts/run_ft3ep_lora_test_eval.sh` + `stage1_eval_routes_forward_only.py`。
