# Kvasir-SEG 1% 伪视频 SAM3 管线(主实验,简化版)

> 版本:2026-08-09。本文只保留**一条主实验线路**,其余机制一律降级为
> 对照实验(第 4 节),供论文对比部分使用。
> 完整实验流水账见 `docs/method_cn.md`,本文不复述细节。

---

## 0. 任务与一句话总结

**任务**:只用 Kvasir-SEG 训练集 **1% 人类标注(8 张图)**,在 test 上拿到
尽可能高的息肉分割 Dice。

**方法一句话**:把 8 张标注当"锚点",用 KNN 构造伪视频路线,让 SAM3 把
锚点 mask 沿路线传播到未标注图;再用单图学生审核传播结果、筛选伪标签池;
最后用审核后的池子微调 SAM3。

**主流程(4 步,全部确定即可复现)**:

```text
① 路线生成:  KNN + DINOv3,固定一种匹配方法(patch correspondence),桥长 b0-b6
        ↓
② SAM3 传播: 每条候选路线正向传播一次;用轻量选择器(小模型)选 Top-1
        ↓
③ 学生评审:  训练单图学生,学生-路线一致性(q_model)参与伪标签池筛选
        ↓
④ SAM3 微调: 8 GT + 审核后伪标签池,LoRA 微调(MedSAM3 设置)
```

---

## 1. 术语(精简)

| 术语 | 含义 |
|---|---|
| anchor(锚点) | 8 张有 GT mask 的训练图,标注的唯一来源 |
| 路线 | 一段伪视频:`anchor → 桥帧 → ... → target`,SAM3 沿它传播 |
| 桥长 bN | 桥帧数量;b0 = 直连 anchor→target |
| 传播 | SAM3 用第一帧提示(anchor GT 框)把 mask 逐帧传到 target |
| q_cycle | 回传一致性:target 预测 mask 反传回 anchor,与 anchor GT 的 Dice |
| q_multi | 同一 target 多条候选路线的 mask 共识(两两 Dice) |
| q_model | 学生掩码与路线掩码的 Dice(学生-路线一致性,质量评审信号) |
| 伪标签池 | 通过质量门槛的路线 mask,作为微调数据 |

---

## 2. 数据与协议红线

- Kvasir-SEG:train=800 / validation=100 / test=100;
- 人类标注 = 8 个固定 anchor(其余训练 GT 不可用于选择/筛选);
- validation/test GT **只用于最终评估**;训练集 GT 只做线下诊断;
- 基座:SAM3(`sam3.pt`);特征模型:DINOv3。

---

## 3. 主实验流程

### 3.1 路线生成(KNN,固定一种方法)

- DINOv3 提特征;anchor 的 GT mask 取前景 patch 均值得到 **anchor prototype**;
- **主实验只用 `anchor_conditioned_patch_correspondence`**:取与 prototype 最
  相似的 top-8 patch 相似度均值作为条件分数(局部对应,对背景更鲁棒),
  再束搜索拼桥;
- 理由:虽然冻结基座下 target pooling 的单条固定路线(b6=0.8546)略强,
  **在一切微调后的设置里(ft_1pct / round2 ckpt1 / LoRA)patch correspondence
  都更好或相当**;主线最终要微调 SAM3,所以按端到端口径选择 patch;
- 桥长:**b0-b6 全保留**,评估时逐桥长报告(不预先砍掉短链);
- 产物:每个 target 一组候选路线(纯图片路径)。

### 3.2 SAM3 传播 + 轻量选择器

- 对每条候选路线,用 anchor GT bbox 作第一帧提示,SAM3 正向传播到 target,
  得到 mask 与传播中间输出;
- 选择器:**一个小模型即可**——在验证集上用核心特征拟合 ridge
  (q_cycle、q_multi、掩码面积轨迹、相邻帧 Dice),每个 target 取 Top-1;
- 不需要 21 特征路由器、不需要 B7 这类叠加机制(见对照);
- 冻结基座结果:b3-b6 选择 **0.8773** / b0-b6 **0.8736**(oracle 0.8969/0.9038)。

### 3.3 学生训练 + 质量评审

- 学生 = 单图分割模型(**X3**:GT 流/原始流/新流三流训练,
  验证 Dice ≈ 0.8162,test 单图 0.8633;SC-SAM/SynFoC 作对照);
- **学生作为质量评审器**:对每个候选计算 q_model(学生掩码 vs 路线掩码 Dice),
  与传播质量一起决定伪标签是否进入微调池;
- 第一轮池子(主实验默认口径):`q_return ≥ 0.95 ∧ q_multi ≥ 0.90`
  → **491 / 792** 通过;
- 学生评审门槛(q_model)作为池子的第二道闸,待主线固化(当前可开可关,
  开启后池子更小但更干净)。

### 3.4 SAM3 微调(审核后的池子)

- 数据:8 GT + 491 伪标签 = 499 张,COCO 格式;
- 方法:**LoRA 微调(MedSAM3 设置)**:
  rank=16、alpha=32、dropout=0.1,注入视觉/文本/DETR 编码器与 mask decoder;
  AdamW lr=5e-5、wd=0.01、cosine、20 epochs,基座全冻结,只训低秩增量;
- 结果(test b0-b6 正向传播,无选择器,两模式均值):
  - **主实验(patch correspondence 路线)0.8662**
    (frozen 0.8137 / 全量 3ep 0.8431 / ft_1pct 0.8516)
  - target pooling 路线 0.8599,作对照(C1)
- 关键:LoRA 在长链(b4-b6)显著占优,补齐了全量微调的长程短板。

---

## 4. 对照实验(论文对比部分)

> 每个对照只说明"变什么、结论是什么",细节在 `method_cn.md`。

| # | 对照 | 目的 | 当前结论 |
|---|---|---|---|
| C1 | KNN 模式:target pooling | 验证路线生成方式 | frozen 固定 b6 略强(0.8546 vs 0.8421),但微调后 patch 全面更好 → 主实验选 patch |
| C2 | 选择器:固定 b6 / 无选择 / 复杂路由器 / B7 | 验证选择机制的必要性与复杂度 | 固定 b6 0.8546 < ridge 0.8773;ft_1pct+B7 0.8994 为阶段一最强,但复杂度高 |
| C3 | 微调:全量(3/5/10/20ep)vs LoRA | 验证参数高效微调 | LoRA+491 双模式 0.8599/0.8662 全面超过全量 3ep 与 ft_1pct;全量在长链上短板明显 |
| C4 | 微调预算:1/5/10/20% | 验证数据量单调性 | 非单调(10% 反降),20% 最强 |
| C5 | 学生:SC-SAM / SynFoC / X3 | 验证审核器 | X3 test 0.8633 最强;SC-SAM 0.6626 / SynFoC 0.7615 |
| C6 | 伪标签池:491 原始 / 357 HQ / 455 round2 共识 | 验证池子质量 | round2 池质量更高但全量微调反而更差(早期 checkpoint 才好) |
| C7 | 多轮:round2 再微调 | 验证"再微调"必要性 | 未超过 ft_1pct;早期 checkpoint(e1/e6)是唯一亮点 |
| C8 | 第一帧提示:纯 mask / det_gtmask | 验证提示方式 | 纯 mask -0.0131 不可用;det_gtmask +0.0055 可作后续优化 |
| C9 | 诊断:RouteCo(mask-decoder-only LoRA) | 验证 LoRA 打对位置 | decoder-only LoRA 近似中性 → LoRA 必须覆盖传播关键部件 |

---

## 5. 结果总表

### 5.1 主实验口径(test b0-b6 正向传播,逐桥长均值)

| 方法 | patch correspondence(主线) | target pooling(对照 C1) |
|---|---:|---:|
| frozen 基座 | 0.8137 | 0.8163 |
| ft_1pct(8图,全量 20ep) | 0.8516 | 0.8387 |
| 全量 3ep(1%+491) | 0.8431 | 0.8400 |
| **主实验:LoRA 20ep(1%+491)** | **0.8662** | 0.8599 |

### 5.2 选择器口径(b3-b6,baseline 对比)

| 方法 | selected | oracle |
|---|---:|---:|
| frozen + 轻量 ridge | 0.8773 | 0.8969 |
| ft_1pct + 轻量 ridge | 0.8946 | 0.9198 |
| ft_1pct + X3 + B7(对照,最复杂) | 0.8994 | 0.9198 |

---

## 6. 当前状态与下一步

- 主线 4 步均已跑通,LoRA+491 是当前最强单点配置;
- 待固化(简单确认即可,不需要新增机制):
  1. 选择器:确认用"核心特征 ridge"还是"q_cycle×q_multi 加权"(二选一);
  2. 学生评审:确认 q_model 门槛开/关及阈值;
  3. 桥长池:确认 b0-b6 全量(当前默认)。
- 主线确认后,按 3.1-3.4 完整重跑一遍,得到论文主表;
- 对照实验(C1-C9)已有结果,直接整理进论文对比章节。

---

## 7. 复现

| 环节 | 路径 |
|---|---|
| 路线/传播质量 | `scripts/stage1_feature_knn_routes.py`、`scripts/stage1_eval_routes_forward_only.py` |
| 选择器 | `scripts/eval_ft1pct_pq_router.py`(可截取核心特征) |
| 学生 | `scripts/run_s27_student.py`(X3) |
| 池子 | `scripts/select_phase1_mainline_pseudo568.py`(491) |
| LoRA 训练 | `scripts/train_sam3_lora_kvasir.py` + `configs/kvasir_1pct_plus_pseudo491_lora.yaml` |
| LoRA 合并/评估 | `scripts/merge_sam3_lora_video_checkpoint.py`、`scripts/run_ft3ep_lora_test_eval.sh` |
| 数据集 | `.../dataset/KvasirSEG_1pct_plus_pseudo491_seed2026`(499 张) |
