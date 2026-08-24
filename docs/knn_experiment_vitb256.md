# KNN 路线构造重实验（ViT-B / 统一 256×256）

> 创建:2026-08-09。本文档在实验开始前固定协议，避免后续遗漏或口径漂移。

## 0. 目标与动机

- 重新实验 KNN 路线构造，重点是 `anchor_conditioned_patch_correspondence`
  中 **"用什么特征做 KNN 检索"** 这一可实验变量；
- DINOv3 骨干从 vits16 换成 **vitb16**（仓库中无 ViT-T 权重，
  旧实现实际是 ViT-S/16，本次按"小骨干 → 大骨干"换成 ViT-B/16）；
- **统一分辨率保证公平**：SAM3 传播固定 256×256；DINOv3 特征输入固定 256×256；
- 数据集与有/无标注划分沿用之前：Kvasir-SEG train=800 / validation=100 / test=100，
  8 个固定 GT anchor（1% 标注），其余 792 张 train 图作为无标注桥帧池；
- 构造完 KNN 路径后，用 SAM3（冻结基座）在 test 上评估**两种预测模式**：
  直接预测（direct / b0）与伪视频 bridge（b0–b6）；
- 最终把所有变体的逐桥长结果汇总比较。

皮肤病 79%    80%


息肉 clinic kr colondb


## 1. 统一协议（务必遵守）

| 项 | 固定值 |
|---|---|
| 数据集 | Kvasir-SEG 快照，`merged_manifest.jsonl`（train 800 / val 100 / test 100） |
| 标注 | 8 个固定 GT anchor（`support_manifest.jsonl`），桥帧候选 = 全部 train 图 |
| SAM3 checkpoint | 冻结基座 `sam3.pt`（modelscope facebook/sam3），不微调 |
| SAM3 评估分辨率 | **256×256**（`--canvas 256`），forward-only test Dice |
| DINOv3 特征输入 | **256×256**，patch grid 16×16=256 tokens，anchor prototype mask 下采样到 16×16 |
| DINOv3 骨干 | `vitb16`（`dinov3_vitb16_pretrain_lvd1689m.pth`） |
| 特征缓存命名 | `dinov3_{model}_s{size}_features.npz`（区分 224 / 256，防止混用） |
| 路由搜索 | beam width = 32；bridge b0–b6；split = test；target 排除 8 个 support；桥帧不重复 |
| 评估口径 | 每变体 100 targets × 7 桥长 = 700 条路线，逐桥长均值 Dice |
| 输出根目录 | `work/kvasir_1pct_anchors/stage1_feature_knn_vitb256/` |

## 2. KNN 构造方法（本次选用的）

### 2.1 非 anchor 模式：KNN 特征 = 模式自身描述子（无额外变体）

| mode | KNN 相似度 | 说明 |
|---|---|---|
| `t18_corrected` | T18 199-D 全局描述子 | 内部固定 64×64 resize（该描述子的定义），与"特征输入分辨率"无关，保持不变 |
| `dino_global_pooling` | DINOv3 CLS token 余弦 | ViT-B，@256 |
| `dino_patch_average` | DINOv3 patch token 均值余弦 | ViT-B，@256 |

### 2.2 anchor-conditioned 模式：KNN 候选排序可换检索特征（本次核心实验）

候选排序规则不变（与旧实现一致）：
按 **KNN 检索相似度（主）× cond_score（次）× merged_id（三级，确定性）** 排序；
路线打分（`route_score`）也不变：节点 cond 分数的 min/mean + 相邻节点 patch-mean 相似度。

4 种 KNN 检索特征：

| knn_feature | 检索空间 | 含义 |
|---|---|---|
| `patch_mean` | patch token 均值 | **基线**，沿用旧实现 |
| `cls` | CLS token | 全局语义 token 相似度 |
| `pooled` | anchor-conditioned 加权池化描述子 | 每 anchor 一个描述子，相似度为 (A,N,N)，检索"目标相关"图像描述子 |
| `cond` | 纯 anchor 条件分数 | 无尾节点相似度，只按 anchor 相关度排序（消融：锚相关是否足以选桥） |

### 2.3 变体清单（共 11 个）

| mode_key | mode | DINOv3 | knn_feature | 说明 |
|---|---|---|---|---|
| `t18_corrected` | t18_corrected | — | — | 全局描述子基线 |
| `dino_global_pooling` | dino_global_pooling | vitb16 | — | CLS 描述子 |
| `dino_patch_average` | dino_patch_average | vitb16 | — | patch 均值描述子 |
| `anchor_conditioned_target_pooling` | target_pooling | vitb16 | patch_mean | 基线 |
| `anchor_conditioned_target_pooling__knn_cls` | target_pooling | vitb16 | cls | |
| `anchor_conditioned_target_pooling__knn_pooled` | target_pooling | vitb16 | pooled | |
| `anchor_conditioned_target_pooling__knn_cond` | target_pooling | vitb16 | cond | |
| `anchor_conditioned_patch_correspondence` | patch_correspondence | vitb16 | patch_mean | **基线** |
| `anchor_conditioned_patch_correspondence__knn_cls` | patch_correspondence | vitb16 | cls | |
| `anchor_conditioned_patch_correspondence__knn_pooled` | patch_correspondence | vitb16 | pooled | |
| `anchor_conditioned_patch_correspondence__knn_cond` | patch_correspondence | vitb16 | cond | |

## 3. 评估与产出

- 两种预测模式均由同一次 forward-only 传播得到：
  - **直接预测 test** = `direct`（b0，anchor→target 两帧）；
  - **伪视频方法** = bridge b1–b6（b0 同时属于 direct 口径）；
- 产出：
  - 每变体 `eval_base_no_ft_b7_forward/route_family_summary.json`（direct..b6 均值 Dice）；
  - `comparison_summary.json` / `comparison_summary.md`：11 变体 × (direct..b6) 汇总表；
  - 旧参考数字（vits16@224 + SAM3@512，来自 `stage1_feature_knn_b7/`）仅作粗参考，
    因分辨率不同**不参与正式对比**。

## 4. 防遗漏检查清单

- [x] DINOv3 ViT-B @256 forward 验证通过（256 patch tokens）
- [ ] 特征提取固定 `--feature-size 256`（grid 16）
- [ ] SAM3 评估固定 `--canvas 256`
- [ ] T18 保持原定义（内部 64×64 全局描述子），不随"特征输入分辨率"改动
- [ ] 新根目录 `stage1_feature_knn_vitb256/`，不覆盖旧根目录文件
- [ ] 每个变体写 `meta.json`（mode / dinov3_model / feature_size / knn_feature / beam / bridge 范围）
- [ ] 汇总表明确区分"本实验（256×256）"与"旧参考（224 + 512）"

## 5. 已知注意点

- ViT-B 特征提取 batch 用 16（显存安全，24GB GPU）；
- `pooled` KNN 相似度是 per-anchor 的 (A, N, N) 张量；`cond` KNN 时同一 anchor
  的所有 tail 共享同一排序列表（内存优化）；
- 本次只做冻结基座 forward-only 逐桥长对比，**不涉及**传播质量路由器、学生审核、
  B7、微调等环节；ft_1pct 等 checkpoint 的复测列为后续可选步骤；
- 若后续要与旧表（3.4 节）对齐，需在相同分辨率下重跑 vits16，否则不直接可比。

## 6. 结果（2026-08-09 完成，test 100 targets，forward-only Dice）

协议：冻结 SAM3 `sam3.pt`，canvas 256，DINOv3 ViT-B @256（grid 16），
beam width 32，bridge b0–b6，每个变体 700 条路线。

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t18_corrected | — | 0.7421 | 0.7817 | 0.7942 | 0.8340 | 0.8311 | 0.8357 | **0.8573** |
| dino_global_pooling | — | 0.7000 | 0.7849 | 0.8182 | 0.8059 | 0.7984 | 0.8124 | 0.8282 |
| dino_patch_average | — | 0.7325 | 0.7634 | **0.8350** | 0.8277 | 0.8001 | 0.8109 | 0.8121 |
| anchor_conditioned_target_pooling | patch_mean | 0.6893 | 0.7691 | 0.7960 | 0.8052 | 0.7736 | 0.7734 | 0.7739 |
| anchor_conditioned_target_pooling | cls | 0.6893 | 0.7777 | 0.8104 | 0.7909 | 0.7790 | 0.7978 | 0.8126 |
| anchor_conditioned_target_pooling | pooled | 0.6893 | 0.7721 | 0.8090 | 0.7982 | 0.7927 | 0.7770 | 0.7833 |
| anchor_conditioned_target_pooling | cond | 0.6893 | 0.7607 | 0.7831 | 0.7573 | 0.7493 | 0.7725 | 0.7608 |
| anchor_conditioned_patch_correspondence | patch_mean | 0.7035 | 0.7862 | **0.8330** | 0.8103 | 0.7964 | 0.8097 | 0.8081 |
| anchor_conditioned_patch_correspondence | cls | 0.7035 | 0.7960 | 0.8100 | 0.7986 | 0.8117 | 0.8117 | **0.8355** |
| anchor_conditioned_patch_correspondence | pooled | 0.7035 | 0.7912 | 0.8185 | 0.7960 | 0.8049 | 0.8019 | 0.8090 |
| anchor_conditioned_patch_correspondence | cond | 0.7035 | 0.7768 | 0.8123 | 0.7867 | 0.7862 | 0.8003 | 0.8230 |

### 6.1 要点

- 全场最高：**t18_corrected b6 = 0.8573**（@256 下 T18 长桥最强）；
- anchor-conditioned 两种模式下，**KNN 用 cls 均是最优检索特征**：
  patch_correspondence b6 0.8355（基线 patch_mean 0.8081）、b4 0.8117（0.7964）；
  target_pooling b6 0.8126（基线 0.7739）；
- `cond`（纯 anchor 条件分数）检索最弱，说明"仅靠 anchor 相关度"不足以选桥；
- direct（b0）不经过 KNN 选桥，同一 mode 的各 knn 变体 direct 列相同（符合预期）；
- 与旧参考（vits16@224 + SAM3@512）**不可直接对比**（两个分辨率变量同时改变）；
  注意旧口径下 anchor-conditioned 是全场最强（0.8546/0.8421），@256 新口径下
  anchor-conditioned 反而落后于 t18，值得后续做单变量消融（如 vits16@256 + SAM3@256）。

## 7. LoRA 复测：lora_p491_e20（2026-08-09 完成）

同一批 11 个变体路线（ViT-B @256，SAM3 @256），换成
`lora_p491_e20_merged_video.pt`（base sam3.pt + LoRA，8 GT + 491 伪标签，e20）
forward-only 评估，逐桥长 test Dice：

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t18_corrected | — | 0.8424 | 0.8859 | 0.8791 | 0.8815 | 0.8794 | 0.8831 | 0.8880 |
| dino_global_pooling | — | 0.8142 | 0.8498 | 0.8794 | 0.8786 | 0.8665 | 0.8775 | 0.8642 |
| dino_patch_average | — | 0.8197 | 0.8616 | 0.8650 | 0.8816 | 0.8666 | 0.8718 | 0.8768 |
| anchor_conditioned_target_pooling | patch_mean | 0.8030 | 0.8531 | 0.8574 | 0.8467 | 0.8610 | 0.8724 | 0.8888 |
| anchor_conditioned_target_pooling | cls | 0.8030 | 0.8324 | 0.8443 | 0.8283 | 0.8762 | 0.8819 | **0.8978** |
| anchor_conditioned_target_pooling | pooled | 0.8030 | 0.8461 | 0.8383 | 0.8402 | 0.8632 | 0.8676 | 0.8866 |
| anchor_conditioned_target_pooling | cond | 0.8030 | 0.8472 | 0.8497 | 0.8592 | 0.8716 | 0.8848 | 0.8861 |
| anchor_conditioned_patch_correspondence | patch_mean | 0.7991 | 0.8577 | 0.8651 | 0.8529 | 0.8536 | 0.8863 | 0.8558 |
| anchor_conditioned_patch_correspondence | cls | 0.7991 | 0.8403 | 0.8627 | 0.8544 | 0.8821 | 0.8768 | 0.8804 |
| anchor_conditioned_patch_correspondence | pooled | 0.7991 | 0.8496 | 0.8685 | 0.8629 | 0.8742 | 0.8851 | 0.8834 |
| anchor_conditioned_patch_correspondence | cond | 0.7991 | 0.8514 | 0.8624 | 0.8596 | 0.8826 | 0.8720 | 0.8626 |

### 7.1 要点

- LoRA 微调整体大幅提升（每格 +0.03 ~ +0.13，direct +0.10 左右）；
- 新全场最高：**target_pooling knn=cls b6 = 0.8978**（base 0.8126 → +0.085）；
  其次 target_pooling patch_mean b6 0.8888、t18 b6 0.8880；
- **排序反转**：base 下 t18 最强、anchor-conditioned 偏弱；LoRA 下
  anchor-conditioned target_pooling 的长桥（b5/b6）反超 t18，
  与旧口径（vits16@224 + SAM3@512）"anchor-conditioned 最强"的趋势一致；
- base 下最弱的 `cond` KNN 受益最大（target_pooling b6 +0.125、b4 +0.122），
  说明 LoRA 后"锚相关度"信号变得更可靠；
- 汇总产物：`comparison_summary_lora_p491_e20.md/json`、
  `comparison_summary_base_vs_lora.md`。

## 9. SAM3 编码器 KNN（2026-08-13 完成）

把 KNN 特征提取器从 DINOv3 换成 **SAM3 主干编码器**
（32 层 ViT、patch 14、embed 1024、无 CLS token）：
`sam3enc_patch_average`（patch 均值描述子）+
`sam3enc_anchor_conditioned_target_pooling` / `sam3enc_anchor_conditioned_patch_correspondence`
（各配 patch_mean / pooled / cond 三种 KNN 检索特征），共 7 个变体；
特征输入分辨率跑 **1008（原生）与 256（协议对齐，需重算 RoPE）** 两组；
SAM3 评估统一 @256，checkpoint 为 base 与 lora_p491_e20。

### 9.1 结果（test forward-only Dice，每格 100 targets）

**1008 特征 + base @256：**

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_target_pooling | patch_mean | 0.8205 | 0.8762 | 0.8729 | 0.8701 | 0.8655 | **0.8874** | 0.8771 |
| sam3enc_target_pooling | cond | 0.8205 | 0.8811 | 0.8823 | 0.8617 | 0.8615 | 0.8822 | 0.8639 |
| sam3enc_target_pooling | pooled | 0.8205 | 0.8735 | 0.8758 | 0.8714 | 0.8563 | 0.8643 | 0.8769 |
| sam3enc_patch_correspondence | patch_mean | 0.8202 | 0.8321 | 0.8302 | 0.8035 | 0.8173 | 0.8247 | 0.8388 |
| sam3enc_patch_correspondence | pooled | 0.8202 | 0.8499 | 0.8311 | 0.8033 | 0.8451 | 0.8154 | 0.8408 |
| sam3enc_patch_correspondence | cond | 0.8202 | 0.8396 | 0.8018 | 0.7944 | 0.8028 | 0.8120 | 0.8084 |
| sam3enc_patch_average | patch_mean | 0.7698 | 0.8020 | 0.8452 | 0.8672 | 0.8500 | 0.8546 | 0.8550 |

**1008 特征 + lora_p491_e20 @256：**

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_target_pooling | cond | 0.8995 | **0.9154** | 0.9032 | 0.9131 | **0.9154** | 0.9136 | 0.9131 |
| sam3enc_patch_correspondence | patch_mean | 0.8992 | 0.9129 | 0.9098 | 0.9029 | 0.9047 | 0.9149 | **0.9150** |
| sam3enc_target_pooling | patch_mean | 0.8995 | 0.9108 | 0.8991 | 0.8927 | 0.8939 | 0.9082 | 0.8934 |
| sam3enc_patch_correspondence | pooled | 0.8992 | 0.9000 | 0.9135 | 0.8994 | 0.9061 | 0.9066 | 0.9021 |
| sam3enc_patch_average | patch_mean | 0.8261 | 0.8648 | 0.8728 | 0.8995 | 0.8981 | 0.8958 | 0.8993 |

**256 特征（协议对齐）+ base @256：**

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_target_pooling | patch_mean | 0.7981 | 0.8466 | 0.8700 | 0.8623 | 0.8739 | 0.8696 | 0.8740 |
| sam3enc_target_pooling | cond | 0.7981 | 0.8534 | 0.8647 | **0.8841** | 0.8461 | 0.8698 | 0.8462 |
| sam3enc_target_pooling | pooled | 0.7981 | 0.8391 | 0.8466 | 0.8493 | 0.8765 | 0.8657 | 0.8529 |
| sam3enc_patch_correspondence | patch_mean | 0.7066 | 0.7549 | 0.7885 | 0.8594 | 0.8576 | 0.8553 | 0.8536 |
| sam3enc_patch_correspondence | pooled | 0.7066 | 0.7458 | 0.8030 | 0.8652 | 0.8618 | 0.8505 | 0.8603 |
| sam3enc_patch_correspondence | cond | 0.7066 | 0.7355 | 0.8357 | 0.8362 | 0.8499 | 0.8484 | 0.8318 |
| sam3enc_patch_average | patch_mean | 0.7502 | 0.7889 | 0.8187 | 0.8204 | 0.8094 | 0.7795 | 0.7680 |

**256 特征（协议对齐）+ lora_p491_e20 @256：**

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_patch_correspondence | cond | 0.7647 | 0.8679 | 0.8739 | 0.8983 | **0.9082** | 0.8809 | 0.9012 |
| sam3enc_patch_correspondence | patch_mean | 0.7647 | 0.8680 | 0.8722 | 0.8886 | 0.8997 | 0.9036 | 0.8994 |
| sam3enc_patch_correspondence | pooled | 0.7647 | 0.8683 | 0.8719 | 0.8849 | 0.8913 | 0.8985 | 0.8939 |
| sam3enc_target_pooling | patch_mean | 0.8482 | 0.8528 | 0.8659 | 0.8730 | 0.8927 | 0.8879 | 0.8960 |
| sam3enc_target_pooling | cond | 0.8482 | 0.8640 | 0.8277 | 0.8779 | 0.8871 | 0.8946 | 0.8988 |
| sam3enc_target_pooling | pooled | 0.8482 | 0.8597 | 0.8616 | 0.8620 | 0.8944 | 0.8901 | 0.8985 |
| sam3enc_patch_average | patch_mean | 0.8222 | 0.8505 | 0.8861 | 0.8689 | 0.8858 | 0.8930 | 0.8876 |

### 9.2 与 DINOv3 ViT-B @256 对比

| 特征提取器 | checkpoint | 最强固定桥 | 出处 |
|---|---:|---|---|
| DINOv3 ViT-B @256（含 t18） | base | 0.8573 | t18 b6 |
| DINOv3 ViT-B @256（anchor 模式） | base | 0.8126 | target_pooling knn=cls b6 |
| SAM3-enc @256 | base | 0.8841 | target_pooling cond b3 |
| **SAM3-enc @1008** | base | **0.8874** | target_pooling patch_mean b5 |
| DINOv3 ViT-B @256 | lora | 0.8978 | target_pooling knn=cls b6 |
| SAM3-enc @256 | lora | 0.9082 | patch_correspondence cond b4 |
| **SAM3-enc @1008** | lora | **0.9154** | target_pooling cond b1/b4 |

### 9.3 要点

- **SAM3 编码器 KNN 全面优于 DINOv3**：base 下 0.8874 vs 0.8573（t18）/
  0.8126（纯 DINOv3 anchor）；lora 下 **0.9154 vs 0.8978（+0.018）**；
- **direct（b0）显著提升**：1008 特征下 base direct 0.82（DINOv3 0.69–0.74）、
  lora direct 0.90（DINOv3 0.80）——说明 SAM3 编码器特征让"选哪个 anchor 直传"
  更准，这是不经过选桥的纯特征收益；
- **1008 原生 > 256 协议对齐**（lora 0.9154 vs 0.9082；base 0.8874 vs 0.8841），
  符合预期（256 需要 RoPE 重算 + 窗口填充妥协）；
- KNN 检索特征偏好改变：DINOv3 下 cls 最优；SAM3 编码器（无 CLS）下
  **patch_mean / cond 最强**，pooled 稍弱；
- 产物：`stage1_feature_knn_sam3enc_s1008/` 与 `_s256/` 下各 4 份
  `comparison_summary_*.md/json`。

## 8. 后续：路线选择/路由器实验

KNN 路线之后的选择/路由阶段单独成文：
[docs/route_selector_experiment_vitb256.md](route_selector_experiment_vitb256.md)。

当前结论（lora_p491_e20 @256，2026-08-10）：

- 固定桥长最高：target_pooling knn=cls b6 = 0.8978；
- 传播质量 + 学生审核（复用 256-px X3/S2/S3）后，
  **lin_cal_mean P1 = 0.9035**（validation 校准
  q_return 0.80 / q_multi 0.05 / q_model 0.15），B7_mean P1 = 0.9014，
  B7_X3 P2 = 0.9022，均超过固定 b6；
- 正式报告与逐 target 明细：
  `work/kvasir_1pct_anchors/route_selector_vitb256/route_selector_report_lora_p491_e20.md`
  及同目录 `.json/.csv`。
- base checkpoint 的同类实验已完成（2026-08-11）：最优为纯 q_model_mean
  P1 = 0.8945（比 base 最强固定桥 t18 b6 0.8573 高 +0.036），
  报告见 `work/kvasir_1pct_anchors/route_selector_vitb256/route_selector_report_base.md`。
