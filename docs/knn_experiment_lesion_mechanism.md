# Lesion KNN 机制拆解实验(为什么 0.77 → 0.89?)

> 创建:2026-08-16。前置:第一轮 pseudo mask → mask 内 SAM3 特征 → 第二轮 KNN
> 在 test 上 direct 0.7698 → 0.8863(+0.117)。本组实验拆解这个巨大提升
> 到底来自 anchor 选择、bridge 选择、还是 pseudo mask 本身携带的信息。
> 统一口径:base @256,SAM3-enc @1008 特征,test 100 targets,b0–b6,beam 32。

## E1:Anchor/Bridge 解耦(lesion 改善谁?)

| 组合 | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| G/G(global anchor, global bridge) | 0.7698 | 0.8020 | 0.8452 | 0.8672 | 0.8500 | 0.8546 | 0.8550 |
| L/L(lesion, lesion) | **0.8863** | **0.8921** | 0.8871 | 0.8650 | 0.8772 | 0.8832 | 0.8836 |
| L/G(lesion anchor, global bridge) | **0.8863** | 0.8719 | 0.8381 | 0.8178 | 0.8347 | 0.8083 | 0.8172 |
| G/L(global anchor, lesion bridge) | 0.7698 | 0.8399 | 0.8414 | 0.8240 | 0.8380 | 0.8559 | 0.8377 |

**结论**:lesion 的收益**主要来自 anchor 检索**(direct +0.117,100% anchor 效应);
bridge 选择有独立贡献但较小(b1:global anchor 下 lesion bridge +0.038;
lesion anchor 下 +0.070)。**解释 B(better anchor retrieval)主导,解释 A
(better pseudo-video routing)次要但存在**。

## E2:Round0 → Round1 真正增益

| 口径 | Dice |
|---|---|
| Round0(第一轮 global,b0-b6 fusion,test_pseudo_masks_round1) | 0.8979 |
| Round1 global 重跑 per-target best | 0.9022 |
| **Round1 F_lesion per-target best** | **0.9300** |
| Round1 F_lesion direct(固定 b0) | 0.8863 |

**结论**:lesion 单桥(direct)低于 Round0,但 **per-target best 超越 Round0
+0.032**——增益真实存在,但藏在"每个 target 的最优路线"里,不是固定桥长。
支持"按 target 自适应选桥长"方向。

## E3:GT mask 上限(仅分析,绝不作为方法)

| mask | direct | b1 | b2 | b5 | b6 | per-target best |
|---|---:|---:|---:|---:|---:|---:|
| 伪 mask(round1) | 0.8863 | 0.8921 | 0.8871 | 0.8832 | 0.8836 | 0.9300 |
| **GT mask** | 0.8829 | 0.8885 | 0.8940 | 0.9190 | 0.9028 | **0.9414** |

**结论**:GT 并没有大幅超过伪 mask(direct 甚至略低 0.003;per-target best
高 0.011)——伪 mask 已捕捉到绝大部分病灶信息。direct 上 GT < 伪,暗示
**伪 mask 携带的"SAM3 可分割性"特征本身有利**(它来自 SAM3 预测,与传播
输出同分布)。提升空间 ≈ 1 个点,不是"mask 不准"的锅。

## E4:mask 破坏(因果证据)

| mask 变体 | direct | b1 | per-target best |
|---|---:|---:|---:|
| 正常伪 mask | 0.8863 | 0.8921 | 0.9300 |
| 膨胀(dilate) | 0.8747 | 0.8881 | 0.9295 |
| 腐蚀(erode) | 0.8854 | 0.8803 | 0.9226 |
| bbox(外接框) | 0.8785 | 0.8888 | 0.9225 |
| 随机移位(random roll) | 0.8644 | 0.8912 | 0.9210 |
| F_global(对照) | 0.7698 | 0.8020 | 0.9022 |

**结论**:所有破坏版本都保留大部分增益(direct 0.86–0.89 vs 基线 0.77)。
**提升不完全依赖精确病灶定位**:即使 mask 被膨胀/bbox/随机移位,lesion 式
描述子仍远优于 global。增益更多来自"mask 圈选(过滤背景 + 编码 SAM3 预测
特性)",而非像素级定位准确。

## E5:相似度 vs 真实传播质量的 Spearman 相关

60 对 train 图真实 SAM3 传播 i→j,q = 传播 mask 与 j 伪 mask 的 Dice:

| 相似度 | ρ | p |
|---|---:|---:|
| s_global | 0.191 | 0.143(不显著) |
| **s_lesion** | **0.277** | **0.032(显著)** |

**结论**:lesion 相似度是传播质量的显著代理(ρ≈0.28 vs 0.19),"
病灶相似 → 可传播"成立,但相关性中等——相似度只解释 ~8% 方差,
**真正的传播质量估计(cycle consistency / 传播感知检索)仍有大空间**。

## 综合结论(论文叙事)

1. **病灶特征有效,机制 = 更好的 anchor 检索为主 + 更好的 bridge 为辅**;
2. **不是"精确定位病灶"**:GT 不超伪 mask,mask 破坏仍保留增益;
   增益来自"mask 圈选"(过滤 task-irrelevant 背景)+ 伪 mask 的 SAM3 预测特性;
3. **伪 mask 已接近上限**,第一轮 → 第二轮 self-refinement 的净增益
   在 per-target best 口径为 +0.032;
4. **下一步最有价值的不是更细的特征,而是传播感知**:
   E5 显示 lesion 相似度只能解释 8% 传播质量方差;直接度量
   传播可达性(cycle consistency)或按 target 自适应选桥长(per-target
   best 0.9300 vs 固定桥长均值 0.88)是明确的方向。

## 产物位置

- routes/eval:`work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_lesion_knn/`
  (sam3enc_lesion / _global_lesion / _lesion_anchor_global_bridge /
  _global_anchor_lesion_bridge / __gt_lesion / __lesion_corrupt_{dilate,erode,bbox,random})
- E5:`work/kvasir_1pct_anchors/e5_transport_correlation.json`
- 描述子:`features/sam3enc_gt_lesion_descriptors_s1008.npz`(分析用,GT 仅 test)
