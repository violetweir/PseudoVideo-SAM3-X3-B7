# Lesion-Centric KNN 系列实验(交接文档)

> 版本: 2026-08-16
> 状态: 实验完成,结论已收敛;本文档供下一个 agent 直接接手
> 对应 docs 下细档:`knn_experiment_lesion_mechanism.md`、
> `knn_experiment_negcontrol_transport.md`(本文件是合并后的交接版)

## 1. 背景与统一协议

问题:无序医学图像 → KNN 组伪视频 → SAM3 传播。核心假设:
**"图像相似"应该升级为"病灶相似"→ 最终"mask 可传播"**。

统一口径(所有数字):
- 数据集 Kvasir-SEG 快照,train 800 / val 100 / test 100;8 个 GT anchor
- 特征:SAM3 主干编码器 @1008(patch 14,grid 72),L2 归一化描述子
- 评估:冻结 base `sam3.pt`,`--canvas 256`,forward-only Dice,beam 32,b0–b6
- lesion 描述子 = 伪 mask(round1)圈选的前景 patch token 均值
  (`sam3enc_mask_descriptors_s1008.npz`,1000 张全覆盖)
- 关键无泄漏约定:test 的 lesion 用 round1 伪 mask(非 GT);GT 只用于分析

## 2. 轮次 1:Lesion KNN 三路对比(核心结果)

| 检索特征 | direct | b1 | b2 | b3 | b4 | b5 | b6 | mean(b1–6) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F_global(整图 patch_mean,基线) | 0.7698 | 0.8020 | 0.8452 | 0.8672 | 0.8500 | 0.8546 | 0.8550 | 0.8457 |
| **F_lesion(病灶)** | **0.8863** | **0.8921** | 0.8871 | 0.8650 | 0.8772 | 0.8832 | 0.8836 | **0.8814** |
| F_global+F_lesion(拼接) | 0.8785 | 0.8870 | 0.8720 | 0.8689 | 0.8760 | 0.8743 | 0.8747 | 0.8755 |

**结论:病灶特征 >> 整图特征(+0.117 direct);拼接反而稀释,单独 lesion 最优。**
实现:stage1_feature_knn_routes.py 的 `sam3enc_lesion` / `sam3enc_global_lesion` mode。

## 3. 轮次 2:机制拆解(E1–E5)

### E1 anchor/bridge 解耦——lesion 主要改善 anchor 检索

| 组合 | direct | b1 |
|---|---:|---:|
| G/G(基线) | 0.7698 | 0.8020 |
| L/G(lesion anchor, global bridge) | **0.8863** | 0.8719 |
| G/L(global anchor, lesion bridge) | 0.7698 | 0.8399 |
| L/L(全 lesion) | 0.8863 | **0.8921** |

→ direct 提升 100% 来自 anchor 选择(+0.117);bridge 独立贡献 +0.038(b1)。
实现:mode `sam3enc_lesion_anchor_global_bridge` / `_global_anchor_lesion_bridge`(route_score 用 anchor_sim/sim 分开)。

### E2 Round0→Round1 真实增益

- Round0(第一轮 global,b0-b6 fusion 伪 mask)test Dice = **0.8979**
- Round1 F_lesion per-target best = **0.9300**(+0.032);direct 单桥 0.8863 < Round0
- → 增益真实,但藏在"每 target 最优路线"里,支持**自适应桥长**。

### E3 GT 上限(仅分析)

- GT mask direct 0.8829(< 伪 mask 0.8863);per-target best 0.9414 vs 伪 0.9300
- → 伪 mask 已接近上限;伪 mask 携带的 SAM3 预测特性本身有利。

### E4 mask 破坏(因果)

| 变体 | direct | per-target best |
|---|---:|---:|
| 伪 mask | 0.8863 | 0.9300 |
| 膨胀 | 0.8747 | 0.9295 |
| 腐蚀 | 0.8854 | 0.9226 |
| bbox | 0.8785 | 0.9225 |
| 随机移位 | 0.8644 | 0.9210 |

→ 增益不完全依赖精确定位。

### E5 传播相关性(60 对真实传播)

- ρ(global, q) = 0.191(p=0.14 不显著);ρ(lesion, q) = **0.277**(p=0.032 显著)
- → lesion 相似度是传播质量的显著代理,但只解释 ~8% 方差。

## 4. 轮次 3:真 Negative Controls + Transport Proxies

### 4.1 Negative controls(解释 0.8863 为什么高)

| mask 变体 | direct |
|---|---:|
| 伪 mask(病灶) | 0.8863 |
| GT mask | 0.8829 |
| fixed crop(固定中心,非病灶) | 0.8464 |
| random non-overlap(确定不重叠) | 0.8584 |
| shuffled(像素打乱) | 0.8078 |
| 整图基线 | 0.7698 |

**两层分解:主因 = 空间聚焦(圈选任意区域 +0.08~0.09);次因 = 病灶定位(+0.028)。**
shuffled 回落到 0.808 → 收益来自空间局部性,非 token 数量。

### 4.2 Transport proxies 全军覆没(validation, 700 条)

| 代理 | 定义 | ρ vs GT Dice | p |
|---|---:|---:|---:|
| q_model | SAM3 传播置信度 | 0.029 | 0.446 |
| q_multi | forward_candidate_count | 0.130 | 0.0006 |
| path_bottleneck | 路径最小边相似度 | **0.138** | 0.0003 |
| path_mean | 路径平均边相似度 | 0.109 | 0.004 |
| cycle | T→B→A 往返 mask 稳定性 | −0.090 | 0.372 |

- proxy-oracle(代理选最优路线)≈ 随机平均(0.84~0.85 vs 0.8435),GT-oracle=0.8972
- **朴素无监督代理全部失败,连 lesion 相似度(0.277)都不如**;
  达到 0.4/0.6 的目标需要更聪明的估计或小规模监督校准。

## 5. 收敛结论(给下一个 agent 的要点)

1. **lesion KNN(病灶相似检索)有效且机制清楚**:
   `F_lesion`(伪 mask 圈选前景均值)是当前最强检索特征,direct 0.8863;
2. **相似度天花板已到**:E5 ρ=0.277 说明"病灶相似→可传播"成立但弱;
   朴素无监督代理(path bottleneck/置信度/候选数/cycle)无法预测传播质量;
3. **真正的缺口 = "无 GT 判断哪条传播路径可靠"**;
4. **两个有实数据支持的高价值方向**:
   - 自适应桥长 / 每 target 选最优路线(per-target best 0.930 vs 固定桥长均值 0.881,~+0.05 实空间)
   - validation 上校准传播质量估计器(小规模监督/半监督,吃 proxy-oracle 与 GT-oracle 的 5 个点)

## 5.5 轮次 4(C3):路径不变性实验 —— 已更新,详见 11_c3_path_invariance_handoff.md

- 核心发现:**consensus(最终预测一致性)是远超边级代理的质量信号**
  - 多 anchor 池 route 级 ρ(consensus, GT) = 0.70–0.73(validation)
  - 对比:lesion 相似度 0.277、path_bottleneck 0.138、q_multi 0.130、cycle −0.090
- selector(medoid/anchor-balanced/cross-anchor support)相对均值 +0.02~0.03,regret 0.03~0.05;
  规则已在 validation 冻结(τ=0.85),test 一次性汇报完成:
  C3-C 冻结 selector = **0.8947**(test 100,vs 候选均值 0.863、oracle 0.930),
  与既有池 medoid 0.896 持平;ρ(consensus,GT) test 0.65 / validation 0.70-0.73(见 11 号文档 6.5/7.5)。
- 补充:pseudo vs GT lesion 检索 paired bootstrap(test 100)= Δ+0.0035,
  95% CI [−0.0009, +0.0115],p=0.376 → "粗定位足够",统计等价。
  产物:work/kvasir_1pct_anchors/pseudo_vs_gt_lesion_paired_bootstrap.json

## 6. 产物位置(接手起点)

| 内容 | 路径 |
|---|---|
| routes/eval(lesion 全系列) | `work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_lesion_knn/` |
| 伪 mask 描述子 | `.../features/sam3enc_mask_descriptors_s1008.npz` |
| GT-lesion 描述子(仅分析) | `.../features/sam3enc_gt_lesion_descriptors_s1008.npz` |
| patch tokens 缓存(1000×5184×1024 fp16) | `work/.../stage1_feature_knn_sam3enc_s1008/features/sam3_base_s1008_patches.npz` |
| E5 相关性 | `work/kvasir_1pct_anchors/e5_transport_correlation.json` |
| transport 代理分析 | `work/kvasir_1pct_anchors/transport_proxies_v1.json` / `_v2_cycle.json`、`e5_cycle_proxy.json` |
| 核心脚本 | `scripts/stage1_feature_knn_routes.py`(lesion/解耦/neg-control modes) |
| 分析脚本 | `scripts/analyze_transport_proxies.py`、`scripts/eval_cycle_proxy.py`、`scripts/eval_transport_correlation.py` |

## 7. 注意点(避免踩坑)

- 描述子 npz 需按 merged_manifest 顺序重排(load_lesion_descriptors 已处理);
- negative-control / GT-lesion 的 mode_key 带后缀(`__gt_lesion`、`__lesion_negcontrol_*`),
  eval 时 --mode 必须用全名;
- validation eval 目录名带 `_validation` 后缀(eval 脚本自动);
- cycle 代理在 100 条 b1 上算过(ρ 负),更大规模/不同定义(如 multi-seed 分歧)未测;
- base 冻结、canvas 256 是评估口径基线;lora_p491_e20 是已训练的 checkpoint,
  本轮未在 lesion 系列上复测(可作后续加分项)。
