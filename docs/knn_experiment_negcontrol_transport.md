# Negative Controls + Transport-Aware Proxies(机制收尾实验)

> 创建:2026-08-16。承接 `knn_experiment_lesion_mechanism.md`:① 用真正的
> negative controls 解释 0.8863 为什么高;② 在 validation 上测试 5 个
> 无监督"传播质量代理"能否预测 route GT Dice(目标 ρ:0.4~0.6)。

## 组1:Negative Controls(把 0.8863 拆开)

构造与病灶**确定不重叠/打乱**的 mask 描述子(test,base @256):

| mask 变体 | direct | b1 | b2 | b6 |
|---|---:|---:|---:|---:|
| 伪 mask(病灶) | **0.8863** | 0.8921 | 0.8871 | 0.8836 |
| GT mask | 0.8829 | 0.8885 | 0.8940 | 0.9028 |
| fixed crop(固定中心,非病灶) | 0.8464 | 0.8703 | 0.8757 | 0.8699 |
| random non-overlap(不与病灶重叠) | 0.8584 | 0.8848 | 0.8727 | 0.8731 |
| shuffled mask(像素打乱) | 0.8078 | 0.8495 | 0.8818 | 0.8785 |
| F_global(整图基线) | 0.7698 | 0.8020 | 0.8452 | 0.8550 |

**结论(0.8863 为什么高,两层分解):**
1. **主因 = 空间聚焦(mask 圈选过滤背景)**:即使圈的是**固定中心/随机
   非病灶区域**,direct 也有 0.846~0.858,比整图 0.770 高 **+0.08~0.09**;
   shuffled(圈内 token 来自全图,等价下采样)回落到 0.808,证明收益来自
   **空间局部性**而非 token 数量;
2. **次因 = 病灶定位**:在聚焦基础上对准病灶,再 +0.028(0.886 vs 0.858);
   这也解释了 GT 不比伪 mask 好(两者都聚焦病灶,定位差只值 1 个点)。

## 组2:Transport-Aware Proxies(validation,无训练)

5 个无监督代理 × 700 条 validation 路线(lesion KNN):

| proxy | 定义 | ρ vs GT Dice | p | proxy-oracle |
|---|---:|---:|---:|---:|
| q_model | SAM3 传播自报置信度 | 0.029 | 0.446 | 0.8409 |
| q_multi | forward_candidate_count | 0.130 | 0.0006 | 0.8495 |
| path_bottleneck | 路径最小边相似度 | **0.138** | 0.0003 | 0.8452 |
| path_mean | 路径平均边相似度 | 0.109 | 0.004 | 0.8377 |
| cycle | 往返 mask 稳定性(T→B→A) | **−0.090** | 0.372 | 0.8415 |
| (对照)ρ(F_lesion, Q_GT) 见 E5 | | 0.277 | | |
| per-target 平均 / GT-oracle | | | | 0.8435 / 0.8972 |

**结论:朴素传播代理全部失败。**
- 最强 path_bottleneck 也只有 ρ=0.138,**远低于 lesion 相似度的 0.277,
  更达不到 0.4/0.6 的目标**;
- SAM3 自报置信度(q_model)几乎与质量无关(0.029);
- **cycle(往返一致性)是负相关**(−0.09,不显著)——朴素往返传播不能
  指示传播质量(方向不对称 + 100 条 b1 小样本);
- proxy-oracle(0.84~0.85)≈ per-target 平均(0.8435),即用这些代理
  "选最优路线"和随机选没有区别;与 GT-oracle(0.897)的 5 个点差距
  完全拿不到。

## 对论文方向的结论(收窄)

1. **KNN/lesion 检索已吃到"相似度"能提供的全部收益**(E5 ρ=0.277 就是
   相似度的天花板);朴素无监督代理连相似度都不如;
2. **"无 GT 判断传播路径可靠"这个缺口真实存在,且比预想更难**:
   SAM3 自报置信度、多候选数、路径瓶颈、往返一致性都不行——需要
   ① 更聪明的传播质量估计(如 multi-seed 分歧、mask 在帧间的
   consistency 而非往返),或 ② 小规模有监督(validation 校准一个小
   回归/排序头),或 ③ 直接用 per-target best 信号做自适应桥长
   (数据上 +0.05 的空间是实的);
3. 下一步建议(供讨论):**先做"自适应桥长/路线选择"的监督或
   半监督校准**(validation 上有 GT,校准量小),而不是继续堆
   无监督代理。

## 产物

- 组1 eval:`.../stage1_feature_knn_sam3enc_s1008_lesion_knn/sam3enc_lesion__lesion_negcontrol_{fixed_crop,random_nonoverlap,shuffled}/`
- 组2:`work/kvasir_1pct_anchors/transport_proxies_v1.json`(4 代理)、
  `transport_proxies_v2_cycle.json`(5 代理)、`e5_cycle_proxy.json`(cycle 明细)
