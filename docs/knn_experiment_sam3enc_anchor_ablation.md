# SAM3-enc @1008 标注数量消融(8 → 6/4/3/2/1 anchor)

> 创建:2026-08-15。目的:固定 KNN 变体,只减少初始 GT anchor 数量,
> 观察 Dice 如何变化,回答"最低需要多少张初始标注图像"。
> 本次只跑 base checkpoint(用户决定不跑 lora)。

## 0. 固定协议

| 项 | 值 |
|---|---|
| 数据集 | Kvasir-SEG 快照(同前,train 800 / val 100 / test 100) |
| KNN 变体 | `sam3enc_anchor_conditioned_target_pooling` + `--knn-feature cond`(固定,不做其它变体) |
| 特征 | SAM3-enc @**1008** 原生(batch 1;共享 patch token 缓存 `sam3_base_s1008_patches.npz` 1000×5184×1024 fp16,每个 N 只重算 per-anchor 量) |
| anchor 选择 | seed=42,`numpy default_rng(42).permutation` 对 merged_id 排序后的 8 anchor 洗牌,取前 N(**嵌套**:8⊃6⊃4⊃3⊃2⊃1) |
| 评估 | 冻结 base `sam3.pt`,`--canvas 256`,`eval_base_no_ft_b7_forward`,test 100 targets × b0–b6 = 700 条/变体 |
| 输出根目录 | `work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_anchor_ablation/n{N}/` |

## 1. anchor 选择记录(可复现,seed=42)

`full_ordered_ids`(permutation 后顺序,取前 N):

1. `kvasir-seg::cju3v72v5h1qz0818fggilwtq` ← n=1 唯一的 anchor
2. `kvasir-seg::cju5cu8qkb84x08186jwo8yin`
3. `kvasir-seg::cju34ymm8d6700799uop0cw33`
4. `kvasir-seg::cjyzkpsbjdsjq07211dfi4sru`
5. `kvasir-seg::cju8bj2ssrmlm0871gc2ug2rs`
6. `kvasir-seg::cju2pmhtr17a00855cvpelzb0`
7. `kvasir-seg::cju7aklv31h4309871m29l4e7`
8. `kvasir-seg::cju1c4fcu40hl07992b8gj0c8`

每 N 的完整 `anchor_selection.json`(含选中/排除 id、mask 路径、时间戳)与
`protocol/support_manifest_n{N}.jsonl` 在各自 `n{N}/` 目录。

## 2. 结果(base @256,test forward-only Dice)

| n | direct | b1 | b2 | b3 | b4 | b5 | b6 | mean(b1–b6) | vs n=8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8(§9 参考) | 0.8205 | 0.8811 | 0.8823 | 0.8617 | 0.8615 | 0.8822 | 0.8639 | 0.8721 | — |
| 6 | 0.8198 | 0.8369 | 0.8539 | 0.8377 | 0.8518 | 0.8391 | 0.8647 | 0.8474 | −0.025 |
| 4 | 0.7757 | 0.8008 | 0.8352 | 0.7979 | 0.8312 | 0.8239 | 0.8411 | 0.8217 | −0.050 |
| 3 | 0.8480 | 0.8594 | 0.8370 | 0.8416 | 0.7686 | 0.7970 | 0.8402 | 0.8240 | −0.048 |
| 2 | 0.7089 | 0.7757 | 0.7870 | 0.7820 | 0.8285 | 0.7420 | 0.7979 | 0.7855 | −0.087 |
| 1 | 0.8517 | 0.8731 | 0.8794 | 0.8571 | 0.8401 | 0.8537 | 0.8541 | 0.8596 | −0.013 |

### 2.1 要点

- **非单调**:8→6(−0.025)→4(−0.050)→3(−0.048)→2(−0.087)→1(−0.013);
  n=1 几乎追平 n=8(mean b1–b6 差 0.013,direct 反而最高 0.8517);
- **n=2 最差**(direct 0.7089,mean 0.7855):seed=42 嵌套选择下,第 2 个
  anchor(`cju5cu8qkb84x08186jwo8yin`)是拖累项;
- **结论:anchor 数量不是主因,个体代表性才是**。单 anchor 足够好时
  (n=1)接近 8 个;掺入差 anchor 时(n=2)反而崩。
- **待补验证**:多 seed 重复(区分数量效应 vs 个体质量效应);
  以及"如何自动挑最优单 anchor"(如按 anchor 与 train 分布的相似度排序)。

## 3. 多 seed 验证(2026-08-16 完成,用户选定)

固定协议同上,seed ∈ {42, 7, 123, 0, 8, 31},全部 n=1(base @256)。

| seed | 单 anchor(末 18 位) | direct | mean(b1–b6) | vs n=8 |
|---|---:|---:|---:|---:|
| 42 | v5h1qz0818fggilwtq | 0.8517 | 0.8596 | −0.013 |
| 8 | v5h1qz0818fggilwtq | 0.8517 | 0.8596 | −0.013 |
| 7 | cu40hl07992b8gj0c8 | 0.7935 | 0.8527 | −0.019 |
| 123 | cu40hl07992b8gj0c8 | 0.7935 | 0.8527 | −0.019 |
| 0 | m8d6700799uop0cw33 | 0.8480 | 0.8240 | −0.048 |
| 31 | qkb84x08186jwo8yin | 0.7089 | 0.7855 | −0.087 |
| **n=1 汇总(4 个唯一 anchor)** | | | **0.8390 ± 0.027** | **−0.033(保持率 96.2%)** |

### 3.1 多 seed 结论(修正 2.1)

- 单 anchor 表现横跨 −0.013 ~ −0.087,**由 anchor 个体质量决定**;
  "1 张就够"修正为"**1 张好 anchor 就够(98.5%),选错 anchor 掉 8.7%**";
- seed31 的单 anchor(qkb84x…)正是 seed42 n=2 的"拖后腿"anchor——
  差 anchor 稳定地差,单独用与混入都拖累(内部一致性证据);
- 数量只起平滑作用(n=6 损失 2.5%,稀释坏 anchor),收益递减;
- **actionable 结论:anchor 选择策略 > anchor 数量**——自动挑最优单
  anchor(如 anchor 与 train 分布的相似度/覆盖度排序)比堆数量划算。

## 3. 产物

- `n{N}/anchor_selection.json` — 每次选择的完整记录(复现依据)
- `n{N}/protocol/support_manifest_n{N}.jsonl` — 该 N 的 anchor manifest
- `n{N}/features/sam3_base_s1008_features.npz` — 重算的 per-anchor 特征
- `n{N}/sam3enc_anchor_conditioned_target_pooling__knn_cond/test_pool0_stage1/routes.jsonl`
- `n{N}/sam3enc_anchor_conditioned_target_pooling__knn_cond/eval_base_no_ft_b7_forward/route_results.jsonl`
