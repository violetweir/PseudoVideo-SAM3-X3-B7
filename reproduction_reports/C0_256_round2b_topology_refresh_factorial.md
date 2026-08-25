# C0-256 Round-2B：SAM3-e33 KNN Topology Refresh 2×2 因子实验

> 实验状态：已定义，流水线准备启动。  
> 服务器：`violet@222.31.141.50`。  
> 实验根目录：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_round2b_e33_topology_factorial/`。

## 1. 实验问题

Round-2A 固定 `G0 = KNN(F_SAM3-base)`，只把 propagation teacher 从 SAM3-base 换成 SAM3-e33。因此不能判断 e33 视觉编码器是否也改善了 KNN topology。

Round-2B 重建 `G1 = KNN(F_SAM3-e33)`，但保持输入分辨率 256×256、8 个 anchors、patch-mean descriptor、beam width 32、两种 route mode 和 b0-b6 范围不变。

| 因子组合 | KNN topology | Propagation teacher | 目的 |
|---|---|---|---|
| G0_Tbase | SAM3-base | SAM3-base | 原始 SAM3-KNN baseline |
| G0_Te33 | SAM3-base | SAM3-e33 | 复用 Round-2A，测 teacher-only |
| G1_Tbase | SAM3-e33 | SAM3-base | 测 topology-only |
| G1_Te33 | SAM3-e33 | SAM3-e33 | Round-2B 完整组合 |

所有 B7 均固定使用同一个第一轮 X3 validation-best prediction，避免引入 X4/student 变化。

## 2. 首先检查 topology 是否真的改变

1. 从已合并的 SAM3-e33 video checkpoint 提取 1000 张图的 256×256 vision-trunk descriptor。
2. 报告 base/e33 descriptor cosine、L2 差异和 anchor prototype 差异。
3. 对 train/validation/test query 计算相对 train bridge 候选池的 top-1、top-5、top-10、top-32 邻居重叠率。
4. 仅重建 validation/test 的 target-pooling 与 patch-correspondence b0-b6 route。
5. 报告实际 route、anchor 和 bridge sequence 的变更数。

训练集图像在这里仅作为 KNN 的 bridge 候选库使用，**不执行 train propagation**。

### 2.1 已完成的 e33 feature audit

使用上一轮已选定的 e33 merged video checkpoint，已实际提取全部 `1000 × 1024` 个 256×256 trunk descriptors：

| 指标 | 结果 |
|---|---:|
| base/e33 同图 descriptor cosine，mean | 0.5864504397337890 |
| base/e33 同图 descriptor cosine，median | 0.5769403610865601 |
| base/e33 同图 descriptor cosine，min | 0.3739515407303054 |
| descriptor mean L2 delta | 0.9066411828763438 |
| descriptor max absolute delta | 0.7566581368446350 |
| 8 个 anchor prototype 平均 cosine | 0.6292819207003604 |

相对 train bridge 候选库的邻居重叠率：

| Query split | Top-1 overlap | Top-5 overlap | Top-10 overlap | Top-32 overlap |
|---|---:|---:|---:|---:|
| train | 0.1075 | 0.1175 | 0.148125 | 0.199140625 |
| validation | 0.0700 | 0.0960 | 0.116000 | 0.182500000 |
| test | 0.0700 | 0.1120 | 0.132000 | 0.195937500 |

当前只能据此确认 e33 encoder **明显重塑了 topology**；它是否改善实际 propagation 或 B7，必须等待固定 teacher 的 2×2 验证结果，不能由邻居变化幅度直接推断。

### 2.2 已完成的实际 route audit

validation/test 共四组新路线已经全部完成，每组恰好 100 个 target × b0-b6 = 700 条：

| Split | Mode | Changed routes / 700 | Changed anchor / 700 | Mean bridge Jaccard |
|---|---|---:|---:|---:|
| validation | target-pooling | 676 / 700 | 547 / 700 | 0.177137 |
| validation | patch-correspondence | 660 / 700 | 425 / 700 | 0.168122 |
| test | target-pooling | 663 / 700 | 493 / 700 | 0.184377 |
| test | patch-correspondence | 657 / 700 | 434 / 700 | 0.165940 |

因此 e33 topology 在四组路线中改变了约 94%—97% 的实际传播路径。当前已进入 `G1_Tbase` 的 validation propagation；后续自动执行 `G1_Te33` validation，再运行两组 test。

## 3. 评价协议与因子分解

- validation/test 各 100 个 target。
- 两模式分别报告 b0-b6，每个 split 每模式 700 条路线。
- 合并两模式后，固定 X3-best，报告真实 B7 Dice、oracle Dice 与 oracle gap。
- 历史 `G0_Tbase` 与 `G0_Te33` 的传播产物直接复用。
- 完全相同的 checkpoint 和 route_id 允许复用旧 inference，新的路线才运行 propagation。
- 优先完成 validation 的两组新 teacher 对照，再运行 test。

已经用完全一致的 frozen X3+B7 脚本重新核对 `G0` 两组历史对照：

| Topology × teacher | Validation B7 Dice | Test B7 Dice |
|---|---:|---:|
| G0_Tbase | 0.8338662059873969 | 0.8954317676013283 |
| G0_Te33 | 0.8838539036343016 | 0.9063803067687748 |
| G1_Tbase | running | pending |
| G1_Te33 | pending | pending |

```text
teacher effect @ G0 = G0_Te33 - G0_Tbase
topology effect @ Tbase = G1_Tbase - G0_Tbase
topology effect @ Te33 = G1_Te33 - G0_Te33
interaction = G1_Te33 - G1_Tbase - G0_Te33 + G0_Tbase
```

## 4. 执行命令

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
DEVICE=0 bash scripts/run_c0_256_round2b_topology_refresh_factorial.sh
```

实验完成后，流水线自动把本文替换为包含完整特征漂移、topology audit、validation/test 2×2 因子表及两模式 b0-b6 明细的最终报告。

本轮边界：**不进行 train propagation、不生成新的 pseudo pool、不训练新的 X4 或 SAM3**。
