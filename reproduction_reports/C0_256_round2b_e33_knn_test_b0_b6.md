# Round-2B：SAM3-e33 KNN 与 SAM3-e33 teacher 的 Test b0-b6 对照

> 更新时间：2026-08-25 14:42:30  
> 固定传播 checkpoint = SAM3-e33；唯一变化 = KNN topology 从 SAM3-base@256 切换到 SAM3-e33@256。  
> 只评估 test b0-b6，不运行 B7、validation、base teacher、train propagation 或任何训练。

## target 与 patch 完整对照

| Bridge | base KNN target | e33 KNN target | target 变化 | base KNN patch | e33 KNN patch | patch 变化 |
|---|---:|---:|---:|---:|---:|---:|
| b0 | 0.796551 | 0.873091 | +0.076540 | 0.828143 | 0.864121 | +0.035979 |
| b1 | 0.809615 | 0.853760 | +0.044144 | 0.840549 | 0.813613 | -0.026936 |
| b2 | 0.875972 | 0.858138 | -0.017835 | 0.856388 | 0.868473 | +0.012084 |
| b3 | 0.891441 | 0.855363 | -0.036078 | 0.878680 | 0.867232 | -0.011449 |
| b4 | 0.904009 | 0.851389 | -0.052620 | 0.885240 | 0.889696 | +0.004457 |
| b5 | 0.897443 | 0.883953 | -0.013491 | 0.893036 | 0.883861 | -0.009176 |
| b6 | 0.899980 | 0.890895 | -0.009085 | 0.908151 | 0.883148 | -0.025003 |

## 两模式 combined

| Bridge | base KNN + e33 teacher | e33 KNN + e33 teacher | Delta |
|---|---:|---:|---:|
| b0 | 0.812347 | 0.868606 | +0.056259 |
| b1 | 0.825082 | 0.833686 | +0.008604 |
| b2 | 0.866180 | 0.863305 | -0.002875 |
| b3 | 0.885061 | 0.861297 | -0.023763 |
| b4 | 0.894624 | 0.870543 | -0.024082 |
| b5 | 0.895240 | 0.883907 | -0.011333 |
| b6 | 0.904065 | 0.887022 | -0.017044 |

## target_pooling

| Bridge | base KNN + e33 teacher | e33 KNN + e33 teacher | Delta |
|---|---:|---:|---:|
| b0 | 0.796551 | 0.873091 | +0.076540 |
| b1 | 0.809615 | 0.853760 | +0.044144 |
| b2 | 0.875972 | 0.858138 | -0.017835 |
| b3 | 0.891441 | 0.855363 | -0.036078 |
| b4 | 0.904009 | 0.851389 | -0.052620 |
| b5 | 0.897443 | 0.883953 | -0.013491 |
| b6 | 0.899980 | 0.890895 | -0.009085 |

## patch_correspondence

| Bridge | base KNN + e33 teacher | e33 KNN + e33 teacher | Delta |
|---|---:|---:|---:|
| b0 | 0.828143 | 0.864121 | +0.035979 |
| b1 | 0.840549 | 0.813613 | -0.026936 |
| b2 | 0.856388 | 0.868473 | +0.012084 |
| b3 | 0.878680 | 0.867232 | -0.011449 |
| b4 | 0.885240 | 0.889696 | +0.004457 |
| b5 | 0.893036 | 0.883861 | -0.009176 |
| b6 | 0.908151 | 0.883148 | -0.025003 |

## 复现命令

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
DEVICE=0 bash scripts/run_c0_256_round2b_e33_test_bridges_only.sh
```

- 旧 topology 指标：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_round2a_fixed_knn_e33/two_mode_b0_b6_test.json`
- e33 topology 指标：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_round2b_e33_topology_factorial/teachers/e33/two_mode_b0_b6_test.json`
- 对照 JSON：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_round2b_e33_topology_factorial/e33_test_priority/base_knn_vs_e33_knn_test_b0_b6.json`
