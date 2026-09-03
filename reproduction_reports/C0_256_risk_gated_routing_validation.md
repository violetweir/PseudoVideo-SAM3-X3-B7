# C0-256 风险门控换路：validation target-disjoint OOF 反事实实验

> 生成时间：2026-08-25T15:51:48+08:00  
> 仓库：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7`  
> 只使用 frozen validation，100 targets 各一条 baseline vs q_return 决策；无 GPU、无 test、无 SAM3 重传播。

## 1. 研究问题与先验冻结边界

固定上一轮 `R_base=argmax(path_mean_similarity)` 与 `R_return=argmax(q_return)`；不重新进行 14 选 1，不修改候选、KNN、B7 或 teacher。模型唯一决策是：接受现有 q_return 换路，还是回退到现有 KNN mean baseline。

## 2. 实际输入文件与 SHA256

| 输入角色 | validation 文件 | SHA256 |
|---|---|---|
| `frozen_validation_target_decisions` | `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_propagation_risk_analysis/qreturn_switch_analysis.jsonl` | `188adc3569cbc1348118425b7d5de4118422c26789e545aa5f9816f97947bf3a` |
| `frozen_validation_routing_summary` | `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_propagation_aware_routing/validation_summary.json` | `7c3727962b71ddb1dd65cb88b7eaee1840cf59b29ca28f8923a5497e377690de` |

## 3. 预注册训练标签、12 个特征与固定参数

目标只提供一个决策样本；有益换路 `ΔDice>0` 共 54 个，无益换路 `ΔDice≤0` 共 46 个。由于 `ΔDice≤-0.10` 只有两个 target，不能对五折独立划分稳定训练严重风险标签，因此预先固定训练标签为 **无益换路 `risk=1[ΔDice≤0]`**；严重/灾难错误只做最终评价。

固定的 12 个不含 target GT 的部署侧输入：

- `q_return_candidate`
- `q_multi_candidate`
- `q_model_candidate`
- `delta_q_return`
- `delta_q_multi`
- `delta_q_model`
- `trace_area_max_rel_delta_candidate`
- `delta_trace_area_max_rel_delta`
- `trace_adjacent_dice_mean_candidate`
- `trace_adjacent_dice_last_candidate`
- `path_mean_similarity_candidate`
- `delta_path_mean_similarity`

固定模型 A：standard scaler + logistic regression，C=1，class_weight=balanced。
固定模型 B：decision tree，max_depth=3，min_samples_leaf=10，class_weight=balanced。
冻结规则：OOF predicted risk < 0.5 才允许真实路线切换；0.5 为固定默认概率门槛，没有使用 validation GT 扫阈值，没有手写 q_multi / 面积规则。

## 4. target-disjoint 五折协议

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)`；每 target 只有一条 row，每个 OOF target 在拟合其预测所用模型时均不参与训练。

| 模型 | fold | train targets | OOF targets | target overlap | OOF severe targets |
|---|---:|---:|---:|---:|---:|
| `logistic` | 1 | 80 | 20 | 0 | 1 |
| `logistic` | 2 | 80 | 20 | 0 | 0 |
| `logistic` | 3 | 80 | 20 | 0 | 0 |
| `logistic` | 4 | 80 | 20 | 0 | 0 |
| `logistic` | 5 | 80 | 20 | 0 | 1 |
| `tree` | 1 | 80 | 20 | 0 | 1 |
| `tree` | 2 | 80 | 20 | 0 | 0 |
| `tree` | 3 | 80 | 20 | 0 | 0 |
| `tree` | 4 | 80 | 20 | 0 | 0 |
| `tree` | 5 | 80 | 20 | 0 | 1 |

## 5. 主要终点评价：最终 Selected Dice 与负尾部

| 方法 | Selected Dice | 相对 KNN Δ | 允许换路 | win / tie / loss | 严重退化 | 灾难退化 |
|---|---:|---:|---:|---|---:|---:|
| KNN mean 安全基准 | 0.887591 | 0.000000 | 0 | 0 / 100 / 0 | 0 | 0 |
| q_return 永远接受 | 0.883055 | -0.004536 | 83 | 54 / 18 / 28 | 2 | 2 |
| 风险门控 Logistic | 0.885822 | -0.001768 | 39 | 25 / 61 / 14 | 1 | 1 |
| 风险门控浅层树 | 0.886012 | -0.001579 | 49 | 32 / 52 / 16 | 1 | 1 |
| GT Oracle Gate（仅评价） | 0.891395 | 0.003805 | 54 | 54 / 46 / 0 | 0 | 0 |

GT Oracle Gate 只用于理论上限：当且仅当 target GT 显示 `ΔDice>0` 才换路，不会作为实际门控输入或训练中的候选选择规则。

## 6. 换路收益与负尾部代价

| 方法 | 正确换路 | 错误换路 | 总正收益 | 总负损失 | 最大单次改善 | 最大单次退化 |
|---|---:|---:|---:|---:|---:|---:|
| KNN mean 安全基准 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| q_return 永远接受 | 54 | 29 | 0.380451 | -0.834043 | 0.117177 | -0.514066 |
| 风险门控 Logistic | 25 | 14 | 0.105525 | -0.282367 | 0.015758 | -0.208569 |
| 风险门控浅层树 | 32 | 17 | 0.109391 | -0.267263 | 0.013706 | -0.208569 |
| GT Oracle Gate（仅评价） | 54 | 0 | 0.380451 | 0.000000 | 0.117177 | 0.000000 |

## 7. paired target bootstrap：与冻结 KNN baseline 比较

| 方法 | mean Δ | median Δ | 95% CI | win / tie / loss |
|---|---:|---:|---|---|
| q_return 永远接受 | -0.004536 | 0.000336 | [-0.017142, 0.004023] | 54 / 18 / 28 |
| 风险门控 Logistic | -0.001768 | 0.000000 | [-0.006474, 0.000972] | 25 / 61 / 14 |
| 风险门控浅层树 | -0.001579 | 0.000000 | [-0.006221, 0.001056] | 32 / 52 / 16 |
| GT Oracle Gate（仅评价） | 0.003805 | 0.000336 | [0.001838, 0.006759] | 54 / 46 / 0 |

bootstrap 每次只对 100 个 target 配对重采样，10000 次，seed=2026；从未将 1300 条候选作为独立样本。

## 8. 两个历史灾难 target 的真实 OOF 审计

| target | q_return ΔDice | logistic OOF risk | logistic 换路 | tree OOF risk | tree 换路 |
|---|---:|---:|---|---:|---|
| `kvasir-seg::cju7b1ygu1msd0801hywhy0mc` | -0.514066 | 0.999667 | False | 0.606742 | False |
| `kvasir-seg::cju1ewnoh5z030855vpex9uzt` | -0.208569 | 0.093986 | True | 0.000000 | True |

- logistic 实际拦截灾难：1/2。
- shallow tree 实际拦截灾难：1/2。
- 两个 target 的 OOF 预测均来自不含该 target 的训练 folds；没有任何手写特例。

## 9. 次要诊断：无益换路标签 AUC

| 模型 | OOF ROC-AUC | OOF PR-AUC |
|---|---:|---:|
| `logistic` | 0.542271 | 0.499587 |
| `tree` | 0.618760 | 0.637609 |

上述 AUC 只是次要诊断；是否继续以 Selected Dice、catastrophic cases 和 paired bootstrap 为准。

## 10. 结论与唯一下一步

实验分类：`C_catastrophic_failures_not_reliably_prevented`。

预注册 target-level OOF gate 未能可靠拦截全部已知灾难切换；现有 q_return/q_multi/q_model/轨迹/形态信息不支持安全门控。

唯一建议：暂停继续堆叠风险门控模型，改为研究新的 validation-only 语义漂移检测信号；不读取 test。

本轮没有读取或运行 test，没有 GPU/SAM3 propagation，没有重新生成 KNN/路线，没有修改 B7、训练 SAM3/Student 或训练神经网络；到 validation OOF 实验结束即停止。
