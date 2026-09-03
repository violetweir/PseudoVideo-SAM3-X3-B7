# C0-256 传播风险诊断：定位“高一致性但错误传播”的失败机制

> 生成时间：2026-08-25T15:39:27+08:00  
> 仓库：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7`  
> 范围：仅 frozen validation、仅 CPU 离线诊断；不读取 test、不运行 propagation、不修改 B7。

## 1. 背景与上一轮冻结结论

此前 q_return 的 candidate-level Spearman 0.216292 高于 KNN mean 的 0.043077，但 q_return Top-1 Dice 0.883055 低于冻结 KNN mean 基准 0.887591。因此本轮不再测试 q_return 直接替代 KNN，而诊断高一致性错误传播能否被其他无监督信号识别。

## 2. 数据、输入 SHA256 与冻结协议

| 输入角色 | validation 文件 | SHA256 |
|---|---|---|
| `prior_validation_candidates` | `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_propagation_aware_routing/validation_candidates.jsonl` | `9555b3ef525b024af470def044290c16dbb3bf9ea1b0b0cfb9183000a257d65e` |
| `prior_validation_per_target` | `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_propagation_aware_routing/validation_per_target.jsonl` | `5d394c0006944c9235748136b238a2ac83c7dd6c45b7cc2290112c48cfa8f3a8` |
| `prior_validation_summary` | `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_propagation_aware_routing/validation_summary.json` | `7c3727962b71ddb1dd65cb88b7eaee1840cf59b29ca28f8923a5497e377690de` |
| `historical_validation_b7_full_trace_audit` | `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_round2a_fixed_knn_e33/b7_calibration/validation_all_candidates.jsonl` | `b26e9dd13687f5fbe14e6a0974e032852d8a483d6bfcc6003f988d8583fb1dd3` |

100 targets × 14 冻结候选 = 1400 candidates。安全基准固定为上一轮相同 tie-break 的 `argmax(path_mean_similarity)`，绝不重新选择 KNN baseline。

可用数值字段：`["b7", "bridge_count", "cycle_candidate_count", "cycle_sam_score", "final_candidate_count", "forward_sam_score", "path_bottleneck_similarity", "path_mean_similarity", "q_cycle", "q_model", "q_multi", "q_return", "trace_adjacent_dice_last", "trace_adjacent_dice_mean", "trace_adjacent_dice_min", "trace_area_final", "trace_area_max", "trace_area_max_rel_delta", "trace_area_min", "trace_bbox_h_max_rel_delta", "trace_bbox_w_max_rel_delta", "trace_candidate_count_final", "trace_candidate_count_max", "trace_centroid_max_step", "trace_component_final", "trace_component_max", "trace_empty_count", "trace_sam_score_final", "trace_sam_score_mean", "trace_sam_score_min"]`。
可选面积字段状态：`{"forward_mask_area": "missing_not_recomputed", "mask_area": "missing_not_recomputed", "mask_area_change": "derived_without_gt_from_trace_area_max_rel_delta", "mask_area_ratio": "derived_without_gt_from_trace_area_final", "returned_mask_area": "missing_not_recomputed"}`。
`mask_area_ratio` 仅由已有 `trace_area_final` 派生；`mask_area_change` 仅由已有 `trace_area_max_rel_delta` 派生；返回 mask 面积未保存，因此不臆造也不重跑 GPU。
所有 `gt_*` 原始字段都排除在模型/风险特征之外：`["gt_dice_evaluation_only", "gt_gt_area_ratio_evaluation_only", "gt_iou_evaluation_only", "gt_precision_evaluation_only", "gt_pred_area_ratio_evaluation_only", "gt_recall_evaluation_only"]`。

## 3. 评价性风险定义

- 明确改善：`ΔDice > 0.01`。
- 基本安全：`-0.01 ≤ ΔDice ≤ 0.01`。
- 中度退化：`-0.10 < ΔDice < -0.01`。
- 严重退化：`ΔDice ≤ -0.10`；灾难性退化：`ΔDice ≤ -0.20`。
- 这些 GT 阈值仅用于回顾性标签与机制评价，不是部署阈值，也不会访问 test。

## 4. q_return 失败集合严格复现

- 改善 / 持平 / 退化：54 / 18 / 28。
- 平均 ΔDice：-0.004536；中位 ΔDice：0.000336。
- 真实 physical route switch：83；相同 route：17。
- q_return 所选 target 中严重退化：2；灾难性退化：2。
- 全部非基准 physical switch candidates：1300；严重 96，灾难 90。
- 决策相关 competitive 子集 `q_return(candidate) >= q_return(baseline)`：470 候选 / 83 targets；严重 27 条，分布在 8 个独立 target；灾难 23 条。
- 高 q_return 上四分位 physical switches：325；严重 7。

## 5. 四组 q_multi / q_model / B7 / q_return 分布

### 所有候选换路

| 组别 | N | q_return mean / median | q_multi mean / median | q_model mean / median | b7 mean / median |
|---|---:|---:|---:|---:|---:|
| 明确改善 ΔDice > 0.01 | 101 | 0.936730 / 0.966114 | 0.848060 / 0.911583 | 0.796849 / 0.877798 | 0.821402 / 0.872915 |
| 基本安全 -0.01 ≤ ΔDice ≤ 0.01 | 974 | 0.955632 / 0.968239 | 0.964188 / 0.994157 | 0.904627 / 0.966896 | 0.922379 / 0.971004 |
| 中度退化 -0.10 < ΔDice < -0.01 | 129 | 0.938258 / 0.963038 | 0.860071 / 0.911473 | 0.727099 / 0.853445 | 0.774589 / 0.869502 |
| 严重退化 ΔDice ≤ -0.10 | 96 | 0.410957 / 0.000000 | 0.390062 / 0.307692 | 0.264906 / 0.000000 | 0.306085 / 0.000157 |

### q_return 实际切换

| 组别 | N | q_return mean / median | q_multi mean / median | q_model mean / median | b7 mean / median |
|---|---:|---:|---:|---:|---:|
| 明确改善 ΔDice > 0.01 | 10 | 0.973143 / 0.971676 | 0.832695 / 0.933776 | 0.801320 / 0.881383 | 0.829592 / 0.883879 |
| 基本安全 -0.01 ≤ ΔDice ≤ 0.01 | 68 | 0.959598 / 0.973409 | 0.951746 / 0.990639 | 0.889837 / 0.965274 | 0.907087 / 0.967105 |
| 中度退化 -0.10 < ΔDice < -0.01 | 3 | 0.971110 / 0.968889 | 0.869680 / 0.894182 | 0.498484 / 0.460307 | 0.658554 / 0.654380 |
| 严重退化 ΔDice ≤ -0.10 | 2 | 0.971500 / 0.971500 | 0.801024 / 0.801024 | 0.809764 / 0.809764 | 0.831626 / 0.831626 |

### 决策相关 competitive 候选

| 组别 | N | q_return mean / median | q_multi mean / median | q_model mean / median | b7 mean / median |
|---|---:|---:|---:|---:|---:|
| 明确改善 ΔDice > 0.01 | 37 | 0.946003 / 0.970027 | 0.825747 / 0.908618 | 0.784689 / 0.881473 | 0.803482 / 0.867391 |
| 基本安全 -0.01 ≤ ΔDice ≤ 0.01 | 390 | 0.950677 / 0.972275 | 0.964869 / 0.994538 | 0.907927 / 0.965659 | 0.920446 / 0.968596 |
| 中度退化 -0.10 < ΔDice < -0.01 | 16 | 0.848765 / 0.968187 | 0.848281 / 0.898711 | 0.667318 / 0.832387 | 0.660994 / 0.750490 |
| 严重退化 ΔDice ≤ -0.10 | 27 | 0.932288 / 0.967213 | 0.789643 / 0.890683 | 0.634051 / 0.561408 | 0.741102 / 0.756451 |

完整 mean / median / std / Q25 / Q75 / min / max 和相对 baseline 变化见 `risk_group_summary.json`。

## 6. q_return 高但 q_multi 低：无监督四分位二维诊断

所有阈值只是 candidate 特征自身的无监督四分位，不看 GT，也不会用作部署规则：q_return Q75=0.971261，q_multi Q25=0.915754，q_multi Q75=0.995929。

| 描述区域 | N | 严重退化 | 严重比例 | 平均 ΔDice |
|---|---:|---:|---:|---:|
| `high_return_low_peer_agreement` | 31 | 7 | 0.225806 | -0.108535 |
| `high_return_high_peer_agreement` | 145 | 0 | 0.000000 | -0.000095 |

## 7. Student disagreement 与历史 B7 的互补性

- `q_multi` 单变量描述 ROC-AUC=0.958351，PR-AUC=0.766221。
- `q_model` 单变量描述 ROC-AUC=0.946922，PR-AUC=0.646402。
- `q_return-q_multi` ROC-AUC=0.508418；`q_return-q_model` ROC-AUC=0.605568。
- `Δq_multi` ROC-AUC=0.759517；`Δq_model` ROC-AUC=0.693002。
- 去除对 q_return 直接无竞争力的候选后：competitive `q_multi` ROC-AUC=0.914639，competitive `q_model` ROC-AUC=0.875387。
B7 全部来自上一轮冻结 audit，不重算、调指数或选择新权重；其失败和成功均只做事后机制解释。

## 8. 局部传播轨迹与形态漂移

`trace_adjacent_dice_min` 描述 ROC-AUC=0.655008，PR-AUC=0.113111。同时审计 `trace_adjacent_dice_mean`、面积跳变、bbox 跳变、质心位移、空 mask 和 SAM score；不存在的 returned-mask 面积直接标为 missing。

## 9. bridge depth 风险：q_return 实际选择

| bridge | N | 平均 Δ | 中位 Δ | 严重退化 | 灾难退化 | 最差 Δ |
|---|---:|---:|---:|---:|---:|---:|
| b0 | 16 | -0.000898 | 0.000779 | 0 (0.000) | 0 (0.000) | -0.023963 |
| b2 | 6 | -0.036011 | -0.002310 | 1 (0.167) | 1 (0.167) | -0.208569 |
| b3 | 8 | 0.020440 | 0.005223 | 0 (0.000) | 0 (0.000) | -0.004008 |
| b4 | 18 | 0.003453 | 0.000897 | 0 (0.000) | 0 (0.000) | -0.003509 |
| b5 | 27 | 0.002272 | 0.000467 | 0 (0.000) | 0 (0.000) | -0.014585 |
| b6 | 25 | -0.020407 | 0.000000 | 1 (0.040) | 1 (0.040) | -0.514066 |

不依据任何 depth 结果删除 b0/b6 或缩小候选空间。

## 10. route mode 风险：q_return 实际 physical switch

| mode | switches | 改善 / 持平 / 退化 | 严重 | 平均 Δ | 最差 Δ | q_multi | q_model | 局部最小 Dice |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `sam3enc_anchor_conditioned_patch_correspondence` | 39 | 24 / 0 / 15 | 1 | -0.011594 | -0.514066 | 0.958250 | 0.889072 | 0.072448 |
| `sam3enc_anchor_conditioned_target_pooling` | 44 | 30 / 1 / 13 | 1 | -0.000032 | -0.208569 | 0.906478 | 0.840076 | 0.154955 |

## 11. 单一无监督特征风险判别

全部 physical switch candidates 的严重退化先验比例=0.073846。方向按 validation 描述性比较展示，不构成冻结阈值或 test 方法。

| 风险特征 | 风险方向（仅描述） | ROC-AUC | PR-AUC | 严重样本 / N |
|---|---|---:|---:|---:|
| `peer_disagreement` | `higher_values_indicate_risk` | 0.958351 | 0.766221 | 96 / 1300 |
| `q_multi` | `lower_values_indicate_risk` | 0.958351 | 0.766221 | 96 / 1300 |
| `b7` | `lower_values_indicate_risk` | 0.957416 | 0.751853 | 96 / 1300 |
| `q_model` | `lower_values_indicate_risk` | 0.946922 | 0.646402 | 96 / 1300 |
| `student_disagreement` | `higher_values_indicate_risk` | 0.946922 | 0.646402 | 96 / 1300 |
| `delta_mask_area_ratio` | `lower_values_indicate_risk` | 0.933001 | 0.879254 | 96 / 1300 |
| `delta_trace_area_final` | `lower_values_indicate_risk` | 0.933001 | 0.879254 | 96 / 1300 |
| `baseline_peer_disagreement` | `higher_values_indicate_risk` | 0.918734 | 0.452746 | 96 / 1300 |
| `baseline_q_multi` | `lower_values_indicate_risk` | 0.918734 | 0.452746 | 96 / 1300 |
| `baseline_q_return_minus_q_multi` | `higher_values_indicate_risk` | 0.909962 | 0.450723 | 96 / 1300 |
| `baseline_abs_q_return_minus_q_multi` | `higher_values_indicate_risk` | 0.895903 | 0.360630 | 96 / 1300 |
| `mask_area_ratio` | `lower_values_indicate_risk` | 0.882471 | 0.742419 | 96 / 1300 |
| `trace_area_final` | `lower_values_indicate_risk` | 0.882471 | 0.742419 | 96 / 1300 |
| `baseline_b7` | `lower_values_indicate_risk` | 0.873521 | 0.269943 | 96 / 1300 |
| `delta_trace_area_max` | `lower_values_indicate_risk` | 0.871730 | 0.492188 | 96 / 1300 |

### q_return 至少不低于基准：决策相关候选

| 风险特征 | 风险方向（仅描述） | ROC-AUC | PR-AUC | 严重样本 / N |
|---|---|---:|---:|---:|
| `baseline_peer_disagreement` | `higher_values_indicate_risk` | 0.931736 | 0.527660 | 27 / 470 |
| `baseline_q_multi` | `lower_values_indicate_risk` | 0.931736 | 0.527660 | 27 / 470 |
| `baseline_q_model` | `lower_values_indicate_risk` | 0.930482 | 0.452265 | 27 / 470 |
| `baseline_student_disagreement` | `higher_values_indicate_risk` | 0.930482 | 0.452265 | 27 / 470 |
| `baseline_b7` | `lower_values_indicate_risk` | 0.915601 | 0.333389 | 27 / 470 |
| `peer_disagreement` | `higher_values_indicate_risk` | 0.914639 | 0.301055 | 27 / 470 |
| `q_multi` | `lower_values_indicate_risk` | 0.914639 | 0.301055 | 27 / 470 |
| `baseline_abs_q_return_minus_q_model` | `higher_values_indicate_risk` | 0.914012 | 0.331815 | 27 / 470 |
| `baseline_q_return_minus_q_model` | `higher_values_indicate_risk` | 0.904314 | 0.451753 | 27 / 470 |
| `b7` | `lower_values_indicate_risk` | 0.903269 | 0.257476 | 27 / 470 |
| `baseline_q_return_minus_q_multi` | `higher_values_indicate_risk` | 0.900803 | 0.520064 | 27 / 470 |
| `q_return_minus_q_multi` | `higher_values_indicate_risk` | 0.900426 | 0.242311 | 27 / 470 |

该子集排除了显然无法赢过 q_return 基准的低一致性候选，避免大量空 mask / q_return≈0 的易分类失败人为抬高对真正错误自洽问题的判断。

### 仅 q_return 实际切换的单变量诊断

| 风险特征 | 风险方向（仅描述） | ROC-AUC | PR-AUC | 严重样本 / N |
|---|---|---:|---:|---:|
| `delta_abs_q_return_minus_q_model` | `lower_values_indicate_risk` | 1.000000 | 1.000000 | 2 / 83 |
| `delta_b7` | `higher_values_indicate_risk` | 1.000000 | 1.000000 | 2 / 83 |
| `delta_q_model` | `higher_values_indicate_risk` | 1.000000 | 1.000000 | 2 / 83 |
| `delta_q_return_minus_q_model` | `lower_values_indicate_risk` | 1.000000 | 1.000000 | 2 / 83 |
| `delta_student_disagreement` | `lower_values_indicate_risk` | 1.000000 | 1.000000 | 2 / 83 |
| `delta_cycle_sam_score` | `higher_values_indicate_risk` | 0.969136 | 0.642857 | 2 / 83 |
| `baseline_peer_disagreement` | `higher_values_indicate_risk` | 0.956790 | 0.611111 | 2 / 83 |
| `baseline_q_multi` | `lower_values_indicate_risk` | 0.956790 | 0.611111 | 2 / 83 |
| `baseline_q_return_minus_q_multi` | `higher_values_indicate_risk` | 0.956790 | 0.611111 | 2 / 83 |
| `baseline_abs_q_return_minus_q_multi` | `higher_values_indicate_risk` | 0.944444 | 0.350000 | 2 / 83 |

实际 q_return severe 样本极少，上述子集 AUC 仅用于案例定位，不能作为稳定泛化证据。

## 12. 改善 vs 严重退化：效应量

| 特征 | beneficial mean | dangerous mean | dangerous - beneficial | Mann-Whitney p | Cliff's δ |
|---|---:|---:|---:|---:|---:|
| `delta_mask_area_ratio` | 0.006733 | -0.135251 | -0.141983 | 5.990e-24 | 0.832715 |
| `delta_trace_area_final` | 0.006733 | -0.135251 | -0.141983 | 5.990e-24 | 0.832715 |
| `q_model` | 0.796849 | 0.264906 | -0.531943 | 1.072e-22 | 0.799711 |
| `student_disagreement` | 0.203151 | 0.735094 | 0.531943 | 1.072e-22 | -0.799711 |
| `b7` | 0.821402 | 0.306085 | -0.515317 | 3.150e-22 | 0.799505 |
| `mask_area_ratio` | 0.147497 | 0.028230 | -0.119267 | 2.790e-21 | 0.772896 |
| `trace_area_final` | 0.147497 | 0.028230 | -0.119267 | 2.790e-21 | 0.772896 |
| `peer_disagreement` | 0.151940 | 0.609938 | 0.457999 | 1.495e-19 | -0.745875 |
| `q_multi` | 0.848060 | 0.390062 | -0.457999 | 1.495e-19 | 0.745875 |
| `delta_final_candidate_count` | 0.000000 | -0.572917 | -0.572917 | 4.059e-19 | 0.572917 |
| `delta_trace_candidate_count_final` | 0.000000 | -0.572917 | -0.572917 | 4.059e-19 | 0.572917 |
| `final_candidate_count` | 1.000000 | 0.427083 | -0.572917 | 4.059e-19 | 0.572917 |
| `trace_candidate_count_final` | 1.000000 | 0.427083 | -0.572917 | 4.059e-19 | 0.572917 |
| `delta_trace_empty_count` | 0.000000 | 0.656250 | 0.656250 | 6.183e-19 | -0.572917 |
| `trace_empty_count` | 0.000000 | 0.656250 | 0.656250 | 6.183e-19 | -0.572917 |

## 13. 联合信号：target-level 五折交叉验证

同一 target 的全部候选始终留在同一个 fold；特征只包含 candidate/base 无监督分数、轨迹、bridge、mode 及相对变化。固定超参数，不训练神经网络，不接触 test。

| 训练/评估 cohort 与 CPU 诊断模型 | cohort ROC-AUC | cohort PR-AUC | q_return 实际 switch ROC-AUC | q_return 实际 switch PR-AUC |
|---|---:|---:|---:|---:|
| `all switches / logistic` | 0.789616 | 0.660976 | 0.222222 | 0.022833 |
| `all switches / tree` | 0.906657 | 0.796002 | 0.740741 | 0.512048 |
| `q_return-competitive / logistic` | 0.623861 | 0.099987 | 0.388889 | 0.038364 |
| `q_return-competitive / tree` | 0.719672 | 0.322082 | 0.709877 | 0.512048 |

competitive cohort 仅有 8 个独立严重 target；q_return OOF 子集只有 2 个严重 target。candidate 数量不能冒充独立 target 样本量；任何 OOF 结论均需谨慎。完整系数、树规则、fold target overlap=0 与全部 OOF 指标保存在相应 `risk_cv_*.json`。

## 14. q_return 所选严重失败清单

| target | ΔDice | q_return | q_multi | q_model | B7 | mode | bridge |
|---|---:|---:|---:|---:|---:|---|---:|
| `kvasir-seg::cju7b1ygu1msd0801hywhy0mc` | -0.514066 | 0.977502 | 0.891835 | 0.713472 | 0.830784 | `sam3enc_anchor_conditioned_patch_correspondence` | 6 |
| `kvasir-seg::cju1ewnoh5z030855vpex9uzt` | -0.208569 | 0.965497 | 0.710212 | 0.906057 | 0.832468 | `sam3enc_anchor_conditioned_target_pooling` | 2 |

## 15. 灾难性失败逐例机制诊断

### 灾难案例 1：`kvasir-seg::cju7b1ygu1msd0801hywhy0mc`

- 基准：`sam3enc_anchor_conditioned_target_pooling` / b6 / `7d0f5cbc984949dd11e94a33`；Dice=0.514066。
- q_return 候选：`sam3enc_anchor_conditioned_patch_correspondence` / b6 / `e239f069a3794dbea15eb576`；Dice=0.000000，ΔDice=-0.514066。

| 无监督字段 | 安全基准 | q_return 候选 | 候选 - 基准 |
|---|---:|---:|---:|
| `q_return` | 0.967983 | 0.977502 | 0.009519 |
| `q_multi` | 0.381352 | 0.891835 | 0.510484 |
| `q_model` | 0.497086 | 0.713472 | 0.216385 |
| `b7` | 0.510830 | 0.830784 | 0.319954 |
| `path_mean_similarity` | 0.911368 | 0.813042 | -0.098325 |
| `path_bottleneck_similarity` | 0.828125 | 0.613281 | -0.214844 |
| `forward_sam_score` | 0.980488 | 0.980488 | 0.000000 |
| `trace_adjacent_dice_min` | 0.000000 | 0.000000 | 0.000000 |
| `trace_adjacent_dice_mean` | 0.377509 | 0.273869 | -0.103640 |
| `trace_adjacent_dice_last` | 0.610792 | 0.000000 | -0.610792 |
| `trace_sam_score_min` | 0.980488 | 0.980488 | 0.000000 |
| `trace_sam_score_mean` | 0.980488 | 0.980488 | 0.000000 |
| `trace_area_final` | 0.103500 | 0.021011 | -0.082489 |
| `trace_area_max_rel_delta` | 0.981314 | 1.775542 | 0.794229 |
| `trace_centroid_max_step` | 0.355786 | 0.253781 | -0.102004 |
| `q_return_minus_q_multi` | 0.586631 | 0.085667 | -0.500964 |
| `q_return_minus_q_model` | 0.470897 | 0.264030 | -0.206866 |

- 五折 target-disjoint OOF logistic 风险概率：0.009051；OOF shallow tree 风险概率：1.000000。
- 决策相关 competitive cohort 的 OOF logistic 风险概率：0.059922；OOF shallow tree 风险概率：0.987670。
- 逐例机制诊断：候选间一致性 q_multi 反而相对安全基准提高 0.510484；Student 一致性 q_model 反而相对安全基准提高 0.216385；历史 B7 反而相对安全基准提高 0.319954；局部相邻帧平均 Dice 相对安全基准下降 -0.103640；传播面积最大相对跳变增加 0.794229。

### 灾难案例 2：`kvasir-seg::cju1ewnoh5z030855vpex9uzt`

- 基准：`sam3enc_anchor_conditioned_target_pooling` / b6 / `2b175dc35699f4fdd574fb35`；Dice=0.974958。
- q_return 候选：`sam3enc_anchor_conditioned_target_pooling` / b2 / `139da78e64f3f7f0adc62c2e`；Dice=0.766389，ΔDice=-0.208569。

| 无监督字段 | 安全基准 | q_return 候选 | 候选 - 基准 |
|---|---:|---:|---:|
| `q_return` | 0.963402 | 0.965497 | 0.002095 |
| `q_multi` | 0.775912 | 0.710212 | -0.065700 |
| `q_model` | 0.683097 | 0.906057 | 0.222960 |
| `b7` | 0.769980 | 0.832468 | 0.062488 |
| `path_mean_similarity` | 0.912410 | 0.902982 | -0.009428 |
| `path_bottleneck_similarity` | 0.851562 | 0.851562 | 0.000000 |
| `forward_sam_score` | 0.980488 | 0.980488 | 0.000000 |
| `trace_adjacent_dice_min` | 0.096594 | 0.303333 | 0.206740 |
| `trace_adjacent_dice_mean` | 0.350382 | 0.357157 | 0.006775 |
| `trace_adjacent_dice_last` | 0.351308 | 0.303333 | -0.047975 |
| `trace_sam_score_min` | 0.980488 | 0.980488 | 0.000000 |
| `trace_sam_score_mean` | 0.980488 | 0.980488 | 0.000000 |
| `trace_area_final` | 0.098038 | 0.150955 | 0.052917 |
| `trace_area_max_rel_delta` | 0.981314 | 3.544734 | 2.563420 |
| `trace_centroid_max_step` | 0.470154 | 0.266177 | -0.203977 |
| `q_return_minus_q_multi` | 0.187491 | 0.255286 | 0.067795 |
| `q_return_minus_q_model` | 0.280305 | 0.059441 | -0.220865 |

- 五折 target-disjoint OOF logistic 风险概率：0.000005；OOF shallow tree 风险概率：0.000000。
- 决策相关 competitive cohort 的 OOF logistic 风险概率：0.000000；OOF shallow tree 风险概率：0.000000。
- 逐例机制诊断：候选间一致性 q_multi 相对安全基准下降 -0.065700；Student 一致性 q_model 反而相对安全基准提高 0.222960；历史 B7 反而相对安全基准提高 0.062488；传播面积最大相对跳变增加 2.563420。

## 16. 六个核心问题与主要机制发现

1. 高 q_return + 低 q_multi：低-peer 区严重率 0.225806，高-peer 区 0.000000；q_multi ROC-AUC=0.958351。
2. Student disagreement：q_model ROC-AUC=0.946922，Δq_model ROC-AUC=0.693002；灾难案例中 q_model 相对基准明显下降 0/2。
3. 局部轨迹：trace 最小 adjacent Dice ROC-AUC=0.655008；灾难案例中局部最小值相对基准明显下降 0/2。
4. bridge/mode 集中情况见前述固定空间分层表；任何观察都不删除 mode/depth。
5. 现有信号联合：target-disjoint logistic ROC-AUC=0.789616，tree ROC-AUC=0.906657；competitive cohort logistic/tree ROC-AUC=0.623861/0.719672。
6. 描述性最佳单指标：`peer_disagreement`，ROC-AUC=0.958351，PR-AUC=0.766221；这个选择存在 validation 多重比较乐观偏差，不是可部署规则。

## 17. 是否值得进入风险感知路线切换阶段

诊断结论：`mechanistic_signal_justifies_one_preregistered_validation_only_counterfactual; current_models_are_not_validated_safe_deployment_gates`。可以论证开展一次严格预注册的 validation-only 反事实实验，但不能宣称当前 logistic/tree 已能安全拦截实际 q_return 灾难案例；必须同时检查 competitive cohort OOF 与逐例失败，而且实际 catastrophic 只有两个独立 target。

## 18. 下一步唯一建议与停止边界

唯一建议：如另行授权，预注册一次 **冻结 KNN mean 安全基准 + 无监督多信号风险门控的 target-disjoint validation-only 路线切换反事实评估**，明确比较 selected Dice、负尾部、paired bootstrap 与灾难率；在协议冻结前不查看 test。

本轮不部署任何阈值/模型，不修改 B7，不运行 GPU/SAM3 propagation，不训练 SAM3/Student/learned router，不读取 test，不自动进入下一轮。
