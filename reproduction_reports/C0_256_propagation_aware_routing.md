# C0-256 传播感知选路机制验证：KNN 图像相似性 vs SAM3 返回一致性

> 生成时间：2026-08-25T15:22:36+08:00  
> 仓库：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7`  
> 实验类型：冻结历史候选的离线 CPU 统计；无新 GPU propagation、无训练、无参数搜索。

## 1. 研究问题与实验动机

对于同一个目标图像的 14 条冻结伪视频候选路线，比较现成 KNN 路径相似度与 SAM3 自身传播返回一致性，判断哪一种分数更能排序真实传播 Dice，并选出更好的 Top-1 路线。

## 2. 与 Round2A / Round2B 的关系

主实验固定 Round2A：SAM3-base@256 KNN topology + 已冻结的 SAM3-e33 propagation teacher。Route 图、anchors、两种 mode、b0-b6 和传播结果均不改变。

Round2B 辅助检查：e33 topology + e33 teacher 仅有历史 test 传播，缺少完整 validation q_cycle/GT，因此按协议跳过且不启动 GPU。

## 3. 数据协议与候选路线来源

Round2A 入口：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_round2a_fixed_knn_e33/quality_root`。其中 validation/test propagation 目录是历史 e33 全量评估结果的符号链接；不跟随符号链接的 `find -type d` 会误以为这两个 split 缺失。

每个 split 固定为 100 targets × 2 route modes × b0-b6 = 1400 candidates。每个 target 的 14 个候选始终共同参与每一种 selector。

## 4. 所有实际读取输入文件及 SHA256

| 角色 | 输入文件 | 实际解析路径 | SHA256 |
|---|---|---|---|
| `validation_sam3enc_anchor_conditioned_target_pooling_frozen_propagation` | `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_round2a_fixed_knn_e33/quality_root/sam3enc_anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl` | `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/e33_full_evaluation/quality_root/sam3enc_anchor_conditioned_target_pooling/propagation_quality_validation/propagation_quality.jsonl` | `75ede921a8b2a7062612a28e0db2ea0684bdd5f18def91256915c8b2ceb0e28b` |
| `validation_sam3enc_anchor_conditioned_patch_correspondence_frozen_propagation` | `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_round2a_fixed_knn_e33/quality_root/sam3enc_anchor_conditioned_patch_correspondence/propagation_quality_validation/propagation_quality.jsonl` | `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/e33_full_evaluation/quality_root/sam3enc_anchor_conditioned_patch_correspondence/propagation_quality_validation/propagation_quality.jsonl` | `c3692d71c035f3c7e33e898cc26dab8005e80706695d1b7b1ab58332c0840424` |
| `historical_b7_all_candidates` | `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_round2a_fixed_knn_e33/b7_calibration/validation_all_candidates.jsonl` | `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_round2a_fixed_knn_e33/b7_calibration/validation_all_candidates.jsonl` | `b26e9dd13687f5fbe14e6a0974e032852d8a483d6bfcc6003f988d8583fb1dd3` |

## 5. KNN 相似度、q_return 与既有 B7 定义

- `path_bottleneck_similarity`：冻结路线中最弱边的现有 KNN 特征相似度。
- `path_mean_similarity`：冻结路线各边的现有 KNN 平均特征相似度。
- `q_return = q_cycle = Dice(anchor mask, returned anchor mask)`；直接读取已有传播结果，不重新运行 SAM3。
- B7 仅作为上下文，读取历史冻结结果；既有公式 `B7=(q_return*q_multi²*q_model²)^0.2` 不被重新计算或调权。

## 6. GT 使用边界与可审计 tie-break

A/B/C 路线选择只调用 `max(proxy, route_id, route_mode)`；B7 使用历史固定 `max(b7, q_multi, q_return, route_id)` 或直接读取完整的冻结历史选择。`gt_dice_evaluation_only` 不进入这些 key、公式、阈值或候选筛选。

GT 只用于选择后的 Dice 评价、candidate/target 排序相关性、GT Oracle 和 paired-target bootstrap。validation 上表现更好的 KNN 方法仅作为事后比较基线，并在读取 test 之前冻结；这一步不会改变任何路线选择方法或参数。

## Validation：候选数量审计

- targets：100；candidates：1400。
- 每 target 候选数：`{"14": 100}`。
- bridge 分布：`{"0": 200, "1": 200, "2": 200, "3": 200, "4": 200, "5": 200, "6": 200}`。
- route mode 分布：`{"sam3enc_anchor_conditioned_patch_correspondence": 700, "sam3enc_anchor_conditioned_target_pooling": 700}`。
- 每个 target × route mode × bridge_count 均恰好一条；未静默过滤失败、缺失或额外候选。

## Validation：candidate-level 相关性

| 指标 | Spearman ρ | p-value | Kendall τ | Kendall p-value | N |
|---|---:|---:|---:|---:|---:|
| KNN 瓶颈相似度 | 0.015387 | 5.651e-01 | 0.008655 | 6.331e-01 | 1400 |
| KNN 平均相似度 | 0.043077 | 1.072e-01 | 0.029596 | 9.757e-02 | 1400 |
| 传播返回一致性 q_return | 0.216292 | 2.775e-16 | 0.150272 | 4.401e-17 | 1400 |
| 既有 B7（冻结 X3） | 0.574399 | 1.060e-123 | 0.413038 | 2.779e-118 | 1400 |

全局相关性只作为辅助证据，不能替代同一 target 候选池中的排序和 Top-1 评价。

## Validation：target 内部排序相关性

| 指标 | target 内平均 ρ | 中位数 ρ | 正相关 target | 有定义 target |
|---|---:|---:|---:|---:|
| KNN 瓶颈相似度 | -0.070998 | -0.053158 | 43/100 (0.430) | 100 |
| KNN 平均相似度 | 0.074183 | 0.148930 | 65/100 (0.650) | 100 |
| 传播返回一致性 q_return | 0.334082 | 0.470847 | 77/99 (0.778) | 99 |

- q_return target 内 ρ 高于 KNN 瓶颈相似度：78/99 (0.788)。
- q_return target 内 ρ 高于 KNN 平均相似度：71/99 (0.717)。

## Validation：逐 target Top-1 与 oracle gap

| 选择方法 | Selected Dice | GT Oracle | Oracle Gap |
|---|---:|---:|---:|
| KNN 瓶颈相似度 | 0.839251 | 0.905122 | 0.065871 |
| KNN 平均相似度 | 0.887591 | 0.905122 | 0.017531 |
| 传播返回一致性 q_return | 0.883055 | 0.905122 | 0.022067 |
| 既有 B7（冻结 X3） | 0.883854 | 0.905122 | 0.021268 |
| GT Oracle（仅评价） | 0.905122 | 0.905122 | 0.000000 |

主要 KNN 对照：**KNN 平均相似度**；来源：`validation_evaluation_strongest_knn_baseline`。

## Validation：paired target bootstrap

- 改善 / 持平 / 退化：54 / 18 / 28。
- 平均 Δ Dice：-0.004536；中位数 Δ Dice：0.000336。
- 95% paired-target bootstrap CI：[-0.017142, 0.004023]；resamples=10000，seed=2026。
- bootstrap 的独立重采样单位是同一个 target 的配对 Δ，不是 1400 条候选。
- 尾部风险审计：最大单例退化 -0.514066，最大单例改善 0.117177，两例最大退化合计 -0.722635；全部改善合计 0.380451，全部退化合计 -0.834043。因此胜场多于败场不保证平均 selected Dice 提升；此处不剔除异常 target，也不据此调 selector。

## Validation：route switch 与获益分布

- 相同 route_id：17；不同 route_id：83。
- switch targets 中改善 / 持平 / 退化：54 / 1 / 28。
- bridge 选择分布：`{"knn_mean": {"4": 1, "5": 4, "6": 95}, "q_return": {"0": 16, "2": 6, "3": 8, "4": 18, "5": 27, "6": 25}}`。
- mode 选择分布：`{"knn_mean": {"sam3enc_anchor_conditioned_target_pooling": 100}, "q_return": {"sam3enc_anchor_conditioned_patch_correspondence": 39, "sam3enc_anchor_conditioned_target_pooling": 61}}`。
- switch 按 q_return bridge 分组：`{"0": {"degraded_target_count": 7, "improved_target_count": 9, "mean_delta": -0.0008975957657251423, "target_count": 16, "tied_target_count": 0}, "2": {"degraded_target_count": 4, "improved_target_count": 2, "mean_delta": -0.03601058290607689, "target_count": 6, "tied_target_count": 0}, "3": {"degraded_target_count": 2, "improved_target_count": 6, "mean_delta": 0.020440222869913566, "target_count": 8, "tied_target_count": 0}, "4": {"degraded_target_count": 4, "improved_target_count": 13, "mean_delta": 0.003656096579161553, "target_count": 17, "tied_target_count": 0}, "5": {"degraded_target_count": 7, "improved_target_count": 18, "mean_delta": 0.002359345852905682, "target_count": 26, "tied_target_count": 1}, "6": {"degraded_target_count": 4, "improved_target_count": 6, "mean_delta": -0.05101860508803435, "target_count": 10, "tied_target_count": 0}}`。
- switch 按 q_return mode 分组：`{"sam3enc_anchor_conditioned_patch_correspondence": {"degraded_target_count": 15, "improved_target_count": 24, "mean_delta": -0.011594057418969258, "target_count": 39, "tied_target_count": 0}, "sam3enc_anchor_conditioned_target_pooling": {"degraded_target_count": 13, "improved_target_count": 30, "mean_delta": -3.237327836371371e-05, "target_count": 44, "tied_target_count": 1}}`。

## Validation：q_return 改善最大的最多 10 个 target

| target | KNN mode/b/route | KNN sim / q_return / Dice | q_return mode/b/route | q_return sim / q_return / Dice | Δ Dice |
|---|---|---|---|---|---:|
| `kvasir-seg::cju7eotqi2qea0871y8yc7tqh` | target_pooling / b6 / `b21ec4e6549fdc2b33a82ac3` | 0.909829 / 0.961114 / 0.269387 | target_pooling / b3 / `55d6b85767e903c50b90e162` | 0.901954 / 0.961845 / 0.386564 | 0.117177 |
| `kvasir-seg::cju42nm68lpyo0818xvvqmupq` | target_pooling / b6 / `0ca343d2d2d30f77ec3ceb45` | 0.911465 / 0.965059 / 0.919726 | target_pooling / b4 / `3a5cd3e8bbc067e20c23cb7e` | 0.907993 / 0.967065 / 0.954576 | 0.034851 |
| `kvasir-seg::cju1d96gsv62d09881b3wecw2` | target_pooling / b6 / `b6aee2dbadcbd7ba58c2082b` | 0.912306 / 0.967446 / 0.898133 | patch_correspondence / b5 / `16e526e70ac13f18f7e452fb` | 0.807260 / 0.969671 / 0.927538 | 0.029405 |
| `kvasir-seg::cju5klveuff6w0871wbibgh3m` | target_pooling / b6 / `44d4b4a287afde76ceb7a838` | 0.900362 / 0.976898 / 0.920546 | target_pooling / b3 / `8ebeda700112d074813e65ea` | 0.896162 / 0.979276 / 0.949210 | 0.028664 |
| `kvasir-seg::cju3yht87j83m08507yk1u1fg` | target_pooling / b6 / `e98b698b5f8e589ae8fc418c` | 0.910032 / 0.966163 / 0.907410 | target_pooling / b5 / `cb5d7d4e606629d339fa0eb6` | 0.908953 / 0.968288 / 0.923168 | 0.015758 |
| `kvasir-seg::cju5yimthmlv80850zhoc90c2` | target_pooling / b6 / `4ae4cca578756a405eb76cd3` | 0.907091 / 0.977351 / 0.936409 | target_pooling / b0 / `c512856468be8882aea7264c` | 0.769531 / 0.983373 / 0.950115 | 0.013706 |
| `kvasir-seg::cju7ap09p1kz10850ldccjebj` | target_pooling / b6 / `5a34f327dd341fb3be0e85f5` | 0.921638 / 0.972683 / 0.909851 | patch_correspondence / b3 / `090db61fb196c0ccbbdad4d6` | 0.806359 / 0.981299 / 0.923204 | 0.013353 |
| `kvasir-seg::cju5gucasds9d0801019axylx` | target_pooling / b6 / `7ce2d688a2ede1d96bfff276` | 0.911669 / 0.968279 / 0.804695 | target_pooling / b0 / `89adb3fbe831eb1e5911404a` | 0.824219 / 0.970027 / 0.815990 | 0.011296 |
| `kvasir-seg::cju2qfie4rvz508357kad9z5o` | target_pooling / b6 / `8419c2bc5553d3257624b62b` | 0.921828 / 0.971819 / 0.874855 | patch_correspondence / b5 / `8f02f67b24bdd8545f42961c` | 0.812238 / 0.973325 / 0.885801 | 0.010947 |
| `kvasir-seg::cju88y1mwoln50871emyfny1g` | target_pooling / b6 / `625c8b21e23c26740ad08dec` | 0.909711 / 0.968916 / 0.897134 | patch_correspondence / b5 / `f52bed066de2fa6cc00bc4f4` | 0.811093 / 0.977264 / 0.907708 | 0.010574 |

## Validation：q_return 退化最大的最多 10 个 target

| target | KNN mode/b/route | KNN sim / q_return / Dice | q_return mode/b/route | q_return sim / q_return / Dice | Δ Dice |
|---|---|---|---|---|---:|
| `kvasir-seg::cju7b1ygu1msd0801hywhy0mc` | target_pooling / b6 / `7d0f5cbc984949dd11e94a33` | 0.911368 / 0.967983 / 0.514066 | patch_correspondence / b6 / `e239f069a3794dbea15eb576` | 0.813042 / 0.977502 / 0.000000 | -0.514066 |
| `kvasir-seg::cju1ewnoh5z030855vpex9uzt` | target_pooling / b6 / `2b175dc35699f4fdd574fb35` | 0.912410 / 0.963402 / 0.974958 | target_pooling / b2 / `139da78e64f3f7f0adc62c2e` | 0.902982 / 0.965497 / 0.766389 | -0.208569 |
| `kvasir-seg::cju5i39mreass0817au8p22zy` | target_pooling / b6 / `09254ef99f62f74a37dc81e0` | 0.911338 / 0.965706 / 0.951991 | target_pooling / b0 / `a5641f7c47691681f054cba5` | 0.835938 / 0.967156 / 0.928028 | -0.023963 |
| `kvasir-seg::cju88itqbny720987hxizbj5y` | target_pooling / b6 / `57317dd11fc09b1c22e46199` | 0.912362 / 0.968463 / 0.882976 | target_pooling / b0 / `a2879e1caf5a42c8d6a873f5` | 0.835938 / 0.968889 / 0.863122 | -0.019854 |
| `kvasir-seg::cju5cjh3xattc0817j2vbulzi` | target_pooling / b6 / `fe329431ae361c6714030b51` | 0.912429 / 0.960151 / 0.841596 | patch_correspondence / b5 / `8a5d4d90057998e4a7c83f83` | 0.817950 / 0.977286 / 0.827011 | -0.014585 |
| `kvasir-seg::cju7f0ec32txj08184asb8w5f` | target_pooling / b6 / `d9bc1ddec08cc7269cbaec62` | 0.909336 / 0.964291 / 0.961174 | patch_correspondence / b2 / `0bbc8dbbca5e90dc6cb96b69` | 0.803165 / 0.981467 / 0.952768 | -0.008406 |
| `kvasir-seg::cju42g865lorv07552ytz6xxa` | target_pooling / b6 / `6fa29e1497f4120ac4b432e6` | 0.921484 / 0.970224 / 0.940746 | patch_correspondence / b6 / `6301def214bec6bd03946a90` | 0.811219 / 0.971810 / 0.933494 | -0.007252 |
| `kvasir-seg::cju5bycdkalkb09875f7bfrvx` | target_pooling / b6 / `4f9ab42b6b9ec4027cd9e6b0` | 0.912645 / 0.966777 / 0.748725 | target_pooling / b0 / `9ea29459924a3eeda8ad45ad` | 0.832031 / 0.968076 / 0.743358 | -0.005366 |
| `kvasir-seg::cju0s2a9ekvms080138tjjpxr` | target_pooling / b6 / `a606acc84cbdb8f29490ee8a` | 0.911846 / 0.965358 / 0.980441 | patch_correspondence / b3 / `2c2f064674dba14b6eaf4905` | 0.802816 / 0.967204 / 0.976432 | -0.004008 |
| `kvasir-seg::cju3xzvnzj0hd0755xprz39nj` | target_pooling / b6 / `f5231579c6d4b17b0e1c9e8f` | 0.909150 / 0.969253 / 0.855430 | patch_correspondence / b4 / `ee3ee158ed588393b8ef34cb` | 0.808833 / 0.974309 / 0.851920 | -0.003509 |

## Validation gate 与 test 冻结协议

- q_return Spearman 高于两个 KNN 指标：**True**。
- q_return Top-1 selected Dice 高于两个 KNN selector：**False**。
- paired bootstrap 95% CI 下界大于 0：**False**。
- gate 最终通过：**False**。
- validation 冻结 KNN 主对照：`knn_mean`。
- 只有前两条同时为真，程序才会打开 test propagation / 历史 B7 选择文件；不会根据 test Dice 调分数、换 baseline、删除 mode 或缩小 bridge 范围。

## Test 状态

Validation gate 未通过，因此未读取任何 test 候选内容或 test selector 结果，也没有生成 test_candidates.jsonl、test_per_target.jsonl 或 test_summary.json。

## 最终结论：是否支持“图像相似性 ≠ 传播兼容性”

q_return 能反映候选的平均质量，但未提高同一 target 的 Top-1 selected Dice，不能独立承担最终路线排序。

## 下一步建议与明确停止边界

停止单独用 q_return 替代 KNN 的方案；如后续另行授权，可研究 q_return、q_multi、传播轨迹和学生信息的联合排序。

本轮没有运行 GPU 推理，没有重算 q_cycle，没有修改历史 Round2A/Round2B artifact，没有调整 B7、bridge 范围或训练 SAM3、Student、X5、learned router。
