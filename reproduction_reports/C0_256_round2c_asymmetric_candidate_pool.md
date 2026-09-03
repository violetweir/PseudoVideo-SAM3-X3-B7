# C0-256 Round-2C：Asymmetric Correspondence Anchor Pool

> 状态：validation 离线组合分析已完成。  
> 未运行任何新模型：只重组已有的 800 条 e33 b0 propagation，并按原 B7 公式复算。

## 1. 预定义候选池

| Pool | Candidate 1 | Candidate 2 | Candidate 3 |
|---|---|---|---|
| C0 | Forward prototype #1 | Forward prototype #2 | Forward prototype #3 |
| C1 | Forward prototype #1 | Forward token-topk #1 | Reverse token-topk (X3) #1 |
| C2 | Forward prototype #1 | Reverse token-topk (X3) #1 | Reverse token-topk (X3) #2 |
| C3 | Forward prototype #1 | Forward prototype #2 | Reverse token-topk (X3) #1 |

若候选重复，则从该 ranker 的下一名补齐；不会混合成单一乘法或加法分数。

## 2. Validation 结果

| Pool | Candidate oracle | B7 selected Dice | Oracle gap | Mean B7 |
|---|---:|---:|---:|---:|
| C0_forward_proto_top3 | 0.874527 | 0.866638 | 0.007889 | 0.879165 |
| C1_forward_forward_reverse | 0.877150 | 0.866951 | 0.010199 | 0.847429 |
| C2_forward_reverse_reverse | 0.876224 | 0.864176 | 0.012048 | 0.842576 |
| C3_forward_forward_reverse_alt | 0.874893 | 0.863658 | 0.011235 | 0.853766 |

## 3. 解释边界

Candidate oracle 使用 validation GT 仅作事后审计；候选构造和 B7 选择均未读取 target GT。
B7 保持原定义：`(q_return × q_multi² × q_model²)^0.2`，其中 X3-best 固定。

这些结果只检验 b0 anchor candidate pool。它们不能直接外推为 bridge reranking 或 test 结论。
