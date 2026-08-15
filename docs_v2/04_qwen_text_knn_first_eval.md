# Qwen3.5 文本描述 KNN：第一版评估

> 版本: 2026-08-14
> 状态: 已生成 train/validation/test/anchor 的文本描述并构建纯文本 KNN 路线

## 1. 方法

- 第一轮 pseudo mask 生成后，用 Qwen3.5-4B 对每张 mask 输出结构化文本描述：

```text
shape=<round|oval|irregular|elongated>;
size=<small|medium|large>;
position=<left|center|right|top|bottom|diffuse>;
boundary=<smooth|irregular>;
components=<single|multiple>;
spread=<local|multi_region>
```

- 将 6 个分类字段转成 one-hot 向量，用 cosine 作为 KNN 相似度；
- 只用 8 个固定 GT anchor 和 792 个 train bridge，构建 b0-b6 路线；
- SAM3 base @256 评估。

## 2. 结果

| 口径 | direct | b1 | b2 | b3 | b4 | b5 | b6 | 最强桥 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.7479 | 0.8002 | 0.7949 | 0.8230 | **0.8308** | 0.8192 | 0.8139 | b4 0.8308 |
| test | 0.7377 | 0.7763 | 0.7731 | 0.8007 | 0.8288 | **0.8394** | 0.8276 | b5 0.8394 |

## 3. 和第一轮 SAM3-enc KNN 对比

| 方法 | validation 最强 | test 最强 |
|---|---:|---:|
| Qwen 文本 KNN | 0.8308 | 0.8394 |
| SAM3-enc @1008 target_pooling patch_mean | 0.8430 | 0.8874 |
| SAM3-enc base b0-b6 fusion | 0.8515 | 0.8979 |

结论：纯 Qwen 文本描述单独做 KNN 明显弱于 SAM3-enc 特征，但它提供的是另一类
语义/形状信息，下一步应作为辅助相似度与 SAM3-enc 主特征组合，而不是单独使用。

## 4. 产物

```text
work/kvasir_1pct_anchors/qwen_text_knn_v1/
work/kvasir_1pct_anchors/qwen_text_knn_routes/
  qwen_text_knn/validation_pool0_stage1/routes.jsonl
  qwen_text_knn/test_pool0_stage1/routes.jsonl
  qwen_text_knn/eval_base_no_ft_b7_forward_validation/
  qwen_text_knn/eval_base_no_ft_b7_forward/
```
