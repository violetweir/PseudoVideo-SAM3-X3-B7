# Mask-conditioned SAM3-enc + Qwen 文本混合 KNN（λ=0.2 首版）

> 版本: 2026-08-14
> 状态: 已完成 mask-conditioned 视觉描述子提取与首版混合 KNN 评估

## 1. 方法

- 视觉部分改为 mask-conditioned：
  - SAM3 encoder 提取 patch token；
  - 用 pseudo/GT mask 选前景 token；
  - 取前景 token 均值并 L2 归一化；
- 文本部分仍是 Qwen3.5 mask 文本描述 one-hot 向量；
- 路径相似度：

```text
transition = (1 - λ) * cos(mask_visual) + λ * cos(qwen_text)
λ = 0.2
```

## 2. 结果

| 口径 | direct | b1 | b2 | b3 | b4 | b5 | b6 | 最强 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.7791 | 0.8206 | 0.8364 | 0.8501 | **0.8627** | 0.8543 | 0.8501 | b4 0.8627 |
| test | 0.7889 | 0.8463 | 0.8742 | **0.8832** | 0.8760 | 0.8813 | 0.8765 | b3 0.8832 |

## 3. 对比

| 方法 | validation 最强 | test 最强 |
|---|---:|---:|
| Qwen 纯文本 KNN | 0.8308 | 0.8394 |
| SAM3-enc 原图纯视觉 KNN | 0.8430 | 0.8874 |
| 原图视觉 + Qwen λ=0.2 | 0.8816 | 0.8929 |
| **mask 视觉 + Qwen λ=0.2** | **0.8627** | **0.8832** |

结论：

- 当前 mask-conditioned 视觉版本低于原图视觉版本；
- 但高于 Qwen 纯文本版本；
- 可能需要继续调 λ，或在路径评分中给 mask 视觉和文本不同权重。
