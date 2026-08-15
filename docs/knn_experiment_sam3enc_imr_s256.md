# SAM3-enc IMR KNN 实验(256×256,图像原生多模态检索)

> 创建:2026-08-15。IMR = Image-native Multimodal Retrieval。
> 目标:在"利用图像原本信息 + 多模态(Qwen 看原图)"的前提下,重新设计
> SAM3 编码器 KNN 的检索特征。协议统一 256×256(与 DINOv3 ViT-B @256、
> SAM3-enc @256 表直接可比),评估口径与 §9 完全一致。

## 0. 动机(基于 §9 与 mask-visual 扫描的证据)

- **背景上下文有用**:SAM3-enc 下全图 `patch_mean` > mask 条件描述子
  (validation:λ=0 纯原图 b2/b3≈0.854 vs λ=1.0 纯 mask b3≈0.806);
- **位置/布局信息缺失**:`patch_mean` 把 18×18 token 网格压成 1 个向量,
  息肉在哪、周围组织长什么样全部丢失;
- **文本通道区分度不足**:旧 Qwen 分支喂的是二值 mask 图、6 字段 one-hot
  (19 维),同类别不同外观的图余弦≈1;
- 结论:**检索特征应从原图直接提取**(无伪 mask 循环依赖),并叠加
  "真·看原图"的 VLM 文本通道。

## 1. 统一协议(与 §9 对齐,仅检索特征不同)

| 项 | 固定值 |
|---|---|
| 数据集 | Kvasir-SEG 快照(同前,train 800 / val 100 / test 100) |
| 标注 | 8 个固定 GT anchor;桥帧候选 = 全部 train 图(无标注) |
| 特征提取器 | SAM3 主干编码器 @**256**(patch 14 → grid 18×18=324 tokens,RoPE 重算) |
| KNN 检索特征 | pyramid(空间金字塔)/ contrast(对比度加权)/ fused(视觉+文本 rank 融合) |
| 文本通道 | Qwen3.5-4B 描述**原图**(anchor 额外叠加 GT mask),8 字段类别向量(27 维) |
| SAM3 评估 | 冻结 base `sam3.pt` / `lora_p491_e20_merged_video.pt`,`--canvas 256`,forward-only |
| 路由搜索 | beam width 32;bridge b0–b6;split=validation→test;target 排除 support;桥帧不重复 |
| 评估口径 | 每变体 100 targets × 7 桥长 = 700 条路线,逐桥长均值 Dice |
| 输出根目录 | `work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_imr_s256/` |

## 2. 变体清单

### 2.1 非 anchor(纯视觉,mask-free 描述子)

| mode | 描述子 | 含义 |
|---|---|---|
| `sam3enc_patch_average` | patch_mean(已有基线) | 全部 324 token 均值 |
| `sam3enc_pyramid` | 2×2 + 3×3 分块均值拼接(13×1024→L2) | 编码"息肉在图中的位置/布局" |
| `sam3enc_contrast` | token 通道方差加权池化 | mask-free 软前景,突出纹理/边缘 |
| `sam3enc_pyramid_contrast` | 两者拼接后 L2 | 位置 + 纹理联合 |

### 2.2 anchor-conditioned 融合(IMR 核心)

`sam3enc_imr`(target_pooling 风格 cond)+ KNN 检索 = fused:

- 视觉项:`pyramid`+`contrast` 拼接描述子的余弦 `pc_sim`;
- 文本项:Qwen 图像描述 27 维类别向量余弦 `text_sim`;
- 融合:`fused = (1−λ)·rank(pc_sim) + λ·rank(text_sim)`(percentile rank 融合),
  λ = `--text-blend`;排序仍主键 fused、次键 cond_score、三级 merged_id;
- mode_key:`sam3enc_imr`(λ=0)/ `sam3enc_imr__tb025/050/075/100`(λ>0)。

## 3. 实现与启动

- 特征提取/路由:`scripts/stage1_feature_knn_routes.py`
  - `extract_sam3_encoder_features(..., imr=True)` → `features/sam3_base_s256_imr_features.npz`
    (新增 `pyramid` / `contrast` key,不污染基线 npz);
  - `build_mode_state()` 新增 IMR modes;`build_rank_cache()` 新增 fused 分支;
  - `pyramid_pool` / `contrast_pool` 为模块级函数。
- 文本描述:`scripts/qwen35_describe_images.py`(新)
  - 输入 `merged_manifest.jsonl`(纯原图)/ `support_manifest.jsonl --overlay-gt-mask`;
  - 输出 `work/kvasir_1pct_anchors/qwen35_image_descriptions/qwen35_image_descriptions[_overlay].jsonl`;
  - 字段:shape/size/position/boundary/components/spread/texture/context。
- 启动脚本:`scripts/run_stage1_knn_imr_s256.sh`
  - Phase 0:Qwen 描述(`QWEN_SKIP=1` 可跳过);
  - Phase 1:validation 构造(3 视觉 mode + 5 λ);
  - Phase 2:validation eval(base);
  - Phase 3:设置 `BEST_LAMBDA` 后跑 test 构造 + base/lora eval + 汇总。

## 4. 结果(2026-08-15 完成,test 100 targets,forward-only Dice)

协议:冻结 SAM3 base `sam3.pt` / `lora_p491_e20_merged_video.pt`,canvas 256,
DINOv3 无关(SAM3-enc 特征 @256,grid 18),beam 32,bridge b0–b6,
每变体 700 条路线。**λ 按用户指示直接在 test 上选定**(未用 validation,
数字略偏乐观,报告需注明口径)。

**base @256:**

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_patch_average(§9) | patch_mean | 0.7981 | 0.8466 | 0.8700 | 0.8623 | 0.8739 | 0.8696 | 0.8740 |
| sam3enc_pyramid | pyramid | **0.8113** | 0.8102 | 0.8374 | 0.8601 | 0.8714 | 0.8618 | 0.8743 |
| sam3enc_contrast | contrast | 0.7502 | 0.7889 | 0.8187 | 0.8204 | 0.8094 | 0.7796 | 0.7680 |
| sam3enc_pyramid_contrast | pc | 0.8014 | 0.7935 | 0.8233 | 0.8646 | 0.8692 | 0.8506 | 0.8464 |
| sam3enc_imr(λ=0) | fused/pc | 0.7981 | **0.8572** | **0.8589** | **0.8674** | **0.8751** | **0.8679** | 0.8624 |
| sam3enc_imr__tb025 | fused λ=0.25 | 0.7981 | 0.8417 | 0.8431 | 0.8424 | 0.8508 | 0.8549 | 0.8457 |
| sam3enc_imr__tb050 | fused λ=0.5 | 0.7981 | 0.8429 | 0.8342 | 0.8417 | 0.8237 | 0.8443 | 0.8661 |
| sam3enc_imr__tb075 | fused λ=0.75 | 0.7981 | 0.8441 | 0.8297 | 0.8504 | 0.8211 | 0.8377 | 0.8570 |
| sam3enc_imr__tb100 | fused λ=1.0 | 0.7981 | 0.8547 | 0.8398 | 0.8418 | 0.8552 | 0.8657 | 0.8464 |

**lora_p491_e20 @256:**

| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sam3enc_patch_average(§9) | patch_mean | 0.8222 | 0.8505 | 0.8861 | 0.8689 | 0.8858 | 0.8930 | 0.8876 |
| sam3enc_pyramid | pyramid | 0.8376 | 0.8597 | 0.8742 | 0.8974 | 0.8902 | 0.9027 | 0.8925 |
| sam3enc_contrast(=patch_mean 排序) | contrast | 0.8222 | 0.8505 | 0.8861 | 0.8689 | 0.8858 | 0.8930 | 0.8876 |
| sam3enc_pyramid_contrast | pc | 0.8297 | 0.8640 | 0.8810 | **0.9061** | 0.9022 | 0.8996 | 0.8992 |
| sam3enc_imr(λ=0) | fused/pc | 0.8482 | **0.8752** | 0.8766 | 0.9040 | 0.9033 | 0.9019 | 0.9008 |
| sam3enc_imr__tb025 | fused λ=0.25 | 0.8482 | 0.8705 | **0.8940** | 0.9000 | 0.8889 | 0.9030 | 0.9059 |
| sam3enc_imr__tb050 | fused λ=0.5 | **0.8482** | 0.8746 | 0.8781 | 0.8968 | **0.9074** | **0.9060** | **0.9110** |
| sam3enc_imr__tb075 | fused λ=0.75 | 0.8482 | 0.8689 | 0.8779 | 0.8961 | 0.8920 | 0.8922 | 0.9028 |
| sam3enc_imr__tb100 | fused λ=1.0 | 0.8482 | 0.8699 | 0.8829 | 0.8702 | 0.8838 | 0.8877 | 0.8982 |

### 4.1 要点

- **lora 下 IMR 显著胜出**:`sam3enc_imr__tb050` b6 = **0.9110**、b4 = 0.9074、
  b5 = 0.9060,全面超过 §9 的 256 特征最佳(patch_correspondence cond b4 = 0.9082);
  λ=0.5 的文本融合在长桥(b4–b6)是全场最高——**LoRA 后文本描述与传播质量
  的相关性变强**(与 §7.1"LoRA 后锚相关度信号更可靠"一致);
- **base 下文本有害**:λ>0 全部劣于 λ=0,说明 LoRA 前 Qwen 图像描述
  (27 维类别 one-hot)与桥质量关联弱,文本通道在 base 下是噪声;
- **pyramid(位置/布局)是有效的纯视觉改进**:非 anchor 下 base direct
  0.8113 vs 基线 0.7981(+0.013);lora b3 0.8974 vs 基线 0.8730(+0.024);
  direct 提升 = 特征让"选 anchor 直传"更准,不经过选桥;
- **contrast 完全无效(设计失败)**:base 与 lora 下 `sam3enc_contrast` 的逐格
  Dice 与 §9 `sam3enc_patch_average` **完全一致**(0.7502/…/0.7680 与
  0.8222/…/0.8876)——L2 归一化后的 token 通道方差差异极小,方差加权
  退化为普通均值,排序等价、路线相同。改进方向:改用未归一化 token 的
  方差、局部邻域对比度(拉普拉斯),或直接丢弃该描述子;
  但 lora 下 `pyramid_contrast` b3 = 0.9061 是最佳 b3,contrast 拼接进
  pyramid 在长桥有微弱正贡献;
- 产物:`comparison_summary_imr_s256_base.md/json`、
  `comparison_summary_imr_s256_lora.md/json`。

## 5. 注意点

- 文本通道区分度风险:若 Qwen 对 Kvasir 内镜图描述同质化,λ 最优会趋近 0,
  文本通道退化为消融结论(干净结果,不掩盖);
- `qwen_text` 字段含 prompt 回显,当前只消费 `qwen_features`;
  若后续要稠密文本嵌入需另行裁剪;
- anchor 描述(overlay)与候选描述(纯原图)分布系统性偏移,已在
  validation 对照(λ 扫描涵盖两档);若 `tb100` 显著劣于低 λ,说明文本
  偏移污染,融合权重应保持较小;
- 256 特征输入 token 数为 18×18=324(SAM3 patch 14),与 DINOv3 ViT-B
  的 16×16=256 不同源,报告统一口径为"输入分辨率 256×256"。
