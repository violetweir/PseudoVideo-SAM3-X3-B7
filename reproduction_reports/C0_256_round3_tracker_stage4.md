# C0-256 Round3：Stage-IV Pseudo-Video Tracker Adaptation

> 协议冻结日期：2026-08-26（Asia/Shanghai）  
> 仓库：`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7`  
> 状态：协议与代码已冻结；结果表仅由实际运行产物回填。

## 1. 研究问题

Round2B 显示 e33 topology 提高 b0，却降低多数 b3-b6；因此本轮固定 `G0=KNN(F_SAM3-base@256)` 与 e33 image-side，只训练 tracker/memory，检验长链传播能否改善。

## 2. Frozen protocol

- 初始化：e33 image-side + base tracker/memory。
- checkpoint SHA256：`2eb33d5be28baf8d579eddfec2a75d2ee7b8a0c26a600603983da2db99a52280`。
- target routes SHA256：`4ee121f46bffda6a5c9eae529451dd966f0ba532db36987b72a63afab72933a2`。
- patch routes SHA256：`eefe9de51015840b756e1bc8d4eb63d20709490f17281032eb8eb0a2fe3410fb`。
- 524 pseudo manifest SHA256：`7f8e76ed708e9fd2b28b36e1d7672b179af7bae3a34bb493b06f905b8331da88`。
- 只允许 train split；8 个 anchors 使用 GT，其余监督只来自冻结 pseudo pool。
- anchor frame 输入 box；bridge/target 无额外 prompt；`text_str=None`。
- Stage4-A 的 long-chain/path-consistency loss 默认关闭。

## 3. Trainable module audit

T1 仅训练实际源码中的 `tracker.maskmem_backbone`、`tracker.transformer` 及直接 memory temporal embeddings。T2 训练完整 `tracker`；`detector`（vision/text/geometry/DETR/image segmentation）始终冻结。实际参数数目见 `work/rerun_c0_256_round3_tracker_stage4/module_audit*.json`。

## 4. Dataset

由 `scripts/build_round3_tracker_sequences.py` 生成 manifest，不复制 RGB。最终 sequence 数量、b0-b6 分布与监督帧比例见 `sequence_manifests/train_sequences.summary.json`。

## 5. Training

AdamW，seed=2026，BF16，sequence batch=1，gradient accumulation=1；T1/T2 各 20k optimizer steps，validation checkpoint 每 1000 steps，另固定保存 2k/5k/10k/15k/20k。trainer 支持任意 gradient accumulation，当前首轮为保持两组完全同配方并控制总时长，在正式训练前冻结为 1。所有状态和 RNG 可 resume。

## 6. Validation checkpoint selection

在任何 test 前冻结：

```text
ValLong = mean(validation combined Dice at b3,b4,b5,b6)
tie-break: higher b6, then earlier checkpoint
```

严禁 Direct Dice 或 test 指标参与 Stage4 checkpoint 选择。

## 7. Validation b0-b6

待实际运行后从 JSON/TSV 回填。

## 8. Test b0-b6

待 T1/T2 validation-best 冻结后各运行一次正式 test。

## 9. Gain-vs-depth

待实际运行后报告 `Delta Dice = Tk - T0` 与 `mean(b3-b6)`。

## 10. Freeze audit

每组保存 image-side before/after hash、逐 frozen tensor hash、首个 backward frozen grad 与 changed tensor count；任何不一致立即终止。

## 11. 结论

只根据最终真实结果填写，不根据 test 修改 topology、loss、horizon 或 checkpoint。
