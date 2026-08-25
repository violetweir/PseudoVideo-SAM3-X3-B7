# ISIC18 SAM3-KNN@256 target-pooling B0-B6 test

Completed: 2026-08-25 00:33 (Asia/Shanghai)

## Protocol

- Dataset: ISIC2018
- Split sizes: train 2075, validation 259, test 260
- Frozen human anchors: 21 (train only)
- Feature model: frozen base `sam3.pt` image trunk
- Feature input: 256 x 256
- Descriptor: L2-normalized patch mean, width 1024
- Route mode: `sam3enc_anchor_conditioned_target_pooling`
- KNN feature: `patch_mean`
- Beam width: 32
- Routes: B0-B6, one route per bridge count and test target
- Propagation checkpoint: base `sam3.pt`
- Propagation canvas: 256 x 256
- GPU: physical GPU 0 (`CUDA_VISIBLE_DEVICES=0`)
- Test GT was used only for final Dice evaluation, never for route search or inference.

Protocol root:

```text
work/isic18_pseudovideo_full/protocol
```

Experiment root:

```text
work/isic18_sam3knn_s256_base/stage1_feature_knn_b0_b6
```

## Commands

```bash
CUDA_VISIBLE_DEVICES=0 /home/violet/anaconda3/envs/sam3/bin/python \
  scripts/stage1_feature_knn_routes.py \
  --mode sam3enc_anchor_conditioned_target_pooling \
  --feature-source sam3_base \
  --feature-size 256 \
  --knn-feature patch_mean \
  --split test \
  --min-bridge 0 \
  --max-bridge 6 \
  --beam-width 32 \
  --protocol-root work/isic18_pseudovideo_full/protocol \
  --output-root work/isic18_sam3knn_s256_base/stage1_feature_knn_b0_b6

CUDA_VISIBLE_DEVICES=0 /home/violet/anaconda3/envs/sam3/bin/python \
  scripts/eval_route_forward_dice.py \
  --checkpoint /Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt \
  --mode sam3enc_anchor_conditioned_target_pooling \
  --root work/isic18_sam3knn_s256_base/stage1_feature_knn_b0_b6 \
  --split test \
  --canvas 256 \
  --resume
```

The first 179 routes were evaluated with the original
`eval_route_propagation_quality.py`, including the target-to-anchor return
cycle. The remaining 1641 routes used the Dice-only evaluator, which removes
only the post-forward return cycle. Route construction, forward propagation,
saved forward masks, and Dice computation are identical. The partial cycle
statistics are intentionally not reported.

## Results

| Bridge | N | Mean Dice | Std | Median |
|---|---:|---:|---:|---:|
| B0 | 260 | 0.715039 | 0.324960 | 0.884640 |
| B1 | 260 | 0.710054 | 0.325364 | 0.886180 |
| B2 | 260 | 0.716470 | 0.326796 | 0.893312 |
| B3 | 260 | 0.722435 | 0.320421 | 0.895222 |
| B4 | 260 | 0.724580 | 0.319677 | 0.894637 |
| B5 | 260 | 0.727414 | 0.320291 | 0.894824 |
| B6 | 260 | **0.728152** | 0.322136 | 0.896343 |

- Macro mean across all B0-B6 routes: `0.720592`
- Evaluation-only per-target oracle over B0-B6: `0.768580`
- Best fixed bridge: B6
- B6 minus B0: `+0.013113` Dice

## Audit

- Frozen routes: 1820
- Successful result rows: 1820
- Unique result route IDs: 1820
- Route/result ID sets match: yes
- Rows per bridge: 260 for every B0-B6
- Routes per target: exactly 7 for all 260 targets
- Missing Dice values: 0
- Rows where target GT participated in search/inference: 0

Primary artifacts:

```text
work/isic18_sam3knn_s256_base/stage1_feature_knn_b0_b6/features/sam3_base_s256_features.npz
work/isic18_sam3knn_s256_base/stage1_feature_knn_b0_b6/sam3enc_anchor_conditioned_target_pooling/test_pool0_stage1/routes.jsonl
work/isic18_sam3knn_s256_base/stage1_feature_knn_b0_b6/sam3enc_anchor_conditioned_target_pooling/propagation_quality_test/propagation_quality.jsonl
work/isic18_sam3knn_s256_base/stage1_feature_knn_b0_b6/sam3enc_anchor_conditioned_target_pooling/propagation_quality_test/summary.json
```
