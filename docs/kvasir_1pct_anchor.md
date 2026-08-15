# Kvasir 1% Anchor Method (WACV2027 Follow-Up)

Local WACV2027 follow-up experiments on **Kvasir-SEG only**. The KNN route
graph is frozen; all improvements come from the route-quality selector, the
route generators, and SAM3 fine-tuning budgets. The current method is the
**candidate-invariant propagation-quality router**, optionally combined with a
1% full SAM3 fine-tune.

## Protocol

- Dataset snapshot:
  `/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T02_fresh_polyp_hf_sources/raw_hf_snapshots/kvasir-seg/snapshot`
- Splits: `train=800`, `validation=100`, `test=100`
- GT anchors: 1% of train = `8` fixed support masks
  (`work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt`)
- SAM3 base checkpoint:
  `/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt`
- DINOv3 weights:
  `/Data_8TB/lht/MK-UNet/teacher/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- Outputs: `work/kvasir_1pct_anchors/`
- Validation/test GT is evaluation-only. Train masks beyond the 8 anchors are
  not used for route search, pseudo-label selection, or training.

## Current Method Line

```text
Frozen KNN graph (8 anchors)
  -> route proposals: anchor-conditioned target pooling + patch correspondence
  -> candidate lengths: b3-b6 only (b7 collapses, short routes are weaker)
  -> per-route propagation-quality features (no GT)
  -> ridge scorer trained on validation route features
  -> per-target Top-1 selected mask

Optional: replace frozen SAM3 with the 1% full fine-tune (ft_1pct) before
route propagation, keeping the same frozen KNN graph and scorer protocol.
```

### Route Generators

- `anchor_conditioned_target_pooling`
- `anchor_conditioned_patch_correspondence`

`bN` means `N` bridge frames between the anchor and the query. `direct`
(b0) has no bridge. The stable propagation region is `b4-b6`; `b7` collapses
for every feature mode, so the mainline keeps `b3-b6`.

### Propagation-Quality Features (21, no GT)

`q_cycle`, `cycle_success`, `cycle_sam_score`, mask-area trajectory
(`trace_area_*`), empty-mask count, connected components, centroid/box
drift, adjacent-frame Dice, SAM-score trajectory, and candidate counts.

`q_cycle` is the fresh-state anchor cycle consistency: the predicted query
mask is converted back to a prompt and propagated to the anchor; Dice with
the anchor mask measures identity survival.

## Results

### Frozen SAM3: Frozen-Feature KNN Stage1 (per-bridge test Dice)

No weighted selector. Each cell is the mean forward-only test Dice for one
fixed route family.

| feature mode | direct | b1 | b2 | b3 | b4 | b5 | b6 | b7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T18 corrected | 0.747869 | 0.775949 | 0.756490 | 0.785808 | 0.821736 | 0.816809 | 0.819200 | 0.143392 |
| DINO global pooling | 0.740844 | 0.748290 | 0.758326 | 0.786777 | 0.777536 | 0.818397 | 0.818565 | 0.092024 |
| DINO patch average | 0.728634 | 0.726893 | 0.756876 | 0.746756 | 0.799814 | 0.829179 | 0.826294 | 0.046968 |
| anchor-conditioned target pooling | 0.741306 | 0.760176 | 0.827209 | 0.840837 | 0.841258 | 0.848618 | 0.854627 | 0.032027 |
| anchor-conditioned patch correspondence | 0.757814 | 0.773501 | 0.822305 | 0.832819 | 0.833990 | 0.833577 | 0.842078 | 0.070807 |

Strongest fixed route: `anchor-conditioned target pooling + b6 = 0.854627`.

### Frozen SAM3: Propagation-Quality Router Progression

| selector | candidates | Test Dice | Oracle | Oracle gap |
|---|---|---:|---:|---:|
| fixed best route | target pooling `b6` | 0.854627 | - | - |
| v2 absolute ridge router | target pooling `b0-b7` | 0.854437 | 0.898969 | 0.044531 |
| propagation-quality router | target pooling `b0-b7` | 0.872938 | 0.898969 | 0.026030 |
| propagation-quality router | target pooling `b0-b6` | 0.873599 | 0.898969 | 0.025370 |
| propagation-quality router | target + patch `b0-b6` | 0.873626 | 0.903846 | 0.030220 |
| propagation-quality router | target + patch `b3-b6` | **0.877299** | 0.896918 | 0.019619 |

Conclusions:

- Full `C0-C7` validation training did not close the oracle gap by itself.
- Propagation-quality features are useful: for target pooling, Top-1 Dice
  improves from `0.854437` to `0.872938`.

### 1% Full Fine-Tune + Propagation-Quality Router (current best)

The `ft_1pct` checkpoint (`finetune_1pct_seed2026`, epoch 20, 160 train
steps, merged at `work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt`)
is propagated over the same frozen KNN routes. The ridge scorer is retrained
on **validation propagation-quality features of the ft_1pct model**; no GT is
used in selection. Evaluated 2026-08-06.

| scheme | selected Dice | oracle | gap | acc | spearman |
|---|---:|---:|---:|---:|---:|
| **b3-b6 target+patch (mainline)** | **0.894648** | 0.919805 | 0.025157 | 0.28 | 0.352 |
| b3-b6 target pooling | 0.886241 | 0.903572 | 0.017331 | 0.37 | 0.376 |
| b3-b6 patch correspondence | 0.884695 | 0.901863 | 0.017168 | 0.31 | 0.311 |
| b0-b6 target+patch (reference) | 0.877061 | 0.921800 | 0.044739 | 0.17 | 0.364 |
| b0-b6 target pooling | 0.868192 | 0.911798 | 0.043606 | 0.14 | 0.361 |
| b0-b6 patch correspondence | 0.890865 | 0.907062 | 0.016198 | 0.25 | 0.355 |

Test per-bridge Dice (ft_1pct vs frozen base):

| bridge | ft_1pct target | frozen target | ft_1pct patch | frozen patch |
|---|---:|---:|---:|---:|
| direct | 0.770790 | 0.741306 | 0.797878 | 0.757814 |
| b1 | 0.792198 | 0.760176 | 0.816105 | 0.773501 |
| b2 | 0.857394 | 0.827209 | 0.857803 | 0.822305 |
| b3 | 0.852228 | 0.840837 | 0.859885 | 0.832819 |
| b4 | 0.860925 | 0.841258 | 0.884293 | 0.833990 |
| b5 | 0.873081 | 0.848618 | 0.865846 | 0.833577 |
| b6 | 0.864322 | 0.854627 | 0.879738 | 0.842078 |

The 1% fine-tune improves every bridge length for both modes
(+0.02 to +0.05). With fine-tuning, the patch mode at `b0-b6` alone
(0.890865) outperforms the union scorer on the same pool (0.877061); the
`b3-b6` union remains the mainline setting.

### SAM3 Fine-Tuning Budgets (weighted 3-route selector)

Weighted route selector over the original three route families
(`direct`, `one_bridge`, `two_bridges`):

| model | weighted Dice | direct | one_bridge | two_bridges | all-route mean | oracle |
|---|---:|---:|---:|---:|---:|---:|
| base_no_ft | 0.830103 | 0.767542 | 0.819048 | 0.761372 | 0.782654 | 0.879040 |
| ft_1pct | 0.874330 | 0.831090 | 0.856687 | 0.820172 | 0.835983 | 0.902679 |
| ft_5pct | 0.882543 | 0.814541 | 0.871664 | 0.837800 | 0.841335 | 0.911454 |
| ft_10pct | 0.844562 | 0.768585 | 0.846834 | 0.815985 | 0.810468 | 0.899441 |
| ft_20pct_latest_after_crash | 0.914321 | 0.856197 | 0.901934 | 0.875345 | 0.877825 | 0.931738 |

Longer-route weighted runs over 3-7 route families:

| model | weighted 3 | weighted 4 | weighted 5 | weighted 6 | weighted 7 | oracle 7 |
|---|---:|---:|---:|---:|---:|---:|
| base_no_ft | 0.830103 | 0.856771 | 0.849015 | 0.861283 | 0.864143 | 0.913565 |
| ft_1pct | 0.874330 | 0.888891 | 0.894013 | 0.901913 | 0.900234 | 0.928200 |
| ft_5pct | 0.882543 | 0.891466 | 0.883936 | 0.886997 | 0.895276 | 0.938849 |
| ft_10pct | 0.844562 | 0.889791 | 0.884999 | 0.887835 | 0.901470 | 0.932109 |
| ft_20pct_latest_after_crash | 0.914321 | 0.914936 | 0.914313 | 0.914289 | 0.910493 | 0.942206 |

Fine-tuning is useful; the 20% checkpoint is strongest, while smaller budgets
are not strictly monotonic.

## Current Status and Known Findings

### RouteCo-SAM3 v1 (see [routeco_sam3_v1.md](routeco_sam3_v1.md))

- Adapter-only training (memory-read adapter + mask-decoder LoRA) at 400
  steps is **roughly neutral on the oracle**: official-pipeline validation
  oracle goes `0.8694 -> 0.8614` (unified), per-mode `-0.002~-0.003`.
- Route-level behavior is a robustness/precision trade: low-dice routes
  improve, high-dice routes degrade (Spearman between frozen Dice and
  delta `~ -0.24`).
- As a **second candidate pool**, RouteCo has real upside: a perfect
  frozen-or-RouteCo per-target pick reaches `0.8778` vs frozen-only
  `0.8694`; 43% of routes are better with RouteCo.
- The `S->T` gate is inert because the student is not wired in yet
  (`student_variance = 0`, `gate_s_to_t = 1.0` for all steps). Wiring the
  student (weak/strong augmentation variance, routeco-vs-student
  disagreement) is the planned next step and is expected to act as both the
  gate and a new router feature.
- Earlier "oracle drop" numbers mixing the core and official forward
  pipelines overstate the effect; always compare same-pipeline evals.

### 1% GT + HQ Pseudo Fine-Tune (data quality audit)

- The HQ pseudo set is `8 GT + 357 pseudo` labels selected from the RouteCo
  warm-start manifest with `q_cycle >= 0.95`, route variance `<= 0.01`,
  disagreement `<= 0.05`, area ratio `0.001-0.6`.
- Protocol-level contamination: **zero** (all targets are train images, no
  validation/test GT used, anchors are exactly the 8 fixed supports).
- Label quality vs train GT: median Dice `0.956`, `85% >= 0.9`, but
  `10/357 < 0.5` and `16/357 < 0.7`.
- Those bad labels are "confidently wrong": they pass every route-internal
  signal (cycle consistency, multi-route disagreement, cross-mode IoU,
  cross-anchor IoU). Route-internal features cannot reliably filter them
  (best single feature AUROC ~0.71, threshold rules lose ~50% of good
  labels). Removing them requires an independent information source
  (single-image student audit / weak-strong augmentation variance).
- The only completed HQ fine-tune used `lr_scale=0.025` (4x lower than the
  1% GT repro at `0.1`) and its validation AP stayed flat
  (`0.6386 -> 0.6356`), so its low test Dice (`0.8367` route3) is
  confounded and not evidence against pseudo labels. All `lr_scale=0.1` HQ
  runs crashed with a matcher NaN (`ValueError: matrix contains invalid
  numeric entries`); a finite-guard matcher patch exists in
  `scripts/patch_remote_sam3_matcher_finite_guard.py`. A clean same-LR HQ
  comparison is still pending.

## Launch Commands

Prepare the protocol if missing:

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
# work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl
# work/kvasir_1pct_anchors/protocol/support_manifest.jsonl
cat work/kvasir_1pct_anchors/protocol/protocol_summary.json
```

Build frozen KNN routes up to `b7` for all five feature modes and evaluate
the base checkpoint:

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_stage1_b7_routes.sh
bash scripts/run_stage1_b7_eval_forward.sh
```

Build validation routes and run frozen forward eval for the router training
split:

```bash
bash scripts/run_stage1_b7_validation_eval_forward.sh
```

Propagation-quality feature extraction for the two main route generators:

```bash
bash scripts/run_propagation_quality_main_modes.sh
```

Summaries and router eval:

```bash
python3 scripts/summarize_candidate_invariant_v2.py
python3 scripts/analyze_propagation_quality_router.py
python3 scripts/eval_bridge_range_quality_router.py --min-bridge 3 --max-bridge 6
```

Evaluate a fine-tuned checkpoint (e.g. `ft_1pct`) with the propagation-quality
router over `direct..b6` and produce both the `b3-b6` mainline and `b0-b6`
reference summaries:

```bash
export CUDA_VISIBLE_DEVICES=1
bash scripts/run_eval_ft1pct_pq_router.sh
```

## Output Artifacts

```text
work/kvasir_1pct_anchors/
  protocol/                          frozen split + support manifests
  stage1_feature_knn_b7/             frozen per-bridge eval + quality features
  stage1_feature_knn_b7_ft1pct/      ft_1pct per-bridge eval + quality features
  model_routes/                      weighted 3-route summaries
  model_routes_max6/                 weighted 3-7 route summaries
  propagation_quality_router_v1/     first PQ router summary
  propagation_quality_router_b3_b6_check/   frozen mainline summary
  propagation_quality_router_ft1pct_b3_b6_check/  ft_1pct mainline summary
  propagation_quality_router_ft1pct_b0_b6_check/  ft_1pct b0-b6 reference
  summaries/                         consolidated tables (incl. ft1pct router md)
  routeco_sam3_v1/                   RouteCo training + pseudo manifest
```
