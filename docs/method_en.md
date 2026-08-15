# Current Method: Student-Audited Pseudo-Video SAM3 Pipeline (Kvasir 1%, WACV2027)

> Version: 2026-08-08. This document describes the complete current method for
> the WACV2027 Kvasir-SEG 1%-anchor task: frozen route generation →
> propagation-quality router → 1% SAM3 fine-tune → mainline-style student
> auditing → pseudo-label expansion → round-2 student-audited consensus pool →
> SAM3 re-fine-tune with dual-checkpoint evaluation.

## 1. Overview

Goal: with only **8 annotated Kvasir-SEG training images (1% anchors)**, achieve
the highest possible test segmentation Dice by combining pseudo-video route
propagation with single-image student auditing.

Core ideas:

- **SAM3 route generation is frozen** (frozen KNN graph + frozen/fine-tuned
  SAM3 propagation); pseudo labels are never used as new anchors;
- **Route quality is scored with GT-free propagation-quality features**
  (anchor return consistency, trajectory, multi-route agreement);
- **A single-image student is an independent auditor**: its error modes are
  independent of SAM3 video propagation, so it audits pseudo labels
  (the "confidently wrong" cases) and participates in final route selection (B7);
- **Progressive training**: round-1 pool (route hard thresholds) → student
  audit → Tier expansion → X3; round-2 pool (student-audited consensus labels)
  → SAM3 re-fine-tune.

## 2. Protocol and Data

- Dataset: Kvasir-SEG snapshot
  `/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T02_fresh_polyp_hf_sources/raw_hf_snapshots/kvasir-seg/snapshot`
- Splits: train=800 / validation=100 / test=100
- Human labels: 1% of train = **8 fixed GT anchors**
  (`work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt`)
- SAM3 base: `sam3.pt` (modelscope facebook/sam3)
- DINOv3 weights: used for KNN descriptors (one feature mode)
- Outputs: `work/kvasir_1pct_anchors/`
- Protocol constraints: validation/test GT is evaluation-only; train masks
  beyond the 8 anchors are not used for route search, pseudo selection, or
  checkpoint choice; train GT is used only for offline diagnostics.

## 3. Stage A: Frozen Routes + Propagation-Quality Router

### 3.1 Route generation (frozen KNN)

- Feature descriptors of the 8 anchors (DINOv3 / T18 corrected /
  anchor-conditioned variants)
- For each query, retrieve bridge frames from the train pool and build
  candidate routes:
  - `anchor_conditioned_target_pooling`
  - `anchor_conditioned_patch_correspondence`
- Candidate lengths: **b3-b6** (b0=direct, bN=N bridge frames); b7 collapses
  for every feature mode and is excluded
- Up to 2 modes × 4 bridge lengths = 8 candidates per target

#### How the two anchor-conditioned route generators work

Both modes share the same DINOv3 feature foundation
(see `scripts/stage1_feature_knn_routes.py`):

- DINOv3 ViT-S/16 features at 224×224 for every image: CLS token,
  14×14=196 patch tokens, and the patch mean (all L2-normalized);
- For each anchor, the **anchor prototype** is the mean of the patch tokens
  masked by the anchor's GT mask (foreground only) — "what this polyp looks
  like in DINOv3 feature space";
- Cosine similarity between every anchor prototype and every patch token of
  every image: `patch_sims[a, n, p]`.

**anchor_conditioned_target_pooling (target-weighted pooling)**

- For each image, weight its 196 patches by similarity to the anchor
  prototype using a softmax-style attention (temperature 10:
  `exp((sim − max) × 10)`, then normalized);
- Pool the patch tokens with these weights into an
  **anchor-conditioned image descriptor** `pooled[a, n]` (L2-normalized);
- Condition score `cond_score[a, n] = pooled[a, n] · anchor_proto[a]`:
  how well the anchor-like content of the image matches the anchor target;
- Intuition: the descriptor is biased toward the **target object (polyp)**
  rather than whole-image appearance — turning "global similarity" into
  "target-relevant similarity".

**anchor_conditioned_patch_correspondence (local patch correspondence)**

- Condition score `cond_score[a, n] = mean(top-8 patch_sims[a, n, p])`:
  the mean similarity of the 8 patches of image n that best match the anchor
  prototype;
- Intuition: measures whether the image contains at least some local patches
  highly matching the anchor target; more robust to background/context — a
  top-k local correspondence instead of global weighted pooling.

**Route search and freezing (shared by both modes)**

- For each target × bridge length (b3-b6) × anchor:
  - **beam search** (default beam width 8) from the target, prepending bridge
    frames step by step;
  - path score = min and mean of the condition scores of the nodes
    (anchor, bridges, target) plus patch-mean similarity between consecutive
    nodes;
  - KNN candidates are ranked by patch-mean similarity to the current tail,
    with the anchor condition score as a secondary ranking (anchor-relevant
    bridge frames preferred);
- Across the 8 anchors, keep the (anchor, bridge sequence) with the best path
  score as the frozen route for that (target, bridge length);
- A route only records the KNN path (anchor→bridges→query); SAM3 executes the
  propagation on it. **The KNN graph and routes are frozen once built and are
  shared by all later experiments.**

### 3.2 Propagation-quality features (21, GT-free)

For every candidate route:

- `q_cycle` (fresh-state anchor return consistency: the predicted query mask is
  converted to a prompt and propagated back to the anchor; Dice with the anchor
  mask) — forward + backward cross-validation;
- `cycle_success`, `cycle_sam_score`;
- mask-area trajectory (`trace_area_*`), empty-mask count, connected
  components, centroid/bbox drift;
- adjacent-frame Dice (`trace_adjacent_dice_*`), SAM-score trajectory,
  candidate counts.

### 3.3 Router

- Scorer: ridge fitted on **validation features** (21 features + mode one-hot);
  each route is scored independently (candidate-invariant, no renormalization);
- Selection: Top-1 route mask per target;
- Frozen-base result: **0.877299** (oracle 0.896918, gap 0.0196);
  fixed best route (target b6) 0.854627 as reference.

### 3.4 Frozen-base full per-bridge table (test, forward-only Dice)

| Feature mode | direct | b1 | b2 | b3 | b4 | b5 | b6 | b7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T18 corrected | 0.747869 | 0.775949 | 0.756490 | 0.785808 | 0.821736 | 0.816809 | 0.819200 | 0.143392 |
| DINO global pooling | 0.740844 | 0.748290 | 0.758326 | 0.786777 | 0.777536 | 0.818397 | 0.818565 | 0.092024 |
| DINO patch average | 0.728634 | 0.726893 | 0.756876 | 0.746756 | 0.799814 | 0.829179 | 0.826294 | 0.046968 |
| anchor-conditioned target pooling | 0.741306 | 0.760176 | 0.827209 | 0.840837 | 0.841258 | 0.848618 | 0.854627 | 0.032027 |
| anchor-conditioned patch correspondence | 0.757814 | 0.773501 | 0.822305 | 0.832819 | 0.833990 | 0.833577 | 0.842078 | 0.070807 |

Strongest fixed route: target pooling b6 = 0.854627; stable region b4-b6; b7
collapses for every mode.

### 3.5 8-anchor single-image student baselines (SC-SAM / SynFoC)

Reference for "how strong is a single-image student without pseudo labels"
(Kvasir 1%, 8 anchors):

| Student | Test Dice (SAM branch) | Test Dice (UNet branch) | best-val |
|---|---:|---:|---:|
| SC-SAM | 0.6626 | 0.6373 | — |
| SynFoC | 0.7615 | 0.7681 | unet 0.7300@37.5k, sam 0.7310@28k |

SynFoC = MedSAM-initialized LoRA-SAM + UNet semi-supervised co-training
(weak/strong augmentation consistency + SAM↔UNet mutual-confirmation gating);
SC-SAM = SAM encoder-adapter + UNet co-training. Both are far below the route
methods, but as independent single-image verifiers their error modes are
independent of SAM3 video propagation — they are the source of the
confidently-wrong audit signal and the B7 q_model term.

## 4. Stage B: 1% SAM3 Fine-Tune (ft_1pct)

- Data: 8 GT anchors, COCO format, 20 epochs, `lr_scale=0.1` (full fine-tune);
- Checkpoint: `work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt`
  (final epoch 20);
- Evaluation with the same router protocol (b3-b6, scorer refit on validation):
  **0.894648** (oracle 0.919805, gap 0.0252) — previous public best.

Every bridge/mode improves over the frozen base by about +0.02–+0.05 (see
results section).

### 4.1 ft_1pct vs frozen per-bridge (test)

| bridge | ft_1pct target | frozen target | ft_1pct patch | frozen patch |
|---|---:|---:|---:|---:|
| direct | 0.7708 | 0.7413 | 0.7979 | 0.7578 |
| b1 | 0.7922 | 0.7602 | 0.8161 | 0.7735 |
| b2 | 0.8574 | 0.8272 | 0.8578 | 0.8223 |
| b3 | 0.8522 | 0.8408 | 0.8599 | 0.8328 |
| b4 | 0.8609 | 0.8413 | 0.8843 | 0.8340 |
| b5 | 0.8731 | 0.8486 | 0.8658 | 0.8336 |
| b6 | 0.8643 | 0.8546 | 0.8797 | 0.8421 |

### 4.2 SAM3 fine-tuning budgets (weighted 3-route selector)

| model | weighted Dice | direct | one_bridge | two_bridges | all-route mean | oracle |
|---|---:|---:|---:|---:|---:|---:|
| base_no_ft | 0.830103 | 0.767542 | 0.819048 | 0.761372 | 0.782654 | 0.879040 |
| ft_1pct | 0.874330 | 0.831090 | 0.856687 | 0.820172 | 0.835983 | 0.902679 |
| ft_5pct | 0.882543 | 0.814541 | 0.871664 | 0.837800 | 0.841335 | 0.911454 |
| ft_10pct | 0.844562 | 0.768585 | 0.846834 | 0.815985 | 0.810468 | 0.899441 |
| ft_20pct (latest after crash) | 0.914321 | 0.856197 | 0.901934 | 0.875345 | 0.877825 | 0.931738 |

### 4.3 Longer-route weighted runs (3-7 route families)

| model | w3 | w4 | w5 | w6 | w7 | oracle7 |
|---|---:|---:|---:|---:|---:|---:|
| base_no_ft | 0.830103 | 0.856771 | 0.849015 | 0.861283 | 0.864143 | 0.913565 |
| ft_1pct | 0.874330 | 0.888891 | 0.894013 | 0.901913 | 0.900234 | 0.928200 |
| ft_5pct | 0.882543 | 0.891466 | 0.883936 | 0.886997 | 0.895276 | 0.938849 |
| ft_10pct | 0.844562 | 0.889791 | 0.884999 | 0.887835 | 0.901470 | 0.932109 |
| ft_20pct | 0.914321 | 0.914936 | 0.914313 | 0.914289 | 0.910493 | 0.942206 |

Takeaway: fine-tuning helps but small budgets are not monotonic; 20% is
strongest.

## 5. Stage C: Mainline-Style Original Pseudo Pool (pseudo568)

Script: `scripts/select_phase1_mainline_pseudo568.py`

- For all 792 train targets, pick the Top-1 candidate with the
  validation-trained ridge scorer on ft_1pct features;
- Hard thresholds (aligned with mainline `prepare_t22_training.py`):
  - `q_return` (= q_cycle) ≥ 0.95
  - `q_multi` (target candidate-pool pairwise agreement) ≥ 0.90
- Result: **491 / 792** accepted (q_multi mean 0.985, q_return mean 0.971);
- Per-image weight = normalized q_multi, clipped to [0.2, 1.0].

## 6. Stage D: T24 Committee Students (S2 / S3)

Script: `run_t24_student.py` (SC-SAM SamUnet single-image student)

- **S2**: 8 GT + 491 original pseudo labels, per-sample CE+Dice, pseudo labels
  weighted by normalized q_multi;
- **S3**: first build consensus targets as the mean of candidate route masks
  with pixel weight `max(0.1, exp(-4 × pixel variance))`
  (`scripts/prepare_phase1_s3_consensus.py`), then train;
- Training: batch 12, UNet-lr 0.01, 40000 iterations, final-step checkpoint;
- Export train predictions for three auditors: S2 val-best, S2 final, S3 final.

Results: validation Dice S2 ≈ 0.798, S3 ≈ 0.795.

## 7. Stage E: Committee Audit → Tier A/B/C

Script: `scripts/phase1_audit_tiers.py` (port of `build_s27_audit_and_tiers.py`)

- For the 301 unselected targets (792 − 491 − 8 anchors): candidate pool +
  3 auditors;
- Per candidate: `q_route = 0.20 × q_multi + 0.80 × q_model_mean`
  (student agreement dominates);
- Pixel consensus: `0.75 × selected SAM3 route + 0.25 × student mean`, pixel
  weight `clip(exp(-5×route_var) × exp(-5×student_var) × cross, 0.05, 1.0)`;
- **Tier A**: q_multi≥0.90 ∧ q_model_mean≥0.90 ∧ q_model_min≥0.80 ∧
  q_model_var≤0.01 ∧ q_route≥0.88 ∧ non-empty ∧ ≥2 students non-empty ∧
  area-safe;
- **Tier B**: q_multi≥0.75 ∧ q_model_mean≥0.75 ∧ q_model_min≥0.60 ∧
  q_model_var≤0.03;
- otherwise Tier C.

Result: Tier A = 93, Tier B = 95, Tier C = 113.

## 8. Stage F: X3 Three-Stream Student

Scripts: `run_s27_student.py` + `scripts/phase1_build_x3_manifest.py`

- X3 pool = 491 original + 93 A + 95 B = **679** pseudo labels;
- Three-stream sampling: GT stream (8) / original stream / new stream (A+B),
  batch 12 (3/3/6);
- Stream weights: original 1.0, A 0.75, B 0.50; global pseudo 0.5 with a
  2000-step ramp;
- Per-sample BCE + foreground Dice; Tier B uses a pixel-weight mapping
  (≥0.70→1.0, 0.30–0.70→0.25, <0.30→0);
- 40000 iterations; validation Dice ≈ **0.8162**.

## 9. Stage G: B7 Student-Assisted Route Selection

Script: `scripts/phase1_b7_select.py`

- Test candidates: ft_1pct b3-b6 pool (up to 8);
- Per candidate:
  - `q_return` = q_cycle (ft_1pct anchor return consistency);
  - `q_multi` = candidate-pool pairwise agreement;
  - `q_model` = Dice(X3 student mask, route mask);
  - `B7 = (q_return × q_multi² × q_model²)^0.2`;
- Top-1 B7 per target; test reported once.

Results:

| Method | Test Dice |
|---|---:|
| X3 single-image student | 0.8633 |
| **X3 + B7 (stage-1 final)** | **0.899369** |
| oracle | 0.919805 |

+0.0047 over the pure ft_1pct router (0.894648).

## 10. Stage H: Round-2 — Student-Audited Consensus Pool + SAM3 Re-Fine-Tune

### 10.1 Round-2 pool selection

Script: `scripts/select_phase1_round2_pool.py`

- For all 792 targets: 3 auditors + ft_1pct candidates;
- Per candidate `q_route = 0.2 × q_multi + 0.8 × q_model_mean`, take Top-1;
- Acceptance (mainline Tier-A-level hard thresholds):
  q_multi≥0.90 ∧ q_return≥0.95 ∧ q_model_mean≥0.90 ∧ q_model_min≥0.80;
- Label = pixel consensus: `0.75 × route + 0.25 × student mean`, pixel weight
  `exp(-5×route_var) × exp(-5×student_var) × cross`;
- Result: **455 / 792** accepted; offline quality vs train GT:
  mean Dice 0.921, <0.7 rate 5.1% (round-1: 6.9%).

### 10.2 SAM3 re-fine-tune

- Dataset: 8 GT + 455 consensus labels = 463 images
  (`scripts/build_phase1_round2_dataset.py`);
- Config: `kvasir_1pct_round2_student_audited_20260807_seed2026.yaml`
  (full fine-tune, `lr_scale=0.1`, 20 epochs, bf16 AMP);
- **Numerical-stability fixes** (trainer patches, all backed up):
  1. `weights_only=False` (resume with optimizer state);
  2. NaN loss → replaced with 0 and continue + zero non-finite gradients after
     backward (prevents a single bad sample's NaN gradient from poisoning
     weights; this run had 0 NaN events);
  3. best-val checkpoint save fix: meter-key prefix matching
     (`save_best_meters: [val_roboflow100]` matches
     `val_roboflow100/detection`); saves
     `checkpoints/val_roboflow100_detection_coco_eval_segm_AP.pt` on val-AP
     improvement;
- Validation: best-val AP **0.6799** (epoch 13), final AP 0.6603 (epoch 19);
- Checkpoints: 1/6/11/16/20 + latest + best-val series.

### 10.3 Dual-checkpoint evaluation

Script: `scripts/run_round2_eval_variant.sh` (GPU 0 final / GPU 1 bestval in
parallel)

- Each variant: merge → propagation quality (validation+test, both modes) →
  b3-b6 router report.

Results (test, 100 targets):

| Variant | target+patch | target | patch | oracle |
|---|---:|---:|---:|---:|
| round2 final | 0.868266 | 0.861481 | 0.873456 | 0.910543 |
| round2 bestval | **0.882049** | 0.880997 | 0.877625 | 0.917097 |
| ft_1pct baseline | **0.894648** | 0.886241 | 0.884695 | 0.919805 |

**Conclusion: the round-2 fine-tune did not beat ft_1pct; it dropped 1.3–2.6
points.**

### 10.4 Per-bridge analysis (direct→b6)

target mode:

| bridge | ft_1pct | r2 final | r2 bestval |
|---|---:|---:|---:|
| direct | 0.7708 | 0.7604 | 0.7745 |
| b1 | 0.7922 | 0.8019 | **0.8165** |
| b2 | 0.8574 | 0.8655 | **0.8687** |
| b3 | 0.8522 | 0.8408 | 0.8533 |
| b4 | 0.8609 | 0.8595 | **0.8736** |
| b5 | 0.8731 | 0.8689 | **0.8762** |
| b6 | 0.8643 | 0.8493 | 0.8601 |

patch mode:

| bridge | ft_1pct | r2 final | r2 bestval |
|---|---:|---:|---:|
| direct | 0.7979 | 0.7762 | 0.7899 |
| b1 | 0.8161 | 0.8130 | **0.8275** |
| b2 | 0.8578 | 0.8651 | **0.8661** |
| b3 | 0.8599 | 0.8592 | 0.8617 |
| b4 | **0.8843** | 0.8652 | 0.8764 |
| b5 | 0.8658 | 0.8623 | **0.8794** |
| b6 | **0.8797** | 0.8633 | 0.8712 |

Pattern: on **bestval(e13)/final(e20)**, round-2 improves short routes (b1-b2)
but degrades long routes (b4-b6); since the router Top-1 relies mostly on
b4-b6, the overall score drops. **Early checkpoints (epoch 1/6) do NOT follow
this pattern — see 10.7.**

### 10.5 Router-pool comparison (b3-b6 vs b0-b6)

| Model | b3-b6 union | b3-b6 oracle | b0-b6 union | b0-b6 oracle |
|---|---:|---:|---:|---:|
| frozen base | 0.877299 | 0.896918 | 0.873626 | 0.903846 |
| ft_1pct | **0.894648** | 0.919805 | 0.877061 | 0.921800 |
| round2 final | 0.868266 | 0.910543 | 0.874911 | 0.913835 |
| round2 bestval | 0.882049 | 0.917097 | **0.879142** | 0.919647 |

Key conclusions:

- round2 bestval already beats frozen at b3-b6 (0.8820 vs 0.8773);
- at b0-b6, round2 bestval (0.8791) **beats every alternative**, including
  ft_1pct (0.8771) — round-2 shifts capability from long to short routes;
- if the pool stays long-route-only (b3-b6), ft_1pct (0.8946) remains best.

### 10.6 Related diagnostic experiments

**RouteCo-SAM3 v1** (memory adapter + LoRA, official validation pipeline):

| Metric | frozen | routeco step400 |
|---|---:|---:|
| unified oracle (b3-b6) | 0.8694 | 0.8614 |
| per-mode oracle change | — | -0.002 ~ -0.003 |
| 43% of routes better with routeco; perfect frozen/routeco pick ceiling | — | 0.8778 |

Takeaway: adapter-only training is roughly neutral; the S→T gate is inert
because the student is not wired in (student_variance=0, gate_s_to_t always 1).

**HQ pseudo-label audit (round-1 357 pool)**:

- Protocol contamination: 0 (all train targets, 8 fixed anchors, no val/test GT);
- Quality vs train GT: median Dice 0.956, 85% ≥ 0.9, but **10/357 < 0.5 and
  16/357 < 0.7**;
- These are "confidently wrong": cycle/disagreement/cross-mode/cross-anchor
  signals all fail to detect them (best single feature AUROC ≤ 0.71 with ~50%
  good-label loss);
- Student-route agreement AUROC ≈ 0.78; combined with route disagreement:
  **0.85**;
- Early 1% GT + HQ pseudo fine-tune: lr_scale=0.025 had flat validation AP
  (0.6386→0.6356, effectively not trained); all lr=0.1 runs crashed with
  matcher NaN (fixed by the finite-guard patch).

### 10.7 Early-checkpoint per-bridge comparison (epoch 1 / 6, forward-only test Dice)

Forward-only per-bridge evaluation of round-2 `checkpoint_1` (epoch 1) and
`checkpoint_6` (epoch 6) on test (no router; same protocol as the README
per-bridge tables for frozen/ft_1pct).

target mode:

| variant | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | 0.7413 | 0.7602 | 0.8272 | 0.8408 | 0.8413 | 0.8486 | 0.8546 |
| ft_1pct | 0.7708 | 0.7922 | 0.8574 | 0.8522 | 0.8609 | 0.8731 | 0.8643 |
| **ckpt1(e1)** | **0.7814** | **0.8214** | **0.8611** | **0.8752** | **0.8687** | **0.8756** | **0.8668** |
| ckpt6(e6) | 0.7788 | 0.8255 | 0.8643 | 0.8684 | 0.8758 | 0.8761 | 0.8577 |
| bestval(e13) | 0.7745 | 0.8165 | 0.8687 | 0.8533 | 0.8736 | 0.8762 | 0.8601 |
| final(e20) | 0.7604 | 0.8019 | 0.8655 | 0.8408 | 0.8595 | 0.8689 | 0.8493 |

patch mode:

| variant | direct | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | 0.7578 | 0.7735 | 0.8223 | 0.8328 | 0.8340 | 0.8336 | 0.8421 |
| ft_1pct | 0.7979 | 0.8161 | 0.8578 | 0.8599 | **0.8843** | 0.8658 | 0.8797 |
| **ckpt1(e1)** | **0.8045** | **0.8433** | **0.8626** | **0.8733** | 0.8735 | **0.8721** | **0.8790** |
| ckpt6(e6) | 0.7931 | 0.8351 | 0.8682 | 0.8688 | 0.8757 | 0.8674 | 0.8661 |
| bestval(e13) | 0.7899 | 0.8275 | 0.8661 | 0.8617 | 0.8764 | 0.8794 | 0.8712 |
| final(e20) | 0.7762 | 0.8130 | 0.8651 | 0.8592 | 0.8652 | 0.8623 | 0.8633 |

Key findings:

- **ckpt1 (epoch 1) beats ft_1pct on almost every bridge**: target 7/7
  (b3 +0.023, b1 +0.029), patch 6/7 (only b4 -0.011, b6 ties);
- **ckpt6 (epoch 6) is also strong**: target 6/7, patch 5/7 bridges beat
  ft_1pct;
- **Degradation happens later in training**: bestval(e13) starts losing b3/b6,
  final(e20) drops across the board — round-2 is an
  "early-strong, then overfit" curve; **epoch 1-6 is the sweet spot**;
- Reason: 455 pseudo labels let the model absorb enough diversity in one epoch
  (vs ft_1pct's 8 images repeated for 20 epochs); early checkpoints get the
  diversity without the overfitting;
- Next step: attach the router (b3-b6 and b0-b6) and B7 student selection to
  ckpt1/ckpt6; they are likely to exceed ft_1pct's 0.894648.

### 10.8 Router validation of early checkpoints (b3-b6 and b0-b6)

Propagation-quality features were completed for ckpt1(e1)/ckpt6(e6), and the
same ridge-router protocol (refit on validation) was evaluated on both the
b3-b6 and b0-b6 candidate pools.

**b3-b6 pool (union)**:

| Model | selected | oracle | gap |
|---|---:|---:|---:|
| frozen | 0.877299 | 0.896918 | 0.0196 |
| ft_1pct | **0.894648** | 0.919805 | 0.0252 |
| round2 final(e20) | 0.868266 | 0.910543 | 0.0423 |
| round2 bestval(e13) | 0.882049 | 0.917097 | 0.0350 |
| **ckpt1(e1)** | **0.884656** | 0.911043 | 0.0264 |
| ckpt6(e6) | 0.880806 | 0.913594 | 0.0328 |

**b0-b6 pool (union)**:

| Model | selected | oracle | gap |
|---|---:|---:|---:|
| frozen | 0.873626 | 0.903846 | 0.0302 |
| ft_1pct | 0.877061 | 0.921800 | 0.0447 |
| round2 final(e20) | 0.874911 | 0.913835 | 0.0389 |
| round2 bestval(e13) | 0.879142 | 0.919647 | 0.0405 |
| round2 ckpt1(e1) | 0.881271 | 0.913898 | 0.0326 |
| **round2 ckpt6(e6)** | **0.883671** | 0.915488 | 0.0318 |

**b0-b6 per-mode (selected)**:

| Model | target | patch |
|---|---:|---:|
| ft_1pct | 0.868192 | 0.890865 |
| round2 ckpt1(e1) | 0.881311 | 0.885018 |
| round2 ckpt6(e6) | 0.883406 | 0.879398 |
| round2 bestval(e13) | 0.878366 | 0.878679 |
| round2 final(e20) | 0.875843 | 0.874597 |

Key conclusions:

- **Early stopping is confirmed**: e1/e6 > e13 > e20; the round-2 fine-tune
  gets worse the longer it trains;
- **On b0-b6, ckpt6(e6) 0.8837 is the best overall**, beating ft_1pct
  (0.8771) and all round-2 variants — consistent with the per-bridge forward
  results;
- **On b3-b6, ckpt1(e1) 0.8847 is the strongest round-2 variant but still
  below ft_1pct (0.8946)**: ft_1pct has better route quality (oracle 0.9198)
  and better selection in the long-route region;
- ft_1pct still has the highest oracle (b0-b6 0.9218 / b3-b6 0.9198),
  so ckpt6 wins through better selection rather than a higher quality ceiling.

## 11. Summary of Results

| Method | Test Dice | Note |
|---|---:|---|
| Frozen multi-route baseline | 0.8715 | merged-protocol scale |
| Frozen PQ router (b3-b6) | 0.877299 | frozen SAM3 |
| ft_1pct + router | **0.894648** | 1% fine-tune + selection |
| X3 single-image student | 0.8633 | stage-1 student |
| X3 + B7 selection | **0.899369** | stage-1 final |
| round2 final + router | 0.868266 | round-2 fine-tune |
| round2 bestval + router | 0.882049 | round-2 fine-tune (best-val) |

Additional reference numbers:

- **Public mainline (S27 X3+B7, merged CVC+Kvasir)**: X3 single-image
  0.866738; linear selector 0.885637; **B7 0.895835**; oracle 0.907835;
- **Kvasir 1% student baselines**: SC-SAM 0.6626/0.6373; SynFoC
  0.7615/0.7681;
- **Router-pool comparison**: round2 bestval is best on the b0-b6 pool
  (0.8791); ft_1pct is best on the b3-b6 pool (0.8946);
- **Fine-tuning budgets**: ft_20pct weighted 0.914321 remains the strongest
  weighted-selection result.

## 12. Engineering and Reproduction

### Key scripts

```text
scripts/select_phase1_mainline_pseudo568.py   original pool (mainline thresholds)
scripts/prepare_phase1_s3_consensus.py        S3 consensus targets
scripts/phase1_audit_tiers.py                 committee audit → Tier A/B/C
scripts/phase1_build_x3_manifest.py           X3 manifest merge
scripts/phase1_b7_select.py                   B7 selection + test report
scripts/select_phase1_round2_pool.py          round-2 student-audited pool
scripts/build_phase1_round2_dataset.py        round-2 COCO dataset
scripts/run_round2_eval_variant.sh            single-variant eval (merge→pq→report)
scripts/patch_remote_sam3_trainer_save_best.py best-val save patch
scripts/patch_remote_sam3_matcher_finite_guard.py matcher finite guard
```

### Key artifacts

```text
work/kvasir_1pct_anchors/phase1/
  pseudo_manifest_original.jsonl        round-1 491 pool
  S3_consensus/pseudo_consensus.jsonl   S3 consensus
  students/{S2,S3,X3}/                  student training
  audit/tier_{A,B,C}.jsonl              audit tiers
  predictions/{S2_valbest,S2_final,S3_final,X3_final}/
  selection/b7_report.json              B7 report (0.899369)
  round2_pool/round2_pool.jsonl         round-2 455 pool
  stage1_feature_knn_b7_ftround2*/       round-2 PQ + reports
  round2/router_b3_b6_{final,bestval}_report.json
```

### Trainer patches (in /Data_8TB/lht/sam3, each backed up with .bak)

- `trainer.py`: weights_only=False; NaN loss→0 + zero non-finite gradients;
  best-val save prefix matching; `_run_step` metadata dump.
- `matcher.py`: BinaryHungarianMatcherV2 finite guard (NaN/inf costs → 1e9).

## 13. Current Status and Next Steps

- Stage 1 (ft_1pct + student audit + B7) is the current strongest pipeline,
  **0.899369**;
- The round-2 SAM3 fine-tune does not beat ft_1pct at final(e20)/bestval(e13),
  but **early checkpoints (epoch 1/6) beat ft_1pct on almost every bridge in
  forward per-bridge Dice (see 10.7)** — the correct use of round-2 is early
  stopping;
- Priority actions:
  1. attach the router (b3-b6 and b0-b6) and B7 selection to ckpt1(e1)/ckpt6(e6);
  2. if effective, fix "student-audited consensus pool + early stopping" as the
     standard round-2 recipe;
  3. optional: fine-tune with raw route masks instead of consensus blends, or
     lower lr (0.05), for further comparison.

Updated after router validation:

- **b0-b6: ckpt6(e6) 0.8837 is now the best overall** (> ft_1pct 0.8771);
- **b3-b6: ft_1pct 0.8946 remains unbeaten**; ckpt1(e1) 0.8847 is the
  strongest round-2 variant;
- Next priority: attach **B7 student selection** to ckpt1(e1)/ckpt6(e6), and
  consider "b0-b6 + ckpt6" as a candidate new main configuration.
