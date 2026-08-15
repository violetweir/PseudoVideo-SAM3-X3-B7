# Phase 1: Kvasir 1% Student-Audited Pseudo Labels (ft_1pct Router + S27-Style Student)

## Objective

Transfer the public mainline (S27 X3 + B7) student-training machinery to the
WACV2027 Kvasir 1% protocol, on top of the current best SAM3 method:
`ft_1pct` + b3-b6 propagation-quality router (`0.894648` test Dice).

The router is the best available pseudo-label and route-selection engine, but
it is selection-limited (oracle `0.919805`, gap `0.025`).  The mainline has
already shown that a single-image student auditor adds an independent signal
(`q_model`) that improves route selection by ~2.5 points on the merged
protocol.  Phase 1 reproduces that loop on Kvasir 1% and, in parallel,
uses the student as an auditor for the "confidently wrong" pseudo labels.

## Baseline

- Protocol: Kvasir-SEG only, train=800 / validation=100 / test=100, 8 fixed
  GT anchors (`work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt`).
- SAM3: `ft_1pct` full fine-tune (8 GT, epoch 20, 160 train steps), merged at
  `work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt`.
- Router: validation-trained ridge over 21 propagation-quality features,
  candidates = `anchor_conditioned_target_pooling` +
  `anchor_conditioned_patch_correspondence`, lengths `b3-b6` only.
- Best result: `0.894648` (oracle `0.919805`).

## Mainline Machinery Being Transferred

From the public S27 X3+B7 mainline (scripts + README):

1. **T24 committee students**: train 3 single-image `SamUnet` students
   (SC-SAM UNet branch, `Model/model.py`) from the fixed high-confidence
   pseudo pool, 40000 iterations, validation every 200.
2. **Audit into tiers** (`build_s27_audit_and_tiers.py`): for each remaining
   target, combine frozen SAM3 routes + student predictions + SAM probability
   maps into pixel-consensus maps and per-image scores; assign Tier A/B/C.
3. **S27 X3 three-stream training** (`run_s27_student.py`): GT stream +
   original pseudo stream + tier stream, per-stream batch sizes and weights
   (original 1.0, tier A 0.75, tier B 0.50, global pseudo 0.5 with 2000-step
   ramp), per-sample BCE + foreground Dice, final-40k checkpoint.
4. **B7 selection**: `(q_return * q_multi^2 * q_model^2)^0.2`, where
   `q_model` is student-vs-route Dice.

## Phase 1 Pipeline

```text
Stage 0  ft_1pct + b3-b6 propagation-quality router (existing)
           -> per train target: Top-1 pseudo mask + quality features + 8 candidates
Stage 1  initial student (T24 analog)
           -> 8 GT + HQ pseudo pool (weighted by student-route agreement)
           -> 3 x SamUnet students, 40000 iters, validation-only selection
Stage 2  committee audit (build_s27_audit_and_tiers analog)
           -> student-route disagreement + route disagreement + aug variance
              (validated combo AUROC 0.85 on the HQ 357 audit set)
           -> pixel consensus maps + Tier A/B/C + per-image weights
Stage 3  X3 three-stream student (S27 analog)
           -> 8 GT + original HQ + Tier A/B, gt-bs 3 / original-bs 3 / new-bs 6
Stage 4  B7-style route selection on ft_1pct b3-b6 candidates
           -> q_return (ft_1pct cycle) + q_multi (candidate-pool consensus)
              + q_model (X3 student vs route mask)
           -> single test report
```

Differences from the public mainline:

- `q_multi` is defined over the 8-candidate pool (`1 - route disagreement`)
  instead of 3 routes.
- The initial pseudo pool is the router-selected HQ set (357) rather than 568.
- Student-route agreement is used both as a Stage-1 training weight and as a
  Stage-2 audit signal (the confident-error mitigation).

## Stage 0/1 Artifacts

```text
work/kvasir_1pct_anchors/phase1/
  pseudo_manifest_original.jsonl   S27-compatible original pool (sample_type)
  students/S1*/                    initial students (train.jsonl, student_final.pth)
  audit/                           committee scores + tiers
  students/X3/                     final three-stream student
  selection/                       B7-style test report
```

## Hyperparameters (Stage 1 initial student)

```text
model          SamUnet (SC-SAM UNet branch)
input          256x256 RGB, ImageNet normalization
optimizer      SGD momentum 0.9, weight decay 1e-4
lr             UNet-lr 0.01, linear decay to 0
iterations     40000
val interval   200
streams        gt-bs 3, original-bs 3, new-bs 0 (Stage 1)
loss           per-sample CE + foreground Dice on GT;
               per-sample CE + foreground Dice on original with image weight
quality weight explicit_quality_weight = clip(0.25 + 0.75 * min(sam_route,
               unet_route), 0.05, 1.0)
checkpoint     final-40k; validation-only (test never used for selection)
```

## Expected Outcomes

- A single-image student whose validation Dice is a faithful lower-bound
  baseline for the Kvasir 1% protocol (SC-SAM/SynFoC baselines: 0.64-0.77).
- Stage 2 audit should separate the 16 confidently-wrong HQ labels from the
  341 good ones better than route-internal features (combo AUROC target >= 0.85).
- Stage 4 B7-style selection should exceed `0.894648` by using the student
  as an independent route-quality signal.

## Protocol Constraints

- Validation/test GT is evaluation-only; never used for pseudo selection,
  tier assignment, or checkpoint choice.
- Train masks beyond the 8 anchors are used for offline audit diagnostics
  only (they are not part of the deployed selection rule).
