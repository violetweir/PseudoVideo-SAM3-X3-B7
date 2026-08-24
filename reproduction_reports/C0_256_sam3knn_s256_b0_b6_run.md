# C0-256-base → SAM3-KNN@256, routes b0-b6

## Objective

Run a controlled continuation of `C0_256_base` with the KNN feature extractor
changed from DINOv3 ViT-S/16 at 224 to the frozen base SAM3 image trunk at
256.  Preserve the original anchors, data splits, beam width, propagation
checkpoint, propagation canvas, student architectures, seeds, and selection
logic.

The route pool is intentionally expanded to `direct/bridge_0` through
`bridge_6` for train, validation, and test.  No b0-b2 result is discarded.

## Controlled variables

| component | C0-256-base | this run |
|---|---|---|
| KNN backbone | DINOv3 ViT-S/16 | base SAM3 image trunk |
| feature input | 224x224 | 256x256 |
| descriptor | L2-normalized patch mean | L2-normalized patch mean |
| descriptor width | 384 | 1024 |
| anchors | frozen eight human masks | unchanged |
| route modes | anchor-conditioned target/patch | SAM3-encoder analogues |
| beam width | 32 | 32 |
| propagation model | base `sam3.pt` | unchanged |
| propagation canvas | 256 | 256 |
| saved bridge range | train b3-b6; val/test b0-b6 | all splits b0-b6 |

SAM3 feature checkpoint:

```text
/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
```

Feature cache:

```text
work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/features/
sam3_base_s256_features.npz
```

The cache records `feature_source=sam3_base`, `feature_size=256`, eight
1024-dimensional anchor prototypes, and 1000 1024-dimensional patch-mean
descriptors.  Its SHA-256 is
`ab8ca194f9c789ad9f6b641d382ed848751f29b23ac0bdb29a9c6722fd599907`.

## Experiment layout

```text
work/rerun_c0_256_sam3knn_s256_base/
└── stage1_feature_knn_b0_b6/
    ├── features/
    ├── sam3enc_anchor_conditioned_target_pooling/
    └── sam3enc_anchor_conditioned_patch_correspondence/
```

Compatibility aliases with the old DINO-era mode names are created only
inside this isolated root.  This lets the frozen downstream scripts read the
new candidates without changing or overwriting any C0-256-base artifact.

## Downstream changes required

1. Generate and retain b0-b6 routes for every split.  Expected counts per
   mode are 5544 train routes, 700 validation routes, and 700 test routes.
2. Run SAM3 propagation on all seven bridge levels and report per-level Dice
   and cycle consistency, not only the B7-selected scalar.
3. Pass `--min-bridge 0 --max-bridge 6` explicitly to the Router, original
   pseudo-label selector, S3 consensus, committee audit, and final B7 selector.
4. Also compute a b3-b6 view offline from the same outputs for an
   apples-to-apples comparison with C0-256-base; do not rerun propagation.
5. Rebuild S2/S3, committee audit, X3 manifest, and X3 predictions because the
   route masks and pseudo-label pool change.  Old student predictions cannot
   be used as the official final selector.
6. Select checkpoints and protocol choices on validation.  Run official test
   once after the new X3 validation-best checkpoint is frozen.
7. Preserve ordinary validation Dice during all student/SAM3 training.  For
   any later SAM3 fine-tuning, log both loss and direct Dice at every epoch.

## Calibration points caused by b0-b6

Expanding each target from eight candidates (two modes x b3-b6) to fourteen
candidates (two modes x b0-b6) changes several score distributions even when
the underlying masks do not change.  The following values must therefore be
checked on validation rather than silently inherited:

- `q_multi`: now averages agreement against thirteen peers instead of seven;
  the old `tau_multi=0.90` acceptance threshold may change coverage.
- Router ridge features: refit on the new validation candidate pool and report
  both b0-b6 and b3-b6 selections.
- S3 pixel consensus/variance: now aggregates fourteen masks; retain the
  frozen `beta=4` first, but audit probability and weight distributions.
- Committee audit: route variance and top-route bridge histograms change;
  report Tier A/B/C counts before accepting the old tier thresholds.
- B7 `q_multi` and geometric score distributions: recompute from the complete
  candidate pool and never compare `b7_mean` with GT Dice.  The official
  performance field remains selected GT Dice on validation/test.

## Current execution

Route generation:

```text
scripts/run_c0_256_sam3knn_s256_routes.sh
work/rerun_c0_256_sam3knn_s256_base/routes.log
```

Validation-first propagation on GPU1:

```text
scripts/supervise_c0_256_sam3knn_s256_validation.sh
work/rerun_c0_256_sam3knn_s256_base/validation_propagation.log
```

Test propagation is deliberately not launched at this stage.

## Validation result (base SAM3 propagation)

Both route modes completed 700 validation routes at canvas 256.  The table is
the mean GT Dice across the two modes.

| route | DINOv3 ViT-S@224 | SAM3 trunk@256 | delta |
|---|---:|---:|---:|
| direct / b0 | 0.745905 | 0.762231 | +0.016326 |
| b1 | 0.778472 | 0.795550 | +0.017078 |
| b2 | 0.817718 | 0.802799 | -0.014919 |
| b3 | 0.811044 | 0.793372 | -0.017673 |
| b4 | 0.833340 | 0.788871 | -0.044469 |
| b5 | 0.825969 | 0.813166 | -0.012803 |
| b6 | 0.828536 | 0.816890 | -0.011646 |

The two SAM3 modes are strongly asymmetric.  Target-pooling reaches 0.842460
at b6, while patch-correspondence reaches only 0.791321 at b6.  Nevertheless,
the combined SAM3 candidate oracle is 0.889432 versus 0.889047 for DINO, so
the candidate set is not intrinsically worse; selection is the bottleneck.

Five-fold target-level validation Router diagnostics:

| candidate range | DINO selected | SAM3 selected | DINO oracle | SAM3 oracle |
|---|---:|---:|---:|---:|
| b0-b6 | 0.847886 | 0.841517 | 0.889047 | 0.889432 |
| b3-b6 | 0.831619 | 0.836823 | 0.868303 | 0.878693 |

The old DINO-trained X3+B7 selector is not suitable as the official selector
for the new routes: its diagnostic SAM3 b0-b6 selected Dice is 0.838158 versus
0.853511 on DINO.  A new student/X3 must be trained before the final decision.

Train propagation completed for both modes: 5,544 routes per mode and 11,088
routes total.  The b0-b6 manifest accepted 410/792 targets; the b3-b6 control
accepted 389/792.  The b0-b6 S3 consensus contains 410 targets.

## S2/S3 validation and user-authorized test audit

Both students trained for 40,000 iterations with ordinary validation every
200 iterations.  Checkpoint selection did not use test GT.

| checkpoint | validation Dice | iteration | test Dice | test IoU |
|---|---:|---:|---:|---:|
| S2 val-best | 0.820152 | 32,400 | 0.855042 | 0.771485 |
| S2 final | 0.809828 | 40,000 | 0.852245 | 0.767905 |
| S3 val-best | 0.816272 | 20,200 | 0.845245 | 0.763529 |
| S3 final | 0.812164 | 40,000 | **0.862539** | **0.786166** |

The user explicitly authorized this four-checkpoint test audit.  The result is
recorded in `s2_s3_test_results.json`; it is diagnostic only and must not be
used to change the already frozen validation-based checkpoint policy.

Committee audit produced Tier A/B/C counts of 94/128/160.  The new X3 manifest
contains 632 targets (410 original + 94 Tier A + 128 Tier B).  X3 val-best is
0.826032 at iteration 28,800 and X3 final is 0.814216 at iteration 40,000.
Both checkpoints exist; test evaluation for X3 remains deferred.

## Frozen B7 rule and LoRA dataset

X3 val-best and final predictions were exported for train/validation only.
The validation threshold sweep used both SAM3-KNN modes and all b0-b6 routes:

| selector | minimum B7 | kept | coverage | selected Val Dice |
|---|---:|---:|---:|---:|
| X3-best, b0-b6 | 0.90 | 64/100 | 64% | 0.946575 |
| X3-best, b0-b6 | **0.94** | **54/100** | **54%** | **0.950447** |
| X3-best, b0-b6 | 0.96 | 43/100 | 43% | 0.953728 |
| X3-final, b0-b6 | 0.94 | 52/100 | 52% | 0.950145 |
| X3-best, b3-b6 control | 0.94 | 58/100 | 58% | 0.939349 |

The frozen rule is X3 val-best + both SAM3-KNN modes + b0-b6 + B7 >= 0.94.
Applied to train without GT selection, it keeps 486/792 pseudo labels.  The
MedSAM3 dataset contains 486 pseudo labels plus eight human anchors (494 train
images) and 100 real validation images.

The 50-epoch full-module LoRA run started on GPU0 at 2026-08-24 01:09.  The
trainer now reports train loss, validation loss, and ordinary direct semantic
Val Dice every epoch, saving every epoch plus separate loss-best and
direct-Dice-best LoRA weights.  A validation-only downstream propagation/B7
queue was prepared for GPU1 with no test stage, then stopped as noted below.

The untouched base `sam3.pt` was also evaluated on all 100 validation images
with exactly the same direct-Dice protocol used inside the LoRA trainer
(class probability >= 0.5, mask probability >= 0.5, foreground-query union at
1008 resolution).  Its direct Val Dice is 0.332041, median 0, IoU 0.309719,
and nonempty rate 0.46.  This raw text-prompt result is distinct from route
`direct/b0`, which already uses anchor-conditioned propagation: target pooling
0.773972, patch correspondence 0.750490, and combined 0.762231 on validation.

## User-authorized current-route test

The GPU1 validation checkpoint queue was stopped at the user's request.  GPU1
then evaluated the complete untouched-base-SAM3 test route pool: two modes x
b0-b6 x 100 targets = 1,400 route results.  X3 val-best+B7 reaches test Dice
0.895432 with oracle 0.921471; X3 final+B7 is a diagnostic 0.897684.  The
validation-selected X3 val-best remains the official mainline.  Full per-step,
per-mode, bridge-selection and confidence details are in
`C0_256_sam3knn_s256_b0_b6_test_details.md`.
