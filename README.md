# Pseudo-Video SAM3 X3+B7

Reproduction code for a category-free pseudo-video SAM3 pipeline ending at the
`S27 X3 Final + B7` route selector.

The method starts from 16 fixed train-set GT anchors, builds three frozen
pseudo-video routes per image with SAM3, trains single-image student auditors
from high-confidence pseudo labels, expands the pseudo-label set with a
student-audited committee, trains the final X3 student, and uses the X3 student
to select among frozen SAM3 routes with B7.

## Main Result

Fixed protocol: CVC-ClinicDB + Kvasir-SEG merged splits, 16 train anchors only.
No validation/test masks are used to train SAM3, generate routes, select pseudo
labels, or train the student. Test masks are used only for final reporting.

| Method | Test Dice | CVC Dice | Kvasir Dice |
|---|---:|---:|---:|
| Frozen SAM3 multi-route baseline | 0.8715 | - | - |
| S27 X3 single-image student | 0.866738 | 0.886304 | 0.854802 |
| S27 X3 + validation-selected linear selector | 0.885637 | 0.905077 | 0.873778 |
| S27 X3 + B7 geometric selector | 0.895835 | 0.904713 | 0.890420 |
| Oracle over three frozen SAM3 routes | 0.907835 | - | - |

B7 score:

```text
score = (max(q_return, 1e-6) * max(q_multi, 1e-6)^2 * max(q_model, 1e-6)^2)^0.2
```

## Pipeline

There are two levels of reproduction:

- `scripts/run_method_ladder.py` covers the full experimental ladder, starting
  from single-image SAM3 and moving through two-frame/multi-step pseudo-video,
  SC-SAM/SynFoC low-label students, SAM3 adaptation diagnostics, student
  distillation, and final B7 selection.
- `scripts/run_pipeline.py` is the clean final mainline from the fixed
  pseudo-video protocol to `S27 X3 Final+B7`.

```text
B00 single-image SAM3 baseline
  -> T19 SC-SAM 16GT low-label student baseline
  -> T20 SynFoC 16GT low-label student baseline
  -> T18 two-frame support-to-query pseudo-video
  -> E1 multi-step star/chain/hybrid propagation
  -> T21 frozen three-route pseudo-video
  -> 16 fixed train GT anchors
  -> T21 frozen SAM3 pseudo-video routes
  -> 568 original high-confidence pseudo labels
  -> T24 committee students: S2 val-best, S2 final, S3 final
  -> audit remaining train images into Tier A/B/C
  -> S27 X3 = original568 + Tier A + Tier B
  -> X3 final checkpoint selected by fixed final-step validation protocol
  -> B7 student-assisted route selection on frozen SAM3 routes
```

Route types:

- `direct`: anchor -> query
- `one_bridge`: anchor -> bridge -> query
- `two_bridges`: anchor -> bridge1 -> bridge2 -> query

The pseudo-video paths are fixed by train-only image descriptors and kNN search.
Pseudo labels are used for training students and auditing routes, not as new
SAM3 propagation anchors in this public mainline.

## External Dependencies

This repository vendors the SC-SAM and SynFoC student-baseline code under
`third_party/`. SAM3, datasets, and model checkpoints are still external.

SAM3 commands must run in the SAM3 environment. Student commands must run in the
SC-SAM/student environment.

On the original server these were:

```text
SAM3:    /home/violet/anaconda3/envs/sam3/bin/python
Student: /home/violet/anaconda3/envs/mkunet_mamba/bin/python
```

SC-SAM is loaded through `SC_SAM_ROOT` and defaults to `third_party/SC-SAM`.
SynFoC T20 defaults to `third_party/SynFoC-T20`.

## Data Layout

Prepare merged data as:

```text
data/
  train/metadata.jsonl
  validation/metadata.jsonl
  test/metadata.jsonl
```

Each row needs:

```json
{
  "file_name": "/abs/path/to/image.png",
  "mask_file_name": "/abs/path/to/mask.png",
  "merged_id": "CVC-ClinicDB::156",
  "source_dataset": "CVC-ClinicDB"
}
```

Expected counts are `train=1290`, `validation=161`, `test=161`.

The fixed 16 support IDs are stored in
`protocols/reproduction_v1/support_ids.txt`. The full path-free split protocol
is `protocols/reproduction_v1/splits.jsonl`.

## Quick Start

```bash
cp configs/reproduction.example.toml configs/reproduction.toml
# edit paths in configs/reproduction.toml

python scripts/run_method_ladder.py --config configs/reproduction.toml --dry-run
python scripts/run_pipeline.py --config configs/reproduction.toml --dry-run
python scripts/run_pipeline.py --config configs/reproduction.toml
```

You can resume from any stage:

```bash
python scripts/run_pipeline.py \
  --config configs/reproduction.toml \
  --from-stage s27_x3_train \
  --to-stage t25_b7_test
```

For the original server, the default config already points to the vendored
SC-SAM copy. You can still override it with:

```bash
export SC_SAM_ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/third_party/SC-SAM
```

## Kvasir 1% Anchor Experiments

These are local WACV2027 follow-up experiments on Kvasir-SEG only.  They use
the dataset snapshot at:

```text
/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T02_fresh_polyp_hf_sources/raw_hf_snapshots/kvasir-seg/snapshot
```

Protocol summary:

- split counts: `train=800`, `validation=100`, `test=100`
- GT anchors: 1% of train, `8` fixed support masks
- SAM3 base checkpoint:
  `/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt`
- DINOv3 weights:
  `/Data_8TB/lht/MK-UNet/teacher/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- outputs:
  `work/kvasir_1pct_anchors/`

### SAM3 fine-tuning budgets

The table below reports Kvasir test Dice from the weighted route selector over
the original three route families: `direct`, `one_bridge`, and `two_bridges`.

| model | weighted Dice | direct | one_bridge | two_bridges | all-route mean | oracle |
|---|---:|---:|---:|---:|---:|---:|
| base_no_ft | 0.830103 | 0.767542 | 0.819048 | 0.761372 | 0.782654 | 0.879040 |
| ft_1pct | 0.874330 | 0.831090 | 0.856687 | 0.820172 | 0.835983 | 0.902679 |
| ft_5pct | 0.882543 | 0.814541 | 0.871664 | 0.837800 | 0.841335 | 0.911454 |
| ft_10pct | 0.844562 | 0.768585 | 0.846834 | 0.815985 | 0.810468 | 0.899441 |
| ft_20pct_latest_after_crash | 0.914321 | 0.856197 | 0.901934 | 0.875345 | 0.877825 | 0.931738 |

Longer-route weighted runs over 3-7 route families:

| model | weighted 3 routes | weighted 4 routes | weighted 5 routes | weighted 6 routes | weighted 7 routes | oracle 7 routes |
|---|---:|---:|---:|---:|---:|---:|
| base_no_ft | 0.830103 | 0.856771 | 0.849015 | 0.861283 | 0.864143 | 0.913565 |
| ft_1pct | 0.874330 | 0.888891 | 0.894013 | 0.901913 | 0.900234 | 0.928200 |
| ft_5pct | 0.882543 | 0.891466 | 0.883936 | 0.886997 | 0.895276 | 0.938849 |
| ft_10pct | 0.844562 | 0.889791 | 0.884999 | 0.887835 | 0.901470 | 0.932109 |
| ft_20pct_latest_after_crash | 0.914321 | 0.914936 | 0.914313 | 0.914289 | 0.910493 | 0.942206 |

The main takeaway is that SAM3 fine-tuning is useful on this protocol.  The
20% fine-tuned checkpoint is the strongest weighted result, while smaller
budgets are not strictly monotonic.

### Frozen-feature KNN Stage1

This diagnostic freezes the SAM3 checkpoint to the un-fine-tuned base model and
changes only the route search descriptor.  No weighted selector is used in the
table below: each cell is the mean forward-only test Dice for one fixed route
family.  `bN` means `N` bridge frames between the anchor and query.

| feature mode | direct | b1 | b2 | b3 | b4 | b5 | b6 | b7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T18 corrected | 0.747869 | 0.775949 | 0.756490 | 0.785808 | 0.821736 | 0.816809 | 0.819200 | 0.143392 |
| DINO global pooling | 0.740844 | 0.748290 | 0.758326 | 0.786777 | 0.777536 | 0.818397 | 0.818565 | 0.092024 |
| DINO patch average | 0.728634 | 0.726893 | 0.756876 | 0.746756 | 0.799814 | 0.829179 | 0.826294 | 0.046968 |
| anchor-conditioned target pooling | 0.741306 | 0.760176 | 0.827209 | 0.840837 | 0.841258 | 0.848618 | 0.854627 | 0.032027 |
| anchor-conditioned patch correspondence | 0.757814 | 0.773501 | 0.822305 | 0.832819 | 0.833990 | 0.833577 | 0.842078 | 0.070807 |

The strongest frozen-feature result is
`anchor-conditioned target pooling + b6 = 0.854627`.  The useful route-length
region is `b4-b6`; `b7` collapses for every feature mode, which suggests the
route is beyond the stable propagation length for this SAM3 setting.

Canvas-256 and canvas-512 route hashes were also audited.  For all five feature
modes, the `256` and `512` runs use identical route hashes and identical route
ordering, so canvas differences come from SAM3 execution, not from KNN selecting
different paths.

### Launch Commands

Generate the Kvasir 1% anchor protocol first if it is missing:

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
# The prepared protocol should contain:
# work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl
# work/kvasir_1pct_anchors/protocol/support_manifest.jsonl
cat work/kvasir_1pct_anchors/protocol/protocol_summary.json
```

Build frozen KNN routes up to `b7` for all five feature modes:

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_stage1_b7_routes.sh
```

Forward-only evaluation of the un-fine-tuned SAM3 base checkpoint:

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_stage1_b7_eval_forward.sh
```

Per-mode summaries are written to:

```text
work/kvasir_1pct_anchors/stage1_feature_knn_b7/<feature_mode>/eval_base_no_ft_b7_forward/route_family_summary.json
```

The earlier weighted fine-tuning summaries are stored under:

```text
work/kvasir_1pct_anchors/model_routes/*/summary.json
work/kvasir_1pct_anchors/model_routes_max6/*/summary_max5_max6.json
work/kvasir_1pct_anchors/summaries/route_count_3_to_7_full_table.md
```

## Important Reproduction Notes

- `S27 X3+B7` is the public mainline.
- `S27 X0/X1/X3` use the later unified S27 student trainer. It is not a strict
  bit-level reproduction of the older T24 supervised-loss implementation.
- The S27 trainer uses a foreground per-sample Dice convention for GT; older T24
  used a two-class batch Dice convention. The public project preserves the S27
  implementation that produced the reported X3+B7 result.
- B7 is treated as a fixed sensitivity/mainline selector here because it was
  chosen after the later local analysis. The validation-selected linear selector
  is also reported separately for protocol clarity.
- Do not use validation/test masks to change support anchors, pseudo labels,
  route topology, training data, or checkpoints.

## Repository Map

```text
configs/                 editable machine-specific config
envs/                    example conda environment manifests
protocols/reproduction_v1 fixed path-free split and support protocol
scripts/                 reproduction stages
src/pvseg/               small shared utilities
docs/                    protocol and release notes
third_party/             vendored SC-SAM and SynFoC student baselines
```
