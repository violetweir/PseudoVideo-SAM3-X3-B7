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

```text
16 fixed train GT anchors
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

This repository does not vendor SAM3, SC-SAM, CVC-ClinicDB, Kvasir-SEG, or model
checkpoints. Provide them locally and set paths in `configs/reproduction.toml`.

SAM3 commands must run in the SAM3 environment. Student commands must run in the
SC-SAM/student environment.

On the original server these were:

```text
SAM3:    /home/violet/anaconda3/envs/sam3/bin/python
Student: /home/violet/anaconda3/envs/mkunet_mamba/bin/python
```

SC-SAM is loaded through `SC_SAM_ROOT`. The historical local SC-SAM checkout did
not include a redistributable license, so it is intentionally treated as an
external dependency.

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

For the original server, set:

```bash
export SC_SAM_ROOT=/Data_8TB/lht/SC-SAM
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
third_party/             placeholder for external checkouts, not vendored
```
