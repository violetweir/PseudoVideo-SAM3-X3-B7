# C0-256-base → X3-best+B7 → SAM3 full-module LoRA (50 epochs)

> Initial start: 2026-08-21 23:45 (Asia/Shanghai)  
> Clean restart after dtype fix: 2026-08-22 00:10 (Asia/Shanghai)  
> Status at handoff: running on GPU 0  
> Root: `work/rerun_c0_256_base/medsam3_lora_b7_e50/`

## 1. Frozen pseudo-label selection rule

The training masks are rebuilt from the final C0-256-base selection rule, not
reused from `pseudo_manifest_x3.jsonl`.

- student: `students/X3/student_best.pth`
- route families: target pooling + patch correspondence
- candidate bridges: b3-b6
- score: `B7 = (q_return * q_multi^2 * q_model^2)^0.2`
- selected mask: highest-B7 route for each target
- acceptance threshold: `B7 >= 0.94`
- no train GT is used for route choice or thresholding

The threshold was fixed on validation before being applied to train:

| minimum B7 | validation kept | coverage | selected-mask Dice |
|---:|---:|---:|---:|
| 0.90 | 67/100 | 67% | 0.9294 |
| 0.92 | 61/100 | 61% | 0.9464 |
| **0.94** | **48/100** | **48%** | **0.9544** |
| 0.96 | 35/100 | 35% | 0.9577 |

Frozen `0.94` on train keeps 494/792 pseudo labels.  The audit-only mean Dice
is 0.924585; this number is reported after selection and is never used by the
selector.

## 2. LoRA dataset

| component | count |
|---|---:|
| X3-best+B7 pseudo labels | 494 |
| frozen human GT anchors | 8 |
| training total | **502** |
| real validation masks | 100 |

COCO category text is `colon polyp`, matching the successful historical
MedSAM3-LoRA run.  Images and masks decode at the trainer's fixed 1008×1008
resolution.  Downstream propagation remains at canvas 256.

## 3. Training configuration

- base: untouched `sam3.pt`
- full-module LoRA: 383 modules
- trainable parameters: 17,883,264 (2.08%)
- rank / alpha / dropout: 16 / 32 / 0.1
- AdamW: lr 5e-5, weight decay 0.01
- actual batch size: 1
- actual precision: FP32
- schedule: constant learning rate, matching the historical trainer behavior
- seed: 2026
- horizon: 50 epochs
- checkpointing: every epoch plus `best_lora_weights.pt` and
  `last_lora_weights.pt`

The 50th epoch is only the training horizon.  Final model selection must use
downstream C0-256 propagation/B7 results across intermediate epochs, not assume
that epoch 50 is best.

## 4. Important paths

```text
manifests/b7_selected_validation_all.jsonl
manifests/b7_selected_train_all.jsonl
manifests/b7_selected_train_min094.jsonl
data/train/_annotations.coco.json
data/valid/_annotations.coco.json
lora_weights/epoch_N_lora_weights.pt
lora_weights/best_lora_weights.pt
lora_weights/last_lora_weights.pt
train.log
supervisor.log
nohup_supervisor.log
```

Config and scripts:

```text
configs/c0_256_base_b7_medsam3_lora_e50.yaml
scripts/build_c0_256_b7_lora_manifest.py
scripts/prepare_c0_256_b7_medsam3_dataset.py
scripts/train_sam3_lora_kvasir_e50.py
scripts/supervise_c0_256_b7_lora_e50.sh
```

## 5. Launch command

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
nohup env DEVICE=0 bash scripts/supervise_c0_256_b7_lora_e50.sh \
  > work/rerun_c0_256_base/medsam3_lora_b7_e50/nohup_supervisor.log \
  2>&1 < /dev/null &
```

## 6. Downstream checkpoint selection

At minimum evaluate epochs 1, 5, 10, 15, 20, 25, 30, 35, 40, 45 and 50.
Merge only evaluation candidates into full video checkpoints, then run the
same C0-256 route propagation and X3/B7 comparison.  Preserve base and
historical `lora_p491_e20` rows as controls.

## 7. Validation dtype incident and fix

The initial process completed all 502 training samples in epoch 1, then failed
on the first validation item with:

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

Cause: under `torch.no_grad()`, SAM3's fused ViT MLP converts the `fc1`
activation to BF16, while the MedSAM3 LoRA-wrapped `fc2` base Linear remains
FP32.  Historical LoRA runs had no validation split and therefore did not
exercise this inference-only path.

Fixes:

1. validation model forward now runs inside CUDA BF16 autocast;
2. `last_lora_weights.pt` and `epoch_N_lora_weights.pt` are saved immediately
   after training each epoch, before validation;
3. a full one-item validation smoke test (forward, matcher and loss) passed:
   `VALIDATION_SMOKE_OK loss=96.692200 dtype=torch.float32`.

The failed logs were preserved as:

```text
train.pre_dtype_fix.log
supervisor.pre_dtype_fix.log
nohup_supervisor.pre_dtype_fix.log
```

## 8. Parallel checkpoint evaluation on GPU 1

Checkpoint evaluation runs independently from training.  Epoch selection uses
validation only; test is evaluated once for the final validation-best epoch.

Validation nodes:

```text
e1-e12, e15, e20, e25, e30, e35, e40, e45, e50
```

For each node:

1. merge the LoRA delta into a full SAM3 video checkpoint;
2. evaluate both route families at canvas 256 using b3-b6 only;
3. calculate the frozen X3-best+B7 selected validation Dice;
4. update the cross-epoch summary and validation-best epoch.

After all nodes finish, the queue runs the same test propagation and B7 report
only for the validation-best epoch.

```text
checkpoint_eval/validation_checkpoint_summary.json
checkpoint_eval/validation_checkpoint_summary.tsv
checkpoint_eval/best_validation_epoch.txt
checkpoint_eval/test_validation_best_b7/b7_report.json
checkpoint_eval_queue.log
checkpoint_eval_supervisor.log
```

Queue scripts:

```text
scripts/run_c0_256_b7_lora_checkpoint_queue.sh
scripts/supervise_c0_256_b7_lora_checkpoint_queue.sh
scripts/filter_route_pool.py
scripts/summarize_c0_256_b7_lora_checkpoints.py
```
