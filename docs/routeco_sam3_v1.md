# RouteCo-SAM3 v1

RouteCo-SAM3 v1 is a Kvasir 1% follow-up experiment on top of the frozen
`b3-b6` route-pool results.  The goal is to test whether a student model and a
lightweight SAM3 adaptation module can co-train in multi-route memory-transport
space without rebuilding the KNN graph or fine-tuning the full SAM3 encoder.

See [Kvasir 1% Anchor Method](kvasir_1pct_anchor.md) for the surrounding
protocol, results, and current status.

## Fixed Components

- KNN graph is frozen.
- Candidate route generators are frozen:
  - `anchor_conditioned_target_pooling`
  - `anchor_conditioned_patch_correspondence`
- Candidate route lengths are frozen to `b3-b6`.
- Frozen SAM3 prior is retained as a teacher/preservation term.
- Current best propagation-quality pseudo labels are used as the warm start.

## Trainable Components

- Student: initialized or warmed with the current best pseudo labels.
- SAM3: only the memory-read adapter and mask-decoder LoRA are trainable.
- The SAM3 image encoder, base memory stack, and base mask-decoder weights stay
  frozen.

## RouteCo Signals

- Multi-path route uncertainty:
  - mean pixel variance across candidate route masks
  - selected route disagreement with the remaining route masks
- Student augmentation uncertainty:
  - weak/strong prediction variance
  - reserved in the v1 pseudo manifest and filled once student predictions are
    exported with augmentation probes
- Directional gates:
  - `T->S` gate for teacher-to-student pseudo supervision
  - `S->T` gate for student-to-SAM3 adaptation
  - v1 uses scalar confidence gates over route uncertainty and student
    weak/strong variance
- Anchor-cycle loss:
  - already supported by SAM3 route probes and propagation-quality extraction
  - used as a route consistency signal for the adapter

## Explicitly Deferred

- Dynamic KNN rebuilds.
- Spatial memory gates.
- Complex DPO objectives.
- Full SAM3 image encoder fine-tuning.
- Same-iteration bidirectional backpropagation.

## Current Preparation Pipeline

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_routeco_v1_prepare.sh
```

This prepares train split pseudo-video routes without using non-anchor train GT:

```text
Frozen KNN b3-b6 routes
  -> no-GT propagation-quality extraction
  -> validation-trained propagation-quality scorer
  -> train pseudo manifest for RouteCo warm start
```

Main outputs:

```text
work/kvasir_1pct_anchors/routeco_sam3_v1_routes/
work/kvasir_1pct_anchors/routeco_sam3_v1/routeco_v1_pseudo_manifest.jsonl
work/kvasir_1pct_anchors/routeco_sam3_v1/routeco_v1_pseudo_summary.json
```

## Current Training Pipeline

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_routeco_v1_train.sh
```

Default training settings:

```text
trainable: memory-read adapter + mask-decoder LoRA
T->S gate: exp(-route_uncertainty / gate_temperature), clipped by min_gate
S->T gate: exp(-student_variance / student_temperature), clipped by min_gate
loss: gated pseudo terminal loss + gated frozen-prior distillation + gated anchor-cycle loss
checkpoint rule: fixed final step; no validation/test GT checkpoint selection
```

Training outputs:

```text
work/kvasir_1pct_anchors/routeco_sam3_v1/train_routeco_v1/
```

## Method Distinction

SynFoC co-trains two models in single-image prediction space.  RouteCo-SAM3
co-trains a student and a SAM3 memory-transport model over multiple frozen
pseudo-video routes, where disagreement and cycle consistency are properties of
route transport rather than only single-image segmentation outputs.
