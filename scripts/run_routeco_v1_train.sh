#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-work/kvasir_1pct_anchors/routeco_sam3_v1}"
MANIFEST="${MANIFEST:-$ROOT/routeco_v1_pseudo_manifest.jsonl}"
CHECKPOINT="${CHECKPOINT:-/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt}"
OUT="${OUT:-$ROOT/train_routeco_v1}"
STEPS="${STEPS:-400}"
PY="${PY:-/home/violet/anaconda3/envs/sam3/bin/python}"

if [[ ! -s "$MANIFEST" ]]; then
  echo "Missing pseudo manifest: $MANIFEST" >&2
  echo "Run scripts/run_routeco_v1_prepare.sh first." >&2
  exit 2
fi

mkdir -p "$OUT"
"$PY" scripts/train_routeco_sam3_v1.py \
  --pseudo-manifest "$MANIFEST" \
  --base-checkpoint "$CHECKPOINT" \
  --output-dir "$OUT" \
  --steps "$STEPS" \
  --lr "${LR:-1e-4}" \
  --lambda-pseudo "${LAMBDA_PSEUDO:-0.5}" \
  --lambda-cycle "${LAMBDA_CYCLE:-0.1}" \
  --lambda-prior-mask "${LAMBDA_PRIOR_MASK:-0.1}" \
  --lambda-prior-mem "${LAMBDA_PRIOR_MEM:-0.05}" \
  --lambda-prior-ptr "${LAMBDA_PRIOR_PTR:-0.05}" \
  --gate-temperature "${GATE_TEMPERATURE:-0.05}" \
  --student-temperature "${STUDENT_TEMPERATURE:-0.05}" \
  --min-gate "${MIN_GATE:-0.15}" \
  --save-every "${SAVE_EVERY:-100}" \
  --save-step0
