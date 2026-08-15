#!/usr/bin/env bash
# Detached Phase-1 orchestrator (single GPU, survives session interruption).
# Waits for T24 S2, launches S3, then runs Phase B (audit -> X3 -> B7).
# Long trainings are launched with setsid so each step is independently detached.
set -u

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export SC_SAM_ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/third_party/SC-SAM
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/kvasir_1pct_anchors/phase1
DATA=work/kvasir_1pct_anchors/baseline_data
LABELS=work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt
LOG=$PHASE/logs

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/orchestrator.log"; }

wait_complete() {
  local dir="$1" name="$2"
  while [ ! -f "$dir/TRAINING_COMPLETE" ]; do
    local it
    it=$(tail -1 "$dir/train.jsonl" 2>/dev/null | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['iteration'])" 2>/dev/null)
    log "$name iter=$it"
    sleep 300
  done
  log "$name COMPLETE"
}

launch_detached() {
  local out="$1"; shift
  setsid nohup "$@" > "$out" 2>&1 < /dev/null &
  log "launched detached: $* -> $out"
}

log "orchestrator start"

log "wait S2"
wait_complete "$PHASE/students/S2" "S2"

if [ ! -f "$PHASE/students/S3/TRAINING_COMPLETE" ] && ! pgrep -f "run_t24_student.py.*students/S3" >/dev/null; then
  log "launch S3"
  launch_detached "$LOG/S3_detached.log" "$PY" scripts/run_t24_student.py \
    --data-path "$DATA" --labeled-list "$LABELS" \
    --pseudo-manifest "$PHASE/S3_consensus/pseudo_consensus.jsonl" \
    --output-dir "$PHASE/students/S3" --experiment S3 --seed 2026 \
    --max-iterations 40000 --val-interval 200 --num-workers 4
else
  log "S3 already running or complete; skip launch"
fi
log "wait S3"
wait_complete "$PHASE/students/S3" "S3"

log "export T24 predictions"
for entry in "S2 student_best.pth predictions/S2_valbest" "S2 student_final.pth predictions/S2_final" "S3 student_final.pth predictions/S3_final"; do
  set -- $entry
  run=$1; ckpt=$2; out=$3
  if [ ! -f "$PHASE/$out/EXPORT_COMPLETE" ]; then
    "$PY" scripts/export_t25_student_predictions.py \
      --run-dir "$PHASE/students/$run" --checkpoint "$PHASE/students/$run/$ckpt" \
      --output-root "$PHASE/$out"
  fi
done

log "committee audit"
if [ ! -f "$PHASE/audit/summary.json" ]; then
  "$PY" scripts/phase1_audit_tiers.py \
    --train-metadata "$DATA/train/metadata.jsonl" --labeled-list "$LABELS" \
    --quality-root work/kvasir_1pct_anchors/stage1_feature_knn_b7_ft1pct \
    --original-manifest "$PHASE/pseudo_manifest_original.jsonl" \
    --predictions "$PHASE/predictions/S2_valbest/student_predictions_train.jsonl" --predictions-name S2_valbest \
    --predictions "$PHASE/predictions/S2_final/student_predictions_train.jsonl" --predictions-name S2_final \
    --predictions "$PHASE/predictions/S3_final/student_predictions_train.jsonl" --predictions-name S3_final \
    --output-dir "$PHASE/audit"
fi

log "X3 manifest"
if [ ! -f "$PHASE/pseudo_manifest_x3.jsonl" ]; then
  "$PY" scripts/phase1_build_x3_manifest.py \
    --original "$PHASE/pseudo_manifest_original.jsonl" \
    --tier-a "$PHASE/audit/tier_A.jsonl" \
    --tier-b "$PHASE/audit/tier_B.jsonl" \
    --output "$PHASE/pseudo_manifest_x3.jsonl"
fi

log "launch X3 training"
if [ ! -f "$PHASE/students/X3/TRAINING_COMPLETE" ] && ! pgrep -f "run_s27_student.py.*students/X3" >/dev/null; then
  launch_detached "$LOG/X3_detached.log" "$PY" scripts/run_s27_student.py \
    --data-path "$DATA" --labeled-list "$LABELS" \
    --pseudo-manifest "$PHASE/pseudo_manifest_x3.jsonl" \
    --output-dir "$PHASE/students/X3" --experiment X3 --seed 2026 \
    --batch-size 12 --gt-bs 3 --original-bs 3 --new-bs 6 \
    --max-iterations 40000 --val-interval 200 --num-workers 4
else
  log "X3 already running or complete; skip launch"
fi
log "wait X3"
wait_complete "$PHASE/students/X3" "X3"

log "X3 export"
if [ ! -f "$PHASE/predictions/X3_final/EXPORT_COMPLETE" ]; then
  "$PY" scripts/export_t25_student_predictions.py \
    --run-dir "$PHASE/students/X3" --checkpoint "$PHASE/students/X3/student_final.pth" \
    --output-root "$PHASE/predictions/X3_final"
fi

log "B7 selection"
"$PY" scripts/phase1_b7_select.py \
  --quality-root work/kvasir_1pct_anchors/stage1_feature_knn_b7_ft1pct \
  --student-predictions "$PHASE/predictions/X3_final/student_predictions_test.jsonl" \
  --output-dir "$PHASE/selection"

log "PHASE1_PIPELINE_DONE"
