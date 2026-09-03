#!/usr/bin/env bash
# ISIC18 Round1 + Round2A pipeline, inheriting work/isic18_pseudovideo_full
# dataset/protocol settings.
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

SAMPY=/home/violet/anaconda3/envs/sam3/bin/python
MKPY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
BASE_CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt

PHASE=work/isic18_round1_round2a_from_pseudovideo_full
PROTO=work/isic18_pseudovideo_full/protocol
DATA=work/isic18_1pct_protocol/data
LABELS=work/isic18_pseudovideo_full/protocol/frozen_labeled_images.txt
ROUTE_ROOT=$PHASE/round1_sam3knn_s256_base/stage1_feature_knn_b0_b6
ROUND1=$PHASE/round1_sam3knn_s256_base
LORA_RUN=$ROUND1/medsam3_lora_b0_b6_e50
ROUND2A=$PHASE/round2a_fixed_knn_lora_teacher
CACHE_ROOT=$PHASE/isic_resized_cache_s256
LOG=$PHASE/pipeline.log
DEVICE=${DEVICE:-0}
export CUDA_VISIBLE_DEVICES=$DEVICE
export ISIC_RESIZED_CACHE_ROOT=$CACHE_ROOT

MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)

log() {
  mkdir -p "$PHASE"
  echo "[isic18-r1-r2a] $(date '+%F %T') $*" | tee -a "$LOG"
}

marker() {
  [[ -f "$PHASE/$1" ]]
}

run_routes() {
  if marker ROUTES_COMPLETE; then
    log "routes already complete"
    return
  fi
  log "route generation start, protocol=$PROTO"
  mkdir -p "$ROUTE_ROOT/features" "$ROUND1"
  if [[ -f work/isic18_sam3knn_s256_base/stage1_feature_knn_b0_b6/features/sam3_base_s256_features.npz ]]; then
    ln -sfn "$(realpath work/isic18_sam3knn_s256_base/stage1_feature_knn_b0_b6/features/sam3_base_s256_features.npz)" \
      "$ROUTE_ROOT/features/sam3_base_s256_features.npz"
  fi
  for mode in "${MODES[@]}"; do
    for split in train validation test; do
      log "routes $mode $split"
      "$SAMPY" scripts/stage1_feature_knn_routes.py \
        --mode "$mode" \
        --feature-source sam3_base \
        --feature-size 256 \
        --knn-feature patch_mean \
        --split "$split" \
        --min-bridge 0 \
        --max-bridge 6 \
        --beam-width 32 \
        --protocol-root "$PROTO" \
        --output-root "$ROUTE_ROOT" \
        >> "$LOG" 2>&1
    done
  done
  ln -sfn "$(realpath "$ROUTE_ROOT/${MODES[0]}")" "$ROUTE_ROOT/anchor_conditioned_target_pooling"
  ln -sfn "$(realpath "$ROUTE_ROOT/${MODES[1]}")" "$ROUTE_ROOT/anchor_conditioned_patch_correspondence"
  "$SAMPY" - "$ROUTE_ROOT" <<'PY' >> "$LOG" 2>&1
import json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
expected = {"train": 2054 * 7, "validation": 259 * 7, "test": 260 * 7}
modes = ("sam3enc_anchor_conditioned_target_pooling", "sam3enc_anchor_conditioned_patch_correspondence")
report = {}
for mode in modes:
    report[mode] = {}
    for split, n in expected.items():
        path = root / mode / f"{split}_pool0_stage1/routes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        bridges = Counter(int(row["bridge_count"]) for row in rows)
        targets = Counter(row["target_id"] for row in rows)
        if len(rows) != n:
            raise SystemExit(f"{mode}/{split}: {len(rows)} != {n}")
        if any(value != 7 for value in targets.values()):
            raise SystemExit(f"{mode}/{split}: not exactly seven routes per target")
        report[mode][split] = {"routes": len(rows), "targets": len(targets), "bridge_counts": dict(sorted(bridges.items()))}
(root / "route_generation_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY
  touch "$PHASE/ROUTES_COMPLETE"
  log "routes complete"
}

run_base_propagation() {
  if marker BASE_PROP_COMPLETE; then
    log "base propagation already complete"
    return
  fi
  log "base SAM3 propagation start on GPU=$DEVICE"
  for split in validation train test; do
    for mode in "${MODES[@]}"; do
      log "base propagation split=$split mode=$mode"
      "$SAMPY" scripts/eval_route_propagation_quality.py \
        --checkpoint "$BASE_CKPT" \
        --mode "$mode" \
        --root "$ROUTE_ROOT" \
        --split "$split" \
        --canvas 256 \
        --resume \
        >> "$LOG" 2>&1
    done
    "$SAMPY" scripts/summarize_c0_256_bridge_metrics.py \
      --quality-root "$ROUTE_ROOT" \
      --split "$split" \
      --modes "${MODES[@]}" \
      --min-bridge 0 --max-bridge 6 \
      --output-json "$ROUND1/base_${split}_bridge_b0_b6.json" \
      --output-tsv "$ROUND1/base_${split}_bridge_b0_b6.tsv" \
      >> "$LOG" 2>&1
  done
  touch "$PHASE/BASE_PROP_COMPLETE"
  log "base propagation complete"
}

prepare_resized_cache() {
  if marker CACHE_COMPLETE; then
    log "resized cache already complete"
    return
  fi
  log "prepare resized cache start"
  "$MKPY" scripts/prepare_isic_resized_cache.py \
    --data-path "$DATA" \
    --output-root "$CACHE_ROOT" \
    --image-size 256 \
    --splits train validation test \
    >> "$LOG" 2>&1
  touch "$PHASE/CACHE_COMPLETE"
  log "prepare resized cache complete"
}

run_round1_students() {
  if marker X3_TRAIN_COMPLETE; then
    log "Round1 students already complete"
    return
  fi
  log "Round1 pseudo pool + S2/S3/X3 start"
  "$MKPY" scripts/select_phase1_mainline_pseudo568.py \
    --quality-root "$ROUTE_ROOT" \
    --min-bridge 0 --max-bridge 6 \
    --output "$ROUND1/pseudo_manifest_original_b0_b6.jsonl" \
    >> "$LOG" 2>&1
  "$MKPY" scripts/prepare_phase1_s3_consensus.py \
    --original-manifest "$ROUND1/pseudo_manifest_original_b0_b6.jsonl" \
    --quality-root "$ROUTE_ROOT" \
    --min-bridge 0 --max-bridge 6 \
    --output-root "$ROUND1/S3_consensus_b0_b6" \
    >> "$LOG" 2>&1
  for student in S2 S3; do
    case "$student" in
      S2) manifest=$ROUND1/pseudo_manifest_original_b0_b6.jsonl ;;
      S3) manifest=$ROUND1/S3_consensus_b0_b6/pseudo_consensus.jsonl ;;
    esac
    if [[ -f "$ROUND1/students/$student/TRAINING_COMPLETE" ]]; then
      log "train $student already complete"
    else
      log "train $student"
      "$MKPY" scripts/run_t24_student.py \
        --data-path "$DATA" --labeled-list "$LABELS" \
        --pseudo-manifest "$manifest" \
        --output-dir "$ROUND1/students/$student" --experiment "$student" --seed 2026 \
        --max-iterations 40000 --val-interval 200 --num-workers 8 \
        >> "$LOG" 2>&1
    fi
    for entry in "student_best.pth ${student}_valbest" "student_final.pth ${student}_final"; do
      read -r checkpoint output_name <<<"$entry"
      if [[ -f "$ROUND1/predictions/$output_name/EXPORT_COMPLETE" ]]; then
        log "export $student $checkpoint already complete"
      else
        log "export $student $checkpoint"
        "$MKPY" scripts/export_t25_student_predictions.py \
          --run-dir "$ROUND1/students/$student" \
          --checkpoint "$ROUND1/students/$student/$checkpoint" \
          --output-root "$ROUND1/predictions/$output_name" \
          --splits train validation test \
          >> "$LOG" 2>&1
      fi
    done
  done
  log "audit tiers and train X3"
  "$MKPY" scripts/phase1_audit_tiers.py \
    --train-metadata "$DATA/train/metadata.jsonl" \
    --labeled-list "$LABELS" \
    --quality-root "$ROUTE_ROOT" \
    --original-manifest "$ROUND1/pseudo_manifest_original_b0_b6.jsonl" \
    --predictions "$ROUND1/predictions/S2_valbest/student_predictions_train.jsonl" --predictions-name S2_valbest \
    --predictions "$ROUND1/predictions/S2_final/student_predictions_train.jsonl" --predictions-name S2_final \
    --predictions "$ROUND1/predictions/S3_valbest/student_predictions_train.jsonl" --predictions-name S3_valbest \
    --predictions "$ROUND1/predictions/S3_final/student_predictions_train.jsonl" --predictions-name S3_final \
    --output-dir "$ROUND1/audit" \
    >> "$LOG" 2>&1
  "$MKPY" scripts/phase1_build_x3_manifest.py \
    --original "$ROUND1/pseudo_manifest_original_b0_b6.jsonl" \
    --tier-a "$ROUND1/audit/tier_A.jsonl" \
    --tier-b "$ROUND1/audit/tier_B.jsonl" \
    --output "$ROUND1/pseudo_manifest_x3.jsonl" \
    >> "$LOG" 2>&1
  if [[ -f "$ROUND1/students/X3/TRAINING_COMPLETE" ]]; then
    log "train X3 already complete"
  else
    "$MKPY" scripts/run_s27_student.py \
      --data-path "$DATA" --labeled-list "$LABELS" \
      --pseudo-manifest "$ROUND1/pseudo_manifest_x3.jsonl" \
      --output-dir "$ROUND1/students/X3" --experiment X3_isic18_round1 --seed 2026 \
      --batch-size 12 --gt-bs 3 --original-bs 3 --new-bs 6 \
      --max-iterations 40000 --val-interval 200 --num-workers 8 \
      >> "$LOG" 2>&1
  fi
  if [[ -f "$ROUND1/predictions/X3_best/EXPORT_COMPLETE" ]]; then
    log "export X3 best already complete"
  else
    "$MKPY" scripts/export_t25_student_predictions.py \
      --run-dir "$ROUND1/students/X3" \
      --checkpoint "$ROUND1/students/X3/student_best.pth" \
      --output-root "$ROUND1/predictions/X3_best" \
      --splits train validation test \
      >> "$LOG" 2>&1
  fi
  if [[ -f "$ROUND1/predictions/X3_final/EXPORT_COMPLETE" ]]; then
    log "export X3 final already complete"
  else
    "$MKPY" scripts/export_t25_student_predictions.py \
      --run-dir "$ROUND1/students/X3" \
      --checkpoint "$ROUND1/students/X3/student_final.pth" \
      --output-root "$ROUND1/predictions/X3_final" \
      --splits train validation test \
      >> "$LOG" 2>&1
  fi
  touch "$PHASE/X3_TRAIN_COMPLETE"
  log "Round1 S2/S3/X3 complete"
}

calibrate_round1_b7_and_lora() {
  if marker ROUND1_LORA_COMPLETE; then
    log "Round1 LoRA already complete"
    return
  fi
  log "Round1 B7 calibration + LoRA start"
  mkdir -p "$ROUND1/b7_calibration" "$LORA_RUN" configs
  "$SAMPY" scripts/build_c0_256_b7_lora_manifest.py \
    --quality-root "$ROUTE_ROOT" \
    --student-predictions "$ROUND1/predictions/X3_best/student_predictions_validation.jsonl" \
    --split validation --modes "${MODES[@]}" \
    --min-bridge 0 --max-bridge 6 --min-b7 0 --canvas 256 \
    --output "$ROUND1/b7_calibration/validation_all.jsonl" \
    --summary "$ROUND1/b7_calibration/validation_all.summary.json" \
    --all-candidates-output "$ROUND1/b7_calibration/validation_all_candidates.jsonl" \
    >> "$LOG" 2>&1
  "$SAMPY" scripts/summarize_round2a_b7_calibration.py \
    --manifest "$ROUND1/b7_calibration/validation_all.jsonl" \
    --target-dice "${ISIC_B7_TARGET_DICE:-0.90}" \
    --output-json "$ROUND1/b7_calibration/calibration_frontier.json" \
    --output-tsv "$ROUND1/b7_calibration/calibration_frontier.tsv" \
    >> "$LOG" 2>&1
  THRESHOLD=$("$SAMPY" - "$ROUND1/b7_calibration/calibration_frontier.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
rec = data.get("recommended")
print(rec["min_b7"] if rec else 0.0)
PY
)
  echo "$THRESHOLD" > "$ROUND1/b7_calibration/recommended_threshold.txt"
  "$SAMPY" scripts/build_c0_256_b7_lora_manifest.py \
    --quality-root "$ROUTE_ROOT" \
    --student-predictions "$ROUND1/predictions/X3_best/student_predictions_train.jsonl" \
    --split train --modes "${MODES[@]}" \
    --min-bridge 0 --max-bridge 6 --min-b7 "$THRESHOLD" --canvas 256 \
    --sample-type isic18_round1_x3_best_b7 \
    --output "$LORA_RUN/train_b7_lora_manifest.jsonl" \
    --summary "$LORA_RUN/train_b7_lora_manifest.summary.json" \
    >> "$LOG" 2>&1
  "$SAMPY" scripts/build_c0_256_b7_lora_manifest.py \
    --quality-root "$ROUTE_ROOT" \
    --student-predictions "$ROUND1/predictions/X3_best/student_predictions_test.jsonl" \
    --split test --modes "${MODES[@]}" \
    --min-bridge 0 --max-bridge 6 --min-b7 0 --canvas 256 \
    --output "$ROUND1/b7_calibration/test_x3_best_b7.jsonl" \
    --summary "$ROUND1/b7_calibration/test_x3_best_b7.summary.json" \
    >> "$LOG" 2>&1
  "$SAMPY" scripts/prepare_isic18_b7_medsam3_dataset.py \
    --pseudo-manifest "$LORA_RUN/train_b7_lora_manifest.jsonl" \
    --labeled-list "$LABELS" \
    --validation-metadata "$DATA/validation/metadata.jsonl" \
    --output-root "$LORA_RUN/data" \
    >> "$LOG" 2>&1
  cat > configs/isic18_round1_sam3knn_s256_b0_b6_medsam3_lora_e50.yaml <<EOF
model:
  name: "facebook/sam3"
  checkpoint_path: $BASE_CKPT
lora:
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "out_proj", "qkv", "proj", "fc1", "fc2", "c_fc", "c_proj", "linear1", "linear2"]
  apply_to_vision_encoder: true
  apply_to_text_encoder: true
  apply_to_geometry_encoder: true
  apply_to_detr_encoder: true
  apply_to_detr_decoder: true
  apply_to_mask_decoder: true
training:
  data_dir: /Data_8TB/lht/PseudoVideo-SAM3-X3-B7/$LORA_RUN/data
  batch_size: 1
  num_workers: 2
  learning_rate: 5.0e-5
  weight_decay: 0.01
  num_epochs: 50
  seed: 2026
output:
  output_dir: $LORA_RUN/lora_weights
  save_lora_only: true
evaluation:
  metric: "direct_validation_dice"
  class_probability_threshold: 0.5
  mask_probability_threshold: 0.5
  downstream_selection: "ISIC18 SAM3@256-KNN b0-b6 propagation + frozen X3-best B7"
hardware:
  device: "cuda"
  training_resolution: 1008
EOF
  "$SAMPY" scripts/train_sam3_lora_kvasir_e50.py \
    --config configs/isic18_round1_sam3knn_s256_b0_b6_medsam3_lora_e50.yaml \
    --device "$DEVICE" \
    >> "$LOG" 2>&1
  touch "$PHASE/ROUND1_LORA_COMPLETE"
  log "Round1 LoRA complete"
}

run_round2a() {
  if marker ROUND2A_COMPLETE; then
    log "Round2A already complete"
    return
  fi
  log "Round2A start"
  mkdir -p "$ROUND2A" "$ROUND2A/quality_root" "$ROUND2A/b7_calibration"
  LORA_EPOCH=${LORA_EPOCH:-33}
  LORA_WEIGHTS=$LORA_RUN/lora_weights/epoch_${LORA_EPOCH}_lora_weights.pt
  MERGED=$ROUND2A/lora_epoch_${LORA_EPOCH}_merged_video.pt
  if [[ ! -f "$LORA_WEIGHTS" ]]; then
    log "missing $LORA_WEIGHTS"
    return 2
  fi
  if [[ ! -f "$MERGED" ]]; then
    "$SAMPY" scripts/merge_sam3_lora_video_checkpoint.py \
      --base-checkpoint "$BASE_CKPT" \
      --lora-weights "$LORA_WEIGHTS" \
      --output "$MERGED" \
      >> "$LOG" 2>&1
  fi
  for mode in "${MODES[@]}"; do
    mkdir -p "$ROUND2A/quality_root/$mode"
    for split in train validation test; do
      ln -sfn "$(realpath "$ROUTE_ROOT/$mode/${split}_pool0_stage1")" \
        "$ROUND2A/quality_root/$mode/${split}_pool0_stage1"
      log "Round2A propagation split=$split mode=$mode"
      "$SAMPY" scripts/eval_route_propagation_quality.py \
        --checkpoint "$MERGED" --mode "$mode" \
        --root "$ROUND2A/quality_root" --split "$split" \
        --canvas 256 --resume \
        >> "$LOG" 2>&1
    done
  done
  for split in train validation test; do
    "$SAMPY" scripts/summarize_c0_256_bridge_metrics.py \
      --quality-root "$ROUND2A/quality_root" --split "$split" --modes "${MODES[@]}" \
      --min-bridge 0 --max-bridge 6 \
      --output-json "$ROUND2A/two_mode_b0_b6_${split}.json" \
      --output-tsv "$ROUND2A/two_mode_b0_b6_${split}.tsv" \
      >> "$LOG" 2>&1
  done
  "$SAMPY" scripts/build_c0_256_b7_lora_manifest.py \
    --quality-root "$ROUND2A/quality_root" \
    --student-predictions "$ROUND1/predictions/X3_best/student_predictions_validation.jsonl" \
    --split validation --modes "${MODES[@]}" \
    --min-bridge 0 --max-bridge 6 --min-b7 0 --canvas 256 \
    --output "$ROUND2A/b7_calibration/validation_all.jsonl" \
    --summary "$ROUND2A/b7_calibration/validation_all.summary.json" \
    --all-candidates-output "$ROUND2A/b7_calibration/validation_all_candidates.jsonl" \
    >> "$LOG" 2>&1
  "$SAMPY" scripts/summarize_round2a_b7_calibration.py \
    --manifest "$ROUND2A/b7_calibration/validation_all.jsonl" \
    --target-dice "${ISIC_B7_TARGET_DICE:-0.90}" \
    --output-json "$ROUND2A/b7_calibration/calibration_frontier.json" \
    --output-tsv "$ROUND2A/b7_calibration/calibration_frontier.tsv" \
    >> "$LOG" 2>&1
  THRESHOLD=$("$SAMPY" - "$ROUND2A/b7_calibration/calibration_frontier.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
rec = data.get("recommended")
print(rec["min_b7"] if rec else 0.0)
PY
)
  "$SAMPY" scripts/build_c0_256_b7_lora_manifest.py \
    --quality-root "$ROUND2A/quality_root" \
    --student-predictions "$ROUND1/predictions/X3_best/student_predictions_train.jsonl" \
    --split train --modes "${MODES[@]}" \
    --min-bridge 0 --max-bridge 6 --min-b7 "$THRESHOLD" --canvas 256 \
    --sample-type original \
    --output "$ROUND2A/manifests/x4_train_b7_raw.jsonl" \
    --summary "$ROUND2A/manifests/x4_train_b7_raw.summary.json" \
    >> "$LOG" 2>&1
  "$SAMPY" scripts/prepare_round2a_x4_manifest.py \
    --b7-manifest "$ROUND2A/manifests/x4_train_b7_raw.jsonl" \
    --x3-reference "$ROUND1/pseudo_manifest_x3.jsonl" \
    --output "$ROUND2A/manifests/x4_train_b7.jsonl" \
    --summary "$ROUND2A/manifests/x4_train_b7.summary.json" \
    >> "$LOG" 2>&1
  "$MKPY" scripts/run_s27_student.py \
    --data-path "$DATA" --labeled-list "$LABELS" \
    --pseudo-manifest "$ROUND2A/manifests/x4_train_b7.jsonl" \
    --output-dir "$ROUND2A/students/X4" --experiment X4_isic18_round2a --seed 2026 \
    --batch-size 12 --gt-bs 3 --original-bs 3 --new-bs 6 \
    --max-iterations 40000 --val-interval 200 --num-workers 8 \
    --resume \
    >> "$LOG" 2>&1
  "$MKPY" scripts/export_t25_student_predictions.py \
    --run-dir "$ROUND2A/students/X4" \
    --checkpoint "$ROUND2A/students/X4/student_best.pth" \
    --output-root "$ROUND2A/predictions/X4_best" \
    --splits validation test \
    >> "$LOG" 2>&1
  for pred in "$ROUND1/predictions/X3_best/student_predictions_test.jsonl" "$ROUND2A/predictions/X4_best/student_predictions_test.jsonl"; do
    name=$(basename "$(dirname "$pred")")
    "$SAMPY" scripts/build_c0_256_b7_lora_manifest.py \
      --quality-root "$ROUND2A/quality_root" \
      --student-predictions "$pred" \
      --split test --modes "${MODES[@]}" \
      --min-bridge 0 --max-bridge 6 --min-b7 0 --canvas 256 \
      --output "$ROUND2A/${name}_test_b7.jsonl" \
      --summary "$ROUND2A/${name}_test_b7.summary.json" \
      >> "$LOG" 2>&1
  done
  touch "$PHASE/ROUND2A_COMPLETE"
  log "Round2A complete"
}

main() {
  log "pipeline begin, DEVICE=$DEVICE"
  run_routes
  run_base_propagation
  prepare_resized_cache
  run_round1_students
  calibrate_round1_b7_and_lora
  run_round2a
  touch "$PHASE/ALL_COMPLETE"
  log "pipeline all complete"
}

main "$@"
