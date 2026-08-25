#!/usr/bin/env bash
# Round-2B: change only the SAM3 KNN topology and test the 2x2 factorial.
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
BASE=work/rerun_c0_256_sam3knn_s256_base
ROUND2A=work/rerun_c0_256_round2a_fixed_knn_e33
PHASE=work/rerun_c0_256_round2b_e33_topology_factorial
ROUTES=$PHASE/stage1_feature_knn_e33_b0_b6
PROTO=work/kvasir_1pct_anchors/protocol
BASE_FEATURES=work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/features/sam3_base_s256_features.npz
BASE_ROUTES=$BASE/stage1_feature_knn_b0_b6
BASE_CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
E33_ROOT=$BASE/medsam3_lora_b0_b6_e50/e33_full_evaluation
E33_CKPT=$E33_ROOT/e33_merged_video.pt
X3_PREDICTIONS=$BASE/predictions/X3_best
REPORT=work/reproduction_reports/C0_256_round2b_topology_refresh_factorial.md
LOG=$PHASE/pipeline.log
export CUDA_VISIBLE_DEVICES=${DEVICE:-0}

MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)
SPLITS=(validation test)

mkdir -p "$PHASE/features" "$ROUTES/features" "$PHASE/baseline_controls"
rm -f "$PHASE/ROUND2B_FAILED"
trap 'touch "$PHASE/ROUND2B_FAILED"' ERR

echo "[round2b] $(date '+%F %T') extract e33 encoder descriptors on GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "$LOG"
"$PY" scripts/extract_c0_256_round2b_e33_features.py \
  --checkpoint "$E33_CKPT" --base-features "$BASE_FEATURES" \
  --output "$PHASE/features/sam3_e33_s256_features.npz" \
  --summary "$PHASE/e33_feature_audit.json" --feature-size 256 --batch-size 8 \
  2>&1 | tee -a "$LOG"

# The legacy route builder expects this fixed filename. The alias exists only
# inside the isolated G1 topology and points to the provenance-tagged e33 cache.
ln -sfn "$(realpath "$PHASE/features/sam3_e33_s256_features.npz")" \
  "$ROUTES/features/sam3_base_s256_features.npz"

pids=()
for split in "${SPLITS[@]}"; do
  for mode in "${MODES[@]}"; do
    echo "[round2b] $(date '+%F %T') build e33 KNN routes $split $mode" | tee -a "$LOG"
    "$PY" scripts/stage1_feature_knn_routes.py \
      --mode "$mode" --feature-source sam3_base --feature-size 256 \
      --knn-feature patch_mean --split "$split" \
      --min-bridge 0 --max-bridge 6 --beam-width 32 \
      --protocol-root "$PROTO" --output-root "$ROUTES" \
      > "$PHASE/routes_${split}_${mode}.log" 2>&1 &
    pids+=("$!")
  done
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PY" scripts/audit_c0_256_round2b_topology.py routes \
  --base-root "$BASE_ROUTES" --e33-root "$ROUTES" \
  --splits validation test --output "$PHASE/topology_route_audit.json" \
  2>&1 | tee -a "$LOG"
touch "$PHASE/TOPOLOGY_COMPLETE"

build_b7() {
  local quality_root=$1 split=$2 output=$3
  "$PY" scripts/build_c0_256_b7_lora_manifest.py \
    --quality-root "$quality_root" \
    --student-predictions "$X3_PREDICTIONS/student_predictions_${split}.jsonl" \
    --split "$split" --modes "${MODES[@]}" \
    --min-bridge 0 --max-bridge 6 --min-b7 0 --canvas 256 \
    --all-candidates-output "${output}.all_candidates.jsonl" \
    --output "${output}.jsonl" --summary "${output}.summary.json"
}

# Re-evaluate historical cells with the exact same frozen X3 selector and
# current oracle-aware B7 script; no propagation is repeated for G0.
for split in "${SPLITS[@]}"; do
  build_b7 "$BASE_ROUTES" "$split" "$PHASE/baseline_controls/G0_Tbase_${split}" \
    2>&1 | tee -a "$LOG"
  build_b7 "$ROUND2A/quality_root" "$split" "$PHASE/baseline_controls/G0_Te33_${split}" \
    2>&1 | tee -a "$LOG"
done

# Validation-first ordering: complete both teacher cells on validation before
# looking at test. Identical routes already run for the same checkpoint are
# safely reused, including duplicates across the two route-construction modes.
for split in "${SPLITS[@]}"; do
  for teacher in base e33; do
    teacher_root=$PHASE/teachers/$teacher
    quality_root=$teacher_root/quality_root
    if [[ $teacher == base ]]; then
      checkpoint=$BASE_CKPT
      previous_quality_root=$BASE_ROUTES
    else
      checkpoint=$E33_CKPT
      previous_quality_root=$ROUND2A/quality_root
    fi
    mkdir -p "$teacher_root/b7"
    for mode in "${MODES[@]}"; do
      mkdir -p "$quality_root/$mode"
      ln -sfn "$(realpath "$ROUTES/$mode/${split}_pool0_stage1")" \
        "$quality_root/$mode/${split}_pool0_stage1"
      "$PY" scripts/audit_c0_256_round2b_topology.py seed \
        --route-root "$ROUTES" --quality-root "$quality_root" \
        --source-roots "$previous_quality_root" \
        --mode "$mode" --split "$split" 2>&1 | tee -a "$LOG"
      echo "[round2b] $(date '+%F %T') propagate teacher=$teacher split=$split mode=$mode" | tee -a "$LOG"
      "$PY" scripts/eval_route_propagation_quality.py \
        --checkpoint "$checkpoint" --mode "$mode" --root "$quality_root" \
        --split "$split" --canvas 256 --resume 2>&1 | tee -a "$LOG"
    done
    "$PY" scripts/summarize_c0_256_bridge_metrics.py \
      --quality-root "$quality_root" --split "$split" --modes "${MODES[@]}" \
      --min-bridge 0 --max-bridge 6 \
      --output-json "$teacher_root/two_mode_b0_b6_${split}.json" \
      --output-tsv "$teacher_root/two_mode_b0_b6_${split}.tsv" \
      2>&1 | tee -a "$LOG"
    build_b7 "$quality_root" "$split" "$teacher_root/b7/X3_best_${split}" \
      2>&1 | tee -a "$LOG"
    touch "$teacher_root/${split^^}_COMPLETE"
  done
done

"$PY" scripts/summarize_c0_256_round2b_factorial.py \
  --phase "$PHASE" --base-phase "$BASE" --round2a "$ROUND2A" \
  --output-json "$PHASE/factorial_2x2_summary.json" --report "$REPORT" \
  2>&1 | tee -a "$LOG"

touch "$PHASE/ROUND2B_COMPLETE"
echo "[round2b] $(date '+%F %T') complete; no train propagation, pseudo pool, or new training" | tee -a "$LOG"
