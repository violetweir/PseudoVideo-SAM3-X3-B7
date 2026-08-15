#!/usr/bin/env bash
set -euo pipefail

PV_ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
DATA=$PV_ROOT/work/kvasir_1pct_anchors/baseline_data
LABELS=$PV_ROOT/work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt
LOG_DIR=$PV_ROOT/work/kvasir_1pct_anchors/logs
mkdir -p "$LOG_DIR"

run_scsam() {
  export CUDA_VISIBLE_DEVICES=0
  export SC_SAM_ROOT=$PV_ROOT/third_party/SC-SAM
  local OUT=$PV_ROOT/work/kvasir_1pct_anchors/scsam_kvasir_1pct
  mkdir -p "$OUT"
  cd "$PV_ROOT/third_party/SC-SAM"
  "$PY" -u run_merged_scsam.py \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --labeled_list "$LABELS" \
    --sam_checkpoint /Data_8TB/lht/models/sam_vit_b_01ec64.pth \
    --seed 2026 \
    --split_seed 2026 \
    --batch_size 12 \
    --labeled_bs 6 \
    --mixed_iterations 10000 \
    --max_iterations 40000 \
    --val_interval 200 \
    --num_workers 4 \
    --mode train
  "$PY" -u run_merged_scsam.py \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --labeled_list "$LABELS" \
    --sam_checkpoint /Data_8TB/lht/models/sam_vit_b_01ec64.pth \
    --trained_sam_checkpoint "$OUT/sam_best_model.pth" \
    --unet_checkpoint "$OUT/Unet_best_model.pth" \
    --seed 2026 \
    --split_seed 2026 \
    --batch_size 12 \
    --labeled_bs 6 \
    --mixed_iterations 10000 \
    --max_iterations 40000 \
    --val_interval 200 \
    --num_workers 1 \
    --mode test
}

run_synfoc() {
  # SynFoC train.py rewrites CUDA_VISIBLE_DEVICES from --gpu, so pass the physical GPU id here.
  export CUDA_VISIBLE_DEVICES=1
  local OUT=$PV_ROOT/work/kvasir_1pct_anchors/synfoc_kvasir_1pct
  mkdir -p "$OUT"
  cd "$PV_ROOT/third_party/SynFoC-T20"
  "$PY" -u train.py \
    --dataset ClinicDB \
    --dataset_label "Kvasir-SEG 1pct anchors" \
    --data_path "$DATA" \
    --labeled_list "$LABELS" \
    --output_dir "$OUT" \
    --save_name fixed8_seed2026 \
    --model MedSAM \
    --ckpt /Data_8TB/lht/models/medsam_vit_b.pth \
    --max_iterations 40000 \
    --num_eval_iter 500 \
    --label_bs 4 \
    --unlabel_bs 4 \
    --test_bs 8 \
    --img_size 256 \
    --base_lr 0.03 \
    --seed 2026 \
    --AdamW \
    --warmup \
    --save_model \
    --overwrite \
    --gpu 1
}

case "${1:-}" in
  scsam) run_scsam ;;
  synfoc) run_synfoc ;;
  *) echo "usage: $0 scsam|synfoc" >&2; exit 2 ;;
esac
