#!/usr/bin/env bash
# Generate the C0-256 route pools with base-SAM3 trunk descriptors at 256.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
PHASE=work/rerun_c0_256_sam3knn_s256_base
ROUTE_ROOT="$PHASE/stage1_feature_knn_b0_b6"
PROTO=work/kvasir_1pct_anchors/protocol
FEATURE_SOURCE=work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/features/sam3_base_s256_features.npz
LOG="$PHASE/routes.log"

MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)
SPLITS=(train validation test)

mkdir -p "$ROUTE_ROOT/features" "$PHASE"
ln -sfn "$(realpath "$FEATURE_SOURCE")" "$ROUTE_ROOT/features/sam3_base_s256_features.npz"

pids=()
for mode in "${MODES[@]}"; do
  for split in "${SPLITS[@]}"; do
    echo "[routes] $(date '+%F %T') start $mode $split" | tee -a "$LOG"
    "$PY" scripts/stage1_feature_knn_routes.py \
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
      >> "$LOG" 2>&1 &
    pids+=("$!")
  done
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

# Old downstream scripts use the DINO-era directory names.  Keep the new
# canonical SAM3 names and expose read-compatible aliases inside this isolated root.
ln -sfn "$(realpath "$ROUTE_ROOT/${MODES[0]}")" \
  "$ROUTE_ROOT/anchor_conditioned_target_pooling"
ln -sfn "$(realpath "$ROUTE_ROOT/${MODES[1]}")" \
  "$ROUTE_ROOT/anchor_conditioned_patch_correspondence"

"$PY" - "$ROUTE_ROOT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
expected = {"train": 792 * 7, "validation": 100 * 7, "test": 100 * 7}
modes = (
    "sam3enc_anchor_conditioned_target_pooling",
    "sam3enc_anchor_conditioned_patch_correspondence",
)
report = {}
for mode in modes:
    report[mode] = {}
    for split, expected_count in expected.items():
        path = root / mode / f"{split}_pool0_stage1/routes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        bridges = Counter(int(row["bridge_count"]) for row in rows)
        targets = Counter(row["target_id"] for row in rows)
        if len(rows) != expected_count:
            raise SystemExit(f"{mode}/{split}: {len(rows)} != {expected_count}")
        if set(bridges) != set(range(7)) or any(value != len(targets) for value in bridges.values()):
            raise SystemExit(f"{mode}/{split}: invalid bridge coverage {bridges}")
        if any(value != 7 for value in targets.values()):
            raise SystemExit(f"{mode}/{split}: target route count is not seven")
        report[mode][split] = {
            "routes": len(rows),
            "targets": len(targets),
            "bridge_counts": dict(sorted(bridges.items())),
        }
(root / "route_generation_summary.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
PY

touch "$PHASE/ROUTES_COMPLETE"
echo "[routes] $(date '+%F %T') all routes complete" | tee -a "$LOG"
