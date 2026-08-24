#!/usr/bin/env python3
"""Build deterministic anchor-count subsets (8 -> N) for the SAM3-enc KNN ablation.

For each requested N this script:
  1. Reads the 8 GT anchors from the protocol support manifest;
  2. Deterministically orders them (seed-based permutation over merged_id-sorted
     anchors) and keeps the first N -- nested subsets: order[:6] ⊇ order[:4] ⊇ ...
     so the trend "removing anchors one by one" is clean and reproducible;
  3. RECORDS the full selection (seed, algorithm, ordered anchor ids, selected /
     excluded ids, timestamp, support-manifest path) into the output root, so the
     experiment can be reproduced exactly;
  4. Recomputes per-anchor pooled / cond_target / cond_correspondence from the
     cached @1008 patch tokens (extract_sam3_encoder_patches.py) for the N anchors
     only, and writes {output_root}/features/sam3_base_s1008_features.npz in the
     same format stage1_feature_knn_routes.py expects;
  5. Writes {output_root}/protocol/support_manifest_n{N}.jsonl for --support-manifest.

KNN variant fixed: sam3enc_anchor_conditioned_target_pooling + --knn-feature cond.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
STAGE1 = ROOT / "scripts/stage1_feature_knn_routes.py"
spec = importlib.util.spec_from_file_location("stage1", STAGE1)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {STAGE1}")
stage1 = importlib.util.module_from_spec(spec)
sys.modules["stage1"] = stage1
spec.loader.exec_module(stage1)

DEFAULT_PATCHES = (
    ROOT
    / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/sam3_base_s1008_patches.npz"
)


def recompute_cond(records, support, patches_path: Path, source_npz: Path, feature_size: int, output: Path) -> None:
    """Recompute per-anchor pooled / cond scores from cached patch tokens.

    patch_mean is anchor-independent and copied from the existing 8-anchor npz
    (same trunk + normalization), so the cached patches are only used for the
    per-anchor quantities.
    """
    grid = feature_size // 14
    data = np.load(patches_path, mmap_mode="r")
    patches_all = data["patches"]  # (1000, P, C) fp16 mmap
    id_to_idx = {row["merged_id"]: i for i, row in enumerate(records)}
    if source_npz.exists():
        patch_mean = np.load(source_npz)["patch_mean"]
    else:
        sums = np.zeros((len(records), patches_all.shape[2]), dtype=np.float32)
        for start in range(0, len(records), 50):
            end = min(start + 50, len(records))
            sums[start:end] = patches_all[start:end].astype(np.float32).mean(axis=1)
        patch_mean = stage1.l2norm(sums.astype(np.float32))

    proto_list = []
    for anchor in support:
        idx = id_to_idx[anchor["merged_id"]]
        tok = patches_all[idx].astype(np.float32)
        fg = stage1.load_mask_grid(anchor["frozen_mask_path"], grid).reshape(-1)
        sel = tok[fg]
        if len(sel) == 0:
            sel = tok
        proto_list.append(stage1.l2norm(sel.mean(axis=0, keepdims=True))[0])
    proto = np.stack(proto_list).astype(np.float32)  # (A, C)
    A, C = proto.shape

    pooled_rows = np.zeros((A, len(records), C), dtype=np.float32)
    cond_t_rows = np.zeros((A, len(records)), dtype=np.float32)
    cond_c_rows = np.zeros((A, len(records)), dtype=np.float32)
    batch = 50
    for start in range(0, len(records), batch):
        end = min(start + batch, len(records))
        p = patches_all[start:end].astype(np.float32)  # (B, P, C)
        sims = np.einsum("ad,npd->anp", proto, p).astype(np.float32)  # (A, B, P)
        weights = np.exp((sims - sims.max(axis=2, keepdims=True)) * 10.0)
        weights = weights / np.maximum(weights.sum(axis=2, keepdims=True), 1e-12)
        pooled = stage1.l2norm(np.einsum("anp,npd->and", weights, p).astype(np.float32), axis=2)
        pooled_rows[:, start:end] = pooled
        cond_t_rows[:, start:end] = np.einsum("and,ad->an", pooled, proto).astype(np.float32)
        cond_c_rows[:, start:end] = np.sort(sims, axis=2)[:, :, -8:].mean(axis=2).astype(np.float32)
        print(f"cond {end}/{len(records)}", flush=True)

    np.savez_compressed(
        output,
        patch_mean=patch_mean,
        anchor_ids=np.asarray([a["merged_id"] for a in support]),
        anchor_prototypes=proto,
        pooled=pooled_rows,
        cond_target=cond_t_rows,
        cond_correspondence=cond_c_rows,
        feature_source="sam3_base",
        feature_size=feature_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-anchors", type=int, required=True, help="Number of anchors to keep (6/4/3/2/1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/protocol")
    parser.add_argument("--patches", type=Path, default=DEFAULT_PATCHES)
    parser.add_argument(
        "--source-npz",
        type=Path,
        default=ROOT
        / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/sam3_base_s1008_features.npz",
        help="Existing 8-anchor npz (only patch_mean is reused; anchor-independent).",
    )
    parser.add_argument("--feature-size", type=int, default=1008)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_anchor_ablation",
    )
    args = parser.parse_args()
    if args.n_anchors not in (6, 4, 3, 2, 1):
        raise SystemExit("--n-anchors must be one of 6/4/3/2/1")
    if not args.patches.exists():
        raise SystemExit(f"Missing patch cache: {args.patches} (run extract_sam3_encoder_patches.py first)")

    support = stage1.read_jsonl(args.protocol_root / "support_manifest.jsonl")
    if len(support) != 8:
        raise SystemExit(f"Expected 8 anchors, got {len(support)}")

    # Deterministic nested selection: seed permutation over merged_id-sorted anchors
    rng = np.random.default_rng(args.seed)
    ordered = sorted(support, key=lambda a: a["merged_id"])
    perm = rng.permutation(len(ordered)).tolist()
    order = [ordered[i] for i in perm]
    selected = order[: args.n_anchors]
    selected_ids = [a["merged_id"] for a in selected]
    excluded_ids = [a["merged_id"] for a in order[args.n_anchors :]]

    out = args.output_root / (f"n{args.n_anchors}" if args.seed == 42 else f"n{args.n_anchors}_s{args.seed}")
    features_dir = out / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    protocol_out = out / "protocol"
    protocol_out.mkdir(parents=True, exist_ok=True)

    # Record the selection (reproducibility)
    record = {
        "experiment": "sam3enc_1008_anchor_count_ablation",
        "mode": "sam3enc_anchor_conditioned_target_pooling",
        "knn_feature": "cond",
        "feature_source": "sam3_base",
        "feature_size": args.feature_size,
        "n_anchors": args.n_anchors,
        "seed": args.seed,
        "selection_algorithm": "numpy default_rng(seed).permutation over merged_id-sorted anchors, keep first N (nested)",
        "full_ordered_ids": [a["merged_id"] for a in order],
        "selected_anchor_ids": selected_ids,
        "excluded_anchor_ids": excluded_ids,
        "selected_anchor_mask_paths": [a.get("frozen_mask_path") or a.get("mask_path") for a in selected],
        "support_manifest_path": str(protocol_out / f"support_manifest_n{args.n_anchors}.jsonl"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    record_path = out / "anchor_selection.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stage1.write_jsonl(protocol_out / f"support_manifest_n{args.n_anchors}.jsonl", selected)

    # Recompute per-anchor cond from cached patches
    records = stage1.read_jsonl(args.protocol_root / "merged_manifest.jsonl")
    npz_path = features_dir / f"sam3_base_s{args.feature_size}_features.npz"
    recompute_cond(records, selected, args.patches, args.source_npz, args.feature_size, npz_path)

    meta = {k: v for k, v in record.items() if k != "selected_anchor_mask_paths"}
    meta["npz"] = str(npz_path)
    meta["patches_cache"] = str(args.patches)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"n_anchors": args.n_anchors, "selected": selected_ids,
                      "record": str(record_path), "npz": str(npz_path)}, indent=2))


if __name__ == "__main__":
    main()
