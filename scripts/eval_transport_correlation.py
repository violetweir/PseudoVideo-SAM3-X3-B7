#!/usr/bin/env python3
"""E5: does lesion similarity predict real SAM3 transportability better than global?

Sample pairs (i, j) from the train pool, actually propagate mask i -> j with the
frozen SAM3 video model, and compare:
  rho(global_sim, q)  vs  rho(lesion_sim, q)
where q = Dice(propagated mask on j, round-1 pseudo mask of j)  [pseudo GT, analysis only]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
T21_PATH = ROOT / "scripts/run_t21_dynamic_pseudovideo.py"
spec = importlib.util.spec_from_file_location("t21_dynamic", T21_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {T21_PATH}")
t21 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t21
spec.loader.exec_module(t21)

STAGE1 = ROOT / "scripts/stage1_feature_knn_routes.py"
spec = importlib.util.spec_from_file_location("stage1", STAGE1)
stage1 = importlib.util.module_from_spec(spec)
sys.modules["stage1"] = stage1
spec.loader.exec_module(stage1)

SAM3_CKPT = "/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt"


def load_mask(path: str, size=256) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.NEAREST)) > 127


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    return float(2 * inter / (a.sum() + b.sum())) if (a.sum() + b.sum()) else 0.0


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.25, 0.25, 0.5, 0.5)
    h, w = mask.shape
    return (xs.min() / w, ys.min() / h, (xs.max() - xs.min() + 1) / w, (ys.max() - ys.min() + 1) / h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canvas", type=int, default=256)
    parser.add_argument("--protocol-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/protocol")
    parser.add_argument("--output", type=Path, default=ROOT / "work/kvasir_1pct_anchors/e5_transport_correlation.json")
    args = parser.parse_args()

    records = stage1.read_jsonl(args.protocol_root / "merged_manifest.jsonl")
    pseudo_map: dict[str, str] = {}
    for d in ("train_pseudo_masks_round1", "validation_pseudo_masks_round1", "test_pseudo_masks_round1"):
        p = ROOT / "work/kvasir_1pct_anchors" / d / "train_pseudo_masks_round1.jsonl"
        if p.exists():
            for row in stage1.read_jsonl(p):
                pseudo_map[row["target_id"]] = row["pseudo_mask_path"]
    am = ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/anchor_mask_manifest.jsonl"
    if am.exists():
        for row in stage1.read_jsonl(am):
            pseudo_map[row["target_id"]] = row["pseudo_mask_path"]

    train = [r for r in records if r["split"] == "train" and r["merged_id"] in pseudo_map]
    rng = random.Random(args.seed)
    pairs = [(rng.choice(train), rng.choice(train)) for _ in range(args.n_pairs)]

    pm = np.load(ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/sam3_base_s1008_features.npz")["patch_mean"]
    md = np.load(ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/sam3enc_mask_descriptors_s1008.npz")
    les = {str(i): np.asarray(d, np.float32) for i, d in zip(md["ids"].tolist(), md["descriptors"])}
    id2idx = {r["merged_id"]: k for k, r in enumerate(records)}

    from sam3.model_builder import build_sam3_video_model
    model = build_sam3_video_model(checkpoint_path=SAM3_CKPT, load_from_HF=False, device="cuda", compile=False)
    model.eval()

    rows = []
    for n, (i, j) in enumerate(pairs, 1):
        mi = load_mask(pseudo_map[i["merged_id"]], args.canvas)
        box = mask_bbox(mi)
        fwd = t21.propagate(model, [i["image_path"], j["image_path"]], box, args.canvas)
        q = dice(fwd["mask"], load_mask(pseudo_map[j["merged_id"]], args.canvas))
        ii, jj = id2idx[i["merged_id"]], id2idx[j["merged_id"]]
        sg = float(pm[ii] @ pm[jj])
        sl = float(les[i["merged_id"]] @ les[j["merged_id"]])
        rows.append({"i": i["merged_id"], "j": j["merged_id"], "q": q, "s_global": sg, "s_lesion": sl})
        print(f"[{n}/{len(pairs)}] q={q:.3f} sg={sg:.3f} sl={sl:.3f}", flush=True)

    q = np.array([r["q"] for r in rows]); sg = np.array([r["s_global"] for r in rows]); sl = np.array([r["s_lesion"] for r in rows])
    rg, pg = spearmanr(sg, q); rl, pl = spearmanr(sl, q)
    out = {
        "n_pairs": len(rows), "canvas": args.canvas,
        "spearman_global": float(rg), "p_global": float(pg),
        "spearman_lesion": float(rl), "p_lesion": float(pl),
        "q_mean": float(q.mean()),
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
