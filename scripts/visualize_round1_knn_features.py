#!/usr/bin/env python3
"""Visualize round-1 KNN (SAM3-trunk lesion) features: do the feature maps focus on the lesion?

For sample test images: overlay round-1 pseudo mask + GT, and show per-patch cosine saliency
of the actual KNN lesion descriptor (mask-foreground token mean) vs a background prototype,
upsampled from the 72x72 token grid. Also prints quantitative focus stats (inside/outside GT,
rank-AUC). Pure analysis of cached features: no model inference.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
FEAT = ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features"
OUT = ROOT / "work/kvasir_1pct_anchors/visuals/round1_knn_features"
GRID = 72  # 1008 // 14
SIZE = 1008

def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def l2norm(x: np.ndarray, axis=-1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + 1e-12)

def mask_grid(path: str) -> np.ndarray:
    m = Image.open(path).convert("L").resize((GRID, GRID), Image.Resampling.NEAREST)
    return np.asarray(m) > 127

def auc_rank(score: np.ndarray, inside: np.ndarray) -> float:
    inside = inside.reshape(-1)
    s = score.reshape(-1)
    n_in, n_out = int(inside.sum()), int((~inside).sum())
    if n_in == 0 or n_out == 0:
        return float("nan")
    order = np.argsort(np.argsort(s)) + 1  # rank 1..N (ties by position, fine for viz)
    return float((order[inside].sum() - n_in * (n_in + 1) / 2) / (n_in * n_out))

def main() -> None:
    records = read_jsonl(ROOT / "work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl")
    # pseudo masks
    mask_map: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        jp = ROOT / f"work/kvasir_1pct_anchors/{split}_pseudo_masks_round1/train_pseudo_masks_round1.jsonl"
        if jp.exists():
            for row in read_jsonl(jp):
                mask_map[row["target_id"]] = row["pseudo_mask_path"]
    for row in read_jsonl(ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/anchor_mask_manifest.jsonl"):
        mask_map[row["target_id"]] = row["pseudo_mask_path"]
    # features
    patches = np.load(FEAT / "sam3_base_s1008_patches.npz", mmap_mode="r")
    patch_ids = patches["ids"].tolist()
    id_to_row = {str(i): k for k, i in enumerate(patch_ids)}
    md = np.load(FEAT / "sam3enc_mask_descriptors_s1008.npz", mmap_mode="r")
    md_ids = md["ids"].tolist()
    md_map = {str(i): k for k, i in enumerate(md_ids)}
    md_desc = md["descriptors"]  # (1000,1024) float32, L2-normalized
    # pick samples: test targets, by direct-route GT dice from C3 test eval (best 2 + worst 2)
    eval_rows = read_jsonl(ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_lesion_knn/sam3enc_lesion_c3/eval_c3_forward/route_results.jsonl")
    direct = {}
    for r in eval_rows:
        if r.get("route_type") == "direct" and r.get("status") == "success":
            direct.setdefault(r["target_id"], []).append(r["gt_dice_evaluation_only"])
    direct = {t: float(np.mean(v)) for t, v in direct.items()}
    ranked = sorted(direct, key=direct.get)
    samples = ranked[:2] + ranked[-2:]
    print("sample targets (direct dice):", [(t, round(direct[t], 4)) for t in samples])

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAVE_MPL = True
    except Exception:
        HAVE_MPL = False
    stats = {}
    for si, tid in enumerate(samples):
        rec = next(r for r in records if r["merged_id"] == tid)
        img = Image.open(rec["image_path"]).convert("RGB")
        W, H = img.size
        raw = FEAT / "sam3_base_s1008_patches_raw.npy"
        if raw.exists():
            parr = np.load(raw, mmap_mode="r")
            tokens = np.asarray(parr[id_to_row[tid]], dtype=np.float32)
        else:
            tokens = np.asarray(patches["patches"][id_to_row[tid]], dtype=np.float32)  # slow if compressed
        tokens = l2norm(tokens)
        fg_pseudo = mask_grid(mask_map.get(tid, rec["mask_path"]))
        fg_gt = mask_grid(rec["mask_path"])
        # actual KNN lesion descriptor (from npz) and a background prototype
        lesion = l2norm(md_desc[md_map[tid]].astype(np.float32))
        bg = l2norm(tokens[~fg_pseudo.reshape(-1)].mean(axis=0))
        sim_lesion = (tokens @ lesion).reshape(GRID, GRID)
        contrast = (tokens @ lesion - tokens @ bg).reshape(GRID, GRID)
        # focus stats on the 72x72 grid
        stats[tid] = {
            "direct_dice": round(direct[tid], 4),
            "auc_sim_lesion_vs_gt": round(auc_rank(sim_lesion, fg_gt), 4),
            "auc_contrast_vs_gt": round(auc_rank(contrast, fg_gt), 4),
            "sim_lesion_in_gt": round(float(sim_lesion[fg_gt].mean()), 4),
            "sim_lesion_out_gt": round(float(sim_lesion[~fg_gt].mean()), 4),
            "contrast_in_gt": round(float(contrast[fg_gt].mean()), 4),
            "contrast_out_gt": round(float(contrast[~fg_gt].mean()), 4),
            "sim_lesion_in_pseudo": round(float(sim_lesion[fg_pseudo].mean()), 4),
            "sim_lesion_out_pseudo": round(float(sim_lesion[~fg_pseudo].mean()), 4),
        }
        # upsampled heatmaps at image resolution
        def up(m: np.ndarray) -> np.ndarray:
            return np.asarray(Image.fromarray(m).resize((W, H), Image.BICUBIC))
        heat_l = up(sim_lesion)
        heat_c = up(contrast)
        if HAVE_MPL:
            fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
            axes[0].imshow(img); axes[0].set_title(f"original\n{tid[:20]}")
            axes[1].imshow(img)
            pm = np.array(Image.open(mask_map.get(tid, rec["mask_path"])).convert("L").resize((W, H), Image.NEAREST)) > 127
            axes[1].imshow(np.dstack([pm, np.zeros_like(pm), np.zeros_like(pm)]).astype(np.float32), alpha=0.45); axes[1].set_title("+ round1 pseudo mask")
            axes[2].imshow(img)
            gm = np.array(Image.open(rec["mask_path"]).convert("L").resize((W, H), Image.NEAREST)) > 127
            axes[2].imshow(np.dstack([np.zeros_like(gm), gm, np.zeros_like(gm)]).astype(np.float32), alpha=0.45); axes[2].set_title("+ GT mask")
            im4 = axes[3].imshow(heat_l, cmap="inferno", alpha=0.75); axes[3].imshow(img, alpha=0.35); axes[3].set_title("sim to lesion descriptor")
            fig.colorbar(im4, ax=axes[3], fraction=0.046)
            im5 = axes[4].imshow(heat_c, cmap="coolwarm", vmin=-np.abs(heat_c).max(), vmax=np.abs(heat_c).max(), alpha=0.75); axes[4].imshow(img, alpha=0.35); axes[4].set_title("lesion vs bg contrast")
            fig.colorbar(im5, ax=axes[4], fraction=0.046)
            for ax in axes: ax.axis("off")
            fig.tight_layout()
            fig.savefig(OUT / f"sample{si+1}_{tid.split('::')[-1]}.png", dpi=110)
            plt.close(fig)
        else:
            print(f"[no matplotlib] sample {tid} saved stats only")
    Path(OUT / "focus_stats.json").write_text(json.dumps({"grid": GRID, "samples": stats}, indent=2))
    print(json.dumps(stats, indent=2))
    print("outputs ->", OUT)

if __name__ == "__main__":
    main()
