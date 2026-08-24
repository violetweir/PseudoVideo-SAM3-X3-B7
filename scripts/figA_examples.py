#!/usr/bin/env python3
"""figA: three representative examples (low / mid / high lesion focus by contrast AUC)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
FEAT = ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features"
OUT = ROOT / "work/kvasir_1pct_anchors/paper_figures/feature_focus"
GRID = 72

def read_jsonl(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def l2norm(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True); return x / (n + 1e-12)
def mask_grid(path):
    m = Image.open(path).convert("L").resize((GRID, GRID), Image.Resampling.NEAREST)
    return np.asarray(m) > 127

per = json.loads((OUT / "focus_per_image.json").read_text())
test = [r for r in per if r["split"] == "test" and r.get("pseudo_lesion_contrast_auc_gt") is not None]
test.sort(key=lambda r: r["pseudo_lesion_contrast_auc_gt"])
samples = [test[0], test[len(test)//2], test[-1]]
print("samples:", [(s["target_id"].split("::")[-1], round(s["pseudo_lesion_contrast_auc_gt"],3), s.get("dice_direct")) for s in samples])

records = {r["merged_id"]: r for r in read_jsonl(ROOT / "work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl")}
mask_map = {}
for split in ("train", "validation", "test"):
    jp = ROOT / f"work/kvasir_1pct_anchors/{split}_pseudo_masks_round1/train_pseudo_masks_round1.jsonl"
    if jp.exists():
        for row in read_jsonl(jp): mask_map[row["target_id"]] = row["pseudo_mask_path"]
for row in read_jsonl(ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/anchor_mask_manifest.jsonl"):
    mask_map[row["target_id"]] = row["pseudo_mask_path"]
patch_ids = [str(i) for i in np.load(FEAT / "sam3_base_s1008_patches.npz", mmap_mode="r")["ids"].tolist()]
id_to_row = {i: k for k, i in enumerate(patch_ids)}
parr = np.load(FEAT / "sam3_base_s1008_patches_raw.npy", mmap_mode="r")
md = np.load(FEAT / "sam3enc_mask_descriptors_s1008.npz", mmap_mode="r")
md_ids = [str(i) for i in md["ids"].tolist()]
md_row = {i: k for k, i in enumerate(md_ids)}
md_desc = l2norm(md["descriptors"].astype(np.float32))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def render(rec, tid, fname):
    img = Image.open(rec["image_path"]).convert("RGB")
    W, H = img.size
    tokens = l2norm(np.asarray(parr[id_to_row[tid]], dtype=np.float32))
    fg_pseudo = mask_grid(mask_map.get(tid, rec["mask_path"]))
    fg_gt = mask_grid(rec["mask_path"])
    lesion = l2norm(md_desc[md_row[tid]].astype(np.float32))
    bg = l2norm(tokens[~fg_pseudo.reshape(-1)].mean(axis=0))
    sim = (tokens @ lesion).reshape(GRID, GRID)
    contr = sim - (tokens @ bg).reshape(GRID, GRID)
    def up(m): return np.asarray(Image.fromarray(m).resize((W, H), Image.BICUBIC))
    pm = np.array(Image.open(mask_map.get(tid, rec["mask_path"])).convert("L").resize((W, H), Image.NEAREST)) > 127
    gm = np.array(Image.open(rec["mask_path"]).convert("L").resize((W, H), Image.NEAREST)) > 127
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
    axes[0].imshow(img); axes[0].set_title("input")
    axes[1].imshow(img); axes[1].imshow(np.dstack([pm, np.zeros_like(pm), np.zeros_like(pm)]).astype(np.float32), alpha=0.45); axes[1].set_title("round-1 pseudo mask")
    axes[2].imshow(img); axes[2].imshow(np.dstack([np.zeros_like(gm), gm, np.zeros_like(gm)]).astype(np.float32), alpha=0.45); axes[2].set_title("GT mask")
    im4 = axes[3].imshow(up(sim), cmap="inferno", alpha=0.75); axes[3].imshow(img, alpha=0.35); axes[3].set_title("saliency to lesion descriptor")
    fig.colorbar(im4, ax=axes[3], fraction=0.046)
    im5 = axes[4].imshow(up(contr), cmap="coolwarm", vmin=-np.abs(up(contr)).max(), vmax=np.abs(up(contr)).max(), alpha=0.75); axes[4].imshow(img, alpha=0.35); axes[4].set_title("lesion vs background")
    fig.colorbar(im5, ax=axes[4], fraction=0.046)
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)

for i, s in enumerate(samples):
    render(records[s["target_id"]], s["target_id"], OUT / f"figA_example{i+1}_{s['target_id'].split('::')[-1]}.png")
print("figA saved ->", OUT)
