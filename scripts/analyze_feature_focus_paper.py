#!/usr/bin/env python3
"""Paper-grade verification of 'mask-grounded SAM3 features focus on the lesion'.

For every validation+test image (200), computes token-level rank-AUC of each descriptor
variant's saliency against the GT mask on the 72x72 grid:
  pseudo-lesion | GT-lesion | global patch_mean (baseline) |
  negcontrol fixed_crop / random_nonoverlap / shuffled | corrupt dilate/erode/bbox/random
Also correlates pseudo-lesion focus AUC with direct-route propagation Dice.
Outputs stats JSON + publication figures (examples, AUC distributions, AUC-vs-Dice).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
FEAT = ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features"
FEAT_L = ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_lesion_knn/features"
OUT = ROOT / "work/kvasir_1pct_anchors/paper_figures/feature_focus"
GRID = 72

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
    order = np.argsort(np.argsort(s)) + 1
    return float((order[inside].sum() - n_in * (n_in + 1) / 2) / (n_in * n_out))

def load_desc(path: Path) -> dict:
    d = np.load(path, mmap_mode="r")
    if "ids" in d.files:
        ids = [str(i) for i in d["ids"].tolist()]
        desc = d["descriptors"].astype(np.float32)
        return {i: k for k, i in enumerate(ids)}, l2norm(desc)
    # global patch-mean npz: rows in merged_manifest order, no ids key
    desc = d["patch_mean"].astype(np.float32)
    return None, l2norm(desc)

def main() -> None:
    records = read_jsonl(ROOT / "work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl")
    rec_by_id = {r["merged_id"]: r for r in records}
    mask_map: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        jp = ROOT / f"work/kvasir_1pct_anchors/{split}_pseudo_masks_round1/train_pseudo_masks_round1.jsonl"
        if jp.exists():
            for row in read_jsonl(jp):
                mask_map[row["target_id"]] = row["pseudo_mask_path"]
    for row in read_jsonl(ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1/anchor_mask_manifest.jsonl"):
        mask_map[row["target_id"]] = row["pseudo_mask_path"]

    variants = {
        "pseudo_lesion": FEAT / "sam3enc_mask_descriptors_s1008.npz",
        "gt_lesion": FEAT_L / "sam3enc_gt_lesion_descriptors_s1008.npz",
        "global_patch_mean": FEAT / "sam3_base_s1008_features.npz",
        "negcontrol_fixed_crop": FEAT_L / "sam3enc_lesion_negcontrol_fixed_crop_s1008.npz",
        "negcontrol_random_nonoverlap": FEAT_L / "sam3enc_lesion_negcontrol_random_nonoverlap_s1008.npz",
        "negcontrol_shuffled": FEAT_L / "sam3enc_lesion_negcontrol_shuffled_s1008.npz",
        "corrupt_dilate": FEAT_L / "sam3enc_lesion_corrupt_dilate_s1008.npz",
        "corrupt_erode": FEAT_L / "sam3enc_lesion_corrupt_erode_s1008.npz",
        "corrupt_bbox": FEAT_L / "sam3enc_lesion_corrupt_bbox_s1008.npz",
        "corrupt_random_shift": FEAT_L / "sam3enc_lesion_corrupt_random_s1008.npz",
    }
    desc_data = {}
    for name, p in variants.items():
        if not p.exists():
            print(f"[skip missing] {name} -> {p}")
            continue
        idmap, desc = load_desc(p)
        desc_data[name] = (idmap, desc)
        print(f"loaded {name}: {desc.shape}")

    raw = FEAT / "sam3_base_s1008_patches_raw.npy"
    if raw.exists():
        patches_arr = np.load(raw, mmap_mode="r")
        patch_ids = [str(i) for i in np.load(FEAT / "sam3_base_s1008_patches.npz", mmap_mode="r")["ids"].tolist()]
        def get_tokens(row):
            return np.asarray(patches_arr[row], dtype=np.float32)
    else:
        _pz = np.load(FEAT / "sam3_base_s1008_patches.npz", mmap_mode="r")
        patch_ids = [str(i) for i in _pz["ids"].tolist()]
        def get_tokens(row):
            return np.asarray(_pz["patches"][row], dtype=np.float32)
    id_to_row = {i: k for k, i in enumerate(patch_ids)}

    # targets: validation + test
    targets = [r for r in records if r["split"] in ("validation", "test")]
    # direct dice from lesion b7 evals
    def direct_dice(eval_dir: Path) -> dict:
        rows = read_jsonl(eval_dir / "route_results.jsonl")
        out = {}
        for r in rows:
            if r.get("route_type") == "direct" and r.get("status") == "success":
                out[r["target_id"]] = float(r["gt_dice_evaluation_only"])
        return out
    dice_test = direct_dice(ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_lesion_knn/sam3enc_lesion/eval_base_no_ft_b7_forward")
    dice_val = direct_dice(ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_lesion_knn/sam3enc_lesion/eval_base_no_ft_b7_forward_validation")

    rows_out = []
    for ti, rec in enumerate(targets):
        tid = rec["merged_id"]
        row = id_to_row.get(tid)
        if row is None:
            continue
        tokens = get_tokens(row)
        tokens = l2norm(tokens)
        fg_gt = mask_grid(rec["mask_path"])
        fg_pseudo = mask_grid(mask_map.get(tid, rec["mask_path"]))
        entry = {"target_id": tid, "split": rec["split"],
                 "dice_direct": dice_test.get(tid, dice_val.get(tid)),
                 "gt_tokens": int(fg_gt.sum())}
        bg_proto = l2norm(tokens[~fg_pseudo.reshape(-1)].mean(axis=0))
        for name, (idmap, desc) in desc_data.items():
            pos = row if idmap is None else idmap.get(tid)
            if pos is None or pos >= len(desc):
                entry[name + "_auc_gt"] = None
                continue
            d = desc[pos].astype(np.float32)
            if d.shape[0] != 1024:  # global npz may store patch_mean as (1000,1024) too
                continue
            sal = (tokens @ d).reshape(GRID, GRID)
            entry[name + "_auc_gt"] = round(auc_rank(sal, fg_gt), 4)
            if name in ("pseudo_lesion", "gt_lesion"):
                contr = sal - (tokens @ bg_proto).reshape(GRID, GRID)
                entry[name + "_contrast_auc_gt"] = round(auc_rank(contr, fg_gt), 4)
            if name == "pseudo_lesion":
                entry["pseudo_auc_pseudo"] = round(auc_rank(sal, fg_pseudo), 4)
                entry["sim_in_gt"] = round(float(sal[fg_gt].mean()), 4)
                entry["sim_out_gt"] = round(float(sal[~fg_gt].mean()), 4)
        rows_out.append(entry)
        if (ti + 1) % 50 == 0:
            print(f"{ti+1}/{len(targets)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "focus_per_image.json").write_text(json.dumps(rows_out, indent=2))
    print("saved focus_per_image.json with", len(rows_out), "images")

    # aggregates
    names = [n for n in variants if n in desc_data and rows_out and rows_out[0].get(n + "_auc_gt") is not None]
    agg = {}
    for n in names:
        vals = np.array([r[n + "_auc_gt"] for r in rows_out if r[n + "_auc_gt"] is not None], dtype=float)
        agg[n] = {
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "median": round(float(np.median(vals)), 4),
            "q25": round(float(np.percentile(vals, 25)), 4),
            "q75": round(float(np.percentile(vals, 75)), 4),
        }
    # paired wilcoxon pseudo vs others
    from scipy import stats
    wil = {}
    base = "pseudo_lesion"
    for n in names:
        if n == base: continue
        pairs = [(r[base + "_auc_gt"], r[n + "_auc_gt"]) for r in rows_out
                 if r[base + "_auc_gt"] is not None and r[n + "_auc_gt"] is not None]
        if len(pairs) > 5:
            a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
            w, pv = stats.wilcoxon(a, b)
            wil[n] = {"n": len(pairs), "delta_mean": round(float((a - b).mean()), 4), "p": float(pv)}
    # contrast AUC for pseudo/gt
    for key in ("pseudo_lesion_contrast_auc_gt", "gt_lesion_contrast_auc_gt"):
        vals = np.array([r[key] for r in rows_out if r.get(key) is not None], float)
        if len(vals):
            agg[key] = {"n": int(len(vals)), "mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4),
                        "median": round(float(np.median(vals)), 4)}
    # correlation AUC(pseudo) vs dice
    corr = {}
    for split in ("validation", "test"):
        sub = [r for r in rows_out if r["split"] == split and r.get("pseudo_lesion_auc_gt") is not None and r.get("dice_direct") is not None]
        if len(sub) >= 5:
            a = np.array([r["pseudo_lesion_auc_gt"] for r in sub], float)
            b = np.array([r["dice_direct"] for r in sub], float)
            rho, p = stats.spearmanr(a, b)
            corr[split] = {"n": len(sub), "spearman_rho": round(float(rho), 4), "p": float(p)}
    out_stats = {"grid": GRID, "n_images": len(rows_out), "by_variant": agg, "wilcoxon_vs_pseudo": wil, "auc_vs_dice": corr}
    (OUT / "focus_stats_paper.json").write_text(json.dumps(out_stats, indent=2))
    print(json.dumps(out_stats, indent=2))

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Fig B: AUC distributions per variant
    fig, ax = plt.subplots(figsize=(9, 5.5))
    order = ["pseudo_lesion", "gt_lesion", "corrupt_dilate", "corrupt_erode", "corrupt_bbox",
             "corrupt_random_shift", "negcontrol_fixed_crop", "negcontrol_random_nonoverlap",
             "global_patch_mean", "negcontrol_shuffled"]
    labels = {"pseudo_lesion": "pseudo-lesion", "gt_lesion": "GT-lesion",
              "global_patch_mean": "global patch-mean",
              "negcontrol_fixed_crop": "fixed-crop", "negcontrol_random_nonoverlap": "random non-overlap",
              "negcontrol_shuffled": "shuffled tokens", "corrupt_dilate": "dilated mask",
              "corrupt_erode": "eroded mask", "corrupt_bbox": "bbox mask", "corrupt_random_shift": "random shift"}
    colors = {"pseudo_lesion": "#c0392b", "gt_lesion": "#27ae60", "global_patch_mean": "#7f8c8d",
              "negcontrol_fixed_crop": "#95a5a6", "negcontrol_random_nonoverlap": "#bdc3c7",
              "negcontrol_shuffled": "#e0e0e0", "corrupt_dilate": "#e67e22", "corrupt_erode": "#f1c40f",
              "corrupt_bbox": "#e67e22", "corrupt_random_shift": "#f39c12"}
    for n in order:
        if n not in agg: continue
        vals = np.array([r[n + "_auc_gt"] for r in rows_out if r[n + "_auc_gt"] is not None], float)
        if len(vals) == 0: continue
        ax.hist(vals, bins=24, range=(0.3, 1.0), density=True, alpha=0.55, color=colors.get(n, "#666"), label=f"{labels.get(n, n)} (mean {agg[n]['mean']:.2f})")
    ax.axvline(0.5, ls="--", color="k", lw=0.8)
    ax.set_xlabel("rank-AUC of token saliency vs GT (0.5 = chance)")
    ax.set_ylabel("density")
    ax.set_title("Mask-grounded descriptors focus the SAM3 feature map on the lesion")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "figB_auc_distributions.png", dpi=150)
    fig.savefig(OUT / "figB_auc_distributions.pdf")
    plt.close(fig)

    # Fig C: AUC(pseudo) vs direct Dice
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for split, m in (("validation", "o"), ("test", "s")):
        sub = [r for r in rows_out if r["split"] == split and r.get("pseudo_lesion_auc_gt") is not None and r.get("dice_direct") is not None]
        if not sub: continue
        a = [r["pseudo_lesion_auc_gt"] for r in sub]; b = [r["dice_direct"] for r in sub]
        ax.scatter(a, b, marker=m, s=22, alpha=0.7, label=split)
    ax.set_xlabel("focus AUC (pseudo-lesion saliency vs GT)")
    ax.set_ylabel("direct-route propagation Dice")
    ttl = "Lesion focus strength correlates with propagation quality"
    for split, info in corr.items():
        ttl += f"\n{split}: Spearman rho={info['spearman_rho']} (p={info['p']:.2e}, n={info['n']})"
    ax.set_title(ttl, fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figC_auc_vs_dice.png", dpi=150)
    fig.savefig(OUT / "figC_auc_vs_dice.pdf")
    plt.close(fig)
    print("figures ->", OUT)

if __name__ == "__main__":
    main()
