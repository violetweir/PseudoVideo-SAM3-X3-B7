#!/usr/bin/env python3
"""C1/C2 (+C3-lite): cross-route consensus analysis, grouped by (target, anchor).
Zero new inference. Usage: python scripts/analyze_consensus.py <mode_dir> [out_json]
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

def load_mask(p):
    return np.array(Image.open(str(p)).convert("L")) > 127

def dice(a, b):
    inter = (a & b).sum()
    denom = a.sum() + b.sum()
    return 2.0 * inter / (denom + 1e-9) if denom > 0 else 0.0

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 3: return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    d = rx - ry
    return 1.0 - 6.0 * (d * d).sum() / (n * (n * n - 1))

def analyze(eval_dir: Path):
    lines = [json.loads(l) for l in (eval_dir / "route_results.jsonl").open()]
    lines = [r for r in lines if r.get("status") == "success"]
    by_pair = {}
    for r in lines:
        by_pair.setdefault((r["target_id"], r["anchor_id"]), []).append(r)

    per_target = {}
    for (tid, aid), routes in by_pair.items():
        routes = sorted(routes, key=lambda r: r.get("bridge_count", 0))
        masks = [load_mask(Path(r["forward_mask_path"])) for r in routes]
        K = len(routes)
        C = np.eye(K)
        for i in range(K):
            for j in range(i + 1, K):
                d = dice(masks[i], masks[j])
                C[i, j] = C[j, i] = d
        Q = (C.sum(axis=1) - 1.0) / (K - 1.0)
        gt = np.array([r["gt_dice_evaluation_only"] for r in routes])
        per_target.setdefault(tid, []).append(dict(anchor=aid, routes=routes, masks=masks, C=C, Q=Q, gt=gt))

    results = []
    route_Q_all, route_gt_all = [], []
    for tid, groups in per_target.items():
        # --- C1: medoid over ALL routes of target (cross-anchor pool) ---
        all_masks = [m for g in groups for m in g["masks"]]
        all_gt = np.array([g2 for g in groups for g2 in g["gt"]])
        n_all = len(all_masks)
        C_all = np.eye(n_all)
        for i in range(n_all):
            for j in range(i + 1, n_all):
                d = dice(all_masks[i], all_masks[j])
                C_all[i, j] = C_all[j, i] = d
        Q_all = (C_all.sum(axis=1) - 1.0) / (n_all - 1.0)
        med_all = int(np.argmax(Q_all))
        # --- within-anchor medoids ---
        within = []
        for g in groups:
            if len(g["routes"]) >= 2:
                m = int(np.argmax(g["Q"]))
                within.append(dict(anchor=g["anchor"], Q=g["Q"][m], gt=g["gt"][m], mean_agr=float(g["C"].mean())))
        # cross-anchor consensus: pick anchor whose within-anchor medoid is most self-consistent
        best_anchor_med = max(within, key=lambda w: w["Q"]) if within else None
        # majority vote over all routes
        V_all = (np.stack(all_masks).astype(np.int16).sum(0) > (n_all / 2))
        gt_mask = np.array(Image.open(groups[0]["routes"][0]["target_mask_path_evaluation_only"]).convert("L").resize((256, 256), Image.NEAREST)) > 127
        # within-anchor medoid majority vote
        if len(within) >= 2:
            V_within = (np.stack([w_m for w_m in [groups[gi]["masks"][int(np.argmax(groups[gi]["Q"]))] for gi in range(len(groups)) if len(groups[gi]["routes"]) >= 2]]).astype(np.int16).sum(0) > (len(within) / 2))
            mv_within_dice = dice(V_within, gt_mask)
        else:
            mv_within_dice = float("nan")
        route_Q_all += list(Q_all); route_gt_all += list(all_gt)
        results.append(dict(
            target=tid, n_anchors=len(groups), n_routes=n_all,
            gt_all=all_gt.tolist(), Q_all=Q_all.tolist(), med_all=med_all,
            med_all_dice=float(all_gt[med_all]),
            best_anchor_med_dice=float(best_anchor_med["gt"]) if best_anchor_med else None,
            best_anchor_med_Q=float(best_anchor_med["Q"]) if best_anchor_med else None,
            within_n=len(within),
            mean_agreement=float(np.mean([w["mean_agr"] for w in within])) if within else float("nan"),
            mv_all_dice=float(dice(V_all, gt_mask)),
            mv_within_dice=mv_within_dice,
        ))
    n = len(results)
    gt_mat = np.array([r["gt_all"] for r in results])
    med_dice = np.array([r["med_all_dice"] for r in results])
    best_anchor_med = np.array([r["best_anchor_med_dice"] if r["best_anchor_med_dice"] is not None else r["med_all_dice"] for r in results])
    mv_all = np.array([r["mv_all_dice"] for r in results])
    mv_within = np.array([r["mv_within_dice"] for r in results])
    fixed_k = {}
    for k in range(gt_mat.shape[1]):  # index-based, sorted by bridge within... not per-b exactly
        pass
    # fixed route_type means (group across targets by route_type of the k-th sorted route is not exact; use route_type label)
    from collections import defaultdict
    rt_sums = defaultdict(list)
    for gs in per_target.values():
        for g in gs:
            for r, gt in zip(g["routes"], g["gt"]):
                rt_sums[r["route_type"]].append(gt)
    fixed_rt = {k: float(np.mean(v)) for k, v in sorted(rt_sums.items())}
    mean_agr = np.array([r["mean_agreement"] for r in results])
    oracle = gt_mat.max(axis=1)
    out = {
        "pool": str(eval_dir), "n_targets": n,
        "fixed_route_type_mean_dice": fixed_rt,
        "per_target_mean_dice": float(gt_mat.mean()),
        "oracle_per_target_best": float(oracle.mean()),
        "medoid_all_routes_mean_dice": float(med_dice.mean()),
        "best_within_anchor_medoid_mean_dice": float(best_anchor_med.mean()),
        "majority_vote_all_mean_dice": float(mv_all.mean()),
        "majority_vote_within_anchor_medoids_mean_dice": float(np.nanmean(mv_within)),
        "gap_oracle_minus_medoid_all": float(oracle.mean() - med_dice.mean()),
        "medoid_beats_best_fixed_frac": float((med_dice >= max(fixed_rt.values())).mean()),
        "route_level_spearman_Q_gt": float(spearman(route_Q_all, route_gt_all)),
        "target_level_spearman_meanagr_vs_mean_gt": float(spearman(mean_agr, gt_mat.mean(axis=1))),
        "target_level_spearman_meanagr_vs_oracle": float(spearman(mean_agr, oracle)),
        "target_level_spearman_Qmed_all_vs_gtmed_all": float(spearman([r["Q_all"][r["med_all"]] for r in results], med_dice)),
    }
    return out

if __name__ == "__main__":
    mode_dir = Path(sys.argv[1])
    outs = {}
    for ed in sorted((mode_dir).glob("eval_base_no_ft_b7_forward*")):
        if not (ed / "route_results.jsonl").exists(): continue
        tag = "validation" if ed.name.endswith("_validation") else "test"
        outs[tag] = analyze(ed)
        print(f"===== {tag} =====")
        print(json.dumps(outs[tag], indent=2))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(outs, indent=2))
