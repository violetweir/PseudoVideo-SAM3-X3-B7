#!/usr/bin/env python3
"""C3 analysis: consensus selectors and independence metrics on a C3 route pool.
Usage: python scripts/analyze_c3.py <eval_dir> [out_json]
Reported per group:
  C3-B direct-only (3 anchors x b0) | C3-C top1 (3 anchors x {b0,b1,b2} top beam)
  C3-D all (with diversity, top-2 beams) | C3-A within-anchor (anchor medoids)
Selectors: ordinary medoid | anchor-balanced | cross-anchor support (tau scan)
Metrics: route-level Spearman rho(Q,GT), selector regret vs oracle, selection gain vs per-target mean.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

def load_mask(p):
    return np.array(Image.open(str(p)).convert("L")) > 127

def dice(a, b):
    inter = (a & b).sum(); den = a.sum() + b.sum()
    return 2.0 * inter / (den + 1e-9) if den > 0 else 0.0

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n < 3: return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    d = rx - ry
    return 1.0 - 6.0 * (d * d).sum() / (n * (n * n - 1))

def pair_dice_matrix(masks):
    K = len(masks); C = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            d = dice(masks[i], masks[j]); C[i, j] = C[j, i] = d
    return C

def overlap_jaccard(route_a, route_b):
    va = {route_a["anchor_id"], *route_a["bridge_ids"], route_a["target_id"]}
    vb = {route_b["anchor_id"], *route_b["bridge_ids"], route_b["target_id"]}
    inter = len(va & vb); union = len(va | vb)
    return inter / union if union else 0.0

def selectors_for(C, masks, gt, routes, anchor_ids, tau=0.85):
    """Return dict of selector -> (selected dice, score vector info)."""
    K = len(routes)
    Q_med = (C.sum(axis=1) - 1.0) / (K - 1.0)
    med = int(np.argmax(Q_med))
    # anchor-balanced: each anchor votes with mean agreement of its routes
    uniq = sorted(set(anchor_ids))
    Q_ab = np.zeros(K)
    for i in range(K):
        parts = []
        for a in uniq:
            js = [j for j in range(K) if anchor_ids[j] == a and j != i]
            if js:
                parts.append(float(np.mean(C[i, js])))
        Q_ab[i] = float(np.mean(parts)) if parts else 0.0
    ab = int(np.argmax(Q_ab))
    # cross-anchor support: count anchors with a similar candidate above tau; tie-break by Q_ab
    support = np.zeros(K, dtype=int)
    for i in range(K):
        for a in uniq:
            js = [j for j in range(K) if anchor_ids[j] == a]
            if any(C[i, j] > tau for j in js):
                support[i] += 1
    cand = sorted(range(K), key=lambda i: (-support[i], -Q_ab[i]))
    cs = cand[0]
    return {
        "medoid": dict(idx=med, dice=float(gt[med])),
        "anchor_balanced": dict(idx=ab, dice=float(gt[ab])),
        "cross_anchor_support": dict(idx=cs, dice=float(gt[cs]), support=int(support[cs])),
        "Q_med": Q_med.tolist(), "Q_ab": Q_ab.tolist(), "support": support.tolist(),
    }

def analyze(eval_dir: Path, tau: float = 0.85):
    lines = [json.loads(l) for l in (eval_dir / "route_results.jsonl").open()]
    lines = [r for r in lines if r.get("status") == "success"]
    by_target = {}
    for r in lines:
        by_target.setdefault(r["target_id"], []).append(r)

    agg = {g: {"sel": {s: [] for s in ("medoid", "anchor_balanced", "cross_anchor_support")},
               "mean": [], "oracle": [], "rho_Q_gt": [], "pair_overlap": [], "rho_pair_dice_diff": []}
           for g in ("B_direct", "C_top1", "D_all", "A_within")}
    route_pool = {g: {"Q": [], "gt": [], "overlap": []} for g in ("B_direct", "C_top1", "D_all")}

    for tid, routes in by_target.items():
        routes = sorted(routes, key=lambda r: (r["anchor_id"], int(r["bridge_count"]), r["route_id"]))
        anchor_ids = [r["anchor_id"] for r in routes]
        masks = [load_mask(Path(r["forward_mask_path"])) for r in routes]
        gt = np.array([r["gt_dice_evaluation_only"] for r in routes])
        C = pair_dice_matrix(masks)
        # overlap (frames shared) between all route pairs
        ov = np.zeros((len(routes), len(routes)))
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                o = overlap_jaccard(routes[i], routes[j])
                ov[i, j] = ov[j, i] = o

        # --- C3-B: direct only (one per anchor) ---
        idx_b = [i for i in range(len(routes)) if routes[i]["bridge_count"] == 0]
        # --- C3-C: best beam per (anchor, depth) by route score -> 3 anchors x {b0,b1,b2} ---
        best_by_key = {}
        for i in range(len(routes)):
            key = (routes[i]["anchor_id"], routes[i]["bridge_count"])
            cur = best_by_key.get(key)
            if cur is None or routes[i]["path_mean_similarity"] > routes[cur]["path_mean_similarity"]:
                best_by_key[key] = i
        idx_c = sorted(best_by_key.values())
        # --- C3-D: all routes ---
        idx_d = list(range(len(routes)))
        # --- C3-A: within-anchor medoid for anchors with >=2 routes (multi-depth) ---
        a_med_dice = []
        for a in sorted(set(anchor_ids)):
            js = [i for i in range(len(routes)) if anchor_ids[i] == a]
            if len(js) >= 2:
                sub = js
                Qs = [(C[i].sum() - 1.0) / (len(sub) - 1.0) for i in sub]
                m = sub[int(np.argmax(Qs))]
                a_med_dice.append(float(gt[m]))
        if a_med_dice:
            agg["A_within"]["mean"].append(float(np.mean(a_med_dice)))
            agg["A_within"]["oracle"].append(float(np.max([gt[j] for j in range(len(routes))])))
            # selected = best within-anchor medoid by its within-agreement (uses all routes)
            sel_by_agr = max([(i, (C[i].sum() - 1.0) / (len([k for k in range(len(routes)) if anchor_ids[k] == anchor_ids[i]]) - 1.0)) for i in range(len(routes)) if len([k for k in range(len(routes)) if anchor_ids[k] == anchor_ids[i]]) >= 2], key=lambda t: t[1])
            agg["A_within"]["sel"]["medoid"].append(float(gt[sel_by_agr[0]]))
            agg["A_within"]["sel"]["anchor_balanced"].append(float(gt[sel_by_agr[0]]))
            agg["A_within"]["sel"]["cross_anchor_support"].append(float(gt[sel_by_agr[0]]))

        for gname, idx in (("B_direct", idx_b), ("C_top1", idx_c), ("D_all", idx_d)):
            sub_r = [routes[i] for i in idx]
            sub_gt = np.array([gt[i] for i in idx])
            sub_C = C[np.ix_(idx, idx)]
            sub_masks = [masks[i] for i in idx]
            sub_anchors = [anchor_ids[i] for i in idx]
            sel = selectors_for(sub_C, sub_masks, sub_gt, sub_r, sub_anchors, tau)
            agg[gname]["mean"].append(float(sub_gt.mean()))
            agg[gname]["oracle"].append(float(sub_gt.max()))
            for s in ("medoid", "anchor_balanced", "cross_anchor_support"):
                agg[gname]["sel"][s].append(sel[s]["dice"])
            route_pool[gname]["Q"] += sel["Q_med"]
            route_pool[gname]["gt"] += sub_gt.tolist()
            # pair overlap stats within group
            ops = [ov[i, j] for k, i in enumerate(idx) for j in idx[k + 1:]]
            route_pool[gname]["overlap"] += ops
            diffs = [abs(gt[i] - gt[j]) for k, i in enumerate(idx) for j in idx[k + 1:]]
            if len(ops) >= 3:
                agg[gname]["rho_pair_dice_diff"].append(spearman(ops, diffs))

    out = {}
    for gname in ("B_direct", "C_top1", "D_all", "A_within"):
        a = agg[gname]
        n = len(a["mean"])
        out[gname] = {
            "n_targets": n,
            "per_target_mean": float(np.mean(a["mean"])),
            "oracle": float(np.mean(a["oracle"])),
            **{f"sel_{s}": float(np.mean(a["sel"][s])) for s in ("medoid", "anchor_balanced", "cross_anchor_support")},
            **{f"regret_{s}": float(np.mean(a["oracle"]) - float(np.mean(a["sel"][s]))) for s in ("medoid", "anchor_balanced", "cross_anchor_support")},
            "gain_medoid_vs_mean": float(np.mean(a["sel"]["medoid"]) - np.mean(a["mean"])),
        }
        if gname in route_pool:
            out[gname]["route_level_rho_Q_gt"] = float(spearman(route_pool[gname]["Q"], route_pool[gname]["gt"]))
            out[gname]["mean_pair_overlap"] = float(np.mean(route_pool[gname]["overlap"])) if route_pool[gname]["overlap"] else float("nan")
            out[gname]["rho_overlap_vs_absdice_diff"] = float(np.mean(a["rho_pair_dice_diff"])) if a["rho_pair_dice_diff"] else float("nan")
    out["config"] = {"tau": tau, "eval_dir": str(eval_dir)}
    return out

if __name__ == "__main__":
    ed = Path(sys.argv[1])
    tau = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85
    res = analyze(ed, tau)
    print(json.dumps(res, indent=2))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(res, indent=2))
