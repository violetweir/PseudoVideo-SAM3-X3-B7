#!/usr/bin/env python3
"""Transport-aware proxies vs route GT Dice (validation, no training).

Proxies (all computed WITHOUT GT):
  q_model      = forward_sam_score        (SAM3 propagation confidence)
  q_multi      = forward_candidate_count  (multi-candidate disagreement proxy)
  path_bottleneck = path_bottleneck_similarity (min edge similarity along path)
  path_mean    = path_mean_similarity
  cycle        = optional: round-trip mask stability (separate script)

Reports per-proxy Spearman rho vs gt_dice, plus ranking ability:
  per-target best-route Dice selected by the proxy ("proxy-oracle") vs
  the true per-target best ("GT-oracle") and the per-target mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_lesion_knn")
    parser.add_argument("--mode", default="sam3enc_lesion")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--eval-name", default="eval_base_no_ft_b7_forward")
    parser.add_argument("--cycle-json", type=Path, default=None, help="Optional cycle proxy per route_id.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.split == "test":
        p = args.root / args.mode / args.eval_name / "route_results.jsonl"
    else:
        p = args.root / args.mode / f"{args.eval_name}_{args.split}" / "route_results.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    print(f"routes: {len(rows)} ({args.split})")

    cycle_map = {}
    if args.cycle_json and args.cycle_json.exists():
        cycle_map = json.loads(args.cycle_json.read_text())

    proxies = {
        "q_model": [float(r.get("forward_sam_score", float("nan"))) for r in rows],
        "q_multi": [float(r.get("forward_candidate_count", float("nan"))) for r in rows],
        "path_bottleneck": [float(r.get("path_bottleneck_similarity", float("nan"))) for r in rows],
        "path_mean": [float(r.get("path_mean_similarity", float("nan"))) for r in rows],
    }
    if cycle_map:
        proxies["cycle"] = [float(cycle_map.get(r["route_id"], float("nan"))) for r in rows]
    q = np.array([float(r["gt_dice_evaluation_only"]) for r in rows])

    print(f"\n{'proxy':>16s} {'rho':>7s} {'p':>8s} {'proxy-oracle':>13s} {'GT-oracle':>10s} {'oracle gap':>11s} {'proxy>=avg':>11s}")
    # per-target structure
    by_t: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_t.setdefault(r["target_id"], []).append(i)
    gt_oracle = np.mean([max(q[idx]) for idx in by_t.values()])
    avg = np.mean(q)
    results = {}
    for name, vals in proxies.items():
        v = np.array(vals, dtype=np.float64)
        ok = ~(np.isnan(v) | np.isnan(q))
        rho, pv = spearmanr(v[ok], q[ok])
        # proxy-oracle: per target pick the route with max proxy
        picks = []
        for idx in by_t.values():
            sub = [(v[i], q[i]) for i in idx if not np.isnan(v[i])]
            if sub:
                picks.append(max(sub, key=lambda t: t[0])[1])
        proxy_oracle = np.mean(picks) if picks else float("nan")
        results[name] = {"rho": float(rho), "p": float(pv), "proxy_oracle": float(proxy_oracle)}
        print(f"{name:>16s} {rho:>7.3f} {pv:>8.4f} {proxy_oracle:>13.4f} {gt_oracle:>10.4f} {gt_oracle-proxy_oracle:>11.4f} {proxy_oracle-avg:>+11.4f}")
    print(f"\n参考: per-target 平均 Dice={avg:.4f}, GT-oracle(每target 真最优)={gt_oracle:.4f}")

    if args.output:
        out = {"n": len(rows), "per_target_mean": float(avg), "gt_oracle": float(gt_oracle),
               "proxies": results}
        args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"written: {args.output}")


if __name__ == "__main__":
    main()
