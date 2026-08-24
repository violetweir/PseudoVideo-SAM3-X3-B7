#!/usr/bin/env python3
"""Freeze C3 selector rules on validation, then apply once to test.
Usage:
  python scripts/analyze_c3_freeze.py <val_eval_dir> <rules_out.json> [--taus 0.80,0.85,0.90]
  python scripts/analyze_c3_freeze.py <val_eval_dir> <rules_out.json> --apply-test <test_eval_dir> <test_report_out.json>
Rule selection: per group (B_direct/C_top1/D_all) choose selector with lowest validation
regret, tie-break by (cross_anchor_support > anchor_balanced > medoid) then larger tau.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from analyze_c3 import analyze

SELECTOR_PREF = {"cross_anchor_support": 0, "anchor_balanced": 1, "medoid": 2}
GROUPS = ("B_direct", "C_top1", "D_all")

def pick_rule(val_result: dict) -> dict:
    rules = {}
    for g in GROUPS:
        best = None
        for sel, pref in SELECTOR_PREF.items():
            key = f"regret_{sel}"
            regret = val_result[g].get(key)
            if regret is None:
                continue
            if best is None or regret < best[1] or (abs(regret - best[1]) < 1e-9 and pref < best[2]):
                best = (sel, float(regret), pref)
        rules[g] = {"selector": best[0], "regret_val": best[1]} if best else None
    return {"rules": rules, "tau": val_result.get("config", {}).get("tau")}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("val_eval_dir")
    ap.add_argument("rules_out")
    ap.add_argument("--taus", default="0.80,0.85,0.90")
    ap.add_argument("--apply-test", nargs=2, metavar=("TEST_EVAL_DIR", "TEST_REPORT_OUT"))
    args = ap.parse_args()

    taus = [float(t) for t in args.taus.split(",")]
    best_val, best_rules = None, None
    for tau in taus:
        r = analyze(Path(args.val_eval_dir), tau)
        rules = pick_rule(r)
        # score: total regret across groups (lower better)
        score = sum(rules["rules"][g]["regret_val"] for g in GROUPS if rules["rules"][g])
        print(f"tau={tau:.2f} total_regret={score:.4f}")
        if best_val is None or score < best_val:
            best_val, best_rules = score, (r, rules)
    val_result, rules = best_rules
    Path(args.rules_out).write_text(json.dumps(rules, indent=2) + "\n")
    print(json.dumps({"val_regret_total": best_val, "rules": rules["rules"]}, indent=2))

    if args.apply_test:
        test_dir, report_out = args.apply_test
        r = analyze(Path(test_dir), rules["tau"])
        rep = {"frozen_rules": rules, "test": r}
        Path(report_out).write_text(json.dumps(rep, indent=2) + "\n")
        print("===== TEST (frozen rules, one-shot) =====")
        print(json.dumps(r, indent=2))

if __name__ == "__main__":
    main()
