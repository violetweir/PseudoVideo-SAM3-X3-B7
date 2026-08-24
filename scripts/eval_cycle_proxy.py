#!/usr/bin/env python3
"""Cycle-consistency proxy on validation b1 routes (analysis only).

For each validation route with bridge_count=1 (anchor -> bridge -> target):
  forward  = propagate([anchor, bridge, target])   -> M_T
  backward = propagate([target, bridge, anchor])   -> M_A'
  cycle    = Dice(M_A', anchor_mask)               [mask stability through round trip]

Output JSON: {route_id: cycle} for the b1 routes, for analyze_transport_proxies.py --cycle-json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
T21_PATH = ROOT / "scripts/run_t21_dynamic_pseudovideo.py"
spec = importlib.util.spec_from_file_location("t21_dynamic", T21_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {T21_PATH}")
t21 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t21
spec.loader.exec_module(t21)

SAM3_CKPT = "/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt"


def load_mask(path: str, size=256) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.NEAREST)) > 127


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    return float(2 * inter / (a.sum() + b.sum())) if (a.sum() + b.sum()) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_lesion_knn")
    parser.add_argument("--mode", default="sam3enc_lesion")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--canvas", type=int, default=256)
    parser.add_argument("--output", type=Path, default=ROOT / "work/kvasir_1pct_anchors/e5_cycle_proxy.json")
    args = parser.parse_args()

    routes_path = args.root / args.mode / f"{args.split}_pool0_stage1" / "routes.jsonl"
    routes = [json.loads(l) for l in routes_path.read_text().splitlines() if l.strip()]
    b1 = [r for r in routes if r["bridge_count"] == 1]
    print(f"b1 routes: {len(b1)}")

    from sam3.model_builder import build_sam3_video_model
    model = build_sam3_video_model(checkpoint_path=SAM3_CKPT, load_from_HF=False, device="cuda", compile=False)
    model.eval()

    out: dict[str, float] = {}
    for n, route in enumerate(b1, 1):
        a = route["anchor_image_path"]
        b = route["bridge_image_paths"][0]
        t = route["target_image_path"]
        box = route["anchor_box_xywh_normalized"]
        # backward: T -> B -> A, then compare A-roundtrip mask with anchor GT mask
        back = t21.propagate(model, [t, b, a], box, args.canvas)
        anchor_gt = load_mask(route["anchor_mask_path"], args.canvas)
        cyc = dice(back["mask"], anchor_gt)
        out[route["route_id"]] = cyc
        print(f"[{n}/{len(b1)}] cycle={cyc:.3f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out) + "\n")
    print(f"written: {args.output} ({len(out)} routes)")


if __name__ == "__main__":
    main()
