#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import v2 as transforms


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
T21_PATH = ROOT / "scripts/run_t21_dynamic_pseudovideo.py"
spec = importlib.util.spec_from_file_location("t21_dynamic", T21_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {T21_PATH}")
t21 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t21
spec.loader.exec_module(t21)


MODES = (
    "t18_corrected",
    "dino_global_pooling",
    "dino_patch_average",
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
)
ROUTE_TYPES = {0: "direct", **{idx: f"bridge_{idx}" for idx in range(1, 8)}}


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_rgb_tensor(path: str, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(TF.resize(image, [size, size], interpolation=TF.InterpolationMode.BICUBIC, antialias=True))
    return tensor


def load_mask_grid(path: str, grid: int) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((grid, grid), Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def extract_dinov3_features(records: list[dict[str, Any]], support: list[dict[str, Any]], output: Path) -> None:
    feature_path = output / "dinov3_vits16_features.npz"
    if feature_path.exists():
        return
    repo = "/Data_8TB/lht/MK-UNet/dinov3"
    weights = "/Data_8TB/lht/MK-UNet/teacher/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    sys.path.insert(0, repo)
    model = torch.hub.load(repo, "dinov3_vits16", source="local", weights=weights).cuda().eval()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    cls_rows, patch_rows = [], []
    with torch.no_grad():
        for start in range(0, len(records), 32):
            batch = torch.stack([normalize(load_rgb_tensor(row["image_path"], 224)) for row in records[start:start + 32]]).cuda()
            out = model(batch, is_training=True, masks=None)
            cls_rows.append(out["x_norm_clstoken"].float().cpu().numpy())
            patch_rows.append(out["x_norm_patchtokens"].float().cpu().numpy())
            print(f"features {min(start + 32, len(records))}/{len(records)}", flush=True)
    cls = l2norm(np.concatenate(cls_rows, axis=0).astype(np.float32))
    patches = l2norm(np.concatenate(patch_rows, axis=0).astype(np.float32))
    patch_mean = l2norm(patches.mean(axis=1).astype(np.float32))
    id_to_idx = {row["merged_id"]: i for i, row in enumerate(records)}
    anchor_ids = [row["merged_id"] for row in support]
    anchor_prototypes = []
    for anchor in support:
        idx = id_to_idx[anchor["merged_id"]]
        fg = load_mask_grid(anchor["frozen_mask_path"], 14).reshape(-1)
        tokens = patches[idx][fg]
        if len(tokens) == 0:
            tokens = patches[idx]
        anchor_prototypes.append(l2norm(tokens.mean(axis=0, keepdims=True))[0])
    np.savez_compressed(
        feature_path,
        cls=cls,
        patches=patches,
        patch_mean=patch_mean,
        anchor_ids=np.asarray(anchor_ids),
        anchor_prototypes=np.asarray(anchor_prototypes, dtype=np.float32),
    )


def t18_descriptors(records: list[dict[str, Any]], output: Path) -> np.ndarray:
    path = output / "t18_corrected_descriptors.npz"
    if path.exists():
        return np.load(path)["descriptors"]
    rows = np.stack([t21.descriptor(row["image_path"]) for row in records]).astype(np.float32)
    rows = l2norm(rows)
    np.savez_compressed(path, descriptors=rows)
    return rows


def build_mode_state(mode: str, records: list[dict[str, Any]], support: list[dict[str, Any]], feature_root: Path) -> dict[str, Any]:
    if mode == "t18_corrected":
        desc = t18_descriptors(records, feature_root)
        return {"mode": mode, "desc": desc, "sim": desc @ desc.T}
    extract_dinov3_features(records, support, feature_root)
    data = np.load(feature_root / "dinov3_vits16_features.npz")
    if mode == "dino_global_pooling":
        desc = data["cls"]
        return {"mode": mode, "desc": desc, "sim": desc @ desc.T}
    if mode == "dino_patch_average":
        desc = data["patch_mean"]
        return {"mode": mode, "desc": desc, "sim": desc @ desc.T}
    anchor_ids = data["anchor_ids"].tolist()
    anchor_proto = data["anchor_prototypes"]
    patches = data["patches"]
    patch_mean = data["patch_mean"]
    id_to_anchor = {anchor_id: i for i, anchor_id in enumerate(anchor_ids)}
    patch_sims = np.einsum("ad,npd->anp", anchor_proto, patches).astype(np.float32)
    if mode == "anchor_conditioned_patch_correspondence":
        cond_scores = np.sort(patch_sims, axis=2)[:, :, -8:].mean(axis=2)
    else:
        weights = np.exp((patch_sims - patch_sims.max(axis=2, keepdims=True)) * 10.0)
        weights = weights / np.maximum(weights.sum(axis=2, keepdims=True), 1e-12)
        pooled = np.einsum("anp,npd->and", weights, patches).astype(np.float32)
        pooled = l2norm(pooled, axis=2)
        cond_scores = np.einsum("and,ad->an", pooled, anchor_proto).astype(np.float32)
    return {
        "mode": mode,
        "patch_mean": patch_mean,
        "sim": patch_mean @ patch_mean.T,
        "cond_scores": cond_scores,
        "id_to_anchor": id_to_anchor,
    }


def conditioned_score(state: dict[str, Any], anchor_id: str, node_idx: int) -> float:
    return float(state["cond_scores"][state["id_to_anchor"][anchor_id], node_idx])


def route_score(state: dict[str, Any], anchor_id: str, path: list[int], target_idx: int) -> tuple[float, float]:
    nodes = [*path, target_idx]
    if state["mode"].startswith("anchor_conditioned"):
        cond = [conditioned_score(state, anchor_id, node) for node in nodes]
        transitions = []
        prev = None
        for node in nodes:
            if prev is not None:
                transitions.append(float(state["sim"][prev, node]))
            prev = node
        values = cond + transitions
    else:
        anchor_idx = state["id_to_index"][anchor_id]
        nodes2 = [anchor_idx, *path, target_idx]
        values = [float(state["sim"][nodes2[i], nodes2[i + 1]]) for i in range(len(nodes2) - 1)]
    return min(values), float(np.mean(values))


def build_rank_cache(
    state: dict[str, Any],
    records: list[dict[str, Any]],
    support: list[dict[str, Any]],
    train_indices: list[int],
) -> dict[tuple[str, int], list[int]]:
    cache: dict[tuple[str, int], list[int]] = {}
    if state["mode"].startswith("anchor_conditioned"):
        for anchor in t21.human_pool(support, 512):
            anchor_id = anchor["anchor_id"]
            cond = np.asarray([conditioned_score(state, anchor_id, i) for i in train_indices])
            ids = np.asarray([records[i]["merged_id"] for i in train_indices])
            for tail in range(len(records)):
                sims = np.asarray([float(state["sim"][i, tail]) for i in train_indices])
                order = np.lexsort((ids, cond, sims))[::-1]
                cache[(anchor_id, tail)] = [train_indices[int(pos)] for pos in order]
    else:
        ids = np.asarray([records[i]["merged_id"] for i in train_indices])
        for tail in range(len(records)):
            sims = np.asarray([float(state["sim"][i, tail]) for i in train_indices])
            order = np.lexsort((ids, sims))[::-1]
            ranked = [train_indices[int(pos)] for pos in order]
            cache[("", tail)] = ranked
    return cache


def top_ranked_nodes(
    rank_cache: dict[tuple[str, int], list[int]],
    state: dict[str, Any],
    anchor_id: str,
    tail: int,
    used: set[int],
    beam_width: int,
) -> list[int]:
    key = (anchor_id, tail) if state["mode"].startswith("anchor_conditioned") else ("", tail)
    out = []
    for node in rank_cache[key]:
        if node not in used:
            out.append(node)
            if len(out) >= beam_width:
                break
    return out


def make_route(
    target: dict[str, Any],
    bridge_count: int,
    anchor: dict[str, Any],
    bridge_indices: list[int],
    score: tuple[float, float],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    route_key = (
        f"{target['merged_id']}::{ROUTE_TYPES[bridge_count]}::{anchor['anchor_id']}::"
        f"{anchor['mask_sha256']}::" + "::".join(records[index]["merged_id"] for index in bridge_indices)
    )
    return {
        "route_id": hashlib.sha256(route_key.encode()).hexdigest()[:24],
        "route_type": ROUTE_TYPES[bridge_count],
        "bridge_count": bridge_count,
        "target_id": target["merged_id"],
        "target_split": target["split"],
        "target_source_dataset": target["source_dataset"],
        "target_image_path": target["image_path"],
        "target_mask_path_evaluation_only": target["mask_path"],
        "anchor_id": anchor["anchor_id"],
        "anchor_generation": anchor["generation"],
        "anchor_is_human": anchor["is_human"],
        "anchor_image_path": anchor["image_path"],
        "anchor_mask_path": anchor["mask_path"],
        "anchor_mask_sha256": anchor["mask_sha256"],
        "anchor_box_xywh_normalized": anchor["box_xywh_normalized"],
        "bridge_ids": [records[index]["merged_id"] for index in bridge_indices],
        "bridge_image_paths": [records[index]["image_path"] for index in bridge_indices],
        "path_bottleneck_similarity": score[0],
        "path_mean_similarity": score[1],
        "target_gt_used_for_search_or_inference": False,
    }


def freeze_routes(
    mode: str,
    output_root: Path,
    records: list[dict[str, Any]],
    support: list[dict[str, Any]],
    state: dict[str, Any],
    split: str,
    max_bridge: int,
    beam_width: int,
) -> list[dict[str, Any]]:
    phase = output_root / mode / f"{split}_pool0_stage1"
    routes_path = phase / "routes.jsonl"
    if routes_path.exists():
        return read_jsonl(routes_path)
    phase.mkdir(parents=True, exist_ok=True)
    id_to_index = {row["merged_id"]: i for i, row in enumerate(records)}
    state["id_to_index"] = id_to_index
    train_indices = [i for i, row in enumerate(records) if row["split"] == "train"]
    anchors = t21.human_pool(support, 512)
    targets = [row for row in records if row["split"] == split]
    rank_cache = build_rank_cache(state, records, support, train_indices)
    routes = []
    for target_no, target in enumerate(sorted(targets, key=lambda row: row["merged_id"]), start=1):
        if target_no == 1 or target_no % 10 == 0:
            print(f"{mode}: target {target_no}/{len(targets)}", flush=True)
        target_idx = id_to_index[target["merged_id"]]
        for bridge_count in range(max_bridge + 1):
            candidates = []
            for anchor in anchors:
                forbidden = {target_idx, id_to_index[anchor["anchor_id"]]}
                beams: list[tuple[list[int], tuple[float, float]]] = [([], route_score(state, anchor["anchor_id"], [], target_idx))]
                for depth in range(bridge_count):
                    expanded = []
                    for path, _ in beams:
                        used = forbidden | set(path)
                        tail = target_idx if not path else path[0]
                        ranked = top_ranked_nodes(
                            rank_cache,
                            state,
                            anchor["anchor_id"],
                            tail,
                            used,
                            beam_width,
                        )
                        for node in ranked:
                            new_path = [node, *path]
                            expanded.append((new_path, route_score(state, anchor["anchor_id"], new_path, target_idx)))
                    beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:beam_width]
                for path, score in beams:
                    if len(path) == bridge_count:
                        candidates.append((score, anchor["anchor_id"], anchor, path))
            score, _, anchor, path = max(candidates, key=lambda item: (item[0], item[1]))
            routes.append(make_route(target, bridge_count, anchor, path, score, records))
    write_jsonl(routes_path, routes)
    return routes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-bridge", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn")
    args = parser.parse_args()
    protocol = ROOT / "work/kvasir_1pct_anchors/protocol"
    records = read_jsonl(protocol / "merged_manifest.jsonl")
    support = read_jsonl(protocol / "support_manifest.jsonl")
    feature_root = args.output_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    state = build_mode_state(args.mode, records, support, feature_root)
    routes = freeze_routes(
        args.mode,
        args.output_root,
        records,
        support,
        state,
        args.split,
        args.max_bridge,
        args.beam_width,
    )
    print(json.dumps({"mode": args.mode, "split": args.split, "routes": len(routes), "max_bridge": args.max_bridge}, indent=2))


if __name__ == "__main__":
    main()
