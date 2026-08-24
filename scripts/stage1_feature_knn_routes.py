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
    "sam3enc_patch_average",
    "sam3enc_lesion",
    "sam3enc_global_lesion",
    "sam3enc_lesion_anchor_global_bridge",
    "sam3enc_global_anchor_lesion_bridge",
    "sam3enc_pyramid",
    "sam3enc_contrast",
    "sam3enc_pyramid_contrast",
    "sam3enc_imr",
    "sam3enc_mask_visual",
    "sam3enc_original_mask_joint",
    "sam3enc_anchor_conditioned_target_pooling",
    "sam3enc_anchor_conditioned_patch_correspondence",
    "sam3enc_fpn_foreground_transport",
)
ROUTE_TYPES = {0: "direct", **{idx: f"bridge_{idx}" for idx in range(1, 8)}}
DINOV3_MODELS = {
    "vits16": {
        "hub_fn": "dinov3_vits16",
        "weights": "/Data_8TB/lht/MK-UNet/teacher/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "feature_name": "dinov3_vits16_features.npz",
    },
    "vitb16": {
        "hub_fn": "dinov3_vitb16",
        "weights": "/Data_8TB/lht/MK-UNet/teacher/dinov3_vitb16_pretrain_lvd1689m.pth",
        "feature_name": "dinov3_vitb16_features.npz",
    },
}
KNN_FEATURES = ("patch_mean", "cls", "pooled", "cond")
ANCHOR_MODES = (
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
    "sam3enc_anchor_conditioned_target_pooling",
    "sam3enc_anchor_conditioned_patch_correspondence",
    "sam3enc_imr",
    "sam3enc_fpn_foreground_transport",
)
# IMR 描述子 modes (SAM3 encoder @feature_size, mask-free visual descriptors)
IMR_VISUAL_MODES = ("sam3enc_pyramid", "sam3enc_contrast", "sam3enc_pyramid_contrast")
FEATURE_SOURCES = ("dinov3_vits16", "dinov3_vitb16", "sam3_base")
SAM3_CKPT = "/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt"
# Lesion (mask-foreground) descriptors from round-1 pseudo masks, @1008.
# Reused by sam3enc_lesion / sam3enc_global_lesion modes (anchor-independent).
MASK_DESC_NPZ = (
    ROOT
    / "work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/sam3enc_mask_descriptors_s1008.npz"
)


def load_lesion_descriptors(
    records: list[dict[str, Any]], path: Path = MASK_DESC_NPZ
) -> np.ndarray:
    """F_lesion: mean of SAM3-trunk patch tokens inside the round-1 pseudo mask,
    L2-normalized, aligned to records order (1000 rows)."""
    if not path.exists():
        raise SystemExit(f"Missing lesion descriptors: {path} (run extract_sam3_mask_descriptors.py)")
    data = np.load(path)
    by_id = {str(record_id): pos for pos, record_id in enumerate(data["ids"].tolist())}
    rows = []
    for record in records:
        pos = by_id.get(record["merged_id"])
        if pos is None:
            raise SystemExit(f"Missing lesion descriptor for {record['merged_id']}")
        rows.append(data["descriptors"][pos])
    return l2norm(np.asarray(rows, dtype=np.float32))


def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def qwen_text_vector(features: dict[str, Any] | None) -> np.ndarray:
    fields = {
        "shape": ["round", "oval", "irregular", "elongated"],
        "size": ["small", "medium", "large"],
        "position": ["left", "center", "right", "top", "bottom", "diffuse"],
        "boundary": ["smooth", "irregular"],
        "components": ["single", "multiple"],
        "spread": ["local", "multi_region"],
    }
    out: list[float] = []
    for field, allowed in fields.items():
        value = ((features or {}).get(field) or "").lower()
        out.extend([1.0 if value == candidate else 0.0 for candidate in allowed])
    vec = np.asarray(out, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def load_qwen_text_sim(
    records: list[dict[str, Any]],
    qwen_desc_root: Path,
) -> np.ndarray:
    """Build text-feature similarity for all records in manifest order."""
    desc_rows: dict[str, dict[str, Any]] = {}
    desc_roots = [
        qwen_desc_root,
        qwen_desc_root.parent / "validation_pseudo_masks_round1",
        qwen_desc_root.parent / "test_pseudo_masks_round1",
    ]
    for desc_root in desc_roots:
        for path in desc_root.glob("*mask_descriptions.jsonl"):
            for row in read_jsonl(path):
                desc_rows[row["target_id"]] = row.get("qwen_features") or {}
    anchor_path = qwen_desc_root / "qwen35_anchor_descriptions.jsonl"
    if anchor_path.exists():
        for row in read_jsonl(anchor_path):
            desc_rows[row["target_id"]] = row.get("qwen_features") or {}
    matrix = np.stack([qwen_text_vector(desc_rows.get(row["merged_id"])) for row in records])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return matrix @ matrix.T


def qwen_image_text_vector(features: dict[str, Any] | None) -> np.ndarray:
    """IMR text descriptor: extended categorical fields from Qwen describing the ORIGINAL image.

    Adds texture / context on top of the legacy mask fields, so the vector is
    denser (8 fields vs 6) and carries surrounding-tissue information.
    """
    fields = {
        "shape": ["round", "oval", "irregular", "elongated"],
        "size": ["small", "medium", "large"],
        "position": ["left", "center", "right", "top", "bottom", "diffuse"],
        "boundary": ["smooth", "irregular"],
        "components": ["single", "multiple"],
        "spread": ["local", "multi_region"],
        "texture": ["smooth", "lobulated", "nodular", "pedunculated"],
        "context": ["normal", "erythema", "hemorrhage", "necrotic"],
    }
    out: list[float] = []
    for field, allowed in fields.items():
        value = ((features or {}).get(field) or "").lower()
        out.extend([1.0 if value == candidate else 0.0 for candidate in allowed])
    vec = np.asarray(out, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def load_qwen_image_text_sim(
    records: list[dict[str, Any]],
    qwen_desc_root: Path,
) -> np.ndarray:
    """Build image-text similarity (Qwen describes the original image) for all records."""
    desc_rows: dict[str, dict[str, Any]] = {}
    for name in ("qwen35_image_descriptions.jsonl", "qwen35_image_descriptions_overlay.jsonl"):
        path = qwen_desc_root / name
        if not path.exists():
            continue
        for row in read_jsonl(path):
            desc_rows[row["target_id"]] = row.get("qwen_features") or {}
    matrix = np.stack([qwen_image_text_vector(desc_rows.get(row["merged_id"])) for row in records])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return matrix @ matrix.T


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Deterministic ascending ranks in [0, 1] (for cross-source rank fusion)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    if len(values) > 1:
        ranks /= float(len(values) - 1)
    return ranks


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


def extract_dinov3_features(
    records: list[dict[str, Any]],
    support: list[dict[str, Any]],
    output: Path,
    dinov3_model: str = "vits16",
    feature_size: int = 256,
) -> None:
    cfg = DINOV3_MODELS[dinov3_model]
    feature_path = output / f"dinov3_{dinov3_model}_s{feature_size}_features.npz"
    if feature_path.exists():
        return
    repo = "/Data_8TB/lht/MK-UNet/dinov3"
    sys.path.insert(0, repo)
    model = torch.hub.load(repo, cfg["hub_fn"], source="local", weights=cfg["weights"]).cuda().eval()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    cls_rows, patch_rows = [], []
    with torch.no_grad():
        for start in range(0, len(records), 16):
            batch = torch.stack([normalize(load_rgb_tensor(row["image_path"], feature_size)) for row in records[start:start + 16]]).cuda()
            out = model(batch, is_training=True, masks=None)
            cls_rows.append(out["x_norm_clstoken"].float().cpu().numpy())
            patch_rows.append(out["x_norm_patchtokens"].float().cpu().numpy())
            print(f"features {min(start + 16, len(records))}/{len(records)}", flush=True)
    cls = l2norm(np.concatenate(cls_rows, axis=0).astype(np.float32))
    patches = l2norm(np.concatenate(patch_rows, axis=0).astype(np.float32))
    patch_mean = l2norm(patches.mean(axis=1).astype(np.float32))
    id_to_idx = {row["merged_id"]: i for i, row in enumerate(records)}
    anchor_ids = [row["merged_id"] for row in support]
    anchor_prototypes = []
    for anchor in support:
        idx = id_to_idx[anchor["merged_id"]]
        grid = feature_size // 16
        fg = load_mask_grid(anchor["frozen_mask_path"], grid).reshape(-1)
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
        dinov3_model=dinov3_model,
        feature_size=feature_size,
    )


def prepare_sam3_trunk(trunk: Any, feature_size: int) -> None:
    """Recompute RoPE buffers for non-native trunk input sizes (native grid = 1008//14)."""
    if feature_size == 1008:
        return
    grid = feature_size // 14
    for blk in trunk.blocks:
        attn = blk.attn
        if not (attn.use_rope and attn.freqs_cis is not None):
            continue
        if blk.window_size > 0:
            # windowed attention stays at the fixed 24x24 window
            continue
        pt = tuple(attn.rope_pt_size)
        scale = 1.0
        if attn.rope_interp:
            scale = pt[0] / float(grid)
        freqs = attn.compute_cis(end_x=grid, end_y=grid, scale_pos=scale)
        freqs = freqs.to(attn.freqs_cis.device)
        attn.register_buffer("freqs_cis", freqs)


def pyramid_pool(tokens_b: torch.Tensor, g: int) -> torch.Tensor:
    """Spatial pyramid pooling over the (B, G, G, C) grid: 2x2 + 3x3 region means."""
    t = tokens_b.view(tokens_b.shape[0], g, g, tokens_b.shape[-1])
    parts = []
    for k in (2, 3):
        s = g // k
        regions = (
            t.view(t.shape[0], k, s, k, s, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(t.shape[0], k * k, s * s, -1)
        )
        parts.append(regions.mean(dim=2))  # (B, k*k, C)
    return torch.nn.functional.normalize(
        torch.cat([p.reshape(p.shape[0], -1) for p in parts], dim=1), dim=-1
    )


def contrast_pool(tokens_b: torch.Tensor) -> torch.Tensor:
    """Variance-weighted pooling: high-texture tokens (edges, folds) get more weight."""
    w = tokens_b.std(dim=-1, keepdim=True) + 1.0  # (B, P, 1)
    pooled = (tokens_b * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-12)
    return torch.nn.functional.normalize(pooled, dim=-1)


def extract_sam3_encoder_features(
    records: list[dict[str, Any]],
    support: list[dict[str, Any]],
    output: Path,
    feature_size: int = 1008,
    imr: bool = False,
) -> None:
    """Extract SAM3-trunk descriptors.

    imr=True additionally computes the mask-free IMR visual descriptors
    (spatial pyramid + contrast-weighted pooling) and writes them to a
    separate npz (sam3_base_s{size}_imr_features.npz) so the cached
    baseline file is never invalidated.
    """
    feature_path = output / (
        f"sam3_base_s{feature_size}_imr_features.npz" if imr else f"sam3_base_s{feature_size}_features.npz"
    )
    if feature_path.exists():
        return
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(
        checkpoint_path=SAM3_CKPT, load_from_HF=False, device="cuda", compile=False
    )
    model.eval()
    trunk = model.detector.backbone.vision_backbone.trunk
    prepare_sam3_trunk(trunk, feature_size)
    grid = feature_size // 14
    id_to_idx = {row["merged_id"]: i for i, row in enumerate(records)}
    anchor_ids = [row["merged_id"] for row in support]

    def preprocess(x: torch.Tensor) -> torch.Tensor:
        return (x - 0.5) / 0.5

    def trunk_tokens(x: torch.Tensor) -> torch.Tensor:
        feats = trunk(x)[0]  # (B, C, H, W)
        tokens = feats.flatten(2).permute(0, 2, 1)  # (B, P, C)
        return torch.nn.functional.normalize(tokens, dim=-1)

    # Pass 0: anchor prototypes from the 8 labeled images
    anchor_prototypes = []
    with torch.no_grad():
        for anchor in support:
            idx = id_to_idx[anchor["merged_id"]]
            x = preprocess(load_rgb_tensor(anchor["image_path"], feature_size)).unsqueeze(0).cuda()
            tokens = trunk_tokens(x)[0]
            fg = load_mask_grid(anchor["frozen_mask_path"], grid).reshape(-1)
            tok = tokens[fg]
            if len(tok) == 0:
                tok = tokens
            anchor_prototypes.append(l2norm(tok.mean(dim=0, keepdim=True).float().cpu().numpy())[0])
    proto_t = torch.from_numpy(np.asarray(anchor_prototypes, dtype=np.float32)).cuda()

    # Pass 1: all records -> patch_mean + anchor-conditioned pooled / cond scores (+ IMR descriptors)
    patch_mean_rows, pooled_rows, cond_t_rows, cond_c_rows = [], [], [], []
    pyramid_rows, contrast_rows = [], []
    batch = 1 if feature_size >= 1008 else 8
    with torch.no_grad():
        for start in range(0, len(records), batch):
            chunk = records[start : start + batch]
            x = torch.stack([preprocess(load_rgb_tensor(row["image_path"], feature_size)) for row in chunk]).cuda()
            tokens = trunk_tokens(x)  # (B, P, C)
            patch_mean_rows.append(torch.nn.functional.normalize(tokens.mean(dim=1), dim=-1).float().cpu().numpy())
            if imr:
                pyramid_rows.append(pyramid_pool(tokens, grid).float().cpu().numpy())
                contrast_rows.append(contrast_pool(tokens).float().cpu().numpy())
            sims = torch.einsum("ad,bpd->abp", proto_t, tokens)  # (A, B, P)
            weights = torch.exp((sims - sims.amax(dim=2, keepdim=True)) * 10.0)
            weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-12)
            pooled = torch.nn.functional.normalize(torch.einsum("abp,bpd->abd", weights, tokens), dim=-1)
            pooled_rows.append(pooled.float().cpu().numpy())
            cond_t_rows.append(torch.einsum("abd,ad->ab", pooled, proto_t).float().cpu().numpy())
            topk = torch.topk(sims, k=8, dim=2).values.mean(dim=2)
            cond_c_rows.append(topk.float().cpu().numpy())
            print(f"sam3 features {min(start + batch, len(records))}/{len(records)}", flush=True)

    payload: dict[str, Any] = {
        "patch_mean": np.concatenate(patch_mean_rows, axis=0).astype(np.float32),
        "anchor_ids": np.asarray(anchor_ids),
        "anchor_prototypes": np.asarray(anchor_prototypes, dtype=np.float32),
        "pooled": np.concatenate(pooled_rows, axis=1).astype(np.float32),
        "cond_target": np.concatenate(cond_t_rows, axis=1).astype(np.float32),
        "cond_correspondence": np.concatenate(cond_c_rows, axis=1).astype(np.float32),
        "feature_source": "sam3_base",
        "feature_size": feature_size,
    }
    if imr:
        payload["pyramid"] = np.concatenate(pyramid_rows, axis=0).astype(np.float32)
        payload["contrast"] = np.concatenate(contrast_rows, axis=0).astype(np.float32)
    np.savez_compressed(feature_path, **payload)


def extract_sam3_fpn_transport_features(
    records: list[dict[str, Any]],
    support: list[dict[str, Any]],
    output: Path,
    feature_size: int = 256,
) -> None:
    """Extract anchor-conditioned descriptors from SAM3's tracker FPN.

    Unlike the trunk baselines, this uses the exact three feature levels consumed
    by SAM3 propagation. Each anchor supplies foreground/background prototypes;
    their margin selects a soft foreground in every candidate image. The saved
    descriptor combines foreground appearance and foreground/background contrast.
    """
    feature_path = output / f"sam3_base_s{feature_size}_fpn_transport_features.npz"
    if feature_path.exists():
        return
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(
        checkpoint_path=SAM3_CKPT, load_from_HF=False, device="cuda", compile=False
    )
    model.eval()
    trunk = model.detector.backbone.vision_backbone.trunk
    prepare_sam3_trunk(trunk, feature_size)
    anchor_ids = [row["merged_id"] for row in support]

    def preprocess(x: torch.Tensor) -> torch.Tensor:
        return (x - 0.5) / 0.5

    def fpn_levels(x: torch.Tensor) -> list[torch.Tensor]:
        out = model.detector.backbone.forward_image(x)["sam2_backbone_out"]
        return [torch.nn.functional.normalize(feat, dim=1) for feat in out["backbone_fpn"]]

    fg_protos: list[list[torch.Tensor]] = [[] for _ in range(3)]
    bg_protos: list[list[torch.Tensor]] = [[] for _ in range(3)]
    anchor_desc_rows: list[np.ndarray] = []
    anchor_centers: list[list[float]] = []
    anchor_spreads: list[float] = []
    with torch.no_grad():
        for anchor in support:
            x = preprocess(load_rgb_tensor(anchor["image_path"], feature_size)).unsqueeze(0).cuda()
            levels = fpn_levels(x)
            desc_parts = []
            for level_no, feat in enumerate(levels):
                _, channels, height, width = feat.shape
                tokens = feat[0].flatten(1).T
                fg = torch.from_numpy(
                    load_mask_grid(anchor["frozen_mask_path"], height).reshape(-1)
                ).to(tokens.device)
                if not bool(fg.any()):
                    fg = torch.ones_like(fg)
                bg = ~fg
                fg_proto = torch.nn.functional.normalize(tokens[fg].mean(0), dim=0)
                bg_proto = torch.nn.functional.normalize(
                    tokens[bg].mean(0) if bool(bg.any()) else tokens.mean(0), dim=0
                )
                fg_protos[level_no].append(fg_proto)
                bg_protos[level_no].append(bg_proto)
                contrast = torch.nn.functional.normalize(fg_proto - bg_proto, dim=0)
                desc_parts.extend([fg_proto, contrast])
            anchor_desc_rows.append(
                torch.nn.functional.normalize(torch.cat(desc_parts), dim=0).float().cpu().numpy()
            )
            mask = load_mask_grid(anchor["frozen_mask_path"], 72)
            ys, xs = np.where(mask)
            if len(xs):
                cx, cy = float(xs.mean() / 71.0), float(ys.mean() / 71.0)
                spread = float(np.sqrt(xs.var() + ys.var()) / 71.0)
            else:
                cx, cy, spread = 0.5, 0.5, 0.5
            anchor_centers.append([cx, cy])
            anchor_spreads.append(max(spread, 1e-3))

    fg_proto_t = [torch.stack(rows) for rows in fg_protos]
    bg_proto_t = [torch.stack(rows) for rows in bg_protos]
    desc_rows, center_rows, spread_rows = [], [], []
    batch = 4
    with torch.no_grad():
        for start in range(0, len(records), batch):
            chunk = records[start : start + batch]
            x = torch.stack(
                [preprocess(load_rgb_tensor(row["image_path"], feature_size)) for row in chunk]
            ).cuda()
            levels = fpn_levels(x)
            level_descs = []
            finest_centers = None
            finest_spreads = None
            for level_no, feat in enumerate(levels):
                bsz, channels, height, width = feat.shape
                tokens = feat.flatten(2).permute(0, 2, 1)
                fg_sim = torch.einsum("ad,bpd->abp", fg_proto_t[level_no], tokens)
                bg_sim = torch.einsum("ad,bpd->abp", bg_proto_t[level_no], tokens)
                margin = fg_sim - bg_sim
                top_count = max(4, int(round(height * width * 0.125)))
                top_values, top_indices = torch.topk(margin, k=top_count, dim=2)
                expanded = tokens.unsqueeze(0).expand(len(support), -1, -1, -1)
                selected = torch.gather(
                    expanded,
                    2,
                    top_indices.unsqueeze(-1).expand(-1, -1, -1, channels),
                )
                weights = torch.softmax(top_values * 8.0, dim=2)
                fg_pool = torch.nn.functional.normalize(
                    (selected * weights.unsqueeze(-1)).sum(dim=2), dim=-1
                )

                low_values, low_indices = torch.topk(-margin, k=top_count, dim=2)
                selected_bg = torch.gather(
                    expanded,
                    2,
                    low_indices.unsqueeze(-1).expand(-1, -1, -1, channels),
                )
                bg_weights = torch.softmax(low_values * 4.0, dim=2)
                bg_pool = torch.nn.functional.normalize(
                    (selected_bg * bg_weights.unsqueeze(-1)).sum(dim=2), dim=-1
                )
                contrast = torch.nn.functional.normalize(fg_pool - bg_pool, dim=-1)
                level_descs.extend([fg_pool, contrast])

                if level_no == 0:
                    yy, xx = torch.meshgrid(
                        torch.linspace(0.0, 1.0, height, device=feat.device),
                        torch.linspace(0.0, 1.0, width, device=feat.device),
                        indexing="ij",
                    )
                    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
                    selected_coords = coords[top_indices]
                    centers = (selected_coords * weights.unsqueeze(-1)).sum(dim=2)
                    variance = (
                        (selected_coords - centers.unsqueeze(2)).square().sum(dim=-1) * weights
                    ).sum(dim=2)
                    finest_centers = centers
                    finest_spreads = variance.clamp_min(1e-6).sqrt()

            descriptor = torch.nn.functional.normalize(torch.cat(level_descs, dim=-1), dim=-1)
            desc_rows.append(descriptor.float().cpu().numpy())
            center_rows.append(finest_centers.float().cpu().numpy())
            spread_rows.append(finest_spreads.float().cpu().numpy())
            print(f"sam3 fpn transport {min(start + batch, len(records))}/{len(records)}", flush=True)

    np.savez_compressed(
        feature_path,
        descriptors=np.concatenate(desc_rows, axis=1).astype(np.float32),
        centers=np.concatenate(center_rows, axis=1).astype(np.float32),
        spreads=np.concatenate(spread_rows, axis=1).astype(np.float32),
        anchor_descriptors=np.asarray(anchor_desc_rows, dtype=np.float32),
        anchor_centers=np.asarray(anchor_centers, dtype=np.float32),
        anchor_spreads=np.asarray(anchor_spreads, dtype=np.float32),
        anchor_ids=np.asarray(anchor_ids),
        feature_source="sam3_base_fpn_transport",
        feature_size=feature_size,
    )


def extract_encoder_features(
    records: list[dict[str, Any]],
    support: list[dict[str, Any]],
    output: Path,
    feature_source: str,
    feature_size: int,
    imr: bool = False,
) -> None:
    if feature_source.startswith("dinov3_"):
        extract_dinov3_features(
            records, support, output, feature_source.split("_", 1)[1], feature_size
        )
    else:
        extract_sam3_encoder_features(records, support, output, feature_size, imr=imr)


def t18_descriptors(records: list[dict[str, Any]], output: Path) -> np.ndarray:
    path = output / "t18_corrected_descriptors.npz"
    if path.exists():
        return np.load(path)["descriptors"]
    rows = np.stack([t21.descriptor(row["image_path"]) for row in records]).astype(np.float32)
    rows = l2norm(rows)
    np.savez_compressed(path, descriptors=rows)
    return rows


def build_mode_state(
    mode: str,
    records: list[dict[str, Any]],
    support: list[dict[str, Any]],
    feature_root: Path,
    feature_source: str = "dinov3_vits16",
    knn_feature: str = "patch_mean",
    feature_size: int = 256,
    text_blend: float = 0.0,
    qwen_desc_root: Path = ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1",
    mask_visual_fraction: float = 0.0,
    lesion_desc_path: Path = MASK_DESC_NPZ,
) -> dict[str, Any]:
    if mode == "t18_corrected":
        desc = t18_descriptors(records, feature_root)
        return {"mode": mode, "desc": desc, "sim": desc @ desc.T}
    if mode.startswith("sam3enc"):
        if mode == "sam3enc_fpn_foreground_transport":
            extract_sam3_fpn_transport_features(records, support, feature_root, feature_size)
            data = np.load(feature_root / f"sam3_base_s{feature_size}_fpn_transport_features.npz")
            descriptors = data["descriptors"]  # (A, N, D)
            semantic = np.einsum("and,amd->anm", descriptors, descriptors).astype(np.float32)
            centers = data["centers"]
            spreads = data["spreads"]
            center_delta = centers[:, :, None, :] - centers[:, None, :, :]
            center_score = np.exp(-np.square(center_delta).sum(axis=-1) / (2.0 * 0.25**2))
            spread_delta = np.abs(
                np.log(np.maximum(spreads[:, :, None], 1e-4))
                - np.log(np.maximum(spreads[:, None, :], 1e-4))
            )
            geometry = center_score * np.exp(-spread_delta)
            edge_sim = (0.9 * semantic + 0.1 * geometry).astype(np.float32)
            cond_scores = np.einsum(
                "ad,and->an", data["anchor_descriptors"], descriptors
            ).astype(np.float32)
            return {
                "mode": mode,
                "sim": semantic.mean(axis=0),
                "knn_sim": edge_sim,
                "edge_sim": edge_sim,
                "knn_feature": "pooled",
                "cond_scores": cond_scores,
                "id_to_anchor": {
                    anchor_id: i for i, anchor_id in enumerate(data["anchor_ids"].tolist())
                },
                "text_blend": 0.0,
            }
        if mode in ("sam3enc_mask_visual", "sam3enc_original_mask_joint"):
            mask_path = feature_root / "sam3enc_mask_descriptors_s1008.npz"
            if not mask_path.exists():
                raise SystemExit(f"Missing mask visual descriptors: {mask_path}")
            orig_path = feature_root / "sam3_base_s1008_features.npz"
            if not orig_path.exists():
                raise SystemExit(f"Missing original visual descriptors: {orig_path}")
            mask_data = np.load(mask_path)
            orig_data = np.load(orig_path)
            id_order = mask_data["ids"].tolist()
            id_to_idx = {str(i): pos for pos, i in enumerate(id_order)}
            mask_rows = []
            orig_rows = []
            orig_patch_mean = orig_data["patch_mean"] if "patch_mean" in orig_data else None
            if orig_patch_mean is None:
                raise SystemExit("Original SAM3 feature npz has no patch_mean")
            for record in records:
                pos = id_to_idx.get(record["merged_id"])
                if pos is None:
                    raise SystemExit(f"Missing mask descriptor for {record['merged_id']}")
                mask_rows.append(mask_data["descriptors"][pos])
                orig_rows.append(orig_patch_mean[len(orig_rows)])
            mask_desc = l2norm(np.asarray(mask_rows, dtype=np.float32))
            orig_desc = l2norm(np.asarray(orig_rows, dtype=np.float32))
            mask_sim = mask_desc @ mask_desc.T
            orig_sim = orig_desc @ orig_desc.T
            frac = float(np.clip(mask_visual_fraction, 0.0, 1.0))
            if mode == "sam3enc_original_mask_joint":
                desc = l2norm(np.concatenate([orig_desc, mask_desc], axis=1).astype(np.float32))
                state = {"mode": mode, "desc": desc, "sim": (desc @ desc.T).astype(np.float32)}
            else:
                visual_sim = (1.0 - frac) * orig_sim + frac * mask_sim
                state = {"mode": mode, "desc": mask_desc, "sim": visual_sim.astype(np.float32)}
            if text_blend > 0.0:
                state["text_sim"] = load_qwen_text_sim(records, qwen_desc_root)
                state["text_blend"] = text_blend
            else:
                state["text_blend"] = 0.0
            return state
        if knn_feature == "cls":
            raise SystemExit("--knn-feature cls is unavailable for the SAM3 encoder (no CLS token)")
        if mode in IMR_VISUAL_MODES or mode == "sam3enc_imr":
            extract_sam3_encoder_features(records, support, feature_root, feature_size, imr=True)
            data = np.load(feature_root / f"sam3_base_s{feature_size}_imr_features.npz")
            pyramid = data["pyramid"]
            contrast = data["contrast"]
            if mode == "sam3enc_pyramid":
                return {"mode": mode, "desc": pyramid, "sim": pyramid @ pyramid.T}
            if mode == "sam3enc_contrast":
                return {"mode": mode, "desc": contrast, "sim": contrast @ contrast.T}
            if mode == "sam3enc_pyramid_contrast":
                desc = l2norm(np.concatenate([pyramid, contrast], axis=1).astype(np.float32))
                return {"mode": mode, "desc": desc, "sim": desc @ desc.T}
            # sam3enc_imr: anchor-conditioned target-pooling cond + fused KNN
            # (pyramid+contrast visual rank fused with Qwen image-text rank by --text-blend).
            pc_desc = l2norm(np.concatenate([pyramid, contrast], axis=1).astype(np.float32))
            pc_sim = pc_desc @ pc_desc.T
            state = {
                "mode": mode,
                "patch_mean": data["patch_mean"],
                "sim": pc_sim,
                "knn_sim": pc_sim,
                "knn_feature": "fused",
                "cond_scores": data["cond_target"],
                "id_to_anchor": {anchor_id: i for i, anchor_id in enumerate(data["anchor_ids"].tolist())},
            }
            if text_blend > 0.0:
                state["text_sim"] = load_qwen_image_text_sim(records, qwen_desc_root)
                state["text_blend"] = text_blend
            else:
                state["text_blend"] = 0.0
            return state
        extract_sam3_encoder_features(records, support, feature_root, feature_size)
        data = np.load(feature_root / f"sam3_base_s{feature_size}_features.npz")
        patch_mean = data["patch_mean"]
        if mode == "sam3enc_patch_average":
            return {"mode": mode, "desc": patch_mean, "sim": patch_mean @ patch_mean.T}
        if mode in (
            "sam3enc_lesion",
            "sam3enc_global_lesion",
            "sam3enc_lesion_anchor_global_bridge",
            "sam3enc_global_anchor_lesion_bridge",
        ):
            lesion = load_lesion_descriptors(records, lesion_desc_path)
            if mode == "sam3enc_lesion":
                return {"mode": mode, "desc": lesion, "sim": lesion @ lesion.T}
            if mode == "sam3enc_global_lesion":
                desc = l2norm(np.concatenate([patch_mean, lesion], axis=1).astype(np.float32))
                return {"mode": mode, "desc": desc, "sim": desc @ desc.T}
            # E1 anchor/bridge decoupling: anchor edge uses anchor_desc, bridge edges use bridge_desc
            if mode == "sam3enc_lesion_anchor_global_bridge":
                anchor_desc, bridge_desc = lesion, patch_mean
            else:
                anchor_desc, bridge_desc = patch_mean, lesion
            return {
                "mode": mode,
                "desc": bridge_desc,
                "sim": bridge_desc @ bridge_desc.T,
                "anchor_desc": anchor_desc,
                "anchor_sim": anchor_desc @ anchor_desc.T,
            }
        cond_scores = data["cond_correspondence" if mode.endswith("patch_correspondence") else "cond_target"]
        if knn_feature == "patch_mean":
            knn_sim = patch_mean @ patch_mean.T
        elif knn_feature == "pooled":
            knn_sim = np.einsum("and,atd->ant", data["pooled"], data["pooled"]).astype(np.float32)
        else:
            knn_sim = None
        state = {
            "mode": mode,
            "patch_mean": patch_mean,
            "sim": patch_mean @ patch_mean.T,
            "knn_sim": knn_sim,
            "knn_feature": knn_feature,
            "cond_scores": cond_scores,
            "id_to_anchor": {anchor_id: i for i, anchor_id in enumerate(data["anchor_ids"].tolist())},
        }
        if text_blend > 0.0:
            state["text_sim"] = load_qwen_text_sim(records, qwen_desc_root)
            state["text_blend"] = text_blend
        else:
            state["text_blend"] = 0.0
        return state
    extract_encoder_features(records, support, feature_root, feature_source, feature_size)
    dinov3_model = feature_source.split("_", 1)[1]
    data = np.load(feature_root / f"dinov3_{dinov3_model}_s{feature_size}_features.npz")
    if mode == "dino_global_pooling":
        desc = data["cls"]
        return {"mode": mode, "desc": desc, "sim": desc @ desc.T}
    if mode == "dino_patch_average":
        desc = data["patch_mean"]
        return {"mode": mode, "desc": desc, "sim": desc @ desc.T}
    if knn_feature not in KNN_FEATURES:
        raise ValueError(f"Unknown --knn-feature: {knn_feature}")
    anchor_ids = data["anchor_ids"].tolist()
    anchor_proto = data["anchor_prototypes"]
    patches = data["patches"]
    cls = data["cls"]
    patch_mean = data["patch_mean"]
    id_to_anchor = {anchor_id: i for i, anchor_id in enumerate(anchor_ids)}
    patch_sims = np.einsum("ad,npd->anp", anchor_proto, patches).astype(np.float32)
    weights = np.exp((patch_sims - patch_sims.max(axis=2, keepdims=True)) * 10.0)
    weights = weights / np.maximum(weights.sum(axis=2, keepdims=True), 1e-12)
    pooled = l2norm(np.einsum("anp,npd->and", weights, patches).astype(np.float32), axis=2)
    if mode == "anchor_conditioned_patch_correspondence":
        cond_scores = np.sort(patch_sims, axis=2)[:, :, -8:].mean(axis=2)
    else:
        cond_scores = np.einsum("and,ad->an", pooled, anchor_proto).astype(np.float32)
    if knn_feature == "patch_mean":
        knn_sim = patch_mean @ patch_mean.T
    elif knn_feature == "cls":
        knn_sim = cls @ cls.T
    elif knn_feature == "pooled":
        knn_sim = np.einsum("and,atd->ant", pooled, pooled).astype(np.float32)
    else:
        knn_sim = None
    return {
        "mode": mode,
        "patch_mean": patch_mean,
        "sim": patch_mean @ patch_mean.T,
        "knn_sim": knn_sim,
        "knn_feature": knn_feature,
        "cond_scores": cond_scores,
        "id_to_anchor": id_to_anchor,
    }


def conditioned_score(state: dict[str, Any], anchor_id: str, node_idx: int) -> float:
    return float(state["cond_scores"][state["id_to_anchor"][anchor_id], node_idx])


def route_score(state: dict[str, Any], anchor_id: str, path: list[int], target_idx: int) -> tuple[float, float]:
    nodes = [*path, target_idx]
    if state["mode"] in ANCHOR_MODES:
        cond = [conditioned_score(state, anchor_id, node) for node in nodes]
        transitions = []
        prev = None
        for node in nodes:
            if prev is not None:
                if "edge_sim" in state:
                    anchor_idx = state["id_to_anchor"][anchor_id]
                    base = float(state["edge_sim"][anchor_idx, prev, node])
                else:
                    base = float(state["sim"][prev, node])
                if state.get("text_blend", 0.0) > 0.0 and "text_sim" in state:
                    blend = float(state["text_blend"])
                    text = float(state["text_sim"][prev, node])
                    transitions.append((1.0 - blend) * base + blend * text)
                else:
                    transitions.append(base)
            prev = node
        values = cond + transitions
    else:
        anchor_idx = state["id_to_index"][anchor_id]
        nodes2 = [anchor_idx, *path, target_idx]
        anchor_sim = state.get("anchor_sim")
        if state.get("text_blend", 0.0) > 0.0 and "text_sim" in state:
            blend = float(state["text_blend"])
            values = []
            for i in range(len(nodes2) - 1):
                base = (
                    float(anchor_sim[nodes2[i], nodes2[i + 1]])
                    if anchor_sim is not None and i == 0
                    else float(state["sim"][nodes2[i], nodes2[i + 1]])
                )
                text = float(state["text_sim"][nodes2[i], nodes2[i + 1]])
                values.append((1.0 - blend) * base + blend * text)
        else:
            values = []
            for i in range(len(nodes2) - 1):
                if anchor_sim is not None and i == 0:
                    values.append(float(anchor_sim[nodes2[0], nodes2[1]]))
                else:
                    values.append(float(state["sim"][nodes2[i], nodes2[i + 1]]))
    return min(values), float(np.mean(values))


def build_rank_cache(
    state: dict[str, Any],
    records: list[dict[str, Any]],
    support: list[dict[str, Any]],
    train_indices: list[int],
) -> dict[tuple[str, int], list[int]]:
    cache: dict[tuple[str, int], list[int]] = {}
    if state["mode"] in ANCHOR_MODES:
        knn_feature = state["knn_feature"]
        knn_sim = state["knn_sim"]
        text_blend = state.get("text_blend", 0.0)
        text_sim = state.get("text_sim")
        train_text = text_sim[train_indices] if (text_blend > 0.0 and text_sim is not None) else None
        for anchor in t21.human_pool(support, 512):
            anchor_id = anchor["anchor_id"]
            anchor_idx = state["id_to_anchor"][anchor_id]
            cond = np.asarray([conditioned_score(state, anchor_id, i) for i in train_indices])
            ids = np.asarray([records[i]["merged_id"] for i in train_indices])
            if knn_feature == "cond":
                order = np.lexsort((ids, cond))[::-1]
                ranked = [train_indices[int(pos)] for pos in order]
                for tail in range(len(records)):
                    cache[(anchor_id, tail)] = ranked
                continue
            for tail in range(len(records)):
                if knn_feature == "pooled":
                    sims = np.asarray([float(knn_sim[anchor_idx, i, tail]) for i in train_indices])
                else:
                    sims = np.asarray([float(knn_sim[i, tail]) for i in train_indices])
                if knn_feature == "fused" and train_text is not None:
                    text = np.asarray([float(train_text[j, tail]) for j in range(len(train_indices))])
                    sims = (1.0 - text_blend) * percentile_ranks(sims) + text_blend * percentile_ranks(text)
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
    key = (anchor_id, tail) if state["mode"] in ANCHOR_MODES else ("", tail)
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
    mode_key: str,
    output_root: Path,
    records: list[dict[str, Any]],
    support: list[dict[str, Any]],
    state: dict[str, Any],
    split: str,
    min_bridge: int,
    max_bridge: int,
    beam_width: int,
    exclude_support_targets: bool,
    extend_existing_routes: bool,
) -> list[dict[str, Any]]:
    phase = output_root / mode_key / f"{split}_pool0_stage1"
    routes_path = phase / "routes.jsonl"
    if routes_path.exists() and not extend_existing_routes:
        return read_jsonl(routes_path)
    phase.mkdir(parents=True, exist_ok=True)
    id_to_index = {row["merged_id"]: i for i, row in enumerate(records)}
    state["id_to_index"] = id_to_index
    train_indices = [i for i, row in enumerate(records) if row["split"] == "train"]
    anchors = t21.human_pool(support, 512)
    support_ids = {row["merged_id"] for row in support}
    targets = [
        row
        for row in records
        if row["split"] == split
        and not (exclude_support_targets and row["merged_id"] in support_ids)
    ]
    rank_cache = build_rank_cache(state, records, support, train_indices)
    routes = read_jsonl(routes_path) if routes_path.exists() else []
    existing = {(row["target_id"], int(row["bridge_count"])) for row in routes}
    for target_no, target in enumerate(sorted(targets, key=lambda row: row["merged_id"]), start=1):
        if target_no == 1 or target_no % 10 == 0:
            print(f"{mode_key}: target {target_no}/{len(targets)}", flush=True)
        target_idx = id_to_index[target["merged_id"]]
        for bridge_count in range(min_bridge, max_bridge + 1):
            if (target["merged_id"], bridge_count) in existing:
                continue
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
    parser.add_argument("--dinov3-model", choices=tuple(DINOV3_MODELS), default="vits16")
    parser.add_argument(
        "--feature-source",
        choices=FEATURE_SOURCES,
        default=None,
        help="Feature extractor for KNN descriptors (default: dinov3_{--dinov3-model}).",
    )
    parser.add_argument("--feature-size", type=int, default=256, help="Feature input resolution (square).")
    parser.add_argument("--text-blend", type=float, default=0.0, help="Blend weight for Qwen text similarity in path score.")
    parser.add_argument(
        "--mask-visual-fraction",
        type=float,
        default=0.0,
        help="Within the visual term, fraction from mask-conditioned SAM3 descriptor (0=original, 1=mask).",
    )
    parser.add_argument(
        "--qwen-desc-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/train_pseudo_masks_round1",
        help="Directory containing qwen35_*_descriptions.jsonl files.",
    )
    parser.add_argument(
        "--knn-feature",
        choices=KNN_FEATURES,
        default="patch_mean",
        help="Feature used for KNN candidate ranking in anchor-conditioned modes.",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--min-bridge", type=int, default=0)
    parser.add_argument("--max-bridge", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument(
        "--extend-existing-routes",
        action="store_true",
        help="Add missing target/bridge rows to an existing routes.jsonl without replacing rows.",
    )
    parser.add_argument(
        "--include-support-targets",
        action="store_true",
        help="Only for diagnostics. Train pseudo-label routes should leave support anchors out.",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn")
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/protocol",
        help="Directory containing merged_manifest.jsonl and support_manifest.jsonl.",
    )
    parser.add_argument(
        "--support-manifest",
        type=Path,
        default=None,
        help="Override the support (anchor) manifest. Defaults to {--protocol-root}/support_manifest.jsonl.",
    )
    parser.add_argument(
        "--lesion-desc-npz",
        type=Path,
        default=MASK_DESC_NPZ,
        help="Lesion (mask-foreground) descriptor npz for sam3enc_lesion / _global_lesion / ablation modes.",
    )
    args = parser.parse_args()
    protocol = args.protocol_root
    records = read_jsonl(protocol / "merged_manifest.jsonl")
    support_path = args.support_manifest or protocol / "support_manifest.jsonl"
    support = read_jsonl(support_path)
    feature_root = args.output_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    mode_key = args.mode
    if args.mode in ANCHOR_MODES:
        if args.knn_feature != "patch_mean":
            mode_key = f"{args.mode}__knn_{args.knn_feature}"
    elif args.knn_feature != "patch_mean":
        raise SystemExit(f"--knn-feature only applies to anchor-conditioned modes, got mode={args.mode}")
    if args.mode == "sam3enc_imr" and args.text_blend > 0.0:
        mode_key = f"{mode_key}__tb{int(round(args.text_blend * 100)):03d}"
    if args.mode in ("sam3enc_lesion", "sam3enc_global_lesion") and args.lesion_desc_npz != MASK_DESC_NPZ:
        tag = args.lesion_desc_npz.stem.replace("sam3enc_", "").replace("_s1008", "").replace("_descriptors", "")
        mode_key = f"{mode_key}__{tag}"
    state = build_mode_state(
        args.mode,
        records,
        support,
        feature_root,
        args.feature_source or f"dinov3_{args.dinov3_model}",
        args.knn_feature,
        args.feature_size,
        args.text_blend,
        args.qwen_desc_root,
        args.mask_visual_fraction,
        args.lesion_desc_npz,
    )
    routes = freeze_routes(
        mode_key,
        args.output_root,
        records,
        support,
        state,
        args.split,
        args.min_bridge,
        args.max_bridge,
        args.beam_width,
        not args.include_support_targets,
        args.extend_existing_routes,
    )
    meta = {
        "mode": args.mode,
        "mode_key": mode_key,
        "dinov3_model": args.dinov3_model,
        "feature_source": args.feature_source or f"dinov3_{args.dinov3_model}",
        "feature_size": args.feature_size,
        "knn_feature": (
            "fused"
            if args.mode == "sam3enc_imr"
            else "fpn_foreground_transport"
            if args.mode == "sam3enc_fpn_foreground_transport"
            else args.knn_feature
        ),
        "text_blend": args.text_blend,
        "beam_width": args.beam_width,
        "min_bridge": args.min_bridge,
        "max_bridge": args.max_bridge,
        "split": args.split,
        "routes": len(routes),
    }
    meta_path = args.output_root / mode_key / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"mode": args.mode, "split": args.split, "routes": len(routes), "min_bridge": args.min_bridge, "max_bridge": args.max_bridge}, indent=2))


if __name__ == "__main__":
    main()
