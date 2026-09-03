#!/usr/bin/env python3
"""Round3 Stage-IV tracker/memory adaptation with a frozen SAM3 image model."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sam3.model_builder import build_sam3_video_model

REPO = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from round3_lora import inject_lora
MEMORY_PREFIXES = (
    "maskmem_backbone.", "transformer.", "maskmem_tpos_enc", "no_mem_embed",
    "no_mem_pos_enc", "no_obj_embed_spatial",
)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()

def hash_named_parameters(named_parameters) -> tuple[str, dict[str, str]]:
    overall = hashlib.sha256()
    per_tensor: dict[str, str] = {}
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        payload = tensor_bytes(parameter)
        digest = hashlib.sha256(payload).hexdigest()
        per_tensor[name] = digest
        overall.update(name.encode())
        overall.update(str(tuple(parameter.shape)).encode())
        overall.update(str(parameter.dtype).encode())
        overall.update(bytes.fromhex(digest))
    return overall.hexdigest(), per_tensor

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_memory_parameter(name: str) -> bool:
    return name.startswith(MEMORY_PREFIXES)

def configure_trainable(model: Any, policy: str, lora_config: dict[str, Any] | None = None) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    lora_audit = None
    if policy == "memory":
        for name, parameter in model.tracker.named_parameters():
            parameter.requires_grad_(is_memory_parameter(name))
    elif policy == "full_tracker":
        for parameter in model.tracker.parameters():
            parameter.requires_grad_(True)
    elif policy in {"memory_attention_lora", "memory_attention_decoder_lora"}:
        if lora_config is None:
            raise RuntimeError(f"LoRA config is required for {policy}")
        lora_audit = inject_lora(
            model.tracker, policy,
            rank=int(lora_config["rank"]),
            alpha=float(lora_config["alpha"]),
            dropout=float(lora_config.get("dropout", 0.0)),
        )
    elif policy != "frozen":
        raise ValueError(policy)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    frozen = [(name, p) for name, p in model.named_parameters() if not p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    audit = {
        "policy": policy,
        "frozen_modules": sorted({name.rsplit(".", 1)[0] for name, _ in frozen}),
        "trainable_modules": sorted({name.rsplit(".", 1)[0] for name, _ in trainable}),
        "frozen_params": sum(p.numel() for _, p in frozen),
        "trainable_params": sum(p.numel() for _, p in trainable),
        "total_params": total,
        "trainable_parameter_names": [name for name, _ in trainable],
    }
    if lora_audit is not None:
        audit["lora"] = lora_audit
        invalid = [name for name, _ in trainable if not (name.endswith(".lora_A") or name.endswith(".lora_B"))]
        if invalid:
            raise RuntimeError(f"Non-LoRA trainable parameters escaped policy {policy}: {invalid}")
    if audit["frozen_params"] + audit["trainable_params"] != total:
        raise RuntimeError("Parameter accounting mismatch")
    if any(name.startswith("detector.") for name, _ in trainable):
        raise RuntimeError("Image/detector parameter escaped the freeze boundary")
    return audit


def current_lora_update_norms(tracker: Any) -> dict[str, float]:
    parameters = dict(tracker.named_parameters())
    effective: list[float] = []
    right_factors: list[float] = []
    for name, left in parameters.items():
        if not name.endswith(".lora_A"):
            continue
        prefix = name.removesuffix(".lora_A")
        right = parameters[f"{prefix}.lora_B"]
        module = tracker.get_submodule(prefix)
        scaling = float(module.alpha) / int(module.rank)
        left_cpu = left.detach().float().cpu()
        right_cpu = right.detach().float().cpu()
        effective.append(float((right_cpu @ left_cpu).norm() * scaling))
        right_factors.append(float(right_cpu.norm()))
    if not effective:
        raise RuntimeError("No LoRA parameter pairs found while recording update norm")
    return {
        "module_count": len(effective),
        "effective_total": math.sqrt(sum(value * value for value in effective)),
        "effective_maximum": max(effective),
        "effective_mean": sum(effective) / len(effective),
        "lora_B_total": math.sqrt(sum(value * value for value in right_factors)),
    }

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)

def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

class FrozenSpatialCache:
    def __init__(self, model: Any, root: Path, canvas: int, device: torch.device):
        self.model = model
        self.root = root
        self.canvas = canvas
        self.device = device
        self.root.mkdir(parents=True, exist_ok=True)
        self.pos_path = self.root / "vision_pos_enc.pt"

    def key(self, image_path: str) -> str:
        return hashlib.sha1(image_path.encode()).hexdigest()

    def path(self, image_path: str) -> Path:
        return self.root / self.key(image_path)[:2] / f"{self.key(image_path)}.pt"

    def preprocess(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB").resize((self.canvas, self.canvas), Image.Resampling.BILINEAR)
        image = image.resize((1008, 1008), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        return (tensor - 0.5) / 0.5

    @torch.no_grad()
    def create(self, image_path: str) -> None:
        target = self.path(image_path)
        if target.exists():
            return
        tensor = self.preprocess(image_path)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model.detector.backbone.forward_image(tensor)["sam2_backbone_out"]
        payload = {"image_path": image_path, "backbone_fpn": [x.detach().cpu().to(torch.bfloat16) for x in output["backbone_fpn"]]}
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        if not self.pos_path.exists():
            temporary_pos = self.pos_path.with_suffix(".tmp")
            torch.save([x.detach().cpu().to(torch.bfloat16) for x in output["vision_pos_enc"]], temporary_pos)
            temporary_pos.replace(self.pos_path)

    def load(self, image_paths: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        entries = []
        for image_path in image_paths:
            target = self.path(image_path)
            if not target.exists():
                self.create(image_path)
            entries.append(torch.load(target, map_location="cpu", weights_only=True))
        fpn = [torch.cat([entry["backbone_fpn"][level] for entry in entries], dim=0).to(self.device, non_blocking=True) for level in range(3)]
        position = torch.load(self.pos_path, map_location="cpu", weights_only=True)
        position = [item.to(self.device, non_blocking=True).expand(len(entries), -1, -1, -1) for item in position]
        return fpn, position

class Round3Trainer:
    def __init__(self, config_path: Path, group: str, device: int, smoke: bool = False):
        self.config_path = config_path
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.group = group
        self.smoke = smoke
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        self.seed = int(self.config["experiment"]["seed"])
        set_seed(self.seed)
        self.rng = random.Random(self.seed)
        protocol = self.config["protocol"]
        self.base_checkpoint = resolve(protocol["initialization_checkpoint"])
        actual_checkpoint_hash = sha256_file(self.base_checkpoint)
        if actual_checkpoint_hash != protocol["initialization_sha256"]:
            raise RuntimeError(f"Initialization checkpoint SHA256 mismatch: {actual_checkpoint_hash}")
        self.sequence_path = resolve(protocol["sequence_manifest"])
        self.rows = read_jsonl(self.sequence_path)
        self.by_mode_bridge: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            self.by_mode_bridge[(row["mode"], int(row["bridge"]))].append(row)
        self.modes = list(self.config["sampling"]["mode_ratio"])
        run_name = {
            "T0": "T0_frozen", "T1": "T1_memory", "T2": "T2_full_tracker",
            "T3": "T3_memory_attention_lora", "T4": "T4_memory_attention_decoder_lora",
        }[group]
        self.run_dir = resolve(self.config["experiment"]["root"]) / run_name
        if smoke:
            self.run_dir = self.run_dir / "smoke"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, self.run_dir / "config.yaml")
        subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, stdout=(self.run_dir / "git_commit.txt").open("w"))
        environment = subprocess.run([sys.executable, "-m", "torch.utils.collect_env"], text=True, capture_output=True, check=False)
        (self.run_dir / "environment.txt").write_text(environment.stdout + environment.stderr, encoding="utf-8")
        (self.run_dir / "checkpoint_sha256.txt").write_text(actual_checkpoint_hash + "  " + str(self.base_checkpoint) + "\n", encoding="utf-8")
        (self.run_dir / "routes_sha256.txt").write_text("\n".join(f"{v}  {k}" for k, v in protocol["train_route_sha256"].items()) + "\n", encoding="utf-8")
        self.model = build_sam3_video_model(checkpoint_path=str(self.base_checkpoint), load_from_HF=False, device="cuda", compile=False)
        self.audit = configure_trainable(
            self.model, self.config["groups"][group]["trainable"], self.config.get("lora")
        )
        audit_root = resolve(self.config["experiment"]["root"]) / "module_audit"
        save_json(audit_root / f"module_audit_{group}.json", self.audit)
        save_json(self.run_dir / "module_audit.json", self.audit)
        if group in {"T2", "T4"}:
            save_json(resolve(self.config["experiment"]["root"]) / "module_audit.json", self.audit)
        self.image_hash_before, self.image_tensor_hash_before = hash_named_parameters(self.model.detector.named_parameters())
        self.frozen_tracker_before = {
            name: hashlib.sha256(tensor_bytes(parameter)).hexdigest()
            for name, parameter in self.model.tracker.named_parameters() if not parameter.requires_grad
        }
        self.cache = FrozenSpatialCache(self.model, resolve(protocol["spatial_feature_cache"]), int(self.config["experiment"]["canvas"]), self.device)
        self.tracker = self.model.tracker
        # The public video builder creates an inference predictor and therefore does
        # not initialize this training-only flag used by Sam3TrackerBase.  False is
        # required by our protocol: GT/pseudo masks are loss targets, never inputs
        # used to teacher-force object presence or memory selection.
        self.tracker.teacher_force_obj_scores_for_mem = False
        # Inference construction also omits the optional training augmentation
        # controls.  Keep spatial-memory dropout disabled for deterministic Stage4-A.
        self.tracker.prob_to_dropout_spatial_mem = 0.0
        self.tracker.rng = np.random.default_rng(self.seed)
        self.tracker.train()
        self.model.detector.eval()
        self.sample_counts: Counter[str] = Counter()
        self.mode_loss_sum: Counter[str] = Counter()
        self.mode_loss_count: Counter[str] = Counter()
        self.first_backward_checked = False
        trainable = [p for p in self.tracker.parameters() if p.requires_grad]
        self.max_steps = int(self.config["groups"][group]["steps"])
        self.lr = float(self.config["groups"][group].get("lr", 0.0))
        self.optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=float(self.config["training"]["weight_decay"])) if trainable else None
        warmup = int(self.config["training"]["warmup_steps"])
        min_ratio = float(self.config["training"]["min_lr_ratio"])
        def schedule(step: int) -> float:
            if step < warmup:
                return max(step, 1) / max(warmup, 1)
            progress = (step - warmup) / max(self.max_steps - warmup, 1)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule) if self.optimizer else None
        self.global_step = 0

    def cache_all(self) -> None:
        unique = {}
        for row in self.rows:
            unique.update(zip(row["frame_ids"], row["frames"]))
        started = time.time()
        for index, (frame_id, image_path) in enumerate(sorted(unique.items()), 1):
            self.cache.create(image_path)
            if index % 10 == 0 or index == len(unique):
                print(json.dumps({"cached": index, "total": len(unique), "frame_id": frame_id, "seconds": round(time.time() - started, 1)}), flush=True)
        save_json(self.cache.root / "summary.json", {"count": len(unique), "checkpoint_sha256": self.config["protocol"]["initialization_sha256"], "canvas": self.config["experiment"]["canvas"]})

    def release_detector_gpu(self) -> None:
        self.model.detector.to("cpu")
        torch.cuda.empty_cache()

    def curriculum_bridges(self, step: int) -> list[int]:
        fraction = step / max(self.max_steps, 1)
        for phase in self.config["curriculum"]:
            if fraction <= float(phase["until_fraction"]):
                return [int(x) for x in phase["bridges"]]
        return [int(x) for x in self.config["curriculum"][-1]["bridges"]]

    def sample(self, micro_index: int = 0) -> dict[str, Any]:
        mode = self.modes[(self.global_step + micro_index) % len(self.modes)]
        if self.smoke and self.global_step == 0:
            bridge = 6
        else:
            bridge = self.rng.choice(self.curriculum_bridges(self.global_step))
        pool = self.by_mode_bridge[(mode, bridge)]
        row = pool[self.rng.randrange(len(pool))]
        self.sample_counts[f"b{bridge}"] += 1
        self.sample_counts[mode] += 1
        return row

    def make_backbone_out(self, row: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        fpn, position = self.cache.load(row["frames"])
        fpn[0] = self.tracker.sam_mask_decoder.conv_s0(fpn[0])
        fpn[1] = self.tracker.sam_mask_decoder.conv_s1(fpn[1])
        box = row["anchor_box_xywh_normalized"]
        x1, y1, width, height = [float(x) for x in box]
        coords = torch.tensor([[[x1, y1], [x1 + width, y1 + height]]], device=self.device, dtype=torch.float32) * 1008.0
        labels = torch.tensor([[2, 3]], device=self.device, dtype=torch.int32)
        num_frames = len(row["frames"])
        backbone_out = {
            "backbone_fpn": fpn, "vision_pos_enc": position, "num_frames": num_frames,
            "init_cond_frames": [0], "frames_to_add_correction_pt": [],
            "frames_not_in_init_cond": list(range(1, num_frames)),
            "point_inputs_per_frame": {0: {"point_coords": coords, "point_labels": labels}},
            "mask_inputs_per_frame": {},
        }
        input_batch = SimpleNamespace(
            img_batch=torch.zeros(num_frames, 3, 1, 1, device=self.device, dtype=torch.bfloat16),
            find_inputs=[SimpleNamespace(img_ids=torch.tensor([index], device=self.device)) for index in range(num_frames)],
        )
        return backbone_out, input_batch

    def load_target(self, path: str) -> torch.Tensor:
        mask = Image.open(path).convert("L").resize((int(self.config["experiment"]["canvas"]),) * 2, Image.Resampling.NEAREST)
        array = (np.asarray(mask) > 127).astype(np.float32)
        return torch.from_numpy(array).to(self.device)[None, None]

    def forward_loss(self, row: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        backbone_out, input_batch = self.make_backbone_out(row)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_dict = self.tracker.forward_tracking(backbone_out, input_batch, return_dict=True)
            all_outputs = {**output_dict["cond_frame_outputs"], **output_dict["non_cond_frame_outputs"]}
            terms = []
            metrics = Counter()
            total_weight = 0.0
            for supervision in row["supervision"]:
                frame_index = int(supervision["frame_index"])
                output = all_outputs[frame_index]
                logits = F.interpolate(output["pred_masks_high_res"].float(), size=(256, 256), mode="bilinear", align_corners=False)
                target = self.load_target(supervision["mask"])
                probabilities = logits.sigmoid()
                intersection = (probabilities * target).sum()
                dice = 1.0 - (2.0 * intersection + 1.0) / (probabilities.sum() + target.sum() + 1.0)
                bce = F.binary_cross_entropy_with_logits(logits, target)
                with torch.no_grad():
                    predicted = probabilities >= 0.5
                    target_bool = target.bool()
                    inter = (predicted & target_bool).sum().float()
                    union = (predicted | target_bool).sum().float()
                    true_iou = inter / union.clamp_min(1.0)
                iou_prediction = output.get("iou_score")
                iou_loss = F.mse_loss(iou_prediction.float().mean(), true_iou) if iou_prediction is not None else logits.sum() * 0.0
                object_logits = output.get("object_score_logits")
                object_loss = F.binary_cross_entropy_with_logits(object_logits.float(), torch.ones_like(object_logits.float())) if object_logits is not None else logits.sum() * 0.0
                cfg = self.config["loss"]
                loss = float(cfg["dice_weight"]) * dice + float(cfg["bce_weight"]) * bce + float(cfg["iou_weight"]) * iou_loss + float(cfg["object_weight"]) * object_loss
                weight = float(supervision["weight"]) if cfg["pseudo_quality_weight"] else 1.0
                if frame_index == 0:
                    weight *= float(cfg["anchor_frame_weight"])
                if frame_index == len(row["frames"]) - 1:
                    weight *= float(cfg["terminal_frame_weight"])
                terms.append(loss * weight)
                total_weight += weight
                metrics["dice_loss"] += float(dice.detach())
                metrics["bce_loss"] += float(bce.detach())
                metrics["iou_loss"] += float(iou_loss.detach())
                metrics["object_loss"] += float(object_loss.detach())
            loss = sum(terms) / max(total_weight, 1e-8)
        memory_count = sum(1 for output in all_outputs.values() if output.get("maskmem_features") is not None)
        runtime = {"frame_count": len(all_outputs), "memory_frame_count": memory_count, "terminal_logits": all_outputs[len(row["frames"]) - 1]["pred_masks_high_res"]}
        return loss, dict(metrics), runtime

    def frozen_grad_audit(self) -> tuple[int, list[str]]:
        names = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad or parameter.grad is None:
                continue
            if torch.count_nonzero(parameter.grad).item() != 0:
                names.append(name)
        return len(names), names

    def save_checkpoint(self, path: Path) -> None:
        state = {
            "group": self.group, "global_step": self.global_step,
            "tracker_state": self.tracker.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "python_rng": random.getstate(), "local_rng": self.rng.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "sample_counts": dict(self.sample_counts),
            "base_checkpoint": str(self.base_checkpoint), "base_checkpoint_sha256": self.config["protocol"]["initialization_sha256"],
            "lora": self.audit.get("lora"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def resume(self) -> None:
        path = self.run_dir / "checkpoints" / "last.pt"
        if not path.exists():
            return
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.tracker.load_state_dict(checkpoint["tracker_state"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state(self.optimizer, self.device)
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = int(checkpoint["global_step"])
        random.setstate(checkpoint["python_rng"])
        self.rng.setstate(checkpoint["local_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        self.sample_counts.update(checkpoint.get("sample_counts", {}))
        print(f"Resumed {self.group} at step {self.global_step}", flush=True)

    def finish_freeze_audit(self, nonzero_frozen_grad_count: int = 0, names: list[str] | None = None) -> dict[str, Any]:
        image_hash_after, image_tensor_hash_after = hash_named_parameters(self.model.detector.named_parameters())
        frozen_tracker_after = {
            name: hashlib.sha256(tensor_bytes(parameter)).hexdigest()
            for name, parameter in self.model.tracker.named_parameters() if not parameter.requires_grad
        }
        changed = [f"detector.{name}" for name, digest in self.image_tensor_hash_before.items() if image_tensor_hash_after.get(name) != digest]
        changed += [f"tracker.{name}" for name, digest in self.frozen_tracker_before.items() if frozen_tracker_after.get(name) != digest]
        audit = {
            "image_side_hash_before": self.image_hash_before,
            "image_side_hash_after": image_hash_after,
            "changed_frozen_tensor_count": len(changed),
            "changed_frozen_tensors": changed,
            "nonzero_frozen_grad_count": nonzero_frozen_grad_count,
            "nonzero_frozen_grad_names": names or [],
            "initialization_checkpoint_sha256_before": self.config["protocol"]["initialization_sha256"],
            "initialization_checkpoint_sha256_after": sha256_file(self.base_checkpoint),
        }
        save_json(self.run_dir / "freeze_audit.json", audit)
        if audit["changed_frozen_tensor_count"] or audit["nonzero_frozen_grad_count"] or audit["image_side_hash_before"] != audit["image_side_hash_after"]:
            raise RuntimeError(f"Freeze audit failed: {audit}")
        return audit

    def checkpoint_reload_audit(self, row: dict[str, Any], checkpoint_path: Path) -> None:
        self.tracker.eval()
        with torch.no_grad():
            _, _, before = self.forward_loss(row)
            before_logits = before["terminal_logits"].detach().float().cpu()
            _, _, repeated = self.forward_loss(row)
            repeated_logits = repeated["terminal_logits"].detach().float().cpu()
        clone = copy.deepcopy(self.tracker).to(self.device).eval()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        clone.load_state_dict(payload["tracker_state"], strict=True)
        original = self.tracker
        self.tracker = clone
        with torch.no_grad():
            _, _, after = self.forward_loss(row)
            after_logits = after["terminal_logits"].detach().float().cpu()
        self.tracker = original
        maximum_error = float((before_logits - after_logits).abs().max())
        before_probability = before_logits.sigmoid()
        after_probability = after_logits.sigmoid()
        maximum_probability_error = float((before_probability - after_probability).abs().max())
        before_binary = before_logits >= 0
        after_binary = after_logits >= 0
        repeated_binary = repeated_logits >= 0
        identical_binary_output = bool(torch.equal(before_binary, after_binary))
        disagreement = int(torch.count_nonzero(before_binary != after_binary))
        repeat_disagreement = int(torch.count_nonzero(before_binary != repeated_binary))
        def binary_dice(left: torch.Tensor, right: torch.Tensor) -> float:
            intersection = int(torch.count_nonzero(left & right))
            return 2.0 * intersection / max(int(torch.count_nonzero(left)) + int(torch.count_nonzero(right)), 1)
        reload_dice = binary_dice(before_binary, after_binary)
        repeat_dice = binary_dice(before_binary, repeated_binary)
        numerically_consistent = reload_dice >= 0.999
        save_json(self.run_dir / "checkpoint_reload_audit.json", {
            "max_abs_logit_error": maximum_error,
            "max_abs_probability_error": maximum_probability_error,
            "bitwise_identical_logits": bool(torch.equal(before_logits, after_logits)),
            "identical_binary_output": identical_binary_output,
            "binary_disagreement_pixels": disagreement,
            "binary_mask_dice": reload_dice,
            "same_model_repeat_disagreement_pixels": repeat_disagreement,
            "same_model_repeat_binary_mask_dice": repeat_dice,
            "numerically_consistent_with_bf16_repeat": numerically_consistent,
            "checkpoint": str(checkpoint_path),
            "precision_note": "Independent BF16 CUDA forwards are compared at the actual binary-mask output boundary.",
        })
        del clone
        self.tracker.train()
        if not numerically_consistent:
            raise RuntimeError(f"Checkpoint reload failed BF16 binary-mask Dice threshold: reload_dice={reload_dice}")

    def train(self, max_steps_override: int | None = None) -> None:
        if self.optimizer is None:
            raise RuntimeError("T0 is evaluation-only")
        self.release_detector_gpu()
        self.resume()
        target_steps = min(max_steps_override, self.max_steps) if max_steps_override else self.max_steps
        accumulation = 1 if self.smoke else int(self.config["training"]["gradient_accumulation"])
        validation_interval = int(self.config["training"]["validation_interval"])
        train_stats = self.run_dir / "train_stats.jsonl"
        self.optimizer.zero_grad(set_to_none=True)
        nonzero_count = 0
        nonzero_names: list[str] = []
        peak_samples = []
        last_row = None
        initial_checkpoint = self.run_dir / "checkpoints" / "step_000000.pt"
        if self.global_step == 0 and 0 in self.config["training"]["save_steps"] and not initial_checkpoint.exists():
            self.save_checkpoint(initial_checkpoint)
        while self.global_step < target_steps:
            started = time.time()
            aggregate = Counter()
            for micro in range(accumulation):
                row = self.sample(micro)
                last_row = row
                try:
                    loss, metrics, runtime = self.forward_loss(row)
                except Exception:
                    save_json(self.run_dir / "nan_inf_dump.json", {"reason": "forward_exception", "route": row, "step": self.global_step})
                    raise
                if not torch.isfinite(loss):
                    save_json(self.run_dir / "nan_inf_dump.json", {"reason": "nonfinite_loss", "loss": float(loss.detach()), "route": row, "step": self.global_step})
                    raise FloatingPointError("Non-finite Stage4 loss")
                (loss / accumulation).backward()
                aggregate["loss"] += float(loss.detach())
                for key, value in metrics.items():
                    aggregate[key] += value
                aggregate["memory_frame_count"] += runtime["memory_frame_count"]
                aggregate["frame_count"] += runtime["frame_count"]
                self.mode_loss_sum[row["mode"]] += float(loss.detach())
                self.mode_loss_count[row["mode"]] += 1
                if not self.first_backward_checked:
                    nonzero_count, nonzero_names = self.frozen_grad_audit()
                    trainable_grad_count = sum(1 for p in self.tracker.parameters() if p.requires_grad and p.grad is not None and torch.count_nonzero(p.grad).item() > 0)
                    save_json(self.run_dir / "first_backward_audit.json", {"trainable_nonzero_grad_tensor_count": trainable_grad_count, "nonzero_frozen_grad_count": nonzero_count, "nonzero_frozen_grad_names": nonzero_names})
                    if nonzero_count or trainable_grad_count == 0:
                        raise RuntimeError("Gradient boundary audit failed")
                    self.first_backward_checked = True
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.tracker.parameters() if p.requires_grad],
                float(self.config["training"]["grad_clip_norm"]),
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            peak_samples.append(torch.cuda.max_memory_allocated() / 2**30)
            row_out = {
                "step": self.global_step, "loss": aggregate["loss"] / accumulation,
                "lr": self.optimizer.param_groups[0]["lr"], "seconds": time.time() - started,
                "bridge": int(last_row["bridge"]), "mode": last_row["mode"], "route_id": last_row["route_id"],
                "sample_counts": dict(self.sample_counts), "cuda_peak_gib": peak_samples[-1],
                "memory_frame_ratio": aggregate["memory_frame_count"] / max(aggregate["frame_count"], 1),
                "trainable_gradient_norm_pre_clip": float(gradient_norm.detach().cpu()),
            }
            if self.audit.get("lora") is not None:
                row_out["lora_update_norm"] = current_lora_update_norms(self.tracker)
            with train_stats.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row_out, sort_keys=True) + "\n")
                handle.flush()
            if self.global_step % 10 == 0 or self.global_step == target_steps:
                print(json.dumps(row_out, sort_keys=True), flush=True)
            should_save = (
                (validation_interval > 0 and self.global_step % validation_interval == 0)
                or self.global_step in self.config["training"]["save_steps"]
                or self.global_step == target_steps
            )
            if should_save:
                checkpoint = self.run_dir / "checkpoints" / f"step_{self.global_step:06d}.pt"
                self.save_checkpoint(checkpoint)
                shutil.copy2(checkpoint, self.run_dir / "checkpoints" / "last.pt")
        last_path = self.run_dir / "checkpoints" / f"step_{self.global_step:06d}.pt"
        if self.smoke and last_row is None:
            last_row = self.by_mode_bridge[(self.modes[0], 6)][0]
        if self.smoke and last_row is not None:
            self.checkpoint_reload_audit(last_row, last_path)
            historical_rows = read_jsonl(train_stats) if train_stats.exists() else []
            audited_peaks = peak_samples or [float(row["cuda_peak_gib"]) for row in historical_rows if "cuda_peak_gib" in row]
            memory_updated = any(float(row.get("memory_frame_ratio", 0.0)) > 0 for row in historical_rows)
            save_json(self.run_dir / "smoke_audit.json", {
                "b6_forward_backward": self.sample_counts.get("b6", 0) > 0,
                "memory_updated": memory_updated, "checkpoint_saved": last_path.exists(),
                "cuda_peak_samples_gib": audited_peaks,
                "unbounded_growth_detected": len(audited_peaks) >= 4 and audited_peaks[-1] > audited_peaks[0] * 1.5,
            })
        freeze = self.finish_freeze_audit(nonzero_count, nonzero_names)
        save_json(self.run_dir / "training_summary.json", {
            "group": self.group, "steps": self.global_step, "sample_counts": dict(self.sample_counts),
            "mode_mean_loss": {mode: self.mode_loss_sum[mode] / max(self.mode_loss_count[mode], 1) for mode in self.modes},
            "freeze_audit": freeze, "nan_or_inf": False,
        })

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO / "configs/c0_256_round3_tracker_stage4.yaml")
    parser.add_argument("--group", choices=("T0", "T1", "T2", "T3", "T4"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    trainer = Round3Trainer(args.config.resolve(), args.group, args.device, args.smoke)
    if args.cache_only:
        trainer.cache_all()
    elif args.audit_only:
        trainer.finish_freeze_audit()
    else:
        trainer.train(args.max_steps)

if __name__ == "__main__":
    main()
