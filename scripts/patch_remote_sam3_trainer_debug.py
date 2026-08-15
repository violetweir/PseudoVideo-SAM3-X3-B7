from pathlib import Path

p = Path("/Data_8TB/lht/sam3/sam3/train/trainer.py")
backup = p.with_suffix(p.suffix + ".codex_debug_bak")
if not backup.exists():
    backup.write_text(p.read_text())

s = p.read_text()
old = """        find_stages = model(batch)
        find_targets = [
"""
new = """        try:
            find_stages = model(batch)
        except Exception:
            logging.exception("MODEL_FORWARD_FAILED during %s; dumping batch metadata", phase)
            try:
                for stage_idx, meta in enumerate(getattr(batch, "find_metadatas", [])):
                    dump = {}
                    for field in ["coco_image_id", "original_image_id", "object_id", "original_category_id", "original_size"]:
                        value = getattr(meta, field, None)
                        if hasattr(value, "detach"):
                            value = value.detach().cpu().tolist()
                        dump[field] = value
                    logging.error("MODEL_FORWARD_FAILED metadata stage=%s %s", stage_idx, dump)
                for stage_idx, target in enumerate(getattr(batch, "find_targets", [])):
                    boxes = getattr(target, "boxes", None)
                    boxes_padded = getattr(target, "boxes_padded", None)
                    num_boxes = getattr(target, "num_boxes", None)

                    def tensor_stats(x):
                        if not hasattr(x, "detach"):
                            return str(x)
                        x_cpu = x.detach().cpu()
                        finite = torch.isfinite(x_cpu) if x_cpu.is_floating_point() else torch.ones_like(x_cpu, dtype=torch.bool)
                        out = {"shape": list(x_cpu.shape), "finite": bool(finite.all().item())}
                        if x_cpu.numel() and x_cpu.is_floating_point():
                            safe = torch.nan_to_num(x_cpu)
                            out.update({"min": float(safe.min().item()), "max": float(safe.max().item())})
                        elif x_cpu.numel():
                            out.update({"min": int(x_cpu.min().item()), "max": int(x_cpu.max().item())})
                        return out

                    logging.error(
                        "MODEL_FORWARD_FAILED target stage=%s num_boxes=%s boxes=%s boxes_padded=%s",
                        stage_idx,
                        tensor_stats(num_boxes),
                        tensor_stats(boxes),
                        tensor_stats(boxes_padded),
                    )
            except Exception:
                logging.exception("MODEL_FORWARD_FAILED metadata dump itself failed")
            raise
        find_targets = [
"""

if "MODEL_FORWARD_FAILED during" in s:
    print("trainer debug already patched")
elif old in s:
    p.write_text(s.replace(old, new, 1))
    print("patched trainer debug")
else:
    raise SystemExit("target text not found")
