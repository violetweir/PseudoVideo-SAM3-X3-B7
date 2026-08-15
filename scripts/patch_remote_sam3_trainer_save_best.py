#!/usr/bin/env python3
"""Enable SAM3 trainer best-checkpoint saving on validation AP improvement.

The SAM3 trainer reads `meter.is_better`, but no meter defines it, so the
built-in `save_best_meters` path is dead code.  This patch treats the
val_roboflow100 COCO segm AP meter as higher-is-better and saves
`checkpoints/<meter_key>.pt` (overwriting on each improvement) when
`checkpoint.save_best_meters` lists the meter key.
"""

from pathlib import Path


p = Path("/Data_8TB/lht/sam3/sam3/train/trainer.py")
backup = p.with_suffix(p.suffix + ".codex_save_best_bak")
if not backup.exists():
    backup.write_text(p.read_text())

s = p.read_text()
old = """                if is_better_check is None:
                    continue

                tracked_meter_key = os.path.join(key, meter_subkey)
                if tracked_meter_key not in self.best_meter_values or is_better_check(
                    meter_value,
                    self.best_meter_values[tracked_meter_key],
                ):"""
new = """                sub_is_better = is_better_check
                if (
                    sub_is_better is None
                    and key.startswith("val_")
                    and "coco_eval_segm_AP" in meter_subkey
                ):
                    sub_is_better = lambda a, b: a > b
                if sub_is_better is None:
                    continue

                tracked_meter_key = os.path.join(key, meter_subkey)
                if tracked_meter_key not in self.best_meter_values or sub_is_better(
                    meter_value,
                    self.best_meter_values[tracked_meter_key],
                ):"""
if "sub_is_better" in s:
    print("save-best patch already applied")
elif old in s:
    p.write_text(s.replace(old, new, 1))
    print("patched trainer save-best")
else:
    raise SystemExit("target text not found")
