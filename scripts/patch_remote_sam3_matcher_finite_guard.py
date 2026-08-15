from pathlib import Path

p = Path("/Data_8TB/lht/sam3/sam3/train/matcher.py")
backup = p.with_suffix(p.suffix + ".codex_finite_guard_bak")
if not backup.exists():
    backup.write_text(p.read_text())

s = p.read_text()
if "import logging" not in s:
    s = s.replace("import numpy as np\n", "import logging\n\nimport numpy as np\n", 1)

old = """        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        # assign a very high cost (1e9) to invalid outputs and targets, so that we can
"""
new = """        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        if not torch.isfinite(C).all():
            bad = ~torch.isfinite(C)
            logging.warning(
                "BinaryHungarianMatcherV2 finite guard replacing invalid costs: "
                "bad=%s total=%s logits_finite=%s pred_boxes_finite=%s "
                "target_boxes_finite=%s cost_bbox_finite=%s cost_giou_finite=%s "
                "cost_class_finite=%s",
                int(bad.sum().item()),
                C.numel(),
                bool(torch.isfinite(out_score).all().item()),
                bool(torch.isfinite(out_bbox).all().item()),
                bool(torch.isfinite(tgt_bbox).all().item()),
                bool(torch.isfinite(cost_bbox).all().item()),
                bool(torch.isfinite(cost_giou).all().item()),
                bool(torch.isfinite(cost_class).all().item()),
            )
            C = torch.nan_to_num(C, nan=1e9, posinf=1e9, neginf=1e9)
        # assign a very high cost (1e9) to invalid outputs and targets, so that we can
"""

if "BinaryHungarianMatcherV2 finite guard replacing invalid costs" in s:
    print("matcher finite guard already patched")
elif old in s:
    p.write_text(s.replace(old, new, 1))
    print("patched matcher finite guard")
else:
    raise SystemExit("target text not found")
