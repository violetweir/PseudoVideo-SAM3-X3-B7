#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
rows = []
for name, value in payload["tracker_state"].items():
    if name.endswith(("lora_A", "lora_B")):
        rows.append({"name": name, "dtype": str(value.dtype), "norm": float(value.float().norm()), "nonzero": int(torch.count_nonzero(value))})
print(json.dumps({
    "step": payload["global_step"],
    "lr": payload["optimizer"]["param_groups"][0]["lr"],
    "lora_A": [row for row in rows if row["name"].endswith("lora_A")],
    "lora_B": [row for row in rows if row["name"].endswith("lora_B")],
    "effective_update_norms": [
        float((value.float() @ payload["tracker_state"][name.removesuffix(".lora_B") + ".lora_A"].float()).norm() * 2.0)
        for name, value in payload["tracker_state"].items() if name.endswith(".lora_B")
    ],
}, indent=2))
