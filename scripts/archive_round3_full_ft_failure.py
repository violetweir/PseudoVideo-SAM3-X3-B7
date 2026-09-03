#!/usr/bin/env python3
"""Freeze T1/T2 as a stopped full-parameter adaptation failure ablation."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_round3_tracker_stage4")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def completed(run_name: str) -> list[dict[str, Any]]:
    return [load(path) for path in sorted((ROOT / run_name / "validation").glob("step_*/b0_b6_validation.json"))]


def main() -> None:
    groups = {"T1": "T1_memory", "T2": "T2_full_tracker"}
    results = {}
    for group, run_name in groups.items():
        rows = completed(run_name)
        rows.sort(key=lambda row: row["checkpoint_step"])
        best = sorted(rows, key=lambda row: (-row["mean_b3_b6"], -row["combined"]["b6"], row["checkpoint_step"]))[0]
        results[group] = {
            "completed_checkpoint_steps": [row["checkpoint_step"] for row in rows],
            "completed_checkpoint_count": len(rows),
            "best_completed_step": best["checkpoint_step"],
            "best_completed_mean_b3_b6": best["mean_b3_b6"],
            "best_completed_b0": best["combined"]["b0"],
            "best_completed_b6": best["combined"]["b6"],
            "later_partial_validation_excluded": True,
        }
    payload = {
        "status": "stopped_and_archived",
        "stopped_at": datetime.now().astimezone().isoformat(),
        "reason": "Stage4 full-parameter adaptation failure ablation reached a decisive conclusion; checkpoints 16k-20k were intentionally not completed.",
        "restart_policy": "T3/T4 must initialize from T0 e33 image side plus original SAM3-base tracker, never from T1/T2.",
        "interpretation": {
            "T1": "memory-decoder interface drift followed by empty-mask collapse",
            "T2": "current-frame shortcut/path collapse with B0 approximately equal to B6",
        },
        "groups": results,
    }
    path = ROOT / "summaries/full_parameter_adaptation_failure_ablation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    (ROOT / "ROUND3_STAGE4_FULL_FT_STOPPED").touch()
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

