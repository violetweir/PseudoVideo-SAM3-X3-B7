#!/usr/bin/env python3
"""Create the final T0/T3/T4 constrained pseudo-temporal LoRA report."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
CFG = REPO / "configs/c0_256_round3_tracker_lora.yaml"
RUNS = {"T3": "T3_memory_attention_lora", "T4": "T4_memory_attention_decoder_lora"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def row(name: str, result: dict[str, Any]) -> str:
    combined = result["combined"]
    return f"| {name} | " + " | ".join(f"{combined[f'b{x}']:.6f}" for x in range(7)) + f" | {result['mean_b3_b6']:.6f} | {result['b6_b0_gap']:+.6f} |"


def main() -> None:
    config = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    root = REPO / config["experiment"]["root"]
    old_root = REPO / "work/rerun_c0_256_round3_tracker_stage4"
    t0_validation = load(old_root / "T0_frozen/validation/e33_baseline/b0_b6_validation.json")
    t0_test = load(old_root / "T0_frozen/test/e33_baseline/b0_b6_test.json")
    supervision = load(root / "protocol/supervision_audit.json")
    audits = {group: load(root / "module_audit" / f"module_audit_{group}.json") for group in RUNS}
    best = {group: load(root / run / "best_checkpoint.json") for group, run in RUNS.items()}
    best_validation = {group: load(Path(best[group]["result_path"])) for group in RUNS}
    best_test = {group: load(root / run / "test/best/b0_b6_test.json") for group, run in RUNS.items()}
    freeze = {group: load(root / run / "freeze_audit.json") for group, run in RUNS.items()}
    path_results = {}
    all_validation = {}
    for group, run in RUNS.items():
        all_validation[group] = [load(path) for path in sorted((root / run / "validation").glob("step_*/b0_b6_validation.json"))]
        best_tag = f"step_{best[group]['step']:06d}"
        path_results[group] = load(root / run / "path_sensitivity" / best_tag / "path_sensitivity.json")
    gains = {
        group: {
            **{f"delta_b{x}": best_validation[group]["combined"][f"b{x}"] - t0_validation["combined"][f"b{x}"] for x in range(7)},
            "delta_mean_b3_b6": best_validation[group]["mean_b3_b6"] - t0_validation["mean_b3_b6"],
        }
        for group in RUNS
    }
    success = {
        group: (
            best_validation[group]["mean_b3_b6"] > t0_validation["mean_b3_b6"]
            and not best_validation[group]["empty_mask_collapse"]
            and path_results[group]["normal_minus_shuffle"] > 0
        )
        for group in RUNS
    }
    conclusion = (
        "At least one constrained LoRA variant preserved path sensitivity and exceeded the frozen T0 long-chain validation score."
        if any(success.values()) else
        "Neither constrained LoRA variant simultaneously exceeded T0, avoided collapse/shortcut behavior, and retained positive path sensitivity."
    )
    payload = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "supervision_audit": supervision,
        "trainable_parameter_audit": {group: {key: audits[group][key] for key in ("trainable_params", "frozen_params", "total_params", "lora")} for group in RUNS},
        "best_checkpoint": best,
        "best_validation": best_validation,
        "best_test": best_test,
        "validation_gain_over_T0": gains,
        "path_sensitivity_best": path_results,
        "freeze_audit": freeze,
        "all_validation_checkpoints": all_validation,
        "success": success,
        "conclusion": conclusion,
    }
    summary_path = root / "summaries/final_round3_lora_report.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    module_lines = []
    for group in RUNS:
        module_lines.append(f"### {group} ({audits[group]['lora']['target_count']} modules)\n")
        module_lines.extend(f"- `{name}`" for name in audits[group]["lora"]["target_modules"])
    report = f"""# Round3 T3/T4：Constrained Pseudo-Temporal LoRA Adaptation

> Completed: {payload['completed_at']}  
> Root: `{root}`

## Supervision audit

```json
{json.dumps({key: supervision[key] for key in ('num_supervised_positive_frames','num_supervised_empty_frames','num_unsupervised_frames','num_object_positive','num_object_negative','num_object_ignore','bug_found')}, indent=2)}
```

Bridge frames without pseudo masks are IGNORE for both segmentation and object presence. Bug found: **{supervision['bug_found']}**.

## LoRA boundary

| Group | Trainable | Frozen | Total | Targets |
|---|---:|---:|---:|---:|
| T3 | {audits['T3']['trainable_params']:,} | {audits['T3']['frozen_params']:,} | {audits['T3']['total_params']:,} | {audits['T3']['lora']['target_count']} |
| T4 | {audits['T4']['trainable_params']:,} | {audits['T4']['frozen_params']:,} | {audits['T4']['total_params']:,} | {audits['T4']['lora']['target_count']} |

## Validation-best

| Model | b0 | b1 | b2 | b3 | b4 | b5 | b6 | Mean b3-b6 | b6-b0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{row('T0', t0_validation)}
{row('T3', best_validation['T3'])}
{row('T4', best_validation['T4'])}

T3 best step: {best['T3']['step']}; T4 best step: {best['T4']['step']}.

## Validation gain over T0

```json
{json.dumps(gains, indent=2, sort_keys=True)}
```

## Formal test

| Model | b0 | b1 | b2 | b3 | b4 | b5 | b6 | Mean b3-b6 | b6-b0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{row('T0', t0_test)}
{row('T3', best_test['T3'])}
{row('T4', best_test['T4'])}

## 20-target B6 path sensitivity

| Group | Normal | Shuffle | Normal-Shuffle |
|---|---:|---:|---:|
| T3 | {path_results['T3']['normal']:.6f} | {path_results['T3']['shuffle']:.6f} | {path_results['T3']['normal_minus_shuffle']:+.6f} |
| T4 | {path_results['T4']['normal']:.6f} | {path_results['T4']['shuffle']:.6f} | {path_results['T4']['normal_minus_shuffle']:+.6f} |

## Freeze integrity

- T3 changed frozen tensors: {freeze['T3']['changed_frozen_tensor_count']}; nonzero frozen grads: {freeze['T3']['nonzero_frozen_grad_count']}.
- T4 changed frozen tensors: {freeze['T4']['changed_frozen_tensor_count']}; nonzero frozen grads: {freeze['T4']['nonzero_frozen_grad_count']}.
- Initialization hash: `{config['protocol']['initialization_sha256']}`.

## Actual inserted modules

{chr(10).join(module_lines)}

## Conclusion

{conclusion}

Did LoRA preserve the original SAM3 long-chain temporal prior? **{'Yes' if any(success.values()) else 'No under the predefined success criterion'}**.
"""
    report_path = REPO / "reproduction_reports/C0_256_round3_tracker_lora_t3_t4.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()

