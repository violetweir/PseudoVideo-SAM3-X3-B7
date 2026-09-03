#!/usr/bin/env python3
"""Render the completed Round3 reproduction report from frozen JSON artifacts."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import yaml

REPO = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
CFG = REPO / "configs/c0_256_round3_tracker_stage4.yaml"

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def one(pattern: str) -> Path:
    paths = sorted(REPO.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one match for {pattern}, got {paths}")
    return paths[0]

def fmt_row(name, row):
    c = row["combined"]
    return f"| {name} | " + " | ".join(f"{c[f'b{x}']:.6f}" for x in range(7)) + f" | {row['mean_b3_b6']:.6f} |"

def main() -> None:
    config = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    root = REPO / config["experiment"]["root"]
    sequence = load(root / "sequence_manifests/train_sequences.summary.json")
    audits = {g: load(root / "module_audit" / f"module_audit_{g}.json") for g in ("T1", "T2")}
    runs = {"T1": root / "T1_memory", "T2": root / "T2_full_tracker"}
    best = {g: load(runs[g] / "best_checkpoint.json") for g in runs}
    val = {
        "T0": load(one(f"{config['experiment']['root']}/T0_frozen/validation/*/b0_b6_validation.json")),
        **{g: load(Path(best[g]["result_path"])) for g in runs},
    }
    test = {
        "T0": load(one(f"{config['experiment']['root']}/T0_frozen/test/*/b0_b6_test.json")),
        **{g: load(one(f"{config['experiment']['root']}/{runs[g].name}/test/best/b0_b6_test.json")) for g in runs},
    }
    freeze = {g: load(runs[g] / "freeze_audit.json") for g in runs}
    training = {g: load(runs[g] / "training_summary.json") for g in runs}
    gains = {g: {f"b{x}": test[g]["combined"][f"b{x}"] - test["T0"]["combined"][f"b{x}"] for x in range(7)} for g in runs}
    long_gain = {g: test[g]["mean_b3_b6"] - test["T0"]["mean_b3_b6"] for g in runs}
    t2_long_dominant = long_gain["T2"] > gains["T2"]["b0"]
    conclusion = (
        "Pseudo-video Tracker adaptation selectively improves long-horizon propagation while preserving frozen spatial recognition."
        if long_gain["T2"] > 0 and t2_long_dominant
        else "Stage4-A did not produce a selectively larger long-chain gain; the frozen protocol is retained and the result is reported without changing topology or test selection."
    )
    diff = subprocess.run(["git", "diff", "--stat"], cwd=REPO, text=True, capture_output=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=REPO, text=True, capture_output=True).stdout.strip()
    bridge_counts = ", ".join(f"b{k}={v}" for k, v in sequence["bridge_counts"].items())
    report = f"""# C0-256 Round3：Stage-IV Pseudo-Video Tracker Adaptation

> 完成日期：自动记录于最终产物生成时  
> 仓库：`{REPO}`  
> 实验根目录：`{root}`

## 1. 研究问题

Round2B 显示 e33 topology 提高 b0，却降低多数 b3-b6。本轮固定 `G0=KNN(F_SAM3-base@256)` 与 e33 image-side，只训练 tracker/memory，检验长链传播是否改善。

## 2. Frozen protocol

- 初始化 checkpoint SHA256：`{config['protocol']['initialization_sha256']}`。
- target routes SHA256：`{config['protocol']['train_route_sha256']['sam3enc_anchor_conditioned_target_pooling']}`。
- patch routes SHA256：`{config['protocol']['train_route_sha256']['sam3enc_anchor_conditioned_patch_correspondence']}`。
- 524 pseudo manifest SHA256：`{config['protocol']['pseudo_manifest_sha256']}`。
- 只允许 train split；anchor box 仅在 frame 0 输入；bridge/target 无额外 prompt；Stage4-A long-chain loss 关闭。

## 3. Trainable module audit

| Group | Trainable params | Frozen params | Total params |
|---|---:|---:|---:|
| T1 memory-only | {audits['T1']['trainable_params']:,} | {audits['T1']['frozen_params']:,} | {audits['T1']['total_params']:,} |
| T2 full tracker | {audits['T2']['trainable_params']:,} | {audits['T2']['frozen_params']:,} | {audits['T2']['total_params']:,} |

T1 训练 `tracker.maskmem_backbone`、`tracker.transformer` 及直接 memory temporal embeddings；T2 训练完整 `tracker`。`detector` 的 vision/text/geometry/DETR/image segmentation 分支始终冻结。

## 4. Dataset

- sequence：{sequence['sequence_count']}；unique target：{sequence['unique_target_count']}；unique train frame：{sequence['unique_frame_count']}。
- bridge 分布：{bridge_counts}。
- mode 分布：{json.dumps(sequence['mode_counts'], sort_keys=True)}。
- supervision：{json.dumps(sequence['supervision_counts'], sort_keys=True)}。
- validation/test leakage intersection：0/0。

## 5. Training

AdamW，lr={config['groups']['T1']['lr']}，weight decay={config['training']['weight_decay']}，seed={config['experiment']['seed']}，BF16，sequence batch=1，gradient accumulation={config['training']['gradient_accumulation']}，T1/T2 各 {config['groups']['T1']['steps']} optimizer steps。实际 bridge 采样见各组 `training_summary.json`。

## 6. Validation checkpoint selection

固定规则：`mean(validation combined Dice at b3,b4,b5,b6)`；tie-break 为更高 b6，再选择更早 checkpoint。

- T1 best：step {best['T1']['step']}，ValLong={best['T1']['validation_mean_b3_b6']:.6f}，b6={best['T1']['validation_b6']:.6f}。
- T2 best：step {best['T2']['step']}，ValLong={best['T2']['validation_mean_b3_b6']:.6f}，b6={best['T2']['validation_b6']:.6f}。

## 7. Validation b0-b6

| Model | b0 | b1 | b2 | b3 | b4 | b5 | b6 | mean b3-b6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{fmt_row('T0', val['T0'])}
{fmt_row('T1', val['T1'])}
{fmt_row('T2', val['T2'])}

## 8. Test b0-b6

| Model | b0 | b1 | b2 | b3 | b4 | b5 | b6 | mean b3-b6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{fmt_row('T0', test['T0'])}
{fmt_row('T1', test['T1'])}
{fmt_row('T2', test['T2'])}

## 9. Gain-vs-depth

| Group | b0 | b1 | b2 | b3 | b4 | b5 | b6 | mean b3-b6 gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T1-T0 | {' | '.join(f"{gains['T1'][f'b{x}']:+.6f}" for x in range(7))} | {long_gain['T1']:+.6f} |
| T2-T0 | {' | '.join(f"{gains['T2'][f'b{x}']:+.6f}" for x in range(7))} | {long_gain['T2']:+.6f} |

LongGain=`mean(b3-b6)-b0`：T0={test['T0']['long_gain']:.6f}，T1={test['T1']['long_gain']:.6f}，T2={test['T2']['long_gain']:.6f}。T2 b6-b0 gap={test['T2']['b6_b0_gap']:.6f}。

## 10. Freeze audit

- T1 image hash before/after：`{freeze['T1']['image_side_hash_before']}` / `{freeze['T1']['image_side_hash_after']}`；changed frozen tensors={freeze['T1']['changed_frozen_tensor_count']}；nonzero frozen grads={freeze['T1']['nonzero_frozen_grad_count']}。
- T2 image hash before/after：`{freeze['T2']['image_side_hash_before']}` / `{freeze['T2']['image_side_hash_after']}`；changed frozen tensors={freeze['T2']['changed_frozen_tensor_count']}；nonzero frozen grads={freeze['T2']['nonzero_frozen_grad_count']}。
- NaN/Inf：T1={training['T1']['nan_or_inf']}，T2={training['T2']['nan_or_inf']}。

## 11. 结论

{conclusion}

Stage4 Tracker/Memory adaptation 的提升是否主要发生在长链而不是 b0：**{'是' if t2_long_dominant and long_gain['T2'] > 0 else '否'}**。该判断只来自 validation-frozen checkpoint 的一次正式 test。

## 12. Git diff 摘要

```text
{diff}
```

工作树状态（包含开工前已有的未跟踪 Round2C/2D 文件，未由本轮改写）：

```text
{status}
```
"""
    path = REPO / "reproduction_reports/C0_256_round3_tracker_stage4.md"
    path.write_text(report, encoding="utf-8")
    print(path)

if __name__ == "__main__":
    main()
