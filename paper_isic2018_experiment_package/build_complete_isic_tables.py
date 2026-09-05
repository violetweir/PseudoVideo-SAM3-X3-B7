import ast
import csv
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
PHASE = PROJECT / "work/isic18_round1_round2a_from_pseudovideo_full"
OLD = PROJECT / "work/isic18_pseudovideo_full"
BASELINES = PROJECT / "work/isic18_1pct_protocol"
PKG = PROJECT / "paper_isic2018_experiment_package"
OUT = PKG / "complete_experiment_tables"


ROOTS = [PHASE, OLD, BASELINES]
SCRIPTS = [
    PROJECT / "scripts/run_isic18_round1_round2a_from_pvf.sh",
    PROJECT / "scripts/run_round2a_epoch27_gpu1.sh",
    PROJECT / "scripts/run_epoch27_sam3_longchain_test_gpu1.sh",
    PROJECT / "scripts/run_t24_student.py",
    PROJECT / "scripts/run_s27_student.py",
    PROJECT / "scripts/export_t25_student_predictions.py",
    PROJECT / "scripts/prepare_isic_resized_cache.py",
    PROJECT / "scripts/prepare_isic18_b7_medsam3_dataset.py",
    PROJECT / "scripts/eval_sam3_lora_direct_split.py",
    PROJECT / "scripts/merge_sam3_lora_video_checkpoint.py",
    PKG / "scripts_snapshot/run_round2a_epoch27_gpu1.sh",
    PKG / "scripts_snapshot/run_epoch27_sam3_longchain_test_gpu1.sh",
    BASELINES / "synfoc/train.py",
]


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def read_text(path: Path, limit=None) -> str:
    data = path.read_text(encoding="utf-8", errors="ignore")
    return data if limit is None else data[:limit]


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten(v, key)
    elif isinstance(obj, list):
        if len(obj) <= 20 and all(not isinstance(x, (dict, list)) for x in obj):
            yield prefix, json.dumps(obj, ensure_ascii=False)
        else:
            yield f"{prefix}.__len__", len(obj)
            for i, v in enumerate(obj[:20]):
                yield from flatten(v, f"{prefix}.{i}")
    else:
        yield prefix, obj


def scalar_type(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int) and not isinstance(v, bool):
        return "int"
    if isinstance(v, float):
        return "float"
    if v is None:
        return "null"
    return "str"


def safe_float(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
        return float(v)
    return None


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def iter_files():
    for root in ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                yield p


def file_kind(p: Path) -> str:
    s = str(p)
    if "contaminated" in s:
        return "contaminated_or_superseded"
    if "/benchmarks/" in s:
        return "runtime_benchmark"
    if "/students/" in s:
        return "student_training"
    if "/predictions/" in s:
        return "student_predictions"
    if "b7_calibration" in s or s.endswith("_b7.summary.json"):
        return "b7_selector_or_calibration"
    if "direct_" in p.name:
        return "sam3_direct_eval"
    if "propagation_quality" in s or "bridge" in p.name or "routes" in p.name:
        return "route_or_bridge"
    if "synfoc" in s or "scsam" in s:
        return "baseline"
    if "isic_records" in s:
        return "consolidated_record"
    if "protocol" in s:
        return "protocol_or_split"
    return "artifact"


def line_count(path: Path):
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return ""


def extract_command_blocks(script: Path):
    if not script.exists():
        return []
    lines = script.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        starts = (
            '"$SAMPY"' in stripped or '"$MKPY"' in stripped or
            "python " in stripped or stripped.startswith("CUDA_VISIBLE_DEVICES=") or
            stripped.startswith("nohup ") or stripped.startswith("bash ") or
            stripped.startswith("./") or stripped.startswith("scp ") or stripped.startswith("ssh ")
        )
        if starts:
            block = [stripped]
            while block[-1].endswith("\\") and i + 1 < len(lines):
                i += 1
                block.append(lines[i].strip())
            cmd = " ".join(x[:-1].strip() if x.endswith("\\") else x for x in block)
            rows.append({
                "source_script": rel(script),
                "line": i + 1 - len(block) + 1,
                "command_template": cmd,
                "notes": "extracted from script; shell variables preserved",
            })
        i += 1
    return rows


def extract_log_steps(log: Path):
    if not log.exists():
        return []
    rows = []
    pat = re.compile(r"^\[(?P<tag>[^\]]+)\]\s+(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(?P<msg>.*)")
    try:
        for n, line in enumerate(log.open(encoding="utf-8", errors="ignore"), start=1):
            m = pat.match(line.strip())
            if m:
                rows.append({
                    "log": rel(log),
                    "line": n,
                    "timestamp": m.group("time"),
                    "tag": m.group("tag"),
                    "step_message": m.group("msg"),
                })
    except Exception:
        pass
    return rows


def summarize_jsonl(path: Path):
    rows = []
    numeric = defaultdict(list)
    keys = set()
    count = 0
    first = None
    try:
        for line in path.open(encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if first is None:
                first = obj
            count += 1
            if isinstance(obj, dict):
                for k, v in flatten(obj):
                    keys.add(k)
                    fv = safe_float(v)
                    if fv is not None:
                        numeric[k].append(fv)
    except Exception:
        pass
    for k, vals in sorted(numeric.items()):
        if not vals:
            continue
        rows.append({
            "path": rel(path),
            "records": count,
            "field": k,
            "count_numeric": len(vals),
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "last": vals[-1],
            "keys_seen": ";".join(sorted(list(keys))[:80]),
        })
    if not rows:
        rows.append({
            "path": rel(path),
            "records": count,
            "field": "",
            "count_numeric": 0,
            "mean": "",
            "min": "",
            "max": "",
            "last": "",
            "keys_seen": ";".join(sorted(list(keys))[:80]),
        })
    return rows


def curve_rows(path: Path):
    rows = []
    if not path.exists() or path.suffix != ".jsonl":
        return rows
    try:
        for line in path.open(encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            flat = dict(flatten(obj))
            lower_keys = {k.lower(): k for k in flat}
            has_metric = any(x in lower_keys for x in ["iteration", "dice", "iou", "loss", "val_dice"])
            if not has_metric:
                continue
            rows.append({
                "path": rel(path),
                "iteration": flat.get("iteration", flat.get("iter", "")),
                "epoch": flat.get("epoch", ""),
                "loss": flat.get("loss", flat.get("train_loss", "")),
                "dice": flat.get("dice", flat.get("val_dice", flat.get("direct_val_dice", ""))),
                "iou": flat.get("iou", ""),
                "nonempty_rate": flat.get("nonempty_rate", ""),
                "lr": flat.get("lr", ""),
                "pseudo_weight": flat.get("pseudo_weight", ""),
                "raw_json": json.dumps(obj, ensure_ascii=False),
            })
    except Exception:
        pass
    return rows


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)

    all_files = list(iter_files())
    inv = []
    for p in sorted(all_files):
        st = p.stat()
        inv.append({
            "path": rel(p),
            "kind": file_kind(p),
            "suffix": p.suffix,
            "size_bytes": st.st_size,
            "modified_time": st.st_mtime,
            "line_count": line_count(p) if p.suffix in [".jsonl", ".log", ".md", ".csv", ".tsv", ".sh", ".py"] else "",
        })
    write_csv(OUT / "tables/all_isic_artifacts.csv", inv, ["path", "kind", "suffix", "size_bytes", "modified_time", "line_count"])

    json_metric_rows = []
    for p in sorted([x for x in all_files if x.suffix == ".json"]):
        obj = load_json(p)
        if obj is None:
            continue
        for k, v in flatten(obj):
            if isinstance(v, (dict, list)):
                continue
            json_metric_rows.append({
                "path": rel(p),
                "kind": file_kind(p),
                "metric_key": k,
                "metric_type": scalar_type(v),
                "metric_value": json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v,
            })
    write_csv(OUT / "tables/all_json_scalar_metrics_long.csv", json_metric_rows, ["path", "kind", "metric_key", "metric_type", "metric_value"])

    jsonl_summary_rows = []
    validation_curve_rows = []
    train_curve_rows = []
    for p in sorted([x for x in all_files if x.suffix == ".jsonl"]):
        jsonl_summary_rows.extend(summarize_jsonl(p))
        rows = curve_rows(p)
        if p.name == "validation.jsonl" or "val_stats" in p.name:
            validation_curve_rows.extend(rows)
        elif p.name in ["train.jsonl", "train_b7_lora_manifest.jsonl"] or "/train" in str(p):
            train_curve_rows.extend(rows)
    write_csv(OUT / "tables/all_jsonl_numeric_summaries.csv", jsonl_summary_rows, ["path", "records", "field", "count_numeric", "mean", "min", "max", "last", "keys_seen"])
    write_csv(OUT / "tables/validation_curves_extracted.csv", validation_curve_rows, ["path", "iteration", "epoch", "loss", "dice", "iou", "nonempty_rate", "lr", "pseudo_weight", "raw_json"])
    write_csv(OUT / "tables/train_curves_extracted.csv", train_curve_rows, ["path", "iteration", "epoch", "loss", "dice", "iou", "nonempty_rate", "lr", "pseudo_weight", "raw_json"])

    cmd_rows = []
    for s in SCRIPTS:
        cmd_rows.extend(extract_command_blocks(s))
    write_csv(OUT / "tables/commands_extracted_from_scripts.csv", cmd_rows, ["source_script", "line", "command_template", "notes"])

    log_rows = []
    for p in sorted(all_files):
        if p.suffix == ".log" and any(name in p.name for name in ["pipeline", "nohup", "launch", "bench"]):
            log_rows.extend(extract_log_steps(p))
    write_csv(OUT / "tables/pipeline_log_step_timeline.csv", log_rows, ["log", "line", "timestamp", "tag", "step_message"])

    mpath = PHASE / "isic_records/metrics.json"
    metrics = load_json(mpath) or {}
    final_rows = []
    main = metrics.get("main_test_dice", {})
    for k, v in main.items():
        final_rows.append({
            "group": "main_test_dice",
            "experiment": k,
            "split": "test",
            "metric": "dice",
            "value": v,
            "resolution_or_scope": "see notes/source",
            "source": rel(mpath),
            "notes": "",
        })
    for name, d in (metrics.get("students") or {}).items():
        for key, metric in [
            ("best_validation_dice", "best_validation_dice"),
            ("diagnostic_best_validation_dice", "diagnostic_best_validation_dice"),
            ("final_validation_dice", "final_validation_dice"),
            ("test_dice", "test_dice"),
            ("test_dice_recomputed", "test_dice_recomputed_256"),
            ("test_dice_recomputed_256", "test_dice_recomputed_256"),
            ("test_dice_recomputed_original_size", "test_dice_recomputed_original_size"),
        ]:
            if key in d:
                final_rows.append({
                    "group": "student",
                    "experiment": name,
                    "split": "validation" if "validation" in metric else "test",
                    "metric": metric,
                    "value": d[key],
                    "resolution_or_scope": "256 unless original_size is stated",
                    "source": d.get("summary", rel(mpath)),
                    "notes": f"iteration={d.get('best_iteration') or d.get('diagnostic_best_iteration') or d.get('final_iteration') or ''}",
                })
    for name in ["sam3_ep27_direct_text_only", "sam3_epoch50_direct_text_only", "sam3_epoch50_direct_text_only_s256"]:
        d = metrics.get(name)
        if d:
            for key in ["direct_dice", "direct_iou", "nonempty_rate", "valid_query_count_mean"]:
                if key in d:
                    final_rows.append({
                        "group": "sam3_direct",
                        "experiment": name,
                        "split": d.get("split", ""),
                        "metric": key,
                        "value": d[key],
                        "resolution_or_scope": d.get("metric_resolution", "1008 if not explicitly marked s256"),
                        "source": d.get("output", rel(mpath)),
                        "notes": "text/category only; no image prompt; no route propagation; actual category skin lesion",
                    })
    for name, d in (metrics.get("b7_selector") or {}).items():
        for key in ["selected_gt_dice_evaluation_only", "oracle_gt_dice_evaluation_only", "coverage"]:
            final_rows.append({
                "group": "b7_selector",
                "experiment": name,
                "split": "test",
                "metric": key,
                "value": d.get(key, ""),
                "resolution_or_scope": "selector/evaluation-only",
                "source": rel(mpath),
                "notes": "not standalone student-mask Dice",
            })
    write_csv(OUT / "tables/final_metric_ledger_long.csv", final_rows, ["group", "experiment", "split", "metric", "value", "resolution_or_scope", "source", "notes"])

    legacy_rows = []
    t21 = load_json(OLD / "t21_dynamic_pseudovideo/summary.json") or {}
    old_s2 = load_json(OLD / "committee_students/S2_final/summary.json") or {}
    proto = load_json(OLD / "protocol/protocol_summary.json") or {}
    pseudo_proto = load_json(OLD / "pseudo1522/protocol/protocol.json") or {}
    for split in ["round0_train", "test_pool0"]:
        d = t21.get(split, {})
        if d:
            legacy_rows.append({
                "experiment": "old_t21_dynamic_pseudovideo",
                "stage_or_split": split,
                "metric": "dice_mean",
                "value": d.get("dice_mean", ""),
                "source": "work/isic18_pseudovideo_full/t21_dynamic_pseudovideo/summary.json",
                "notes": f"n={d.get('n','')}; quality_mean={d.get('quality_mean','')}; accepted_count={d.get('accepted_count','')}",
            })
            legacy_rows.append({
                "experiment": "old_t21_dynamic_pseudovideo",
                "stage_or_split": split,
                "metric": "quality_gt_dice_pearson",
                "value": d.get("quality_gt_dice_pearson", ""),
                "source": "work/isic18_pseudovideo_full/t21_dynamic_pseudovideo/summary.json",
                "notes": f"spearman={d.get('quality_gt_dice_spearman','')}",
            })
    if old_s2:
        legacy_rows.extend([
            {"experiment": "old_committee_S2_final", "stage_or_split": "validation", "metric": "best_validation_dice", "value": old_s2.get("best_validation_dice", ""), "source": "work/isic18_pseudovideo_full/committee_students/S2_final/summary.json", "notes": f"best_iteration={old_s2.get('best_iteration','')}; final_iteration={old_s2.get('final_iteration','')}"},
            {"experiment": "old_committee_S2_final", "stage_or_split": "test", "metric": "test_dice", "value": (old_s2.get("test") or {}).get("dice", ""), "source": "work/isic18_pseudovideo_full/committee_students/S2_final/summary.json", "notes": f"test_iou={(old_s2.get('test') or {}).get('iou','')}"},
        ])
    if proto:
        legacy_rows.append({"experiment": "old_protocol", "stage_or_split": "split_counts", "metric": "counts", "value": json.dumps(proto.get("counts", {}), ensure_ascii=False), "source": "work/isic18_pseudovideo_full/protocol/protocol_summary.json", "notes": f"support_count={proto.get('support_count','')}"})
    if pseudo_proto:
        legacy_rows.append({"experiment": "old_pseudo1522_protocol", "stage_or_split": "pseudo_pool", "metric": "pseudo_count", "value": pseudo_proto.get("pseudo_count", ""), "source": "work/isic18_pseudovideo_full/pseudo1522/protocol/protocol.json", "notes": f"anchor_count={pseudo_proto.get('anchor_count','')}; route_type_counts={json.dumps(pseudo_proto.get('route_type_counts',{}), ensure_ascii=False)}"})
    write_csv(OUT / "tables/legacy_isic18_pseudovideo_full.csv", legacy_rows, ["experiment", "stage_or_split", "metric", "value", "source", "notes"])

    manual_steps = [
        ("0", "Old baseline protocol", "Use previous isic18_pseudovideo_full split/protocol", "no new command; inherited artifacts", "work/isic18_pseudovideo_full/protocol/protocol_summary.json"),
        ("1", "Route generation", "Generate b0-b6 KNN routes for train/validation/test and two SAM3 modes", "scripts/run_isic18_round1_round2a_from_pvf.sh::run_routes", f"{rel(PHASE)}/round1_sam3knn_s256_base/stage1_feature_knn_b0_b6/route_generation_summary.json"),
        ("2", "Base propagation", "Evaluate base SAM3 propagation quality for b0-b6", "scripts/run_isic18_round1_round2a_from_pvf.sh::run_base_propagation", f"{rel(PHASE)}/round1_sam3knn_s256_base/base_*_bridge_b0_b6.tsv"),
        ("3", "Resize cache", "Prepare 256 resized image/mask cache", "scripts/run_isic18_round1_round2a_from_pvf.sh::prepare_resized_cache", f"{rel(PHASE)}/isic_resized_cache_s256/summary.json"),
        ("4", "S2", "Train Round1 S2 student on original pseudo pool", "scripts/run_isic18_round1_round2a_from_pvf.sh::run_round1_students", f"{rel(PHASE)}/round1_sam3knn_s256_base/students/S2/summary.json"),
        ("5", "S3", "Train Round1 S3 student on consensus pseudo pool", "scripts/run_isic18_round1_round2a_from_pvf.sh::run_round1_students", f"{rel(PHASE)}/round1_sam3knn_s256_base/students/S3/summary.json"),
        ("6", "Audit/X3 manifest", "Audit candidates and build X3 pseudo manifest", "phase1_audit_tiers.py + phase1_build_x3_manifest.py", f"{rel(PHASE)}/round1_sam3knn_s256_base/audit/summary.json"),
        ("7", "X3", "Train Round1 X3 student", "scripts/run_isic18_round1_round2a_from_pvf.sh::run_round1_students", f"{rel(PHASE)}/round1_sam3knn_s256_base/students/X3/summary.json"),
        ("8", "Round1 B7 calibration", "Calibrate B7 selector from X3_best and base bridges", "scripts/run_isic18_round1_round2a_from_pvf.sh::calibrate_round1_b7_and_lora", f"{rel(PHASE)}/round1_sam3knn_s256_base/b7_calibration/validation_all.summary.json"),
        ("9", "SAM3 LoRA", "Train SAM3 LoRA with skin lesion category prompt", "scripts/run_isic18_round1_round2a_from_pvf.sh::calibrate_round1_b7_and_lora", f"{rel(PHASE)}/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/lora_weights/val_stats.json"),
        ("10", "Clean Round2A epoch27", "Merge epoch27 LoRA, regenerate teacher bridges, train X4", "scripts/run_round2a_epoch27_gpu1.sh", f"{rel(PHASE)}/round2a_fixed_knn_lora_teacher/students/X4/summary.json"),
        ("11", "Long-chain epoch27 test", "Evaluate b0-b7 long chain for epoch27 LoRA", "scripts/run_epoch27_sam3_longchain_test_gpu1.sh", f"{rel(PHASE)}/round2a_epoch27_longchain_test_b0_b7/test_bridge_b0_b7.tsv"),
        ("12", "Bridge benefit analysis", "Compare target_pooling bridge benefit before/after LoRA", "analysis script/run from generated outputs", f"{rel(PHASE)}/analysis/target_pooling_bridge_benefit_b0_b6/combined_summary.json"),
        ("13", "SAM3 direct evals", "Evaluate text-only SAM3 direct, no image prompt", "scripts/eval_sam3_lora_direct_split.py with category prompt", f"{rel(PHASE)}/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/direct_test_*_merged*.json"),
        ("14", "Baselines", "SynFoC and SCSAM baseline tests", "work/isic18_1pct_protocol/logs/*launch.log and rerun log", "work/isic18_1pct_protocol/synfoc/summary.json; work/isic18_1pct_protocol/logs/scsam_test_best_rerun_20260903_160852.log"),
    ]
    write_csv(OUT / "tables/manual_experiment_step_map.csv",
              [{"step_id": a, "step_name": b, "purpose": c, "command_or_entry": d, "primary_output": e} for a, b, c, d, e in manual_steps],
              ["step_id", "step_name", "purpose", "command_or_entry", "primary_output"])

    actual_commands = [
        {
            "scope": "main pipeline",
            "command": "cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7 && DEVICE=0 bash scripts/run_isic18_round1_round2a_from_pvf.sh",
            "status": "superseded/restarted in parts",
            "log_or_output": "work/isic18_round1_round2a_from_pseudovideo_full/pipeline.log",
            "notes": "Main Round1+Round2A entry; later clean Round2A used epoch27 separately.",
        },
        {
            "scope": "Round2A clean epoch27",
            "command": "cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7 && CUDA_VISIBLE_DEVICES=1 bash paper_isic2018_experiment_package/scripts_snapshot/run_round2a_epoch27_gpu1.sh",
            "status": "complete",
            "log_or_output": "work/isic18_round1_round2a_from_pseudovideo_full/pipeline_round2a_epoch27_gpu1.log",
            "notes": "Helper script snapshot records the clean rerun logic; original dirty Round2A was moved aside.",
        },
        {
            "scope": "epoch27 long-chain test",
            "command": "cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7 && CUDA_VISIBLE_DEVICES=1 bash paper_isic2018_experiment_package/scripts_snapshot/run_epoch27_sam3_longchain_test_gpu1.sh",
            "status": "complete; b7 abnormal",
            "log_or_output": "work/isic18_round1_round2a_from_pseudovideo_full/pipeline_epoch27_longchain_test_gpu1.log",
            "notes": "Evaluated b0-b7 chain using epoch27 LoRA teacher.",
        },
        {
            "scope": "SAM3 epoch27 direct test",
            "command": "CUDA_VISIBLE_DEVICES=1 /home/violet/anaconda3/envs/sam3/bin/python scripts/eval_sam3_lora_direct_split.py --config work/isic18_round1_round2a_from_pseudovideo_full/isic_records/isic18_sam3_ep27_merged_direct_config.yaml --split test --output work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/direct_test_epoch27_category_merged.json --prompt-mode category",
            "status": "complete",
            "log_or_output": "work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/direct_test_epoch27_category_merged.json",
            "notes": "Text/category-only; no image prompt; metric resolution 1008.",
        },
        {
            "scope": "SAM3 epoch27 direct validation",
            "command": "CUDA_VISIBLE_DEVICES=1 /home/violet/anaconda3/envs/sam3/bin/python scripts/eval_sam3_lora_direct_split.py --config work/isic18_round1_round2a_from_pseudovideo_full/isic_records/isic18_sam3_ep27_merged_direct_config.yaml --split valid --output work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/direct_valid_epoch27_category_merged.json --prompt-mode category",
            "status": "complete",
            "log_or_output": "work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/direct_valid_epoch27_category_merged.json",
            "notes": "Independent validation re-eval close to training val.",
        },
        {
            "scope": "SAM3 epoch50 direct test 1008",
            "command": "CUDA_VISIBLE_DEVICES=1 /home/violet/anaconda3/envs/sam3/bin/python scripts/eval_sam3_lora_direct_split.py --config work/isic18_round1_round2a_from_pseudovideo_full/isic_records/isic18_sam3_epoch50_merged_direct_config.yaml --split test --output work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/direct_test_epoch50_category_merged.json --prompt-mode category",
            "status": "complete",
            "log_or_output": "work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/direct_test_epoch50_category_merged.json",
            "notes": "Text/category-only; no image prompt; metric resolution 1008.",
        },
        {
            "scope": "SAM3 epoch50 direct test 256",
            "command": "CUDA_VISIBLE_DEVICES=1 /home/violet/anaconda3/envs/sam3/bin/python scripts/eval_sam3_lora_direct_split.py --config work/isic18_round1_round2a_from_pseudovideo_full/isic_records/isic18_sam3_epoch50_merged_direct_config.yaml --split test --output work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/direct_test_epoch50_category_merged_s256.json --prompt-mode category --effective-resolution 256",
            "status": "complete",
            "log_or_output": "work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/direct_test_epoch50_category_merged_s256.json",
            "notes": "Fair comparison to 256-resolution student masks.",
        },
        {
            "scope": "SCSAM test rerun",
            "command": "see work/isic18_1pct_protocol/logs/scsam_test_best_rerun_20260903_160852.log",
            "status": "complete",
            "log_or_output": "work/isic18_1pct_protocol/logs/scsam_test_best_rerun_20260903_160852.log",
            "notes": "SCSAM-SAM Dice 0.852200; SCSAM-UNet Dice 0.830954.",
        },
        {
            "scope": "SynFoC baseline",
            "command": "see work/isic18_1pct_protocol/logs/synfoc_launch.log and work/isic18_1pct_protocol/synfoc/train.py",
            "status": "complete",
            "log_or_output": "work/isic18_1pct_protocol/synfoc/summary.json",
            "notes": "SynFoC-SAM Dice 0.873360; SynFoC-UNet Dice 0.841385.",
        },
    ]
    write_csv(OUT / "tables/manual_actual_or_equivalent_commands.csv", actual_commands, ["scope", "command", "status", "log_or_output", "notes"])

    readme = """# Complete ISIC2018 Experiment Tables

This directory is the detailed ISIC2018 experiment ledger. It complements the concise paper-facing files one level above.

## Key Tables

- `tables/final_metric_ledger_long.csv`: curated metric ledger for main results, students, SAM3 direct tests, and B7 selector.
- `tables/manual_experiment_step_map.csv`: human-readable step map from protocol inheritance to baselines.
- `tables/manual_actual_or_equivalent_commands.csv`: actual or equivalent top-level commands used for the major runs.
- `tables/legacy_isic18_pseudovideo_full.csv`: explicit table for the old inherited ISIC run.
- `tables/commands_extracted_from_scripts.csv`: command templates extracted from shell/Python scripts. Variables such as `$SAMPY`, `$DATA`, `$ROUND1` are preserved.
- `tables/pipeline_log_step_timeline.csv`: timestamped step messages extracted from pipeline/nohup/launch logs.
- `tables/all_json_scalar_metrics_long.csv`: scalar values from every JSON artifact under ISIC-related experiment roots.
- `tables/all_jsonl_numeric_summaries.csv`: record counts and numeric summaries for JSONL artifacts.
- `tables/validation_curves_extracted.csv`: validation curves extracted from JSONL logs when available.
- `tables/train_curves_extracted.csv`: training curves extracted from JSONL logs when available.
- `tables/all_isic_artifacts.csv`: full inventory of ISIC-related artifacts.

## Resolution Warning

SAM3 direct text-only has both 1008 and 256 metric-resolution runs. For comparison against 256 student masks, use the `sam3_epoch50_direct_text_only_s256` row.
"""
    write_text(OUT / "README.md", readme)

    # Copy the compact source files too.
    raw = OUT / "raw"
    for src in [mpath, PHASE / "isic_records/README.md"]:
        if src.exists():
            shutil.copy2(src, raw / src.name)

    print(f"WROTE {OUT}")
    print(f"files={len(inv)} json_scalars={len(json_metric_rows)} jsonl_summaries={len(jsonl_summary_rows)} commands={len(cmd_rows)} log_steps={len(log_rows)}")


if __name__ == "__main__":
    main()
