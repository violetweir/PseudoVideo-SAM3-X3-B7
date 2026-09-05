import csv
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
ROOTS = [
    PROJECT / "work/kvasir_1pct_anchors",
    PROJECT / "work/clinicdb_external_kvasir8",
]
SCRIPT_FILES = [
    PROJECT / "scripts",
    PROJECT / "paper_kvasir_experiment_package",
]
PKG = PROJECT / "paper_kvasir_experiment_package"
TABLES = PKG / "tables"
RAW = PKG / "raw_records"


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        if len(obj) <= 30 and all(not isinstance(x, (dict, list)) for x in obj):
            yield prefix, json.dumps(obj, ensure_ascii=False)
        else:
            yield f"{prefix}.__len__", len(obj)
            for i, v in enumerate(obj[:30]):
                yield from flatten(v, f"{prefix}.{i}")
    else:
        yield prefix, obj


def safe_num(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
        return float(v)
    return None


def value_type(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int) and not isinstance(v, bool):
        return "int"
    if isinstance(v, float):
        return "float"
    if v is None:
        return "null"
    return "str"


def iter_files():
    for root in ROOTS:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file():
                    yield p


def kind(path: Path):
    s = rel(path).lower()
    if "scsam" in s or "synfoc" in s:
        return "baseline"
    if "lora" in s:
        return "lora_or_lora_eval"
    if "route_family_summary" in s or "propagation_quality" in s or "routes.jsonl" in s:
        return "route_or_propagation"
    if "candidate_invariant_router" in s or "unified_candidate_router" in s or "router" in s:
        return "router_or_selector"
    if "mask_prompt" in s or "triple" in s or "hybrid" in s or "qwen" in s or "text" in s:
        return "prompt_or_multimodal_ablation"
    if "protocol" in s or "metadata" in s or "baseline_data" in s:
        return "protocol_or_data"
    if "clinicdb_external" in s:
        return "clinicdb_external_eval"
    if path.suffix == ".log":
        return "log"
    return "artifact"


def line_count(path: Path):
    if path.suffix.lower() not in {".jsonl", ".log", ".md", ".csv", ".tsv", ".py", ".sh", ".txt"}:
        return ""
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return ""


def summarize_jsonl(path: Path):
    count = 0
    keys = set()
    nums = defaultdict(list)
    try:
        for line in path.open(encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            count += 1
            if isinstance(obj, dict):
                for k, v in flatten(obj):
                    keys.add(k)
                    n = safe_num(v)
                    if n is not None:
                        nums[k].append(n)
    except Exception:
        pass
    rows = []
    for k, vals in sorted(nums.items()):
        rows.append({
            "path": rel(path),
            "kind": kind(path),
            "records": count,
            "field": k,
            "count_numeric": len(vals),
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
            "last": vals[-1],
            "keys_seen": ";".join(sorted(keys)[:120]),
        })
    if not rows:
        rows.append({
            "path": rel(path),
            "kind": kind(path),
            "records": count,
            "field": "",
            "count_numeric": 0,
            "mean": "",
            "min": "",
            "max": "",
            "last": "",
            "keys_seen": ";".join(sorted(keys)[:120]),
        })
    return rows


def route_rows(path: Path, data):
    rows = []
    flat = dict(flatten(data))
    for k, v in flat.items():
        if k.endswith(".dice") or k == "direct.dice":
            n = safe_num(v)
            if n is None:
                continue
            family = k.rsplit(".", 1)[0]
            rows.append({
                "path": rel(path),
                "root": rel(path.parent),
                "family_or_bridge": family,
                "dice": n,
                "kind": kind(path),
            })
    return rows


def extract_log_metrics(path: Path):
    rows = []
    metric_pat = re.compile(r"(dice|iou|hd95|asd|loss|val[_ ]?dice|mean[_ ]?dice)[:= ]+([0-9]*\.?[0-9]+)", re.I)
    try:
        for idx, line in enumerate(path.open(encoding="utf-8", errors="ignore"), start=1):
            low = line.lower()
            if not any(token in low for token in ["dice", "iou", "hd95", "asd", "loss", "epoch", "iteration", "test", "val"]):
                continue
            for m in metric_pat.finditer(line):
                rows.append({
                    "path": rel(path),
                    "line": idx,
                    "metric": m.group(1),
                    "value": m.group(2),
                    "line_text": line.strip()[:600],
                })
    except Exception:
        pass
    return rows


def extract_commands(path: Path):
    rows = []
    if path.suffix.lower() not in {".sh", ".py", ".log", ".md"}:
        return rows
    cmd_re = re.compile(r"(CUDA_VISIBLE_DEVICES=\S+\s+)?(/home/violet/[^\s]+/python|python|bash|nohup|scp|ssh|\$SAMPY|\$MKPY|\"\$SAMPY\"|\"\$MKPY\")\b")
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return rows
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if cmd_re.search(stripped):
            block = [stripped]
            while block[-1].endswith("\\") and i + 1 < len(lines):
                i += 1
                block.append(lines[i].strip())
            cmd = " ".join(x[:-1].strip() if x.endswith("\\") else x for x in block)
            rows.append({
                "source": rel(path),
                "line": i + 1 - len(block) + 1,
                "command_or_template": cmd[:2500],
                "notes": "auto-extracted; verify shell variables for exact rerun",
            })
        i += 1
    return rows


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def fmt(x):
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def main():
    if PKG.exists():
        shutil.rmtree(PKG)
    TABLES.mkdir(parents=True)
    RAW.mkdir(parents=True)

    files = sorted(iter_files())
    inventory = []
    for p in files:
        st = p.stat()
        inventory.append({
            "path": rel(p),
            "kind": kind(p),
            "suffix": p.suffix,
            "size_bytes": st.st_size,
            "modified_time": st.st_mtime,
            "line_count": line_count(p),
        })
    write_csv(TABLES / "all_kvasir_artifacts.csv", inventory, ["path", "kind", "suffix", "size_bytes", "modified_time", "line_count"])

    json_scalar = []
    route_metric = []
    for p in [x for x in files if x.suffix.lower() == ".json"]:
        data = load_json(p)
        if data is None:
            continue
        for k, v in flatten(data):
            if isinstance(v, (dict, list)):
                continue
            json_scalar.append({
                "path": rel(p),
                "kind": kind(p),
                "metric_key": k,
                "metric_type": value_type(v),
                "metric_value": json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v,
            })
        route_metric.extend(route_rows(p, data))
    write_csv(TABLES / "all_json_scalar_metrics_long.csv", json_scalar, ["path", "kind", "metric_key", "metric_type", "metric_value"])
    write_csv(TABLES / "route_family_and_bridge_dice.csv", route_metric, ["path", "root", "family_or_bridge", "dice", "kind"])

    jsonl_summaries = []
    for p in [x for x in files if x.suffix.lower() == ".jsonl"]:
        jsonl_summaries.extend(summarize_jsonl(p))
    write_csv(TABLES / "all_jsonl_numeric_summaries.csv", jsonl_summaries, ["path", "kind", "records", "field", "count_numeric", "mean", "min", "max", "last", "keys_seen"])

    commands = []
    log_metrics = []
    for p in files:
        if p.suffix.lower() in {".sh", ".py", ".log", ".md"}:
            commands.extend(extract_commands(p))
        if p.suffix.lower() == ".log":
            log_metrics.extend(extract_log_metrics(p))
    for sroot in SCRIPT_FILES:
        if sroot.is_file():
            commands.extend(extract_commands(sroot))
        elif sroot.exists():
            for p in sroot.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".sh", ".py", ".md"}:
                    commands.extend(extract_commands(p))
    write_csv(TABLES / "commands_extracted.csv", commands, ["source", "line", "command_or_template", "notes"])
    write_csv(TABLES / "log_metric_lines.csv", log_metrics, ["path", "line", "metric", "value", "line_text"])

    curated = []
    def add(group, experiment, split, metric, value, source, notes="", resolution=""):
        curated.append({
            "group": group,
            "experiment": experiment,
            "split": split,
            "metric": metric,
            "value": value,
            "resolution_or_scope": resolution,
            "source": source,
            "notes": notes,
        })

    base_summary = load_json(PROJECT / "work/kvasir_1pct_anchors/baseline_data/summary.json") or {}
    synfoc = load_json(PROJECT / "work/kvasir_1pct_anchors/synfoc_kvasir_1pct/summary.json") or {}
    c3 = load_json(PROJECT / "work/kvasir_1pct_anchors/c3_test_report.json") or {}
    repro = load_json(PROJECT / "work/kvasir_1pct_anchors/ft_1pct_repro_20260806_results.json") or {}
    hq = load_json(PROJECT / "work/kvasir_1pct_anchors/ft_1pct_plus_hq_pseudo_finite_guard_lr025_eval1_20260806_results.json") or {}
    clinic_proto = load_json(PROJECT / "work/clinicdb_external_kvasir8/protocol/protocol_summary.json") or {}

    for split, count in (base_summary.get("counts") or {}).items():
        add("dataset", "kvasir_1pct_anchors", split, "count", count, "work/kvasir_1pct_anchors/baseline_data/summary.json")
    if synfoc:
        add("baseline", "SynFoC-Kvasir-SAM", "test", "dice", (synfoc.get("test") or {}).get("sam_dice", ""), "work/kvasir_1pct_anchors/synfoc_kvasir_1pct/summary.json")
        add("baseline", "SynFoC-Kvasir-UNet", "test", "dice", (synfoc.get("test") or {}).get("unet_dice", ""), "work/kvasir_1pct_anchors/synfoc_kvasir_1pct/summary.json")
        add("baseline", "SynFoC-Kvasir-SAM", "validation", "best_sam_dice", (synfoc.get("best_validation") or {}).get("sam_dice", ""), "work/kvasir_1pct_anchors/synfoc_kvasir_1pct/summary.json", notes=f"iteration={(synfoc.get('best_validation') or {}).get('sam_iteration','')}")
    for name in ["B_direct", "C_top1", "D_all", "A_within"]:
        d = ((c3.get("test") or {}).get(name) or {})
        if d:
            add("router", f"c3_{name}", "test", "per_target_mean", d.get("per_target_mean", ""), "work/kvasir_1pct_anchors/c3_test_report.json")
            add("router", f"c3_{name}", "test", "oracle", d.get("oracle", ""), "work/kvasir_1pct_anchors/c3_test_report.json")
            add("router", f"c3_{name}", "test", "sel_medoid", d.get("sel_medoid", ""), "work/kvasir_1pct_anchors/c3_test_report.json")
    for label, data, source in [
        ("ft_1pct_repro_20260806", repro, "work/kvasir_1pct_anchors/ft_1pct_repro_20260806_results.json"),
        ("ft_1pct_plus_hq_pseudo_lr025_eval1", hq, "work/kvasir_1pct_anchors/ft_1pct_plus_hq_pseudo_finite_guard_lr025_eval1_20260806_results.json"),
    ]:
        if data:
            for key in ["max_bridge_5", "max_bridge_6"]:
                d = ((data.get("max6_summary") or {}).get(key) or {})
                if d:
                    add("main_or_legacy_kvasir", label, "test", f"{key}.dice_mean", d.get("dice_mean", ""), source)
            d = (((data.get("route3_summary") or {}).get("test_pool0")) or {})
            if d:
                add("main_or_legacy_kvasir", label, "test_pool0", "dice_mean", d.get("dice_mean", ""), source, notes=f"n={d.get('n','')}; quality_mean={d.get('quality_mean','')}")
    if clinic_proto:
        for key in ["support_count", "target_count", "bridge_pool_count"]:
            add("clinicdb_external", "clinicdb_external_kvasir8", "", key, clinic_proto.get(key, ""), "work/clinicdb_external_kvasir8/protocol/protocol_summary.json")

    # Add every route dice as curated too, because many Kvasir ablations are route-family experiments.
    for r in route_metric:
        add("route_or_bridge", r["root"], "", r["family_or_bridge"] + ".dice", r["dice"], r["path"], resolution="varies; see source path")

    write_csv(TABLES / "curated_metric_ledger_long.csv", curated, ["group", "experiment", "split", "metric", "value", "resolution_or_scope", "source", "notes"])

    best_routes = sorted(
        [r for r in route_metric if isinstance(r["dice"], float)],
        key=lambda r: r["dice"],
        reverse=True,
    )[:80]
    write_csv(TABLES / "top_route_family_dice.csv", best_routes, ["path", "root", "family_or_bridge", "dice", "kind"])

    aggregate_routes = [
        r for r in route_metric
        if isinstance(r["dice"], float)
        and "::" not in r["family_or_bridge"]
        and (
            r["path"].endswith("route_family_summary.json")
            or "/propagation_quality_" in r["path"]
        )
    ]
    best_aggregate_routes = sorted(aggregate_routes, key=lambda r: r["dice"], reverse=True)[:120]
    write_csv(TABLES / "top_aggregate_route_family_dice.csv", best_aggregate_routes, ["path", "root", "family_or_bridge", "dice", "kind"])

    command_manual = [
        {
            "scope": "Kvasir root",
            "command_or_entry": "work/kvasir_1pct_anchors",
            "status": "existing experiment root",
            "output": "paper_kvasir_experiment_package",
            "notes": "Main Kvasir 1pct anchor experiments.",
        },
        {
            "scope": "ClinicDB external Kvasir8",
            "command_or_entry": "work/clinicdb_external_kvasir8",
            "status": "existing external evaluation root",
            "output": "work/clinicdb_external_kvasir8",
            "notes": "External ClinicDB evaluation using Kvasir anchors/support.",
        },
        {
            "scope": "SynFoC baseline",
            "command_or_entry": "see work/kvasir_1pct_anchors/logs/kvasir_1pct_synfoc_baseline.log and work/kvasir_1pct_anchors/synfoc_kvasir_1pct/protocol.json",
            "status": "complete",
            "output": "work/kvasir_1pct_anchors/synfoc_kvasir_1pct/summary.json",
            "notes": "SAM test Dice and UNet test Dice are in curated_metric_ledger_long.csv.",
        },
        {
            "scope": "SCSAM baseline",
            "command_or_entry": "see work/kvasir_1pct_anchors/logs/kvasir_1pct_scsam_baseline.log",
            "status": "existing logs",
            "output": "work/kvasir_1pct_anchors/scsam_kvasir_1pct",
            "notes": "Exact metric lines, if present, are extracted into log_metric_lines.csv.",
        },
        {
            "scope": "LoRA p491/e20 route eval",
            "command_or_entry": "see work/kvasir_1pct_anchors/lora_experiment/* and stage1_feature_knn_vitb256/*/propagation_quality_*_lora_p491_e20",
            "status": "complete artifacts found",
            "output": "route_family_and_bridge_dice.csv",
            "notes": "Route-family Dice table includes all discovered bridge/direct rows.",
        },
    ]
    write_csv(TABLES / "manual_experiment_step_and_command_map.csv", command_manual, ["scope", "command_or_entry", "status", "output", "notes"])

    # Copy a few compact records.
    for src in [
        PROJECT / "work/kvasir_1pct_anchors/baseline_data/summary.json",
        PROJECT / "work/kvasir_1pct_anchors/synfoc_kvasir_1pct/summary.json",
        PROJECT / "work/kvasir_1pct_anchors/c3_test_report.json",
        PROJECT / "work/kvasir_1pct_anchors/ft_1pct_repro_20260806_results.json",
        PROJECT / "work/kvasir_1pct_anchors/ft_1pct_plus_hq_pseudo_finite_guard_lr025_eval1_20260806_results.json",
        PROJECT / "work/clinicdb_external_kvasir8/protocol/protocol_summary.json",
    ]:
        if src.exists():
            shutil.copy2(src, RAW / (rel(src).replace("/", "__")))

    split = base_summary.get("counts") or {}
    top_md_rows = [[r["root"], r["family_or_bridge"], fmt(r["dice"])] for r in best_aggregate_routes[:20]]
    curated_head = []
    for r in curated:
        if r["group"] in {"dataset", "baseline", "main_or_legacy_kvasir", "router", "clinicdb_external"}:
            curated_head.append([r["group"], r["experiment"], r["split"], r["metric"], fmt(r["value"])])
        if len(curated_head) >= 40:
            break

    readme = f"""# Kvasir Experiment Package

This folder consolidates Kvasir-related experiments from:

- `work/kvasir_1pct_anchors`
- `work/clinicdb_external_kvasir8`

It mirrors the ISIC organization, but Kvasir has more exploratory router, prompt, LoRA, and external-evaluation branches.

## Dataset Snapshot

- train: `{split.get('train', '')}`
- validation: `{split.get('validation', '')}`
- test: `{split.get('test', '')}`

## Main Tables

- `tables/curated_metric_ledger_long.csv`: curated ledger covering dataset counts, SynFoC, C3/router summaries, known finetune/repro summaries, ClinicDB external setup, and all route-family Dice rows.
- `tables/route_family_and_bridge_dice.csv`: direct/bridge Dice extracted from every discovered route-family or propagation summary JSON.
- `tables/top_aggregate_route_family_dice.csv`: top aggregate route-family Dice rows, sorted descending. Start here for paper tables.
- `tables/top_route_family_dice.csv`: top route-family Dice rows including per-target selector details, useful for auditing but not directly paper-ready.
- `tables/all_json_scalar_metrics_long.csv`: all scalar metrics from every JSON.
- `tables/all_jsonl_numeric_summaries.csv`: JSONL record counts and numeric summaries.
- `tables/log_metric_lines.csv`: metric-looking lines extracted from logs.
- `tables/commands_extracted.csv`: command lines/templates extracted from `.sh`, `.py`, `.log`, and `.md`.
- `tables/manual_experiment_step_and_command_map.csv`: human-readable command/source map.
- `tables/all_kvasir_artifacts.csv`: complete artifact inventory.

## Curated Highlights

{md_table(["Group", "Experiment", "Split", "Metric", "Value"], curated_head)}

## Top Aggregate Route/Bridge Dice Rows

{md_table(["Root", "Family/Bridge", "Dice"], top_md_rows)}

## Notes

The package is intentionally exhaustive. For paper tables, start from `curated_metric_ledger_long.csv` and `top_route_family_dice.csv`; for auditing/reproduction, use the all-JSON/all-JSONL/commands/log tables.
"""
    write_text(PKG / "README.md", readme)

    protocol = f"""# Kvasir Protocol And Organization

Primary root:

`work/kvasir_1pct_anchors`

External root:

`work/clinicdb_external_kvasir8`

The primary Kvasir split discovered from `baseline_data/summary.json` is train/validation/test = `{split.get('train','')}`/`{split.get('validation','')}`/`{split.get('test','')}`.

The external ClinicDB root records support/target/bridge-pool counts in `work/clinicdb_external_kvasir8/protocol/protocol_summary.json`.
"""
    write_text(PKG / "01_protocol.md", protocol)

    results = f"""# Kvasir Results Overview

This overview is generated from discovered artifacts. The exhaustive source of truth is the CSV table set under `tables/`.

## Top Aggregate Route/Bridge Results

{md_table(["Root", "Family/Bridge", "Dice"], top_md_rows)}

## Baseline And Router Highlights

{md_table(["Group", "Experiment", "Split", "Metric", "Value"], curated_head)}
"""
    write_text(PKG / "02_results_overview.md", results)

    print(f"WROTE {PKG}")
    print(f"files={len(inventory)} json_scalars={len(json_scalar)} jsonl_summaries={len(jsonl_summaries)} route_rows={len(route_metric)} commands={len(commands)} log_metric_lines={len(log_metrics)} curated={len(curated)}")


if __name__ == "__main__":
    main()
