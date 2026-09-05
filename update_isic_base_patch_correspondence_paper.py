import csv
import json
from pathlib import Path


PROJECT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
PKG = PROJECT / "paper_isic2018_experiment_package"
TABLES = PKG / "tables"
COMPLETE = PKG / "complete_experiment_tables" / "tables"
BASE_JSON = PROJECT / "work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/base_test_bridge_b0_b6.json"
BASE_TSV = PROJECT / "work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/base_test_bridge_b0_b6.tsv"


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def replace_between(text, start, end, replacement):
    if start in text and end in text:
        a = text.index(start)
        b = text.index(end, a)
        return text[:a] + replacement + "\n\n" + text[b:]
    return text.rstrip() + "\n\n" + replacement + "\n"


data = json.loads(BASE_JSON.read_text())
modes = data["modes"]
target = modes["sam3enc_anchor_conditioned_target_pooling"]
patch = modes["sam3enc_anchor_conditioned_patch_correspondence"]
combined = data["combined"]

rows = []
for mode_name, vals in [
    ("target_pooling", target),
    ("patch_correspondence", patch),
    ("combined", combined),
]:
    for i in range(7):
        k = f"bridge_{i}"
        rows.append({
            "mode": mode_name,
            "bridge": f"b{i}",
            "test_dice": vals[k]["dice"],
            "median_dice": vals[k].get("dice_median", ""),
            "n": vals[k].get("n", ""),
            "q_cycle": vals[k].get("q_cycle", ""),
            "source": str(BASE_TSV.relative_to(PROJECT)),
        })

patch_rows = [r for r in rows if r["mode"] == "patch_correspondence"]
write_csv(TABLES / "base_sam3_knn_patch_correspondence_b0_b6.csv", patch_rows, ["mode", "bridge", "test_dice", "median_dice", "n", "q_cycle", "source"])
write_csv(TABLES / "base_sam3_knn_two_modes_b0_b6.csv", rows, ["mode", "bridge", "test_dice", "median_dice", "n", "q_cycle", "source"])

if COMPLETE.exists():
    write_csv(COMPLETE / "base_sam3_knn_two_modes_b0_b6.csv", rows, ["mode", "bridge", "test_dice", "median_dice", "n", "q_cycle", "source"])
    ledger = read_csv(COMPLETE / "final_metric_ledger_long.csv")
    ledger = [
        r for r in ledger
        if not (
            r.get("group") == "base_sam3_knn"
            and r.get("experiment") in {"target_pooling", "patch_correspondence", "combined"}
        )
    ]
    for r in rows:
        ledger.append({
            "group": "base_sam3_knn",
            "experiment": r["mode"],
            "split": "test",
            "metric": f"{r['bridge']}.dice",
            "value": r["test_dice"],
            "resolution_or_scope": "256; base SAM3; no LoRA",
            "source": str(BASE_JSON.relative_to(PROJECT)),
            "notes": f"n={r['n']}; median_dice={r['median_dice']}; q_cycle={r['q_cycle']}",
        })
    write_csv_rows(COMPLETE / "final_metric_ledger_long.csv", ledger, ["group", "experiment", "split", "metric", "value", "resolution_or_scope", "source", "notes"])

table_lines = [
    "| Mode | b0 | b1 | b2 | b3 | b4 | b5 | b6 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for mode_name, vals in [
    ("target_pooling", target),
    ("patch_correspondence", patch),
    ("combined", combined),
]:
    table_lines.append(
        "| "
        + mode_name
        + " | "
        + " | ".join(f"{vals[f'bridge_{i}']['dice']:.6f}" for i in range(7))
        + " |"
    )
two_mode_table = "\n".join(table_lines)

section = (
    "### Base SAM3-KNN Test, Target Pooling And Patch Correspondence\n\n"
    + two_mode_table
    + "\n\n"
    + "Source: `work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/base_test_bridge_b0_b6.json` and `.tsv`. "
    + "These are base SAM3, no-LoRA, 256-canvas test results. Patch Correspondence was already present in the original base test summary; it had simply not been split into a paper table before this update."
)

bridge_md = PKG / "04_bridge_analysis.md"
text = bridge_md.read_text(encoding="utf-8")
old_start = "## Base SAM3-KNN Target Pooling, b0-b6"
old_end = "## Epoch27 LoRA Long-Chain Test, b0-b7"
replacement = (
    "## Base SAM3-KNN Test, b0-b6\n\n"
    + two_mode_table
    + "\n\n"
    + "The Patch Correspondence row is now included explicitly. The combined row averages both modes as in `base_test_bridge_b0_b6.tsv`.\n\n"
)
if old_start in text and old_end in text:
    a = text.index(old_start)
    b = text.index(old_end, a)
    text = text[:a] + replacement + text[b:]
elif "## Base SAM3-KNN Test, b0-b6" not in text:
    text = text.rstrip() + "\n\n" + replacement
bridge_md.write_text(text, encoding="utf-8", newline="\n")

readme = PKG / "README.md"
txt = readme.read_text(encoding="utf-8")
note = "- `tables/base_sam3_knn_patch_correspondence_b0_b6.csv`: base SAM3 Patch Correspondence test Dice, b0-b6.\n- `tables/base_sam3_knn_two_modes_b0_b6.csv`: base SAM3 Target Pooling, Patch Correspondence, and combined test Dice, b0-b6."
if "base_sam3_knn_patch_correspondence_b0_b6.csv" not in txt:
    txt = txt.replace("- `tables/`: CSV tables for manuscript or spreadsheet import.", "- `tables/`: CSV tables for manuscript or spreadsheet import.\n" + note)
readme.write_text(txt, encoding="utf-8", newline="\n")

overview = PKG / "02_main_results.md"
if overview.exists():
    txt = overview.read_text(encoding="utf-8")
    if "Base SAM3-KNN Test, Target Pooling And Patch Correspondence" not in txt:
        txt = txt.rstrip() + "\n\n" + section + "\n"
    overview.write_text(txt, encoding="utf-8", newline="\n")

print("UPDATED")
print(TABLES / "base_sam3_knn_patch_correspondence_b0_b6.csv")
print(TABLES / "base_sam3_knn_two_modes_b0_b6.csv")
