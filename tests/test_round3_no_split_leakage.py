import json
from pathlib import Path

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def test_round3_no_split_leakage():
    manifest = rows(ROOT / "work/rerun_c0_256_round3_tracker_stage4/sequence_manifests/train_sequences.jsonl")
    protocol = rows(ROOT / "work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl")
    by_split = {split: {x["merged_id"] for x in protocol if x["split"] == split} for split in ("train", "validation", "test")}
    train_sequence_ids = {frame_id for row in manifest for frame_id in row["frame_ids"]}
    assert train_sequence_ids <= by_split["train"]
    assert not train_sequence_ids & by_split["validation"]
    assert not train_sequence_ids & by_split["test"]
