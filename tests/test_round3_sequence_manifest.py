import json
from collections import Counter
from pathlib import Path

ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")

def test_round3_sequence_manifest_schema_and_balance():
    path = ROOT / "work/rerun_c0_256_round3_tracker_stage4/sequence_manifests/train_sequences.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    assert len(rows) == 524 * 7 * 2
    modes = Counter(row["mode"] for row in rows)
    bridges = Counter(row["bridge"] for row in rows)
    assert len(modes) == 2 and len(set(modes.values())) == 1
    assert set(bridges) == set(range(7)) and len(set(bridges.values())) == 1
    for row in rows:
        assert len(row["frames"]) == row["bridge"] + 2
        assert row["supervision"][0]["source"] == "gt_anchor"
        assert row["supervision"][-1]["source"] == "frozen_pseudo"
        assert row["supervision"][-1]["frame_index"] == len(row["frames"]) - 1
        assert all(item["source"] in {"gt_anchor", "frozen_pseudo"} for item in row["supervision"])
