from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_fixed_protocol_counts() -> None:
    rows = read_jsonl(ROOT / "protocols/reproduction_v1/splits.jsonl")
    assert len(rows) == 1612
    assert Counter(row["split"] for row in rows) == {
        "train": 1290,
        "validation": 161,
        "test": 161,
    }


def test_support_ids_are_fixed_train_balanced() -> None:
    support = read_jsonl(ROOT / "protocols/reproduction_v1/support_ids.jsonl")
    assert len(support) == 16
    assert len({row["merged_id"] for row in support}) == 16
    assert {row["split"] for row in support} == {"train"}
    assert Counter(row["source_dataset"] for row in support) == {
        "CVC-ClinicDB": 8,
        "kvasir-seg": 8,
    }
