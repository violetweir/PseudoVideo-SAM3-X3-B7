#!/usr/bin/env python3
"""Build a zero-target-domain-supervision ClinicDB transfer protocol.

Bridge nodes and human anchors come only from Kvasir train.  ClinicDB contributes
only its 61 test targets; its train and validation images are not included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
DEFAULT_MERGED = Path(
    "/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/"
    "T17_autonomous_target_1pct/protocol/merged_manifest.jsonl"
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )


def write_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen protocol mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kvasir-protocol",
        type=Path,
        default=ROOT / "work/kvasir_1pct_anchors/protocol",
    )
    parser.add_argument("--merged-polyp-manifest", type=Path, default=DEFAULT_MERGED)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "work/clinicdb_external_kvasir8/protocol",
    )
    args = parser.parse_args()

    kvasir_rows = read_jsonl(args.kvasir_protocol / "merged_manifest.jsonl")
    source_rows = read_jsonl(args.merged_polyp_manifest)
    kvasir_train = [
        row
        for row in kvasir_rows
        if row["source_dataset"] == "kvasir-seg" and row["split"] == "train"
    ]
    clinic_test = [
        row
        for row in source_rows
        if row["source_dataset"] == "CVC-ClinicDB" and row["split"] == "test"
    ]
    support = read_jsonl(args.kvasir_protocol / "support_manifest.jsonl")
    if len(kvasir_train) != 800 or len(clinic_test) != 61 or len(support) != 8:
        raise RuntimeError(
            f"Unexpected train/test/support counts: {len(kvasir_train)}/{len(clinic_test)}/{len(support)}"
        )
    train_ids = {row["merged_id"] for row in kvasir_train}
    if any(row["merged_id"] not in train_ids for row in support):
        raise RuntimeError("A support anchor is not in Kvasir train")
    if any(row["source_dataset"] != "kvasir-seg" for row in support):
        raise RuntimeError("Protocol contains a non-Kvasir support anchor")

    records = sorted(kvasir_train + clinic_test, key=lambda row: row["merged_id"])
    records_payload = jsonl_bytes(records)
    support_payload = jsonl_bytes(sorted(support, key=lambda row: row["merged_id"]))
    write_or_verify(args.output_root / "merged_manifest.jsonl", records_payload)
    write_or_verify(args.output_root / "support_manifest.jsonl", support_payload)
    summary = {
        "name": "ClinicDB external transfer with frozen Kvasir-8 support",
        "bridge_pool": "Kvasir-SEG train only",
        "bridge_pool_count": len(kvasir_train),
        "support_source": "Kvasir-SEG train only",
        "support_count": len(support),
        "target_source": "CVC-ClinicDB test only",
        "target_count": len(clinic_test),
        "clinicdb_train_or_validation_included": False,
        "target_gt_role": "evaluation_only",
        "records_sha256": hashlib.sha256(records_payload).hexdigest(),
        "support_sha256": hashlib.sha256(support_payload).hexdigest(),
    }
    summary_payload = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_or_verify(args.output_root / "protocol_summary.json", summary_payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("CLINICDB_EXTERNAL_PROTOCOL_DONE")


if __name__ == "__main__":
    main()
