#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v3_promotion_acceptance import PARTITIONS, SCIENTIFIC_FIELDS
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _payload(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((root / "cases").glob("*.json")):
        record = _load(path)
        records.append({field: record[field] for field in SCIENTIFIC_FIELDS})
    return sorted(records, key=lambda row: row["case_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two CAGE-v3 promotion acceptance repeats")
    parser.add_argument("--repeat-a", type=Path, required=True)
    parser.add_argument("--repeat-b", type=Path, required=True)
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {output}")
    roots = (args.repeat_a.resolve(), args.repeat_b.resolve())
    expected = 12 if args.partition == "cage_qwen3" else 3
    summaries = [_load(root / "summary.json") for root in roots]
    for summary in summaries:
        checks = (
            summary.get("status") == "pass",
            summary.get("claim_eligible") is False,
            summary.get("partition") == args.partition,
            summary.get("stage") == "promotion_acceptance",
            summary.get("expected_cases") == expected,
            summary.get("completed_cases") == expected,
            summary.get("failure_records") == 0,
            summary.get("full_holdout_authorized") is False,
            summary.get("pg19_test_accessed") is False,
        )
        if not all(checks):
            raise ValueError("promotion acceptance repeat summary mismatch")
    payloads = [_payload(root) for root in roots]
    if any(len(payload) != expected for payload in payloads):
        raise ValueError("promotion acceptance repeat case count mismatch")
    first = {row["case_id"]: row for row in payloads[0]}
    second = {row["case_id"]: row for row in payloads[1]}
    mismatch = sorted(
        case_id for case_id in set(first) | set(second) if first.get(case_id) != second.get(case_id)
    )
    report = {
        "schema_version": 1,
        "status": "pass" if not mismatch else "fail",
        "claim_eligible": False,
        "partition": args.partition,
        "case_count": expected,
        "repeat_a": str(roots[0]),
        "repeat_b": str(roots[1]),
        "required_consistency": "bitwise_equal_json_numeric_payload",
        "scientific_fields": list(SCIENTIFIC_FIELDS),
        "mismatch_case_ids": mismatch,
        "scientific_payload_sha256": canonical_sha256(payloads[0]) if not mismatch else None,
        "comparator_sha256": file_sha256(Path(__file__).resolve()),
        "full_holdout_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
