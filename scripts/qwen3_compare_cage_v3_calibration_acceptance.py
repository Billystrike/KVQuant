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

from utils.qwen3_cage_v3_execution import SCIENTIFIC_FIELDS, canonical_sha256, load_json
from utils.qwen3_cage_v3_protocol import file_sha256


def _records(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary = load_json(root / "summary.json")
    if summary.get("status") != "pass" or summary.get("stage") != "calibration_acceptance":
        raise RuntimeError(f"CAGE-v3 calibration acceptance is incomplete: {root}")
    if summary.get("claim_eligible") is not False or summary.get("failure_records") != 0:
        raise RuntimeError(f"CAGE-v3 calibration acceptance boundary/failures mismatch: {root}")
    cases = {}
    for path in sorted((root / "cases").glob("*.json")):
        record = load_json(path)
        if record.get("status") != "completed":
            raise RuntimeError(f"incomplete CAGE-v3 acceptance case: {path}")
        cases[record["case_id"]] = record
    if len(cases) != summary.get("completed_cases") or len(cases) != 6:
        raise RuntimeError(f"CAGE-v3 acceptance case count mismatch: {root}")
    return summary, cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two fresh CAGE-v3 calibration acceptances")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first_root = args.first.resolve()
    second_root = args.second.resolve()
    if first_root == second_root:
        raise RuntimeError("CAGE-v3 acceptance directories must be distinct")
    first_summary, first_cases = _records(first_root)
    second_summary, second_cases = _records(second_root)
    if first_summary["identity"] != second_summary["identity"]:
        raise RuntimeError("CAGE-v3 fresh acceptance identities differ")
    if set(first_cases) != set(second_cases):
        raise RuntimeError("CAGE-v3 fresh acceptance case IDs differ")
    mismatches = []
    payload = []
    for case_id in sorted(first_cases):
        left = {field: first_cases[case_id][field] for field in SCIENTIFIC_FIELDS}
        right = {field: second_cases[case_id][field] for field in SCIENTIFIC_FIELDS}
        if left != right:
            mismatches.append(case_id)
        payload.append(left)
    report = {
        "schema_version": 1,
        "status": "pass" if not mismatches else "fail",
        "claim_eligible": False,
        "stage": "calibration_acceptance_comparison",
        "case_count": len(payload),
        "compare_fields": list(SCIENTIFIC_FIELDS),
        "required_consistency": "bitwise_equal_json_numeric_payload",
        "mismatch_case_ids": mismatches,
        "scientific_payload_sha256": canonical_sha256(payload),
        "first": {"path": str(first_root), "summary_sha256": file_sha256(first_root / "summary.json")},
        "second": {"path": str(second_root), "summary_sha256": file_sha256(second_root / "summary.json")},
        "comparator_sha256": file_sha256(Path(__file__).resolve()),
    }
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite comparison: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if mismatches:
        raise RuntimeError(f"CAGE-v3 calibration acceptance mismatches: {mismatches}")


if __name__ == "__main__":
    main()
