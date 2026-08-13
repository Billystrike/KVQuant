#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256
from utils.qwen3_cage_v4_dtqi_acceptance import SCIENTIFIC_FIELDS, STAGE, load_json


def _payload(root: Path) -> tuple[dict, list[dict]]:
    summary = load_json(root / "summary.json")
    expected_summary = (
        summary.get("schema_version") == 1,
        summary.get("status") == "pass",
        summary.get("claim_eligible") is False,
        summary.get("stage") == STAGE,
        summary.get("expected_cases") == 3,
        summary.get("completed_cases") == 3,
        summary.get("new_cases") == 3,
        summary.get("resumed_cases") == 0,
        summary.get("failure_records") == 0,
        summary.get("full_screen_authorized") is False,
        summary.get("holdout_accessed") is False,
        summary.get("pg19_test_accessed") is False,
    )
    if not all(expected_summary):
        raise RuntimeError(f"incomplete DTQI acceptance repeat: {root}")
    records = []
    for path in sorted((root / "cases").glob("*.json")):
        value = load_json(path)
        if value.get("status") != "completed" or path.stem != value.get("case_id"):
            raise RuntimeError(f"invalid DTQI acceptance case: {path}")
        records.append({field: value[field] for field in SCIENTIFIC_FIELDS})
    if len(records) != 3:
        raise RuntimeError(f"DTQI acceptance case count mismatch: {root}")
    records.sort(key=lambda record: record["case_id"])
    return summary, records


def compare_repeats(first_root: Path, second_root: Path) -> dict:
    if first_root.resolve() == second_root.resolve():
        raise RuntimeError("DTQI acceptance repeat directories must differ")
    first_summary, first = _payload(first_root.resolve())
    second_summary, second = _payload(second_root.resolve())
    if first_summary["identity"] != second_summary["identity"]:
        raise RuntimeError("DTQI acceptance execution identities differ")
    if first_summary["model"] != second_summary["model"]:
        raise RuntimeError("DTQI acceptance model identities differ")
    first_by_id = {record["case_id"]: record for record in first}
    second_by_id = {record["case_id"]: record for record in second}
    mismatches = sorted(
        case_id
        for case_id in set(first_by_id) | set(second_by_id)
        if first_by_id.get(case_id) != second_by_id.get(case_id)
    )
    return {
        "schema_version": 1,
        "status": "pass" if not mismatches else "fail",
        "claim_eligible": False,
        "stage": "gpu_acceptance_repeat_comparison",
        "case_count": 3,
        "fields": list(SCIENTIFIC_FIELDS),
        "required_consistency": "bitwise_equal_json_numeric_payload",
        "mismatch_case_ids": mismatches,
        "scientific_payload_sha256": canonical_sha256(first),
        "comparator_sha256": file_sha256(Path(__file__).resolve()),
        "repeat_a": str(first_root.resolve()),
        "repeat_b": str(second_root.resolve()),
        "full_screen_authorized": False,
        "holdout_accessed": False,
        "pg19_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fresh DTQI GPU acceptance repeats")
    parser.add_argument("--repeat-a", type=Path, required=True)
    parser.add_argument("--repeat-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_repeats(args.repeat_a, args.repeat_b)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DTQI acceptance comparison: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["status"] != "pass":
        raise RuntimeError(f"DTQI acceptance mismatches: {result['mismatch_case_ids']}")


if __name__ == "__main__":
    main()
