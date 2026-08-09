#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_perturbation_protocol import file_sha256


COMPARE_FIELDS = (
    "case_id",
    "method",
    "input",
    "memory",
    "layer_metrics",
    "aggregates",
    "cache",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two fresh CAGE-v2 round1 acceptances")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--partition", choices=("cage_qwen3", "kitty_qwen3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _records(root: Path, partition: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary = _load(root / "summary.json")
    if summary.get("status") != "pass" or summary.get("stage") != "acceptance":
        raise RuntimeError(f"acceptance summary is not complete: {root}")
    if summary.get("partition") != partition or summary.get("claim_eligible") is not False:
        raise RuntimeError(f"acceptance summary partition/claim boundary mismatch: {root}")
    if summary.get("failure_records") != 0:
        raise RuntimeError(f"acceptance contains failures: {root}")
    cases = {}
    for path in sorted((root / "cases").glob("*.json")):
        record = _load(path)
        if record.get("status") != "completed":
            raise RuntimeError(f"acceptance case is incomplete: {path}")
        cases[record["case_id"]] = record
    if len(cases) != summary.get("completed_cases"):
        raise RuntimeError(f"acceptance case count mismatch: {root}")
    return summary, cases


def main() -> None:
    args = _parse_args()
    first_root = args.first.resolve()
    second_root = args.second.resolve()
    if first_root == second_root:
        raise RuntimeError("acceptance directories must be distinct fresh outputs")
    first_summary, first_cases = _records(first_root, args.partition)
    second_summary, second_cases = _records(second_root, args.partition)
    if first_summary["identity"] != second_summary["identity"]:
        raise RuntimeError("fresh acceptance run identities differ")
    if set(first_cases) != set(second_cases):
        raise RuntimeError("fresh acceptance case IDs differ")

    mismatches = []
    payload = []
    for case_id in sorted(first_cases):
        left = {field: first_cases[case_id][field] for field in COMPARE_FIELDS}
        right = {field: second_cases[case_id][field] for field in COMPARE_FIELDS}
        if left != right:
            mismatches.append(case_id)
        payload.append(left)
    report = {
        "schema_version": 1,
        "status": "pass" if not mismatches else "fail",
        "claim_eligible": False,
        "partition": args.partition,
        "case_count": len(payload),
        "compare_fields": list(COMPARE_FIELDS),
        "required_consistency": "bitwise_equal_json_numeric_payload",
        "mismatch_case_ids": mismatches,
        "scientific_payload_sha256": _canonical_sha256(payload),
        "first": {
            "path": str(first_root),
            "summary_sha256": file_sha256(first_root / "summary.json"),
        },
        "second": {
            "path": str(second_root),
            "summary_sha256": file_sha256(second_root / "summary.json"),
        },
        "comparator_sha256": file_sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if mismatches:
        raise RuntimeError(f"round1 acceptance mismatches: {mismatches}")


if __name__ == "__main__":
    main()
