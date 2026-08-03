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

from utils.qwen3_formal import Qwen3FormalError, atomic_write_json, file_sha256


PARTITION = "kitty_qwen3"
EXPECTED_CASES = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two fresh Qwen3 Kitty-partition acceptance runs"
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3FormalError(f"cannot load acceptance JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Qwen3FormalError(f"acceptance JSON must be an object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scientific_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "partition": record.get("partition"),
        "method": record.get("method"),
        "input": record.get("input"),
        "scoring": record.get("scoring"),
        "cache": record.get("cache"),
    }


def main() -> None:
    args = _parse_args()
    left = args.left.resolve()
    right = args.right.resolve()
    if left == right:
        raise Qwen3FormalError("acceptance repeat directories must be different")
    left_summary = _load(left / "summary.json")
    right_summary = _load(right / "summary.json")
    expected = {
        "status": "pass",
        "partition": PARTITION,
        "stage": "acceptance",
        "expected_cases": EXPECTED_CASES,
        "completed_cases": EXPECTED_CASES,
        "failure_records": 0,
    }
    for name, summary in (("left", left_summary), ("right", right_summary)):
        if any(summary.get(field) != value for field, value in expected.items()):
            raise Qwen3FormalError(f"{name} acceptance summary failed its completion gate")
    if left_summary["identity"] != right_summary["identity"]:
        raise Qwen3FormalError("acceptance repeats have different execution identities")
    left_lock = _load(left / "run_identity.json")
    right_lock = _load(right / "run_identity.json")
    if left_lock != right_lock:
        raise Qwen3FormalError("acceptance repeats have different run locks")
    case_ids = left_lock.get("expected_case_ids")
    if not isinstance(case_ids, list) or len(case_ids) != EXPECTED_CASES:
        raise Qwen3FormalError(
            f"acceptance run lock must contain {EXPECTED_CASES} case IDs"
        )

    mismatches = []
    payloads = []
    for case_id in case_ids:
        left_record = _load(left / "cases" / f"{case_id}.json")
        right_record = _load(right / "cases" / f"{case_id}.json")
        left_payload = _scientific_payload(left_record)
        right_payload = _scientific_payload(right_record)
        left_digest = _canonical_sha256(left_payload)
        right_digest = _canonical_sha256(right_payload)
        if left_digest != right_digest:
            mismatches.append(
                {
                    "case_id": case_id,
                    "left_scientific_sha256": left_digest,
                    "right_scientific_sha256": right_digest,
                }
            )
        payloads.append(left_payload)
    report = {
        "schema_version": 1,
        "status": "pass" if not mismatches else "fail",
        "partition": PARTITION,
        "stage": "acceptance_repeat_comparison",
        "case_count": len(case_ids),
        "bitwise_equal_scientific_cases": len(case_ids) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "scientific_payload_sha256": _canonical_sha256(payloads),
        "execution_identity": left_summary["identity"],
        "left": str(left),
        "right": str(right),
        "left_summary_sha256": file_sha256(left / "summary.json"),
        "right_summary_sha256": file_sha256(right / "summary.json"),
    }
    output = args.output.resolve()
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"acceptance_comparison_json: {output}")
    print(f"acceptance_comparison_json_sha256: {file_sha256(output)}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
