#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llama2_cage_v3_transfer_quality_acceptance import (
    SCIENTIFIC_FIELDS,
    expand_acceptance_cases,
    load_execution,
    load_json,
    load_server_input_manifest,
    require,
    validate_completed_case,
)
from utils.llama2_cage_v3_transfer_quality_protocol import load_transfer_quality_protocol
from utils.qwen3_cage_v4_data import canonical_sha256


def _write_fresh(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite acceptance comparison: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two frozen Llama-2 transfer-quality acceptance repeats")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--repeat-a", type=Path, required=True)
    parser.add_argument("--repeat-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execution, execution_sha = load_execution(args.execution.resolve(), repo_root=REPO_ROOT)
    protocol, _ = load_transfer_quality_protocol(REPO_ROOT / execution["protocol"]["path"], repo_root=REPO_ROOT)
    manifest = load_server_input_manifest(execution, protocol=protocol)
    expected = expand_acceptance_cases(execution=execution, execution_sha256=execution_sha, protocol=protocol, input_manifest=manifest)
    payloads = {}
    for repeat, root in (("a", args.repeat_a.resolve()), ("b", args.repeat_b.resolve())):
        summary = load_json(root / "summary.json")
        require(summary.get("status") == "pass" and summary.get("repeat") == repeat, f"repeat {repeat} summary changed")
        require(summary.get("completed_cases") == 12 and summary.get("failure_records") == 0, f"repeat {repeat} incomplete")
        records = []
        for case in expected:
            record = load_json(root / "cases" / f"{case['case_id']}.json")
            validate_completed_case(record, case)
            records.append({field: record[field] for field in SCIENTIFIC_FIELDS})
        require(canonical_sha256(records) == summary["scientific_payload_sha256"], f"repeat {repeat} summary payload mismatch")
        payloads[repeat] = records
    mismatches = [left["case_id"] for left, right in zip(payloads["a"], payloads["b"]) if left != right]
    report = {
        "schema_version": 1,
        "comparison_id": "llama2-7b-cage-v3-transfer-quality-acceptance-repeat-comparison-v1",
        "status": "pass" if not mismatches else "fail",
        "claim_eligible": False,
        "case_count": 12,
        "required_consistency": "bitwise_equal_json_scientific_payload",
        "scientific_fields": list(SCIENTIFIC_FIELDS),
        "telemetry_excluded": True,
        "mismatch_case_ids": mismatches,
        "scientific_payload_sha256": canonical_sha256(payloads["a"]),
        "full_600_case_execution_authorized": False,
    }
    _write_fresh(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    require(not mismatches, "acceptance repeat scientific payload mismatch")


if __name__ == "__main__":
    main()
