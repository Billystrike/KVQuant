#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v3_promotion_acceptance import (
    PARTITIONS,
    expand_acceptance_cases,
    load_acceptance_execution,
)
from utils.qwen3_cage_v4_data import (
    canonical_sha256,
    file_sha256,
    load_data_protocol,
    validate_input_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CAGE-v3 promotion acceptance execution")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    args = parser.parse_args()
    execution, execution_sha256, protocol, protocol_sha256 = load_acceptance_execution(
        args.execution.resolve(), repo_root=REPO_ROOT, verify_server_artifacts=True
    )
    manifest_path = args.input_manifest.resolve()
    if str(manifest_path) != execution["input_manifest"]["path"]:
        raise RuntimeError("input manifest path differs from frozen execution")
    if file_sha256(manifest_path) != execution["input_manifest"]["sha256"]:
        raise RuntimeError("input manifest hash differs from frozen execution")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = REPO_ROOT / protocol["input_receipt"]["data_protocol_path"]
    data_protocol, data_sha256 = load_data_protocol(data_path)
    if data_sha256 != protocol["input_receipt"]["data_protocol_sha256"]:
        raise RuntimeError("PG-19 data protocol hash mismatch")
    validate_input_manifest(manifest, protocol=data_protocol, protocol_sha256=data_sha256)
    expanded = {
        partition: expand_acceptance_cases(
            execution=execution,
            execution_sha256=execution_sha256,
            protocol=protocol,
            manifest=manifest,
            partition=partition,
            repo_root=REPO_ROOT,
        )
        for partition in PARTITIONS
    }
    report = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "execution_id": execution["execution_id"],
        "execution_sha256": execution_sha256,
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": file_sha256(manifest_path),
        "static_preflight_sha256": execution["static_preflight"]["output_sha256"],
        "partitions": {
            partition: {
                "case_count_per_repeat": len(cases),
                "case_ids": [case["case_id"] for case in cases],
                "case_ids_sha256": canonical_sha256([case["case_id"] for case in cases]),
                "method_ids": [case["method"]["metric_method_id"] for case in cases],
            }
            for partition, cases in expanded.items()
        },
        "authorization": execution["authorization"],
        "full_holdout_authorized": False,
        "pg19_test_accessed": False,
        "holdout_method_metrics_read": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
