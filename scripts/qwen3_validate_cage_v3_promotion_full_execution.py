#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v3_promotion_full import (
    EXPECTED_CASE_ID_HASHES,
    expand_full_holdout_cases,
    load_full_execution,
)
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256, load_data_protocol, validate_input_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen CAGE-v3 promotion full-holdout execution")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execution, execution_sha256, protocol, protocol_sha256, gate, gate_sha256 = load_full_execution(
        args.execution.resolve(),
        repo_root=REPO_ROOT,
        verify_server_artifacts=True,
    )
    manifest_path = args.input_manifest.resolve()
    if str(manifest_path) != execution["input_manifest"]["path"] or file_sha256(manifest_path) != execution["input_manifest"]["sha256"]:
        raise RuntimeError("promotion full input manifest differs from execution receipt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = REPO_ROOT / protocol["input_receipt"]["data_protocol_path"]
    data_protocol, data_sha256 = load_data_protocol(data_path)
    if data_sha256 != protocol["input_receipt"]["data_protocol_sha256"]:
        raise RuntimeError("promotion full data protocol mismatch")
    validate_input_manifest(manifest, protocol=data_protocol, protocol_sha256=data_sha256)
    expanded = {
        partition: expand_full_holdout_cases(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            gate_sha256=gate_sha256,
            manifest=manifest,
            partition=partition,
            repo_root=REPO_ROOT,
        )
        for partition in ("cage_qwen3", "kitty_qwen3")
    }
    for partition, cases in expanded.items():
        if canonical_sha256([case["case_id"] for case in cases]) != EXPECTED_CASE_ID_HASHES[partition]:
            raise RuntimeError(f"{partition} promotion full case IDs differ from execution receipt")
    report = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "execution_id": execution["execution_id"],
        "execution_sha256": execution_sha256,
        "gate_sha256": gate_sha256,
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": file_sha256(manifest_path),
        "partitions": {
            partition: {
                "case_count": len(cases),
                "case_ids_sha256": canonical_sha256([case["case_id"] for case in cases]),
                "method_counts": dict(Counter(case["method"]["metric_method_id"] for case in cases)),
            }
            for partition, cases in expanded.items()
        },
        "authorization": execution["authorization"],
        "holdout_method_metrics_read": False,
        "pg19_test_accessed": False,
        "interpretation_performed": False,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite promotion full execution preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
