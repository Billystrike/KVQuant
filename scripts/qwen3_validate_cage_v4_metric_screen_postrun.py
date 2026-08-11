#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256
from utils.qwen3_cage_v4_execution import load_screen_execution
from utils.qwen3_cage_v4_postrun import (
    expected_memory_report,
    load_json,
    validate_artifact_manifest,
    validate_partition,
)


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jointly validate the 720-case CAGE-v4 metric screen")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite metric screen postrun audit: {output_path}")

    execution, execution_sha256, _, protocol_sha256, _, expanded = load_screen_execution(
        args.execution.resolve(), repo_root=REPO_ROOT, verify_artifacts=True
    )
    if expanded is None:
        raise RuntimeError("metric screen cases were not expanded")
    artifacts = load_json(args.artifact_manifest.resolve())
    validate_artifact_manifest(
        artifacts,
        execution_sha256=execution_sha256,
        protocol_sha256=protocol_sha256,
        input_manifest_sha256=execution["input_manifest"]["sha256"],
        gate_sha256=execution["acceptance_gate"]["sha256"],
    )

    reports = {}
    joint_payload = []
    for partition in ("cage_qwen3", "kitty_qwen3"):
        expected_cases = [
            {
                "case_id": case["case_id"],
                "method": case["method"],
                "input": case["input"],
                "memory": expected_memory_report(case["method"]),
            }
            for case in expanded[partition]
        ]
        report = validate_partition(
            partition,
            artifacts["partitions"][partition],
            expected_cases=expected_cases,
            execution_source_commit=artifacts["execution_source_commit"],
            execution_sha256=execution_sha256,
            protocol_sha256=protocol_sha256,
            input_manifest_sha256=execution["input_manifest"]["sha256"],
            gate_sha256=execution["acceptance_gate"]["sha256"],
            runner_sha256=artifacts["runner_sha256"],
            accepted_runtime_sha256=artifacts["accepted_runtime_sha256"],
        )
        joint_payload.extend(
            {"partition": partition, **record} for record in report.pop("scientific_payload")
        )
        reports[partition] = report
    joint_payload.sort(key=lambda item: (item["partition"], item["case_id"]))
    audit = {
        "schema_version": 1,
        "status": "pass",
        "audit_id": "qwen3-8b-cage-v4-pg19-metric-screen-postrun-v1",
        "claim_eligible": False,
        "interpretation_performed": False,
        "holdout_accessed": False,
        "cage_v4_candidate_executed": False,
        "execution_sha256": execution_sha256,
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": execution["input_manifest"]["sha256"],
        "acceptance_gate_sha256": execution["acceptance_gate"]["sha256"],
        "artifact_manifest_sha256": file_sha256(args.artifact_manifest.resolve()),
        "execution_source_commit": artifacts["execution_source_commit"],
        "case_count": sum(report["case_count"] for report in reports.values()),
        "compressed_case_count": sum(
            report["compressed_case_count"] for report in reports.values()
        ),
        "target_token_count": sum(report["target_token_count"] for report in reports.values()),
        "layer_record_count": sum(report["layer_record_count"] for report in reports.values()),
        "failure_count": 0,
        "joint_scientific_payload_sha256": canonical_sha256(joint_payload),
        "partitions": reports,
    }
    expected = artifacts["joint_expected"]
    for name in (
        "case_count",
        "compressed_case_count",
        "target_token_count",
        "layer_record_count",
        "failure_count",
    ):
        if audit[name] != expected[name]:
            raise RuntimeError(f"joint postrun {name} mismatch")
    _write_atomic(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
