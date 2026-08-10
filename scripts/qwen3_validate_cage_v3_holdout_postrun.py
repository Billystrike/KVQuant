#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qwen3_run_cage_v2_round1 import _memory_report
from scripts.qwen3_run_cage_v3_holdout import _validate_full_gate
from utils.qwen3_cage_v3_execution import canonical_sha256, load_json
from utils.qwen3_cage_v3_holdout import expand_holdout_cases, load_holdout_execution
from utils.qwen3_cage_v3_holdout_postrun import validate_artifact_manifest, validate_partition
from utils.qwen3_cage_v3_protocol import file_sha256


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jointly validate the 90-case CAGE-v3 development holdout")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--acceptance-gate", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    execution, execution_sha256, protocol, manifest, plan, decision = load_holdout_execution(
        args.execution.resolve(), repo_root=REPO_ROOT, verify_artifacts=True
    )
    gate_path = args.acceptance_gate.resolve()
    gate = _validate_full_gate(gate_path, execution=execution, execution_sha256=execution_sha256)
    gate_sha256 = file_sha256(gate_path)
    artifacts_path = args.artifact_manifest.resolve()
    artifacts = load_json(artifacts_path)
    validate_artifact_manifest(
        artifacts,
        execution_sha256=execution_sha256,
        protocol_sha256=execution["protocol"]["sha256"],
        input_manifest_sha256=execution["input_manifest"]["sha256"],
        quota_plan_sha256=execution["quota_plan"]["sha256"],
        screen_decision_sha256=execution["screen_decision"]["sha256"],
        gate_sha256=gate_sha256,
    )

    partition_reports = {}
    joint_payload = []
    for partition in ("cage_qwen3", "kitty_qwen3"):
        expanded = expand_holdout_cases(
            execution=execution,
            execution_sha256=execution_sha256,
            protocol=protocol,
            manifest=manifest,
            plan=plan,
            partition=partition,
            stage="holdout_full",
        )
        expected = {
            case["case_id"]: {
                "method": case["method"],
                "input": case["input"],
                "memory": _memory_report(case["method"]),
            }
            for case in expanded
        }
        report = validate_partition(
            partition,
            artifacts["partitions"][partition],
            expected_cases=expected,
            execution_source_commit=artifacts["execution_source_commit"],
            expected_source_sha256=gate["acceptance_source"]["source_sha256"],
            execution_sha256=execution_sha256,
            protocol_sha256=execution["protocol"]["sha256"],
            input_manifest_sha256=execution["input_manifest"]["sha256"],
            quota_plan_sha256=execution["quota_plan"]["sha256"],
            screen_decision_sha256=execution["screen_decision"]["sha256"],
        )
        joint_payload.extend({"partition": partition, **row} for row in report.pop("scientific_payload"))
        partition_reports[partition] = report

    joint_payload.sort(key=lambda item: (item["partition"], item["case_id"]))
    output = {
        "schema_version": 1,
        "status": "pass",
        "audit_id": "qwen3-8b-cage-v3-development-holdout-postrun-v1",
        "claim_eligible": False,
        "interpretation_performed": False,
        "holdout_metrics_consumed": True,
        "reserved_unseen_metrics_consumed": False,
        "end_to_end_quality_authorized": False,
        "execution_sha256": execution_sha256,
        "protocol_sha256": execution["protocol"]["sha256"],
        "manifest_sha256": execution["input_manifest"]["sha256"],
        "quota_plan_sha256": execution["quota_plan"]["sha256"],
        "screen_decision_sha256": execution["screen_decision"]["sha256"],
        "selected_family_id": decision["decision"]["selected_family_id"],
        "acceptance_gate_sha256": gate_sha256,
        "artifact_manifest_sha256": file_sha256(artifacts_path),
        "execution_source_commit": artifacts["execution_source_commit"],
        "case_count": sum(row["case_count"] for row in partition_reports.values()),
        "layer_record_count": sum(row["layer_record_count"] for row in partition_reports.values()),
        "failure_count": 0,
        "joint_scientific_payload_sha256": canonical_sha256(joint_payload),
        "partitions": partition_reports,
    }
    if output["case_count"] != 90 or output["layer_record_count"] != 3240:
        raise RuntimeError("joint CAGE-v3 holdout totals mismatch")
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite CAGE-v3 holdout postrun audit: {output_path}")
    _atomic_write(output_path, output)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
