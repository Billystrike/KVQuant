#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qwen3_run_cage_v2_round1 import _expand_cases, _load_protocol, _memory_report
from utils.qwen3_cage_v2_gate import file_sha256, load_cage_v2_acceptance_gate, load_json
from utils.qwen3_cage_v2_postrun import validate_artifact_manifest, validate_partition
from utils.qwen3_cage_v2_protocol import validate_cage_v2_dev_manifest


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jointly validate the 145-case CAGE-v2 round1 screen")
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol, protocol_sha256 = _load_protocol(args.protocol.resolve())
    input_manifest = load_json(args.manifest.resolve())
    input_manifest_sha256 = file_sha256(args.manifest.resolve())
    validate_cage_v2_dev_manifest(
        input_manifest,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    gate, gate_sha256 = load_cage_v2_acceptance_gate(
        args.gate.resolve(),
        protocol_path=args.protocol.resolve(),
        manifest_path=args.manifest.resolve(),
        verify_artifacts=True,
    )
    artifacts = load_json(args.artifact_manifest.resolve())
    validate_artifact_manifest(
        artifacts,
        protocol_sha256=protocol_sha256,
        input_manifest_sha256=input_manifest_sha256,
        gate_sha256=gate_sha256,
    )
    if artifacts.get("execution_source_commit") != gate["acceptance_source"]["cage_commit"]:
        raise RuntimeError("screen artifact and acceptance execution commits differ")

    partition_reports = {}
    joint_payload = []
    for partition in ("cage_qwen3", "kitty_qwen3"):
        expanded = _expand_cases(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            manifest=input_manifest,
            manifest_sha256=input_manifest_sha256,
            partition=partition,
            stage="screen",
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
            accepted_source=gate["acceptance_source"],
            protocol_sha256=protocol_sha256,
            input_manifest_sha256=input_manifest_sha256,
        )
        joint_payload.extend(
            {"partition": partition, **record} for record in report.pop("scientific_payload")
        )
        partition_reports[partition] = report

    joint_payload.sort(key=lambda item: (item["partition"], item["case_id"]))
    from utils.qwen3_cage_v2_gate import canonical_sha256

    output = {
        "schema_version": 1,
        "status": "pass",
        "audit_id": "qwen3-8b-cage-v2-round1-development-screen-postrun-v1",
        "claim_eligible": False,
        "interpretation_performed": False,
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": input_manifest_sha256,
        "acceptance_gate_sha256": gate_sha256,
        "artifact_manifest_sha256": file_sha256(args.artifact_manifest.resolve()),
        "execution_source_commit": artifacts["execution_source_commit"],
        "case_count": sum(record["case_count"] for record in partition_reports.values()),
        "layer_record_count": sum(record["layer_record_count"] for record in partition_reports.values()),
        "failure_count": 0,
        "joint_scientific_payload_sha256": canonical_sha256(joint_payload),
        "partitions": partition_reports,
    }
    if output["case_count"] != 145 or output["layer_record_count"] != 5220:
        raise RuntimeError("joint postrun totals mismatch")
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
