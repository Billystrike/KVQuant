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
from scripts.qwen3_run_cage_v3_screen import _validate_full_gate
from utils.qwen3_cage_v3_analysis import build_screen_analysis, validate_screen_receipt
from utils.qwen3_cage_v3_execution import canonical_sha256, load_json
from utils.qwen3_cage_v3_postrun import (
    validate_artifact_manifest,
    validate_failed_pre_case_attempt,
    validate_failed_validation_attempt,
    validate_partition,
    validate_validation_attempt_manifest,
)
from utils.qwen3_cage_v3_protocol import file_sha256
from utils.qwen3_cage_v3_screen import expand_screen_cases, load_screen_execution


EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_execution_v1.json"
GATE_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_acceptance_gate_v1.json"
ARTIFACT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_artifacts_v1.json"
ATTEMPT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_screen_postrun_attempts_v1.json"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _render_markdown(analysis: dict[str, object]) -> str:
    lines = [
        "# Qwen3-8B CAGE-v3 development screen",
        "",
        "> Development-only local perturbation selection evidence; not an end-to-end quality claim.",
        "",
        "| Candidate | Memory | Beats CAGE-v2 | No worse than Kitty-Pro | Pass | Overall delta vs Kitty-Pro |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in analysis["family_reports"]:
        gates = report["gates"]
        lines.append(
            f"| {report['family_id']} | {gates['memory_all_three']} | "
            f"{gates['beats_cage_v2_all_three']} | {gates['kitty_pro_no_worse_length_count']}/3 | "
            f"{report['family_pass']} | {report['overall_15_case_versus_kitty_pro']['mean_delta']:+.9g} |"
        )
    lines.extend(["", f"Decision: `{analysis['decision_status']}`", ""])
    if analysis["advanced_families"]:
        lines.append(f"Selected family: `{analysis['advanced_families'][0]['family_id']}`")
    else:
        lines.append("Selected family: none; holdout remains unread and unauthorized.")
    lines.extend(["", "All three candidates and all three lengths are retained in the JSON output.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the preregistered CAGE-v3 development screen gate")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    receipt_path = args.receipt.resolve()
    receipt = load_json(receipt_path)
    validate_screen_receipt(receipt, receipt_path=receipt_path)
    audit_path = Path(receipt["audit"]["path"])
    audit_log = Path(receipt["audit"]["execution_log"])
    if file_sha256(audit_path) != receipt["audit"]["sha256"]:
        raise RuntimeError("passed audit hash differs from receipt")
    if file_sha256(audit_log) != receipt["audit"]["execution_log_sha256"]:
        raise RuntimeError("passed audit log hash differs from receipt")
    if audit_log.stat().st_size != receipt["audit"]["execution_log_size_bytes"]:
        raise RuntimeError("passed audit log size differs from receipt")

    execution, execution_sha256, protocol, manifest, plan = load_screen_execution(
        EXECUTION_PATH, repo_root=REPO_ROOT, verify_artifacts=True
    )
    gate = _validate_full_gate(GATE_PATH, execution=execution, execution_sha256=execution_sha256)
    artifacts = load_json(ARTIFACT_PATH)
    validate_artifact_manifest(
        artifacts,
        execution_sha256=execution_sha256,
        protocol_sha256=execution["protocol"]["sha256"],
        input_manifest_sha256=execution["input_manifest"]["sha256"],
        quota_plan_sha256=execution["quota_plan"]["sha256"],
        gate_sha256=file_sha256(GATE_PATH),
    )
    attempts = load_json(ATTEMPT_PATH)
    validate_validation_attempt_manifest(attempts)
    for attempt in attempts["attempts"]:
        validate_failed_validation_attempt(attempt)
    for attempt in artifacts["failed_pre_case_attempts"]:
        validate_failed_pre_case_attempt(attempt)

    audit = load_json(audit_path)
    if audit.get("status") != "pass" or audit.get("interpretation_performed") is not False:
        raise RuntimeError("screen audit is not an uninterpreted pass")
    all_records = []
    joint_payload = []
    for partition in ("cage_qwen3", "kitty_qwen3"):
        expanded = expand_screen_cases(
            execution=execution,
            execution_sha256=execution_sha256,
            protocol=protocol,
            manifest=manifest,
            plan=plan,
            partition=partition,
            stage="screen_full",
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
        )
        payload = report.pop("scientific_payload")
        for key, value in report.items():
            if audit["partitions"][partition].get(key) != value:
                raise RuntimeError(f"{partition} differs from passed audit: {key}")
        root = Path(artifacts["partitions"][partition]["directory"])
        all_records.extend(load_json(path) for path in sorted((root / "cases").glob("*.json")))
        joint_payload.extend({"partition": partition, **row} for row in payload)
    joint_payload.sort(key=lambda row: (row["partition"], row["case_id"]))
    if canonical_sha256(joint_payload) != audit["joint_scientific_payload_sha256"]:
        raise RuntimeError("joint payload differs from passed audit")

    analysis = build_screen_analysis(
        protocol=protocol,
        receipt=receipt,
        receipt_sha256=file_sha256(receipt_path),
        records=all_records,
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite screen analysis: {output_dir}")
    output_dir.mkdir(parents=True)
    analysis_path = output_dir / "qwen3_cage_v3_screen_analysis_v1.json"
    markdown_path = output_dir / "qwen3_cage_v3_screen_analysis_v1.md"
    _write_json(analysis_path, analysis)
    markdown_path.write_text(_render_markdown(analysis), encoding="utf-8")
    result_manifest = {
        "schema_version": 1,
        "status": "pass",
        "analysis_id": analysis["analysis_id"],
        "receipt_sha256": file_sha256(receipt_path),
        "advanced_family_count": analysis["advanced_family_count"],
        "holdout_authorized_after_decision_freeze": analysis["holdout_authorized_after_decision_freeze"],
        "outputs": {
            analysis_path.name: {"sha256": file_sha256(analysis_path), "size_bytes": analysis_path.stat().st_size},
            markdown_path.name: {"sha256": file_sha256(markdown_path), "size_bytes": markdown_path.stat().st_size},
        },
    }
    manifest_path = output_dir / "qwen3_cage_v3_screen_analysis_manifest_v1.json"
    _write_json(manifest_path, result_manifest)
    print(json.dumps(result_manifest, indent=2, sort_keys=True, allow_nan=False))
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
