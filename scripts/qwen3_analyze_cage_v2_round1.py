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
from utils.qwen3_cage_v2_analysis import build_round1_analysis, validate_results_receipt
from utils.qwen3_cage_v2_gate import (
    canonical_sha256,
    file_sha256,
    load_cage_v2_acceptance_gate,
    load_json,
)
from utils.qwen3_cage_v2_postrun import validate_artifact_manifest, validate_partition
from utils.qwen3_cage_v2_protocol import validate_cage_v2_dev_manifest


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render_markdown(analysis: dict) -> str:
    lines = [
        "# Qwen3-8B CAGE-v2 Round-1 development analysis",
        "",
        "> Development-only local perturbation evidence; not an end-to-end quality or population claim.",
        "",
        "## Frozen gate results",
        "",
        "| Family | Budget track | Memory 3/3 | Beats CAGE-v1 3/3 | No worse than Kitty | Track pass | Overall mean delta vs Kitty |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for track in analysis["track_reports"]:
        gates = track["gates"]
        lines.append(
            f"| {track['family_id']} | {track['target']} | "
            f"{gates['memory_all_three']} | {gates['beats_cage_v1_all_three']} | "
            f"{gates['kitty_no_worse_length_count']}/3 | {track['track_pass']} | "
            f"{track['overall_15_case_versus_kitty']['mean_delta']:+.9g} |"
        )
    lines.extend(
        [
            "",
            "## Family advancement",
            "",
            "| Rank | Family | Selected budget | Overall mean delta vs Kitty |",
            "|---:|---|---|---:|",
        ]
    )
    if analysis["advanced_families"]:
        for family in analysis["advanced_families"]:
            lines.append(
                f"| {family['rank']} | {family['family_id']} | {family['selected_target']} | "
                f"{family['selected_overall_kitty_mean_delta']:+.9g} |"
            )
    else:
        lines.append("| - | None | - | - |")
    lines.extend(
        [
            "",
            "Bootstrap intervals are descriptive only and were not used by any advancement gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen CAGE-v2 round1 development results")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    receipt_path = args.receipt.resolve()
    receipt = load_json(receipt_path)
    validate_results_receipt(receipt, receipt_path=receipt_path)
    receipt_sha256 = file_sha256(receipt_path)

    protocol_path = _resolve(receipt["protocol"]["path"])
    manifest_path = _resolve(receipt["input_manifest"]["path"])
    gate_path = _resolve(receipt["acceptance_gate"]["path"])
    artifacts_path = _resolve(receipt["artifact_manifest"]["path"])
    audit_path = _resolve(receipt["joint_postrun_audit"]["path"])
    audit_log_path = _resolve(receipt["joint_postrun_audit"]["log_path"])
    for path, expected, label in (
        (protocol_path, receipt["protocol"]["sha256"], "protocol"),
        (manifest_path, receipt["input_manifest"]["sha256"], "input manifest"),
        (gate_path, receipt["acceptance_gate"]["sha256"], "acceptance gate"),
        (artifacts_path, receipt["artifact_manifest"]["sha256"], "artifact manifest"),
        (audit_path, receipt["joint_postrun_audit"]["sha256"], "postrun audit"),
        (audit_log_path, receipt["joint_postrun_audit"]["log_sha256"], "postrun log"),
    ):
        if file_sha256(path) != expected:
            raise RuntimeError(f"{label} hash differs from receipt")
    if audit_log_path.stat().st_size != receipt["joint_postrun_audit"]["log_size_bytes"]:
        raise RuntimeError("postrun log size differs from receipt")

    protocol, protocol_sha256 = _load_protocol(protocol_path)
    input_manifest = load_json(manifest_path)
    manifest_sha256 = file_sha256(manifest_path)
    validate_cage_v2_dev_manifest(
        input_manifest,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    gate, gate_sha256 = load_cage_v2_acceptance_gate(
        gate_path,
        protocol_path=protocol_path,
        manifest_path=manifest_path,
        verify_artifacts=True,
    )
    artifacts = load_json(artifacts_path)
    validate_artifact_manifest(
        artifacts,
        protocol_sha256=protocol_sha256,
        input_manifest_sha256=manifest_sha256,
        gate_sha256=gate_sha256,
    )
    audit = load_json(audit_path)
    if audit.get("status") != "pass" or audit.get("interpretation_performed") is not False:
        raise RuntimeError("joint audit is not a pre-interpretation pass")

    all_records = []
    joint_payload = []
    for partition in ("cage_qwen3", "kitty_qwen3"):
        expanded = _expand_cases(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            manifest=input_manifest,
            manifest_sha256=manifest_sha256,
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
        validated = validate_partition(
            partition,
            artifacts["partitions"][partition],
            expected_cases=expected,
            accepted_source=gate["acceptance_source"],
            protocol_sha256=protocol_sha256,
            input_manifest_sha256=manifest_sha256,
        )
        payload = validated.pop("scientific_payload")
        for key, value in validated.items():
            if audit["partitions"][partition].get(key) != value:
                raise RuntimeError(f"{partition} audit differs for {key}")
        for path in sorted((Path(artifacts["partitions"][partition]["directory"]) / "cases").glob("*.json")):
            all_records.append(load_json(path))
        if canonical_sha256(sorted(payload, key=lambda item: item["case_id"])) != audit["partitions"][partition]["scientific_payload_sha256"]:
            raise RuntimeError(f"{partition} payload differs from audit")
        joint_payload.extend({"partition": partition, **record} for record in payload)

    joint_payload.sort(key=lambda item: (item["partition"], item["case_id"]))
    if canonical_sha256(joint_payload) != audit["joint_scientific_payload_sha256"]:
        raise RuntimeError("joint scientific payload differs from audit")

    analysis = build_round1_analysis(
        receipt=receipt,
        records=all_records,
        receipt_sha256=receipt_sha256,
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite analysis directory: {output_dir}")
    output_dir.mkdir(parents=True)
    analysis_path = output_dir / "qwen3_cage_v2_round1_analysis_v1.json"
    markdown_path = output_dir / "qwen3_cage_v2_round1_analysis_v1.md"
    _write_json(analysis_path, analysis)
    markdown_path.write_text(_render_markdown(analysis), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "analysis_id": analysis["analysis_id"],
        "receipt_sha256": receipt_sha256,
        "advanced_family_count": analysis["advanced_family_count"],
        "outputs": {
            analysis_path.name: {"sha256": file_sha256(analysis_path), "size_bytes": analysis_path.stat().st_size},
            markdown_path.name: {"sha256": file_sha256(markdown_path), "size_bytes": markdown_path.stat().st_size},
        },
    }
    manifest_path = output_dir / "qwen3_cage_v2_round1_analysis_manifest_v1.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
