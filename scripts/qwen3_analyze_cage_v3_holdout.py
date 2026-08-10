#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from utils.qwen3_cage_v3_holdout_analysis import build_holdout_analysis, validate_holdout_receipt
from utils.qwen3_cage_v3_holdout_postrun import validate_artifact_manifest, validate_partition
from utils.qwen3_cage_v3_protocol import file_sha256


EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_holdout_execution_v1.json"
GATE_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_holdout_acceptance_gate_v1.json"
ARTIFACT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_holdout_full_artifacts_v1.json"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _render_markdown(analysis: dict[str, object]) -> str:
    lines = [
        "# Qwen3-8B CAGE-v3 development holdout",
        "",
        "> Development-only local perturbation evidence; not an end-to-end quality or runtime claim.",
        "",
        "| Length | CAGE-v3 bytes | Kitty-Pro bytes | Delta vs CAGE-v2 | Beats CAGE-v2 | Delta vs Kitty-Pro | No worse than Kitty-Pro |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["lengths"]:
        lines.append(
            f"| {row['prompt_length']} | {row['candidate_model_total_bytes']} | "
            f"{row['kitty_pro_model_total_bytes']} | {row['versus_cage_v2']['mean_delta']:+.9g} | "
            f"{row['beats_cage_v2']} | {row['versus_kitty_pro']['mean_delta']:+.9g} | "
            f"{row['no_worse_than_kitty_pro']} |"
        )
    lines.extend([
        "",
        f"Preregistered holdout gate: `{analysis['holdout_gate_pass']}`",
        "",
        f"Decision: `{analysis['decision_status']}`",
        "",
        "All three lengths and all paired deltas are retained in the JSON output, including unfavorable results.",
        "No bootstrap interval was used for the gate; no seed or resample count was preregistered.",
        "Reserved-unseen anchors 20–49 remain unread.",
        "",
    ])
    return "\n".join(lines)


def _write_length_csv(path: Path, analysis: dict[str, object]) -> None:
    fieldnames = [
        "prompt_length",
        "candidate_model_total_bytes",
        "kitty_pro_model_total_bytes",
        "candidate_minus_kitty_pro_bytes",
        "memory_pass",
        "mean_delta_vs_cage_v2",
        "median_delta_vs_cage_v2",
        "favor_count_vs_cage_v2",
        "tie_count_vs_cage_v2",
        "oppose_count_vs_cage_v2",
        "beats_cage_v2",
        "mean_delta_vs_kitty_pro",
        "median_delta_vs_kitty_pro",
        "favor_count_vs_kitty_pro",
        "tie_count_vs_kitty_pro",
        "oppose_count_vs_kitty_pro",
        "no_worse_than_kitty_pro",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in analysis["lengths"]:
            control = row["versus_cage_v2"]
            kitty = row["versus_kitty_pro"]
            writer.writerow({
                "prompt_length": row["prompt_length"],
                "candidate_model_total_bytes": row["candidate_model_total_bytes"],
                "kitty_pro_model_total_bytes": row["kitty_pro_model_total_bytes"],
                "candidate_minus_kitty_pro_bytes": row["candidate_minus_kitty_pro_bytes"],
                "memory_pass": row["memory_pass"],
                "mean_delta_vs_cage_v2": control["mean_delta"],
                "median_delta_vs_cage_v2": control["median_delta"],
                "favor_count_vs_cage_v2": control["favor_count"],
                "tie_count_vs_cage_v2": control["tie_count"],
                "oppose_count_vs_cage_v2": control["oppose_count"],
                "beats_cage_v2": row["beats_cage_v2"],
                "mean_delta_vs_kitty_pro": kitty["mean_delta"],
                "median_delta_vs_kitty_pro": kitty["median_delta"],
                "favor_count_vs_kitty_pro": kitty["favor_count"],
                "tie_count_vs_kitty_pro": kitty["tie_count"],
                "oppose_count_vs_kitty_pro": kitty["oppose_count"],
                "no_worse_than_kitty_pro": row["no_worse_than_kitty_pro"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the preregistered CAGE-v3 development-holdout gate")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    receipt_path = args.receipt.resolve()
    receipt = load_json(receipt_path)
    validate_holdout_receipt(receipt, receipt_path=receipt_path)
    audit_path = Path(receipt["audit"]["path"])
    audit_log = Path(receipt["audit"]["execution_log"])
    if file_sha256(audit_path) != receipt["audit"]["sha256"]:
        raise RuntimeError("holdout audit hash differs from frozen receipt")
    if audit_path.stat().st_size != receipt["audit"]["size_bytes"]:
        raise RuntimeError("holdout audit size differs from frozen receipt")
    if file_sha256(audit_log) != receipt["audit"]["execution_log_sha256"]:
        raise RuntimeError("holdout audit log hash differs from frozen receipt")
    if audit_log.stat().st_size != receipt["audit"]["execution_log_size_bytes"]:
        raise RuntimeError("holdout audit log size differs from frozen receipt")

    execution, execution_sha256, protocol, manifest, plan, decision = load_holdout_execution(
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
        screen_decision_sha256=execution["screen_decision"]["sha256"],
        gate_sha256=file_sha256(GATE_PATH),
    )
    audit = load_json(audit_path)
    if audit.get("status") != "pass" or audit.get("interpretation_performed") is not False:
        raise RuntimeError("holdout audit is not an uninterpreted pass")
    if audit.get("reserved_unseen_metrics_consumed") is not False:
        raise RuntimeError("reserved-unseen boundary was violated")

    all_records = []
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
        payload = report.pop("scientific_payload")
        for key, value in report.items():
            if audit["partitions"][partition].get(key) != value:
                raise RuntimeError(f"{partition} differs from passed holdout audit: {key}")
        root = Path(artifacts["partitions"][partition]["directory"])
        all_records.extend(load_json(path) for path in sorted((root / "cases").glob("*.json")))
        joint_payload.extend({"partition": partition, **row} for row in payload)
    joint_payload.sort(key=lambda row: (row["partition"], row["case_id"]))
    if canonical_sha256(joint_payload) != audit["joint_scientific_payload_sha256"]:
        raise RuntimeError("joint holdout payload differs from passed audit")

    analysis = build_holdout_analysis(
        protocol=protocol,
        decision=decision,
        receipt=receipt,
        receipt_sha256=file_sha256(receipt_path),
        records=all_records,
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite holdout analysis: {output_dir}")
    output_dir.mkdir(parents=True)
    analysis_path = output_dir / "qwen3_cage_v3_holdout_analysis_v1.json"
    markdown_path = output_dir / "qwen3_cage_v3_holdout_analysis_v1.md"
    csv_path = output_dir / "qwen3_cage_v3_holdout_length_comparisons_v1.csv"
    _write_json(analysis_path, analysis)
    markdown_path.write_text(_render_markdown(analysis), encoding="utf-8")
    _write_length_csv(csv_path, analysis)
    result_manifest = {
        "schema_version": 1,
        "status": "pass",
        "analysis_id": analysis["analysis_id"],
        "receipt_sha256": file_sha256(receipt_path),
        "holdout_gate_pass": analysis["holdout_gate_pass"],
        "next_protocol_may_be_frozen": analysis["next_protocol_may_be_frozen"],
        "reserved_unseen_metrics_consumed": False,
        "outputs": {
            path.name: {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            for path in (analysis_path, markdown_path, csv_path)
        },
    }
    manifest_path = output_dir / "qwen3_cage_v3_holdout_analysis_manifest_v1.json"
    _write_json(manifest_path, result_manifest)
    print(json.dumps(result_manifest, indent=2, sort_keys=True, allow_nan=False))
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
