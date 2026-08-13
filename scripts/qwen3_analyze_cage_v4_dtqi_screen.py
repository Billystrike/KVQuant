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

from utils.qwen3_cage_v4_analysis import validate_results_receipt as validate_baseline_receipt
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256, load_data_protocol, validate_input_manifest
from utils.qwen3_cage_v4_dtqi_acceptance import load_json
from utils.qwen3_cage_v4_dtqi_analysis import BASELINES, CANDIDATE, build_analysis, validate_results_receipt
from utils.qwen3_cage_v4_dtqi_screen import expand_screen_cases, load_screen_execution as load_dtqi_execution
from utils.qwen3_cage_v4_dtqi_screen_postrun import validate_artifact_manifest as validate_dtqi_artifacts, validate_screen
from utils.qwen3_cage_v4_execution import load_screen_execution as load_baseline_execution
from utils.qwen3_cage_v4_postrun import expected_memory_report, validate_artifact_manifest as validate_baseline_artifacts, validate_attempt_manifest, validate_partition


BASELINE_EXECUTION = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_screen_execution_v1.json"
BASELINE_ARTIFACTS = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_screen_artifacts_v1.json"
BASELINE_ATTEMPTS = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_screen_postrun_attempts_v1.json"
DTQI_EXECUTION = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_dtqi_screen_execution_v1.json"
DTQI_ARTIFACTS = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_dtqi_screen_artifacts_v1.json"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _baseline_records(receipt: dict) -> list[dict]:
    baseline_execution, execution_sha, _, protocol_sha, _, expanded = load_baseline_execution(BASELINE_EXECUTION, repo_root=REPO_ROOT, verify_artifacts=True)
    if expanded is None:
        raise RuntimeError("baseline cases were not expanded")
    artifacts = load_json(BASELINE_ARTIFACTS)
    validate_baseline_artifacts(
        artifacts,
        execution_sha256=execution_sha,
        protocol_sha256=protocol_sha,
        input_manifest_sha256=baseline_execution["input_manifest"]["sha256"],
        gate_sha256=baseline_execution["acceptance_gate"]["sha256"],
    )
    attempts = load_json(BASELINE_ATTEMPTS)
    validate_attempt_manifest(attempts, verify_artifacts=True)
    records = []
    for partition in ("cage_qwen3", "kitty_qwen3"):
        expected = [{"case_id": case["case_id"], "method": case["method"], "input": case["input"], "memory": expected_memory_report(case["method"])} for case in expanded[partition]]
        validate_partition(
            partition,
            artifacts["partitions"][partition],
            expected_cases=expected,
            execution_source_commit=artifacts["execution_source_commit"],
            execution_sha256=execution_sha,
            protocol_sha256=protocol_sha,
            input_manifest_sha256=baseline_execution["input_manifest"]["sha256"],
            gate_sha256=baseline_execution["acceptance_gate"]["sha256"],
            runner_sha256=artifacts["runner_sha256"],
            accepted_runtime_sha256=artifacts["accepted_runtime_sha256"],
        )
        root = Path(artifacts["partitions"][partition]["directory"])
        for path in sorted((root / "cases").glob("*.json")):
            value = load_json(path)
            method = value["method"]["metric_method_id"]
            if method in BASELINES:
                records.append({
                    "analysis_method_id": method,
                    "input": value["input"],
                    "memory": value["memory"],
                    "scoring": value["scoring"],
                })
    if len(records) != 240:
        raise RuntimeError("expected exactly 240 reused baseline records")
    return records


def _dtqi_records(receipt: dict) -> list[dict]:
    spec = receipt["dtqi_results"]
    audit_path = Path(spec["postrun_audit_path"])
    log_path = Path(spec["postrun_log_path"])
    for path, sha, size in (
        (audit_path, spec["postrun_audit_sha256"], spec["postrun_audit_size_bytes"]),
        (log_path, spec["postrun_log_sha256"], spec["postrun_log_size_bytes"]),
    ):
        if not path.is_file() or file_sha256(path) != sha or path.stat().st_size != size:
            raise RuntimeError("DTQI postrun receipt artifact mismatch")
    audit = load_json(audit_path)
    if audit.get("status") != "pass" or audit.get("interpretation_performed") is not False:
        raise RuntimeError("DTQI postrun audit is not an uninterpreted pass")
    if audit["screen"]["scientific_payload_sha256"] != spec["scientific_payload_sha256"]:
        raise RuntimeError("DTQI postrun scientific payload mismatch")
    execution, execution_sha, protocol, _, quota = load_dtqi_execution(DTQI_EXECUTION, repo_root=REPO_ROOT, verify_artifacts=True)
    artifacts = load_json(DTQI_ARTIFACTS)
    validate_dtqi_artifacts(artifacts)
    manifest = load_json(execution["input_manifest"]["path"])
    data, data_sha = load_data_protocol(REPO_ROOT / execution["data_protocol"]["path"])
    validate_input_manifest(manifest, protocol=data, protocol_sha256=data_sha)
    cases = expand_screen_cases(execution=execution, execution_sha256=execution_sha, protocol=protocol, quota_plan=quota, input_manifest=manifest)
    report = validate_screen(artifacts, cases)
    if report["scientific_payload_sha256"] != spec["scientific_payload_sha256"]:
        raise RuntimeError("DTQI revalidated payload differs from receipt")
    records = []
    root = Path(artifacts["root"])
    for path in sorted((root / "cases").glob("*.json")):
        value = load_json(path)
        records.append({
            "analysis_method_id": CANDIDATE,
            "input": value["input"],
            "memory": value["memory"],
            "scoring": value["scoring"],
        })
    if len(records) != 120:
        raise RuntimeError("expected exactly 120 DTQI records")
    return records


def _validate_common_grid(records: list[dict]) -> None:
    grids = {}
    for method in (CANDIDATE, *BASELINES):
        grid = {
            (row["input"]["document_id"], row["input"]["anchor_index"], row["input"]["prompt_length"]): row["input"]
            for row in records
            if row["analysis_method_id"] == method
        }
        if len(grid) != 120:
            raise RuntimeError(f"{method} common-grid count mismatch")
        grids[method] = grid
    keys = set(grids[CANDIDATE])
    if any(set(grids[method]) != keys for method in BASELINES):
        raise RuntimeError("candidate and baselines do not share the same document/anchor/length grid")
    for key in keys:
        candidate = grids[CANDIDATE][key]
        for baseline in BASELINES:
            if grids[baseline][key] != candidate:
                raise RuntimeError("candidate and baseline input identities differ")


def _markdown(analysis: dict) -> str:
    lines = [
        "# Qwen3-8B CAGE-v4-DTQI PG-19 screen decision",
        "",
        "> Development-screen decision only; holdout and test were not accessed.",
        "",
        "## Preregistered decision",
        "",
        f"Candidate outcome: **{analysis['success_decision']['candidate_outcome']}**.",
        "",
        "| Baseline | Material overall | CI favorable | All lengths noninferior | Baseline pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for baseline, row in analysis["success_decision"]["per_baseline"].items():
        lines.append(f"| {baseline} | {row['overall_material_superiority_pass']} | {row['overall_uncertainty_pass']} | {row['all_lengths_noninferiority_pass']} | {row['baseline_pass']} |")
    lines.extend(["", "## Paired comparisons", "", "| Baseline | Length | Relative PPL (%) | NLL delta CI95 | Decision check |", "|---|---:|---:|---|---|"])
    for row in analysis["paired_comparisons"]:
        check = row.get("noninferiority_pass", row.get("material_superiority_pass"))
        ci = row["bootstrap_mean_nll_delta_ci95"]
        lines.append(f"| {row['baseline_method']} | {row['prompt_length']} | {row['relative_ppl_percent']:.6f} | [{ci[0]:.9g}, {ci[1]:.9g}] | {check} |")
    lines.extend(["", "All unfavorable results are retained. Local MSE was not used. No runtime or paper claim is authorized.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen CAGE-v4-DTQI screen against two frozen baselines")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.receipt.resolve()
    receipt = load_json(receipt_path)
    validate_results_receipt(receipt, receipt_path=receipt_path)
    baseline_receipt_path = REPO_ROOT / receipt["baseline_results"]["receipt_path"]
    baseline_receipt = load_json(baseline_receipt_path)
    validate_baseline_receipt(baseline_receipt, receipt_path=baseline_receipt_path)
    records = _baseline_records(baseline_receipt) + _dtqi_records(receipt)
    _validate_common_grid(records)
    analysis = build_analysis(receipt_sha256=file_sha256(receipt_path), records=records)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DTQI analysis: {output}")
    output.mkdir(parents=True)
    analysis_path = output / "qwen3_cage_v4_dtqi_screen_analysis_v1.json"
    summary_path = output / "qwen3_cage_v4_dtqi_method_length_summary_v1.csv"
    comparisons_path = output / "qwen3_cage_v4_dtqi_paired_comparisons_v1.csv"
    decision_path = output / "qwen3_cage_v4_dtqi_screen_decision_v1.md"
    _write_json(analysis_path, analysis)
    _write_csv(summary_path, analysis["method_length_summaries"], ["method_id", "prompt_length", "case_count", "document_count", "target_token_count", "packed_bytes", "mean_nll", "perplexity"])
    _write_csv(comparisons_path, analysis["paired_comparisons"], ["candidate_method", "baseline_method", "prompt_length", "paired_document_count", "mean_nll_delta", "median_document_nll_delta", "candidate_favor_count", "tie_count", "baseline_favor_count", "relative_ppl_percent", "bootstrap_resamples", "bootstrap_mean_nll_delta_ci95", "bootstrap_favorable", "noninferiority_margin_relative_ppl_percent", "noninferiority_pass", "material_superiority_threshold_relative_ppl_percent", "material_superiority_pass", "uncertainty_pass"])
    decision_path.write_text(_markdown(analysis), encoding="utf-8")
    outputs = {}
    for path in (analysis_path, summary_path, comparisons_path, decision_path):
        outputs[path.name] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    manifest_out = {
        "schema_version": 1,
        "status": "pass",
        "analysis_id": analysis["analysis_id"],
        "receipt_sha256": file_sha256(receipt_path),
        "case_count": 360,
        "outputs": outputs,
        "candidate_outcome": analysis["success_decision"]["candidate_outcome"],
        "holdout_accessed": False,
        "pg19_test_accessed": False,
    }
    manifest_path = output / "qwen3_cage_v4_dtqi_screen_analysis_manifest_v1.json"
    _write_json(manifest_path, manifest_out)
    print(json.dumps({"status": "pass", "analysis": analysis, "manifest": manifest_out, "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path)}, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
