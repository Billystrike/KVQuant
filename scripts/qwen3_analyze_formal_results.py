#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_formal import (
    Qwen3FormalError,
    expand_partition_cases,
    file_sha256,
    load_acceptance_gate,
    load_execution_config,
    load_formal_protocol,
    validate_completed_result,
    validate_input_manifest,
)
from utils.qwen3_formal_analysis import (
    Qwen3FormalAnalysisError,
    build_formal_analysis,
    canonical_sha256,
    validate_results_receipt,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and analyze the frozen 1,300-case Qwen3 formal results"
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3FormalAnalysisError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Qwen3FormalAnalysisError(f"JSON root must be an object: {path}")
    return value


def _resolve_receipt_path(receipt_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidate = (REPO_ROOT / path).resolve()
    if not candidate.exists():
        candidate = (receipt_path.parent / path).resolve()
    return candidate


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise Qwen3FormalAnalysisError(f"missing {label}: {path}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise Qwen3FormalAnalysisError(
            f"{label} SHA mismatch: expected={expected_sha256}, observed={observed}"
        )


def _load_partition_records(
    *,
    partition: str,
    receipt_record: dict[str, Any],
    cases: list[dict[str, Any]],
    execution: dict[str, Any],
    execution_sha256: str,
    protocol: dict[str, Any],
    protocol_sha256: str,
    input_manifest_sha256: str,
    acceptance_gate_sha256: str,
) -> list[dict[str, Any]]:
    directory = Path(receipt_record["directory"]).resolve()
    if not directory.is_dir():
        raise Qwen3FormalAnalysisError(f"missing partition directory: {directory}")
    _verify_file(
        directory / "run_identity.json",
        receipt_record["run_identity_sha256"],
        f"{partition} run identity",
    )
    _verify_file(
        directory / "summary.json",
        receipt_record["summary_sha256"],
        f"{partition} summary",
    )
    for path_name, hash_name, label in (
        ("archive_path", "archive_sha256", "archive"),
        ("execution_log_path", "execution_log_sha256", "execution log"),
        ("postrun_audit_path", "postrun_audit_sha256", "post-run audit"),
        (
            "postrun_audit_log_path",
            "postrun_audit_log_sha256",
            "post-run audit log",
        ),
    ):
        _verify_file(
            Path(receipt_record[path_name]).resolve(),
            receipt_record[hash_name],
            f"{partition} {label}",
        )

    expected_case_ids = [case["case_id"] for case in cases]
    if canonical_sha256(expected_case_ids) != receipt_record["case_ids_sha256"]:
        raise Qwen3FormalAnalysisError(f"{partition} expanded case ID SHA mismatch")
    expected_lock = {
        "schema_version": 1,
        "partition": partition,
        "stage": "full",
        "execution_id": execution["execution_id"],
        "execution_sha256": execution_sha256,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "source_state": receipt_record["source_state"],
        "expected_case_ids": expected_case_ids,
        "acceptance_gate_sha256": acceptance_gate_sha256,
    }
    run_identity = _load_json(directory / "run_identity.json")
    if run_identity != expected_lock:
        raise Qwen3FormalAnalysisError(f"{partition} run identity reconstruction failed")

    summary = _load_json(directory / "summary.json")
    expected_count = receipt_record["case_count"]
    expected_identity = {
        key: expected_lock[key]
        for key in (
            "execution_id",
            "execution_sha256",
            "protocol_id",
            "protocol_sha256",
            "input_manifest_sha256",
            "source_state",
            "acceptance_gate_sha256",
        )
    }
    expected_summary = {
        "schema_version": 1,
        "status": "pass",
        "partition": partition,
        "stage": "full",
        "expected_cases": expected_count,
        "completed_cases": expected_count,
        "new_cases": expected_count,
        "resumed_cases": 0,
        "failure_records": 0,
        "identity": expected_identity,
        "case_ids_sha256": receipt_record["case_ids_sha256"],
    }
    if any(summary.get(name) != value for name, value in expected_summary.items()):
        raise Qwen3FormalAnalysisError(f"{partition} summary completion gate failed")
    model_identity = summary.get("model")
    checks = model_identity.get("checks") if isinstance(model_identity, dict) else None
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise Qwen3FormalAnalysisError(f"{partition} model identity checks failed")

    case_paths = {path.stem: path for path in (directory / "cases").glob("*.json")}
    if set(case_paths) != set(expected_case_ids):
        raise Qwen3FormalAnalysisError(f"{partition} case file set mismatch")
    failure_files = list((directory / "failures").glob("*.json"))
    if failure_files:
        raise Qwen3FormalAnalysisError(f"{partition} contains failure records")

    records = []
    scientific_payloads = []
    case_file_hashes = []
    for case in cases:
        path = case_paths[case["case_id"]]
        record = _load_json(path)
        validate_completed_result(
            record,
            expected_case=case,
            execution_id=execution["execution_id"],
            execution_sha256=execution_sha256,
            protocol_id=protocol["protocol_id"],
            protocol_sha256=protocol_sha256,
            input_manifest_sha256=input_manifest_sha256,
            source_state=receipt_record["source_state"],
            acceptance_gate_sha256=acceptance_gate_sha256,
        )
        if record.get("stage") != "full" or record.get("model") != model_identity:
            raise Qwen3FormalAnalysisError(
                f"{partition} case {case['case_id']} identity drift"
            )
        records.append(record)
        scientific_payloads.append(
            {
                "case_id": record.get("case_id"),
                "partition": record.get("partition"),
                "method": record.get("method"),
                "input": record.get("input"),
                "scoring": record.get("scoring"),
                "cache": record.get("cache"),
            }
        )
        case_file_hashes.append(
            {"case_id": case["case_id"], "sha256": file_sha256(path)}
        )
    if canonical_sha256(scientific_payloads) != receipt_record[
        "scientific_payload_sha256"
    ]:
        raise Qwen3FormalAnalysisError(f"{partition} scientific payload SHA mismatch")
    expected_manifest_sha = receipt_record.get("case_file_hash_manifest_sha256")
    if expected_manifest_sha is not None and canonical_sha256(
        case_file_hashes
    ) != expected_manifest_sha:
        raise Qwen3FormalAnalysisError(f"{partition} case file manifest SHA mismatch")

    audit = _load_json(Path(receipt_record["postrun_audit_path"]).resolve())
    audit_checks = {
        "status": audit.get("status") == "pass",
        "partition": audit.get("partition") == partition,
        "case_count": audit.get("case_count") == expected_count,
        "case_ids": audit.get("case_ids_sha256")
        == receipt_record["case_ids_sha256"],
        "run_identity": audit.get("run_identity_sha256")
        == receipt_record["run_identity_sha256"],
        "summary": audit.get("summary_sha256") == receipt_record["summary_sha256"],
        "archive": audit.get("archive_sha256") == receipt_record["archive_sha256"],
        "scientific_payload": audit.get("scientific_payload_sha256")
        == receipt_record["scientific_payload_sha256"],
    }
    if not all(audit_checks.values()):
        failures = [name for name, passed in audit_checks.items() if not passed]
        raise Qwen3FormalAnalysisError(
            f"{partition} post-run audit cross-check failed: {failures}"
        )
    return records


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _method_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in analysis["method_summaries"]:
        paired = record["paired_delta_vs_fp16"]
        row = dict(record)
        row.update(
            {
                "mean_delta_nll_vs_fp16": None
                if paired is None
                else paired["mean_delta_nll"],
                "bootstrap_low_vs_fp16": None
                if paired is None
                else paired["bootstrap_95_interval"]["low"],
                "bootstrap_high_vs_fp16": None
                if paired is None
                else paired["bootstrap_95_interval"]["high"],
            }
        )
        rows.append(row)
    return rows


def _comparison_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        stats = record["statistics"]
        row = {name: value for name, value in record.items() if name != "statistics"}
        row.pop("frozen_memory_selections", None)
        row.update(
            {
                "mean_delta_nll": stats["mean_delta_nll"],
                "median_delta_nll": stats["median_delta_nll"],
                "population_standard_deviation": stats[
                    "population_standard_deviation"
                ],
                "minimum_delta_nll": stats["minimum_delta_nll"],
                "maximum_delta_nll": stats["maximum_delta_nll"],
                "favor_count": stats["favor_count"],
                "tie_count": stats["tie_count"],
                "oppose_count": stats["oppose_count"],
                "exp_of_mean_delta": stats["exp_of_mean_delta"],
                "bootstrap_low": stats["bootstrap_95_interval"]["low"],
                "bootstrap_high": stats["bootstrap_95_interval"]["high"],
            }
        )
        rows.append(row)
    return rows


def _format_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B frozen formal analysis v1",
        "",
        "This file is a deterministic rendering of the frozen 1,300-case result set.",
        "Negative paired delta NLL favors the candidate. Intervals describe stability",
        "over the deterministic 50-anchor grid and are not population significance tests.",
        "Memory values are complete active packed paper estimates, not realized CUDA use.",
        "",
        "## Primary matched-memory comparisons",
        "",
        "| Length | Candidate | Baseline | Memory delta | Mean delta NLL | 95% bootstrap interval | F/T/O | exp(mean delta) |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in analysis["primary_matched_comparisons"]:
        stats = record["statistics"]
        interval = stats["bootstrap_95_interval"]
        lines.append(
            "| {length} | {candidate} | {baseline} | {memory:+.3%} | "
            "{mean:+.6f} | [{low:+.6f}, {high:+.6f}] | {favor}/{tie}/{oppose} | "
            "{ratio:.6f} |".format(
                length=record["prompt_length"],
                candidate=record["candidate"],
                baseline=record["baseline"],
                memory=record["candidate_relative_memory_delta_vs_baseline"],
                mean=stats["mean_delta_nll"],
                low=interval["low"],
                high=interval["high"],
                favor=stats["favor_count"],
                tie=stats["tie_count"],
                oppose=stats["oppose_count"],
                ratio=stats["exp_of_mean_delta"],
            )
        )
    lines.extend(
        [
            "",
            "## Exact-byte mechanism comparisons",
            "",
            "| Length | Role | Candidate | Baseline | Mean delta NLL | 95% bootstrap interval | F/T/O |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for record in analysis["mechanism_comparisons"]:
        stats = record["statistics"]
        interval = stats["bootstrap_95_interval"]
        lines.append(
            "| {length} | {role} | {candidate} | {baseline} | {mean:+.6f} | "
            "[{low:+.6f}, {high:+.6f}] | {favor}/{tie}/{oppose} |".format(
                length=record["prompt_length"],
                role=record["role"],
                candidate=record["candidate"],
                baseline=record["baseline"],
                mean=stats["mean_delta_nll"],
                low=interval["low"],
                high=interval["high"],
                favor=stats["favor_count"],
                tie=stats["tie_count"],
                oppose=stats["oppose_count"],
            )
        )
    lines.extend(["", "## Base-point Pareto sets", ""])
    for record in analysis["base_pareto_by_prompt_length"]:
        methods = ", ".join(f"`{value}`" for value in record["pareto_method_ids"])
        lines.append(f"- Length {record['prompt_length']}: {methods}")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This analysis is cache-conditioned continuation NLL, not canonical full-corpus",
            "WikiText-2 perplexity. It does not support CAGE fused-kernel latency, throughput,",
            "or realized CUDA-memory claims. Passkey remains a saturation diagnostic.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    receipt_path = args.receipt.resolve()
    receipt = _load_json(receipt_path)
    validate_results_receipt(receipt)
    receipt_sha256 = file_sha256(receipt_path)

    protocol_path = _resolve_receipt_path(receipt_path, receipt["protocol"]["path"])
    memory_protocol_path = _resolve_receipt_path(
        receipt_path, receipt["memory_protocol"]["path"]
    )
    execution_path = _resolve_receipt_path(receipt_path, receipt["execution"]["path"])
    gate_path = _resolve_receipt_path(receipt_path, receipt["acceptance_gate"]["path"])
    manifest_path = _resolve_receipt_path(receipt_path, receipt["input_manifest"]["path"])
    for path, record, label in (
        (protocol_path, receipt["protocol"], "protocol"),
        (memory_protocol_path, receipt["memory_protocol"], "memory protocol"),
        (execution_path, receipt["execution"], "execution"),
        (gate_path, receipt["acceptance_gate"], "acceptance gate"),
        (manifest_path, receipt["input_manifest"], "input manifest"),
    ):
        _verify_file(path, record["sha256"], label)

    protocol, protocol_sha256 = load_formal_protocol(protocol_path)
    execution, execution_sha256 = load_execution_config(
        execution_path,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    manifest = _load_json(manifest_path)
    validate_input_manifest(
        manifest,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    manifest_sha256 = file_sha256(manifest_path)
    _, acceptance_gate_sha256 = load_acceptance_gate(
        gate_path,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        execution=execution,
        execution_sha256=execution_sha256,
        input_manifest_sha256=manifest_sha256,
    )

    all_records = []
    for partition in ("cage_qwen3", "kitty_qwen3"):
        cases = expand_partition_cases(
            protocol=protocol,
            execution=execution,
            execution_sha256=execution_sha256,
            input_manifest=manifest,
            partition=partition,
            stage="full",
        )
        all_records.extend(
            _load_partition_records(
                partition=partition,
                receipt_record=receipt["partitions"][partition],
                cases=cases,
                execution=execution,
                execution_sha256=execution_sha256,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                input_manifest_sha256=manifest_sha256,
                acceptance_gate_sha256=acceptance_gate_sha256,
            )
        )

    memory_protocol = _load_json(memory_protocol_path)
    analysis = build_formal_analysis(
        protocol=protocol,
        memory_protocol=memory_protocol,
        receipt=receipt,
        records=all_records,
        receipt_sha256=receipt_sha256,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "qwen3_formal_analysis_v1.json"
    method_csv = output_dir / "qwen3_method_length_summary_v1.csv"
    primary_csv = output_dir / "qwen3_primary_matched_comparisons_v1.csv"
    mechanism_csv = output_dir / "qwen3_mechanism_comparisons_v1.csv"
    markdown_path = output_dir / "qwen3_formal_paper_tables_v1.md"
    manifest_output = output_dir / "qwen3_formal_analysis_manifest_v1.json"

    _write_json(analysis_path, analysis)
    _write_csv(
        method_csv,
        [
            "method_id",
            "method_family",
            "prompt_length",
            "anchor_count",
            "target_count",
            "packed_bytes",
            "token_weighted_mean_nll",
            "perplexity",
            "anchor_mean_nll_median",
            "anchor_mean_nll_population_standard_deviation",
            "anchor_mean_nll_minimum",
            "anchor_mean_nll_maximum",
            "is_base_point",
            "mean_delta_nll_vs_fp16",
            "bootstrap_low_vs_fp16",
            "bootstrap_high_vs_fp16",
        ],
        _method_rows(analysis),
    )
    comparison_fields = [
        "order",
        "comparison_id",
        "role",
        "prompt_length",
        "candidate",
        "baseline",
        "candidate_packed_bytes",
        "baseline_packed_bytes",
        "candidate_relative_memory_delta_vs_baseline",
        "mean_delta_nll",
        "median_delta_nll",
        "population_standard_deviation",
        "minimum_delta_nll",
        "maximum_delta_nll",
        "favor_count",
        "tie_count",
        "oppose_count",
        "exp_of_mean_delta",
        "bootstrap_low",
        "bootstrap_high",
    ]
    _write_csv(
        primary_csv,
        comparison_fields,
        _comparison_rows(analysis["primary_matched_comparisons"]),
    )
    _write_csv(
        mechanism_csv,
        comparison_fields,
        _comparison_rows(analysis["mechanism_comparisons"]),
    )
    temporary_markdown = markdown_path.with_name(markdown_path.name + ".tmp")
    temporary_markdown.write_text(_format_markdown(analysis), encoding="utf-8")
    temporary_markdown.replace(markdown_path)

    output_files = [analysis_path, method_csv, primary_csv, mechanism_csv, markdown_path]
    output_manifest = {
        "schema_version": 1,
        "status": "pass",
        "analysis_id": analysis["analysis_id"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "case_count": analysis["case_count"],
        "target_count": analysis["target_count"],
        "outputs": {
            path.name: {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            for path in output_files
        },
    }
    _write_json(manifest_output, output_manifest)
    print(json.dumps(output_manifest, indent=2, sort_keys=True))
    print(f"analysis_manifest: {manifest_output}")
    print(f"analysis_manifest_sha256: {file_sha256(manifest_output)}")


if __name__ == "__main__":
    try:
        main()
    except (Qwen3FormalError, Qwen3FormalAnalysisError) as error:
        raise SystemExit(f"formal analysis failed: {error}") from error
