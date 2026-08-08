#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_formal import load_formal_protocol
from utils.qwen3_perturbation_analysis import (
    PRIMARY_METRIC,
    Qwen3PerturbationAnalysisError,
    build_perturbation_analysis,
    canonical_sha256,
    validate_results_receipt,
)
from utils.qwen3_perturbation_protocol import (
    LAYER_METRICS,
    Qwen3PerturbationError,
    file_sha256,
    load_perturbation_acceptance_gate,
    load_perturbation_protocol,
    scientific_payload_sha256,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen 1,300-case Qwen3 perturbation result set"
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3PerturbationAnalysisError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise Qwen3PerturbationAnalysisError(f"{label} must be a JSON object: {path}")
    return value


def _resolve_receipt_path(receipt_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (receipt_path.parents[1] / path).resolve()


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise Qwen3PerturbationAnalysisError(f"missing {label}: {path}")
    if file_sha256(path) != expected_sha256:
        raise Qwen3PerturbationAnalysisError(f"{label} SHA-256 mismatch: {path}")


def _canonical_case_manifest(paths: list[Path]) -> str:
    return canonical_sha256(
        [{"filename": path.name, "sha256": file_sha256(path)} for path in paths]
    )


def _validate_joint_audit(
    audit: dict[str, Any], receipt: dict[str, Any], *, artifact_sha256: str
) -> None:
    expected = receipt["joint_postrun_audit"]
    checks = {
        "schema": audit.get("schema_version") == 1,
        "status": audit.get("status") == "pass",
        "stage": audit.get("stage") == "joint_full_postrun_validation",
        "artifact_manifest": audit.get("artifact_manifest_sha256") == artifact_sha256,
        "protocol": audit.get("perturbation_protocol_sha256")
        == receipt["perturbation_protocol"]["sha256"],
        "acceptance_gate": audit.get("acceptance_gate_sha256")
        == receipt["acceptance_gate"]["sha256"],
        "quality_protocol": audit.get("quality_protocol_sha256")
        == receipt["quality_protocol"]["sha256"],
        "quality_execution": audit.get("quality_execution_sha256")
        == receipt["quality_execution"]["sha256"],
        "input_manifest": audit.get("input_manifest_sha256")
        == receipt["input_manifest"]["sha256"],
        "source_commit": audit.get("source_commit") == receipt["execution_source_commit"],
        "case_count": audit.get("case_count") == expected["case_count"],
        "layer_record_count": audit.get("layer_record_count")
        == expected["layer_record_count"],
        "failure_count": audit.get("failure_count") == 0,
        "scientific_payload": audit.get("joint_scientific_payload_sha256")
        == expected["joint_scientific_payload_sha256"],
        "no_prior_interpretation": audit.get("interpretation_performed") is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise Qwen3PerturbationAnalysisError(
            f"joint post-run audit cross-check failed: {failures}"
        )
    if set(audit.get("partitions", {})) != {"cage_qwen3", "kitty_qwen3"}:
        raise Qwen3PerturbationAnalysisError("joint audit partition set mismatch")
    for partition, receipt_record in receipt["partitions"].items():
        record = audit["partitions"][partition]
        mapping = {
            "case_count": "case_count",
            "layer_record_count": "layer_record_count",
            "failure_count": "failure_count",
            "run_identity_sha256": "run_identity_sha256",
            "summary_sha256": "summary_sha256",
            "execution_log_sha256": "execution_log_sha256",
            "execution_log_size_bytes": "execution_log_size_bytes",
            "case_ids_sha256": "case_ids_sha256",
            "case_file_hash_manifest_sha256": "canonical_case_file_manifest_sha256",
            "shell_case_file_manifest_sha256": "shell_case_file_manifest_sha256",
            "scientific_payload_sha256": "scientific_payload_sha256",
            "model_identity_sha256": "model_identity_sha256",
        }
        mismatches = [
            audit_name
            for audit_name, receipt_name in mapping.items()
            if record.get(audit_name) != receipt_record.get(receipt_name)
        ]
        if mismatches:
            raise Qwen3PerturbationAnalysisError(
                f"{partition} audit/receipt mismatch: {mismatches}"
            )
        required_true = (
            "all_case_schema_checks_pass",
            "all_aggregate_recomputations_pass",
            "all_memory_checks_pass",
            "all_cache_mechanics_checks_pass",
            "all_model_identity_checks_pass",
        )
        if any(record.get(name) is not True for name in required_true):
            raise Qwen3PerturbationAnalysisError(
                f"{partition} joint audit contains a failed scientific check"
            )


def _load_partition_records(
    partition: str, receipt_record: dict[str, Any]
) -> list[dict[str, Any]]:
    root = Path(receipt_record["directory"])
    identity_path = root / "run_identity.json"
    summary_path = root / "summary.json"
    log_path = Path(receipt_record["execution_log_path"])
    for path, expected, label in (
        (identity_path, receipt_record["run_identity_sha256"], f"{partition} run identity"),
        (summary_path, receipt_record["summary_sha256"], f"{partition} summary"),
        (log_path, receipt_record["execution_log_sha256"], f"{partition} execution log"),
    ):
        _verify_file(path, expected, label)
    if log_path.stat().st_size != receipt_record["execution_log_size_bytes"]:
        raise Qwen3PerturbationAnalysisError(f"{partition} execution log size mismatch")
    identity = _load_json(identity_path, f"{partition} run identity")
    summary = _load_json(summary_path, f"{partition} summary")
    expected_count = receipt_record["case_count"]
    if (
        summary.get("status") != "pass"
        or summary.get("stage") != "full"
        or summary.get("partition") != partition
        or summary.get("expected_cases") != expected_count
        or summary.get("completed_cases") != expected_count
        or summary.get("failure_records") != 0
        or summary.get("identity") != identity
        or summary.get("case_ids_sha256") != receipt_record["case_ids_sha256"]
    ):
        raise Qwen3PerturbationAnalysisError(f"{partition} summary content mismatch")
    if canonical_sha256(summary.get("model")) != receipt_record["model_identity_sha256"]:
        raise Qwen3PerturbationAnalysisError(f"{partition} model identity mismatch")

    paths = sorted((root / "cases").glob("*.json"))
    if len(paths) != expected_count:
        raise Qwen3PerturbationAnalysisError(f"{partition} case count mismatch")
    if _canonical_case_manifest(paths) != receipt_record["canonical_case_file_manifest_sha256"]:
        raise Qwen3PerturbationAnalysisError(f"{partition} case-file manifest mismatch")
    scientific_sha, _ = scientific_payload_sha256(root, expected_case_count=expected_count)
    if scientific_sha != receipt_record["scientific_payload_sha256"]:
        raise Qwen3PerturbationAnalysisError(f"{partition} scientific payload mismatch")

    records = []
    for path in paths:
        record = _load_json(path, f"{partition} case")
        if (
            record.get("schema_version") != 1
            or record.get("status") != "completed"
            or record.get("partition") != partition
            or record.get("stage") != "full"
            or record.get("identity") != identity
            or record.get("model") != summary["model"]
        ):
            raise Qwen3PerturbationAnalysisError(f"{partition} case identity mismatch: {path}")
        records.append(record)
    return records


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
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


def _comparison_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        statistics = record["statistics"]
        row = {name: value for name, value in record.items() if name != "statistics"}
        row.update(
            mean_delta=statistics["mean_delta"],
            median_delta=statistics["median_delta"],
            population_standard_deviation=statistics["population_standard_deviation"],
            minimum_delta=statistics["minimum_delta"],
            maximum_delta=statistics["maximum_delta"],
            favor_count=statistics["favor_count"],
            tie_count=statistics["tie_count"],
            oppose_count=statistics["oppose_count"],
            bootstrap_low=statistics["bootstrap_95_interval"]["low"],
            bootstrap_high=statistics["bootstrap_95_interval"]["high"],
        )
        rows.append(row)
    return rows


def _method_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in analysis["method_summaries"]:
        paired = record["paired_primary_vs_fp16"]
        row = {name: value for name, value in record.items() if name not in {"metrics", "paired_primary_vs_fp16"}}
        row.update(
            mean_delta_vs_fp16=None if paired is None else paired["mean_delta"],
            bootstrap_low_vs_fp16=None if paired is None else paired["bootstrap_95_interval"]["low"],
            bootstrap_high_vs_fp16=None if paired is None else paired["bootstrap_95_interval"]["high"],
        )
        rows.append(row)
    return rows


def _layer_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in analysis["layer_summaries"]:
        row = {name: value for name, value in record.items() if name != "metric_means"}
        row.update(record["metric_means"])
        rows.append(row)
    return rows


def _pareto_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"prompt_length": table["prompt_length"], **point}
        for table in analysis["base_pareto_by_prompt_length"]
        for point in table["points"]
    ]


def _format_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B frozen memory--perturbation analysis v1",
        "",
        "This is a deterministic rendering of all 1,300 cases and 46,800 layer records.",
        "The primary local metric is mean `joint_post_o_proj_mse` over 36 layers.",
        "Negative paired delta favors the candidate. Bootstrap intervals describe",
        "stability over the fixed 50-anchor grid, not population significance.",
        "Memory is a packed paper estimate and is not realized CUDA memory.",
        "",
        "## Primary matched-memory comparisons",
        "",
        "| Length | Candidate | Baseline | Memory delta | Mean metric delta | 95% bootstrap interval | F/T/O |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for record in analysis["primary_matched_comparisons"]:
        stats = record["statistics"]
        interval = stats["bootstrap_95_interval"]
        lines.append(
            "| {length} | {candidate} | {baseline} | {memory:+.3%} | {mean:+.6e} | "
            "[{low:+.6e}, {high:+.6e}] | {favor}/{tie}/{oppose} |".format(
                length=record["prompt_length"],
                candidate=record["candidate"],
                baseline=record["baseline"],
                memory=record["candidate_relative_memory_delta_vs_baseline"],
                mean=stats["mean_delta"],
                low=interval["low"],
                high=interval["high"],
                favor=stats["favor_count"],
                tie=stats["tie_count"],
                oppose=stats["oppose_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Exact-byte mechanism comparisons",
            "",
            "| Length | Role | Candidate | Baseline | Mean metric delta | 95% bootstrap interval | F/T/O |",
            "|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for record in analysis["mechanism_comparisons"]:
        stats = record["statistics"]
        interval = stats["bootstrap_95_interval"]
        lines.append(
            "| {length} | {role} | {candidate} | {baseline} | {mean:+.6e} | "
            "[{low:+.6e}, {high:+.6e}] | {favor}/{tie}/{oppose} |".format(
                length=record["prompt_length"],
                role=record["role"],
                candidate=record["candidate"],
                baseline=record["baseline"],
                mean=stats["mean_delta"],
                low=interval["low"],
                high=interval["high"],
                favor=stats["favor_count"],
                tie=stats["tie_count"],
                oppose=stats["oppose_count"],
            )
        )
    lines.extend(["", "## Base-point Pareto sets", ""])
    for table in analysis["base_pareto_by_prompt_length"]:
        methods = ", ".join(f"`{value}`" for value in table["pareto_method_ids"])
        lines.append(f"- Length {table['prompt_length']}: {methods}")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This post-quality analysis measures a local, cache-conditioned perturbation at",
            "the first teacher-forced decode step. It does not replace end-to-end quality",
            "results and does not support latency, throughput, or realized CUDA-memory claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    receipt_path = args.receipt.resolve()
    receipt = _load_json(receipt_path, "results receipt")
    validate_results_receipt(receipt)
    receipt_sha256 = file_sha256(receipt_path)

    resolved = {
        name: _resolve_receipt_path(receipt_path, receipt[name]["path"])
        for name in (
            "perturbation_protocol",
            "quality_protocol",
            "quality_execution",
            "acceptance_gate",
            "input_manifest",
            "full_artifact_manifest",
        )
    }
    for name, path in resolved.items():
        _verify_file(path, receipt[name]["sha256"], name.replace("_", " "))

    perturbation, perturbation_sha = load_perturbation_protocol(
        resolved["perturbation_protocol"]
    )
    quality, quality_sha = load_formal_protocol(resolved["quality_protocol"])
    if perturbation_sha != receipt["perturbation_protocol"]["sha256"]:
        raise Qwen3PerturbationAnalysisError("perturbation protocol hash drift")
    if quality_sha != receipt["quality_protocol"]["sha256"]:
        raise Qwen3PerturbationAnalysisError("quality protocol hash drift")
    _, gate_sha = load_perturbation_acceptance_gate(
        resolved["acceptance_gate"],
        perturbation_protocol_sha256=perturbation_sha,
        verify_artifacts=False,
    )
    if gate_sha != receipt["acceptance_gate"]["sha256"]:
        raise Qwen3PerturbationAnalysisError("acceptance gate hash drift")

    audit_record = receipt["joint_postrun_audit"]
    audit_path = Path(audit_record["path"])
    audit_log_path = Path(audit_record["log_path"])
    archive_path = Path(receipt["archive"]["path"])
    _verify_file(audit_path, audit_record["sha256"], "joint post-run audit")
    _verify_file(audit_log_path, audit_record["log_sha256"], "joint post-run audit log")
    if audit_log_path.stat().st_size != audit_record["log_size_bytes"]:
        raise Qwen3PerturbationAnalysisError("joint post-run audit log size mismatch")
    _verify_file(archive_path, receipt["archive"]["sha256"], "joint backup archive")
    audit = _load_json(audit_path, "joint post-run audit")
    _validate_joint_audit(
        audit,
        receipt,
        artifact_sha256=receipt["full_artifact_manifest"]["sha256"],
    )

    records = []
    for partition in ("cage_qwen3", "kitty_qwen3"):
        records.extend(_load_partition_records(partition, receipt["partitions"][partition]))
    analysis = build_perturbation_analysis(
        perturbation_protocol=perturbation,
        quality_protocol=quality,
        receipt=receipt,
        records=records,
        receipt_sha256=receipt_sha256,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "qwen3_perturbation_analysis_v1.json"
    method_path = output_dir / "qwen3_perturbation_method_length_summary_v1.csv"
    primary_path = output_dir / "qwen3_perturbation_primary_comparisons_v1.csv"
    mechanism_path = output_dir / "qwen3_perturbation_mechanism_comparisons_v1.csv"
    layer_path = output_dir / "qwen3_perturbation_layer_summary_v1.csv"
    pareto_path = output_dir / "qwen3_perturbation_pareto_v1.csv"
    markdown_path = output_dir / "qwen3_perturbation_paper_tables_v1.md"
    manifest_path = output_dir / "qwen3_perturbation_analysis_manifest_v1.json"

    _write_json(analysis_path, analysis)
    _write_csv(
        method_path,
        [
            "method_id", "method_family", "prompt_length", "anchor_count",
            "layer_record_count", "packed_bytes", "is_base_point", "primary_metric",
            "primary_case_statistic", "primary_mean_across_anchors",
            "primary_median_across_anchors", "primary_population_standard_deviation",
            "primary_minimum", "primary_maximum", "mean_delta_vs_fp16",
            "bootstrap_low_vs_fp16", "bootstrap_high_vs_fp16",
        ],
        _method_rows(analysis),
    )
    comparison_fields = [
        "order", "comparison_id", "role", "prompt_length", "candidate", "baseline",
        "candidate_packed_bytes", "baseline_packed_bytes",
        "candidate_relative_memory_delta_vs_baseline", "mean_delta", "median_delta",
        "population_standard_deviation", "minimum_delta", "maximum_delta",
        "favor_count", "tie_count", "oppose_count", "bootstrap_low", "bootstrap_high",
    ]
    _write_csv(primary_path, comparison_fields, _comparison_rows(analysis["primary_matched_comparisons"]))
    _write_csv(mechanism_path, comparison_fields, _comparison_rows(analysis["mechanism_comparisons"]))
    _write_csv(
        layer_path,
        ["method_id", "method_family", "prompt_length", "layer_idx", "anchor_count", *LAYER_METRICS],
        _layer_rows(analysis),
    )
    _write_csv(
        pareto_path,
        ["prompt_length", "method_id", "packed_bytes", "mean_joint_post_o_proj_mse", "is_pareto", "dominated_by"],
        _pareto_rows(analysis),
    )
    markdown_path.write_text(_format_markdown(analysis), encoding="utf-8")

    outputs = [analysis_path, method_path, primary_path, mechanism_path, layer_path, pareto_path, markdown_path]
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "analysis_id": analysis["analysis_id"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "case_count": analysis["case_count"],
        "layer_record_count": analysis["layer_record_count"],
        "method_length_point_count": analysis["method_length_point_count"],
        "outputs": {
            path.name: {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            for path in outputs
        },
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"analysis_manifest: {manifest_path}")
    print(f"analysis_manifest_sha256: {file_sha256(manifest_path)}")


if __name__ == "__main__":
    try:
        main()
    except (Qwen3PerturbationAnalysisError, Qwen3PerturbationError) as error:
        raise SystemExit(f"perturbation analysis failed: {error}") from error
