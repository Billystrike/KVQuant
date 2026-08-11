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

from utils.qwen3_cage_v4_analysis import (
    build_metric_screen_analysis,
    validate_results_receipt,
)
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256
from utils.qwen3_cage_v4_execution import load_screen_execution
from utils.qwen3_cage_v4_postrun import (
    expected_memory_report,
    load_json,
    validate_artifact_manifest,
    validate_attempt_manifest,
    validate_partition,
)


EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_screen_execution_v1.json"
ARTIFACT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_screen_artifacts_v1.json"
ATTEMPT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_metric_screen_postrun_attempts_v1.json"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _render_markdown(analysis: dict) -> str:
    proxy = analysis["local_proxy_validity"]
    lines = [
        "# Qwen3-8B CAGE-v4 PG-19 metric-validity screen",
        "",
        "> Development-screen evidence only; not a holdout, runtime, or final CAGE-v4 claim.",
        "",
        "## Local-proxy validity gate",
        "",
        "| Length | Spearman | Pass |",
        "|---:|---:|---:|",
    ]
    for row in proxy["per_length"]:
        lines.append(
            f"| {row['prompt_length']} | {row['spearman']:.6f} | {row['spearman_pass']} |"
        )
    lines.extend(
        [
            "",
            f"All-method pair concordance: {proxy['all_method_pair_concordant_count']}/"
            f"{proxy['all_method_pair_count']} ({proxy['all_method_pair_concordance']:.6f}).",
            "",
            f"Frontier pair concordance: {proxy['frontier_pair_concordant_count']}/"
            f"{proxy['frontier_pair_count']} ({proxy['frontier_pair_concordance']:.6f}).",
            "",
            f"Gate pass: **{proxy['gate_pass']}**.",
            "",
            f"Resulting local-MSE role: `{proxy['resulting_role']}`.",
            "",
            "## Method-length summary",
            "",
            "| Method | Length | Packed bytes | Mean NLL | PPL | Local MSE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["method_length_summaries"]:
        local = "—" if row["mean_joint_post_o_proj_mse"] is None else f"{row['mean_joint_post_o_proj_mse']:.9g}"
        lines.append(
            f"| {row['method_id']} | {row['prompt_length']} | {row['packed_bytes']} | "
            f"{row['mean_nll']:.9g} | {row['perplexity']:.9g} | {local} |"
        )
    lines.extend(
        [
            "",
            "All preregistered pairwise comparisons, practical-effect labels, and bootstrap intervals "
            "are retained in the JSON and CSV outputs.",
            "",
            "No CAGE-v4 candidate was selected or executed, and PG-19 holdout/test remained unread.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the frozen CAGE-v4 PG-19 metric screen")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.receipt.resolve()
    receipt = load_json(receipt_path)
    validate_results_receipt(receipt, receipt_path=receipt_path)

    audit_spec = receipt["postrun_audit"]
    audit_path = Path(audit_spec["path"])
    audit_log = Path(audit_spec["execution_log"])
    if file_sha256(audit_path) != audit_spec["sha256"] or audit_path.stat().st_size != audit_spec["size_bytes"]:
        raise RuntimeError("postrun audit differs from results receipt")
    if file_sha256(audit_log) != audit_spec["execution_log_sha256"]:
        raise RuntimeError("postrun audit log hash differs from receipt")
    if audit_log.stat().st_size != audit_spec["execution_log_size_bytes"]:
        raise RuntimeError("postrun audit log size differs from receipt")
    audit = load_json(audit_path)
    if audit.get("status") != "pass" or audit.get("interpretation_performed") is not False:
        raise RuntimeError("postrun audit is not an uninterpreted pass")

    execution, execution_sha256, _, protocol_sha256, _, expanded = load_screen_execution(
        EXECUTION_PATH, repo_root=REPO_ROOT, verify_artifacts=True
    )
    if expanded is None:
        raise RuntimeError("metric screen cases were not expanded")
    artifacts = load_json(ARTIFACT_PATH)
    validate_artifact_manifest(
        artifacts,
        execution_sha256=execution_sha256,
        protocol_sha256=protocol_sha256,
        input_manifest_sha256=execution["input_manifest"]["sha256"],
        gate_sha256=execution["acceptance_gate"]["sha256"],
    )
    attempts = load_json(ATTEMPT_PATH)
    validate_attempt_manifest(attempts, verify_artifacts=True)

    records = []
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
        payload = report.pop("scientific_payload")
        for name, value in report.items():
            if audit["partitions"][partition].get(name) != value:
                raise RuntimeError(f"{partition} differs from passed postrun audit: {name}")
        root = Path(artifacts["partitions"][partition]["directory"])
        records.extend(load_json(path) for path in sorted((root / "cases").glob("*.json")))
        joint_payload.extend({"partition": partition, **row} for row in payload)
    joint_payload.sort(key=lambda row: (row["partition"], row["case_id"]))
    if canonical_sha256(joint_payload) != audit["joint_scientific_payload_sha256"]:
        raise RuntimeError("joint scientific payload differs from passed postrun audit")

    analysis = build_metric_screen_analysis(
        receipt_sha256=file_sha256(receipt_path), records=records
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite metric screen analysis: {output_dir}")
    output_dir.mkdir(parents=True)
    analysis_path = output_dir / "qwen3_cage_v4_metric_screen_analysis_v1.json"
    summary_path = output_dir / "qwen3_cage_v4_method_length_summary_v1.csv"
    comparison_path = output_dir / "qwen3_cage_v4_paired_nll_comparisons_v1.csv"
    proxy_path = output_dir / "qwen3_cage_v4_local_proxy_validity_v1.csv"
    markdown_path = output_dir / "qwen3_cage_v4_metric_screen_analysis_v1.md"
    _write_json(analysis_path, analysis)
    _write_csv(
        summary_path,
        analysis["method_length_summaries"],
        [
            "method_id",
            "prompt_length",
            "case_count",
            "document_count",
            "target_token_count",
            "packed_bytes",
            "mean_nll",
            "perplexity",
            "mean_joint_post_o_proj_mse",
        ],
    )
    comparison_rows = []
    for row in analysis["paired_nll_comparisons"]:
        comparison_rows.append(
            {
                **{key: value for key, value in row.items() if key not in {"practical_effect", "bootstrap_mean_nll_delta_ci95"}},
                "practical_magnitude_label": row["practical_effect"]["magnitude_label"],
                "practical_direction": row["practical_effect"]["direction"],
                "bootstrap_ci95_lower": row["bootstrap_mean_nll_delta_ci95"][0],
                "bootstrap_ci95_upper": row["bootstrap_mean_nll_delta_ci95"][1],
            }
        )
    comparison_fields = list(comparison_rows[0])
    _write_csv(comparison_path, comparison_rows, comparison_fields)
    proxy_rows = [
        {
            "prompt_length": row["prompt_length"],
            "spearman": row["spearman"],
            "spearman_pass": row["spearman_pass"],
            "all_method_pair_count": row["all_method_pairs"]["pair_count"],
            "all_method_pair_concordant_count": row["all_method_pairs"]["concordant_count"],
            "all_method_pair_concordance": row["all_method_pairs"]["concordance"],
            "frontier_pair_count": row["frontier_pairs"]["pair_count"],
            "frontier_pair_concordant_count": row["frontier_pairs"]["concordant_count"],
            "frontier_pair_concordance": row["frontier_pairs"]["concordance"],
        }
        for row in analysis["local_proxy_validity"]["per_length"]
    ]
    _write_csv(proxy_path, proxy_rows, list(proxy_rows[0]))
    markdown_path.write_text(_render_markdown(analysis), encoding="utf-8")
    outputs = {}
    for path in (analysis_path, summary_path, comparison_path, proxy_path, markdown_path):
        outputs[path.name] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "analysis_id": analysis["analysis_id"],
        "claim_eligible": False,
        "receipt_sha256": file_sha256(receipt_path),
        "case_count": 720,
        "document_count": 20,
        "target_token_count": 46_080,
        "local_proxy_gate_pass": analysis["local_proxy_validity"]["gate_pass"],
        "local_proxy_resulting_role": analysis["local_proxy_validity"]["resulting_role"],
        "candidate_selection_performed": False,
        "holdout_accessed": False,
        "outputs": outputs,
    }
    manifest_path = output_dir / "qwen3_cage_v4_metric_screen_analysis_manifest_v1.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
