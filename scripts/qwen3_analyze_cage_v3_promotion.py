#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v3_promotion_analysis import (
    build_promotion_analysis,
    validate_results_receipt,
)
from utils.qwen3_cage_v3_promotion_full import expand_full_holdout_cases, load_full_execution
from utils.qwen3_cage_v3_promotion_full_postrun import build_postrun_audit
from utils.qwen3_cage_v4_data import file_sha256


EXECUTION_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_promotion_full_execution_v1.json"
ARTIFACT_PATH = REPO_ROOT / "configs" / "qwen3_8b_cage_v3_promotion_full_artifacts_v1.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B CAGE-v3 promotion decision",
        "",
        "> Frozen PG-19 validation development-holdout evidence; not PG-19 test, runtime, or a final paper claim.",
        "",
        "## Preregistered promotion gates",
        "",
        "| Baseline | Overall relative PPL (%) | NLL-delta CI95 | Every length noninferior | Gate pass |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = (
        ("cage-v1-kittypro-matched", "CAGE-v1 predecessor"),
        ("kivi-kittypro-matched", "Matched KIVI"),
        ("kitty-pro-25pct", "Kitty-Pro"),
    )
    gate_names = {
        "cage-v1-kittypro-matched": "versus_predecessor",
        "kivi-kittypro-matched": "versus_uniform_baseline",
        "kitty-pro-25pct": "versus_external_baseline",
    }
    for method, label in labels:
        comparison = analysis["comparison_lookup"][method]["overall"]
        ci = comparison["bootstrap_mean_nll_delta_ci95"]
        gate = analysis["promotion_gates"][gate_names[method]]
        lines.append(
            f"| {label} | {comparison['relative_ppl_percent']:+.6f} | "
            f"[{ci[0]:+.9g}, {ci[1]:+.9g}] | "
            f"{gate['all_lengths_noninferiority_pass']} | {gate['pass']} |"
        )
    predecessor = analysis["promotion_gates"]["versus_predecessor"]
    lines.extend(
        [
            "",
            f"CAGE-v1 quality-superiority track: `{predecessor['quality_superiority_track_pass']}`.",
            "",
            f"CAGE-v1 memory-quality Pareto track: `{predecessor['memory_quality_pareto_track_pass']}`.",
            "",
            f"Overall promotion pass: **{analysis['promotion_pass']}**.",
            "",
            f"Decision: `{analysis['candidate_outcome']}`.",
            "",
            "## Per-length paired results",
            "",
            "| Baseline | Length | Relative PPL (%) | CI95 lower | CI95 upper |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method, label in labels:
        for length in analysis["prompt_lengths"]:
            row = analysis["comparison_lookup"][method]["per_length"][str(length)]
            ci = row["bootstrap_mean_nll_delta_ci95"]
            lines.append(
                f"| {label} | {length} | {row['relative_ppl_percent']:+.6f} | "
                f"{ci[0]:+.9g} | {ci[1]:+.9g} |"
            )
    lines.extend(
        [
            "",
            "All five methods, all three lengths, and unfavorable outcomes are retained in the JSON/CSV outputs.",
            "PG-19 test remains unread; no additional CAGE-v3 tuning, Kitty-Llama port, full Llama-2 run, paper claim, or runtime claim is authorized by this analysis alone.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the preregistered CAGE-v3 promotion gate")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    receipt_path = args.receipt.resolve()
    receipt = _load(receipt_path)
    validate_results_receipt(receipt, receipt_path=receipt_path)

    audit_spec = receipt["postrun_audit"]
    audit_path = Path(audit_spec["path"])
    audit_log = Path(audit_spec["execution_log"])
    if file_sha256(audit_path) != audit_spec["sha256"] or audit_path.stat().st_size != audit_spec["size_bytes"]:
        raise RuntimeError("promotion post-run audit differs from frozen receipt")
    if file_sha256(audit_log) != audit_spec["execution_log_sha256"] or audit_log.stat().st_size != audit_spec["execution_log_size_bytes"]:
        raise RuntimeError("promotion post-run log differs from frozen receipt")
    frozen_audit = _load(audit_path)
    rebuilt_audit = build_postrun_audit(ARTIFACT_PATH)
    if rebuilt_audit != frozen_audit:
        raise RuntimeError("promotion artifacts differ from passed post-run audit")
    if frozen_audit.get("status") != "pass" or frozen_audit.get("interpretation_performed") is not False:
        raise RuntimeError("promotion post-run audit is not an uninterpreted pass")

    execution, execution_sha256, protocol, protocol_sha256, _, gate_sha256 = load_full_execution(
        EXECUTION_PATH,
        repo_root=REPO_ROOT,
        verify_server_artifacts=True,
    )
    if execution_sha256 != receipt["frozen_inputs"]["execution_sha256"]:
        raise RuntimeError("promotion execution differs from results receipt")
    manifest_path = Path(execution["input_manifest"]["path"])
    manifest = _load(manifest_path)
    artifacts = _load(ARTIFACT_PATH)

    records = []
    for partition in ("cage_qwen3", "kitty_qwen3"):
        expected_cases = expand_full_holdout_cases(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            gate_sha256=gate_sha256,
            manifest=manifest,
            partition=partition,
            repo_root=REPO_ROOT,
        )
        expected = {case["case_id"]: case for case in expected_cases}
        root = Path(artifacts["partitions"][partition]["output_dir"])
        partition_records = [_load(path) for path in sorted((root / "cases").glob("*.json"))]
        if len(partition_records) != len(expected):
            raise RuntimeError(f"{partition} record count differs from frozen expansion")
        for record in partition_records:
            case = expected.get(record["case_id"])
            if case is None or record.get("method") != case["method"] or record.get("input") != case["input"]:
                raise RuntimeError(f"{partition} case differs from frozen method/input expansion")
        records.extend(partition_records)

    analysis = build_promotion_analysis(
        receipt_sha256=file_sha256(receipt_path),
        protocol=protocol,
        records=records,
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite promotion analysis: {output_dir}")
    output_dir.mkdir(parents=True)

    analysis_path = output_dir / "qwen3_cage_v3_promotion_analysis_v1.json"
    summary_path = output_dir / "qwen3_cage_v3_promotion_method_length_summary_v1.csv"
    comparison_path = output_dir / "qwen3_cage_v3_promotion_paired_comparisons_v1.csv"
    decision_path = output_dir / "qwen3_cage_v3_promotion_decision_v1.md"
    _write_json(analysis_path, analysis)
    _write_csv(
        summary_path,
        analysis["method_length_summaries"],
        (
            "method_id",
            "prompt_length",
            "case_count",
            "document_count",
            "target_token_count",
            "packed_bytes",
            "mean_nll",
            "perplexity",
        ),
    )
    flattened = []
    for row in analysis["candidate_comparisons"]:
        flattened.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"practical_effect", "bootstrap_mean_nll_delta_ci95"}
                },
                "practical_magnitude_label": row["practical_effect"]["magnitude_label"],
                "practical_direction": row["practical_effect"]["direction"],
                "bootstrap_ci95_lower": row["bootstrap_mean_nll_delta_ci95"][0],
                "bootstrap_ci95_upper": row["bootstrap_mean_nll_delta_ci95"][1],
            }
        )
    _write_csv(comparison_path, flattened, tuple(flattened[0]))
    decision_path.write_text(_render_markdown(analysis), encoding="utf-8")

    outputs = {
        path.name: {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
        for path in (analysis_path, summary_path, comparison_path, decision_path)
    }
    result_manifest = {
        "schema_version": 1,
        "status": "pass",
        "analysis_id": analysis["analysis_id"],
        "claim_eligible": False,
        "receipt_sha256": file_sha256(receipt_path),
        "case_count": 600,
        "document_count": 20,
        "target_token_count": 38_400,
        "promotion_pass": analysis["promotion_pass"],
        "candidate_outcome": analysis["candidate_outcome"],
        "pg19_test_accessed": False,
        "additional_v3_tuning_performed": False,
        "analysis_source_sha256": {
            "runner": file_sha256(Path(__file__).resolve()),
            "analysis_utils": file_sha256(
                REPO_ROOT / "utils" / "qwen3_cage_v3_promotion_analysis.py"
            ),
            "statistical_utils": file_sha256(
                REPO_ROOT / "utils" / "qwen3_cage_v4_analysis.py"
            ),
        },
        "outputs": outputs,
    }
    result_manifest_path = output_dir / "qwen3_cage_v3_promotion_analysis_manifest_v1.json"
    _write_json(result_manifest_path, result_manifest)
    print(json.dumps(result_manifest, indent=2, sort_keys=True, allow_nan=False))
    print(decision_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
