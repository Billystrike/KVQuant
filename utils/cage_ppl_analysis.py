"""Read-only analysis for the frozen cache-conditioned continuation PPL pilot."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils.cage_ppl import (
    PPL_ANCHOR_INDICES,
    PPL_DECODE_TARGETS,
    PPL_FULL_RAW_METHODS,
    PPL_PROMPT_LENGTHS,
    PPL_SCHEMA_VERSION,
    validate_completed_ppl_case,
)


ANALYSIS_SCHEMA_VERSION = 1
PARETO_PROMPT_LENGTHS = (512, 1024, 2048, 4095)
PPL_METHOD_IDS = tuple(method["id"] for method in PPL_FULL_RAW_METHODS)


class PPLAnalysisError(ValueError):
    """Raised when frozen PPL or Pareto artifacts cannot support analysis."""


def load_completed_ppl_matrix(
    results_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Strictly validate and load the frozen 200-case full PPL matrix."""

    root = Path(results_dir)
    resolved = _read_json(root / "manifest.resolved.json", "resolved manifest")
    quality = _read_json(root / "summary" / "quality.json", "quality summary")
    _validate_resolved_manifest(resolved)

    expanded = resolved["expanded_cases"]
    expected_ids = [item.get("case_id") for item in expanded]
    if len(expected_ids) != 200 or len(set(expected_ids)) != 200:
        raise PPLAnalysisError(
            f"resolved manifest requires 200 unique cases, got {len(expected_ids)} rows "
            f"and {len(set(expected_ids))} unique IDs"
        )
    if any(not isinstance(case_id, str) or not case_id for case_id in expected_ids):
        raise PPLAnalysisError("resolved expanded_cases contains an invalid case_id")
    expected_id_set = set(expected_ids)

    case_ids = {path.stem for path in (root / "cases").glob("*.json")}
    failure_ids = {path.stem for path in (root / "failures").glob("*.json")}
    if case_ids != expected_id_set:
        raise PPLAnalysisError(_id_set_error("case artifacts", expected_id_set, case_ids))
    if failure_ids:
        raise PPLAnalysisError(f"failure artifacts remain: {sorted(failure_ids)}")

    records = []
    for case_id in expected_ids:
        try:
            records.append(validate_completed_ppl_case(root, case_id))
        except ValueError as error:
            raise PPLAnalysisError(f"invalid completed PPL case {case_id}: {error}") from error

    summary_rows = _read_jsonl(root / "summary" / "cases.jsonl")
    summary_ids = [row.get("case_id") for row in summary_rows]
    if len(summary_ids) != 200 or set(summary_ids) != expected_id_set:
        raise PPLAnalysisError(
            _id_set_error("summary/cases.jsonl", expected_id_set, set(summary_ids))
        )
    if len(summary_ids) != len(set(summary_ids)):
        raise PPLAnalysisError("summary/cases.jsonl contains duplicate case IDs")
    record_by_id = {record["case_id"]: record for record in records}
    if any(row != record_by_id[row["case_id"]] for row in summary_rows):
        raise PPLAnalysisError("summary/cases.jsonl differs from completed case artifacts")

    csv_ids = _read_csv_ids(root / "summary" / "cases.csv")
    if len(csv_ids) != 200 or set(csv_ids) != expected_id_set:
        raise PPLAnalysisError(
            _id_set_error("summary/cases.csv", expected_id_set, set(csv_ids))
        )
    if len(csv_ids) != len(set(csv_ids)):
        raise PPLAnalysisError("summary/cases.csv contains duplicate case IDs")

    expanded_by_id = {item["case_id"]: item for item in expanded}
    for record in records:
        declared = expanded_by_id[record["case_id"]]
        if set(declared) != {"case_id", "method", "input"}:
            raise PPLAnalysisError(
                f"expanded case {record['case_id']} has unexpected fields"
            )
        if declared["method"] != record["method"] or declared["input"] != record["input"]:
            raise PPLAnalysisError(
                f"expanded case {record['case_id']} differs from its completed artifact"
            )

    source_state_json = _canonical_json(resolved["source_state"])
    record_states = {
        _canonical_json(record["provenance"]["source_state"]) for record in records
    }
    if record_states != {source_state_json}:
        raise PPLAnalysisError("completed cases differ from resolved source_state")

    _validate_coverage(records)
    tables = aggregate_ppl_results(records, resolved)
    _validate_quality_summary(quality, tables)
    return resolved, records, quality


def aggregate_ppl_results(
    records: Sequence[dict[str, Any]], resolved_manifest: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Build token-weighted summaries and paired candidate-versus-FP16 tables."""

    catalog = {
        method["id"]: {
            "method_id": method["id"],
            "method_order": order,
            "method": method["method"],
            "resolved_config_json": _canonical_json(method["method_config"]),
        }
        for order, method in enumerate(resolved_manifest["methods"])
    }
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        method_id = record["method"]["id"]
        if method_id not in catalog:
            raise PPLAnalysisError(f"record uses undeclared method {method_id!r}")
        by_method[method_id].append(record)

    method_rows = []
    length_rows = []
    anchor_rows = []
    for method_id in catalog:
        metadata = catalog[method_id]
        method_records = by_method[method_id]
        method_rows.append(_nll_row(metadata, method_records))
        for prompt_length in resolved_manifest["prompt_lengths"]:
            length_rows.append(_nll_row(
                metadata,
                [record for record in method_records
                 if record["input"]["prompt_length"] == prompt_length],
                prompt_length=prompt_length,
            ))
        for anchor_index in resolved_manifest["anchor_indices"]:
            anchor_rows.append(_nll_row(
                metadata,
                [record for record in method_records
                 if record["input"]["anchor_index"] == anchor_index],
                anchor_index=anchor_index,
            ))

    _attach_fp16_normalization(method_rows, ())
    _attach_fp16_normalization(length_rows, ("prompt_length",))
    _attach_fp16_normalization(anchor_rows, ("anchor_index",))
    paired_rows = _paired_vs_fp16(records, catalog, resolved_manifest)
    return {
        "method_summary": method_rows,
        "length_summary": length_rows,
        "anchor_summary": anchor_rows,
        "paired_vs_fp16": paired_rows,
    }


def load_pareto_analysis(
    analysis_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the frozen 40-row core Pareto aggregate table for an exact-length join."""

    root = Path(analysis_dir)
    protocol = _read_json(root / "analysis_protocol.json", "Pareto analysis protocol")
    rows = _read_jsonl(root / "aggregate_points.jsonl")
    if protocol.get("aggregate_point_count") != 40 or len(rows) != 40:
        raise PPLAnalysisError("Pareto analysis must contain exactly 40 aggregate points")
    keys = [(row.get("config_id"), row.get("prompt_length")) for row in rows]
    if len(set(keys)) != 40:
        raise PPLAnalysisError("Pareto aggregate points contain duplicate method/length keys")
    if {key[0] for key in keys} != set(PPL_METHOD_IDS):
        raise PPLAnalysisError("Pareto method IDs differ from the PPL full matrix")
    if {key[1] for key in keys} != set(PARETO_PROMPT_LENGTHS):
        raise PPLAnalysisError("Pareto prompt lengths differ from the frozen core pilot")
    required = {
        "config_id", "method", "prompt_length", "paper_total_bytes",
        "paper_total_mib", "primary_error", "primary_error_sample_pstdev",
        "compression_ratio_vs_fp16", "is_pareto_global",
    }
    for row in rows:
        if not required.issubset(row):
            raise PPLAnalysisError(
                f"Pareto row {row.get('config_id')!r} lacks required fields"
            )
        for field in ("paper_total_bytes", "paper_total_mib", "primary_error"):
            _finite(f"Pareto {field}", row[field], minimum=0.0)
    return protocol, rows


def build_joint_selection(
    length_rows: Sequence[dict[str, Any]], pareto_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join only identical prompt lengths; 4032/4095 remain explicitly unmatched."""

    pareto = {
        (row["config_id"], row["prompt_length"]): row for row in pareto_rows
    }
    joined = []
    for ppl in length_rows:
        key = (ppl["method_id"], ppl["prompt_length"])
        memory = pareto.get(key)
        row = {
            "method_id": ppl["method_id"],
            "method_order": ppl["method_order"],
            "method": ppl["method"],
            "ppl_prompt_length": ppl["prompt_length"],
            "pareto_prompt_length": ppl["prompt_length"] if memory else None,
            "join_status": "exact_prompt_length" if memory else "unmatched_4032_vs_4095",
            "case_count": ppl["case_count"],
            "decode_target_count": ppl["decode_target_count"],
            "decode_mean_nll": ppl["decode_mean_nll"],
            "decode_perplexity": ppl["decode_perplexity"],
            "delta_mean_nll_vs_fp16": ppl["delta_mean_nll_vs_fp16"],
            "ppl_ratio_vs_fp16": ppl["ppl_ratio_vs_fp16"],
            "ppl_increase_percent_vs_fp16": ppl["ppl_increase_percent_vs_fp16"],
            "paper_total_bytes": memory["paper_total_bytes"] if memory else None,
            "paper_total_mib": memory["paper_total_mib"] if memory else None,
            "compression_ratio_vs_fp16": (
                memory["compression_ratio_vs_fp16"] if memory else None
            ),
            "primary_error": memory["primary_error"] if memory else None,
            "primary_error_sample_pstdev": (
                memory["primary_error_sample_pstdev"] if memory else None
            ),
            "is_perturbation_pareto": memory["is_pareto_global"] if memory else None,
            "is_memory_ppl_pareto": None,
            "is_joint_memory_perturbation_ppl_pareto": None,
        }
        joined.append(row)

    for prompt_length in sorted(set(PPL_PROMPT_LENGTHS) & set(PARETO_PROMPT_LENGTHS)):
        group = [row for row in joined if row["ppl_prompt_length"] == prompt_length]
        for row in group:
            row["is_memory_ppl_pareto"] = not any(
                _dominates(other, row, ("paper_total_bytes", "ppl_ratio_vs_fp16"))
                for other in group if other is not row
            )
            row["is_joint_memory_perturbation_ppl_pareto"] = not any(
                _dominates(
                    other,
                    row,
                    ("paper_total_bytes", "primary_error", "ppl_ratio_vs_fp16"),
                )
                for other in group if other is not row
            )
    return sorted(joined, key=lambda row: (row["ppl_prompt_length"], row["method_order"]))


def write_ppl_analysis_outputs(
    analysis_dir: str | Path,
    tables: dict[str, list[dict[str, Any]]],
    *,
    resolved_manifest: dict[str, Any],
    quality_summary: dict[str, Any],
    pareto_protocol: dict[str, Any] | None = None,
    pareto_rows: Sequence[dict[str, Any]] | None = None,
    make_plots: bool = True,
) -> list[Path]:
    """Write deterministic PPL tables, optional exact-length joint table, and figures."""

    destination = Path(analysis_dir)
    if destination.exists() and any(destination.iterdir()):
        raise PPLAnalysisError(f"analysis directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for name in ("method_summary", "length_summary", "anchor_summary", "paired_vs_fp16"):
        rows = tables[name]
        output_paths.append(_write_jsonl(destination / f"{name}.jsonl", rows))
        output_paths.append(_write_csv(destination / f"{name}.csv", rows))

    joint_rows = None
    if pareto_rows is not None:
        joint_rows = build_joint_selection(tables["length_summary"], pareto_rows)
        output_paths.append(_write_jsonl(destination / "joint_selection.jsonl", joint_rows))
        output_paths.append(_write_csv(destination / "joint_selection.csv", joint_rows))

    protocol = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "input_ppl_schema_version": PPL_SCHEMA_VERSION,
        "protocol_stage": "full",
        "case_count": 200,
        "input_group_count": 20,
        "primary_target_count": 12600,
        "method_ids": list(PPL_METHOD_IDS),
        "prompt_lengths": list(PPL_PROMPT_LENGTHS),
        "anchor_indices": list(PPL_ANCHOR_INDICES),
        "primary_metric": "cache-dependent decode NLL over 63 targets per case",
        "aggregation": "token-weighted NLL; perplexity is exp(aggregate mean NLL)",
        "dispersion": "population standard deviation across deterministic case means",
        "paired_analysis": "candidate minus FP16 on shared prompt-length/anchor inputs",
        "overall_scope": "diagnostic because each continuation repeats across four lengths",
        "inference_boundary": "five deterministic anchors do not establish significance",
        "fp16_calibration": {
            "cases": quality_summary["fp16_calibration_cases"],
            "max_mean_abs_token_nll_delta": quality_summary[
                "fp16_calibration_max_mean_abs_token_nll_delta"
            ],
            "max_abs_token_nll_delta": quality_summary[
                "fp16_calibration_max_abs_token_nll_delta"
            ],
            "mean_limit": 0.005,
            "max_limit": 0.03,
        },
        "source_state": resolved_manifest["source_state"],
        "pareto_join": None,
    }
    if joint_rows is not None:
        protocol["pareto_join"] = {
            "input_protocol": pareto_protocol,
            "matched_rows": sum(row["join_status"] == "exact_prompt_length" for row in joint_rows),
            "unmatched_rows": sum(row["join_status"] != "exact_prompt_length" for row in joint_rows),
            "rule": "exact prompt length only",
            "unmatched_contexts": (
                "PPL 4032 and perturbation 4095 are retained as unmatched; no silent remapping"
            ),
            "joint_dominance": (
                "computed separately per matched prompt length by minimizing paper bytes, "
                "perturbation MSE, and PPL ratio"
            ),
        }
    protocol_path = destination / "analysis_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_paths.append(protocol_path)

    markdown_path = destination / "ppl_summary.md"
    markdown_path.write_text(_render_markdown(tables, joint_rows), encoding="utf-8")
    output_paths.append(markdown_path)

    if make_plots:
        output_paths.extend(_plot_ppl(destination, tables))
        if joint_rows is not None:
            output_paths.extend(_plot_joint(destination, joint_rows))
    return output_paths


def _validate_resolved_manifest(resolved: dict[str, Any]) -> None:
    if resolved.get("protocol_stage") != "full":
        raise PPLAnalysisError("PPL analysis requires protocol_stage='full'")
    if resolved.get("prompt_lengths") != list(PPL_PROMPT_LENGTHS):
        raise PPLAnalysisError("resolved PPL prompt lengths differ from the full protocol")
    if resolved.get("anchor_indices") != list(PPL_ANCHOR_INDICES):
        raise PPLAnalysisError("resolved PPL anchors differ from the full protocol")
    methods = resolved.get("methods")
    if not isinstance(methods, list) or [method.get("id") for method in methods] != list(PPL_METHOD_IDS):
        raise PPLAnalysisError("resolved PPL methods differ from the full protocol")
    source_state = resolved.get("source_state")
    if not isinstance(source_state, dict) or source_state.get("dirty") is not False:
        raise PPLAnalysisError("resolved PPL source state must be clean")
    if not isinstance(resolved.get("expanded_cases"), list):
        raise PPLAnalysisError("resolved PPL manifest lacks expanded_cases")


def _validate_coverage(records: Sequence[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    input_hashes: dict[tuple[int, int], set[tuple[Any, ...]]] = defaultdict(set)
    for record in records:
        method_id = record["method"]["id"]
        prompt_length = record["input"]["prompt_length"]
        anchor_index = record["input"]["anchor_index"]
        groups[(method_id, prompt_length, anchor_index)].append(record)
        input_record = record["input"]
        input_hashes[(prompt_length, anchor_index)].add((
            input_record["continuation_start"], input_record["prompt_ids_sha256"],
            input_record["continuation_ids_sha256"], input_record["full_ids_sha256"],
        ))
    expected = {
        (method_id, prompt_length, anchor_index)
        for method_id in PPL_METHOD_IDS
        for prompt_length in PPL_PROMPT_LENGTHS
        for anchor_index in PPL_ANCHOR_INDICES
    }
    if set(groups) != expected or any(len(group) != 1 for group in groups.values()):
        raise PPLAnalysisError("PPL method/length/anchor coverage is incomplete or duplicated")
    if any(len(values) != 1 for values in input_hashes.values()):
        raise PPLAnalysisError("methods do not share identical PPL inputs within a cell")


def _validate_quality_summary(
    quality: dict[str, Any], tables: dict[str, list[dict[str, Any]]]
) -> None:
    expected = {
        "schema_version": PPL_SCHEMA_VERSION,
        "protocol_stage": "full",
        "expected_cases": 200,
        "completed_cases": 200,
        "failure_records": 0,
        "completion_gate": "PASS",
        "quality_gate": "NOT_APPLICABLE",
        "fp16_calibration_cases": 20,
    }
    for field, value in expected.items():
        if quality.get(field) != value:
            raise PPLAnalysisError(f"quality summary {field} differs: {quality.get(field)!r}")
    method_quality = {row["method_id"]: row for row in quality.get("method_quality", [])}
    for row in tables["method_summary"]:
        actual = method_quality.get(row["method_id"])
        if actual is None:
            raise PPLAnalysisError(f"quality summary lacks method {row['method_id']}")
        for field in ("completed_cases", "decode_target_count", "decode_nll_sum", "decode_mean_nll", "decode_perplexity"):
            analysis_field = "case_count" if field == "completed_cases" else field
            if not _equivalent(actual[field], row[analysis_field]):
                raise PPLAnalysisError(
                    f"quality method {row['method_id']} field {field} differs from cases"
                )
    length_quality = {
        (row["method_id"], row["prompt_length"]): row
        for row in quality.get("length_quality", [])
    }
    for row in tables["length_summary"]:
        key = (row["method_id"], row["prompt_length"])
        actual = length_quality.get(key)
        if actual is None:
            raise PPLAnalysisError(f"quality summary lacks method/length {key}")
        for field in ("completed_cases", "decode_target_count", "decode_nll_sum", "decode_mean_nll", "decode_perplexity"):
            analysis_field = "case_count" if field == "completed_cases" else field
            if not _equivalent(actual[field], row[analysis_field]):
                raise PPLAnalysisError(f"quality method/length {key} field {field} differs from cases")
    calibration_mean = _finite(
        "quality FP16 max mean calibration delta",
        quality.get("fp16_calibration_max_mean_abs_token_nll_delta"),
        minimum=0.0,
    )
    calibration_max = _finite(
        "quality FP16 max token calibration delta",
        quality.get("fp16_calibration_max_abs_token_nll_delta"),
        minimum=0.0,
    )
    if calibration_mean > 0.005 or calibration_max > 0.03:
        raise PPLAnalysisError(
            "FP16 incremental-versus-one-shot calibration exceeds the frozen limits"
        )


def _nll_row(
    metadata: dict[str, Any], records: Sequence[dict[str, Any]],
    *, prompt_length: int | None = None, anchor_index: int | None = None,
) -> dict[str, Any]:
    if not records:
        raise PPLAnalysisError(f"empty PPL aggregate for {metadata['method_id']}")
    target_count = sum(record["scoring"]["decode_target_count"] for record in records)
    nll_sum = math.fsum(record["scoring"]["decode_nll_sum"] for record in records)
    mean_nll = nll_sum / target_count
    case_means = [record["scoring"]["decode_mean_nll"] for record in records]
    row = {
        **metadata,
        "case_count": len(records),
        "decode_target_count": target_count,
        "decode_nll_sum": nll_sum,
        "decode_mean_nll": mean_nll,
        "decode_perplexity": math.exp(mean_nll),
        "case_mean_nll_pstdev": statistics.pstdev(case_means),
        "case_mean_nll_min": min(case_means),
        "case_mean_nll_max": max(case_means),
    }
    if prompt_length is not None:
        row["prompt_length"] = prompt_length
    if anchor_index is not None:
        row["anchor_index"] = anchor_index
    return row


def _attach_fp16_normalization(
    rows: Sequence[dict[str, Any]], group_fields: tuple[str, ...]
) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    for key, group in groups.items():
        baselines = [row for row in group if row["method_id"] == "fp16"]
        if len(baselines) != 1:
            raise PPLAnalysisError(f"PPL group {key} requires one FP16 baseline")
        baseline = baselines[0]
        for row in group:
            delta = row["decode_mean_nll"] - baseline["decode_mean_nll"]
            ratio = math.exp(delta)
            row["fp16_decode_mean_nll"] = baseline["decode_mean_nll"]
            row["fp16_decode_perplexity"] = baseline["decode_perplexity"]
            row["delta_mean_nll_vs_fp16"] = delta
            row["ppl_ratio_vs_fp16"] = ratio
            row["ppl_increase_percent_vs_fp16"] = 100.0 * (ratio - 1.0)


def _paired_vs_fp16(
    records: Sequence[dict[str, Any]], catalog: dict[str, dict[str, Any]],
    resolved_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    by_method_input = {
        (record["method"]["id"], record["input"]["prompt_length"], record["input"]["anchor_index"]): record
        for record in records
    }
    rows = []
    for method_id in catalog:
        if method_id == "fp16":
            continue
        for prompt_length in (None, *resolved_manifest["prompt_lengths"]):
            pairs = []
            for length in resolved_manifest["prompt_lengths"]:
                if prompt_length is not None and length != prompt_length:
                    continue
                for anchor_index in resolved_manifest["anchor_indices"]:
                    baseline = by_method_input[("fp16", length, anchor_index)]
                    candidate = by_method_input[(method_id, length, anchor_index)]
                    pairs.append(
                        candidate["scoring"]["decode_mean_nll"]
                        - baseline["scoring"]["decode_mean_nll"]
                    )
            mean_delta = statistics.fmean(pairs)
            row = {
                **catalog[method_id],
                "scope": "overall_diagnostic" if prompt_length is None else "prompt_length",
                "prompt_length": prompt_length,
                "pair_count": len(pairs),
                "mean_paired_delta_nll": mean_delta,
                "paired_delta_nll_pstdev": statistics.pstdev(pairs),
                "paired_delta_nll_min": min(pairs),
                "paired_delta_nll_max": max(pairs),
                "ppl_ratio_from_mean_paired_delta": math.exp(mean_delta),
                "candidate_better_pairs": sum(delta < 0 for delta in pairs),
                "equal_pairs": sum(delta == 0 for delta in pairs),
                "candidate_worse_pairs": sum(delta > 0 for delta in pairs),
            }
            rows.append(row)
    return rows


def _dominates(left: dict[str, Any], right: dict[str, Any], fields: Sequence[str]) -> bool:
    return all(left[field] <= right[field] for field in fields) and any(
        left[field] < right[field] for field in fields
    )


def _render_markdown(
    tables: dict[str, list[dict[str, Any]]],
    joint_rows: Sequence[dict[str, Any]] | None,
) -> str:
    lines = [
        "# CAGE-KV cache-conditioned continuation PPL pilot",
        "",
        "Primary values aggregate the 63 cache-dependent decode targets in each case. "
        "Perplexity is `exp(token-weighted mean NLL)`.",
        "",
        "## Overall diagnostic",
        "",
        "| Method | Cases | PPL | ΔNLL vs FP16 | PPL ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in tables["method_summary"]:
        lines.append(
            f"| {row['method_id']} | {row['case_count']} | {row['decode_perplexity']:.6f} | "
            f"{row['delta_mean_nll_vs_fp16']:+.6f} | {row['ppl_ratio_vs_fp16']:.6f}× |"
        )
    lines.extend([
        "",
        "Overall values are diagnostic because each continuation is repeated under four prompt lengths.",
        "",
        "## Length-stratified PPL ratio vs FP16",
        "",
        "| Method | 512 | 1024 | 2048 | 4032 |",
        "|---|---:|---:|---:|---:|",
    ])
    lookup = {(row["method_id"], row["prompt_length"]): row for row in tables["length_summary"]}
    for method_id in PPL_METHOD_IDS:
        values = [lookup[(method_id, length)]["ppl_ratio_vs_fp16"] for length in PPL_PROMPT_LENGTHS]
        lines.append(f"| {method_id} | " + " | ".join(f"{value:.6f}×" for value in values) + " |")
    if joint_rows is not None:
        matched = [row for row in joint_rows if row["join_status"] == "exact_prompt_length"]
        lines.extend([
            "",
            "## Exact-length joint selection",
            "",
            f"The joint table contains {len(matched)} exact-length rows at 512, 1024, and 2048. "
            "PPL length 4032 is not silently mapped to the perturbation pilot's length 4095.",
            "",
            "`joint_selection.csv` retains memory, perturbation, and PPL objectives and reports "
            "two- and three-objective prompt-local Pareto membership.",
        ])
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The five anchors are deterministic and do not establish population-level significance. "
        "This is cache-conditioned continuation PPL, not canonical full-corpus WikiText-2 PPL.",
        "",
        "CAGE uses the fake-quant path. Runtime and CUDA peaks are diagnostics, not compressed-kernel performance claims.",
    ])
    return "\n".join(lines) + "\n"


def _plot_ppl(destination: Path, tables: dict[str, list[dict[str, Any]]]) -> list[Path]:
    plt = _matplotlib()
    methods = tables["method_summary"]
    lengths = tables["length_summary"]
    quantized_ids = [method_id for method_id in PPL_METHOD_IDS if method_id != "fp16"]
    colors = _method_colors()

    figure, (overall_axis, heatmap_axis) = plt.subplots(
        1, 2, figsize=(14.5, 6.3), gridspec_kw={"width_ratios": [0.9, 1.55]}
    )
    quantized = [row for row in methods if row["method_id"] != "fp16"]
    y_positions = list(range(len(quantized)))
    ratios = [row["ppl_ratio_vs_fp16"] for row in quantized]
    overall_axis.scatter(
        ratios, y_positions,
        c=[colors[row["method_id"]] for row in quantized], s=55, edgecolors="#222222",
    )
    overall_axis.axvline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    overall_axis.set_yticks(y_positions, [row["method_id"] for row in quantized])
    overall_axis.invert_yaxis()
    overall_axis.set_xlabel("PPL ratio vs FP16")
    overall_axis.set_title("Overall diagnostic (20 cases per method)")
    overall_axis.grid(True, axis="x", alpha=0.25)
    for y, row in zip(y_positions, quantized):
        overall_axis.annotate(
            f"{row['ppl_ratio_vs_fp16']:.3f}×", (row["ppl_ratio_vs_fp16"], y),
            xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.5,
        )

    lookup = {(row["method_id"], row["prompt_length"]): row for row in lengths}
    matrix = [[lookup[(method_id, length)]["ppl_ratio_vs_fp16"] for length in PPL_PROMPT_LENGTHS]
              for method_id in quantized_ids]
    image = heatmap_axis.imshow(matrix, cmap="YlOrBr", vmin=1.0, vmax=max(map(max, matrix)), aspect="auto")
    heatmap_axis.set_xticks(range(4), PPL_PROMPT_LENGTHS)
    heatmap_axis.set_yticks(range(len(quantized_ids)), quantized_ids)
    heatmap_axis.set_xlabel("Prompt length")
    heatmap_axis.set_title("Length-stratified PPL ratio vs FP16")
    for row_index, values in enumerate(matrix):
        for column_index, value in enumerate(values):
            normalized = (value - 1.0) / max(max(map(max, matrix)) - 1.0, 1e-12)
            heatmap_axis.text(
                column_index, row_index, f"{value:.3f}×", ha="center", va="center",
                fontsize=7.5, color="#ffffff" if normalized > 0.62 else "#111111",
            )
    colorbar = figure.colorbar(image, ax=heatmap_axis, fraction=0.04, pad=0.025)
    colorbar.set_label("PPL ratio")
    figure.suptitle(
        "CAGE-KV cache-conditioned continuation PPL pilot\n"
        "Five deterministic WikiText-2 anchors; overall panel is diagnostic",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    paths = _save_figure(figure, destination / "ppl_quality", plt)

    paired = [row for row in tables["paired_vs_fp16"] if row["scope"] == "prompt_length"]
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 10.2), squeeze=False)
    for axis, prompt_length in zip(axes.flat, PPL_PROMPT_LENGTHS):
        rows = [row for row in paired if row["prompt_length"] == prompt_length]
        x = list(range(len(rows)))
        means = [row["mean_paired_delta_nll"] for row in rows]
        errors = [row["paired_delta_nll_pstdev"] for row in rows]
        axis.errorbar(x, means, yerr=errors, fmt="none", color="#777777", capsize=3, linewidth=0.9)
        axis.scatter(x, means, c=[colors[row["method_id"]] for row in rows], s=50, edgecolors="#222222", zorder=3)
        axis.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
        axis.set_xticks(x, [row["method_id"] for row in rows], rotation=40, ha="right", fontsize=7.5)
        axis.set_ylabel("Paired ΔNLL (candidate − FP16)")
        axis.set_title(f"Prompt length {prompt_length}")
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle(
        "Paired cache-dependent NLL shifts across five deterministic anchors\n"
        "Error bars are ±1 population σ across paired anchor differences",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    paths.extend(_save_figure(figure, destination / "ppl_paired_delta", plt))
    return paths


def _plot_joint(destination: Path, rows: Sequence[dict[str, Any]]) -> list[Path]:
    plt = _matplotlib()
    matched_lengths = sorted({
        row["ppl_prompt_length"] for row in rows if row["join_status"] == "exact_prompt_length"
    })
    figure, axes = plt.subplots(1, len(matched_lengths), figsize=(15.2, 4.8), squeeze=False)
    colors = _method_colors()
    markers = {"kivi": "o", "cage": "^"}
    for axis, prompt_length in zip(axes.flat, matched_lengths):
        group = [row for row in rows if row["ppl_prompt_length"] == prompt_length]
        fp16 = next(row for row in group if row["method_id"] == "fp16")
        quantized = [row for row in group if row["method_id"] != "fp16"]
        for row in quantized:
            axis.scatter(
                row["paper_total_mib"], row["ppl_ratio_vs_fp16"],
                color=colors[row["method_id"]], marker=markers[row["method"]], s=62,
                edgecolor="#222222" if row["is_perturbation_pareto"] else "none",
                linewidth=0.9, alpha=1.0 if row["is_memory_ppl_pareto"] else 0.4,
            )
        axis.axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
        axis.set_title(f"Prompt length {prompt_length}")
        axis.set_xlabel("Paper-facing packed KV cache (MiB)")
        axis.set_ylabel("PPL ratio vs FP16")
        axis.grid(True, alpha=0.25)
        axis.text(
            0.98, 0.96, f"FP16: {fp16['paper_total_mib']:.1f} MiB, ratio 1.0",
            transform=axis.transAxes, ha="right", va="top", fontsize=7.5,
        )
    figure.suptitle(
        "Exact-length memory–PPL tradeoff\n"
        "Opaque: memory–PPL Pareto; dark outline: perturbation Pareto",
        fontsize=14,
    )
    from matplotlib.lines import Line2D
    quantized = [method_id for method_id in PPL_METHOD_IDS if method_id != "fp16"]
    method_name = {row["method_id"]: row["method"] for row in rows}
    handles = [
        Line2D(
            [0], [0], color="none", marker=markers[method_name[method_id]],
            markerfacecolor=colors[method_id], markeredgecolor="#222222",
            markeredgewidth=0.7, markersize=7,
        )
        for method_id in quantized
    ]
    figure.legend(
        handles, quantized, loc="lower center", ncol=5, fontsize=7.5,
        frameon=False, bbox_to_anchor=(0.5, 0.005),
    )
    figure.tight_layout(rect=(0, 0.12, 1, 0.9))
    return _save_figure(figure, destination / "memory_ppl_tradeoff", plt)


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise PPLAnalysisError("matplotlib is required for plots; install it or pass --no-plots") from error
    return plt


def _save_figure(figure: Any, base: Path, plt: Any) -> list[Path]:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return [png, pdf]


def _method_colors() -> dict[str, str]:
    return {
        "kivi-g32-r32": "#f28e2b", "kivi-g32-r64": "#ff9d4d",
        "kivi-g32-r128": "#ffbe7d", "kivi-g64-r64": "#d55e00",
        "kivi-g64-r128": "#a05a2c", "kivi-g128-r128": "#7f3c0a",
        "cage-r32": "#8cd17d", "cage-r64": "#59a14f", "cage-r128": "#176b3a",
    }


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PPLAnalysisError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PPLAnalysisError(f"{name} {path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise PPLAnalysisError(f"blank line {line_number} in {path}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PPLAnalysisError(f"row {line_number} in {path} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise PPLAnalysisError(f"cannot read JSONL {path}: {error}") from error
    return rows


def _read_csv_ids(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise PPLAnalysisError(f"cannot read CSV {path}: {error}") from error
    if not rows or "case_id" not in rows[0]:
        raise PPLAnalysisError(f"CSV {path} is empty or lacks case_id")
    return [row["case_id"] for row in rows]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    fields = _ordered_fields(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _ordered_fields(rows: Sequence[dict[str, Any]]) -> list[str]:
    fields = sorted({field for row in rows for field in row})
    preferred = [
        "method_id", "method_order", "method", "prompt_length", "anchor_index",
        "ppl_prompt_length", "pareto_prompt_length", "join_status", "case_count",
        "decode_target_count", "decode_mean_nll", "decode_perplexity",
        "delta_mean_nll_vs_fp16", "ppl_ratio_vs_fp16", "paper_total_mib",
        "primary_error", "is_perturbation_pareto", "is_memory_ppl_pareto",
        "is_joint_memory_perturbation_ppl_pareto",
    ]
    ordered = [field for field in preferred if field in fields]
    return ordered + [field for field in fields if field not in ordered]


def _finite(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PPLAnalysisError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise PPLAnalysisError(f"{name} must be a finite real number >= {minimum}")
    return result


def _equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _id_set_error(name: str, expected: set[str], actual: set[str]) -> str:
    return (
        f"{name} differ from expected IDs: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}"
    )


__all__ = [
    "ANALYSIS_SCHEMA_VERSION", "PPLAnalysisError", "aggregate_ppl_results",
    "build_joint_selection", "load_completed_ppl_matrix", "load_pareto_analysis",
    "write_ppl_analysis_outputs",
]
