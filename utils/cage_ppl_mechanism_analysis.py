"""Pre-registered paired analysis for the CAGE PPL mechanism ablation."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils.cage_experiment_config import resolve_method
from utils.cage_ppl import (
    PPL_MECHANISM_ABLATION_RAW_METHODS,
    PPL_PAIRED_ANCHOR_INDICES,
    PPL_PROMPT_LENGTHS,
    PPL_SCHEMA_VERSION,
    validate_completed_ppl_case,
)


ANALYSIS_SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 20260726
BOOTSTRAP_RESAMPLES = 10_000
METHOD_IDS = tuple(method["id"] for method in PPL_MECHANISM_ABLATION_RAW_METHODS)
RESIDUALS = (64, 128)
BASELINE_ROLES = ("fixed-random", "uniform", "k-adaptive", "v-adaptive")
FACTORIAL_EFFECTS = (
    "full_minus_uniform",
    "key_adaptive_minus_uniform",
    "value_adaptive_minus_uniform",
    "interaction_full_minus_key_minus_value_plus_uniform",
)


class MechanismPPLAnalysisError(ValueError):
    """Raised when the frozen 2,000-case study fails strict validation."""


def load_completed_mechanism_matrix(
    results_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate and load the frozen full mechanism-ablation matrix."""

    root = Path(results_dir)
    resolved = _read_json(root / "manifest.resolved.json", "resolved manifest")
    quality = _read_json(root / "summary" / "quality.json", "quality summary")
    _validate_resolved(resolved)
    expanded = resolved["expanded_cases"]
    expected_ids = [item.get("case_id") for item in expanded]
    if len(expected_ids) != 2000 or len(set(expected_ids)) != 2000:
        raise MechanismPPLAnalysisError(
            f"mechanism matrix requires 2000 unique IDs, got {len(expected_ids)} rows "
            f"and {len(set(expected_ids))} unique IDs"
        )
    if any(not isinstance(case_id, str) or not case_id for case_id in expected_ids):
        raise MechanismPPLAnalysisError("expanded_cases contains an invalid case_id")
    expected_set = set(expected_ids)

    case_ids = {path.stem for path in (root / "cases").glob("*.json")}
    failure_ids = {path.stem for path in (root / "failures").glob("*.json")}
    if case_ids != expected_set:
        raise MechanismPPLAnalysisError(_id_set_error("case artifacts", expected_set, case_ids))
    if failure_ids:
        raise MechanismPPLAnalysisError(f"failure artifacts remain: {sorted(failure_ids)}")

    records = []
    for case_id in expected_ids:
        try:
            records.append(validate_completed_ppl_case(root, case_id))
        except ValueError as error:
            raise MechanismPPLAnalysisError(f"invalid completed case {case_id}: {error}") from error

    summary = _read_jsonl(root / "summary" / "cases.jsonl")
    summary_ids = [row.get("case_id") for row in summary]
    if len(summary_ids) != 2000 or set(summary_ids) != expected_set:
        raise MechanismPPLAnalysisError(
            _id_set_error("summary/cases.jsonl", expected_set, set(summary_ids))
        )
    if len(summary_ids) != len(set(summary_ids)):
        raise MechanismPPLAnalysisError("summary/cases.jsonl contains duplicate IDs")
    by_id = {record["case_id"]: record for record in records}
    if any(row != by_id[row["case_id"]] for row in summary):
        raise MechanismPPLAnalysisError("summary/cases.jsonl differs from case artifacts")

    csv_ids = _read_csv_ids(root / "summary" / "cases.csv")
    if len(csv_ids) != 2000 or set(csv_ids) != expected_set:
        raise MechanismPPLAnalysisError(
            _id_set_error("summary/cases.csv", expected_set, set(csv_ids))
        )
    if len(csv_ids) != len(set(csv_ids)):
        raise MechanismPPLAnalysisError("summary/cases.csv contains duplicate IDs")

    expanded_by_id = {item["case_id"]: item for item in expanded}
    for record in records:
        declared = expanded_by_id[record["case_id"]]
        if declared["method"] != record["method"] or declared["input"] != record["input"]:
            raise MechanismPPLAnalysisError(
                f"expanded case {record['case_id']} differs from completed artifact"
            )
    resolved_state = _canonical_json(resolved["source_state"])
    states = {_canonical_json(record["provenance"]["source_state"]) for record in records}
    if states != {resolved_state}:
        raise MechanismPPLAnalysisError("completed cases differ from resolved source state")

    _validate_coverage(records)
    tables = aggregate_mechanism_results(records, resolved)
    _validate_quality(quality, tables)
    return resolved, records, quality


def aggregate_mechanism_results(
    records: Sequence[dict[str, Any]], resolved: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate methods, pre-registered pairs, and the 2x2 decomposition."""

    _validate_resolved(resolved)
    if len(records) != 2000:
        raise MechanismPPLAnalysisError(
            f"mechanism aggregation requires 2000 cases, got {len(records)}"
        )
    _validate_coverage(records)
    catalog = {
        method["id"]: {
            "method_id": method["id"],
            "method_order": order,
            "method": method["method"],
            "resolved_config_json": _canonical_json(method["method_config"]),
        }
        for order, method in enumerate(resolved["methods"])
    }
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_method[record["method"]["id"]].append(record)

    method_rows = [_nll_row(catalog[method_id], by_method[method_id]) for method_id in catalog]
    length_rows = []
    for method_id in catalog:
        for prompt_length in PPL_PROMPT_LENGTHS:
            length_rows.append(_nll_row(
                catalog[method_id],
                [
                    record for record in by_method[method_id]
                    if record["input"]["prompt_length"] == prompt_length
                ],
                prompt_length=prompt_length,
            ))

    lookup = {
        (
            record["method"]["id"],
            record["input"]["prompt_length"],
            record["input"]["anchor_index"],
        ): record
        for record in records
    }
    anchor_rows = _anchor_delta_rows(lookup)
    paired_rows = _paired_rows(anchor_rows, method_rows, length_rows)
    factorial_anchor_rows = _factorial_anchor_rows(lookup)
    factorial_rows = _factorial_rows(factorial_anchor_rows)
    return {
        "method_summary": method_rows,
        "length_summary": length_rows,
        "paired_comparisons": paired_rows,
        "anchor_deltas": anchor_rows,
        "factorial_summary": factorial_rows,
        "factorial_anchor_effects": factorial_anchor_rows,
    }


def write_mechanism_analysis_outputs(
    analysis_dir: str | Path,
    tables: dict[str, list[dict[str, Any]]],
    *,
    resolved_manifest: dict[str, Any],
    quality_summary: dict[str, Any],
    make_plots: bool = True,
) -> list[Path]:
    """Write frozen tables, metadata, Markdown, and paper-facing figures."""

    destination = Path(analysis_dir)
    if destination.exists() and any(destination.iterdir()):
        raise MechanismPPLAnalysisError(f"analysis directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    names = (
        "method_summary", "length_summary", "paired_comparisons",
        "anchor_deltas", "factorial_summary", "factorial_anchor_effects",
    )
    for name in names:
        rows = tables[name]
        outputs.append(_write_jsonl(destination / f"{name}.jsonl", rows))
        outputs.append(_write_csv(destination / f"{name}.csv", rows))

    protocol = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "input_ppl_schema_version": PPL_SCHEMA_VERSION,
        "protocol_stage": "mechanism_ablation_full",
        "case_count": 2000,
        "input_group_count": 200,
        "anchor_count": 50,
        "primary_target_count": 126_000,
        "method_ids": list(METHOD_IDS),
        "prompt_lengths": list(PPL_PROMPT_LENGTHS),
        "primary_comparisons": [
            {
                "residual_length": residual,
                "candidate_method_id": f"cage-r{residual}-full",
                "baseline_method_id": f"cage-r{residual}-{role}",
                "direction": "full minus baseline; negative favors full",
            }
            for residual in RESIDUALS for role in BASELINE_ROLES
        ],
        "factorial_effects": list(FACTORIAL_EFFECTS),
        "anchor_cluster": (
            "one anchor is the resampling unit; overall first averages the four "
            "prompt lengths within each anchor"
        ),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "interval": "2.5th and 97.5th percentiles of paired anchor-cluster means",
            "role": "descriptive stability interval over a deterministic corpus grid",
        },
        "inference_boundary": (
            "systematic deterministic anchors are not random population draws; "
            "bootstrap intervals do not establish population-level significance"
        ),
        "fp16_role": "not rerun; within-CAGE mechanism contrasts only",
        "quality_gate": quality_summary["quality_gate"],
        "source_state": resolved_manifest["source_state"],
    }
    protocol_path = destination / "analysis_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs.append(protocol_path)
    summary_path = destination / "ppl_mechanism_summary.md"
    summary_path.write_text(_render_markdown(tables), encoding="utf-8")
    outputs.append(summary_path)
    if make_plots:
        outputs.extend(_plot_mechanism(destination, tables))
    return outputs


def _validate_resolved(resolved: dict[str, Any]) -> None:
    if resolved.get("protocol_stage") != "mechanism_ablation_full":
        raise MechanismPPLAnalysisError(
            "mechanism analysis requires protocol_stage='mechanism_ablation_full'"
        )
    if resolved.get("prompt_lengths") != list(PPL_PROMPT_LENGTHS):
        raise MechanismPPLAnalysisError("prompt lengths differ from protocol")
    if resolved.get("anchor_indices") != list(PPL_PAIRED_ANCHOR_INDICES):
        raise MechanismPPLAnalysisError("anchors differ from protocol")
    expected = [
        resolve_method(method, index)
        for index, method in enumerate(PPL_MECHANISM_ABLATION_RAW_METHODS)
    ]
    if resolved.get("methods") != expected:
        raise MechanismPPLAnalysisError("methods differ from the frozen mechanism protocol")
    state = resolved.get("source_state")
    if not isinstance(state, dict) or state.get("dirty") is not False:
        raise MechanismPPLAnalysisError("source state must be clean")
    if not isinstance(resolved.get("expanded_cases"), list):
        raise MechanismPPLAnalysisError("resolved manifest lacks expanded_cases")


def _validate_coverage(records: Sequence[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    shared: dict[tuple[int, int], set[tuple[Any, ...]]] = defaultdict(set)
    for record in records:
        input_record = record["input"]
        key = (
            record["method"]["id"], input_record["prompt_length"],
            input_record["anchor_index"],
        )
        groups[key].append(record)
        shared[(input_record["prompt_length"], input_record["anchor_index"])].add((
            input_record["continuation_start"], input_record["prompt_ids_sha256"],
            input_record["continuation_ids_sha256"], input_record["full_ids_sha256"],
        ))
    expected = {
        (method_id, length, anchor)
        for method_id in METHOD_IDS
        for length in PPL_PROMPT_LENGTHS
        for anchor in PPL_PAIRED_ANCHOR_INDICES
    }
    if set(groups) != expected or any(len(group) != 1 for group in groups.values()):
        raise MechanismPPLAnalysisError("method/length/anchor coverage is invalid")
    if len(shared) != 200 or any(len(values) != 1 for values in shared.values()):
        raise MechanismPPLAnalysisError("methods do not share identical inputs")


def _validate_quality(
    quality: dict[str, Any], tables: dict[str, list[dict[str, Any]]]
) -> None:
    expected = {
        "schema_version": PPL_SCHEMA_VERSION,
        "protocol_stage": "mechanism_ablation_full",
        "expected_cases": 2000,
        "completed_cases": 2000,
        "failure_records": 0,
        "completion_gate": "PASS",
        "quality_gate": "NOT_APPLICABLE",
        "fp16_calibration_cases": 0,
        "fp16_calibration_max_mean_abs_token_nll_delta": None,
        "fp16_calibration_max_abs_token_nll_delta": None,
    }
    for field, value in expected.items():
        if quality.get(field) != value:
            raise MechanismPPLAnalysisError(f"quality {field} differs: {quality.get(field)!r}")
    method_quality = {row["method_id"]: row for row in quality.get("method_quality", [])}
    length_quality = {
        (row["method_id"], row["prompt_length"]): row
        for row in quality.get("length_quality", [])
    }
    for row in tables["method_summary"]:
        _compare_quality_row(method_quality.get(row["method_id"]), row, row["method_id"])
    for row in tables["length_summary"]:
        key = (row["method_id"], row["prompt_length"])
        _compare_quality_row(length_quality.get(key), row, str(key))


def _compare_quality_row(actual: Any, expected: dict[str, Any], name: str) -> None:
    if not isinstance(actual, dict):
        raise MechanismPPLAnalysisError(f"quality lacks {name}")
    for field in ("decode_target_count", "decode_nll_sum", "decode_mean_nll", "decode_perplexity"):
        if not _equivalent(actual[field], expected[field]):
            raise MechanismPPLAnalysisError(f"quality {name} field {field} differs")


def _nll_row(
    metadata: dict[str, Any], records: Sequence[dict[str, Any]],
    *, prompt_length: int | None = None,
) -> dict[str, Any]:
    if not records:
        raise MechanismPPLAnalysisError(f"empty aggregate for {metadata['method_id']}")
    target_count = sum(record["scoring"]["decode_target_count"] for record in records)
    nll_sum = math.fsum(record["scoring"]["decode_nll_sum"] for record in records)
    mean = nll_sum / target_count
    case_means = [record["scoring"]["decode_mean_nll"] for record in records]
    row = {
        **metadata,
        "case_count": len(records),
        "anchor_count": len({record["input"]["anchor_index"] for record in records}),
        "decode_target_count": target_count,
        "decode_nll_sum": nll_sum,
        "decode_mean_nll": mean,
        "decode_perplexity": math.exp(mean),
        "case_mean_nll_pstdev": statistics.pstdev(case_means),
        "case_mean_nll_min": min(case_means),
        "case_mean_nll_max": max(case_means),
    }
    if prompt_length is not None:
        row["prompt_length"] = prompt_length
    return row


def _comparisons() -> list[dict[str, Any]]:
    rows = []
    order = 0
    for residual in RESIDUALS:
        for role in BASELINE_ROLES:
            rows.append({
                "comparison_id": f"r{residual}_full_vs_{role}",
                "comparison_order": order,
                "residual_length": residual,
                "baseline_role": role,
                "candidate_method_id": f"cage-r{residual}-full",
                "baseline_method_id": f"cage-r{residual}-{role}",
            })
            order += 1
    return rows


def _anchor_delta_rows(
    lookup: dict[tuple[str, int, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for comparison in _comparisons():
        for prompt_length in PPL_PROMPT_LENGTHS:
            for anchor_index in PPL_PAIRED_ANCHOR_INDICES:
                candidate = lookup[(comparison["candidate_method_id"], prompt_length, anchor_index)]
                baseline = lookup[(comparison["baseline_method_id"], prompt_length, anchor_index)]
                candidate_mean = candidate["scoring"]["decode_mean_nll"]
                baseline_mean = baseline["scoring"]["decode_mean_nll"]
                rows.append({
                    **comparison,
                    "prompt_length": prompt_length,
                    "anchor_index": anchor_index,
                    "continuation_start": candidate["input"]["continuation_start"],
                    "continuation_ids_sha256": candidate["input"]["continuation_ids_sha256"],
                    "candidate_case_id": candidate["case_id"],
                    "baseline_case_id": baseline["case_id"],
                    "candidate_mean_nll": candidate_mean,
                    "baseline_mean_nll": baseline_mean,
                    "delta_mean_nll_full_minus_baseline": candidate_mean - baseline_mean,
                })
    return rows


def _paired_rows(
    delta_rows: Sequence[dict[str, Any]],
    method_rows: Sequence[dict[str, Any]],
    length_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    method_lookup = {row["method_id"]: row for row in method_rows}
    length_lookup = {(row["method_id"], row["prompt_length"]): row for row in length_rows}
    rows = []
    for comparison in _comparisons():
        selected = [row for row in delta_rows if row["comparison_id"] == comparison["comparison_id"]]
        for prompt_length in (None, *PPL_PROMPT_LENGTHS):
            if prompt_length is None:
                by_anchor: dict[int, list[float]] = defaultdict(list)
                for row in selected:
                    by_anchor[row["anchor_index"]].append(row["delta_mean_nll_full_minus_baseline"])
                deltas = [statistics.fmean(by_anchor[index]) for index in PPL_PAIRED_ANCHOR_INDICES]
                candidate = method_lookup[comparison["candidate_method_id"]]
                baseline = method_lookup[comparison["baseline_method_id"]]
                scope = "overall_anchor_clustered"
                case_pairs = 200
                scope_order = 0
            else:
                deltas = [
                    row["delta_mean_nll_full_minus_baseline"]
                    for row in selected if row["prompt_length"] == prompt_length
                ]
                candidate = length_lookup[(comparison["candidate_method_id"], prompt_length)]
                baseline = length_lookup[(comparison["baseline_method_id"], prompt_length)]
                scope = "prompt_length"
                case_pairs = 50
                scope_order = PPL_PROMPT_LENGTHS.index(prompt_length) + 1
            seed = BOOTSTRAP_SEED + comparison["comparison_order"] * 10 + scope_order
            low, high = _bootstrap_mean_interval(deltas, seed=seed)
            mean_delta = statistics.fmean(deltas)
            rows.append({
                **comparison,
                "scope": scope,
                "prompt_length": prompt_length,
                "anchor_count": len(deltas),
                "case_pair_count": case_pairs,
                "candidate_decode_perplexity": candidate["decode_perplexity"],
                "baseline_decode_perplexity": baseline["decode_perplexity"],
                "candidate_ppl_ratio_to_baseline": math.exp(mean_delta),
                "mean_paired_delta_nll": mean_delta,
                "median_paired_delta_nll": statistics.median(deltas),
                "paired_delta_nll_pstdev": statistics.pstdev(deltas),
                "paired_delta_nll_min": min(deltas),
                "paired_delta_nll_max": max(deltas),
                "candidate_better_anchors": sum(delta < 0 for delta in deltas),
                "equal_anchors": sum(delta == 0 for delta in deltas),
                "candidate_worse_anchors": sum(delta > 0 for delta in deltas),
                "bootstrap_mean_delta_nll_ci95_low": low,
                "bootstrap_mean_delta_nll_ci95_high": high,
                "bootstrap_ppl_ratio_ci95_low": math.exp(low),
                "bootstrap_ppl_ratio_ci95_high": math.exp(high),
                "bootstrap_seed": seed,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            })
    return rows


def _factorial_anchor_rows(
    lookup: dict[tuple[str, int, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for residual in RESIDUALS:
        ids = {
            "full": f"cage-r{residual}-full",
            "key": f"cage-r{residual}-k-adaptive",
            "value": f"cage-r{residual}-v-adaptive",
            "uniform": f"cage-r{residual}-uniform",
        }
        for prompt_length in PPL_PROMPT_LENGTHS:
            for anchor_index in PPL_PAIRED_ANCHOR_INDICES:
                means = {
                    role: lookup[(method_id, prompt_length, anchor_index)]["scoring"]["decode_mean_nll"]
                    for role, method_id in ids.items()
                }
                rows.append({
                    "residual_length": residual,
                    "prompt_length": prompt_length,
                    "anchor_index": anchor_index,
                    "full_mean_nll": means["full"],
                    "key_adaptive_mean_nll": means["key"],
                    "value_adaptive_mean_nll": means["value"],
                    "uniform_mean_nll": means["uniform"],
                    "full_minus_uniform": means["full"] - means["uniform"],
                    "key_adaptive_minus_uniform": means["key"] - means["uniform"],
                    "value_adaptive_minus_uniform": means["value"] - means["uniform"],
                    "interaction_full_minus_key_minus_value_plus_uniform": (
                        means["full"] - means["key"] - means["value"] + means["uniform"]
                    ),
                })
    return rows


def _factorial_rows(anchor_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for residual in RESIDUALS:
        residual_rows = [row for row in anchor_rows if row["residual_length"] == residual]
        for prompt_length in (None, *PPL_PROMPT_LENGTHS):
            if prompt_length is None:
                by_anchor = {
                    anchor: [
                        row for row in residual_rows if row["anchor_index"] == anchor
                    ]
                    for anchor in PPL_PAIRED_ANCHOR_INDICES
                }
                clustered = {
                    effect: [statistics.fmean(row[effect] for row in by_anchor[anchor])
                             for anchor in PPL_PAIRED_ANCHOR_INDICES]
                    for effect in FACTORIAL_EFFECTS
                }
                scope = "overall_anchor_clustered"
                scope_order = 0
                case_groups = 200
            else:
                selected = [row for row in residual_rows if row["prompt_length"] == prompt_length]
                clustered = {effect: [row[effect] for row in selected] for effect in FACTORIAL_EFFECTS}
                scope = "prompt_length"
                scope_order = PPL_PROMPT_LENGTHS.index(prompt_length) + 1
                case_groups = 50
            output: dict[str, Any] = {
                "residual_length": residual,
                "scope": scope,
                "prompt_length": prompt_length,
                "anchor_count": 50,
                "case_group_count": case_groups,
            }
            for effect_order, effect in enumerate(FACTORIAL_EFFECTS):
                values = clustered[effect]
                seed = BOOTSTRAP_SEED + 1000 + residual * 10 + scope_order + effect_order * 100
                low, high = _bootstrap_mean_interval(values, seed=seed)
                output[f"mean_{effect}"] = statistics.fmean(values)
                output[f"pstdev_{effect}"] = statistics.pstdev(values)
                output[f"bootstrap_{effect}_ci95_low"] = low
                output[f"bootstrap_{effect}_ci95_high"] = high
            rows.append(output)
    return rows


def _bootstrap_mean_interval(values: Sequence[float], *, seed: int) -> tuple[float, float]:
    if len(values) != 50:
        raise MechanismPPLAnalysisError(
            f"bootstrap requires 50 anchor values, got {len(values)}"
        )
    generator = random.Random(seed)
    means = [
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    means.sort()
    return means[249], means[9749]


def _render_markdown(tables: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# CAGE-KV paired PPL mechanism ablation",
        "",
        "Primary contrasts are paired cache-dependent decode NLL differences over 50 "
        "deterministic WikiText-2 anchors. Negative `full - baseline` favors full CAGE.",
        "",
        "> Bootstrap intervals are descriptive stability intervals over the frozen corpus "
        "grid, not population-significance claims. FP16 and KIVI are not rerun here.",
        "",
        "## Overall paired contrasts",
        "",
        "| Residual | Baseline | Delta NLL | PPL ratio | Bootstrap 95% | Better/equal/worse anchors |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in tables["paired_comparisons"]:
        if row["scope"] != "overall_anchor_clustered":
            continue
        lines.append(
            f"| {row['residual_length']} | {row['baseline_role']} | "
            f"{row['mean_paired_delta_nll']:+.6f} | {row['candidate_ppl_ratio_to_baseline']:.6f} | "
            f"[{row['bootstrap_mean_delta_nll_ci95_low']:+.6f}, "
            f"{row['bootstrap_mean_delta_nll_ci95_high']:+.6f}] | "
            f"{row['candidate_better_anchors']}/{row['equal_anchors']}/"
            f"{row['candidate_worse_anchors']} |"
        )
    lines.extend([
        "",
        "## Overall factorial decomposition",
        "",
        "| Residual | Full-uniform | Key-uniform | Value-uniform | Interaction |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in tables["factorial_summary"]:
        if row["scope"] != "overall_anchor_clustered":
            continue
        lines.append(
            f"| {row['residual_length']} | {row['mean_full_minus_uniform']:+.6f} | "
            f"{row['mean_key_adaptive_minus_uniform']:+.6f} | "
            f"{row['mean_value_adaptive_minus_uniform']:+.6f} | "
            f"{row['mean_interaction_full_minus_key_minus_value_plus_uniform']:+.6f} |"
        )
    return "\n".join(lines) + "\n"


def _plot_mechanism(
    destination: Path, tables: dict[str, list[dict[str, Any]]]
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise MechanismPPLAnalysisError(
            "matplotlib is required for plots; install it or pass --no-plots"
        ) from error

    colors = {
        "fixed-random": "#7f3c0a", "uniform": "#e15759",
        "k-adaptive": "#59a14f", "v-adaptive": "#76b7b2",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), sharey=True)
    for axis, residual in zip(axes, RESIDUALS):
        for role in BASELINE_ROLES:
            rows = [
                row for row in tables["paired_comparisons"]
                if row["residual_length"] == residual
                and row["baseline_role"] == role
                and row["scope"] == "prompt_length"
            ]
            means = [row["mean_paired_delta_nll"] for row in rows]
            lower = [mean - row["bootstrap_mean_delta_nll_ci95_low"] for mean, row in zip(means, rows)]
            upper = [row["bootstrap_mean_delta_nll_ci95_high"] - mean for mean, row in zip(means, rows)]
            axis.errorbar(
                PPL_PROMPT_LENGTHS, means, yerr=[lower, upper], marker="o",
                color=colors[role], capsize=3, linewidth=1.1, label=role,
            )
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_xscale("log", base=2)
        axis.set_xticks(PPL_PROMPT_LENGTHS, PPL_PROMPT_LENGTHS)
        axis.set_xlabel("Prompt length")
        axis.set_ylabel("Paired delta NLL (full - baseline)")
        axis.set_title(f"Residual length {residual}")
        axis.grid(True, alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("CAGE-KV paired PPL mechanism contrasts\nDescriptive paired-bootstrap 95% intervals")
    figure.tight_layout(rect=(0, 0.1, 1, 0.9))
    outputs = _save_figure(figure, destination / "ppl_mechanism_contrasts", plt)

    effect_styles = {
        "full_minus_uniform": ("full - uniform", "#176b3a", "o"),
        "key_adaptive_minus_uniform": ("Key-only - uniform", "#59a14f", "^"),
        "value_adaptive_minus_uniform": ("Value-only - uniform", "#76b7b2", "v"),
        "interaction_full_minus_key_minus_value_plus_uniform": ("interaction", "#b07aa1", "s"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), sharey=True)
    for axis, residual in zip(axes, RESIDUALS):
        rows = [
            row for row in tables["factorial_summary"]
            if row["residual_length"] == residual and row["scope"] == "prompt_length"
        ]
        for effect, (label, color, marker) in effect_styles.items():
            axis.plot(
                PPL_PROMPT_LENGTHS, [row[f"mean_{effect}"] for row in rows],
                marker=marker, color=color, linewidth=1.1, label=label,
            )
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_xscale("log", base=2)
        axis.set_xticks(PPL_PROMPT_LENGTHS, PPL_PROMPT_LENGTHS)
        axis.set_xlabel("Prompt length")
        axis.set_ylabel("Paired factorial effect on decode NLL")
        axis.set_title(f"Residual length {residual}")
        axis.grid(True, alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("CAGE-KV paired PPL factorial decomposition")
    figure.tight_layout(rect=(0, 0.1, 1, 0.92))
    outputs.extend(_save_figure(figure, destination / "ppl_factorial_decomposition", plt))
    return outputs


def _save_figure(figure: Any, base: Path, plt: Any) -> list[Path]:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return [png, pdf]


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MechanismPPLAnalysisError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise MechanismPPLAnalysisError(f"{name} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise MechanismPPLAnalysisError(f"blank line {line_number} in {path}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise MechanismPPLAnalysisError(f"row {line_number} in {path} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise MechanismPPLAnalysisError(f"cannot read JSONL {path}: {error}") from error
    return rows


def _read_csv_ids(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise MechanismPPLAnalysisError(f"cannot read CSV {path}: {error}") from error
    if not rows or "case_id" not in rows[0]:
        raise MechanismPPLAnalysisError(f"CSV {path} is empty or lacks case_id")
    return [row["case_id"] for row in rows]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ))
            handle.write("\n")
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    fields = sorted({field for row in rows for field in row})
    preferred = [
        "comparison_id", "comparison_order", "residual_length", "baseline_role",
        "candidate_method_id", "baseline_method_id", "scope", "prompt_length",
        "anchor_index", "method_id", "method_order", "case_count", "anchor_count",
        "decode_target_count", "decode_mean_nll", "decode_perplexity",
        "mean_paired_delta_nll", "candidate_ppl_ratio_to_baseline",
    ]
    ordered = [field for field in preferred if field in fields]
    ordered.extend(field for field in fields if field not in ordered)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


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
    "ANALYSIS_SCHEMA_VERSION", "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED",
    "MechanismPPLAnalysisError", "aggregate_mechanism_results",
    "load_completed_mechanism_matrix", "write_mechanism_analysis_outputs",
]
