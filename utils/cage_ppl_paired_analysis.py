"""Pre-registered read-only analysis for the 50-anchor paired PPL study."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils.cage_ppl import (
    PPL_DECODE_TARGETS,
    PPL_PAIRED_ANCHOR_INDICES,
    PPL_PAIRED_RAW_METHODS,
    PPL_PROMPT_LENGTHS,
    PPL_SCHEMA_VERSION,
    validate_completed_ppl_case,
)


PAIRED_ANALYSIS_SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_RESAMPLES = 10_000
PAIRED_METHOD_IDS = tuple(method["id"] for method in PPL_PAIRED_RAW_METHODS)
DECLARED_COMPARISONS = (
    {
        "comparison_id": "cage-r128_vs_kivi-g32-r128",
        "candidate_method_id": "cage-r128",
        "baseline_method_id": "kivi-g32-r128",
        "rationale": "same residual length; high-fidelity KIVI reference",
    },
    {
        "comparison_id": "cage-r64_vs_kivi-g64-r64",
        "candidate_method_id": "cage-r64",
        "baseline_method_id": "kivi-g64-r64",
        "rationale": "same residual length; memory-efficient KIVI operating point",
    },
)


class PairedPPLAnalysisError(ValueError):
    """Raised when frozen paired-study artifacts fail strict validation."""


def load_completed_paired_matrix(
    results_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate and load the frozen 1,000-case paired PPL matrix."""

    root = Path(results_dir)
    resolved = _read_json(root / "manifest.resolved.json", "resolved manifest")
    quality = _read_json(root / "summary" / "quality.json", "quality summary")
    _validate_resolved(resolved)
    expanded = resolved["expanded_cases"]
    expected_ids = [item.get("case_id") for item in expanded]
    if len(expected_ids) != 1000 or len(set(expected_ids)) != 1000:
        raise PairedPPLAnalysisError(
            f"paired manifest requires 1000 unique cases, got {len(expected_ids)} rows "
            f"and {len(set(expected_ids))} unique IDs"
        )
    if any(not isinstance(case_id, str) or not case_id for case_id in expected_ids):
        raise PairedPPLAnalysisError("expanded_cases contains an invalid case_id")
    expected_set = set(expected_ids)

    case_ids = {path.stem for path in (root / "cases").glob("*.json")}
    failure_ids = {path.stem for path in (root / "failures").glob("*.json")}
    if case_ids != expected_set:
        raise PairedPPLAnalysisError(_id_set_error("case artifacts", expected_set, case_ids))
    if failure_ids:
        raise PairedPPLAnalysisError(f"failure artifacts remain: {sorted(failure_ids)}")

    records = []
    for case_id in expected_ids:
        try:
            records.append(validate_completed_ppl_case(root, case_id))
        except ValueError as error:
            raise PairedPPLAnalysisError(f"invalid completed case {case_id}: {error}") from error

    summary = _read_jsonl(root / "summary" / "cases.jsonl")
    summary_ids = [row.get("case_id") for row in summary]
    if len(summary_ids) != 1000 or set(summary_ids) != expected_set:
        raise PairedPPLAnalysisError(
            _id_set_error("summary/cases.jsonl", expected_set, set(summary_ids))
        )
    if len(summary_ids) != len(set(summary_ids)):
        raise PairedPPLAnalysisError("summary/cases.jsonl contains duplicate IDs")
    by_id = {record["case_id"]: record for record in records}
    if any(row != by_id[row["case_id"]] for row in summary):
        raise PairedPPLAnalysisError("summary/cases.jsonl differs from case artifacts")

    csv_ids = _read_csv_ids(root / "summary" / "cases.csv")
    if len(csv_ids) != 1000 or set(csv_ids) != expected_set:
        raise PairedPPLAnalysisError(
            _id_set_error("summary/cases.csv", expected_set, set(csv_ids))
        )
    if len(csv_ids) != len(set(csv_ids)):
        raise PairedPPLAnalysisError("summary/cases.csv contains duplicate IDs")

    expanded_by_id = {item["case_id"]: item for item in expanded}
    for record in records:
        declared = expanded_by_id[record["case_id"]]
        if declared["method"] != record["method"] or declared["input"] != record["input"]:
            raise PairedPPLAnalysisError(
                f"expanded case {record['case_id']} differs from completed artifact"
            )
    resolved_state = _canonical_json(resolved["source_state"])
    states = {
        _canonical_json(record["provenance"]["source_state"]) for record in records
    }
    if states != {resolved_state}:
        raise PairedPPLAnalysisError("completed cases differ from resolved source state")

    _validate_coverage(records)
    tables = aggregate_paired_results(records, resolved)
    _validate_quality(quality, tables)
    return resolved, records, quality


def aggregate_paired_results(
    records: Sequence[dict[str, Any]], resolved: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate methods and declared paired comparisons over 50 anchor clusters."""

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
    _attach_fp16(method_rows, ())
    _attach_fp16(length_rows, ("prompt_length",))

    record_lookup = {
        (
            record["method"]["id"],
            record["input"]["prompt_length"],
            record["input"]["anchor_index"],
        ): record
        for record in records
    }
    delta_rows = _anchor_delta_rows(record_lookup)
    paired_rows = _paired_rows(delta_rows, method_rows, length_rows)
    return {
        "method_summary": method_rows,
        "length_summary": length_rows,
        "paired_comparisons": paired_rows,
        "anchor_deltas": delta_rows,
    }


def write_paired_analysis_outputs(
    analysis_dir: str | Path,
    tables: dict[str, list[dict[str, Any]]],
    *,
    resolved_manifest: dict[str, Any],
    quality_summary: dict[str, Any],
    make_plots: bool = True,
) -> list[Path]:
    """Write pre-registered paired tables, metadata, Markdown, and figures."""

    destination = Path(analysis_dir)
    if destination.exists() and any(destination.iterdir()):
        raise PairedPPLAnalysisError(f"analysis directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name in ("method_summary", "length_summary", "paired_comparisons", "anchor_deltas"):
        rows = tables[name]
        outputs.append(_write_jsonl(destination / f"{name}.jsonl", rows))
        outputs.append(_write_csv(destination / f"{name}.csv", rows))

    protocol = {
        "schema_version": PAIRED_ANALYSIS_SCHEMA_VERSION,
        "input_ppl_schema_version": PPL_SCHEMA_VERSION,
        "protocol_stage": "paired_full",
        "case_count": 1000,
        "input_group_count": 200,
        "anchor_count": 50,
        "primary_target_count": 63_000,
        "method_ids": list(PAIRED_METHOD_IDS),
        "prompt_lengths": list(PPL_PROMPT_LENGTHS),
        "declared_comparisons": list(DECLARED_COMPARISONS),
        "paired_direction": "candidate CAGE mean NLL minus baseline KIVI mean NLL",
        "anchor_cluster": (
            "one anchor is the resampling unit; overall diagnostic first averages "
            "the four prompt lengths within each anchor"
        ),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "interval": "2.5th and 97.5th percentiles of paired mean delta NLL",
            "role": "descriptive stability interval over a deterministic corpus grid",
        },
        "inference_boundary": (
            "anchors are systematic deterministic locations, not random population draws; "
            "bootstrap intervals do not establish population-level significance"
        ),
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
    }
    protocol_path = destination / "analysis_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs.append(protocol_path)
    summary_path = destination / "paired_ppl_summary.md"
    summary_path.write_text(_render_markdown(tables), encoding="utf-8")
    outputs.append(summary_path)
    if make_plots:
        outputs.extend(_plot_paired(destination, tables))
    return outputs


def _validate_resolved(resolved: dict[str, Any]) -> None:
    if resolved.get("protocol_stage") != "paired_full":
        raise PairedPPLAnalysisError("paired analysis requires protocol_stage='paired_full'")
    if resolved.get("prompt_lengths") != list(PPL_PROMPT_LENGTHS):
        raise PairedPPLAnalysisError("paired prompt lengths differ from protocol")
    if resolved.get("anchor_indices") != list(PPL_PAIRED_ANCHOR_INDICES):
        raise PairedPPLAnalysisError("paired anchors differ from protocol")
    methods = resolved.get("methods")
    if not isinstance(methods, list) or [method.get("id") for method in methods] != list(PAIRED_METHOD_IDS):
        raise PairedPPLAnalysisError("paired methods differ from protocol")
    state = resolved.get("source_state")
    if not isinstance(state, dict) or state.get("dirty") is not False:
        raise PairedPPLAnalysisError("paired source state must be clean")
    if not isinstance(resolved.get("expanded_cases"), list):
        raise PairedPPLAnalysisError("paired manifest lacks expanded_cases")


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
        for method_id in PAIRED_METHOD_IDS
        for length in PPL_PROMPT_LENGTHS
        for anchor in PPL_PAIRED_ANCHOR_INDICES
    }
    if set(groups) != expected or any(len(group) != 1 for group in groups.values()):
        raise PairedPPLAnalysisError("paired method/length/anchor coverage is invalid")
    if any(len(values) != 1 for values in shared.values()):
        raise PairedPPLAnalysisError("paired methods do not share identical inputs")


def _validate_quality(
    quality: dict[str, Any], tables: dict[str, list[dict[str, Any]]]
) -> None:
    expected = {
        "schema_version": PPL_SCHEMA_VERSION,
        "protocol_stage": "paired_full",
        "expected_cases": 1000,
        "completed_cases": 1000,
        "failure_records": 0,
        "completion_gate": "PASS",
        "quality_gate": "NOT_APPLICABLE",
        "fp16_calibration_cases": 200,
    }
    for field, value in expected.items():
        if quality.get(field) != value:
            raise PairedPPLAnalysisError(f"quality {field} differs: {quality.get(field)!r}")
    method_quality = {row["method_id"]: row for row in quality.get("method_quality", [])}
    for row in tables["method_summary"]:
        actual = method_quality.get(row["method_id"])
        if actual is None:
            raise PairedPPLAnalysisError(f"quality lacks method {row['method_id']}")
        for field in ("decode_target_count", "decode_nll_sum", "decode_mean_nll", "decode_perplexity"):
            if not _equivalent(actual[field], row[field]):
                raise PairedPPLAnalysisError(
                    f"quality method {row['method_id']} field {field} differs"
                )
    mean_delta = _finite(
        "FP16 max mean calibration delta",
        quality.get("fp16_calibration_max_mean_abs_token_nll_delta"), minimum=0.0,
    )
    max_delta = _finite(
        "FP16 max token calibration delta",
        quality.get("fp16_calibration_max_abs_token_nll_delta"), minimum=0.0,
    )
    if mean_delta > 0.005 or max_delta > 0.03:
        raise PairedPPLAnalysisError("FP16 calibration exceeds frozen limits")


def _nll_row(
    metadata: dict[str, Any], records: Sequence[dict[str, Any]],
    *, prompt_length: int | None = None,
) -> dict[str, Any]:
    if not records:
        raise PairedPPLAnalysisError(f"empty aggregate for {metadata['method_id']}")
    count = sum(record["scoring"]["decode_target_count"] for record in records)
    nll_sum = math.fsum(record["scoring"]["decode_nll_sum"] for record in records)
    mean = nll_sum / count
    case_means = [record["scoring"]["decode_mean_nll"] for record in records]
    row = {
        **metadata,
        "case_count": len(records),
        "anchor_count": len({record["input"]["anchor_index"] for record in records}),
        "decode_target_count": count,
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


def _attach_fp16(rows: Sequence[dict[str, Any]], fields: tuple[str, ...]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    for key, group in groups.items():
        baselines = [row for row in group if row["method_id"] == "fp16"]
        if len(baselines) != 1:
            raise PairedPPLAnalysisError(f"group {key} requires one FP16 row")
        baseline = baselines[0]
        for row in group:
            delta = row["decode_mean_nll"] - baseline["decode_mean_nll"]
            row["delta_mean_nll_vs_fp16"] = delta
            row["ppl_ratio_vs_fp16"] = math.exp(delta)


def _anchor_delta_rows(
    lookup: dict[tuple[str, int, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for comparison_order, comparison in enumerate(DECLARED_COMPARISONS):
        candidate_id = comparison["candidate_method_id"]
        baseline_id = comparison["baseline_method_id"]
        for prompt_length in PPL_PROMPT_LENGTHS:
            for anchor_index in PPL_PAIRED_ANCHOR_INDICES:
                candidate = lookup[(candidate_id, prompt_length, anchor_index)]
                baseline = lookup[(baseline_id, prompt_length, anchor_index)]
                candidate_mean = candidate["scoring"]["decode_mean_nll"]
                baseline_mean = baseline["scoring"]["decode_mean_nll"]
                rows.append({
                    "comparison_id": comparison["comparison_id"],
                    "comparison_order": comparison_order,
                    "candidate_method_id": candidate_id,
                    "baseline_method_id": baseline_id,
                    "prompt_length": prompt_length,
                    "anchor_index": anchor_index,
                    "continuation_start": candidate["input"]["continuation_start"],
                    "continuation_ids_sha256": candidate["input"]["continuation_ids_sha256"],
                    "candidate_case_id": candidate["case_id"],
                    "baseline_case_id": baseline["case_id"],
                    "candidate_mean_nll": candidate_mean,
                    "baseline_mean_nll": baseline_mean,
                    "delta_mean_nll_candidate_minus_baseline": candidate_mean - baseline_mean,
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
    for comparison_order, comparison in enumerate(DECLARED_COMPARISONS):
        comparison_rows = [
            row for row in delta_rows if row["comparison_id"] == comparison["comparison_id"]
        ]
        for prompt_length in (None, *PPL_PROMPT_LENGTHS):
            if prompt_length is None:
                by_anchor: dict[int, list[float]] = defaultdict(list)
                for row in comparison_rows:
                    by_anchor[row["anchor_index"]].append(
                        row["delta_mean_nll_candidate_minus_baseline"]
                    )
                deltas = [statistics.fmean(by_anchor[index]) for index in PPL_PAIRED_ANCHOR_INDICES]
                candidate = method_lookup[comparison["candidate_method_id"]]
                baseline = method_lookup[comparison["baseline_method_id"]]
                case_pairs = 200
                scope = "overall_diagnostic_anchor_clustered"
            else:
                deltas = [
                    row["delta_mean_nll_candidate_minus_baseline"]
                    for row in comparison_rows if row["prompt_length"] == prompt_length
                ]
                candidate = length_lookup[(comparison["candidate_method_id"], prompt_length)]
                baseline = length_lookup[(comparison["baseline_method_id"], prompt_length)]
                case_pairs = 50
                scope = "prompt_length"
            mean_delta = statistics.fmean(deltas)
            low, high = _bootstrap_mean_interval(
                deltas,
                seed=BOOTSTRAP_SEED + comparison_order * 10 + (
                    0 if prompt_length is None else PPL_PROMPT_LENGTHS.index(prompt_length) + 1
                ),
            )
            rows.append({
                **comparison,
                "comparison_order": comparison_order,
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
                "bootstrap_seed": BOOTSTRAP_SEED + comparison_order * 10 + (
                    0 if prompt_length is None else PPL_PROMPT_LENGTHS.index(prompt_length) + 1
                ),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            })
    return rows


def _bootstrap_mean_interval(values: Sequence[float], *, seed: int) -> tuple[float, float]:
    if len(values) != 50:
        raise PairedPPLAnalysisError(f"bootstrap requires 50 anchor values, got {len(values)}")
    generator = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        means.append(statistics.fmean(generator.choices(values, k=len(values))))
    means.sort()
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _render_markdown(tables: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# CAGE-KV 50-anchor paired PPL study",
        "",
        "Paired direction is CAGE candidate mean NLL minus KIVI baseline mean NLL; "
        "negative values favor CAGE.",
        "",
        "| Comparison | Scope | Length | CAGE/KIVI PPL ratio | Mean ΔNLL | Descriptive bootstrap 95% interval | Better anchors |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in tables["paired_comparisons"]:
        length = row["prompt_length"] if row["prompt_length"] is not None else "all"
        lines.append(
            f"| {row['comparison_id']} | {row['scope']} | {length} | "
            f"{row['candidate_ppl_ratio_to_baseline']:.6f}× | "
            f"{row['mean_paired_delta_nll']:+.6f} | "
            f"[{row['bootstrap_mean_delta_nll_ci95_low']:+.6f}, "
            f"{row['bootstrap_mean_delta_nll_ci95_high']:+.6f}] | "
            f"{row['candidate_better_anchors']}/{row['anchor_count']} |"
        )
    lines.extend([
        "",
        "The 50 anchors form a deterministic, approximately even grid over the frozen corpus. "
        "Bootstrap intervals describe stability over that grid and are not population-level "
        "significance claims.",
        "",
        "Overall rows first average the four prompt-length deltas within each anchor, so the "
        "resampling unit remains one corpus anchor.",
    ])
    return "\n".join(lines) + "\n"


def _plot_paired(
    destination: Path, tables: dict[str, list[dict[str, Any]]]
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise PairedPPLAnalysisError(
            "matplotlib is required for plots; install it or pass --no-plots"
        ) from error

    rows = [
        row for row in tables["paired_comparisons"] if row["scope"] == "prompt_length"
    ]
    colors = ["#176b3a", "#59a14f"]
    figure, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), sharey=True)
    for axis, comparison, color in zip(axes, DECLARED_COMPARISONS, colors):
        selected = [row for row in rows if row["comparison_id"] == comparison["comparison_id"]]
        x = list(range(len(PPL_PROMPT_LENGTHS)))
        means = [row["mean_paired_delta_nll"] for row in selected]
        lower = [
            row["mean_paired_delta_nll"] - row["bootstrap_mean_delta_nll_ci95_low"]
            for row in selected
        ]
        upper = [
            row["bootstrap_mean_delta_nll_ci95_high"] - row["mean_paired_delta_nll"]
            for row in selected
        ]
        axis.errorbar(
            x, means, yerr=[lower, upper], fmt="o-", color=color,
            capsize=4, linewidth=1.2, markersize=6,
        )
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_xticks(x, PPL_PROMPT_LENGTHS)
        axis.set_xlabel("Prompt length")
        axis.set_ylabel("Paired ΔNLL (CAGE − KIVI)")
        axis.set_title(comparison["comparison_id"].replace("_", "\n"))
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        "CAGE-KV 50-anchor direct paired comparison\n"
        "Bars are pre-registered descriptive paired-bootstrap 95% intervals",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    paths = _save_figure(figure, destination / "paired_delta_nll", plt)

    delta_rows = tables["anchor_deltas"]
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.6), squeeze=False)
    for axis, prompt_length in zip(axes.flat, PPL_PROMPT_LENGTHS):
        for comparison, color in zip(DECLARED_COMPARISONS, colors):
            selected = [
                row for row in delta_rows
                if row["comparison_id"] == comparison["comparison_id"]
                and row["prompt_length"] == prompt_length
            ]
            axis.plot(
                [row["anchor_index"] for row in selected],
                [row["delta_mean_nll_candidate_minus_baseline"] for row in selected],
                marker="o", markersize=2.5, linewidth=0.8, alpha=0.8,
                color=color, label=comparison["comparison_id"],
            )
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_title(f"Prompt length {prompt_length}")
        axis.set_xlabel("Deterministic anchor index")
        axis.set_ylabel("Paired ΔNLL (CAGE − KIVI)")
        axis.grid(True, alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    figure.suptitle("Per-anchor paired NLL differences", fontsize=14)
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    paths.extend(_save_figure(figure, destination / "paired_anchor_deltas", plt))
    return paths


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
        raise PairedPPLAnalysisError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PairedPPLAnalysisError(f"{name} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise PairedPPLAnalysisError(f"blank line {line_number} in {path}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PairedPPLAnalysisError(f"row {line_number} in {path} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise PairedPPLAnalysisError(f"cannot read JSONL {path}: {error}") from error
    return rows


def _read_csv_ids(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise PairedPPLAnalysisError(f"cannot read CSV {path}: {error}") from error
    if not rows or "case_id" not in rows[0]:
        raise PairedPPLAnalysisError(f"CSV {path} is empty or lacks case_id")
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
    fields = _ordered_fields(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _ordered_fields(rows: Sequence[dict[str, Any]]) -> list[str]:
    fields = sorted({field for row in rows for field in row})
    preferred = [
        "comparison_id", "comparison_order", "candidate_method_id",
        "baseline_method_id", "scope", "prompt_length", "anchor_index",
        "method_id", "method_order", "method", "case_count", "anchor_count",
        "decode_target_count", "decode_mean_nll", "decode_perplexity",
        "mean_paired_delta_nll", "bootstrap_mean_delta_nll_ci95_low",
        "bootstrap_mean_delta_nll_ci95_high", "candidate_ppl_ratio_to_baseline",
    ]
    ordered = [field for field in preferred if field in fields]
    return ordered + [field for field in fields if field not in ordered]


def _finite(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairedPPLAnalysisError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise PairedPPLAnalysisError(f"{name} must be finite and >= {minimum}")
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
    "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED", "DECLARED_COMPARISONS",
    "PAIRED_ANALYSIS_SCHEMA_VERSION", "PairedPPLAnalysisError",
    "aggregate_paired_results", "load_completed_paired_matrix",
    "write_paired_analysis_outputs",
]
