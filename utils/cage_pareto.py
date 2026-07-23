"""Read-only aggregation and Pareto analysis for completed CAGE pilot results."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils.cage_experiment_config import validate_resolved_manifest
from utils.cage_experiment_schema import BYTE_FIELDS, METRIC_NAMES, validate_completed_point


PRIMARY_METRIC = "joint_post_o_proj_mse"
MIB = 1024**2
EXPECTED_LAYER_COUNT = 32


class ParetoAnalysisError(ValueError):
    """Raised when completed artifacts cannot support the declared analysis."""


def load_completed_matrix(results_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate and load the frozen matrix without recomputing source-dependent run IDs."""

    root = Path(results_dir)
    resolved_path = root / "manifest.resolved.json"
    try:
        resolved = validate_resolved_manifest(
            json.loads(resolved_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ParetoAnalysisError(f"invalid resolved manifest {resolved_path}: {error}") from error

    summary_rows = _read_jsonl(root / "summary" / "runs.jsonl")
    summary_ids = [row.get("run_id") for row in summary_rows]
    if any(not isinstance(run_id, str) or not run_id for run_id in summary_ids):
        raise ParetoAnalysisError("summary/runs.jsonl contains a missing or invalid run_id")
    if len(summary_ids) != len(set(summary_ids)):
        raise ParetoAnalysisError("summary/runs.jsonl contains duplicate run IDs")
    if any(row.get("status") != "completed" for row in summary_rows):
        raise ParetoAnalysisError("summary/runs.jsonl contains a non-completed row")

    expected_count = (
        len(resolved["methods"])
        * len(resolved["sample_ids"])
        * len(resolved["prompt_lengths"])
    )
    if len(summary_ids) != expected_count:
        raise ParetoAnalysisError(
            f"summary contains {len(summary_ids)} runs; resolved matrix requires {expected_count}"
        )

    summary_id_set = set(summary_ids)
    run_ids = {path.stem for path in (root / "runs").glob("*.json")}
    layer_ids = {path.stem for path in (root / "layers").glob("*.jsonl")}
    failure_ids = {path.stem for path in (root / "failures").glob("*.json")}
    if run_ids != summary_id_set:
        raise ParetoAnalysisError(_id_set_error("run artifacts", summary_id_set, run_ids))
    if layer_ids != summary_id_set:
        raise ParetoAnalysisError(_id_set_error("layer artifacts", summary_id_set, layer_ids))
    if failure_ids:
        raise ParetoAnalysisError(f"failure artifacts remain: {sorted(failure_ids)}")

    runs = []
    for run_id in sorted(summary_ids):
        try:
            run = validate_completed_point(root, run_id)
        except ValueError as error:
            raise ParetoAnalysisError(f"invalid completed point {run_id}: {error}") from error
        if run["measurement"]["layer_count"] != EXPECTED_LAYER_COUNT:
            raise ParetoAnalysisError(
                f"run {run_id} has {run['measurement']['layer_count']} layers; "
                f"the Llama-2-7B pilot requires {EXPECTED_LAYER_COUNT}"
            )
        for field in ("reference", "dtype", "device"):
            if run["model"].get(field) != resolved["model"][field]:
                raise ParetoAnalysisError(
                    f"run {run_id} model.{field} differs from the resolved manifest"
                )
        runs.append(run)

    source_states = {
        _canonical_json(run["provenance"]["source_state"])
        for run in runs
    }
    if len(source_states) != 1:
        raise ParetoAnalysisError(
            f"completed matrix mixes {len(source_states)} source states"
        )
    source_state = json.loads(next(iter(source_states)))
    if source_state.get("dirty") is not False:
        raise ParetoAnalysisError(f"completed matrix source state is dirty: {source_state}")
    return resolved, runs


def aggregate_points(
    runs: Sequence[dict[str, Any]],
    resolved_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate three sample runs into one method-configuration/prompt point."""

    methods = _method_catalog(resolved_manifest)
    selected_samples = tuple(resolved_manifest["sample_ids"])
    selected_lengths = tuple(resolved_manifest["prompt_lengths"])
    selected_sample_set = set(selected_samples)
    selected_length_set = set(selected_lengths)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    seen_combinations: set[tuple[str, str, int]] = set()

    for run in runs:
        identity = (
            run["method"]["name"],
            _canonical_json(run["method"]["resolved_config"]),
        )
        if identity not in methods:
            raise ParetoAnalysisError(
                f"run {run['run_id']} has a method configuration absent from the resolved manifest"
            )
        method = methods[identity]
        sample_id = run["input"]["sample_id"]
        prompt_length = run["input"]["prompt_length"]
        if sample_id not in selected_sample_set:
            raise ParetoAnalysisError(f"run {run['run_id']} has unselected sample {sample_id!r}")
        if prompt_length not in selected_length_set:
            raise ParetoAnalysisError(
                f"run {run['run_id']} has unselected prompt length {prompt_length}"
            )
        combination = (method["config_id"], sample_id, prompt_length)
        if combination in seen_combinations:
            raise ParetoAnalysisError(f"duplicate scientific combination {combination}")
        seen_combinations.add(combination)
        groups[(method["config_id"], prompt_length)].append(run)

    points = []
    ordered_methods = sorted(methods.values(), key=lambda item: item["config_order"])
    for method in ordered_methods:
        for prompt_length in selected_lengths:
            group = groups.get((method["config_id"], prompt_length), [])
            actual_samples = {run["input"]["sample_id"] for run in group}
            if actual_samples != selected_sample_set or len(group) != len(selected_samples):
                raise ParetoAnalysisError(
                    f"{method['config_id']} prompt {prompt_length} sample coverage differs: "
                    f"expected {sorted(selected_sample_set)}, got {sorted(actual_samples)}"
                )
            points.append(_aggregate_group(group, method, prompt_length))

    expected_points = len(ordered_methods) * len(selected_lengths)
    if len(points) != expected_points:
        raise ParetoAnalysisError(f"aggregated {len(points)} points; expected {expected_points}")

    _attach_fp16_normalization(points)
    _attach_pareto_flags(points)
    _attach_sample_pareto_stability(points)
    return sorted(points, key=lambda row: (row["prompt_length"], row["config_order"]))


def build_trends(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-configuration prompt-length changes without cross-length Pareto claims."""

    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        by_config[point["config_id"]].append(point)

    trends = []
    for config_id, rows in sorted(
        by_config.items(), key=lambda item: min(row["config_order"] for row in item[1])
    ):
        previous = None
        for point in sorted(rows, key=lambda row: row["prompt_length"]):
            trend = {
                "config_id": config_id,
                "method": point["method"],
                "prompt_length": point["prompt_length"],
                "paper_total_bytes": point["paper_total_bytes"],
                "paper_total_mib": point["paper_total_mib"],
                "paper_bytes_per_prompt_token": point["paper_bytes_per_prompt_token"],
                "primary_error": point["primary_error"],
                "primary_error_sample_pstdev": point["primary_error_sample_pstdev"],
                "compression_ratio_vs_fp16": point["compression_ratio_vs_fp16"],
                "is_pareto_global": point["is_pareto_global"],
                "previous_prompt_length": None,
                "paper_growth_ratio_vs_previous": None,
                "primary_error_delta_vs_previous": None,
                "primary_error_ratio_vs_previous": None,
            }
            if previous is not None:
                trend["previous_prompt_length"] = previous["prompt_length"]
                trend["paper_growth_ratio_vs_previous"] = (
                    point["paper_total_bytes"] / previous["paper_total_bytes"]
                )
                trend["primary_error_delta_vs_previous"] = (
                    point["primary_error"] - previous["primary_error"]
                )
                if previous["primary_error"] != 0:
                    trend["primary_error_ratio_vs_previous"] = (
                        point["primary_error"] / previous["primary_error"]
                    )
            trends.append(trend)
            previous = point
    return trends


def build_sample_points(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand aggregate points into auditable per-sample Pareto rows."""

    rows = []
    for point in points:
        errors = json.loads(point["primary_error_by_sample_json"])
        for sample_order, sample_id in enumerate(json.loads(point["sample_ids_json"])):
            rows.append({
                "config_id": point["config_id"],
                "config_order": point["config_order"],
                "method": point["method"],
                "prompt_length": point["prompt_length"],
                "sample_id": sample_id,
                "sample_order": sample_order,
                "paper_total_bytes": point["paper_total_bytes"],
                "paper_total_mib": point["paper_total_mib"],
                "primary_error": errors[sample_id],
            })

    by_sample_length: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample_length[(row["sample_id"], row["prompt_length"])].append(row)
    for group in by_sample_length.values():
        for row in group:
            row["dominated_by_count_global"] = sum(
                _dominates(other, row) for other in group if other is not row
            )
            row["is_pareto_global"] = row["dominated_by_count_global"] == 0

    return sorted(
        rows,
        key=lambda row: (
            row["prompt_length"],
            row["sample_order"],
            row["config_order"],
        ),
    )


def write_analysis_outputs(
    analysis_dir: str | Path,
    points: Sequence[dict[str, Any]],
    trends: Sequence[dict[str, Any]],
    *,
    resolved_manifest: dict[str, Any],
    run_count: int,
    make_plots: bool = True,
) -> list[Path]:
    """Write deterministic tables, protocol metadata, Markdown, and optional figures."""

    destination = Path(analysis_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ParetoAnalysisError(f"analysis directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    ordered_points = list(points)
    pareto_points = [point for point in ordered_points if point["is_pareto_global"]]
    sample_points = build_sample_points(ordered_points)
    output_paths = [
        _write_jsonl(destination / "aggregate_points.jsonl", ordered_points),
        _write_csv(destination / "aggregate_points.csv", ordered_points),
        _write_jsonl(destination / "pareto_points.jsonl", pareto_points),
        _write_csv(destination / "pareto_points.csv", pareto_points),
        _write_jsonl(destination / "sample_points.jsonl", sample_points),
        _write_csv(destination / "sample_points.csv", sample_points),
        _write_jsonl(destination / "trends.jsonl", trends),
        _write_csv(destination / "trends.csv", trends),
    ]

    source_states = sorted(
        {
            _canonical_json(run_source)
            for run_source in _source_states_from_points(ordered_points)
        }
    )
    protocol = {
        "schema_version": 2,
        "run_count": run_count,
        "aggregate_point_count": len(ordered_points),
        "pareto_point_count": len(pareto_points),
        "sample_point_count": len(sample_points),
        "sample_aggregation": "arithmetic mean over run-level layer means",
        "sample_dispersion": "population standard deviation",
        "sample_pareto": "computed separately within each sample and prompt length",
        "primary_error_metric": PRIMARY_METRIC,
        "primary_error_field": "metrics_aggregate.joint_post_o_proj_mse.mean",
        "memory_axis": "memory.paper_estimate.total_bytes",
        "pareto_scope": "separate within each prompt length",
        "pareto_dominance": "both objectives no worse and at least one strictly better",
        "fp16_included": True,
        "runtime_tensors_role": "diagnostic fake-quant tensor storage only",
        "cuda_peak_role": "whole-point diagnostic only; not a cache or kernel memory claim",
        "source_states": [json.loads(value) for value in source_states],
        "resolved_dimensions": {
            "sample_ids": resolved_manifest["sample_ids"],
            "prompt_lengths": resolved_manifest["prompt_lengths"],
            "method_ids": [method["id"] for method in resolved_manifest["methods"]],
        },
    }
    protocol_path = destination / "analysis_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_paths.append(protocol_path)

    markdown_path = destination / "pareto_summary.md"
    markdown_path.write_text(_render_markdown(ordered_points), encoding="utf-8")
    output_paths.append(markdown_path)

    if make_plots:
        output_paths.extend(_plot_pareto(destination, ordered_points))
    return output_paths


def _aggregate_group(
    group: Sequence[dict[str, Any]],
    method: dict[str, Any],
    prompt_length: int,
) -> dict[str, Any]:
    ordered = sorted(group, key=lambda run: run["input"]["sample_id"])
    paper = _require_identical_memory(ordered, "paper_estimate")
    runtime = _require_identical_memory(ordered, "runtime_tensors")
    _require_fp16_invariants(ordered, method["method"])

    row: dict[str, Any] = {
        "config_id": method["config_id"],
        "config_order": method["config_order"],
        "method": method["method"],
        "method_config_json": _canonical_json(method["method_config"]),
        "prompt_length": prompt_length,
        "sample_count": len(ordered),
        "sample_ids_json": _canonical_json([run["input"]["sample_id"] for run in ordered]),
        "run_ids_json": _canonical_json([run["run_id"] for run in ordered]),
        "source_state_json": _canonical_json(ordered[0]["provenance"]["source_state"]),
        "paper_cache_type": paper["cache_type"],
        "runtime_tensor_cache_type": runtime["cache_type"],
        "paper_total_mib": paper["total_bytes"] / MIB,
        "runtime_tensor_total_mib": runtime["total_bytes"] / MIB,
        "paper_bytes_per_prompt_token": paper["total_bytes"] / prompt_length,
    }
    for field in BYTE_FIELDS:
        row[f"paper_{field}"] = paper[field]
        row[f"runtime_tensor_{field}"] = runtime[field]

    for field in ("max_allocated_bytes", "max_reserved_bytes"):
        values = [run["memory"]["cuda_peak_diagnostic"][field] for run in ordered]
        row[f"cuda_peak_diagnostic_{field}_sample_mean"] = statistics.fmean(values)
        row[f"cuda_peak_diagnostic_{field}_sample_min"] = min(values)
        row[f"cuda_peak_diagnostic_{field}_sample_max"] = max(values)

    for metric in METRIC_NAMES:
        layer_means = [run["metrics_aggregate"][metric]["mean"] for run in ordered]
        layer_medians = [run["metrics_aggregate"][metric]["median"] for run in ordered]
        layer_maxima = [run["metrics_aggregate"][metric]["max"] for run in ordered]
        prefix = f"{metric}.layer_mean"
        row[f"{prefix}.sample_mean"] = statistics.fmean(layer_means)
        row[f"{prefix}.sample_pstdev"] = statistics.pstdev(layer_means)
        row[f"{prefix}.sample_min"] = min(layer_means)
        row[f"{prefix}.sample_max"] = max(layer_means)
        row[f"{metric}.layer_median.sample_mean"] = statistics.fmean(layer_medians)
        row[f"{metric}.layer_max.sample_mean"] = statistics.fmean(layer_maxima)

    primary_prefix = f"{PRIMARY_METRIC}.layer_mean"
    row["primary_error"] = row[f"{primary_prefix}.sample_mean"]
    row["primary_error_sample_pstdev"] = row[f"{primary_prefix}.sample_pstdev"]
    row["primary_error_sample_min"] = row[f"{primary_prefix}.sample_min"]
    row["primary_error_sample_max"] = row[f"{primary_prefix}.sample_max"]
    row["primary_error_by_sample_json"] = _canonical_json({
        run["input"]["sample_id"]: run["metrics_aggregate"][PRIMARY_METRIC]["mean"]
        for run in ordered
    })
    return row


def _attach_fp16_normalization(points: Sequence[dict[str, Any]]) -> None:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        by_length[point["prompt_length"]].append(point)
    for prompt_length, rows in by_length.items():
        baselines = [row for row in rows if row["method"] == "fp16"]
        if len(baselines) != 1:
            raise ParetoAnalysisError(
                f"prompt {prompt_length} requires exactly one FP16 baseline, got {len(baselines)}"
            )
        fp16_bytes = baselines[0]["paper_total_bytes"]
        for row in rows:
            fraction = row["paper_total_bytes"] / fp16_bytes
            row["memory_fraction_of_fp16"] = fraction
            row["memory_savings_fraction_vs_fp16"] = 1.0 - fraction
            row["compression_ratio_vs_fp16"] = fp16_bytes / row["paper_total_bytes"]


def _attach_pareto_flags(points: Sequence[dict[str, Any]]) -> None:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        by_length[point["prompt_length"]].append(point)
    for rows in by_length.values():
        for point in rows:
            global_dominators = [
                other for other in rows if other is not point and _dominates(other, point)
            ]
            method_dominators = [
                other
                for other in rows
                if other is not point
                and other["method"] == point["method"]
                and _dominates(other, point)
            ]
            point["dominated_by_count_global"] = len(global_dominators)
            point["dominated_by_count_within_method"] = len(method_dominators)
            point["is_pareto_global"] = not global_dominators
            point["is_pareto_within_method"] = not method_dominators


def _attach_sample_pareto_stability(points: Sequence[dict[str, Any]]) -> None:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        by_length[point["prompt_length"]].append(point)

    for rows in by_length.values():
        sample_ids = json.loads(rows[0]["sample_ids_json"])
        for point in rows[1:]:
            if json.loads(point["sample_ids_json"]) != sample_ids:
                raise ParetoAnalysisError(
                    f"prompt {point['prompt_length']} has inconsistent ordered sample IDs"
                )

        pareto_samples_by_config: dict[str, list[str]] = defaultdict(list)
        for sample_id in sample_ids:
            sample_rows = []
            for point in rows:
                errors = json.loads(point["primary_error_by_sample_json"])
                sample_rows.append({
                    "config_id": point["config_id"],
                    "paper_total_bytes": point["paper_total_bytes"],
                    "primary_error": errors[sample_id],
                })
            for sample_point in sample_rows:
                if not any(
                    _dominates(other, sample_point)
                    for other in sample_rows
                    if other is not sample_point
                ):
                    pareto_samples_by_config[sample_point["config_id"]].append(sample_id)

        for point in rows:
            pareto_samples = pareto_samples_by_config[point["config_id"]]
            point["pareto_sample_count"] = len(pareto_samples)
            point["pareto_sample_fraction"] = len(pareto_samples) / len(sample_ids)
            point["pareto_sample_ids_json"] = _canonical_json(pareto_samples)
            point["is_pareto_all_samples"] = len(pareto_samples) == len(sample_ids)
            point["is_pareto_any_sample"] = bool(pareto_samples)


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_memory = left["paper_total_bytes"]
    right_memory = right["paper_total_bytes"]
    left_error = left["primary_error"]
    right_error = right["primary_error"]
    return (
        left_memory <= right_memory
        and left_error <= right_error
        and (left_memory < right_memory or left_error < right_error)
    )


def _require_identical_memory(
    runs: Sequence[dict[str, Any]], namespace: str
) -> dict[str, Any]:
    reference = runs[0]["memory"][namespace]
    for run in runs[1:]:
        actual = run["memory"][namespace]
        if actual != reference:
            raise ParetoAnalysisError(
                f"{runs[0]['method']['name']} prompt {runs[0]['input']['prompt_length']} "
                f"has sample-dependent {namespace} memory: "
                f"{runs[0]['run_id']} differs from {run['run_id']}"
            )
    return reference


def _require_fp16_invariants(runs: Sequence[dict[str, Any]], method: str) -> None:
    if method != "fp16":
        return
    for run in runs:
        for metric in METRIC_NAMES:
            expected = 1.0 if metric == "topk_attention_overlap" else 0.0
            for statistic_name, value in run["metrics_aggregate"][metric].items():
                if value != expected:
                    raise ParetoAnalysisError(
                        f"FP16 run {run['run_id']} has {metric}.{statistic_name}={value}; "
                        f"expected {expected}"
                    )


def _method_catalog(resolved_manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    catalog = {}
    for order, method in enumerate(resolved_manifest["methods"]):
        identity = (method["method"], _canonical_json(method["method_config"]))
        if identity in catalog:
            raise ParetoAnalysisError(f"duplicate resolved method identity {identity}")
        catalog[identity] = {
            "config_id": method["id"],
            "config_order": order,
            "method": method["method"],
            "method_config": method["method_config"],
        }
    return catalog


def _source_states_from_points(points: Sequence[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for point in points:
        yield json.loads(point["source_state_json"])


def _render_markdown(points: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# CAGE-KV memory–perturbation Pareto summary",
        "",
        "Primary error is the arithmetic mean across samples of each run's 32-layer mean "
        "`joint_post_o_proj_mse`. Memory is `memory.paper_estimate.total_bytes`. "
        "Pareto optimality is computed separately within each prompt length.",
        "",
        "> This is a local teacher-forced cache-perturbation result. It is not a downstream "
        "quality, latency, throughput, or realized-kernel-memory claim.",
        "",
    ]
    lengths = sorted({point["prompt_length"] for point in points})
    for prompt_length in lengths:
        lines.extend([
            f"## Prompt length {prompt_length}",
            "",
            "| Configuration | Method | Paper MiB | Compression vs FP16 | Primary error | Sample σ | Mean Pareto | Sample Pareto |",
            "|---|---:|---:|---:|---:|---:|:---:|:---:|",
        ])
        rows = [point for point in points if point["prompt_length"] == prompt_length]
        for point in sorted(rows, key=lambda row: row["config_order"]):
            lines.append(
                f"| {point['config_id']} | {point['method']} | {point['paper_total_mib']:.3f} | "
                f"{point['compression_ratio_vs_fp16']:.3f}× | {point['primary_error']:.6e} | "
                f"{point['primary_error_sample_pstdev']:.3e} | "
                f"{'yes' if point['is_pareto_global'] else 'no'} | "
                f"{point['pareto_sample_count']}/{point['sample_count']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _plot_pareto(destination: Path, points: Sequence[dict[str, Any]]) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ParetoAnalysisError(
            "matplotlib is required for plots; install it or pass --no-plots"
        ) from error

    lengths = sorted({point["prompt_length"] for point in points})
    columns = 2
    rows_count = math.ceil(len(lengths) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(13.5, 5.6 * rows_count),
        squeeze=False,
    )
    config_colors = {
        "kivi-g32-r32": "#f28e2b",
        "kivi-g32-r64": "#ff9d4d",
        "kivi-g32-r128": "#ffbe7d",
        "kivi-g64-r64": "#d55e00",
        "kivi-g64-r128": "#a05a2c",
        "kivi-g128-r128": "#7f3c0a",
        "cage-r32": "#8cd17d",
        "cage-r64": "#59a14f",
        "cage-r128": "#176b3a",
    }
    markers = {"kivi": "o", "cage": "^"}

    for axis, prompt_length in zip(axes.flat, lengths):
        rows = [point for point in points if point["prompt_length"] == prompt_length]
        fp16 = next(point for point in rows if point["method"] == "fp16")
        quantized = [point for point in rows if point["method"] != "fp16"]
        for point in sorted(quantized, key=lambda row: row["config_order"]):
            color = config_colors[point["config_id"]]
            alpha = 1.0 if point["is_pareto_global"] else 0.35
            axis.errorbar(
                point["paper_total_mib"],
                point["primary_error"],
                yerr=point["primary_error_sample_pstdev"],
                color=color,
                alpha=alpha,
                linewidth=0.8,
                capsize=2,
                fmt="none",
                zorder=2,
            )
            axis.scatter(
                point["paper_total_mib"],
                point["primary_error"],
                color=color,
                marker=markers[point["method"]],
                s=62,
                alpha=alpha,
                edgecolor="#222222" if point["is_pareto_global"] else "none",
                linewidth=0.9,
                zorder=3,
            )
        frontier = sorted(
            (point for point in quantized if point["is_pareto_global"]),
            key=lambda point: point["paper_total_mib"],
        )
        axis.plot(
            [point["paper_total_mib"] for point in frontier],
            [point["primary_error"] for point in frontier],
            color="#333333",
            linewidth=1.2,
            linestyle="--",
            zorder=2,
        )
        x_values = [point["paper_total_mib"] for point in quantized]
        x_padding = max(x_values) - min(x_values)
        x_padding = max(x_padding * 0.08, max(x_values) * 0.01)
        axis.set_xlim(min(x_values) - x_padding, max(x_values) + x_padding)
        axis.set_yscale("log")
        axis.set_title(f"Prompt length {prompt_length}")
        axis.set_xlabel("Paper-facing packed KV cache (MiB)")
        axis.set_ylabel("Joint post-$W_O$ MSE (sample mean of layer means)")
        axis.grid(True, which="both", alpha=0.25)
        axis.text(
            0.98,
            0.04,
            f"FP16 reference: {fp16['paper_total_mib']:.1f} MiB, MSE = 0",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#4c78a8",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85,
                  "edgecolor": "#4c78a8", "linewidth": 0.6},
        )

    for axis in axes.flat[len(lengths):]:
        axis.set_visible(False)
    from matplotlib.lines import Line2D

    first_length_rows = [
        point
        for point in points
        if point["prompt_length"] == lengths[0] and point["method"] != "fp16"
    ]
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="none",
            marker=markers[point["method"]],
            markerfacecolor=config_colors[point["config_id"]],
            markeredgecolor="#222222",
            markeredgewidth=0.7,
            markersize=7,
        )
        for point in sorted(first_length_rows, key=lambda row: row["config_order"])
    ]
    legend_labels = [
        point["config_id"]
        for point in sorted(first_length_rows, key=lambda row: row["config_order"])
    ]
    legend_handles.append(Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.2))
    legend_labels.append("Quantized Pareto front")
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, 0.012),
    )
    figure.suptitle(
        "CAGE-KV local memory–perturbation Pareto pilot\n"
        "Quantized operating region; error bars are ±1 sample population σ",
        fontsize=14,
        y=0.985,
    )
    figure.tight_layout(rect=(0, 0.095, 1, 0.94))

    png_path = destination / "memory_perturbation_pareto.png"
    pdf_path = destination / "memory_perturbation_pareto.pdf"
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path)
    plt.close(figure)
    return [png_path, pdf_path]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ParetoAnalysisError(f"blank line {line_number} in {path}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ParetoAnalysisError(f"row {line_number} in {path} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ParetoAnalysisError(f"cannot read JSONL {path}: {error}") from error
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
            handle.write("\n")
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    fields = _ordered_fields(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _ordered_fields(rows: Sequence[dict[str, Any]]) -> list[str]:
    fields = sorted({field for row in rows for field in row})
    preferred = [
        "config_id",
        "config_order",
        "method",
        "prompt_length",
        "sample_count",
        "paper_total_bytes",
        "paper_total_mib",
        "primary_error",
        "primary_error_sample_pstdev",
        "compression_ratio_vs_fp16",
        "is_pareto_global",
        "is_pareto_within_method",
    ]
    ordered = [field for field in preferred if field in fields]
    return ordered + [field for field in fields if field not in ordered]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _id_set_error(name: str, expected: set[str], actual: set[str]) -> str:
    return (
        f"{name} differ from summary IDs: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}"
    )


__all__ = [
    "PRIMARY_METRIC",
    "ParetoAnalysisError",
    "aggregate_points",
    "build_trends",
    "load_completed_matrix",
    "write_analysis_outputs",
]
