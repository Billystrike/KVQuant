"""Read-only analysis for the frozen Llama-2-7B CAGE mechanism ablation."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils.cage_experiment_schema import BYTE_FIELDS, METRIC_NAMES
from utils.cage_pareto import ParetoAnalysisError, load_completed_matrix


ABLATION_ANALYSIS_SCHEMA_VERSION = 1
PRIMARY_METRIC = "joint_post_o_proj_mse"
MIB = 1024**2
EXPECTED_SAMPLES = ("doc-001", "doc-002", "doc-003")
EXPECTED_LENGTHS = (512, 1024, 2048, 4095)
EXPECTED_METHOD_IDS = (
    "kivi-g64-r64",
    "kivi-g32-r128",
    "cage-r64-full",
    "cage-r64-k-adaptive",
    "cage-r64-v-adaptive",
    "cage-r64-uniform",
    "cage-r64-fixed-random",
    "cage-r128-full",
    "cage-r128-k-adaptive",
    "cage-r128-v-adaptive",
    "cage-r128-uniform",
    "cage-r128-fixed-random",
)
RESIDUALS = (64, 128)
ROLES = ("full", "k-adaptive", "v-adaptive", "uniform", "fixed-random")
CONTRAST_BASELINE_ROLES = ("fixed-random", "uniform", "k-adaptive", "v-adaptive")
KIVI_BY_RESIDUAL = {64: "kivi-g64-r64", 128: "kivi-g32-r128"}


class AblationAnalysisError(ValueError):
    """Raised when frozen ablation artifacts do not match the protocol."""


def load_completed_ablation_matrix(
    results_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Strictly validate and load the frozen 144-point matrix."""

    try:
        resolved, runs = load_completed_matrix(results_dir)
    except ParetoAnalysisError as error:
        raise AblationAnalysisError(str(error)) from error
    _validate_protocol(resolved)
    if len(runs) != 144:
        raise AblationAnalysisError(f"ablation matrix requires 144 runs, got {len(runs)}")
    return resolved, runs


def aggregate_ablation_results(
    runs: Sequence[dict[str, Any]],
    resolved_manifest: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build aggregate, paired, factorial, advancement, and baseline tables."""

    _validate_protocol(resolved_manifest)
    if len(runs) != 144:
        raise AblationAnalysisError(f"ablation aggregation requires 144 runs, got {len(runs)}")
    methods = _method_catalog(resolved_manifest)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    for run in runs:
        identity = (run["method"]["name"], _canonical_json(run["method"]["resolved_config"]))
        if identity not in methods:
            raise AblationAnalysisError(
                f"run {run['run_id']} has a method absent from the frozen manifest"
            )
        method = methods[identity]
        sample_id = run["input"]["sample_id"]
        prompt_length = run["input"]["prompt_length"]
        if sample_id not in EXPECTED_SAMPLES:
            raise AblationAnalysisError(f"run {run['run_id']} has unexpected sample {sample_id!r}")
        if prompt_length not in EXPECTED_LENGTHS:
            raise AblationAnalysisError(
                f"run {run['run_id']} has unexpected prompt length {prompt_length}"
            )
        combination = (method["config_id"], sample_id, prompt_length)
        if combination in seen:
            raise AblationAnalysisError(f"duplicate scientific combination {combination}")
        seen.add(combination)
        groups[(method["config_id"], prompt_length)].append(run)

    points = []
    for method in sorted(methods.values(), key=lambda item: item["config_order"]):
        for prompt_length in EXPECTED_LENGTHS:
            group = groups.get((method["config_id"], prompt_length), [])
            samples = {run["input"]["sample_id"] for run in group}
            if samples != set(EXPECTED_SAMPLES) or len(group) != len(EXPECTED_SAMPLES):
                raise AblationAnalysisError(
                    f"{method['config_id']} prompt {prompt_length} sample coverage differs: "
                    f"expected {list(EXPECTED_SAMPLES)}, got {sorted(samples)}"
                )
            points.append(_aggregate_group(group, method, prompt_length))
    if len(points) != 48:
        raise AblationAnalysisError(f"aggregated {len(points)} points; expected 48")

    point_lookup = {(row["config_id"], row["prompt_length"]): row for row in points}
    sample_rows, contrast_rows = _build_contrasts(point_lookup)
    factorial_rows = _build_factorial_rows(point_lookup)
    advancement_rows = _build_advancement_rows(point_lookup)
    external_rows = _build_external_rows(point_lookup)
    return {
        "aggregate_points": points,
        "sample_contrasts": sample_rows,
        "paired_contrasts": contrast_rows,
        "factorial_decomposition": factorial_rows,
        "advancement_decisions": advancement_rows,
        "external_baselines": external_rows,
    }


def write_ablation_analysis_outputs(
    analysis_dir: str | Path,
    tables: dict[str, list[dict[str, Any]]],
    *,
    resolved_manifest: dict[str, Any],
    run_count: int,
    make_plots: bool = True,
) -> list[Path]:
    """Write deterministic tables, protocol metadata, summary, and figures."""

    destination = Path(analysis_dir)
    if destination.exists() and any(destination.iterdir()):
        raise AblationAnalysisError(f"analysis directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    table_names = (
        "aggregate_points",
        "sample_contrasts",
        "paired_contrasts",
        "factorial_decomposition",
        "advancement_decisions",
        "external_baselines",
    )
    for name in table_names:
        rows = tables[name]
        outputs.append(_write_jsonl(destination / f"{name}.jsonl", rows))
        outputs.append(_write_csv(destination / f"{name}.csv", rows))

    protocol = {
        "schema_version": ABLATION_ANALYSIS_SCHEMA_VERSION,
        "input_run_count": run_count,
        "aggregate_point_count": len(tables["aggregate_points"]),
        "sample_contrast_count": len(tables["sample_contrasts"]),
        "paired_contrast_count": len(tables["paired_contrasts"]),
        "factorial_row_count": len(tables["factorial_decomposition"]),
        "advancement_row_count": len(tables["advancement_decisions"]),
        "external_baseline_row_count": len(tables["external_baselines"]),
        "method_ids": list(EXPECTED_METHOD_IDS),
        "sample_ids": list(EXPECTED_SAMPLES),
        "prompt_lengths": list(EXPECTED_LENGTHS),
        "primary_error_metric": PRIMARY_METRIC,
        "primary_error_field": "metrics_aggregate.joint_post_o_proj_mse.mean",
        "memory_axis": "memory.paper_estimate.total_bytes",
        "sample_aggregation": "arithmetic mean over three natural-text samples",
        "sample_dispersion": "population standard deviation",
        "paired_direction": "full CAGE minus named ablation; negative error favors full",
        "factorial_interaction": "full - key_only - value_only + uniform; negative favors complementary adaptation",
        "fixed_random_role": "same scheduled bucket counts and exact paper/runtime byte budget as full CAGE",
        "advancement_rule": (
            "advance a side-only variant when aggregate local error is not worse than "
            "uniform at at least three of four prompt lengths"
        ),
        "inference_boundary": (
            "three deterministic documents support descriptive paired mechanism evidence "
            "only; no population-significance claim"
        ),
        "claim_boundary": (
            "paper-estimate packed cache and local teacher-forced perturbation only; "
            "no latency, throughput, realized-memory, or fused-kernel claim"
        ),
        "source_state": json.loads(tables["aggregate_points"][0]["source_state_json"]),
        "resolved_output_dir": resolved_manifest["output_dir"],
    }
    protocol_path = destination / "analysis_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    outputs.append(protocol_path)
    summary_path = destination / "ablation_summary.md"
    summary_path.write_text(_render_markdown(tables), encoding="utf-8")
    outputs.append(summary_path)
    if make_plots:
        outputs.extend(_plot_ablation(destination, tables))
    return outputs


def _validate_protocol(resolved: dict[str, Any]) -> None:
    if tuple(resolved.get("sample_ids", ())) != EXPECTED_SAMPLES:
        raise AblationAnalysisError("ablation sample IDs differ from the frozen protocol")
    if tuple(resolved.get("prompt_lengths", ())) != EXPECTED_LENGTHS:
        raise AblationAnalysisError("ablation prompt lengths differ from the frozen protocol")
    methods = resolved.get("methods")
    if not isinstance(methods, list) or tuple(item.get("id") for item in methods) != EXPECTED_METHOD_IDS:
        raise AblationAnalysisError("ablation method IDs/order differ from the frozen protocol")
    for method in methods:
        config_id = method["id"]
        config = method["method_config"]
        if config_id.startswith("kivi-"):
            expected = (64, 64) if config_id == "kivi-g64-r64" else (32, 128)
            if method["method"] != "kivi" or (
                config.get("group_size"), config.get("residual_length")
            ) != expected:
                raise AblationAnalysisError(f"{config_id} differs from the frozen KIVI baseline")
            continue
        residual, role = _parse_cage_id(config_id)
        if method["method"] != "cage" or config.get("residual_length") != residual:
            raise AblationAnalysisError(f"{config_id} residual/method differs from protocol")
        if config.get("cage_ablation") is not True or config.get("cage_assignment_seed") != 1729:
            raise AblationAnalysisError(f"{config_id} lacks frozen ablation identity/seed")
        if config.get("k_bits") != 2 or config.get("v_bits") != 2:
            raise AblationAnalysisError(f"{config_id} must retain 2-bit Key and Value")
        adaptive_k = role in {"full", "k-adaptive", "fixed-random"}
        adaptive_v = role in {"full", "v-adaptive", "fixed-random"}
        _validate_side(config_id, config, "k", adaptive_k, role)
        _validate_side(config_id, config, "v", adaptive_v, role)


def _validate_side(
    config_id: str, config: dict[str, Any], side: str, adaptive: bool, role: str
) -> None:
    prefix = f"cage_{side}_"
    expected_importance = "fixed_random" if role == "fixed-random" else (
        "q2_var" if side == "k" else "wo_var"
    )
    expected_groups = [32, 64, 128] if adaptive else [64]
    expected_clips = [0.999, 0.995, 0.99] if adaptive else [0.995]
    expected_buckets = 3 if adaptive else 1
    actual = (
        config.get(prefix + "importance"),
        config.get(prefix + "group_sizes"),
        config.get(prefix + "clip_percentiles"),
        config.get(prefix + "num_buckets"),
    )
    expected = (expected_importance, expected_groups, expected_clips, expected_buckets)
    if actual != expected:
        raise AblationAnalysisError(
            f"{config_id} {side.upper()} policy differs from protocol: {actual} != {expected}"
        )


def _method_catalog(resolved: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    catalog = {}
    for order, method in enumerate(resolved["methods"]):
        identity = (method["method"], _canonical_json(method["method_config"]))
        residual, role = (None, "external-kivi")
        if method["method"] == "cage":
            residual, role = _parse_cage_id(method["id"])
        elif method["id"] == "kivi-g64-r64":
            residual = 64
        elif method["id"] == "kivi-g32-r128":
            residual = 128
        catalog[identity] = {
            "config_id": method["id"],
            "config_order": order,
            "method": method["method"],
            "method_config": method["method_config"],
            "residual_length": residual,
            "ablation_role": role,
        }
    return catalog


def _aggregate_group(
    group: Sequence[dict[str, Any]], method: dict[str, Any], prompt_length: int
) -> dict[str, Any]:
    ordered = sorted(group, key=lambda run: EXPECTED_SAMPLES.index(run["input"]["sample_id"]))
    paper = _identical_memory(ordered, "paper_estimate")
    runtime = _identical_memory(ordered, "runtime_tensors")
    row: dict[str, Any] = {
        **{key: method[key] for key in (
            "config_id", "config_order", "method", "residual_length", "ablation_role"
        )},
        "method_config_json": _canonical_json(method["method_config"]),
        "prompt_length": prompt_length,
        "sample_count": len(ordered),
        "sample_ids_json": _canonical_json(list(EXPECTED_SAMPLES)),
        "run_ids_json": _canonical_json([run["run_id"] for run in ordered]),
        "source_state_json": _canonical_json(ordered[0]["provenance"]["source_state"]),
        "paper_total_mib": paper["total_bytes"] / MIB,
        "runtime_tensor_total_mib": runtime["total_bytes"] / MIB,
    }
    for field in BYTE_FIELDS:
        row[f"paper_{field}"] = paper[field]
        row[f"runtime_tensor_{field}"] = runtime[field]
    for metric in METRIC_NAMES:
        values = [run["metrics_aggregate"][metric]["mean"] for run in ordered]
        row[f"{metric}.sample_mean"] = statistics.fmean(values)
        row[f"{metric}.sample_pstdev"] = statistics.pstdev(values)
        row[f"{metric}.sample_min"] = min(values)
        row[f"{metric}.sample_max"] = max(values)
    values = [run["metrics_aggregate"][PRIMARY_METRIC]["mean"] for run in ordered]
    row["primary_error"] = statistics.fmean(values)
    row["primary_error_sample_pstdev"] = statistics.pstdev(values)
    row["primary_error_by_sample_json"] = _canonical_json(dict(zip(EXPECTED_SAMPLES, values)))
    return row


def _build_contrasts(
    lookup: dict[tuple[str, int], dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows = []
    aggregate_rows = []
    order = 0
    for residual in RESIDUALS:
        full_id = f"cage-r{residual}-full"
        for prompt_length in EXPECTED_LENGTHS:
            full = lookup[(full_id, prompt_length)]
            full_errors = json.loads(full["primary_error_by_sample_json"])
            for baseline_role in CONTRAST_BASELINE_ROLES:
                baseline_id = f"cage-r{residual}-{baseline_role}"
                baseline = lookup[(baseline_id, prompt_length)]
                baseline_errors = json.loads(baseline["primary_error_by_sample_json"])
                deltas = []
                for sample_order, sample_id in enumerate(EXPECTED_SAMPLES):
                    delta = full_errors[sample_id] - baseline_errors[sample_id]
                    deltas.append(delta)
                    sample_rows.append({
                        "contrast_id": f"full_vs_{baseline_role}",
                        "contrast_order": order,
                        "residual_length": residual,
                        "prompt_length": prompt_length,
                        "candidate_config_id": full_id,
                        "baseline_config_id": baseline_id,
                        "sample_id": sample_id,
                        "sample_order": sample_order,
                        "candidate_primary_error": full_errors[sample_id],
                        "baseline_primary_error": baseline_errors[sample_id],
                        "paired_error_delta_full_minus_baseline": delta,
                    })
                memory_delta = full["paper_total_bytes"] - baseline["paper_total_bytes"]
                if baseline_role == "fixed-random":
                    if memory_delta != 0 or full["runtime_tensor_total_bytes"] != baseline[
                        "runtime_tensor_total_bytes"
                    ]:
                        raise AblationAnalysisError(
                            f"same-budget random control differs at r={residual}, length={prompt_length}"
                        )
                aggregate_rows.append({
                    "contrast_id": f"full_vs_{baseline_role}",
                    "contrast_order": order,
                    "residual_length": residual,
                    "prompt_length": prompt_length,
                    "candidate_config_id": full_id,
                    "baseline_config_id": baseline_id,
                    "sample_count": len(deltas),
                    "candidate_paper_total_bytes": full["paper_total_bytes"],
                    "baseline_paper_total_bytes": baseline["paper_total_bytes"],
                    "paper_memory_delta_bytes": memory_delta,
                    "paper_memory_change_vs_baseline_percent": (
                        100.0 * memory_delta / baseline["paper_total_bytes"]
                    ),
                    "candidate_primary_error": full["primary_error"],
                    "baseline_primary_error": baseline["primary_error"],
                    "mean_paired_error_delta_full_minus_baseline": statistics.fmean(deltas),
                    "paired_error_delta_sample_pstdev": statistics.pstdev(deltas),
                    "candidate_error_change_vs_baseline_percent": (
                        100.0 * (full["primary_error"] - baseline["primary_error"])
                        / baseline["primary_error"]
                    ),
                    "baseline_error_penalty_vs_full_percent": (
                        100.0 * (baseline["primary_error"] - full["primary_error"])
                        / full["primary_error"]
                    ),
                    "full_better_samples": sum(delta < 0 for delta in deltas),
                    "tied_samples": sum(delta == 0 for delta in deltas),
                    "full_worse_samples": sum(delta > 0 for delta in deltas),
                    "same_paper_and_runtime_budget": baseline_role == "fixed-random",
                })
                order += 1
    return sample_rows, aggregate_rows


def _build_factorial_rows(
    lookup: dict[tuple[str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for residual in RESIDUALS:
        for prompt_length in EXPECTED_LENGTHS:
            points = {
                role: lookup[(f"cage-r{residual}-{role}", prompt_length)]
                for role in ("full", "k-adaptive", "v-adaptive", "uniform")
            }
            errors = {role: point["primary_error"] for role, point in points.items()}
            memory = {role: point["paper_total_bytes"] for role, point in points.items()}
            rows.append({
                "residual_length": residual,
                "prompt_length": prompt_length,
                **{f"{role.replace('-', '_')}_primary_error": errors[role] for role in errors},
                **{f"{role.replace('-', '_')}_paper_total_bytes": memory[role] for role in memory},
                "full_error_reduction_vs_uniform_percent": 100.0 * (errors["uniform"] - errors["full"]) / errors["uniform"],
                "key_only_error_reduction_vs_uniform_percent": 100.0 * (errors["uniform"] - errors["k-adaptive"]) / errors["uniform"],
                "value_only_error_reduction_vs_uniform_percent": 100.0 * (errors["uniform"] - errors["v-adaptive"]) / errors["uniform"],
                "key_effect_uniform_context": errors["k-adaptive"] - errors["uniform"],
                "value_effect_uniform_context": errors["v-adaptive"] - errors["uniform"],
                "value_effect_given_adaptive_key": errors["full"] - errors["k-adaptive"],
                "key_effect_given_adaptive_value": errors["full"] - errors["v-adaptive"],
                "error_interaction_full_minus_key_minus_value_plus_uniform": (
                    errors["full"] - errors["k-adaptive"] - errors["v-adaptive"] + errors["uniform"]
                ),
                "full_memory_change_vs_uniform_percent": 100.0 * (memory["full"] - memory["uniform"]) / memory["uniform"],
                "memory_interaction_full_minus_key_minus_value_plus_uniform": (
                    memory["full"] - memory["k-adaptive"] - memory["v-adaptive"] + memory["uniform"]
                ),
            })
    return rows


def _build_advancement_rows(
    lookup: dict[tuple[str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for residual in RESIDUALS:
        for side_role in ("k-adaptive", "v-adaptive"):
            decisions = []
            for prompt_length in EXPECTED_LENGTHS:
                side = lookup[(f"cage-r{residual}-{side_role}", prompt_length)]
                uniform = lookup[(f"cage-r{residual}-uniform", prompt_length)]
                decisions.append(side["primary_error"] <= uniform["primary_error"])
            count = sum(decisions)
            rows.append({
                "residual_length": residual,
                "side_only_role": side_role,
                "not_worse_than_uniform_length_count": count,
                "total_length_count": len(EXPECTED_LENGTHS),
                "required_length_count": 3,
                "advance_to_ppl_ablation": count >= 3,
                "decision_by_length_json": _canonical_json(dict(zip(EXPECTED_LENGTHS, decisions))),
            })
    return rows


def _build_external_rows(
    lookup: dict[tuple[str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for residual in RESIDUALS:
        cage_id = f"cage-r{residual}-full"
        kivi_id = KIVI_BY_RESIDUAL[residual]
        for prompt_length in EXPECTED_LENGTHS:
            cage = lookup[(cage_id, prompt_length)]
            kivi = lookup[(kivi_id, prompt_length)]
            rows.append({
                "residual_length": residual,
                "prompt_length": prompt_length,
                "candidate_config_id": cage_id,
                "baseline_config_id": kivi_id,
                "candidate_paper_total_bytes": cage["paper_total_bytes"],
                "baseline_paper_total_bytes": kivi["paper_total_bytes"],
                "paper_memory_change_vs_kivi_percent": 100.0 * (cage["paper_total_bytes"] - kivi["paper_total_bytes"]) / kivi["paper_total_bytes"],
                "candidate_primary_error": cage["primary_error"],
                "baseline_primary_error": kivi["primary_error"],
                "primary_error_change_vs_kivi_percent": 100.0 * (cage["primary_error"] - kivi["primary_error"]) / kivi["primary_error"],
            })
    return rows


def _render_markdown(tables: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# CAGE-KV Llama-2-7B mechanism ablation",
        "",
        "Primary error is the three-document arithmetic mean of each run's 32-layer mean "
        "`joint_post_o_proj_mse`. Negative full-minus-ablation error favors full CAGE.",
        "",
        "> These are descriptive local teacher-forced perturbation results. Memory is the "
        "paper-facing packed-cache estimate, not fake-quant runtime storage or CUDA peak.",
        "",
        "## Same-budget importance-ordering control",
        "",
        "| Residual | Length | Random error penalty vs full | Full better samples | Same budget |",
        "|---:|---:|---:|---:|:---:|",
    ]
    random_rows = [row for row in tables["paired_contrasts"] if row["contrast_id"] == "full_vs_fixed-random"]
    for row in random_rows:
        lines.append(
            f"| {row['residual_length']} | {row['prompt_length']} | "
            f"{row['baseline_error_penalty_vs_full_percent']:+.2f}% | "
            f"{row['full_better_samples']}/{row['sample_count']} | "
            f"{'yes' if row['same_paper_and_runtime_budget'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Factorial mechanism decomposition",
        "",
        "| Residual | Length | Full error reduction vs uniform | Key-only reduction | Value-only reduction | Full memory change | Interaction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in tables["factorial_decomposition"]:
        lines.append(
            f"| {row['residual_length']} | {row['prompt_length']} | "
            f"{row['full_error_reduction_vs_uniform_percent']:.2f}% | "
            f"{row['key_only_error_reduction_vs_uniform_percent']:.2f}% | "
            f"{row['value_only_error_reduction_vs_uniform_percent']:.2f}% | "
            f"{row['full_memory_change_vs_uniform_percent']:+.2f}% | "
            f"{row['error_interaction_full_minus_key_minus_value_plus_uniform']:+.3e} |"
        )
    lines.extend([
        "",
        "## Frozen PPL-ablation advancement decision",
        "",
        "| Residual | Side-only variant | Lengths not worse than uniform | Advance |",
        "|---:|---|---:|:---:|",
    ])
    for row in tables["advancement_decisions"]:
        lines.append(
            f"| {row['residual_length']} | {row['side_only_role']} | "
            f"{row['not_worse_than_uniform_length_count']}/{row['total_length_count']} | "
            f"{'yes' if row['advance_to_ppl_ablation'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def _plot_ablation(
    destination: Path, tables: dict[str, list[dict[str, Any]]]
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise AblationAnalysisError(
            "matplotlib is required for plots; install it or pass --no-plots"
        ) from error

    colors = {
        "full": "#176b3a", "k-adaptive": "#59a14f", "v-adaptive": "#8cd17d",
        "uniform": "#b6d7a8", "fixed-random": "#7f3c0a", "external-kivi": "#f28e2b",
    }
    markers = {
        "full": "o", "k-adaptive": "^", "v-adaptive": "v", "uniform": "s",
        "fixed-random": "X", "external-kivi": "D",
    }
    figure, axes = plt.subplots(2, 4, figsize=(16.0, 7.8), squeeze=False)
    for row_index, residual in enumerate(RESIDUALS):
        for column_index, prompt_length in enumerate(EXPECTED_LENGTHS):
            axis = axes[row_index][column_index]
            rows = [
                row for row in tables["aggregate_points"]
                if row["residual_length"] == residual and row["prompt_length"] == prompt_length
            ]
            for row in rows:
                role = row["ablation_role"]
                axis.errorbar(
                    row["paper_total_mib"], row["primary_error"],
                    yerr=row["primary_error_sample_pstdev"], fmt=markers[role],
                    color=colors[role], markersize=6, capsize=3,
                    label=role if row_index == 0 and column_index == 0 else None,
                )
            axis.set_title(f"r={residual}, length={prompt_length}")
            axis.set_xlabel("Paper-facing cache (MiB)")
            axis.set_ylabel("Joint post-$W_O$ MSE")
            axis.grid(True, alpha=0.22)
    handles = [
        plt.Line2D([], [], color=colors[role], marker=markers[role], linestyle="None", label=role)
        for role in ("full", "k-adaptive", "v-adaptive", "uniform", "fixed-random", "external-kivi")
    ]
    figure.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    figure.suptitle("CAGE-KV Llama-2-7B mechanism ablation: memory–perturbation operating points")
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    outputs = _save_figure(figure, destination / "ablation_tradeoff", plt)

    figure, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), sharey=True)
    labels = {
        "full_vs_fixed-random": "fixed-random",
        "full_vs_uniform": "uniform",
        "full_vs_k-adaptive": "Key-only",
        "full_vs_v-adaptive": "Value-only",
    }
    for axis, residual in zip(axes, RESIDUALS):
        for contrast_id, label in labels.items():
            rows = [
                row for row in tables["paired_contrasts"]
                if row["residual_length"] == residual and row["contrast_id"] == contrast_id
            ]
            values = [row["baseline_error_penalty_vs_full_percent"] for row in rows]
            spreads = [
                100.0 * row["paired_error_delta_sample_pstdev"] / row["candidate_primary_error"]
                for row in rows
            ]
            axis.errorbar(
                EXPECTED_LENGTHS, values, yerr=spreads, marker="o", capsize=3,
                linewidth=1.1, label=label,
            )
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_xscale("log", base=2)
        axis.set_xticks(EXPECTED_LENGTHS, EXPECTED_LENGTHS)
        axis.set_xlabel("Prompt length")
        axis.set_ylabel("Ablation error penalty vs full CAGE (%)")
        axis.set_title(f"Residual length {residual}")
        axis.grid(True, alpha=0.22)
    handles, labels_text = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels_text, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("Importance ordering and channel-adaptation effects (±1 sample population σ)")
    figure.tight_layout(rect=(0, 0.1, 1, 0.92))
    outputs.extend(_save_figure(figure, destination / "mechanism_effects", plt))
    return outputs


def _identical_memory(runs: Sequence[dict[str, Any]], namespace: str) -> dict[str, Any]:
    reference = runs[0]["memory"][namespace]
    for run in runs[1:]:
        if run["memory"][namespace] != reference:
            raise AblationAnalysisError(
                f"{runs[0]['method']['name']} prompt {runs[0]['input']['prompt_length']} "
                f"has sample-dependent {namespace} memory"
            )
    return reference


def _parse_cage_id(config_id: str) -> tuple[int, str]:
    for residual in RESIDUALS:
        prefix = f"cage-r{residual}-"
        if config_id.startswith(prefix):
            role = config_id[len(prefix):]
            if role in ROLES:
                return residual, role
    raise AblationAnalysisError(f"unrecognized CAGE ablation ID {config_id!r}")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    fields = sorted({field for row in rows for field in row})
    preferred = [
        "config_id", "contrast_id", "contrast_order", "residual_length",
        "prompt_length", "sample_id", "ablation_role", "candidate_config_id",
        "baseline_config_id", "sample_count", "primary_error", "paper_total_mib",
    ]
    ordered = [field for field in preferred if field in fields]
    ordered.extend(field for field in fields if field not in ordered)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save_figure(figure: Any, base: Path, plt: Any) -> list[Path]:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return [png, pdf]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "ABLATION_ANALYSIS_SCHEMA_VERSION", "AblationAnalysisError",
    "aggregate_ablation_results", "load_completed_ablation_matrix",
    "write_ablation_analysis_outputs",
]
