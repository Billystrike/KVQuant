from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from utils.qwen3_formal import formal_method_length_points
from utils.qwen3_formal_analysis import packed_bytes_for_method
from utils.qwen3_perturbation_protocol import LAYER_METRICS, validate_aggregates, validate_layer_records


RESULTS_RECEIPT_ID = "qwen3-8b-cage-kitty-memory-perturbation-results-receipt-v1"
PRIMARY_METRIC = "joint_post_o_proj_mse"


class Qwen3PerturbationAnalysisError(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_results_receipt(receipt: dict[str, Any]) -> None:
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise Qwen3PerturbationAnalysisError("perturbation results receipt schema mismatch")
    if receipt.get("receipt_id") != RESULTS_RECEIPT_ID:
        raise Qwen3PerturbationAnalysisError("perturbation results receipt ID mismatch")
    if receipt.get("status") != (
        "frozen_after_joint_postrun_audit_before_perturbation_interpretation"
    ):
        raise Qwen3PerturbationAnalysisError("perturbation results receipt status mismatch")
    _require_commit("execution_source_commit", receipt.get("execution_source_commit"))
    _require_commit("audit_source_commit", receipt.get("audit_source_commit"))
    for name in (
        "perturbation_protocol",
        "quality_protocol",
        "quality_execution",
        "acceptance_gate",
        "input_manifest",
        "full_artifact_manifest",
    ):
        record = receipt.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise Qwen3PerturbationAnalysisError(f"receipt {name} record is invalid")
        _require_sha256(f"{name}.sha256", record.get("sha256"))

    audit = receipt.get("joint_postrun_audit")
    if not isinstance(audit, dict) or audit.get("status") != "pass":
        raise Qwen3PerturbationAnalysisError("joint post-run audit receipt is invalid")
    if (
        audit.get("case_count") != 1300
        or audit.get("layer_record_count") != 46800
        or audit.get("failure_count") != 0
        or audit.get("interpretation_performed") is not False
    ):
        raise Qwen3PerturbationAnalysisError("joint post-run audit counts/scope mismatch")
    for name in ("path", "log_path"):
        if not isinstance(audit.get(name), str) or not audit[name]:
            raise Qwen3PerturbationAnalysisError(f"joint audit {name} is invalid")
    for name in ("sha256", "log_sha256", "joint_scientific_payload_sha256"):
        _require_sha256(f"joint_postrun_audit.{name}", audit.get(name))
    if audit.get("log_size_bytes") != 9567:
        raise Qwen3PerturbationAnalysisError("joint audit log size mismatch")

    archive = receipt.get("archive")
    if not isinstance(archive, dict) or not isinstance(archive.get("path"), str):
        raise Qwen3PerturbationAnalysisError("joint archive record is invalid")
    _require_sha256("archive.sha256", archive.get("sha256"))

    partitions = receipt.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {
        "cage_qwen3",
        "kitty_qwen3",
    }:
        raise Qwen3PerturbationAnalysisError("perturbation receipt partition set mismatch")
    expected = {
        "cage_qwen3": (1000, 36000, 180940),
        "kitty_qwen3": (300, 10800, 58429),
    }
    for partition, (case_count, layer_count, log_size) in expected.items():
        record = partitions[partition]
        if (
            record.get("case_count") != case_count
            or record.get("layer_record_count") != layer_count
            or record.get("failure_count") != 0
            or record.get("execution_log_size_bytes") != log_size
        ):
            raise Qwen3PerturbationAnalysisError(f"receipt {partition} counts mismatch")
        for name in ("directory", "execution_log_path"):
            if not isinstance(record.get(name), str) or not record[name]:
                raise Qwen3PerturbationAnalysisError(f"receipt {partition}.{name} is invalid")
        for name in (
            "run_identity_sha256",
            "summary_sha256",
            "execution_log_sha256",
            "case_ids_sha256",
            "canonical_case_file_manifest_sha256",
            "shell_case_file_manifest_sha256",
            "scientific_payload_sha256",
            "model_identity_sha256",
        ):
            _require_sha256(f"receipt {partition}.{name}", record.get(name))

    expected_freeze = {
        "analysis_unit": "anchor_case",
        "layer_aggregation": "use_the_exact_case_level_mean_median_and_maximum_recomputed_from_all_36_layers",
        "primary_metric": PRIMARY_METRIC,
        "primary_case_statistic": "mean",
        "paired_delta": "candidate_case_mean_minus_baseline_case_mean",
        "negative_delta_favors_candidate": True,
        "tie_rule": "exact_zero",
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 20260808,
        "bootstrap_rng": "python_random.Random_reinitialized_per_contrast",
        "bootstrap_sampling": "sample_50_paired_anchor_deltas_with_replacement",
        "bootstrap_interval": "two_sided_95_percentile_type7_linear",
        "primary_comparison_source": "inherit_all_10_frozen_quality_primary_matched_comparisons_without_selection",
        "mechanism_comparison_source": "inherit_all_12_frozen_quality_exact_byte_mechanism_comparisons_without_selection",
        "pareto_objectives": [
            "minimize_complete_active_packed_bytes_at_prompt_length",
            "minimize_mean_joint_post_o_proj_mse",
        ],
        "reported_metric_scope": "all_12_frozen_layer_metrics_with_primary_claims_limited_to_joint_post_o_proj_mse",
        "interpretation_scope": "post_quality_local_mechanism_validation_over_the_deterministic_50_anchor_grid",
        "not_end_to_end_quality": True,
        "not_population_inference": True,
        "memory_representation": "packed_paper_estimate_only",
    }
    if receipt.get("analysis_freeze") != expected_freeze:
        raise Qwen3PerturbationAnalysisError("analysis freeze differs from v1 definition")


def paired_delta_statistics(
    candidate: Mapping[int, float],
    baseline: Mapping[int, float],
    *,
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 20260808,
) -> dict[str, Any]:
    anchors = sorted(candidate)
    if anchors != sorted(baseline) or anchors != list(range(50)):
        raise Qwen3PerturbationAnalysisError(
            "paired statistics require the same complete anchor grid 0..49"
        )
    if bootstrap_resamples <= 0:
        raise Qwen3PerturbationAnalysisError("bootstrap_resamples must be positive")
    deltas = [
        _finite_float(f"candidate[{anchor}]", candidate[anchor])
        - _finite_float(f"baseline[{anchor}]", baseline[anchor])
        for anchor in anchors
    ]
    rng = random.Random(bootstrap_seed)
    bootstrap_means = []
    for _ in range(bootstrap_resamples):
        bootstrap_means.append(
            math.fsum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        )
    bootstrap_means.sort()
    favor = sum(value < 0 for value in deltas)
    tie = sum(value == 0 for value in deltas)
    return {
        "paired_anchor_count": len(deltas),
        "delta_definition": "candidate_case_mean_minus_baseline_case_mean",
        "negative_delta_favors_candidate": True,
        "mean_delta": math.fsum(deltas) / len(deltas),
        "median_delta": statistics.median(deltas),
        "population_standard_deviation": statistics.pstdev(deltas),
        "minimum_delta": min(deltas),
        "maximum_delta": max(deltas),
        "favor_count": favor,
        "tie_count": tie,
        "oppose_count": len(deltas) - favor - tie,
        "bootstrap_95_interval": {
            "low": _type7_quantile(bootstrap_means, 0.025),
            "high": _type7_quantile(bootstrap_means, 0.975),
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "rng": "python_random.Random_reinitialized_per_contrast",
            "quantile": "type7_linear",
        },
        "anchor_deltas": [
            {"anchor_index": anchor, "delta": delta}
            for anchor, delta in zip(anchors, deltas)
        ],
    }


def build_perturbation_analysis(
    *,
    perturbation_protocol: dict[str, Any],
    quality_protocol: dict[str, Any],
    receipt: dict[str, Any],
    records: Sequence[dict[str, Any]],
    receipt_sha256: str,
) -> dict[str, Any]:
    validate_results_receipt(receipt)
    _require_sha256("receipt_sha256", receipt_sha256)
    if len(records) != 1300:
        raise Qwen3PerturbationAnalysisError(
            f"perturbation analysis requires 1300 records, got {len(records)}"
        )
    if perturbation_protocol["metrics"]["primary"] != PRIMARY_METRIC:
        raise Qwen3PerturbationAnalysisError("primary perturbation metric drifted")

    index: dict[tuple[str, int, int], dict[str, Any]] = {}
    methods: dict[tuple[str, int], dict[str, Any]] = {}
    point_records: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        method = record.get("method")
        input_record = record.get("input")
        if not isinstance(method, dict) or not isinstance(input_record, dict):
            raise Qwen3PerturbationAnalysisError("analysis record method/input is missing")
        validate_layer_records(record.get("layer_metrics", []))
        validate_aggregates(record.get("aggregates", {}), record["layer_metrics"])
        key = (
            method["id"],
            int(input_record["prompt_length"]),
            int(input_record["anchor_index"]),
        )
        if key in index:
            raise Qwen3PerturbationAnalysisError(f"duplicate perturbation key {key!r}")
        index[key] = record
        point = key[:2]
        if point in methods and methods[point] != method:
            raise Qwen3PerturbationAnalysisError(f"method config drift at {point!r}")
        methods[point] = method
        point_records[point].append(record)

    expected_points = {
        (point["method_id"], point["prompt_length"])
        for partition in ("cage_qwen3", "kitty_qwen3")
        for point in formal_method_length_points(quality_protocol, partition=partition)
    }
    if set(point_records) != expected_points or len(expected_points) != 26:
        raise Qwen3PerturbationAnalysisError("perturbation method-length grid mismatch")

    base_points = _base_point_identities(quality_protocol)
    summaries = []
    primary_by_anchor: dict[tuple[str, int], dict[int, float]] = {}
    layer_summaries = []
    for point in sorted(point_records, key=lambda value: (value[1], value[0])):
        method_id, prompt_length = point
        ordered = sorted(
            point_records[point], key=lambda value: value["input"]["anchor_index"]
        )
        if [value["input"]["anchor_index"] for value in ordered] != list(range(50)):
            raise Qwen3PerturbationAnalysisError(f"incomplete anchor grid at {point!r}")
        memory_values = {int(value["memory"]["model_total_bytes"]) for value in ordered}
        expected_bytes = packed_bytes_for_method(methods[point], prompt_length)
        if memory_values != {expected_bytes}:
            raise Qwen3PerturbationAnalysisError(f"packed-memory drift at {point!r}")

        metric_summaries = {}
        for metric in LAYER_METRICS:
            metric_summaries[metric] = {
                f"case_aggregate_{statistic}": _distribution(
                    [float(value["aggregates"][metric][statistic]) for value in ordered]
                )
                for statistic in ("mean", "median", "maximum")
            }
            metric_summaries[metric]["all_layer_records"] = _distribution(
                [
                    float(layer["metrics"][metric])
                    for value in ordered
                    for layer in value["layer_metrics"]
                ]
            )
        primary_values = {
            int(value["input"]["anchor_index"]): float(
                value["aggregates"][PRIMARY_METRIC]["mean"]
            )
            for value in ordered
        }
        primary_by_anchor[point] = primary_values
        primary_distribution = _distribution(list(primary_values.values()))
        summaries.append(
            {
                "method_id": method_id,
                "method_family": methods[point]["name"],
                "prompt_length": prompt_length,
                "anchor_count": 50,
                "layer_record_count": 1800,
                "packed_bytes": expected_bytes,
                "is_base_point": point in base_points,
                "primary_metric": PRIMARY_METRIC,
                "primary_case_statistic": "mean",
                "primary_mean_across_anchors": primary_distribution["mean"],
                "primary_median_across_anchors": primary_distribution["median"],
                "primary_population_standard_deviation": primary_distribution[
                    "population_standard_deviation"
                ],
                "primary_minimum": primary_distribution["minimum"],
                "primary_maximum": primary_distribution["maximum"],
                "metrics": metric_summaries,
            }
        )
        for layer_idx in range(36):
            layer_summaries.append(
                {
                    "method_id": method_id,
                    "method_family": methods[point]["name"],
                    "prompt_length": prompt_length,
                    "layer_idx": layer_idx,
                    "anchor_count": 50,
                    "metric_means": {
                        metric: math.fsum(
                            float(value["layer_metrics"][layer_idx]["metrics"][metric])
                            for value in ordered
                        )
                        / 50
                        for metric in LAYER_METRICS
                    },
                }
            )

    summary_index = {
        (value["method_id"], value["prompt_length"]): value for value in summaries
    }
    freeze = receipt["analysis_freeze"]
    for summary in summaries:
        point = (summary["method_id"], summary["prompt_length"])
        fp16 = ("fp16", summary["prompt_length"])
        summary["paired_primary_vs_fp16"] = (
            None
            if point == fp16
            else paired_delta_statistics(
                primary_by_anchor[point],
                primary_by_anchor[fp16],
                bootstrap_resamples=freeze["bootstrap_resamples"],
                bootstrap_seed=freeze["bootstrap_seed"],
            )
        )

    primary = []
    for order, contrast in enumerate(quality_protocol["primary_matched_comparisons"], 1):
        prompt_length = contrast["prompt_length"]
        candidate = (contrast["candidate"], prompt_length)
        baseline = (contrast["baseline"], prompt_length)
        candidate_bytes = summary_index[candidate]["packed_bytes"]
        baseline_bytes = summary_index[baseline]["packed_bytes"]
        primary.append(
            {
                "order": order,
                "comparison_id": (
                    f"l{prompt_length}-{contrast['candidate']}-minus-{contrast['baseline']}"
                ),
                "prompt_length": prompt_length,
                "candidate": contrast["candidate"],
                "baseline": contrast["baseline"],
                "candidate_packed_bytes": candidate_bytes,
                "baseline_packed_bytes": baseline_bytes,
                "candidate_relative_memory_delta_vs_baseline": (
                    candidate_bytes - baseline_bytes
                )
                / baseline_bytes,
                "statistics": paired_delta_statistics(
                    primary_by_anchor[candidate],
                    primary_by_anchor[baseline],
                    bootstrap_resamples=freeze["bootstrap_resamples"],
                    bootstrap_seed=freeze["bootstrap_seed"],
                ),
            }
        )

    mechanism = []
    for point in quality_protocol["mechanism_points"]:
        prompt_length = point["prompt_length"]
        full = point["full_method_id"]
        uniform = f"{full}-fixed-uniform"
        contrasts = [
            (full, f"{full}-fixed-random", "full_vs_fixed_random"),
            (full, uniform, "full_vs_fixed_uniform"),
            (full, f"{full}-key-adaptive-only", "value_adaptation_increment"),
            (full, f"{full}-value-adaptive-only", "key_adaptation_increment"),
            (f"{full}-key-adaptive-only", uniform, "key_only_vs_uniform"),
            (f"{full}-value-adaptive-only", uniform, "value_only_vs_uniform"),
        ]
        for candidate_id, baseline_id, role in contrasts:
            candidate = (candidate_id, prompt_length)
            baseline = (baseline_id, prompt_length)
            candidate_bytes = summary_index[candidate]["packed_bytes"]
            baseline_bytes = summary_index[baseline]["packed_bytes"]
            if candidate_bytes != baseline_bytes:
                raise Qwen3PerturbationAnalysisError(
                    f"mechanism contrast {role} is not exactly byte matched"
                )
            mechanism.append(
                {
                    "comparison_id": f"l{prompt_length}-{role}",
                    "role": role,
                    "prompt_length": prompt_length,
                    "candidate": candidate_id,
                    "baseline": baseline_id,
                    "candidate_packed_bytes": candidate_bytes,
                    "baseline_packed_bytes": baseline_bytes,
                    "statistics": paired_delta_statistics(
                        primary_by_anchor[candidate],
                        primary_by_anchor[baseline],
                        bootstrap_resamples=freeze["bootstrap_resamples"],
                        bootstrap_seed=freeze["bootstrap_seed"],
                    ),
                }
            )

    return {
        "schema_version": 1,
        "status": "pass",
        "analysis_id": "qwen3-8b-cage-kitty-memory-perturbation-analysis-v1",
        "receipt_id": receipt["receipt_id"],
        "receipt_sha256": receipt_sha256,
        "perturbation_protocol_id": perturbation_protocol["protocol_id"],
        "perturbation_protocol_sha256": receipt["perturbation_protocol"]["sha256"],
        "quality_protocol_id": quality_protocol["protocol_id"],
        "quality_protocol_sha256": receipt["quality_protocol"]["sha256"],
        "execution_source_commit": receipt["execution_source_commit"],
        "audit_source_commit": receipt["audit_source_commit"],
        "case_count": len(records),
        "layer_record_count": len(records) * 36,
        "method_length_point_count": len(summaries),
        "base_method_length_point_count": sum(value["is_base_point"] for value in summaries),
        "analysis_freeze": freeze,
        "method_summaries": summaries,
        "layer_summaries": layer_summaries,
        "primary_matched_comparisons": primary,
        "mechanism_comparisons": mechanism,
        "base_pareto_by_prompt_length": _pareto_tables(summaries),
        "evidence": {
            "joint_postrun_audit_sha256": receipt["joint_postrun_audit"]["sha256"],
            "joint_scientific_payload_sha256": receipt["joint_postrun_audit"][
                "joint_scientific_payload_sha256"
            ],
            "archive_sha256": receipt["archive"]["sha256"],
            "partitions": receipt["partitions"],
        },
        "claim_boundary": perturbation_protocol["claim_boundary"],
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    normalized = [_finite_float("distribution value", value) for value in values]
    if not normalized:
        raise Qwen3PerturbationAnalysisError("distribution must not be empty")
    return {
        "count": len(normalized),
        "mean": math.fsum(normalized) / len(normalized),
        "median": statistics.median(normalized),
        "population_standard_deviation": statistics.pstdev(normalized),
        "minimum": min(normalized),
        "maximum": max(normalized),
    }


def _base_point_identities(protocol: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (record["method_id"], prompt_length)
        for record in protocol["base_method_length_points"]
        for prompt_length in record["prompt_lengths"]
    }


def _pareto_tables(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    tables = []
    for prompt_length in (1024, 2048, 4032):
        points = [
            value
            for value in summaries
            if value["prompt_length"] == prompt_length and value["is_base_point"]
        ]
        rows = []
        for point in points:
            dominated_by = []
            for other in points:
                if other is point:
                    continue
                no_worse = (
                    other["packed_bytes"] <= point["packed_bytes"]
                    and other["primary_mean_across_anchors"]
                    <= point["primary_mean_across_anchors"]
                )
                strictly_better = (
                    other["packed_bytes"] < point["packed_bytes"]
                    or other["primary_mean_across_anchors"]
                    < point["primary_mean_across_anchors"]
                )
                if no_worse and strictly_better:
                    dominated_by.append(other["method_id"])
            rows.append(
                {
                    "method_id": point["method_id"],
                    "packed_bytes": point["packed_bytes"],
                    "mean_joint_post_o_proj_mse": point["primary_mean_across_anchors"],
                    "is_pareto": not dominated_by,
                    "dominated_by": sorted(dominated_by),
                }
            )
        tables.append(
            {
                "prompt_length": prompt_length,
                "point_count": len(rows),
                "pareto_method_ids": sorted(
                    value["method_id"] for value in rows if value["is_pareto"]
                ),
                "points": sorted(rows, key=lambda value: (value["packed_bytes"], value["method_id"])),
            }
        )
    return tables


def _type7_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values or not 0 <= probability <= 1:
        raise Qwen3PerturbationAnalysisError("invalid quantile request")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Qwen3PerturbationAnalysisError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Qwen3PerturbationAnalysisError(f"{name} must be finite")
    return result


def _require_sha256(name: str, value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise Qwen3PerturbationAnalysisError(f"{name} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise Qwen3PerturbationAnalysisError(f"{name} must be a SHA-256") from error


def _require_commit(name: str, value: Any) -> None:
    if not isinstance(value, str) or len(value) != 40:
        raise Qwen3PerturbationAnalysisError(f"{name} must be a Git commit")
    try:
        int(value, 16)
    except ValueError as error:
        raise Qwen3PerturbationAnalysisError(f"{name} must be a Git commit") from error


__all__ = [
    "PRIMARY_METRIC",
    "Qwen3PerturbationAnalysisError",
    "RESULTS_RECEIPT_ID",
    "build_perturbation_analysis",
    "canonical_sha256",
    "paired_delta_statistics",
    "validate_results_receipt",
]
