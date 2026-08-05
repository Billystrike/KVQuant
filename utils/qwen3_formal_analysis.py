from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from utils.qwen3_formal import formal_method_length_points
from utils.qwen3_memory import (
    estimate_qwen3_cage_bytes,
    estimate_qwen3_fp16_bytes,
    estimate_qwen3_kivi_bytes,
    estimate_qwen3_kitty_bytes,
)


QWEN3_FORMAL_RESULTS_RECEIPT_ID = (
    "qwen3-8b-cage-kitty-formal-results-receipt-v1"
)


class Qwen3FormalAnalysisError(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_results_receipt(receipt: dict[str, Any]) -> None:
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise Qwen3FormalAnalysisError("formal results receipt schema mismatch")
    if receipt.get("receipt_id") != QWEN3_FORMAL_RESULTS_RECEIPT_ID:
        raise Qwen3FormalAnalysisError("formal results receipt ID mismatch")
    if receipt.get("status") != (
        "frozen_after_complete_execution_before_quality_interpretation"
    ):
        raise Qwen3FormalAnalysisError("formal results receipt status mismatch")
    _require_git_commit("source_commit", receipt.get("source_commit"))
    for name in (
        "protocol",
        "memory_protocol",
        "execution",
        "acceptance_gate",
        "input_manifest",
    ):
        record = receipt.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise Qwen3FormalAnalysisError(f"receipt {name} record is invalid")
        _require_sha256(f"{name}.sha256", record.get("sha256"))
    partitions = receipt.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {
        "cage_qwen3",
        "kitty_qwen3",
    }:
        raise Qwen3FormalAnalysisError("receipt partition set mismatch")
    expected_counts = {"cage_qwen3": 1000, "kitty_qwen3": 300}
    for name, expected_count in expected_counts.items():
        record = partitions[name]
        if record.get("case_count") != expected_count:
            raise Qwen3FormalAnalysisError(f"receipt {name} case count mismatch")
        for path_name in (
            "directory",
            "archive_path",
            "execution_log_path",
            "postrun_audit_path",
            "postrun_audit_log_path",
        ):
            if not isinstance(record.get(path_name), str) or not record[path_name]:
                raise Qwen3FormalAnalysisError(
                    f"receipt {name}.{path_name} is invalid"
                )
        for hash_name in (
            "case_ids_sha256",
            "run_identity_sha256",
            "summary_sha256",
            "archive_sha256",
            "execution_log_sha256",
            "postrun_audit_sha256",
            "postrun_audit_log_sha256",
            "scientific_payload_sha256",
        ):
            _require_sha256(f"{name}.{hash_name}", record.get(hash_name))
        source_state = record.get("source_state")
        if not isinstance(source_state, dict) or source_state.get("dirty") is not False:
            raise Qwen3FormalAnalysisError(f"receipt {name} source state mismatch")
        if source_state.get("git_commit") != receipt["source_commit"]:
            raise Qwen3FormalAnalysisError(f"receipt {name} source commit mismatch")
        for commit_name in ("kitty_commit", "transformers_commit"):
            if commit_name in source_state:
                _require_git_commit(
                    f"receipt {name}.{commit_name}", source_state[commit_name]
                )
    freeze = receipt.get("analysis_freeze")
    expected_freeze = {
        "paired_unit": "anchor",
        "delta": "candidate_mean_nll_minus_baseline_mean_nll",
        "negative_delta_favors_candidate": True,
        "tie_rule": "exact_zero",
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 20260803,
        "bootstrap_rng": "python_random.Random_reinitialized_per_contrast",
        "bootstrap_sampling": "sample_50_paired_anchor_deltas_with_replacement",
        "bootstrap_interval": "two_sided_95_percentile_type7_linear",
        "pareto_objectives": [
            "minimize_complete_active_packed_bytes",
            "minimize_token_weighted_mean_nll",
        ],
        "interpretation_scope": (
            "descriptive_stability_over_deterministic_corpus_grid"
        ),
    }
    if freeze != expected_freeze:
        raise Qwen3FormalAnalysisError("analysis freeze differs from v1 definition")


def packed_bytes_for_method(method: Mapping[str, Any], prompt_length: int) -> int:
    name = method.get("name")
    config = method.get("config")
    if not isinstance(config, Mapping):
        raise Qwen3FormalAnalysisError("method config must be a mapping")
    if name == "fp16":
        report = estimate_qwen3_fp16_bytes(seq_len=prompt_length)
    elif name == "cage":
        report = estimate_qwen3_cage_bytes(
            seq_len=prompt_length,
            residual_length=_positive_int(config, "residual_length"),
            key_bucket_sizes=tuple(config.get("key_bucket_sizes", (42, 43, 43))),
            value_bucket_sizes=tuple(config.get("value_bucket_sizes", (42, 43, 43))),
            key_group_sizes=tuple(config["key_group_sizes"]),
            value_group_sizes=tuple(config["value_group_sizes"]),
            bits=_positive_int(config, "bits"),
        )
    elif name == "kivi":
        report = estimate_qwen3_kivi_bytes(
            seq_len=prompt_length,
            group_size=_positive_int(config, "group_size"),
            residual_length=_positive_int(config, "residual_length"),
            bits=_positive_int(config, "bits"),
        )
    elif name == "kitty":
        report = estimate_qwen3_kitty_bytes(
            seq_len=prompt_length,
            boosted_channels=_positive_int(config, "boosted_channels"),
        )
    else:
        raise Qwen3FormalAnalysisError(f"unsupported formal method family {name!r}")
    return int(report["model_total_bytes"])


def paired_delta_statistics(
    candidate: Mapping[int, float],
    baseline: Mapping[int, float],
    *,
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 20260803,
) -> dict[str, Any]:
    anchors = sorted(candidate)
    if anchors != sorted(baseline) or anchors != list(range(50)):
        raise Qwen3FormalAnalysisError(
            "paired statistics require the same complete anchor grid 0..49"
        )
    if bootstrap_resamples <= 0:
        raise Qwen3FormalAnalysisError("bootstrap_resamples must be positive")
    deltas = []
    for anchor in anchors:
        left = _finite_float(f"candidate[{anchor}]", candidate[anchor])
        right = _finite_float(f"baseline[{anchor}]", baseline[anchor])
        deltas.append(left - right)
    mean = math.fsum(deltas) / len(deltas)
    rng = random.Random(bootstrap_seed)
    bootstrap_means = []
    for _ in range(bootstrap_resamples):
        total = math.fsum(deltas[rng.randrange(len(deltas))] for _ in deltas)
        bootstrap_means.append(total / len(deltas))
    bootstrap_means.sort()
    favor = sum(value < 0 for value in deltas)
    tie = sum(value == 0 for value in deltas)
    oppose = len(deltas) - favor - tie
    return {
        "paired_anchor_count": len(deltas),
        "delta_definition": "candidate_mean_nll_minus_baseline_mean_nll",
        "negative_delta_favors_candidate": True,
        "mean_delta_nll": mean,
        "median_delta_nll": statistics.median(deltas),
        "population_standard_deviation": statistics.pstdev(deltas),
        "minimum_delta_nll": min(deltas),
        "maximum_delta_nll": max(deltas),
        "favor_count": favor,
        "tie_count": tie,
        "oppose_count": oppose,
        "exp_of_mean_delta": math.exp(mean),
        "bootstrap_95_interval": {
            "low": _type7_quantile(bootstrap_means, 0.025),
            "high": _type7_quantile(bootstrap_means, 0.975),
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "rng": "python_random.Random_reinitialized_per_contrast",
            "quantile": "type7_linear",
        },
        "anchor_deltas": [
            {"anchor_index": anchor, "delta_nll": delta}
            for anchor, delta in zip(anchors, deltas)
        ],
    }


def build_formal_analysis(
    *,
    protocol: dict[str, Any],
    memory_protocol: dict[str, Any],
    receipt: dict[str, Any],
    records: Sequence[dict[str, Any]],
    receipt_sha256: str,
) -> dict[str, Any]:
    validate_results_receipt(receipt)
    _require_sha256("receipt_sha256", receipt_sha256)
    expected_case_count = protocol["case_counts"]["total_unique_method_length_cases"]
    if len(records) != expected_case_count:
        raise Qwen3FormalAnalysisError(
            f"formal analysis requires {expected_case_count} records, got {len(records)}"
        )
    index: dict[tuple[str, int, int], dict[str, Any]] = {}
    methods: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        method = record.get("method")
        input_record = record.get("input")
        scoring = record.get("scoring")
        if not all(isinstance(value, dict) for value in (method, input_record, scoring)):
            raise Qwen3FormalAnalysisError("analysis record fields are missing")
        key = (
            method["id"],
            int(input_record["prompt_length"]),
            int(input_record["anchor_index"]),
        )
        if key in index:
            raise Qwen3FormalAnalysisError(f"duplicate formal result key {key!r}")
        index[key] = record
        point = key[:2]
        if point in methods and methods[point] != method:
            raise Qwen3FormalAnalysisError(f"method config drift at {point!r}")
        methods[point] = method

    point_records: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for (method_id, prompt_length, _), record in index.items():
        point_records[(method_id, prompt_length)].append(record)
    if any(len(values) != 50 for values in point_records.values()):
        raise Qwen3FormalAnalysisError("every method-length point must have 50 cases")

    base_points = _base_point_identities(protocol)
    expected_points = {
        (point["method_id"], point["prompt_length"])
        for partition in ("cage_qwen3", "kitty_qwen3")
        for point in formal_method_length_points(protocol, partition=partition)
    }
    if set(point_records) != expected_points:
        raise Qwen3FormalAnalysisError("formal result method-length grid mismatch")
    if len(expected_points) != 26:
        raise Qwen3FormalAnalysisError("formal result grid must contain 26 points")

    summaries = []
    anchor_means: dict[tuple[str, int], dict[int, float]] = {}
    for point in sorted(point_records, key=lambda value: (value[1], value[0])):
        method_id, prompt_length = point
        ordered = sorted(
            point_records[point], key=lambda record: record["input"]["anchor_index"]
        )
        anchors = [record["input"]["anchor_index"] for record in ordered]
        if anchors != list(range(50)):
            raise Qwen3FormalAnalysisError(f"incomplete anchor grid at {point!r}")
        token_nlls = []
        means: dict[int, float] = {}
        for record in ordered:
            values = record["scoring"].get("token_nlls")
            if not isinstance(values, list) or len(values) != 64:
                raise Qwen3FormalAnalysisError(f"invalid token NLL vector at {point!r}")
            normalized = [
                _finite_float(f"{point}.token_nlls", value) for value in values
            ]
            token_nlls.extend(normalized)
            anchor = int(record["input"]["anchor_index"])
            observed_mean = _finite_float(
                f"{point}.mean_nll", record["scoring"]["mean_nll"]
            )
            recalculated = math.fsum(normalized) / len(normalized)
            if not math.isclose(observed_mean, recalculated, rel_tol=1e-12, abs_tol=1e-12):
                raise Qwen3FormalAnalysisError(f"mean NLL drift at {point!r}")
            means[anchor] = observed_mean
        anchor_means[point] = means
        token_mean = math.fsum(token_nlls) / len(token_nlls)
        summary = {
            "method_id": method_id,
            "method_family": methods[point]["name"],
            "prompt_length": prompt_length,
            "anchor_count": 50,
            "target_count": len(token_nlls),
            "packed_bytes": packed_bytes_for_method(methods[point], prompt_length),
            "token_weighted_mean_nll": token_mean,
            "perplexity": math.exp(token_mean),
            "anchor_mean_nll_median": statistics.median(means.values()),
            "anchor_mean_nll_population_standard_deviation": statistics.pstdev(
                means.values()
            ),
            "anchor_mean_nll_minimum": min(means.values()),
            "anchor_mean_nll_maximum": max(means.values()),
            "is_base_point": point in base_points,
        }
        summaries.append(summary)

    summary_index = {
        (record["method_id"], record["prompt_length"]): record
        for record in summaries
    }
    for summary in summaries:
        point = (summary["method_id"], summary["prompt_length"])
        fp16_point = ("fp16", summary["prompt_length"])
        if point == fp16_point:
            summary["paired_delta_vs_fp16"] = None
        else:
            summary["paired_delta_vs_fp16"] = paired_delta_statistics(
                anchor_means[point],
                anchor_means[fp16_point],
                bootstrap_resamples=receipt["analysis_freeze"]["bootstrap_resamples"],
                bootstrap_seed=receipt["analysis_freeze"]["bootstrap_seed"],
            )

    _validate_memory_protocol(memory_protocol, methods, summary_index)
    primary = []
    for order, contrast in enumerate(protocol["primary_matched_comparisons"], 1):
        prompt_length = contrast["prompt_length"]
        candidate_point = (contrast["candidate"], prompt_length)
        baseline_point = (contrast["baseline"], prompt_length)
        stats = paired_delta_statistics(
            anchor_means[candidate_point],
            anchor_means[baseline_point],
            bootstrap_resamples=receipt["analysis_freeze"]["bootstrap_resamples"],
            bootstrap_seed=receipt["analysis_freeze"]["bootstrap_seed"],
        )
        candidate_bytes = summary_index[candidate_point]["packed_bytes"]
        baseline_bytes = summary_index[baseline_point]["packed_bytes"]
        primary.append(
            {
                "order": order,
                "comparison_id": (
                    f"l{prompt_length}-{contrast['candidate']}-minus-"
                    f"{contrast['baseline']}"
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
                "frozen_memory_selections": _matching_memory_selections(
                    memory_protocol,
                    prompt_length=prompt_length,
                    candidate=contrast["candidate"],
                    baseline=contrast["baseline"],
                ),
                "statistics": stats,
            }
        )

    mechanism = []
    for point in protocol["mechanism_points"]:
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
        for candidate, baseline, role in contrasts:
            candidate_point = (candidate, prompt_length)
            baseline_point = (baseline, prompt_length)
            candidate_bytes = summary_index[candidate_point]["packed_bytes"]
            baseline_bytes = summary_index[baseline_point]["packed_bytes"]
            if candidate_bytes != baseline_bytes:
                raise Qwen3FormalAnalysisError(
                    f"mechanism contrast {role} is not exactly byte matched"
                )
            mechanism.append(
                {
                    "comparison_id": f"l{prompt_length}-{role}",
                    "role": role,
                    "prompt_length": prompt_length,
                    "candidate": candidate,
                    "baseline": baseline,
                    "candidate_packed_bytes": candidate_bytes,
                    "baseline_packed_bytes": baseline_bytes,
                    "statistics": paired_delta_statistics(
                        anchor_means[candidate_point],
                        anchor_means[baseline_point],
                        bootstrap_resamples=receipt["analysis_freeze"][
                            "bootstrap_resamples"
                        ],
                        bootstrap_seed=receipt["analysis_freeze"]["bootstrap_seed"],
                    ),
                }
            )

    pareto = _pareto_tables(summaries)
    return {
        "schema_version": 1,
        "status": "pass",
        "analysis_id": "qwen3-8b-cage-kitty-formal-analysis-v1",
        "receipt_id": receipt["receipt_id"],
        "receipt_sha256": receipt_sha256,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": receipt["protocol"]["sha256"],
        "memory_protocol_id": memory_protocol["protocol_id"],
        "memory_protocol_sha256": receipt["memory_protocol"]["sha256"],
        "source_commit": receipt["source_commit"],
        "case_count": len(records),
        "target_count": sum(record["target_count"] for record in summaries),
        "method_length_point_count": len(summaries),
        "base_method_length_point_count": sum(
            record["is_base_point"] for record in summaries
        ),
        "analysis_freeze": receipt["analysis_freeze"],
        "method_summaries": summaries,
        "primary_matched_comparisons": primary,
        "mechanism_comparisons": mechanism,
        "base_pareto_by_prompt_length": pareto,
        "evidence": {
            name: {
                key: partition[key]
                for key in (
                    "case_count",
                    "case_ids_sha256",
                    "run_identity_sha256",
                    "summary_sha256",
                    "archive_sha256",
                    "execution_log_sha256",
                    "postrun_audit_sha256",
                    "postrun_audit_log_sha256",
                    "scientific_payload_sha256",
                )
            }
            for name, partition in receipt["partitions"].items()
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def _base_point_identities(protocol: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (record["method_id"], prompt_length)
        for record in protocol["base_method_length_points"]
        for prompt_length in record["prompt_lengths"]
    }


def _validate_memory_protocol(
    memory_protocol: dict[str, Any],
    methods: Mapping[tuple[str, int], dict[str, Any]],
    summaries: Mapping[tuple[str, int], dict[str, Any]],
) -> None:
    if memory_protocol.get("schema_version") != 1:
        raise Qwen3FormalAnalysisError("memory protocol schema mismatch")
    for selection in memory_protocol["matched_memory_matrix"]["selections"]:
        length = selection["prompt_length"]
        target_point = (selection["target_method"], length)
        cage_point = (selection["cage_method"], length)
        if target_point not in methods or cage_point not in methods:
            raise Qwen3FormalAnalysisError("frozen memory selection is not executable")
        if summaries[target_point]["packed_bytes"] != selection["target_bytes"]:
            raise Qwen3FormalAnalysisError("Kitty target byte total drifted")
        if summaries[cage_point]["packed_bytes"] != selection["cage_bytes"]:
            raise Qwen3FormalAnalysisError("CAGE selected byte total drifted")
        if selection["kivi_method"] is not None:
            kivi_point = (selection["kivi_method"], length)
            if summaries[kivi_point]["packed_bytes"] != selection["kivi_bytes"]:
                raise Qwen3FormalAnalysisError("KIVI selected byte total drifted")


def _matching_memory_selections(
    memory_protocol: dict[str, Any],
    *,
    prompt_length: int,
    candidate: str,
    baseline: str,
) -> list[dict[str, Any]]:
    matches = []
    for selection in memory_protocol["matched_memory_matrix"]["selections"]:
        if selection["prompt_length"] != prompt_length:
            continue
        methods = {
            selection["target_method"],
            selection["cage_method"],
            selection.get("kivi_method"),
        }
        if candidate in methods and baseline in methods:
            matches.append(
                {
                    "target_method": selection["target_method"],
                    "target_bytes": selection["target_bytes"],
                    "relative_budget_tolerance": memory_protocol[
                        "matched_memory_matrix"
                    ]["relative_budget_tolerance"],
                }
            )
    if not matches:
        raise Qwen3FormalAnalysisError(
            f"comparison {candidate} vs {baseline} lacks frozen memory evidence"
        )
    return matches


def _pareto_tables(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    tables = []
    for prompt_length in (1024, 2048, 4032):
        points = [
            record
            for record in summaries
            if record["prompt_length"] == prompt_length and record["is_base_point"]
        ]
        rows = []
        for point in points:
            dominated_by = []
            for other in points:
                if other is point:
                    continue
                no_worse = (
                    other["packed_bytes"] <= point["packed_bytes"]
                    and other["token_weighted_mean_nll"]
                    <= point["token_weighted_mean_nll"]
                )
                strictly_better = (
                    other["packed_bytes"] < point["packed_bytes"]
                    or other["token_weighted_mean_nll"]
                    < point["token_weighted_mean_nll"]
                )
                if no_worse and strictly_better:
                    dominated_by.append(other["method_id"])
            rows.append(
                {
                    "method_id": point["method_id"],
                    "packed_bytes": point["packed_bytes"],
                    "token_weighted_mean_nll": point["token_weighted_mean_nll"],
                    "perplexity": point["perplexity"],
                    "is_pareto": not dominated_by,
                    "dominated_by": sorted(dominated_by),
                }
            )
        tables.append(
            {
                "prompt_length": prompt_length,
                "point_count": len(rows),
                "pareto_method_ids": sorted(
                    row["method_id"] for row in rows if row["is_pareto"]
                ),
                "points": sorted(rows, key=lambda row: (row["packed_bytes"], row["method_id"])),
            }
        )
    return tables


def _type7_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise Qwen3FormalAnalysisError("quantile input must not be empty")
    if not 0 <= probability <= 1:
        raise Qwen3FormalAnalysisError("quantile probability must be in [0, 1]")
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
        raise Qwen3FormalAnalysisError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise Qwen3FormalAnalysisError(f"{name} must be finite")
    return numeric


def _positive_int(record: Mapping[str, Any], name: str) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Qwen3FormalAnalysisError(f"{name} must be a positive integer")
    return value


def _require_sha256(name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Qwen3FormalAnalysisError(f"{name} must be a lowercase SHA-256")


def _require_git_commit(name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Qwen3FormalAnalysisError(f"{name} must be a lowercase Git commit")


__all__ = [
    "QWEN3_FORMAL_RESULTS_RECEIPT_ID",
    "Qwen3FormalAnalysisError",
    "build_formal_analysis",
    "canonical_sha256",
    "packed_bytes_for_method",
    "paired_delta_statistics",
    "validate_results_receipt",
]
