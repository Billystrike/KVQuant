from __future__ import annotations

import itertools
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.qwen3_cage_v4_data import file_sha256


EXPECTED_RECEIPT_SHA256 = "fdd363b822d3a34e3c6e078410290d792a2ef608c7d48042c42e4d73737c4268"
LENGTHS = (1024, 2048, 4032)
ALL_METHODS = (
    "fp16",
    "kivi-kittypro-matched",
    "cage-v1-kittypro-matched",
    "kitty-pro-25pct",
    "cage-v2-mixed-sink32-kittypro",
    "cage-v3-sr2-sink32-calibrated",
)
COMPRESSED_METHODS = ALL_METHODS[1:]
FRONTIER_METHODS = (
    "kitty-pro-25pct",
    "cage-v2-mixed-sink32-kittypro",
    "cage-v3-sr2-sink32-calibrated",
)


class CageV4AnalysisError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV4AnalysisError(message)


def validate_results_receipt(receipt: dict[str, Any], *, receipt_path: Path) -> None:
    _require(file_sha256(receipt_path) == EXPECTED_RECEIPT_SHA256, "results receipt hash mismatch")
    _require(receipt.get("schema_version") == 1, "results receipt schema mismatch")
    _require(
        receipt.get("receipt_id")
        == "qwen3-8b-cage-v4-pg19-metric-screen-results-receipt-v1",
        "results receipt ID mismatch",
    )
    _require(
        receipt.get("status")
        == "frozen_after_joint_postrun_audit_before_metric_screen_interpretation",
        "results receipt status mismatch",
    )
    _require(receipt.get("claim_eligible") is False, "results receipt became claim-eligible")
    _require(receipt.get("interpretation_performed") is False, "results already interpreted")
    summary = receipt.get("audit_summary", {})
    _require(
        (
            summary.get("status"),
            summary.get("case_count"),
            summary.get("target_token_count"),
            summary.get("layer_record_count"),
            summary.get("failure_count"),
        )
        == ("pass", 720, 46_080, 21_600, 0),
        "results receipt audit totals mismatch",
    )
    for name in ("holdout_accessed", "cage_v4_candidate_executed", "interpretation_performed"):
        _require(summary.get(name) is False, f"results receipt boundary mismatch: {name}")
    freeze = receipt.get("analysis_freeze", {})
    _require(
        freeze
        == {
            "statistical_unit": "PG-19 document",
            "anchors_clustered_within_document": True,
            "lengths_clustered_within_document_for_overall_effect": True,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20_260_810,
            "bootstrap_interval": "linear-interpolated percentile 95 percent",
            "bootstrap_draws_reused_for_every_comparison": True,
            "relative_ppl_percent": "100 * (exp(candidate_mean_nll_minus_baseline_mean_nll) - 1)",
            "practical_thresholds_percent": {
                "negligible_absolute_below": 0.5,
                "material_absolute_at_least": 1.0,
                "between": "small_effect_cannot_alone_advance_candidate",
            },
            "rank_ties": "average ranks",
            "directional_ties": "concordant only when both axes tie; a tie on exactly one axis is discordant",
            "local_proxy_gate": {
                "minimum_spearman_each_length": 0.8,
                "minimum_all_method_pair_concordance": 0.8,
                "minimum_frontier_pair_concordance": 0.8888888888888888,
                "frontier_pair_count": 9,
            },
            "report_all_methods_lengths_and_pairwise_comparisons": True,
            "candidate_selection_performed": False,
        },
        "analysis freeze changed",
    )
    authorization = receipt.get("authorization", {})
    _require(
        authorization
        == {
            "metric_screen_interpretation": True,
            "local_proxy_role_decision": True,
            "cage_v4_design_after_analysis": True,
            "maximum_future_cage_v4_candidates": 1,
            "cage_v4_candidate_execution": False,
            "holdout_metrics": False,
            "pg19_test_access": False,
            "runtime_claims": False,
        },
        "authorization boundary changed",
    )
    expected_partitions = {
        "cage_qwen3": (600, 17_280, "d3fab5cf92b314f132dcc8b1e002e784dd7f67797e2a9997d4e7cd19d2dbe170"),
        "kitty_qwen3": (120, 4_320, "400f8cdfd3e15dbf76887ae7c445d99b1f5a227b7fa202ba2534f847e4c6e39f"),
    }
    _require(tuple(receipt.get("partitions", {})) == tuple(expected_partitions), "receipt partitions changed")
    for partition, (case_count, layer_count, payload_sha) in expected_partitions.items():
        record = receipt["partitions"][partition]
        _require(record.get("case_count") == case_count, f"{partition} receipt count changed")
        _require(record.get("layer_record_count") == layer_count, f"{partition} receipt layers changed")
        _require(record.get("scientific_payload_sha256") == payload_sha, f"{partition} receipt payload changed")


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "cannot average an empty sequence")
    _require(all(math.isfinite(float(value)) for value in values), "non-finite analysis value")
    return math.fsum(float(value) for value in values) / len(values)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    _require(len(first) == len(second) and len(first) >= 2, "Spearman inputs mismatch")
    x = _average_ranks([float(value) for value in first])
    y = _average_ranks([float(value) for value in second])
    x_mean = _mean(x)
    y_mean = _mean(y)
    numerator = math.fsum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_scale = math.fsum((a - x_mean) ** 2 for a in x)
    y_scale = math.fsum((b - y_mean) ** 2 for b in y)
    _require(x_scale > 0 and y_scale > 0, "Spearman axis is constant")
    return numerator / math.sqrt(x_scale * y_scale)


def directional_concordance(
    truth: Mapping[str, float], proxy: Mapping[str, float], methods: Sequence[str]
) -> dict[str, Any]:
    pairs = []
    concordant = 0
    for first, second in itertools.combinations(methods, 2):
        truth_delta = float(truth[first]) - float(truth[second])
        proxy_delta = float(proxy[first]) - float(proxy[second])
        passed = (truth_delta == 0 and proxy_delta == 0) or truth_delta * proxy_delta > 0
        concordant += int(passed)
        pairs.append(
            {
                "first_method": first,
                "second_method": second,
                "truth_delta": truth_delta,
                "proxy_delta": proxy_delta,
                "concordant": passed,
            }
        )
    return {
        "pair_count": len(pairs),
        "concordant_count": concordant,
        "discordant_count": len(pairs) - concordant,
        "concordance": concordant / len(pairs),
        "pairs": pairs,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(ordered and 0 <= probability <= 1, "invalid percentile request")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_draws(document_count: int, *, resamples: int, seed: int) -> list[list[int]]:
    _require(document_count > 0 and resamples > 0, "invalid bootstrap shape")
    generator = random.Random(seed)
    return [
        [generator.randrange(document_count) for _ in range(document_count)]
        for _ in range(resamples)
    ]


def practical_effect(relative_ppl_percent: float) -> dict[str, str]:
    magnitude = abs(relative_ppl_percent)
    if magnitude < 0.5:
        label = "practically_equivalent"
    elif magnitude < 1.0:
        label = "small_effect_cannot_alone_advance_candidate"
    else:
        label = "material_effect"
    direction = (
        "candidate_better"
        if relative_ppl_percent < 0
        else "candidate_worse"
        if relative_ppl_percent > 0
        else "exact_tie"
    )
    return {"magnitude_label": label, "direction": direction}


def paired_comparison(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    draws: Sequence[Sequence[int]],
) -> dict[str, Any]:
    documents = sorted(candidate)
    _require(documents and documents == sorted(baseline), "paired document grid mismatch")
    deltas = [float(candidate[document]) - float(baseline[document]) for document in documents]
    mean_delta = _mean(deltas)
    bootstrap = [_mean([deltas[index] for index in draw]) for draw in draws]
    relative = 100.0 * math.expm1(mean_delta)
    favor = sum(delta < 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    lower = _percentile(bootstrap, 0.025)
    upper = _percentile(bootstrap, 0.975)
    return {
        "paired_document_count": len(documents),
        "mean_nll_delta": mean_delta,
        "median_document_nll_delta": statistics.median(deltas),
        "minimum_document_nll_delta": min(deltas),
        "maximum_document_nll_delta": max(deltas),
        "candidate_favor_count": favor,
        "tie_count": ties,
        "baseline_favor_count": len(deltas) - favor - ties,
        "relative_ppl_percent": relative,
        "practical_effect": practical_effect(relative),
        "bootstrap_resamples": len(draws),
        "bootstrap_mean_nll_delta_ci95": [lower, upper],
        "bootstrap_favorable": upper < 0,
        "bootstrap_unfavorable": lower > 0,
        "negative_delta_favors_candidate": True,
    }


def _document_index(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    nll: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    local: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    bytes_by_point: dict[tuple[str, int], set[int]] = defaultdict(set)
    anchors: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for record in records:
        method = record["method"]["metric_method_id"]
        length = int(record["input"]["prompt_length"])
        document = record["input"]["document_id"]
        anchor = int(record["input"]["anchor_index"])
        _require(method in ALL_METHODS and length in LENGTHS, "unexpected analysis method/length")
        nll[method][length][document].append(float(record["scoring"]["mean_nll"]))
        anchors[(method, length, document)].add(anchor)
        bytes_by_point[(method, length)].add(int(record["memory"]["model_total_bytes"]))
        if method != "fp16":
            local[method][length][document].append(
                float(record["local_perturbation"]["aggregates"]["joint_post_o_proj_mse"]["mean"])
            )
    for method in ALL_METHODS:
        for length in LENGTHS:
            _require(len(nll[method][length]) == 20, f"{method}/{length} document count mismatch")
            _require(len(bytes_by_point[(method, length)]) == 1, "packed bytes vary within point")
            for document, values in nll[method][length].items():
                _require(len(values) == 2, f"{method}/{length}/{document} anchor count mismatch")
                _require(anchors[(method, length, document)] == {0, 1}, "anchor identity mismatch")
                if method != "fp16":
                    _require(len(local[method][length][document]) == 2, "local anchor count mismatch")
    return {"nll": nll, "local": local, "bytes": bytes_by_point}


def _document_means(
    index: dict[str, Any], method: str, *, length: int | None, metric: str
) -> dict[str, float]:
    source = index[metric]
    if length is not None:
        return {document: _mean(values) for document, values in source[method][length].items()}
    documents = sorted(source[method][LENGTHS[0]])
    return {
        document: _mean(
            [_mean(source[method][prompt_length][document]) for prompt_length in LENGTHS]
        )
        for document in documents
    }


def build_metric_screen_analysis(
    *, receipt_sha256: str, records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    _require(len(records) == 720, "metric screen analysis requires exactly 720 records")
    index = _document_index(records)
    documents = sorted(index["nll"]["fp16"][LENGTHS[0]])
    draws = bootstrap_draws(len(documents), resamples=10_000, seed=20_260_810)
    summaries = []
    summary_lookup = {}
    for method in ALL_METHODS:
        for length in LENGTHS:
            document_nll = _document_means(index, method, length=length, metric="nll")
            local_mse = None
            if method != "fp16":
                local_mse = _mean(
                    list(_document_means(index, method, length=length, metric="local").values())
                )
            row = {
                "method_id": method,
                "prompt_length": length,
                "case_count": 40,
                "document_count": 20,
                "target_token_count": 2_560,
                "packed_bytes": next(iter(index["bytes"][(method, length)])),
                "mean_nll": _mean(list(document_nll.values())),
                "perplexity": math.exp(_mean(list(document_nll.values()))),
                "mean_joint_post_o_proj_mse": local_mse,
            }
            summaries.append(row)
            summary_lookup[(method, length)] = row

    comparisons = []
    for length in (*LENGTHS, None):
        length_label = "overall_equal_length_weight" if length is None else length
        fp16 = _document_means(index, "fp16", length=length, metric="nll")
        for method in COMPRESSED_METHODS:
            report = paired_comparison(
                _document_means(index, method, length=length, metric="nll"), fp16, draws=draws
            )
            comparisons.append(
                {
                    "candidate_method": method,
                    "baseline_method": "fp16",
                    "prompt_length": length_label,
                    **report,
                }
            )
        for candidate, baseline in itertools.combinations(COMPRESSED_METHODS, 2):
            report = paired_comparison(
                _document_means(index, candidate, length=length, metric="nll"),
                _document_means(index, baseline, length=length, metric="nll"),
                draws=draws,
            )
            comparisons.append(
                {
                    "candidate_method": candidate,
                    "baseline_method": baseline,
                    "prompt_length": length_label,
                    **report,
                }
            )

    per_length_proxy = []
    all_pair_rows = []
    frontier_pair_rows = []
    for length in LENGTHS:
        fp16_mean = summary_lookup[("fp16", length)]["mean_nll"]
        truth = {
            method: summary_lookup[(method, length)]["mean_nll"] - fp16_mean
            for method in COMPRESSED_METHODS
        }
        proxy = {
            method: summary_lookup[(method, length)]["mean_joint_post_o_proj_mse"]
            for method in COMPRESSED_METHODS
        }
        spearman = spearman_correlation(
            [truth[method] for method in COMPRESSED_METHODS],
            [proxy[method] for method in COMPRESSED_METHODS],
        )
        all_pairs = directional_concordance(truth, proxy, COMPRESSED_METHODS)
        frontier = directional_concordance(truth, proxy, FRONTIER_METHODS)
        all_pair_rows.extend({"prompt_length": length, **row} for row in all_pairs["pairs"])
        frontier_pair_rows.extend(
            {"prompt_length": length, **row} for row in frontier["pairs"]
        )
        per_length_proxy.append(
            {
                "prompt_length": length,
                "truth_nll_degradation_vs_fp16": truth,
                "proxy_mean_joint_post_o_proj_mse": proxy,
                "spearman": spearman,
                "spearman_pass": spearman >= 0.8,
                "all_method_pairs": {key: value for key, value in all_pairs.items() if key != "pairs"},
                "frontier_pairs": {key: value for key, value in frontier.items() if key != "pairs"},
            }
        )
    all_concordant = sum(row["concordant"] for row in all_pair_rows)
    frontier_concordant = sum(row["concordant"] for row in frontier_pair_rows)
    all_concordance = all_concordant / len(all_pair_rows)
    frontier_concordance = frontier_concordant / len(frontier_pair_rows)
    gate_checks = {
        "spearman_each_length_at_least_0_8": all(row["spearman_pass"] for row in per_length_proxy),
        "all_method_pair_concordance_at_least_0_8": all_concordance >= 0.8,
        "frontier_pair_concordance_at_least_8_of_9": frontier_concordant >= 8,
    }
    gate_pass = all(gate_checks.values())
    return {
        "schema_version": 1,
        "status": "pass",
        "analysis_id": "qwen3-8b-cage-v4-pg19-metric-screen-analysis-v1",
        "claim_eligible": False,
        "receipt_sha256": receipt_sha256,
        "case_count": 720,
        "document_count": 20,
        "target_token_count": 46_080,
        "prompt_lengths": list(LENGTHS),
        "method_length_summaries": summaries,
        "paired_nll_comparisons": comparisons,
        "local_proxy_validity": {
            "per_length": per_length_proxy,
            "all_method_pair_count": len(all_pair_rows),
            "all_method_pair_concordant_count": all_concordant,
            "all_method_pair_concordance": all_concordance,
            "frontier_pair_count": len(frontier_pair_rows),
            "frontier_pair_concordant_count": frontier_concordant,
            "frontier_pair_concordance": frontier_concordance,
            "gate_checks": gate_checks,
            "gate_pass": gate_pass,
            "resulting_role": (
                "secondary_fine_ranking_proxy_never_overrides_nll"
                if gate_pass
                else "coarse_diagnostics_only_future_cage_v4_screen_selection_uses_nll"
            ),
        },
        "bootstrap": {
            "unit": "document",
            "resamples": 10_000,
            "seed": 20_260_810,
            "interval": "linear-interpolated percentile 95 percent",
            "draws_reused_for_every_comparison": True,
        },
        "boundaries": {
            "candidate_selection_performed": False,
            "cage_v4_candidate_executed": False,
            "holdout_accessed": False,
            "pg19_test_accessed": False,
            "runtime_claims_made": False,
        },
    }


__all__ = [
    "ALL_METHODS",
    "COMPRESSED_METHODS",
    "CageV4AnalysisError",
    "EXPECTED_RECEIPT_SHA256",
    "FRONTIER_METHODS",
    "LENGTHS",
    "bootstrap_draws",
    "build_metric_screen_analysis",
    "directional_concordance",
    "paired_comparison",
    "practical_effect",
    "spearman_correlation",
    "validate_results_receipt",
]
