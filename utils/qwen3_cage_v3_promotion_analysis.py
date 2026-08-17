from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.qwen3_cage_v3_promotion_protocol import METHOD_IDS, PROMPT_LENGTHS
from utils.qwen3_cage_v4_analysis import bootstrap_draws, paired_comparison
from utils.qwen3_cage_v4_data import file_sha256


EXPECTED_RECEIPT_SHA256 = "ea8a4a3fe5158e8fd20cdc79f4618464f90d90c09f805e6ffef231ef1bd29066"
CANDIDATE = "cage-v3-sr2-sink32-calibrated"
PREDECESSOR = "cage-v1-kittypro-matched"
UNIFORM_BASELINE = "kivi-kittypro-matched"
EXTERNAL_BASELINE = "kitty-pro-25pct"
QUALITY_REFERENCE = "fp16"
MAIN_BASELINES = (PREDECESSOR, UNIFORM_BASELINE, EXTERNAL_BASELINE)


class CageV3PromotionAnalysisError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3PromotionAnalysisError(message)


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "cannot average an empty sequence")
    converted = [float(value) for value in values]
    _require(all(math.isfinite(value) for value in converted), "non-finite analysis value")
    return math.fsum(converted) / len(converted)


def validate_results_receipt(receipt: Mapping[str, Any], *, receipt_path: Path) -> None:
    _require(file_sha256(receipt_path) == EXPECTED_RECEIPT_SHA256, "promotion results receipt hash mismatch")
    try:
        frozen = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV3PromotionAnalysisError(f"cannot load promotion results receipt: {error}") from error
    _require(receipt == frozen, "promotion results receipt differs from frozen file")
    _require(receipt.get("schema_version") == 1, "promotion results receipt schema mismatch")
    _require(receipt.get("receipt_id") == "qwen3-8b-cage-v3-promotion-results-receipt-v1", "promotion results receipt ID mismatch")
    _require(receipt.get("status") == "frozen_after_joint_full_postrun_before_promotion_interpretation", "promotion results receipt status mismatch")
    _require(receipt.get("claim_eligible") is False and receipt.get("interpretation_performed") is False, "promotion receipt interpretation boundary changed")
    summary = receipt.get("audit_summary", {})
    _require(
        (
            summary.get("status"),
            summary.get("case_count"),
            summary.get("target_token_count"),
            summary.get("failure_count"),
            summary.get("joint_scientific_payload_sha256"),
        )
        == (
            "pass",
            600,
            38_400,
            0,
            "acde4ad3b5e90051b86a5e52c5ab871edae0a477ff406bbf98bb79420eb1f262",
        ),
        "promotion receipt audit summary mismatch",
    )
    _require(summary.get("interpretation_performed") is False, "promotion metrics were already interpreted")
    for name in ("pg19_test_accessed", "llama2_execution_performed", "kitty_llama_port_performed"):
        _require(summary.get(name) is False, f"promotion receipt boundary changed: {name}")
    expected_partitions = {
        "cage_qwen3": (480, 30_720, "d6d1241855c8d40817f7ded16c328d9a738c5b3cd10c487a16e834fd53a561a1"),
        "kitty_qwen3": (120, 7_680, "d64c9ce08f77575071ef052f702947d9257913136d6277d3476031bb039fde20"),
    }
    _require(tuple(receipt.get("partitions", {})) == tuple(expected_partitions), "promotion receipt partitions changed")
    for partition, (case_count, token_count, payload_sha) in expected_partitions.items():
        row = receipt["partitions"][partition]
        _require(row.get("case_count") == case_count and row.get("target_token_count") == token_count, f"{partition} receipt totals mismatch")
        _require(row.get("failure_count") == 0 and row.get("scientific_payload_sha256") == payload_sha, f"{partition} receipt payload mismatch")
    freeze = receipt.get("analysis_freeze", {})
    _require(
        freeze
        == {
            "candidate": CANDIDATE,
            "predecessor": PREDECESSOR,
            "uniform_baseline": UNIFORM_BASELINE,
            "external_baseline": EXTERNAL_BASELINE,
            "quality_reference": QUALITY_REFERENCE,
            "statistical_unit": "PG-19 document",
            "anchors_clustered_within_document": True,
            "lengths_clustered_within_document_for_overall_effect": True,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20_260_816,
            "bootstrap_interval": "linear-interpolated paired document-cluster percentile 95 percent",
            "bootstrap_draws_reused_for_every_comparison": True,
            "per_length_noninferiority_relative_ppl_percent_at_most": 0.5,
            "overall_material_superiority_relative_ppl_percent_at_most": -1.0,
            "overall_noninferiority_relative_ppl_percent_at_most": 0.5,
            "report_all_methods_lengths_and_unfavorable_results": True,
            "decimal_direction_alone_is_never_a_pass": True,
        },
        "promotion analysis freeze changed",
    )
    _require(
        receipt.get("authorization")
        == {
            "promotion_interpretation": True,
            "promotion_decision": True,
            "additional_v3_tuning": False,
            "pg19_test_access": False,
            "llama2_v3_implementation_before_decision": False,
            "kitty_llama_port": False,
            "paper_claims": False,
            "runtime_claims": False,
        },
        "promotion analysis authorization changed",
    )


def _document_index(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    nll: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    anchors: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    bytes_by_point: dict[tuple[str, int], set[int]] = defaultdict(set)
    seen_cases: set[str] = set()
    for record in records:
        case_id = record["case_id"]
        _require(case_id not in seen_cases, "duplicate promotion case record")
        seen_cases.add(case_id)
        method = record["method"]["metric_method_id"]
        length = int(record["input"]["prompt_length"])
        document = record["input"]["document_id"]
        anchor = int(record["input"]["anchor_index"])
        _require(method in METHOD_IDS and length in PROMPT_LENGTHS, "unexpected promotion method/length")
        nll[method][length][document].append(float(record["scoring"]["mean_nll"]))
        anchors[(method, length, document)].add(anchor)
        bytes_by_point[(method, length)].add(int(record["memory"]["model_total_bytes"]))
    _require(len(seen_cases) == 600, "promotion analysis requires exactly 600 unique records")
    for method in METHOD_IDS:
        for length in PROMPT_LENGTHS:
            _require(len(nll[method][length]) == 20, f"{method}/{length} document count mismatch")
            _require(len(bytes_by_point[(method, length)]) == 1, f"{method}/{length} packed bytes vary")
            for document, values in nll[method][length].items():
                _require(len(values) == 2, f"{method}/{length}/{document} anchor count mismatch")
                _require(anchors[(method, length, document)] == {0, 1}, f"{method}/{length}/{document} anchor identity mismatch")
    return {"nll": nll, "bytes": bytes_by_point}


def _document_means(index: dict[str, Any], method: str, length: int | None) -> dict[str, float]:
    source = index["nll"]
    if length is not None:
        return {document: _mean(values) for document, values in source[method][length].items()}
    documents = sorted(source[method][PROMPT_LENGTHS[0]])
    return {
        document: _mean(
            [_mean(source[method][prompt_length][document]) for prompt_length in PROMPT_LENGTHS]
        )
        for document in documents
    }


def _protocol_bytes(protocol: Mapping[str, Any]) -> dict[tuple[str, int], int]:
    result = {}
    for method in protocol["method_grid"]:
        for point in method["points"]:
            result[(method["method_id"], int(point["prompt_length"]))] = int(point["packed_bytes"])
    return result


def build_promotion_analysis(
    *, receipt_sha256: str, protocol: Mapping[str, Any], records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    _require(len(records) == 600, "promotion analysis requires exactly 600 records")
    index = _document_index(records)
    frozen_bytes = _protocol_bytes(protocol)
    summaries = []
    for method in METHOD_IDS:
        for length in PROMPT_LENGTHS:
            observed_bytes = next(iter(index["bytes"][(method, length)]))
            _require(observed_bytes == frozen_bytes[(method, length)], f"{method}/{length} packed bytes mismatch")
            values = _document_means(index, method, length)
            mean_nll = _mean(list(values.values()))
            summaries.append(
                {
                    "method_id": method,
                    "prompt_length": length,
                    "case_count": 40,
                    "document_count": 20,
                    "target_token_count": 2_560,
                    "packed_bytes": observed_bytes,
                    "mean_nll": mean_nll,
                    "perplexity": math.exp(mean_nll),
                }
            )

    documents = sorted(_document_means(index, CANDIDATE, PROMPT_LENGTHS[0]))
    draws = bootstrap_draws(len(documents), resamples=10_000, seed=20_260_816)
    comparisons: dict[str, dict[str, Any]] = {}
    comparison_rows = []
    for baseline in (QUALITY_REFERENCE, *MAIN_BASELINES):
        by_length = {}
        for length in PROMPT_LENGTHS:
            report = paired_comparison(
                _document_means(index, CANDIDATE, length),
                _document_means(index, baseline, length),
                draws=draws,
            )
            by_length[str(length)] = report
            comparison_rows.append(
                {
                    "candidate_method": CANDIDATE,
                    "baseline_method": baseline,
                    "prompt_length": length,
                    **report,
                }
            )
        overall = paired_comparison(
            _document_means(index, CANDIDATE, None),
            _document_means(index, baseline, None),
            draws=draws,
        )
        comparison_rows.append(
            {
                "candidate_method": CANDIDATE,
                "baseline_method": baseline,
                "prompt_length": "overall_equal_length_weight",
                **overall,
            }
        )
        comparisons[baseline] = {"per_length": by_length, "overall": overall}

    noninferiority_limit = 0.5
    material_limit = -1.0
    ci_noninferiority_limit = math.log1p(0.005)

    def all_lengths_noninferior(baseline: str) -> bool:
        return all(
            comparisons[baseline]["per_length"][str(length)]["relative_ppl_percent"]
            <= noninferiority_limit
            for length in PROMPT_LENGTHS
        )

    candidate_total = sum(frozen_bytes[(CANDIDATE, length)] for length in PROMPT_LENGTHS)
    predecessor_total = sum(frozen_bytes[(PREDECESSOR, length)] for length in PROMPT_LENGTHS)
    memory_reduction = 1.0 - candidate_total / predecessor_total
    predecessor_memory_per_length_pass = all(
        frozen_bytes[(CANDIDATE, length)] / frozen_bytes[(PREDECESSOR, length)] - 1.0 <= 0.001
        for length in PROMPT_LENGTHS
    )
    kitty_memory_pass = all(
        frozen_bytes[(CANDIDATE, length)] <= frozen_bytes[(EXTERNAL_BASELINE, length)]
        for length in PROMPT_LENGTHS
    )

    predecessor_overall = comparisons[PREDECESSOR]["overall"]
    predecessor_lengths = all_lengths_noninferior(PREDECESSOR)
    predecessor_quality_track = (
        predecessor_overall["relative_ppl_percent"] <= material_limit
        and predecessor_overall["bootstrap_mean_nll_delta_ci95"][1] < 0.0
        and predecessor_lengths
    )
    predecessor_pareto_track = (
        memory_reduction >= 0.02
        and predecessor_memory_per_length_pass
        and predecessor_overall["relative_ppl_percent"] <= noninferiority_limit
        and predecessor_overall["bootstrap_mean_nll_delta_ci95"][1] < ci_noninferiority_limit
        and predecessor_lengths
    )
    predecessor_pass = predecessor_quality_track or predecessor_pareto_track

    uniform_overall = comparisons[UNIFORM_BASELINE]["overall"]
    uniform_lengths = all_lengths_noninferior(UNIFORM_BASELINE)
    uniform_pass = (
        uniform_overall["relative_ppl_percent"] <= material_limit
        and uniform_overall["bootstrap_mean_nll_delta_ci95"][1] < 0.0
        and uniform_lengths
    )

    external_overall = comparisons[EXTERNAL_BASELINE]["overall"]
    external_lengths = all_lengths_noninferior(EXTERNAL_BASELINE)
    external_pass = (
        kitty_memory_pass
        and external_overall["relative_ppl_percent"] <= noninferiority_limit
        and external_overall["bootstrap_mean_nll_delta_ci95"][1] < ci_noninferiority_limit
        and external_lengths
    )
    promotion_pass = predecessor_pass and uniform_pass and external_pass
    return {
        "schema_version": 1,
        "status": "pass",
        "analysis_id": "qwen3-8b-cage-v3-promotion-analysis-v1",
        "claim_eligible": False,
        "development_holdout_interpretation": True,
        "receipt_sha256": receipt_sha256,
        "case_count": 600,
        "document_count": 20,
        "target_token_count": 38_400,
        "prompt_lengths": list(PROMPT_LENGTHS),
        "method_length_summaries": summaries,
        "candidate_comparisons": comparison_rows,
        "comparison_lookup": comparisons,
        "memory": {
            "representation": "complete active logical packed paper estimate; not realized CUDA allocation",
            "candidate_total_bytes": candidate_total,
            "predecessor_total_bytes": predecessor_total,
            "candidate_relative_total_bytes_vs_predecessor": candidate_total / predecessor_total - 1.0,
            "candidate_memory_reduction_vs_predecessor": memory_reduction,
            "predecessor_pareto_minimum_reduction_pass": memory_reduction >= 0.02,
            "predecessor_per_length_overhead_pass": predecessor_memory_per_length_pass,
            "candidate_not_exceed_kitty_all_lengths": kitty_memory_pass,
        },
        "promotion_gates": {
            "versus_predecessor": {
                "quality_superiority_track_pass": predecessor_quality_track,
                "memory_quality_pareto_track_pass": predecessor_pareto_track,
                "all_lengths_noninferiority_pass": predecessor_lengths,
                "pass": predecessor_pass,
            },
            "versus_uniform_baseline": {
                "all_lengths_noninferiority_pass": uniform_lengths,
                "pass": uniform_pass,
            },
            "versus_external_baseline": {
                "all_lengths_noninferiority_pass": external_lengths,
                "memory_pass": kitty_memory_pass,
                "pass": external_pass,
            },
            "all_three_comparison_gates_pass": promotion_pass,
        },
        "promotion_pass": promotion_pass,
        "candidate_outcome": (
            "promote_cage_v3_to_architecture_normalized_main_method_candidate"
            if promotion_pass
            else "retain_original_cage_and_close_cage_v3_promotion"
        ),
        "bootstrap": {
            "unit": "PG-19 document",
            "resamples": 10_000,
            "seed": 20_260_816,
            "interval": "linear-interpolated paired document-cluster percentile 95 percent",
            "draws_reused_for_every_comparison": True,
        },
        "next_authorization": {
            "architecture_normalized_cage_v3_design_freeze": promotion_pass,
            "staged_llama2_cpu_gpu_acceptance": promotion_pass,
            "llama2_full_experiments": False,
            "kitty_llama_port": False,
            "additional_v3_tuning": False,
            "pg19_test_access": False,
            "paper_claims": False,
            "runtime_claims": False,
        },
        "reporting": {
            "all_methods_lengths_and_unfavorable_results_retained": True,
            "canonical_full_corpus_ppl_claim": False,
            "inference_scope": "frozen development-holdout promotion decision; not a final paper claim",
        },
    }


__all__ = [
    "CANDIDATE",
    "CageV3PromotionAnalysisError",
    "EXPECTED_RECEIPT_SHA256",
    "EXTERNAL_BASELINE",
    "MAIN_BASELINES",
    "PREDECESSOR",
    "QUALITY_REFERENCE",
    "UNIFORM_BASELINE",
    "build_promotion_analysis",
    "validate_results_receipt",
]
