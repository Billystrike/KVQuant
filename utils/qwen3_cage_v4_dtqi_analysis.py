from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.qwen3_cage_v4_analysis import bootstrap_draws, paired_comparison
from utils.qwen3_cage_v4_data import file_sha256


LENGTHS = (1024, 2048, 4032)
CANDIDATE = "cage-v4-dtqi"
BASELINES = ("kitty-pro-25pct", "cage-v3-sr2-sink32-calibrated")
EXPECTED_RECEIPT_SHA256 = "3ef449c20279d9e34f2c36fdbef8c30a88f7afab193df635f4b25a831d0769fd"


class CageV4DTQIAnalysisError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV4DTQIAnalysisError(message)


def validate_results_receipt(receipt: Mapping[str, Any], *, receipt_path: Path) -> None:
    _require(file_sha256(receipt_path) == EXPECTED_RECEIPT_SHA256, "DTQI results receipt hash mismatch")
    _require(receipt.get("schema_version") == 1, "DTQI results receipt schema mismatch")
    _require(receipt.get("receipt_id") == "qwen3-8b-cage-v4-dtqi-pg19-screen-results-receipt-v1", "DTQI results receipt identity mismatch")
    _require(receipt.get("status") == "frozen_after_dtqi_screen_postrun_before_joint_interpretation", "DTQI results receipt status mismatch")
    _require(receipt.get("claim_eligible") is False and receipt.get("interpretation_performed") is False, "DTQI receipt interpretation boundary changed")
    _require(receipt["baseline_results"] == {
        "receipt_path": "configs/qwen3_8b_cage_v4_metric_screen_results_receipt_v1.json",
        "receipt_sha256": "fdd363b822d3a34e3c6e078410290d792a2ef608c7d48042c42e4d73737c4268",
        "methods_reused": list(BASELINES),
        "rerun": False,
    }, "DTQI baseline evidence changed")
    _require(receipt["dtqi_results"] == {
        "postrun_audit_path": "/root/autodl-tmp/kitty_setup_audit/qwen3_cage_v4_dtqi_screen_postrun_e4d493f.json",
        "postrun_audit_sha256": "a420ab6d6ea35c4d6d1c44ae0ff94d7328dac9aa4c58ba5d4918ee59d969aa7a",
        "postrun_audit_size_bytes": 2160,
        "postrun_log_path": "/root/autodl-tmp/kitty_setup_audit/qwen3_cage_v4_dtqi_screen_postrun_e4d493f_20260813T220650.log",
        "postrun_log_sha256": "c04540c0579c419f095d7e1769d2b98d389c5e5abf8319780f816adf26a70722",
        "postrun_log_size_bytes": 8964,
        "scientific_payload_sha256": "b0e89f7952055992dc1b1fe3e0af46d68800d61520b22a42b9f0ae782d57f2e7",
        "case_count": 120,
        "target_token_count": 7680,
        "document_count": 20,
        "anchor_count": 40,
    }, "DTQI result evidence changed")
    _require(receipt["analysis_freeze"] == {
        "candidate": CANDIDATE,
        "baselines": list(BASELINES),
        "statistical_unit": "PG-19 screen document",
        "anchors_clustered_within_document": True,
        "lengths_clustered_within_document_for_overall_effect": True,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 20260810,
        "bootstrap_interval": "linear-interpolated percentile 95 percent",
        "bootstrap_draws_reused_for_all_comparisons": True,
        "relative_ppl_percent": "100 * (exp(candidate_mean_nll_minus_baseline_mean_nll) - 1)",
        "overall_material_superiority_relative_ppl_percent_at_most": -1.0,
        "overall_bootstrap_upper_bound_below_zero": True,
        "per_length_noninferiority_relative_ppl_percent_at_most": 0.5,
        "all_two_baselines_must_pass": True,
        "local_mse_used": False,
        "report_unfavorable_results": True,
    }, "DTQI analysis freeze changed")
    _require(receipt["authorization"] == {
        "joint_screen_interpretation": True,
        "advance_only_if_preregistered_success_policy_passes": True,
        "holdout_metrics": False,
        "pg19_test_access": False,
        "runtime_claims": False,
        "paper_claims": False,
    }, "DTQI analysis authorization changed")
    _require(len(file_sha256(receipt_path)) == 64, "DTQI receipt hash invalid")


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "cannot average empty DTQI values")
    _require(all(math.isfinite(float(value)) for value in values), "non-finite DTQI value")
    return math.fsum(float(value) for value in values) / len(values)


def _index(records: Sequence[dict[str, Any]]) -> tuple[dict, dict]:
    nll = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    packed = defaultdict(set)
    anchors = defaultdict(set)
    for record in records:
        method = record["analysis_method_id"]
        _require(method in (CANDIDATE, *BASELINES), "unexpected DTQI analysis method")
        input_record = record["input"]
        length = int(input_record["prompt_length"])
        document = input_record["document_id"]
        anchor = int(input_record["anchor_index"])
        _require(length in LENGTHS, "unexpected DTQI analysis length")
        nll[method][length][document].append(float(record["scoring"]["mean_nll"]))
        packed[(method, length)].add(int(record["memory"]["model_total_bytes"]))
        anchors[(method, length, document)].add(anchor)
    for method in (CANDIDATE, *BASELINES):
        for length in LENGTHS:
            _require(len(nll[method][length]) == 20, "DTQI document count mismatch")
            _require(len(packed[(method, length)]) == 1, "DTQI packed bytes vary")
            for document, values in nll[method][length].items():
                _require(len(values) == 2 and anchors[(method, length, document)] == {0, 1}, "DTQI anchor grid mismatch")
    return nll, packed


def _document_means(nll: Mapping, method: str, length: int | None) -> dict[str, float]:
    if length is not None:
        return {document: _mean(values) for document, values in nll[method][length].items()}
    documents = sorted(nll[method][LENGTHS[0]])
    return {document: _mean([_mean(nll[method][length][document]) for length in LENGTHS]) for document in documents}


def build_analysis(*, receipt_sha256: str, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    _require(len(records) == 360, "DTQI joint analysis requires exactly 360 records")
    nll, packed = _index(records)
    draws = bootstrap_draws(20, resamples=10000, seed=20260810)
    summaries = []
    for method in (CANDIDATE, *BASELINES):
        for length in LENGTHS:
            values = _document_means(nll, method, length)
            mean_nll = _mean(list(values.values()))
            summaries.append({
                "method_id": method,
                "prompt_length": length,
                "case_count": 40,
                "document_count": 20,
                "target_token_count": 2560,
                "packed_bytes": next(iter(packed[(method, length)])),
                "mean_nll": mean_nll,
                "perplexity": math.exp(mean_nll),
            })
    comparisons = []
    decisions = {}
    for baseline in BASELINES:
        per_length = []
        for length in LENGTHS:
            report = paired_comparison(_document_means(nll, CANDIDATE, length), _document_means(nll, baseline, length), draws=draws)
            report = {"candidate_method": CANDIDATE, "baseline_method": baseline, "prompt_length": length, **report}
            report["noninferiority_margin_relative_ppl_percent"] = 0.5
            report["noninferiority_pass"] = report["relative_ppl_percent"] <= 0.5
            comparisons.append(report)
            per_length.append(report)
        overall = paired_comparison(_document_means(nll, CANDIDATE, None), _document_means(nll, baseline, None), draws=draws)
        overall = {"candidate_method": CANDIDATE, "baseline_method": baseline, "prompt_length": "overall_equal_length_weight", **overall}
        overall["material_superiority_threshold_relative_ppl_percent"] = -1.0
        overall["material_superiority_pass"] = overall["relative_ppl_percent"] <= -1.0
        overall["uncertainty_pass"] = overall["bootstrap_mean_nll_delta_ci95"][1] < 0
        comparisons.append(overall)
        decisions[baseline] = {
            "overall_material_superiority_pass": overall["material_superiority_pass"],
            "overall_uncertainty_pass": overall["uncertainty_pass"],
            "all_lengths_noninferiority_pass": all(row["noninferiority_pass"] for row in per_length),
        }
        decisions[baseline]["baseline_pass"] = all(decisions[baseline].values())
    advance = all(row["baseline_pass"] for row in decisions.values())
    return {
        "schema_version": 1,
        "analysis_id": "qwen3-8b-cage-v4-dtqi-pg19-screen-analysis-v1",
        "status": "pass",
        "claim_eligible": False,
        "receipt_sha256": receipt_sha256,
        "case_count": 360,
        "document_count": 20,
        "target_token_count": 23040,
        "method_length_summaries": summaries,
        "paired_comparisons": comparisons,
        "success_decision": {
            "per_baseline": decisions,
            "all_two_baselines_pass": advance,
            "candidate_outcome": "advance_to_holdout_preflight" if advance else "close_cage_v4_as_negative",
        },
        "bootstrap": {"unit": "document", "resamples": 10000, "seed": 20260810, "draws_reused": True},
        "boundaries": {
            "local_mse_used": False,
            "holdout_accessed": False,
            "pg19_test_accessed": False,
            "runtime_claims_made": False,
            "paper_claims_made": False,
        },
    }


__all__ = ["BASELINES", "CANDIDATE", "CageV4DTQIAnalysisError", "EXPECTED_RECEIPT_SHA256", "LENGTHS", "build_analysis", "validate_results_receipt"]
