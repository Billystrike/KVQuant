from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v4_data import file_sha256


PROTOCOL_ID = "qwen3-8b-cage-v3-pg19-promotion-v1"
METHOD_IDS = (
    "fp16",
    "kivi-kittypro-matched",
    "cage-v1-kittypro-matched",
    "cage-v3-sr2-sink32-calibrated",
    "kitty-pro-25pct",
)
PROMPT_LENGTHS = (1024, 2048, 4032)
PER_LENGTH_NONINFERIORITY_PERCENT = 0.5
MATERIAL_SUPERIORITY_PERCENT = -1.0
OVERALL_NONINFERIORITY_PERCENT = 0.5


class CageV3PromotionProtocolError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3PromotionProtocolError(message)


def load_promotion_protocol(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV3PromotionProtocolError(f"cannot load promotion protocol: {error}") from error
    validate_promotion_protocol(payload)
    return payload, file_sha256(path)


def validate_promotion_protocol(payload: Mapping[str, Any]) -> None:
    _require(isinstance(payload, Mapping), "promotion protocol must be an object")
    _require(payload.get("schema_version") == 1, "promotion protocol schema mismatch")
    _require(payload.get("protocol_id") == PROTOCOL_ID, "promotion protocol identity mismatch")
    _require(
        payload.get("status") == "frozen_before_any_pg19_holdout_method_metric",
        "promotion protocol freeze status mismatch",
    )
    _require(payload.get("claim_eligible") is False, "promotion protocol claim boundary mismatch")

    prior = payload.get("prior_boundary")
    _require(isinstance(prior, Mapping), "promotion prior boundary is missing")
    _require(prior.get("cage_v4_dtqi_outcome") == "close_cage_v4_as_negative", "DTQI closure changed")
    _require(prior.get("no_dtqi_reopening") is True, "DTQI reopening is forbidden")
    _require(prior.get("no_additional_tuning_on_metric_screen") is True, "screen tuning boundary changed")
    for key, sha, size in (
        ("closeout_json", "62207ff61aa4dbbd727751c3c84e8416b298444229a59f75828c2ace0003e7cb", 2645),
        ("closeout_log", "82f689febc4423ae782fc046a49d7b8b57e20a9d45415293764b764d7aecb395", 7529),
    ):
        receipt = prior.get(key)
        _require(isinstance(receipt, Mapping), f"{key} receipt is missing")
        _require(receipt.get("sha256") == sha and receipt.get("size_bytes") == size, f"{key} receipt changed")

    inputs = payload.get("input_receipt")
    _require(isinstance(inputs, Mapping), "promotion input receipt is missing")
    expected_inputs = {
        "data_protocol_sha256": "b5405e6bb9acf964047d0c7e033acdd90b6affac73c28805902de07df024b544",
        "input_manifest_sha256": "7c662aecd392f91f8d4a674af6af01496deba952820babd29045e3193dc5300d",
        "input_manifest_size_bytes": 5880826,
        "case_ids_sha256": "2077fbfffd6e36273dedf4bf69922a984a533d2a7add76eff0596ed052e42416",
        "partition": "holdout",
        "document_count": 20,
        "anchors_per_document": 2,
        "anchor_count": 40,
        "prompt_lengths": list(PROMPT_LENGTHS),
        "continuation_tokens": 64,
        "statistical_unit": "document",
    }
    _require(all(inputs.get(k) == v for k, v in expected_inputs.items()), "promotion input receipt changed")

    scoring = payload.get("scoring")
    _require(isinstance(scoring, Mapping), "promotion scoring policy is missing")
    _require(scoring.get("metric") == "cache_conditioned_continuation_nll", "primary metric changed")
    _require(scoring.get("target_tokens_per_case") == 64, "target count changed")
    _require(scoring.get("local_perturbation_used_for_selection") is False, "local proxy cannot select v3")
    _require(scoring.get("canonical_full_corpus_ppl_claim") is False, "canonical PPL claim is forbidden")

    methods = payload.get("method_grid")
    _require(isinstance(methods, list), "promotion method grid is missing")
    _require([row.get("method_id") for row in methods] == list(METHOD_IDS), "promotion method grid changed")
    for method in methods:
        points = method.get("points")
        _require(isinstance(points, list), "method points are missing")
        _require([point.get("prompt_length") for point in points] == list(PROMPT_LENGTHS), "prompt grid changed")
        _require(all(type(point.get("packed_bytes")) is int and point["packed_bytes"] > 0 for point in points), "packed bytes changed")
    _require([row.get("partition") for row in methods] == ["cage_qwen3"] * 4 + ["kitty_qwen3"], "execution partition changed")

    by_id = {row["method_id"]: row for row in methods}
    candidate = by_id["cage-v3-sr2-sink32-calibrated"]
    predecessor = by_id["cage-v1-kittypro-matched"]
    kitty = by_id["kitty-pro-25pct"]
    candidate_bytes = [point["packed_bytes"] for point in candidate["points"]]
    predecessor_bytes = [point["packed_bytes"] for point in predecessor["points"]]
    kitty_bytes = [point["packed_bytes"] for point in kitty["points"]]
    memory = payload.get("memory_gate")
    _require(isinstance(memory, Mapping), "promotion memory gate is missing")
    _require(sum(candidate_bytes) == memory.get("candidate_total_bytes") == 225967104, "candidate total bytes changed")
    _require(sum(predecessor_bytes) == memory.get("cage_v1_total_bytes") == 230812416, "predecessor total bytes changed")
    expected_delta = sum(candidate_bytes) / sum(predecessor_bytes) - 1.0
    _require(math.isclose(memory.get("candidate_relative_total_bytes_vs_cage_v1"), expected_delta), "memory delta changed")
    _require(memory.get("minimum_total_memory_reduction_vs_cage_v1_for_pareto_track") == 0.02, "Pareto memory threshold changed")
    _require(memory.get("maximum_per_length_relative_overhead_vs_cage_v1") == 0.001, "memory overhead tolerance changed")
    _require(all(c <= k for c, k in zip(candidate_bytes, kitty_bytes, strict=True)), "candidate exceeds Kitty-Pro bytes")
    _require(all(c / p - 1.0 <= 0.001 for c, p in zip(candidate_bytes, predecessor_bytes, strict=True)), "candidate exceeds v1 memory tolerance")

    gate = payload.get("promotion_gate")
    _require(isinstance(gate, Mapping), "promotion decision gate is missing")
    expected_gate = {
        "candidate": "cage-v3-sr2-sink32-calibrated",
        "predecessor": "cage-v1-kittypro-matched",
        "uniform_baseline": "kivi-kittypro-matched",
        "external_baseline": "kitty-pro-25pct",
        "per_length_noninferiority_relative_ppl_percent_at_most": PER_LENGTH_NONINFERIORITY_PERCENT,
        "overall_material_superiority_relative_ppl_percent_at_most": MATERIAL_SUPERIORITY_PERCENT,
        "overall_noninferiority_relative_ppl_percent_at_most": OVERALL_NONINFERIORITY_PERCENT,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 20260816,
        "all_three_comparison_gates_must_pass": True,
        "decimal_direction_alone_is_never_a_pass": True,
    }
    _require(all(gate.get(k) == v for k, v in expected_gate.items()), "promotion thresholds changed")
    _require(gate.get("versus_predecessor", {}).get("pass_rule") == "quality_superiority_track OR memory_quality_pareto_track", "predecessor pass rule changed")

    stages = payload.get("execution_stages")
    _require(isinstance(stages, Mapping), "promotion execution stages are missing")
    _require(stages.get("static_preflight") == {"reads_holdout_method_metrics": False, "authorizes_gpu": False}, "static preflight boundary changed")
    acceptance = stages.get("gpu_acceptance")
    _require(isinstance(acceptance, Mapping), "promotion acceptance stage is missing")
    _require(acceptance.get("document_id") == "pg19-validation-13-4bff8b8c6d7def9e", "acceptance document changed")
    _require(acceptance.get("anchor_index") == 0, "acceptance anchor changed")
    _require(acceptance.get("fresh_repeats_required") == 2, "acceptance repeat count changed")
    _require(acceptance.get("cage_partition_cases_per_repeat") == 12, "CAGE acceptance count changed")
    _require(acceptance.get("kitty_partition_cases_per_repeat") == 3, "Kitty acceptance count changed")
    full = stages.get("full_holdout")
    _require(full == {"authorized_only_after_joint_acceptance_gate": True, "cage_partition_case_count": 480, "kitty_partition_case_count": 120, "total_case_count": 600}, "full holdout counts changed")

    boundary = payload.get("decision_boundary")
    _require(isinstance(boundary, Mapping), "promotion decision boundary is missing")
    for key in (
        "pg19_test_access_authorized",
        "llama2_v3_implementation_authorized_before_promotion_pass",
        "kitty_llama_port_authorized_before_promotion_pass",
        "additional_v3_tuning_after_holdout",
    ):
        _require(boundary.get(key) is False, f"forbidden boundary changed: {key}")


__all__ = [
    "CageV3PromotionProtocolError",
    "MATERIAL_SUPERIORITY_PERCENT",
    "METHOD_IDS",
    "OVERALL_NONINFERIORITY_PERCENT",
    "PER_LENGTH_NONINFERIORITY_PERCENT",
    "PROMPT_LENGTHS",
    "PROTOCOL_ID",
    "load_promotion_protocol",
    "validate_promotion_protocol",
]
