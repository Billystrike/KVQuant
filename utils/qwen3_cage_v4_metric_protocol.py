from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v4_data import file_sha256


EXPECTED_PROTOCOL_ID = "qwen3-8b-cage-v4-pg19-metric-validity-v1"
COMPRESSED_METHODS = (
    "kivi-kittypro-matched",
    "cage-v1-kittypro-matched",
    "kitty-pro-25pct",
    "cage-v2-mixed-sink32-kittypro",
    "cage-v3-sr2-sink32-calibrated",
)
PROMPT_LENGTHS = (1024, 2048, 4032)
NEGLIGIBLE_RELATIVE_PPL_PERCENT = 0.5
MATERIAL_RELATIVE_PPL_PERCENT = 1.0
NONINFERIORITY_RELATIVE_PPL_PERCENT = 0.5


def load_metric_protocol(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_metric_protocol(payload)
    return payload, file_sha256(path)


def relative_ppl_percent(delta_mean_nll: float) -> float:
    if isinstance(delta_mean_nll, bool) or not isinstance(delta_mean_nll, (int, float)):
        raise ValueError("delta_mean_nll must be numeric")
    value = float(delta_mean_nll)
    if not math.isfinite(value):
        raise ValueError("delta_mean_nll must be finite")
    return 100.0 * math.expm1(value)


def practical_effect_label(delta_mean_nll: float) -> str:
    percent = relative_ppl_percent(delta_mean_nll)
    magnitude = abs(percent)
    if magnitude < NEGLIGIBLE_RELATIVE_PPL_PERCENT:
        return "practically_equivalent"
    direction = "favorable" if percent < 0 else "unfavorable"
    if magnitude < MATERIAL_RELATIVE_PPL_PERCENT:
        return f"small_{direction}_not_material"
    return f"material_{direction}"


def validate_metric_protocol(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("metric-validity protocol must be an object")
    if payload.get("schema_version") != 1 or payload.get("claim_eligible") is not False:
        raise ValueError("metric-validity protocol schema/boundary mismatch")
    if payload.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise ValueError("metric-validity protocol identity mismatch")
    inputs = payload.get("input_receipt")
    expected_inputs = {
        "data_protocol_sha256": "b5405e6bb9acf964047d0c7e033acdd90b6affac73c28805902de07df024b544",
        "input_manifest_sha256": "7c662aecd392f91f8d4a674af6af01496deba952820babd29045e3193dc5300d",
        "input_manifest_size_bytes": 5880826,
        "case_ids_sha256": "2077fbfffd6e36273dedf4bf69922a984a533d2a7add76eff0596ed052e42416",
        "manifest_build_log_sha256": "05a6b563a4073d16fc37bc49b9a60c26631c52d9296ee6bd0d851cfce690d84b",
        "documents": 50,
        "anchors": 100,
        "input_cases": 300,
    }
    if not isinstance(inputs, Mapping) or any(inputs.get(k) != v for k, v in expected_inputs.items()):
        raise ValueError("metric-validity input receipt mismatch")
    scoring = payload.get("primary_scoring")
    if not isinstance(scoring, Mapping):
        raise ValueError("metric-validity scoring record is missing")
    if scoring.get("metric") != "cache_conditioned_continuation_nll":
        raise ValueError("metric-validity primary metric mismatch")
    if scoring.get("target_tokens_per_case") != 64:
        raise ValueError("metric-validity target count mismatch")
    if scoring.get("statistical_unit") != "document":
        raise ValueError("metric-validity statistical unit must be document")
    effect = payload.get("practical_effect_policy")
    expected_effect = {
        "negligible_absolute_relative_ppl_percent_below": NEGLIGIBLE_RELATIVE_PPL_PERCENT,
        "material_absolute_relative_ppl_percent_at_least": MATERIAL_RELATIVE_PPL_PERCENT,
        "per_length_noninferiority_margin_relative_ppl_percent": NONINFERIORITY_RELATIVE_PPL_PERCENT,
    }
    if not isinstance(effect, Mapping) or any(effect.get(k) != v for k, v in expected_effect.items()):
        raise ValueError("metric-validity practical-effect thresholds mismatch")
    methods = payload.get("method_grid")
    if not isinstance(methods, list) or len(methods) != 6:
        raise ValueError("metric-validity method grid mismatch")
    method_ids = [method.get("method_id") for method in methods]
    if method_ids != ["fp16", *COMPRESSED_METHODS]:
        raise ValueError("metric-validity method order/identity mismatch")
    for method in methods:
        points = method.get("points")
        if not isinstance(points, list) or [row.get("prompt_length") for row in points] != list(PROMPT_LENGTHS):
            raise ValueError("metric-validity method length points mismatch")
        if any(type(row.get("packed_bytes")) is not int or row["packed_bytes"] <= 0 for row in points):
            raise ValueError("metric-validity packed bytes mismatch")
    execution = payload.get("execution_boundary")
    expected_execution = {
        "input_partition": "screen",
        "screen_documents": 20,
        "anchors_per_document": 2,
        "method_length_points": 18,
        "full_quality_cases": 720,
        "compressed_perturbation_cases": 600,
        "acceptance_repeats_required": 2,
        "acceptance_must_pass_before_full": True,
        "holdout_method_metrics_authorized": False,
        "cage_v4_candidate_execution_authorized": False,
        "full_gpu_execution_authorized": False,
    }
    if execution != expected_execution:
        raise ValueError("metric-validity execution boundary mismatch")
    validity = payload.get("local_proxy_validity_gate")
    if not isinstance(validity, Mapping):
        raise ValueError("local proxy validity gate is missing")
    if validity.get("minimum_spearman_each_length") != 0.8:
        raise ValueError("local proxy Spearman threshold mismatch")
    if validity.get("minimum_all_method_pair_concordance") != 0.8:
        raise ValueError("local proxy concordance threshold mismatch")
    if validity.get("minimum_frontier_pair_concordance") != 8 / 9:
        raise ValueError("local proxy frontier threshold mismatch")
    if payload.get("future_candidate_success_policy", {}).get("maximum_cage_v4_candidates") != 1:
        raise ValueError("metric-validity candidate limit mismatch")


__all__ = [
    "COMPRESSED_METHODS",
    "MATERIAL_RELATIVE_PPL_PERCENT",
    "NEGLIGIBLE_RELATIVE_PPL_PERCENT",
    "NONINFERIORITY_RELATIVE_PPL_PERCENT",
    "PROMPT_LENGTHS",
    "load_metric_protocol",
    "practical_effect_label",
    "relative_ppl_percent",
    "validate_metric_protocol",
]
