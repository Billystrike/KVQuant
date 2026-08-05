from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


PERTURBATION_PROTOCOL_ID = "qwen3-8b-cage-kitty-memory-perturbation-v1"
LAYER_METRICS = (
    "relative_k_reconstruction_error",
    "attention_logit_mse",
    "attention_score_kl",
    "topk_attention_overlap",
    "weighted_key_error",
    "relative_v_reconstruction_error",
    "attention_output_mse",
    "post_o_proj_mse",
    "weighted_value_error",
    "joint_attention_output_mse",
    "joint_post_o_proj_mse",
    "joint_attention_output_relative_error",
)


class Qwen3PerturbationError(ValueError):
    pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_perturbation_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    try:
        protocol = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3PerturbationError(f"cannot load perturbation protocol {source}: {error}") from error
    validate_perturbation_protocol(protocol)
    return protocol, file_sha256(source)


def validate_perturbation_protocol(protocol: dict[str, Any]) -> None:
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise Qwen3PerturbationError("perturbation protocol schema mismatch")
    if protocol.get("protocol_id") != PERTURBATION_PROTOCOL_ID:
        raise Qwen3PerturbationError("perturbation protocol ID mismatch")
    if protocol.get("post_quality_design") is not True:
        raise Qwen3PerturbationError("the perturbation protocol must disclose its post-quality design")
    grid = protocol.get("grid", {})
    if grid.get("quality_selected_subset") is not False:
        raise Qwen3PerturbationError("the perturbation grid must not be quality-selected")
    if (
        grid.get("case_count") != 1300
        or grid.get("layer_count_per_case") != 36
        or grid.get("layer_record_count") != 46800
    ):
        raise Qwen3PerturbationError("the inherited full-grid counts are not frozen correctly")
    measurement = protocol.get("measurement", {})
    if measurement.get("continuation_token_index_used_as_decode_input") != 0:
        raise Qwen3PerturbationError("the teacher-forced measurement token must be continuation token 0")
    if measurement.get("candidate_query_excluded_from_primary_metrics") is not True:
        raise Qwen3PerturbationError("candidate queries must be excluded from primary local metrics")
    metrics = protocol.get("metrics", {})
    if tuple(metrics.get("layer_metrics", ())) != LAYER_METRICS:
        raise Qwen3PerturbationError("layer metric names or order differ from the frozen schema")
    if metrics.get("primary") != "joint_post_o_proj_mse":
        raise Qwen3PerturbationError("primary local perturbation metric mismatch")
    if tuple(metrics.get("run_aggregates", ())) != ("mean", "median", "maximum"):
        raise Qwen3PerturbationError("run aggregate definitions mismatch")
    partitions = protocol.get("partitions", {})
    if set(partitions) != {"cage_qwen3", "kitty_qwen3"}:
        raise Qwen3PerturbationError("perturbation execution partitions mismatch")
    if partitions["cage_qwen3"].get("full_cases") != 1000:
        raise Qwen3PerturbationError("CAGE partition case count mismatch")
    if partitions["kitty_qwen3"].get("full_cases") != 300:
        raise Qwen3PerturbationError("Kitty partition case count mismatch")


def perturbation_case_id(
    *, base_case_id: str, perturbation_protocol_sha256: str
) -> str:
    _require_sha256("perturbation_protocol_sha256", perturbation_protocol_sha256)
    payload = {
        "base_case_id": base_case_id,
        "perturbation_protocol_sha256": perturbation_protocol_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def aggregate_layer_metrics(
    layer_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    if len(layer_records) != 36:
        raise Qwen3PerturbationError("every Qwen3 case must contain exactly 36 layer records")
    indices = [record.get("layer_idx") for record in layer_records]
    if indices != list(range(36)):
        raise Qwen3PerturbationError("layer records must be ordered from layer 0 through 35")
    aggregates: dict[str, dict[str, float]] = {}
    for metric in LAYER_METRICS:
        values = [_finite_number(record.get("metrics", {}).get(metric), metric) for record in layer_records]
        aggregates[metric] = {
            "mean": math.fsum(values) / len(values),
            "median": statistics.median(values),
            "maximum": max(values),
        }
    return aggregates


def validate_layer_records(layer_records: Sequence[Mapping[str, Any]]) -> None:
    expected_metric_set = set(LAYER_METRICS)
    if len(layer_records) != 36:
        raise Qwen3PerturbationError("layer record count mismatch")
    for expected_idx, record in enumerate(layer_records):
        if record.get("layer_idx") != expected_idx:
            raise Qwen3PerturbationError("layer record ordering mismatch")
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != expected_metric_set:
            raise Qwen3PerturbationError("layer metric schema mismatch")
        for name, value in metrics.items():
            numeric = _finite_number(value, name)
            if numeric < 0:
                raise Qwen3PerturbationError(f"layer metric {name} must be nonnegative")
        overlap = float(metrics["topk_attention_overlap"])
        if overlap > 1:
            raise Qwen3PerturbationError("top-k attention overlap must not exceed one")


def validate_aggregates(
    aggregates: Mapping[str, Any], layer_records: Sequence[Mapping[str, Any]]
) -> None:
    expected = aggregate_layer_metrics(layer_records)
    if set(aggregates) != set(expected):
        raise Qwen3PerturbationError("aggregate metric schema mismatch")
    for metric, expected_stats in expected.items():
        actual_stats = aggregates.get(metric)
        if not isinstance(actual_stats, Mapping) or set(actual_stats) != set(expected_stats):
            raise Qwen3PerturbationError(f"aggregate schema mismatch for {metric}")
        for statistic, expected_value in expected_stats.items():
            actual = _finite_number(actual_stats.get(statistic), f"{metric}.{statistic}")
            if not math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12):
                raise Qwen3PerturbationError(f"aggregate {metric}.{statistic} is inconsistent")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Qwen3PerturbationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise Qwen3PerturbationError(f"{name} must be finite")
    return number


def _require_sha256(name: str, value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise Qwen3PerturbationError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise Qwen3PerturbationError(f"{name} must be a SHA-256 hex digest") from error


__all__ = [
    "LAYER_METRICS",
    "PERTURBATION_PROTOCOL_ID",
    "Qwen3PerturbationError",
    "aggregate_layer_metrics",
    "file_sha256",
    "load_perturbation_protocol",
    "perturbation_case_id",
    "validate_aggregates",
    "validate_layer_records",
    "validate_perturbation_protocol",
]
