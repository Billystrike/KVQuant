"""Versioned schemas and completed-point validation for CAGE experiments."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
METRIC_AGGREGATE_REL_TOL = 1e-12
METRIC_AGGREGATE_ABS_TOL = 1e-12

METRIC_NAMES = (
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

BYTE_FIELDS = (
    "key_payload_bytes",
    "value_payload_bytes",
    "key_scale_bytes",
    "value_scale_bytes",
    "key_min_or_zp_bytes",
    "value_min_or_zp_bytes",
    "bucket_index_bytes",
    "residual_full_precision_bytes",
    "payload_only_bytes",
    "metadata_bytes",
    "total_bytes",
)

RUN_FIELDS = frozenset({
    "schema_version",
    "run_id",
    "status",
    "model",
    "method",
    "input",
    "quantization",
    "measurement",
    "memory",
    "metrics_aggregate",
    "runtime_diagnostics",
    "provenance",
})

LAYER_FIELDS = frozenset({
    "schema_version",
    "run_id",
    "layer_index",
    "method",
    "prompt_length",
    "phase",
    "query_source",
    "memory",
    *METRIC_NAMES,
})


class ExperimentPointError(ValueError):
    """A deterministic experiment-point validation failure."""


def _require_exact_fields(name: str, value: dict[str, Any], expected: set[str] | frozenset[str]) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{name} missing required fields {missing}")
    if unknown:
        raise ValueError(f"{name} has unknown fields {unknown}")


def _require_object(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_int(name: str, value: Any, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{name} must be a {qualifier}integer")
    return value


def _require_finite_real(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    if nonnegative and number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _validate_byte_summary(name: str, summary: Any) -> dict[str, Any]:
    value = _require_object(name, summary)
    _require_exact_fields(name, value, {"cache_type", *BYTE_FIELDS})
    for field in BYTE_FIELDS:
        _require_int(f"{name}.{field}", value.get(field))
    if not isinstance(value.get("cache_type"), str) or not value["cache_type"]:
        raise ValueError(f"{name}.cache_type must be a non-empty string")
    expected_payload = value["key_payload_bytes"] + value["value_payload_bytes"]
    if value["payload_only_bytes"] != expected_payload:
        raise ValueError(
            f"invalid byte arithmetic for {name}.payload_only_bytes: "
            f"expected {expected_payload}"
        )
    expected_metadata = sum(value[field] for field in (
        "key_scale_bytes", "value_scale_bytes",
        "key_min_or_zp_bytes", "value_min_or_zp_bytes",
    ))
    if value["metadata_bytes"] != expected_metadata:
        raise ValueError(
            f"invalid byte arithmetic for {name}.metadata_bytes: expected {expected_metadata}"
        )
    expected_total = sum(value[field] for field in BYTE_FIELDS[:8])
    if value["total_bytes"] != expected_total:
        raise ValueError(
            f"invalid byte arithmetic for {name}.total_bytes: expected {expected_total}"
        )
    return value


def aggregate_layer_metrics(
    layer_records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, dict[str, float]]:
    """Return the schema-1 canonical mean/median/max layer aggregates."""

    result = {}
    for name in METRIC_NAMES:
        values = [
            float(record[name])
            for record in layer_records
            if isinstance(record.get(name), (int, float))
            and not isinstance(record.get(name), bool)
        ]
        if values:
            result[name] = {
                "mean": float(statistics.fmean(values)),
                "median": float(statistics.median(values)),
                "max": float(max(values)),
            }
    return result


def _validate_cuda_diagnostic(name: str, value: Any) -> None:
    diagnostic = _require_object(name, value)
    _require_exact_fields(
        name,
        diagnostic,
        {"max_allocated_bytes", "max_reserved_bytes"},
    )
    _require_int(f"{name}.max_allocated_bytes", diagnostic["max_allocated_bytes"])
    _require_int(f"{name}.max_reserved_bytes", diagnostic["max_reserved_bytes"])


def _validate_cache_side(name: str, side: Any, expected: tuple[int, int, int]) -> None:
    value = _require_object(name, side)
    _require_exact_fields(
        name,
        value,
        {"total_tokens", "quantized_history_tokens", "fp16_residual_tokens"},
    )
    actual = tuple(
        _require_int(f"{name}.{field}", value[field])
        for field in ("total_tokens", "quantized_history_tokens", "fp16_residual_tokens")
    )
    if actual != expected:
        raise ValueError(f"invalid cache structure {name}: expected {expected}, got {actual}")
    if actual[1] + actual[2] != actual[0]:
        raise ValueError(f"invalid cache structure {name}: quantized + residual must equal total")


def _validate_cache_structure(memory: dict[str, Any], run: dict[str, Any], row_number: int) -> None:
    cache = _require_object(f"layer row {row_number} cache structure", memory.get("cache_structure"))
    _require_exact_fields(f"layer row {row_number} cache structure", cache, {"key", "value"})
    prompt_length = run["input"]["prompt_length"]
    method = run["method"]["name"]
    if method == "fp16":
        key_expected = value_expected = (prompt_length, 0, prompt_length)
    elif method in {"kivi", "cage"}:
        residual_length = run["method"]["resolved_config"].get("residual_length")
        _require_int("method.resolved_config.residual_length", residual_length, positive=True)
        key_residual = prompt_length % residual_length
        value_residual = min(prompt_length, residual_length)
        key_expected = (prompt_length, prompt_length - key_residual, key_residual)
        value_expected = (prompt_length, prompt_length - value_residual, value_residual)
    else:
        raise ValueError(f"unsupported completed-point method {method!r}")
    _validate_cache_side(f"layer row {row_number} cache structure.key", cache["key"], key_expected)
    _validate_cache_side(f"layer row {row_number} cache structure.value", cache["value"], value_expected)


def _load_layer_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"matching layer JSONL is missing: {path}")
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"invalid layer JSONL {path}: blank line {line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid layer JSON in {path} line {line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(f"invalid layer JSONL {path}: row {line_number} must be an object")
                rows.append(row)
    except OSError as error:
        raise ValueError(f"cannot read layer JSONL {path}: {error}") from error
    if not rows:
        raise ValueError(f"layer JSONL must be non-empty: {path}")
    return rows


def _validate_run(path: Path, run_id: str, record: Any) -> dict[str, Any]:
    run = _require_object(f"completed run {path}", record)
    _require_exact_fields(f"completed run {path}", run, RUN_FIELDS)
    if type(run["schema_version"]) is not int or run["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"completed run {path} schema_version must be integer {SCHEMA_VERSION}")
    if run["run_id"] != run_id or run_id != path.stem:
        raise ValueError(f"completed run {path} run_id must match filename and requested id")
    if run["status"] != "completed":
        raise ValueError(f"run {path} is not completed")
    method = _require_object("run.method", run["method"])
    if not isinstance(method.get("name"), str) or not isinstance(method.get("resolved_config"), dict):
        raise ValueError("run.method must contain name and resolved_config")
    input_record = _require_object("run.input", run["input"])
    _require_int("run.input.prompt_length", input_record.get("prompt_length"), positive=True)
    measurement = _require_object("run.measurement", run["measurement"])
    _require_int("run.measurement.layer_count", measurement.get("layer_count"), positive=True)
    if not isinstance(measurement.get("phase"), str) or not isinstance(measurement.get("query_source"), str):
        raise ValueError("run.measurement must contain phase and query_source")
    memory = _require_object("run.memory", run["memory"])
    _require_exact_fields(
        "run.memory",
        memory,
        {"paper_estimate", "runtime_tensors", "cuda_peak_diagnostic"},
    )
    _validate_byte_summary("run.memory.paper_estimate", memory["paper_estimate"])
    _validate_byte_summary("run.memory.runtime_tensors", memory["runtime_tensors"])
    _validate_cuda_diagnostic("run.memory.cuda_peak_diagnostic", memory["cuda_peak_diagnostic"])
    aggregate = _require_object("run.metrics_aggregate", run["metrics_aggregate"])
    _require_exact_fields("run.metrics_aggregate", aggregate, set(METRIC_NAMES))
    for metric in METRIC_NAMES:
        statistics = _require_object(f"run.metrics_aggregate.{metric}", aggregate[metric])
        _require_exact_fields(
            f"run.metrics_aggregate.{metric}", statistics, {"mean", "median", "max"}
        )
        for statistic, value in statistics.items():
            number = _require_finite_real(
                f"run metric {metric}.{statistic}",
                value,
                nonnegative=metric != "topk_attention_overlap",
            )
            if metric == "topk_attention_overlap" and not 0 <= number <= 1:
                raise ValueError(
                    f"run metric topk_attention_overlap.{statistic} must be in [0, 1]"
                )
    return run


def _validate_layers(run: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    layer_count = run["measurement"]["layer_count"]
    if len(rows) != layer_count:
        raise ValueError(f"layer row count {len(rows)} does not equal measurement.layer_count {layer_count}")
    indices = []
    for row_number, row in enumerate(rows, 1):
        missing_metrics = sorted(set(METRIC_NAMES) - set(row))
        if missing_metrics:
            raise ValueError(f"layer row {row_number} missing metric fields {missing_metrics}")
        _require_exact_fields(f"layer row {row_number}", row, LAYER_FIELDS)
        if type(row["schema_version"]) is not int or row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"layer row {row_number} schema_version must be integer {SCHEMA_VERSION}")
        index = _require_int(f"layer row {row_number}.layer_index", row["layer_index"])
        indices.append(index)
        contexts = {
            "run_id": run["run_id"],
            "method": run["method"]["name"],
            "prompt_length": run["input"]["prompt_length"],
            "phase": run["measurement"]["phase"],
            "query_source": run["measurement"]["query_source"],
        }
        for field, expected in contexts.items():
            if row[field] != expected:
                raise ValueError(
                    f"layer row {row_number}.{field} must match run context {expected!r}"
                )
        memory = _require_object(f"layer row {row_number}.memory", row["memory"])
        _require_exact_fields(
            f"layer row {row_number}.memory",
            memory,
            {
                "paper_estimate",
                "runtime_tensors",
                "cuda_peak_diagnostic",
                "cache_structure",
            },
        )
        _validate_byte_summary(
            f"layer row {row_number}.memory.paper_estimate", memory["paper_estimate"]
        )
        _validate_byte_summary(
            f"layer row {row_number}.memory.runtime_tensors", memory["runtime_tensors"]
        )
        _validate_cuda_diagnostic(
            f"layer row {row_number}.memory.cuda_peak_diagnostic",
            memory["cuda_peak_diagnostic"],
        )
        _validate_cache_structure(memory, run, row_number)
        for metric in METRIC_NAMES:
            number = _require_finite_real(
                f"layer metric {metric} at row {row_number}",
                row[metric],
                nonnegative=metric != "topk_attention_overlap",
            )
            if metric == "topk_attention_overlap" and not 0 <= number <= 1:
                raise ValueError(
                    f"layer metric topk_attention_overlap at row {row_number} "
                    "must be in [0, 1]"
                )
    if sorted(indices) != list(range(layer_count)) or len(set(indices)) != layer_count:
        raise ValueError(f"layer indices must be unique and exactly contiguous 0..{layer_count - 1}")


def _validate_memory_sums(run: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    for namespace in ("paper_estimate", "runtime_tensors"):
        model_summary = run["memory"][namespace]
        for field in BYTE_FIELDS:
            expected = sum(row["memory"][namespace][field] for row in rows)
            if model_summary[field] != expected:
                raise ValueError(
                    f"run memory {namespace}.{field} does not equal recursive layer sum "
                    f"{expected}"
                )


def _validate_metric_aggregates(run: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    expected = aggregate_layer_metrics(rows)
    for metric in METRIC_NAMES:
        for statistic in ("mean", "median", "max"):
            actual_value = float(run["metrics_aggregate"][metric][statistic])
            expected_value = expected[metric][statistic]
            if not math.isclose(
                actual_value,
                expected_value,
                rel_tol=METRIC_AGGREGATE_REL_TOL,
                abs_tol=METRIC_AGGREGATE_ABS_TOL,
            ):
                raise ValueError(
                    f"run metric {metric}.{statistic} does not match layer aggregate "
                    f"{expected_value} within rel_tol={METRIC_AGGREGATE_REL_TOL} and "
                    f"abs_tol={METRIC_AGGREGATE_ABS_TOL}"
                )


def validate_completed_point(output_dir: str | Path, run_id: str) -> dict[str, Any]:
    """Validate and return one authoritative completed run and matching layer file."""

    root = Path(output_dir)
    run_path = root / "runs" / f"{run_id}.json"
    if not run_path.is_file():
        raise ValueError(f"completed run JSON is missing: {run_path}")
    try:
        record = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load completed run JSON {run_path}: {error}") from error
    run = _validate_run(run_path, run_id, record)
    rows = _load_layer_rows(root / "layers" / f"{run_id}.jsonl")
    _validate_layers(run, rows)
    _validate_metric_aggregates(run, rows)
    _validate_memory_sums(run, rows)
    return run


def validate_completed_artifacts(
    run_record: Any,
    layer_rows: list[dict[str, Any]],
    *,
    expected_run_id: str,
    run_path: Path | None = None,
) -> dict[str, Any]:
    """Validate in-memory artifacts before writing or after loading them."""

    path = run_path or Path(f"{expected_run_id}.json")
    run = _validate_run(path, expected_run_id, run_record)
    rows = layer_rows
    _validate_layers(run, rows)
    _validate_metric_aggregates(run, rows)
    _validate_memory_sums(run, rows)
    return run


def is_valid_completed_point(output_dir: str | Path, run_id: str) -> bool:
    try:
        validate_completed_point(output_dir, run_id)
    except ValueError:
        return False
    return True


__all__ = [
    "BYTE_FIELDS",
    "ExperimentPointError",
    "LAYER_FIELDS",
    "METRIC_NAMES",
    "RUN_FIELDS",
    "SCHEMA_VERSION",
    "aggregate_layer_metrics",
    "is_valid_completed_point",
    "validate_completed_artifacts",
    "validate_completed_point",
]
