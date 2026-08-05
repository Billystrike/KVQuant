from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


PERTURBATION_PROTOCOL_ID = "qwen3-8b-cage-kitty-memory-perturbation-v1"
PERTURBATION_ACCEPTANCE_GATE_ID = (
    "qwen3-8b-cage-kitty-memory-perturbation-acceptance-gate-v1"
)
SCIENTIFIC_PAYLOAD_FIELDS = (
    "case_id",
    "base_quality_case_id",
    "partition",
    "stage",
    "method",
    "input",
    "memory",
    "measurement",
    "layer_metrics",
    "aggregates",
)
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


def load_perturbation_acceptance_gate(
    path: str | Path,
    *,
    perturbation_protocol_sha256: str,
    verify_artifacts: bool = False,
) -> tuple[dict[str, Any], str]:
    source = Path(path)
    try:
        gate = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3PerturbationError(f"cannot load perturbation acceptance gate {source}: {error}") from error
    validate_perturbation_acceptance_gate(
        gate,
        perturbation_protocol_sha256=perturbation_protocol_sha256,
    )
    if verify_artifacts:
        verify_perturbation_acceptance_artifacts(gate)
    return gate, file_sha256(source)


def validate_perturbation_acceptance_gate(
    gate: dict[str, Any], *, perturbation_protocol_sha256: str
) -> None:
    _require_sha256("perturbation_protocol_sha256", perturbation_protocol_sha256)
    if not isinstance(gate, dict) or gate.get("schema_version") != 1:
        raise Qwen3PerturbationError("perturbation acceptance gate schema mismatch")
    if gate.get("gate_id") != PERTURBATION_ACCEPTANCE_GATE_ID or gate.get("status") != "pass":
        raise Qwen3PerturbationError("perturbation acceptance gate identity/status mismatch")
    protocol = gate.get("perturbation_protocol", {})
    if protocol.get("protocol_id") != PERTURBATION_PROTOCOL_ID:
        raise Qwen3PerturbationError("acceptance gate protocol ID mismatch")
    if protocol.get("sha256") != perturbation_protocol_sha256:
        raise Qwen3PerturbationError("acceptance gate protocol hash mismatch")
    source = gate.get("acceptance_source", {})
    for name in (
        "runner_sha256",
        "recorder_sha256",
        "qwen3_cage_sha256",
    ):
        _require_sha256(f"acceptance_source.{name}", source.get(name))
    if source.get("kitty_commit") != "dfd2c07b407d6b407179359207c612ab631f3ed1":
        raise Qwen3PerturbationError("acceptance gate Kitty commit mismatch")
    if source.get("transformers_commit") != "37f8b0b53512e6aae0cfd15746c133c101783178":
        raise Qwen3PerturbationError("acceptance gate Transformers commit mismatch")
    if source.get("omp_num_threads_observed") != "0":
        raise Qwen3PerturbationError("acceptance gate OMP observation mismatch")

    partitions = gate.get("partitions", {})
    if set(partitions) != {"cage_qwen3", "kitty_qwen3"}:
        raise Qwen3PerturbationError("acceptance gate partition schema mismatch")
    for partition, expected_count in (("cage_qwen3", 11), ("kitty_qwen3", 3)):
        record = partitions[partition]
        if record.get("case_count_per_repeat") != expected_count:
            raise Qwen3PerturbationError(f"{partition} acceptance case count mismatch")
        _require_sha256(f"{partition}.case_ids_sha256", record.get("case_ids_sha256"))
        _require_sha256(
            f"{partition}.scientific_payload_sha256",
            record.get("scientific_payload_sha256"),
        )
        for repeat in ("repeat_a", "repeat_b"):
            artifact = record.get(repeat, {})
            if not isinstance(artifact.get("output_dir"), str) or not isinstance(
                artifact.get("execution_log"), str
            ):
                raise Qwen3PerturbationError(f"{partition}.{repeat} paths are invalid")
            for name in (
                "run_identity_sha256",
                "summary_sha256",
                "execution_log_sha256",
            ):
                _require_sha256(
                    f"{partition}.{repeat}.{name}", artifact.get(name)
                )
    comparison = gate.get("comparison", {})
    if tuple(comparison.get("fields", ())) != SCIENTIFIC_PAYLOAD_FIELDS:
        raise Qwen3PerturbationError("acceptance scientific payload fields mismatch")
    if comparison.get("required") != "bitwise_equal_canonical_json":
        raise Qwen3PerturbationError("acceptance comparison rule mismatch")
    if comparison.get("cage_qwen3_pass") is not True or comparison.get("kitty_qwen3_pass") is not True:
        raise Qwen3PerturbationError("both acceptance partitions must pass")
    authorization = gate.get("full_run_authorization", {})
    if authorization.get("total_cases") != 1300 or authorization.get("layer_records") != 46800:
        raise Qwen3PerturbationError("full-run gate counts mismatch")
    if authorization.get("quality_grid_mutation_allowed") is not False:
        raise Qwen3PerturbationError("full-run gate must forbid quality-grid mutation")


def scientific_payload_sha256(
    output_dir: str | Path, *, expected_case_count: int
) -> tuple[str, list[dict[str, Any]]]:
    root = Path(output_dir)
    paths = sorted((root / "cases").glob("*.json"))
    if len(paths) != expected_case_count:
        raise Qwen3PerturbationError(
            f"{root} contains {len(paths)} case files, expected {expected_case_count}"
        )
    payload: list[dict[str, Any]] = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Qwen3PerturbationError(f"cannot read acceptance case {path}: {error}") from error
        if not isinstance(record, dict):
            raise Qwen3PerturbationError(f"acceptance case is not an object: {path}")
        try:
            payload.append({name: record[name] for name in SCIENTIFIC_PAYLOAD_FIELDS})
        except KeyError as error:
            raise Qwen3PerturbationError(
                f"acceptance case {path} lacks scientific field {error.args[0]!r}"
            ) from error
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), payload


def verify_perturbation_acceptance_artifacts(gate: dict[str, Any]) -> None:
    for partition, record in gate["partitions"].items():
        expected_count = record["case_count_per_repeat"]
        repeat_payloads = []
        for repeat in ("repeat_a", "repeat_b"):
            artifact = record[repeat]
            root = Path(artifact["output_dir"])
            identity_path = root / "run_identity.json"
            summary_path = root / "summary.json"
            log_path = Path(artifact["execution_log"])
            for label, path, expected_sha in (
                ("run identity", identity_path, artifact["run_identity_sha256"]),
                ("summary", summary_path, artifact["summary_sha256"]),
                ("execution log", log_path, artifact["execution_log_sha256"]),
            ):
                if not path.is_file() or file_sha256(path) != expected_sha:
                    raise Qwen3PerturbationError(
                        f"{partition} {repeat} {label} artifact mismatch"
                    )
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise Qwen3PerturbationError(
                    f"cannot read {partition} {repeat} summary: {error}"
                ) from error
            if (
                summary.get("status") != "pass"
                or summary.get("completed_cases") != expected_count
                or summary.get("failure_records") != 0
                or summary.get("case_ids_sha256") != record["case_ids_sha256"]
            ):
                raise Qwen3PerturbationError(
                    f"{partition} {repeat} summary content mismatch"
                )
            payload_sha, payload = scientific_payload_sha256(
                root, expected_case_count=expected_count
            )
            if payload_sha != record["scientific_payload_sha256"]:
                raise Qwen3PerturbationError(
                    f"{partition} {repeat} scientific payload hash mismatch"
                )
            repeat_payloads.append(payload)
        if repeat_payloads[0] != repeat_payloads[1]:
            raise Qwen3PerturbationError(
                f"{partition} acceptance repeats are not bitwise equal"
            )


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
    "PERTURBATION_ACCEPTANCE_GATE_ID",
    "PERTURBATION_PROTOCOL_ID",
    "Qwen3PerturbationError",
    "SCIENTIFIC_PAYLOAD_FIELDS",
    "aggregate_layer_metrics",
    "file_sha256",
    "load_perturbation_acceptance_gate",
    "load_perturbation_protocol",
    "perturbation_case_id",
    "scientific_payload_sha256",
    "validate_aggregates",
    "validate_layer_records",
    "validate_perturbation_acceptance_gate",
    "validate_perturbation_protocol",
    "verify_perturbation_acceptance_artifacts",
]
