from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v3_execution import canonical_sha256, load_json
from utils.qwen3_cage_v3_protocol import file_sha256
from utils.qwen3_cage_v3_screen import SCIENTIFIC_FIELDS
from utils.qwen3_perturbation_protocol import (
    aggregate_layer_metrics,
    validate_aggregates,
    validate_layer_records,
)


ARTIFACT_SET_ID = "qwen3-8b-cage-v3-development-screen-artifacts-v1"
ATTEMPT_SET_ID = "qwen3-8b-cage-v3-screen-postrun-attempts-v1"
PARTITIONS = ("cage_qwen3", "kitty_qwen3")
EXPECTED_COUNTS = {"cage_qwen3": 60, "kitty_qwen3": 15}
EXPECTED_LAYER_COUNTS = {partition: count * 36 for partition, count in EXPECTED_COUNTS.items()}
EXPECTED_KITTY_COMMIT = "dfd2c07b407d6b407179359207c612ab631f3ed1"
EXPECTED_TRANSFORMERS_COMMIT = "37f8b0b53512e6aae0cfd15746c133c101783178"


class CageV3PostrunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3PostrunError(message)


def validate_artifact_manifest(
    manifest: dict[str, Any],
    *,
    execution_sha256: str,
    protocol_sha256: str,
    input_manifest_sha256: str,
    quota_plan_sha256: str,
    gate_sha256: str,
) -> None:
    _require(manifest.get("schema_version") == 1, "artifact manifest schema mismatch")
    _require(manifest.get("artifact_set_id") == ARTIFACT_SET_ID, "artifact set ID mismatch")
    _require(
        manifest.get("status") == "declared_from_execution_outputs_before_joint_postrun_audit",
        "artifact manifest status mismatch",
    )
    _require(manifest.get("claim_eligible") is False, "screen artifacts must be claim-ineligible")
    _require(manifest.get("interpretation_performed") is False, "artifact declaration performed interpretation")
    for field, expected in (
        ("execution_sha256", execution_sha256),
        ("protocol_sha256", protocol_sha256),
        ("manifest_sha256", input_manifest_sha256),
        ("quota_plan_sha256", quota_plan_sha256),
        ("acceptance_gate_sha256", gate_sha256),
    ):
        _require(manifest.get(field) == expected, f"artifact {field} mismatch")
    source_commit = manifest.get("execution_source_commit")
    _require(isinstance(source_commit, str) and len(source_commit) == 40, "execution source commit is invalid")
    _require(tuple(manifest.get("partitions", {})) == PARTITIONS, "artifact partitions mismatch")
    for partition in PARTITIONS:
        record = manifest["partitions"][partition]
        _require(record.get("case_count") == EXPECTED_COUNTS[partition], f"{partition} case count mismatch")
        _require(
            record.get("layer_record_count") == EXPECTED_LAYER_COUNTS[partition],
            f"{partition} layer count mismatch",
        )
        _require(record.get("failure_count") == 0, f"{partition} declares failures")
        for field in (
            "case_ids_sha256",
            "run_identity_sha256",
            "summary_sha256",
            "shell_case_file_manifest_sha256",
            "execution_log_sha256",
        ):
            value = record.get(field)
            _require(isinstance(value, str) and len(value) == 64, f"{partition}.{field} is invalid")
        _require(record.get("execution_log_size_bytes", 0) > 0, f"{partition} log size is invalid")
        _require(isinstance(record.get("completion_marker"), str), f"{partition} completion marker is invalid")
    failed = manifest.get("failed_pre_case_attempts")
    _require(isinstance(failed, list) and len(failed) == 1, "failed pre-case attempt declaration mismatch")
    attempt = failed[0]
    _require(attempt.get("partition") == "kitty_qwen3", "failed attempt partition mismatch")
    _require(attempt.get("case_execution_started") is False, "failed attempt executed a case")
    _require(attempt.get("observed_literal_length") == 34, "failed attempt observed literal length mismatch")
    _require(attempt.get("expected_literal_length") == 40, "failed attempt expected literal length mismatch")
    for field in (
        "execution_log_sha256",
        "archived_script_sha256",
        "observed_literal_sha256",
        "expected_literal_sha256",
    ):
        value = attempt.get(field)
        _require(isinstance(value, str) and len(value) == 64, f"failed attempt {field} is invalid")
    _require(
        manifest.get("joint_expected")
        == {
            "case_count": 75,
            "layer_record_count": 2700,
            "failure_count": 0,
            "failed_pre_case_attempt_count": 1,
            "interpretation_allowed_before_audit_pass": False,
            "holdout_metrics_consumed": False,
            "reserved_unseen_metrics_consumed": False,
        },
        "joint artifact expectations mismatch",
    )


def shell_case_manifest_sha256(paths: list[Path]) -> str:
    lines = "".join(f"{file_sha256(path)}  cases/{path.name}\n" for path in sorted(paths))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def canonical_case_manifest_sha256(paths: list[Path]) -> str:
    records = [{"name": path.name, "sha256": file_sha256(path)} for path in sorted(paths)]
    return canonical_sha256(records)


def validate_failed_pre_case_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    log_path = Path(attempt["execution_log"])
    script_path = Path(attempt["archived_script"])
    _require(file_sha256(log_path) == attempt["execution_log_sha256"], "failed attempt log hash mismatch")
    _require(log_path.stat().st_size == attempt["execution_log_size_bytes"], "failed attempt log size mismatch")
    _require(file_sha256(script_path) == attempt["archived_script_sha256"], "failed attempt script hash mismatch")
    log_text = log_path.read_text(encoding="utf-8")
    script_text = script_path.read_text(encoding="utf-8")
    _require("ERROR: unexpected Kitty HEAD" in log_text, "failed attempt reason missing from log")
    _require("PIPELINE_STATUS=1" in log_text, "failed attempt status missing from log")
    _require("Loading checkpoint shards" not in log_text, "failed attempt loaded the model")
    observed = attempt["observed_literal"]
    expected = attempt["expected_literal"]
    _require(f"EXPECTED_KITTY={observed}" in script_text, "archived script literal mismatch")
    _require(len(observed) == attempt["observed_literal_length"], "observed literal length mismatch")
    _require(len(expected) == attempt["expected_literal_length"], "expected literal length mismatch")
    _require(hashlib.sha256(observed.encode("ascii")).hexdigest() == attempt["observed_literal_sha256"], "observed literal hash mismatch")
    _require(hashlib.sha256(expected.encode("ascii")).hexdigest() == attempt["expected_literal_sha256"], "expected literal hash mismatch")
    _require(expected == EXPECTED_KITTY_COMMIT, "failed-attempt expected Kitty identity mismatch")
    _require(observed != expected, "failed attempt literal unexpectedly matched")
    return {
        "partition": "kitty_qwen3",
        "reason": attempt["reason"],
        "case_execution_started": False,
        "execution_log_sha256": attempt["execution_log_sha256"],
        "archived_script_sha256": attempt["archived_script_sha256"],
        "observed_literal_length": len(observed),
        "expected_literal_length": len(expected),
    }


def validate_validation_attempt_manifest(manifest: dict[str, Any]) -> None:
    _require(manifest.get("schema_version") == 1, "validation attempt schema mismatch")
    _require(manifest.get("attempt_set_id") == ATTEMPT_SET_ID, "validation attempt set ID mismatch")
    _require(
        manifest.get("status") == "frozen_after_failed_validators_before_retry",
        "validation attempt status mismatch",
    )
    _require(manifest.get("claim_eligible") is False, "validation attempts must be claim-ineligible")
    _require(manifest.get("interpretation_performed") is False, "validation attempt performed interpretation")
    _require(manifest.get("scientific_artifacts_mutated") is False, "validation attempt mutated scientific artifacts")
    attempts = manifest.get("attempts")
    _require(isinstance(attempts, list) and len(attempts) == 2, "validation attempt count mismatch")
    expected = (
        (1, 28, "cage_qwen3 log lacks successful pipeline status"),
        (2, 30, "cage_qwen3 shell case manifest mismatch"),
    )
    for attempt, (index, test_count, failure_message) in zip(attempts, expected):
        _require(attempt.get("attempt_index") == index, "validation attempt index mismatch")
        _require(attempt.get("output_created") is False, "failed validation unexpectedly created output")
        _require(attempt.get("tests_passed_before_failure") == test_count, "failed validation test count mismatch")
        _require(attempt.get("failure_type") == "CageV3PostrunError", "validation failure type mismatch")
        _require(attempt.get("failure_message") == failure_message, "validation failure message mismatch")
        _require(attempt.get("scientific_artifacts_mutated") is False, "failed validation mutated artifacts")
        _require(attempt.get("interpretation_performed") is False, "failed validation performed interpretation")
        for field, length in (("source_commit", 40), ("validator_sha256", 64), ("execution_log_sha256", 64)):
            value = attempt.get(field)
            _require(isinstance(value, str) and len(value) == length, f"validation attempt {field} is invalid")
        _require(attempt.get("execution_log_size_bytes", 0) > 0, "validation attempt log size is invalid")


def validate_failed_validation_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    log_path = Path(attempt["execution_log"])
    output_path = Path(attempt["intended_output"])
    _require(file_sha256(log_path) == attempt["execution_log_sha256"], "validation attempt log hash mismatch")
    _require(log_path.stat().st_size == attempt["execution_log_size_bytes"], "validation attempt log size mismatch")
    text = log_path.read_text(encoding="utf-8")
    _require("Ran 28 tests" in text and "OK" in text, "validation attempt test evidence mismatch")
    _require(attempt["failure_message"] in text, "validation attempt failure message missing")
    _require("JOINT 75-CASE DEEP AUDIT" in text, "validation attempt did not reach deep audit")
    _require(not output_path.exists(), "failed validation output now unexpectedly exists")
    return {
        "attempt_index": attempt["attempt_index"],
        "source_commit": attempt["source_commit"],
        "validator_sha256": attempt["validator_sha256"],
        "execution_log_sha256": attempt["execution_log_sha256"],
        "failure_type": attempt["failure_type"],
        "failure_message": attempt["failure_message"],
        "output_created": False,
        "scientific_artifacts_mutated": False,
        "interpretation_performed": False,
    }


def _validate_cache(record: dict[str, Any]) -> None:
    cache = record.get("cache", {})
    method = record["method"]
    expected_length = record["input"]["prompt_length"] + 1
    _require(cache.get("reported_seq_length") == expected_length, "cache reported length mismatch")
    _require(cache.get("expected_seq_length") == expected_length, "cache expected length mismatch")
    _require(cache.get("layer_count") == 36, "cache layer count mismatch")
    _require(cache.get("tensor_dtypes") == ["torch.float16"], "cache dtype mismatch")
    _require(cache.get("tensors_finite") is True, "cache contains non-finite values")
    config = method["config"]
    if method["name"] == "kitty":
        boosted = config["boosted_channels"]
        _require(
            cache.get("official_config")
            == {
                "sink_length": 32,
                "buffer_length": 128,
                "group_size": 128,
                "kbits": 2,
                "vbits": 2,
                "promote_ratio": boosted / 128,
                "promote_bit": 4,
                "channel_selection": 1,
                "VCache_BitDecoding": False,
                "PostQuant": True,
            },
            "Kitty official configuration mismatch",
        )
        return
    _require(method["name"] == "cage_v2", "unsupported CAGE-v3 screen method")
    residual = config["residual_length"]
    sink = min(expected_length, config["sink_length"])
    non_sink = expected_length - sink
    expected_key = sink + non_sink - non_sink % residual
    expected_value = max(sink, expected_length - residual)
    _require(cache.get("key_quantized_lengths") == [expected_key], "Key cache mechanics mismatch")
    _require(cache.get("value_quantized_lengths") == [expected_value], "Value cache mechanics mismatch")
    _require(cache.get("value_adaptive") is False, "CAGE-v3 Value adaptation changed")
    _require(cache.get("one_bit_channels") == 0, "CAGE-v3 one-bit refinement changed")
    _require(cache.get("one_bit_channels") == config["one_bit_channels"], "one-bit quota mismatch")
    _require(cache.get("two_bit_channels") == config["two_bit_channels"], "two-bit quota mismatch")


def validate_partition(
    partition: str,
    spec: dict[str, Any],
    *,
    expected_cases: dict[str, dict[str, Any]],
    execution_source_commit: str,
    expected_source_sha256: dict[str, str],
    execution_sha256: str,
    protocol_sha256: str,
    input_manifest_sha256: str,
    quota_plan_sha256: str,
) -> dict[str, Any]:
    _require(partition in PARTITIONS, "unsupported postrun partition")
    root = Path(spec["directory"])
    identity_path = root / "run_identity.json"
    summary_path = root / "summary.json"
    _require(file_sha256(identity_path) == spec["run_identity_sha256"], f"{partition} identity hash mismatch")
    _require(file_sha256(summary_path) == spec["summary_sha256"], f"{partition} summary hash mismatch")
    log_path = Path(spec["execution_log"])
    _require(file_sha256(log_path) == spec["execution_log_sha256"], f"{partition} log hash mismatch")
    _require(log_path.stat().st_size == spec["execution_log_size_bytes"], f"{partition} log size mismatch")
    log_text = log_path.read_text(encoding="utf-8")
    _require("Traceback (most recent call last)" not in log_text, f"{partition} log contains traceback")
    _require("ERROR conda.cli.main_run" not in log_text, f"{partition} log contains conda error")
    _require(spec["completion_marker"] in log_text, f"{partition} log lacks completion marker")

    identity = load_json(identity_path)
    summary = load_json(summary_path)
    count = EXPECTED_COUNTS[partition]
    for field, expected in (
        ("schema_version", 1),
        ("status", "pass"),
        ("claim_eligible", False),
        ("partition", partition),
        ("stage", "screen_full"),
        ("expected_cases", count),
        ("completed_cases", count),
        ("new_cases", count),
        ("resumed_cases", 0),
        ("failure_records", 0),
        ("case_ids_sha256", spec["case_ids_sha256"]),
    ):
        _require(summary.get(field) == expected, f"{partition} summary {field} mismatch")
    _require(summary.get("identity") == identity, f"{partition} summary identity mismatch")
    _require(identity.get("expected_case_ids") == list(expected_cases), f"{partition} ordered case IDs mismatch")
    _require(identity.get("execution_sha256") == execution_sha256, f"{partition} execution mismatch")
    _require(identity.get("protocol_sha256") == protocol_sha256, f"{partition} protocol mismatch")
    _require(identity.get("manifest_sha256") == input_manifest_sha256, f"{partition} manifest mismatch")
    _require(identity.get("quota_plan_sha256") == quota_plan_sha256, f"{partition} quota plan mismatch")
    _require(identity.get("experiment") == "qwen3_cage_v3_screen_development_only", "experiment identity mismatch")
    source = identity.get("source_state", {})
    _require(source.get("git_commit") == execution_source_commit, f"{partition} source commit mismatch")
    _require(source.get("dirty") is False, f"{partition} source was dirty")
    if partition == "kitty_qwen3":
        _require(source.get("kitty_commit") == EXPECTED_KITTY_COMMIT, "Kitty commit mismatch")
        _require(source.get("transformers_commit") == EXPECTED_TRANSFORMERS_COMMIT, "Transformers commit mismatch")
        _require(source.get("kitty_dirty") is False, "Kitty source was dirty")
    model = summary.get("model", {})
    _require(model.get("partition") == partition, f"{partition} model partition mismatch")
    _require(all(model.get("checks", {}).values()), f"{partition} model checks failed")
    _require(model.get("source_sha256") == expected_source_sha256, f"{partition} model source hashes mismatch")

    paths = sorted((root / "cases").glob("*.json"))
    _require(len(paths) == count, f"{partition} case file count mismatch")
    failure_paths = sorted((root / "failures").glob("*.json")) if (root / "failures").exists() else []
    _require(not failure_paths, f"{partition} failure files remain")
    _require(
        shell_case_manifest_sha256(paths) == spec["shell_case_file_manifest_sha256"],
        f"{partition} shell case manifest mismatch",
    )
    layer_count = 0
    scientific_payload = []
    seen = set()
    for path in paths:
        record = load_json(path)
        case_id = record.get("case_id")
        _require(path.stem == case_id, f"case filename/ID mismatch: {path}")
        _require(case_id in expected_cases, f"unexpected {partition} case: {case_id}")
        expected = expected_cases[case_id]
        _require(record.get("schema_version") == 1 and record.get("status") == "completed", f"invalid case: {path}")
        _require(record.get("identity") == identity, f"case identity mismatch: {case_id}")
        _require(record.get("model") == model, f"case model mismatch: {case_id}")
        for field in ("method", "input", "memory"):
            _require(record.get(field) == expected[field], f"case {field} mismatch: {case_id}")
        layers = record.get("layer_metrics", [])
        validate_layer_records(layers)
        validate_aggregates(record.get("aggregates", {}), layers)
        _require(aggregate_layer_metrics(layers) == record["aggregates"], f"aggregate mismatch: {case_id}")
        _require(record["aggregates"]["relative_k_reconstruction_error"]["maximum"] > 0, f"no Key perturbation: {case_id}")
        _require(record["aggregates"]["relative_v_reconstruction_error"]["maximum"] > 0, f"no Value perturbation: {case_id}")
        _validate_cache(record)
        seen.add(case_id)
        layer_count += len(layers)
        scientific_payload.append({field: record[field] for field in SCIENTIFIC_FIELDS})
    _require(seen == set(expected_cases), f"{partition} expected case set mismatch")
    _require(layer_count == EXPECTED_LAYER_COUNTS[partition], f"{partition} total layer count mismatch")
    scientific_payload.sort(key=lambda item: item["case_id"])
    return {
        "case_count": len(paths),
        "layer_record_count": layer_count,
        "failure_count": 0,
        "case_ids_sha256": spec["case_ids_sha256"],
        "canonical_case_file_manifest_sha256": canonical_case_manifest_sha256(paths),
        "shell_case_file_manifest_sha256": spec["shell_case_file_manifest_sha256"],
        "scientific_payload_sha256": canonical_sha256(scientific_payload),
        "model_identity_sha256": canonical_sha256(model),
        "scientific_payload": scientific_payload,
    }


__all__ = [
    "ARTIFACT_SET_ID",
    "CageV3PostrunError",
    "canonical_case_manifest_sha256",
    "shell_case_manifest_sha256",
    "validate_artifact_manifest",
    "validate_failed_pre_case_attempt",
    "validate_failed_validation_attempt",
    "validate_partition",
    "validate_validation_attempt_manifest",
]
