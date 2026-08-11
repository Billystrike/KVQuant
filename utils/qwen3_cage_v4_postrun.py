from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256
from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes
from utils.qwen3_formal import formal_scoring
from utils.qwen3_memory import (
    estimate_qwen3_cage_bytes,
    estimate_qwen3_fp16_bytes,
    estimate_qwen3_kivi_bytes,
    estimate_qwen3_kitty_bytes,
)
from utils.qwen3_perturbation_protocol import validate_aggregates, validate_layer_records


ARTIFACT_SET_ID = "qwen3-8b-cage-v4-pg19-metric-screen-artifacts-v1"
PARTITIONS = ("cage_qwen3", "kitty_qwen3")
SCIENTIFIC_FIELDS = (
    "case_id",
    "partition",
    "method",
    "input",
    "memory",
    "scoring",
    "local_perturbation",
    "quality_cache",
)
EXPECTED_COUNTS = {
    "cage_qwen3": {
        "case_count": 600,
        "compressed_case_count": 480,
        "target_token_count": 38_400,
        "layer_record_count": 17_280,
    },
    "kitty_qwen3": {
        "case_count": 120,
        "compressed_case_count": 120,
        "target_token_count": 7_680,
        "layer_record_count": 4_320,
    },
}
EXPECTED_RUNTIME_DEPENDENCIES = {
    "kitty_commit": "dfd2c07b407d6b407179359207c612ab631f3ed1",
    "kitty_dirty": False,
    "transformers_commit": "37f8b0b53512e6aae0cfd15746c133c101783178",
}


class CageV4PostrunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV4PostrunError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV4PostrunError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _require_sha256(name: str, value: Any) -> None:
    _require(isinstance(value, str) and len(value) == 64, f"{name} is not SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise CageV4PostrunError(f"{name} is not SHA-256") from error


def validate_artifact_manifest(
    manifest: dict[str, Any],
    *,
    execution_sha256: str,
    protocol_sha256: str,
    input_manifest_sha256: str,
    gate_sha256: str,
) -> None:
    _require(manifest.get("schema_version") == 1, "artifact manifest schema mismatch")
    _require(manifest.get("artifact_set_id") == ARTIFACT_SET_ID, "artifact set ID mismatch")
    _require(
        manifest.get("status") == "declared_from_execution_outputs_before_joint_postrun_audit",
        "artifact manifest status mismatch",
    )
    _require(manifest.get("claim_eligible") is False, "screen artifacts became claim-eligible")
    _require(manifest.get("interpretation_performed") is False, "premature interpretation recorded")
    for name, expected in (
        ("execution_sha256", execution_sha256),
        ("protocol_sha256", protocol_sha256),
        ("input_manifest_sha256", input_manifest_sha256),
        ("acceptance_gate_sha256", gate_sha256),
    ):
        _require(manifest.get(name) == expected, f"artifact {name} mismatch")
    for name in ("runner_sha256", "accepted_runtime_sha256"):
        _require_sha256(name, manifest.get(name))
    _require(tuple(manifest.get("partitions", {})) == PARTITIONS, "artifact partitions mismatch")
    for partition, expected_counts in EXPECTED_COUNTS.items():
        record = manifest["partitions"][partition]
        for name, expected in expected_counts.items():
            _require(record.get(name) == expected, f"{partition}.{name} mismatch")
        _require(record.get("failure_count") == 0, f"{partition} declares failures")
        _require(
            record.get("new_cases", 0) + record.get("resumed_cases", 0)
            == expected_counts["case_count"],
            f"{partition} new/resumed count mismatch",
        )
        for name in (
            "case_ids_sha256",
            "run_identity_sha256",
            "summary_sha256",
            "shell_case_file_manifest_sha256",
            "execution_log_sha256",
        ):
            _require_sha256(f"{partition}.{name}", record.get(name))
        _require(record.get("execution_log_size_bytes", 0) > 0, f"{partition} log size invalid")
    _require(
        manifest.get("joint_expected")
        == {
            "case_count": 720,
            "compressed_case_count": 600,
            "target_token_count": 46_080,
            "layer_record_count": 21_600,
            "failure_count": 0,
            "interpretation_allowed_before_audit_pass": False,
            "holdout_access": False,
            "cage_v4_candidate_execution": False,
        },
        "joint artifact expectations mismatch",
    )


def shell_case_manifest_sha256(paths: list[Path]) -> str:
    lines = "".join(
        f"{file_sha256(path)}  cases/{path.name}\n" for path in sorted(paths, key=lambda p: p.name)
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def canonical_case_manifest_sha256(paths: list[Path]) -> str:
    records = [
        {"name": path.name, "sha256": file_sha256(path)}
        for path in sorted(paths, key=lambda p: p.name)
    ]
    return canonical_sha256(records)


def expected_memory_report(method: dict[str, Any]) -> dict[str, Any]:
    seq_len = method["prompt_length"]
    config = method["config"]
    name = method["name"]
    if method["runtime_family"] == "formal":
        if name == "fp16":
            report = estimate_qwen3_fp16_bytes(seq_len=seq_len)
        elif name == "kivi":
            report = estimate_qwen3_kivi_bytes(
                seq_len=seq_len,
                group_size=config["group_size"],
                residual_length=config["residual_length"],
                bits=config["bits"],
            )
        elif name == "cage":
            report = estimate_qwen3_cage_bytes(
                seq_len=seq_len,
                residual_length=config["residual_length"],
                key_group_sizes=config["key_group_sizes"],
                value_group_sizes=config["value_group_sizes"],
                bits=config["bits"],
            )
        else:
            raise CageV4PostrunError(f"unsupported formal memory family: {name}")
    elif name == "cage_v2":
        report = estimate_qwen3_cage_v2_bytes(
            seq_len=seq_len,
            residual_length=config["residual_length"],
            one_bit_channels=config["one_bit_channels"],
            two_bit_channels=config["two_bit_channels"],
            sink_length=config["sink_length"],
        )
    elif name == "kitty":
        report = estimate_qwen3_kitty_bytes(
            seq_len=seq_len,
            boosted_channels=config["boosted_channels"],
        )
    else:
        raise CageV4PostrunError(f"unsupported round1 memory family: {name}")
    _require(
        report.get("model_total_bytes") == method["packed_bytes"],
        f"memory report differs from frozen packed bytes: {method['id']}",
    )
    return report


def _validate_quality_cache(record: dict[str, Any]) -> None:
    cache = record.get("quality_cache")
    expected_length = record["input"]["prompt_length"] + 63
    _require(isinstance(cache, dict), "quality cache is missing")
    _require(cache.get("reported_seq_length") == expected_length, "quality cache length mismatch")
    _require(cache.get("expected_seq_length") == expected_length, "expected cache length mismatch")
    _require(cache.get("layer_count") == 36, "quality cache layer count mismatch")
    _require(cache.get("tensors_finite") is True, "quality cache contains non-finite tensors")


def _validate_local_perturbation(record: dict[str, Any]) -> int:
    local = record.get("local_perturbation")
    compressed = record["method"]["metric_method_id"] != "fp16"
    if not compressed:
        _require(local is None, "FP16 record carries local perturbation")
        return 0
    _require(isinstance(local, dict), "compressed record lacks local perturbation")
    _require(local.get("metric") == "joint_post_o_proj_mse", "local metric changed")
    layers = local.get("layer_metrics", [])
    validate_layer_records(layers)
    validate_aggregates(local.get("aggregates", {}), layers)
    _require(
        local["aggregates"]["relative_k_reconstruction_error"]["maximum"] > 0,
        "compressed record did not perturb Key cache",
    )
    _require(
        local["aggregates"]["relative_v_reconstruction_error"]["maximum"] > 0,
        "compressed record did not perturb Value cache",
    )
    return len(layers)


def validate_partition(
    partition: str,
    spec: dict[str, Any],
    *,
    expected_cases: list[dict[str, Any]],
    execution_source_commit: str,
    execution_sha256: str,
    protocol_sha256: str,
    input_manifest_sha256: str,
    gate_sha256: str,
    runner_sha256: str,
    accepted_runtime_sha256: str,
) -> dict[str, Any]:
    _require(partition in PARTITIONS, "unsupported metric screen partition")
    root = Path(spec["directory"])
    identity_path = root / "run_identity.json"
    summary_path = root / "summary.json"
    log_path = Path(spec["execution_log"])
    _require(file_sha256(identity_path) == spec["run_identity_sha256"], f"{partition} identity hash mismatch")
    _require(file_sha256(summary_path) == spec["summary_sha256"], f"{partition} summary hash mismatch")
    _require(file_sha256(log_path) == spec["execution_log_sha256"], f"{partition} log hash mismatch")
    _require(log_path.stat().st_size == spec["execution_log_size_bytes"], f"{partition} log size mismatch")
    log_text = log_path.read_text(encoding="utf-8")
    _require("Traceback (most recent call last)" not in log_text, f"{partition} log contains traceback")
    _require("ERROR conda.cli.main_run" not in log_text, f"{partition} log contains conda error")
    result_marker = "CAGE_FULL_RESULT=PASS" if partition == "cage_qwen3" else "KITTY_FULL_RESULT=PASS"
    _require(result_marker in log_text, f"{partition} log lacks pass marker")
    _require("RUN_STATUS=0" in log_text and "TEE_STATUS=0" in log_text, f"{partition} log status failed")

    identity = load_json(identity_path)
    summary = load_json(summary_path)
    expected_ids = [case["case_id"] for case in expected_cases]
    expected_by_id = {case["case_id"]: case for case in expected_cases}
    _require(len(expected_by_id) == len(expected_cases), f"{partition} expected IDs are duplicated")
    expected_identity = {
        "schema_version": 1,
        "experiment": "qwen3_cage_v4_pg19_metric_validity_screen",
        "claim_eligible": False,
        "partition": partition,
        "stage": "screen_full",
        "execution_sha256": execution_sha256,
        "protocol_sha256": protocol_sha256,
        "acceptance_gate_sha256": gate_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "runner_sha256": runner_sha256,
        "accepted_runtime_sha256": accepted_runtime_sha256,
        "expected_case_ids": expected_ids,
    }
    for name, expected in expected_identity.items():
        _require(identity.get(name) == expected, f"{partition} identity {name} mismatch")
    _require(
        identity.get("runtime_dependency_state") == EXPECTED_RUNTIME_DEPENDENCIES,
        f"{partition} runtime dependency mismatch",
    )
    expected_source = {"git_commit": execution_source_commit, "dirty": False}
    if partition == "kitty_qwen3":
        expected_source.update(
            kitty_commit=EXPECTED_RUNTIME_DEPENDENCIES["kitty_commit"],
            kitty_dirty=False,
            transformers_commit=EXPECTED_RUNTIME_DEPENDENCIES["transformers_commit"],
        )
    _require(identity.get("source_state") == expected_source, f"{partition} source state mismatch")

    for name, expected in (
        ("schema_version", 1),
        ("status", "pass"),
        ("claim_eligible", False),
        ("partition", partition),
        ("stage", "screen_full"),
        ("expected_cases", spec["case_count"]),
        ("completed_cases", spec["case_count"]),
        ("new_cases", spec["new_cases"]),
        ("resumed_cases", spec["resumed_cases"]),
        ("failure_records", 0),
    ):
        _require(summary.get(name) == expected, f"{partition} summary {name} mismatch")
    _require(summary.get("run_identity") == identity, f"{partition} summary identity mismatch")
    model = summary.get("model", {})
    _require(model.get("partition") == partition, f"{partition} model partition mismatch")
    _require(all(model.get("checks", {}).values()), f"{partition} model checks failed")
    _require(model.get("metric_screen_runner_sha256") == runner_sha256, "model runner mismatch")
    _require(model.get("accepted_runtime_sha256") == accepted_runtime_sha256, "model runtime mismatch")

    paths = sorted((root / "cases").glob("*.json"), key=lambda path: path.name)
    _require(len(paths) == spec["case_count"], f"{partition} case file count mismatch")
    failures = sorted((root / "failures").glob("*.json")) if (root / "failures").exists() else []
    _require(not failures, f"{partition} failure files remain")
    _require(
        shell_case_manifest_sha256(paths) == spec["shell_case_file_manifest_sha256"],
        f"{partition} shell case manifest mismatch",
    )

    scientific_payload = []
    layer_record_count = 0
    target_token_count = 0
    compressed_case_count = 0
    observed_ids = set()
    for path in paths:
        record = load_json(path)
        case_id = record.get("case_id")
        _require(path.stem == case_id, f"case filename/ID mismatch: {path}")
        _require(case_id in expected_by_id, f"unexpected {partition} case: {case_id}")
        expected = expected_by_id[case_id]
        _require(record.get("schema_version") == 1, f"case schema mismatch: {case_id}")
        _require(record.get("status") == "completed", f"case status mismatch: {case_id}")
        _require(record.get("partition") == partition, f"case partition mismatch: {case_id}")
        _require(record.get("stage") == "screen_full", f"case stage mismatch: {case_id}")
        _require(record.get("identity") == identity, f"case identity mismatch: {case_id}")
        _require(record.get("model") == model, f"case model mismatch: {case_id}")
        _require(record.get("method") == expected["method"], f"case method mismatch: {case_id}")
        _require(record.get("input") == expected["input"], f"case input mismatch: {case_id}")
        _require(record.get("memory") == expected["memory"], f"case memory mismatch: {case_id}")
        _require(
            record.get("representation_note")
            == "accuracy simulation with packed paper-estimate memory",
            f"case representation note mismatch: {case_id}",
        )
        scoring = record.get("scoring", {})
        _require(scoring == formal_scoring(scoring.get("token_nlls", [])), f"NLL mismatch: {case_id}")
        target_token_count += scoring["target_count"]
        _validate_quality_cache(record)
        local_layers = _validate_local_perturbation(record)
        layer_record_count += local_layers
        if local_layers:
            compressed_case_count += 1
        observed_ids.add(case_id)
        scientific_payload.append({name: record[name] for name in SCIENTIFIC_FIELDS})
    _require(observed_ids == set(expected_ids), f"{partition} expected case set mismatch")
    _require(compressed_case_count == spec["compressed_case_count"], f"{partition} compressed count mismatch")
    _require(target_token_count == spec["target_token_count"], f"{partition} target count mismatch")
    _require(layer_record_count == spec["layer_record_count"], f"{partition} layer count mismatch")
    scientific_payload.sort(key=lambda item: item["case_id"])
    return {
        "case_count": len(paths),
        "compressed_case_count": compressed_case_count,
        "target_token_count": target_token_count,
        "layer_record_count": layer_record_count,
        "failure_count": 0,
        "case_ids_sha256": canonical_sha256(expected_ids),
        "canonical_case_file_manifest_sha256": canonical_case_manifest_sha256(paths),
        "shell_case_file_manifest_sha256": spec["shell_case_file_manifest_sha256"],
        "scientific_payload_sha256": canonical_sha256(scientific_payload),
        "model_identity_sha256": canonical_sha256(model),
        "scientific_payload": scientific_payload,
    }


__all__ = [
    "ARTIFACT_SET_ID",
    "CageV4PostrunError",
    "EXPECTED_COUNTS",
    "SCIENTIFIC_FIELDS",
    "canonical_case_manifest_sha256",
    "expected_memory_report",
    "load_json",
    "shell_case_manifest_sha256",
    "validate_artifact_manifest",
    "validate_partition",
]
