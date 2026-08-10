from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v4_data import (
    canonical_sha256,
    file_sha256,
    load_data_protocol,
    validate_input_manifest,
)
from utils.qwen3_cage_v4_gate import EXPECTED_AUTHORIZATION, load_acceptance_gate
from utils.qwen3_cage_v4_metric_protocol import load_metric_protocol
from utils.qwen3_cage_v4_screen import expand_full_screen_cases


EXECUTION_ID = "qwen3-8b-cage-v4-pg19-metric-screen-execution-v1"
EXPECTED_BOUNDARY = {
    "input_partition": "screen",
    "screen_documents": 20,
    "screen_anchors": 40,
    "continuation_tokens_per_case": 64,
    "run_partitions_in_isolated_environments": True,
    "resume_completed_cases": True,
    "interpret_results_before_both_partitions_finish": False,
    "pg19_holdout_method_metrics": False,
    "pg19_test_access": False,
    "cage_v4_candidate_execution": False,
    "runtime_latency_throughput_or_real_cuda_memory_claim": False,
}
EXPECTED_PARTITIONS = {
    "cage_qwen3": {
        "quality_case_count": 600,
        "compressed_perturbation_case_count": 480,
        "case_ids_sha256": "e57251804031e69d16d9223b7d0b4a25fdc9edc86dd9603e89daef1b3177475b",
    },
    "kitty_qwen3": {
        "quality_case_count": 120,
        "compressed_perturbation_case_count": 120,
        "case_ids_sha256": "19fae6e3d89c9c6d466a2d96fa1230ee0394beb6c6e821a20f1fb88cb77e2e9c",
    },
}


class CageV4ExecutionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV4ExecutionError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV4ExecutionError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def load_screen_execution(
    path: Path, *, repo_root: Path, verify_artifacts: bool
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    dict[str, Any] | None,
    dict[str, list[dict[str, Any]]] | None,
]:
    execution = _load(path)
    static = (
        execution.get("schema_version") == 1,
        execution.get("execution_id") == EXECUTION_ID,
        execution.get("status") == "frozen_after_joint_acceptance_before_full_screen_gpu",
        execution.get("claim_eligible") is False,
        execution.get("partitions") == EXPECTED_PARTITIONS,
        execution.get("execution_boundary") == EXPECTED_BOUNDARY,
    )
    _require(all(static), "metric screen execution freeze mismatch")
    execution_sha256 = file_sha256(path)
    protocol_path = repo_root / execution["protocol"]["path"]
    protocol, protocol_sha256 = load_metric_protocol(protocol_path)
    _require(protocol_sha256 == execution["protocol"]["sha256"], "metric protocol receipt changed")
    gate_path = repo_root / execution["acceptance_gate"]["path"]
    _require(file_sha256(gate_path) == execution["acceptance_gate"]["sha256"], "gate receipt changed")
    gate, _ = load_acceptance_gate(
        gate_path,
        repo_root=repo_root,
        protocol_sha256=protocol_sha256,
        input_manifest_sha256=execution["input_manifest"]["sha256"],
        verify_artifacts=verify_artifacts,
    )
    _require(gate["full_screen_authorization"] == EXPECTED_AUTHORIZATION, "gate authorization changed")
    if not verify_artifacts:
        return execution, execution_sha256, protocol, protocol_sha256, None, None
    preflight = execution["gate_preflight_receipt"]
    preflight_path = Path(preflight["path"])
    preflight_log = Path(preflight["log_path"])
    _require(file_sha256(preflight_path) == preflight["sha256"], "gate preflight hash mismatch")
    _require(preflight_path.stat().st_size == preflight["size_bytes"], "gate preflight size mismatch")
    _require(file_sha256(preflight_log) == preflight["log_sha256"], "gate preflight log mismatch")
    _require(preflight_log.stat().st_size == preflight["log_size_bytes"], "gate preflight log size mismatch")
    preflight_value = _load(preflight_path)
    _require(preflight_value.get("status") == "pass", "gate preflight did not pass")
    manifest_path = Path(execution["input_manifest"]["path"])
    _require(file_sha256(manifest_path) == execution["input_manifest"]["sha256"], "input manifest changed")
    manifest = _load(manifest_path)
    data_path = repo_root / "configs" / "qwen3_8b_cage_v4_pg19_data_protocol_v1.json"
    data_protocol, data_sha256 = load_data_protocol(data_path)
    _require(data_sha256 == protocol["input_receipt"]["data_protocol_sha256"], "data protocol changed")
    validate_input_manifest(manifest, protocol=data_protocol, protocol_sha256=data_sha256)
    expanded = {}
    for partition, receipt in EXPECTED_PARTITIONS.items():
        cases = expand_full_screen_cases(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            manifest=manifest,
            partition=partition,
            repo_root=repo_root,
        )
        case_ids_sha256 = canonical_sha256([case["case_id"] for case in cases])
        _require(case_ids_sha256 == receipt["case_ids_sha256"], f"screen case IDs changed: {partition}")
        preflight_partition = preflight_value["partitions"][partition]
        _require(
            preflight_partition["full_screen_case_ids_sha256"] == case_ids_sha256,
            f"preflight case-ID receipt changed: {partition}",
        )
        _require(
            preflight_partition["full_screen_case_count"] == len(cases),
            f"preflight case count changed: {partition}",
        )
        expanded[partition] = cases
    return execution, execution_sha256, protocol, protocol_sha256, manifest, expanded


__all__ = [
    "CageV4ExecutionError",
    "EXPECTED_BOUNDARY",
    "EXPECTED_PARTITIONS",
    "load_screen_execution",
]
