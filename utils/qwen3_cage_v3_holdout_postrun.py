from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.qwen3_cage_v3_execution import canonical_sha256, load_json
from utils.qwen3_cage_v3_postrun import (
    _validate_cache,
    canonical_case_manifest_sha256,
    shell_case_manifest_sha256,
)
from utils.qwen3_cage_v3_protocol import file_sha256
from utils.qwen3_cage_v3_screen import SCIENTIFIC_FIELDS
from utils.qwen3_perturbation_protocol import (
    aggregate_layer_metrics,
    validate_aggregates,
    validate_layer_records,
)


ARTIFACT_SET_ID = "qwen3-8b-cage-v3-development-holdout-full-artifacts-v1"
PARTITIONS = ("cage_qwen3", "kitty_qwen3")
EXPECTED_COUNTS = {"cage_qwen3": 60, "kitty_qwen3": 30}
EXPECTED_LAYER_COUNTS = {partition: count * 36 for partition, count in EXPECTED_COUNTS.items()}
EXPECTED_KITTY_COMMIT = "dfd2c07b407d6b407179359207c612ab631f3ed1"
EXPECTED_TRANSFORMERS_COMMIT = "37f8b0b53512e6aae0cfd15746c133c101783178"
SELECTED_FAMILY_ID = "pure-sr2-sink32-calibrated"


class CageV3HoldoutPostrunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3HoldoutPostrunError(message)


def validate_artifact_manifest(
    manifest: dict[str, Any],
    *,
    execution_sha256: str,
    protocol_sha256: str,
    input_manifest_sha256: str,
    quota_plan_sha256: str,
    screen_decision_sha256: str,
    gate_sha256: str,
) -> None:
    _require(manifest.get("schema_version") == 1, "holdout artifact schema mismatch")
    _require(manifest.get("artifact_set_id") == ARTIFACT_SET_ID, "holdout artifact set ID mismatch")
    _require(
        manifest.get("status") == "declared_from_holdout_outputs_before_joint_postrun_audit",
        "holdout artifact status mismatch",
    )
    _require(manifest.get("claim_eligible") is False, "holdout artifacts must be claim-ineligible")
    _require(manifest.get("interpretation_performed") is False, "holdout artifacts were interpreted before audit")
    for field, expected in (
        ("execution_sha256", execution_sha256),
        ("protocol_sha256", protocol_sha256),
        ("manifest_sha256", input_manifest_sha256),
        ("quota_plan_sha256", quota_plan_sha256),
        ("screen_decision_sha256", screen_decision_sha256),
        ("acceptance_gate_sha256", gate_sha256),
    ):
        _require(manifest.get(field) == expected, f"holdout artifact {field} mismatch")
    source_commit = manifest.get("execution_source_commit")
    _require(isinstance(source_commit, str) and len(source_commit) == 40, "holdout source commit is invalid")
    _require(tuple(manifest.get("partitions", {})) == PARTITIONS, "holdout artifact partitions mismatch")
    for partition in PARTITIONS:
        record = manifest["partitions"][partition]
        _require(record.get("case_count") == EXPECTED_COUNTS[partition], f"{partition} case count mismatch")
        _require(record.get("layer_record_count") == EXPECTED_LAYER_COUNTS[partition], f"{partition} layer count mismatch")
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
    _require(
        manifest.get("joint_expected")
        == {
            "case_count": 90,
            "layer_record_count": 3240,
            "failure_count": 0,
            "interpretation_allowed_before_audit_pass": False,
            "holdout_metrics_consumed": True,
            "reserved_unseen_metrics_consumed": False,
            "end_to_end_quality_authorized": False,
        },
        "joint holdout expectations mismatch",
    )


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
    screen_decision_sha256: str,
) -> dict[str, Any]:
    _require(partition in PARTITIONS, "unsupported holdout postrun partition")
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
        ("stage", "holdout_full"),
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
    _require(identity.get("screen_decision_sha256") == screen_decision_sha256, f"{partition} decision mismatch")
    _require(identity.get("selected_family_id") == SELECTED_FAMILY_ID, f"{partition} selected family mismatch")
    _require(identity.get("experiment") == "qwen3_cage_v3_holdout_development_only", "holdout experiment identity mismatch")
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
    failures = sorted((root / "failures").glob("*.json")) if (root / "failures").exists() else []
    _require(not failures, f"{partition} failure files remain")
    _require(shell_case_manifest_sha256(paths) == spec["shell_case_file_manifest_sha256"], f"{partition} shell case manifest mismatch")
    scientific_payload = []
    layer_count = 0
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
        _require(record["input"]["anchor_index"] in range(10, 20), f"protected anchor entered holdout: {case_id}")
        family = record["method"]["family_id"]
        allowed = {"kitty-pro-25pct"} if partition == "kitty_qwen3" else {"cage-v2-best-control", SELECTED_FAMILY_ID}
        _require(family in allowed, f"unregistered holdout family: {family}")
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
    "CageV3HoldoutPostrunError",
    "validate_artifact_manifest",
    "validate_partition",
]
