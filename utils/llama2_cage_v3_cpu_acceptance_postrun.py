from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v4_data import file_sha256


MANIFEST_ID = "llama2-7b-cage-v3-cpu-acceptance-artifacts-v1"


class Llama2CageV3CPUPostrunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3CPUPostrunError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3CPUPostrunError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def validate_artifact_manifest(manifest: Mapping[str, Any], *, repo_root: Path) -> None:
    _require(manifest.get("schema_version") == 1, "artifact manifest schema mismatch")
    _require(manifest.get("manifest_id") == MANIFEST_ID, "artifact manifest identity mismatch")
    _require(
        manifest.get("status") == "frozen_after_direct_acceptance_before_gpu_gate_definition",
        "artifact manifest status mismatch",
    )
    _require(manifest.get("claim_eligible") is False, "artifact claim boundary changed")
    _require(manifest.get("source_commit") == "8d45f5628180a49106a274fa4c1ce8ff1c2667b5", "source commit mismatch")
    repair = manifest.get("postrun_administrative_repair", {})
    _require(
        repair.get("failed_attempt_receipt_path") == "configs/llama2_7b_cage_v3_cpu_postrun_failed_attempt_v1.json",
        "postrun failed-attempt receipt path changed",
    )
    _require(
        repair.get("failed_attempt_receipt_sha256") == "153d0844382767b8d86bec46673ef27773b3df1e3cdb7bd6f5a5d5981f65148f",
        "postrun failed-attempt receipt hash changed",
    )
    _require(repair.get("scientific_result_changed") is False, "postrun repair changed the scientific result")
    _require(repair.get("authorization_changed") is False, "postrun repair changed authorization")
    _require(
        file_sha256(repo_root / repair["failed_attempt_receipt_path"]) == repair["failed_attempt_receipt_sha256"],
        "postrun failed-attempt receipt file changed",
    )
    protocol = manifest.get("protocol", {})
    _require(
        protocol
        == {
            "path": "configs/llama2_7b_cage_v3_cpu_acceptance_protocol_v1.json",
            "sha256": "ea5299d08d0dbceabd701fcdd063ec3ab04b8fa5ad4c0ad6573f161c0649f6f1",
        },
        "CPU protocol receipt changed",
    )
    _require(file_sha256(repo_root / protocol["path"]) == protocol["sha256"], "CPU protocol file changed")
    wrapper = manifest.get("wrapper_outcome", {})
    _require(wrapper.get("run_status") == 1 and wrapper.get("tee_status") == 0, "wrapper status changed")
    _require(wrapper.get("scientific_acceptance_completed_before_failure") is True, "acceptance ordering changed")
    _require(wrapper.get("package_check_pass") is False, "package-check failure was hidden")
    _require(wrapper.get("environment_mutation_authorized") is False, "environment mutation was authorized")
    _require(wrapper.get("rerun_scientific_acceptance_required") is False, "scientific rerun boundary changed")
    _require("outside the persisted tee log" in wrapper.get("evidence_scope", ""), "wrapper evidence scope changed")
    execution_log = manifest.get("execution_log", {})
    _require(execution_log.get("post_pipeline_status_lines_included") is False, "tee-log scope changed")
    boundary = manifest.get("postrun_boundary", {})
    _require(boundary.get("gpu_acceptance_gate_definition_authorized_if_audit_passes") is True, "gate-definition boundary changed")
    for key, value in boundary.items():
        if key != "gpu_acceptance_gate_definition_authorized_if_audit_passes":
            _require(value is False, f"postrun authorization changed: {key}")


def _validate_result(result: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    checks = result.get("checks", {})
    required = expected["required_checks"]
    _require(result.get("schema_version") == 1, "result schema mismatch")
    _require(result.get("acceptance_id") == expected["acceptance_id"], "acceptance identity mismatch")
    _require(result.get("protocol_sha256") == expected["protocol_sha256"], "result protocol mismatch")
    _require(result.get("status") == expected["status"], "CPU acceptance did not pass")
    _require(result.get("claim_eligible") is False, "result claim boundary changed")
    _require(result.get("failures") == [], "CPU acceptance contains failures")
    _require(sorted(checks) == sorted(required), "CPU acceptance check set changed")
    _require(len(checks) == expected["check_count"], "CPU acceptance check count mismatch")
    _require(all(value is True for value in checks.values()), "CPU acceptance contains a failed check")
    _require(result.get("device_used") == "cpu", "CPU acceptance used a different device")
    _require(result.get("qwen3_promotion_pass") is False, "Qwen3 promotion result changed")
    _require(result.get("qwen3_conclusion_reopened") is False, "Qwen3 conclusion was reopened")
    _require(result.get("production_model", {}).get("full_model_weights_loaded") is False, "production weights were loaded")
    boundary = result.get("execution_boundary", {})
    _require(boundary and not any(boundary.values()), "result execution boundary changed")
    _require(result.get("attention_details", {}).get("prefill_max_abs_delta") == 0.0, "prefill output was not exact")
    _require(result.get("static_preflight_checks_preserved") is True, "static preflight checks changed")
    return {
        "check_count": len(checks),
        "all_checks_pass": True,
        "device_used": result["device_used"],
        "full_model_weights_loaded": False,
        "prefill_max_abs_delta": result["attention_details"]["prefill_max_abs_delta"],
        "production_architecture": result["production_model"]["architecture"],
    }


def _validate_log(text: str, manifest: Mapping[str, Any]) -> dict[str, bool]:
    required_markers = [
        "Ran 29 tests",
        "OK\nTEST_STATUS=0",
        "ACCEPTANCE_STATUS=0",
        "AUDIT_STATUS=0",
        "=== PACKAGE CHECK ===",
    ]
    checks = {
        "nonempty": bool(text),
        "required_markers": all(marker in text for marker in required_markers),
        "no_traceback": "Traceback (most recent call last)" not in text,
        "package_failure_disclosed": all(
            line in text for line in manifest["wrapper_outcome"]["observed_unrelated_requirements"]
        ),
        "scientific_result_precedes_package_check": text.find("ACCEPTANCE_STATUS=0") < text.find("=== PACKAGE CHECK ==="),
        "post_pipeline_status_correctly_outside_log": all(
            marker not in text
            for marker in (
                "RUN_STATUS=1",
                "TEE_STATUS=0",
                "LLAMA2_CAGE_V3_CPU_ACCEPTANCE_RESULT=FAIL",
            )
        ),
    }
    _require(all(checks.values()), "execution log validation failed")
    return checks


def build_postrun_audit(manifest_path: Path, *, repo_root: Path) -> dict[str, Any]:
    manifest = _load(manifest_path)
    validate_artifact_manifest(manifest, repo_root=repo_root)
    result_spec = manifest["result"]
    log_spec = manifest["execution_log"]
    result_path = Path(result_spec["path"])
    log_path = Path(log_spec["path"])
    _require(result_path.stat().st_size == result_spec["size_bytes"], "result size mismatch")
    _require(file_sha256(result_path) == result_spec["sha256"], "result hash mismatch")
    _require(log_path.stat().st_size == log_spec["size_bytes"], "execution log size mismatch")
    _require(file_sha256(log_path) == log_spec["sha256"], "execution log hash mismatch")
    result = _load(result_path)
    result_audit = _validate_result(result, manifest["expected_acceptance"])
    log_checks = _validate_log(log_path.read_text(encoding="utf-8"), manifest)
    return {
        "schema_version": 1,
        "audit_id": "llama2-7b-cage-v3-cpu-acceptance-postrun-v1",
        "status": "pass",
        "claim_eligible": False,
        "artifact_manifest_path": str(manifest_path),
        "artifact_manifest_sha256": file_sha256(manifest_path),
        "source_commit": manifest["source_commit"],
        "result_sha256": result_spec["sha256"],
        "execution_log_sha256": log_spec["sha256"],
        "scientific_acceptance": result_audit,
        "execution_log_checks": log_checks,
        "wrapper_outcome": manifest["wrapper_outcome"],
        "decision": {
            "cpu_implementation_acceptance_pass": True,
            "broad_environment_package_check_pass": False,
            "scientific_rerun_required": False,
            "gpu_acceptance_gate_definition_authorized": True,
            "gpu_acceptance_execution_authorized": False,
        },
        "postrun_boundary": manifest["postrun_boundary"],
        "next_step": "prospectively freeze a separate Llama-2 CAGE-v3 GPU acceptance protocol before any production-weight GPU execution",
    }


__all__ = [
    "Llama2CageV3CPUPostrunError",
    "build_postrun_audit",
    "validate_artifact_manifest",
]
