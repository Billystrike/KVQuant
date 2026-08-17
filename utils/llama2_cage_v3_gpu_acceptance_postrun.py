from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from utils.llama2_cage_v3_gpu_execution import load_object, require
from utils.qwen3_cage_v4_data import file_sha256


EXPECTED_MANIFEST_SHA256 = "fdac85087799391a37204a05e5632dfb5a16bda820fb468ef396732033be8bc9"


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_file(spec: Mapping[str, Any]) -> Path:
    path = Path(spec["path"])
    require(path.is_file(), f"artifact is missing: {path}")
    require(path.stat().st_size == spec["size_bytes"], f"artifact size mismatch: {path}")
    require(file_sha256(path) == spec["sha256"], f"artifact hash mismatch: {path}")
    return path


def _validate_log(spec: Mapping[str, Any], *, required_markers: tuple[str, ...]) -> dict[str, bool]:
    path = _validate_file(spec)
    text = path.read_text(encoding="utf-8")
    checks = {
        "nonempty": bool(text),
        "required_markers": all(marker in text for marker in required_markers),
        "no_traceback": "Traceback (most recent call last)" not in text,
        "no_conda_error": "ERROR conda.cli" not in text,
        "no_explicit_error": "ERROR:" not in text,
    }
    require(all(checks.values()), f"execution log checks failed: {path}")
    return checks


def _validate_repeat(
    spec: Mapping[str, Any],
    *,
    repeat: str,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    result_path = _validate_file(spec["result"])
    result = load_object(result_path)
    require(result.get("schema_version") == 1, f"repeat {repeat} schema mismatch")
    require(result.get("acceptance_id") == "llama2-7b-cage-v3-production-gpu-acceptance-v1", f"repeat {repeat} identity mismatch")
    require(result.get("status") == "pass", f"repeat {repeat} did not pass")
    require(result.get("claim_eligible") is False, f"repeat {repeat} claim boundary changed")
    require(result.get("repeat") == repeat, f"repeat {repeat} label mismatch")
    require(result.get("protocol_sha256") == manifest["protocol_sha256"], f"repeat {repeat} protocol mismatch")
    require(result.get("preflight_receipt_sha256") == manifest["preflight_receipt_sha256"], f"repeat {repeat} preflight mismatch")
    require(result.get("gate_receipt_sha256") == manifest["gate_receipt_sha256"], f"repeat {repeat} gate mismatch")
    require(len(result.get("weight_receipts", [])) == 2, f"repeat {repeat} weight receipt count mismatch")
    science = result.get("scientific_payload", {})
    require(_canonical_sha256(science) == manifest["expected_scientific_payload_sha256"], f"repeat {repeat} scientific hash mismatch")
    require(science.get("failures") == [], f"repeat {repeat} contains scientific failures")
    require(all(science.get("checks", {}).values()), f"repeat {repeat} contains failed checks")
    require(len(science.get("cache_records", [])) == 32, f"repeat {repeat} cache layer count mismatch")
    require(science.get("quota_counts") == {"16": 11, "32": 10, "48": 11}, f"repeat {repeat} quota mismatch")
    require(science.get("reported_cache_lengths") == [1024, 1025, 1026], f"repeat {repeat} cache lengths mismatch")
    require(science.get("parameter_devices") == ["cuda:0"], f"repeat {repeat} device mismatch")
    require(science.get("parameter_dtypes") == ["torch.float16"], f"repeat {repeat} dtype mismatch")
    require(result.get("execution_boundary") == manifest["preserved_boundary"], f"repeat {repeat} boundary mismatch")
    telemetry = result.get("telemetry", {})
    require(
        set(telemetry) == {"model_load_seconds", "acceptance_run_seconds", "peak_cuda_allocator_bytes"},
        f"repeat {repeat} telemetry fields changed",
    )
    require(
        all(isinstance(telemetry.get(key), (int, float)) and math.isfinite(telemetry[key]) and telemetry[key] >= 0 for key in telemetry),
        f"repeat {repeat} telemetry is invalid",
    )
    upper = repeat.upper()
    log_checks = _validate_log(
        spec["execution_log"],
        required_markers=(
            f"=== LLAMA2 CAGE-V3 GPU ACCEPTANCE {upper} START ===",
            f"LLAMA2_CAGE_V3_GPU_ACCEPTANCE_{upper}_RESULT=PASS",
            "AUDIT_STATUS=0",
            "RUN_STATUS=0",
            "TEE_STATUS=0",
        ),
    )
    return result, science, log_checks


def build_postrun_audit(manifest_path: Path) -> dict[str, Any]:
    require(file_sha256(manifest_path) == EXPECTED_MANIFEST_SHA256, "GPU acceptance artifact manifest hash mismatch")
    manifest = load_object(manifest_path)
    require(manifest.get("schema_version") == 1, "artifact manifest schema mismatch")
    require(manifest.get("manifest_id") == "llama2-7b-cage-v3-production-gpu-acceptance-artifacts-v1", "artifact manifest identity mismatch")
    require(
        manifest.get("status") == "frozen_after_two_repeat_comparison_before_postrun_interpretation",
        "artifact manifest status mismatch",
    )
    require(manifest.get("claim_eligible") is False, "artifact manifest claim boundary changed")
    boundary = manifest.get("preserved_boundary", {})
    require(boundary.get("synthetic_acceptance_only") is True, "synthetic acceptance boundary changed")
    require(not any(value for key, value in boundary.items() if key != "synthetic_acceptance_only"), "artifact boundary expanded")

    repeat_a, science_a, log_a = _validate_repeat(manifest["repeat_a"], repeat="a", manifest=manifest)
    repeat_b, science_b, log_b = _validate_repeat(manifest["repeat_b"], repeat="b", manifest=manifest)
    require(science_a == science_b, "repeat scientific payloads are not bitwise-equal JSON values")

    comparison_path = _validate_file(manifest["comparison"]["result"])
    comparison = load_object(comparison_path)
    require(comparison.get("schema_version") == 1, "comparison schema mismatch")
    require(comparison.get("status") == "pass", "comparison did not pass")
    require(comparison.get("claim_eligible") is False, "comparison claim boundary changed")
    require(comparison.get("scientific_payload_equal") is True, "comparison equality failed")
    require(comparison.get("telemetry_excluded") is True, "comparison telemetry boundary changed")
    require(comparison.get("formal_transfer_authorized") is False, "comparison expanded transfer authorization")
    require(Path(comparison["repeat_a"]) == Path(manifest["repeat_a"]["result"]["path"]), "comparison repeat A path mismatch")
    require(Path(comparison["repeat_b"]) == Path(manifest["repeat_b"]["result"]["path"]), "comparison repeat B path mismatch")
    log_comparison = _validate_log(
        manifest["comparison"]["execution_log"],
        required_markers=(
            "=== LLAMA2 CAGE-V3 GPU ACCEPTANCE A-B COMPARISON START ===",
            "LLAMA2_CAGE_V3_GPU_ACCEPTANCE_AB_COMPARISON_RESULT=PASS",
            "COMPARATOR_STATUS=0",
            "AUDIT_STATUS=0",
            "RUN_STATUS=0",
            "TEE_STATUS=0",
        ),
    )

    return {
        "schema_version": 1,
        "audit_id": "llama2-7b-cage-v3-production-gpu-acceptance-postrun-v1",
        "status": "pass",
        "claim_eligible": False,
        "artifact_manifest_path": str(manifest_path),
        "artifact_manifest_sha256": file_sha256(manifest_path),
        "execution_source_commit": manifest["execution_source_commit"],
        "protocol_sha256": manifest["protocol_sha256"],
        "preflight_receipt_sha256": manifest["preflight_receipt_sha256"],
        "gate_receipt_sha256": manifest["gate_receipt_sha256"],
        "scientific_payload_sha256": manifest["expected_scientific_payload_sha256"],
        "repeat_results": {
            "a": {
                "result_sha256": manifest["repeat_a"]["result"]["sha256"],
                "log_sha256": manifest["repeat_a"]["execution_log"]["sha256"],
                "log_checks": log_a,
                "telemetry": repeat_a["telemetry"],
            },
            "b": {
                "result_sha256": manifest["repeat_b"]["result"]["sha256"],
                "log_sha256": manifest["repeat_b"]["execution_log"]["sha256"],
                "log_checks": log_b,
                "telemetry": repeat_b["telemetry"],
            },
        },
        "comparison": {
            "result_sha256": manifest["comparison"]["result"]["sha256"],
            "log_sha256": manifest["comparison"]["execution_log"]["sha256"],
            "scientific_payload_equal": True,
            "telemetry_excluded": True,
            "log_checks": log_comparison,
        },
        "decision": {
            "production_gpu_acceptance_pass": True,
            "quality_protocol_design_authorized": True,
            "formal_transfer_execution_authorized": False,
            "corpus_access_authorized": False,
            "quality_metric_execution_authorized": False,
            "runtime_claims_authorized": False,
        },
        "next_step": "freeze a separate Llama-2 CAGE-v3 transfer-quality protocol before any corpus access or quality execution",
    }


__all__ = ["EXPECTED_MANIFEST_SHA256", "build_postrun_audit"]
