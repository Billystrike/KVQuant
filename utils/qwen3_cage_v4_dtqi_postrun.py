from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256
from utils.qwen3_cage_v4_dtqi_acceptance import SCIENTIFIC_FIELDS, load_json


MANIFEST_ID = "qwen3-8b-cage-v4-dtqi-gpu-acceptance-artifacts-v1"
AUDIT_ID = "qwen3-8b-cage-v4-dtqi-gpu-acceptance-postrun-v1"


class CageV4DTQIPostrunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV4DTQIPostrunError(message)


def validate_artifact_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema_version") == 1, "artifact manifest schema mismatch")
    _require(manifest.get("manifest_id") == MANIFEST_ID, "artifact manifest identity mismatch")
    _require(
        manifest.get("status")
        == "declared_from_two_completed_gpu_acceptance_repeats_before_postrun_audit",
        "artifact manifest status mismatch",
    )
    _require(manifest.get("claim_eligible") is False, "artifact claim boundary changed")
    _require(
        manifest.get("artifact_source_commit")
        == "5a2cceb95e8975e86246f63a8c555e1da2ce68cb",
        "artifact source commit changed",
    )
    _require(
        manifest.get("execution_sha256")
        == "da87398a5870aa5bc49321995901f85e635f558a3c06551bdcd76c316bd8b972",
        "artifact execution hash changed",
    )
    expected_ids = [
        "7a6850c57f3f9f1965cb0f0e",
        "6844fac8cfc19bc9be5f43f8",
        "0543d4c8d6a20c1add196159",
    ]
    _require(manifest.get("expected_case_ids") == expected_ids, "artifact case IDs changed")
    _require(
        manifest.get("scientific_fields") == list(SCIENTIFIC_FIELDS),
        "artifact scientific fields changed",
    )
    _require(set(manifest.get("repeats", {})) == {"a", "b"}, "artifact repeats changed")
    _require(
        manifest["repeats"]
        == {
            "a": {
                "root": "/root/autodl-tmp/qwen3_cage_v4/dtqi_gpu_acceptance_a_5a2cceb",
                "run_identity_sha256": "ee831268a0e82818cf249a64b6bfbd091456ed9d433e253c084a733d6fb7118b",
                "summary_sha256": "61fde69ded44ff44072d3a4a93c3915c1f459f376e993efc6baf7da084be5f4f",
                "execution_log": "/root/autodl-tmp/kitty_setup_audit/qwen3_cage_v4_dtqi_gpu_acceptance_a_5a2cceb_20260813T212652.log",
                "execution_log_sha256": "c26ca9bd9a1210a2d1d3353e0d8140bacafc5bc61f22c2d17f683de34108589b",
                "execution_log_size_bytes": 12319,
            },
            "b": {
                "root": "/root/autodl-tmp/qwen3_cage_v4/dtqi_gpu_acceptance_b_5a2cceb",
                "run_identity_sha256": "ee831268a0e82818cf249a64b6bfbd091456ed9d433e253c084a733d6fb7118b",
                "summary_sha256": "61fde69ded44ff44072d3a4a93c3915c1f459f376e993efc6baf7da084be5f4f",
                "execution_log": "/root/autodl-tmp/kitty_setup_audit/qwen3_cage_v4_dtqi_gpu_acceptance_b_5a2cceb_20260813T212917.log",
                "execution_log_sha256": "b963484fdc91c4ed0ed286e489ab1905a4297c11cc48bcc477d43ba8c983d24b",
                "execution_log_size_bytes": 10959,
            },
        },
        "artifact repeat evidence changed",
    )
    _require(
        manifest.get("comparison")
        == {
            "path": "/root/autodl-tmp/kitty_setup_audit/qwen3_cage_v4_dtqi_gpu_acceptance_comparison_5a2cceb.json",
            "sha256": "a5d2630d285ce8c0bea5ba3a2d143550ff3446740469c5e718b8087c4a506021",
            "size_bytes": 787,
            "scientific_payload_sha256": "e94f84b8446ab5eda22ef4e0fe9c637cbe69cc5caf7ca65b1a311a529ba2cb73",
            "comparator_sha256": "3a5206ff5340f40533d23651d9d45cf745fcdde6acd351f51e67c4001851bb38",
        },
        "artifact comparison evidence changed",
    )
    _require(
        manifest.get("execution_boundary")
        == {
            "gpu_acceptance_completed": True,
            "gpu_full_screen_authorized": False,
            "holdout_method_metrics_authorized": False,
            "pg19_test_access_authorized": False,
            "interpretation_authorized": False,
            "runtime_claims_authorized": False,
            "paper_claims_authorized": False,
        },
        "artifact execution boundary changed",
    )


def _validate_log(path: Path, spec: Mapping[str, Any], repeat: str) -> dict[str, Any]:
    _require(path.is_file(), f"repeat {repeat} execution log is missing")
    _require(file_sha256(path) == spec["execution_log_sha256"], f"repeat {repeat} log hash mismatch")
    _require(path.stat().st_size == spec["execution_log_size_bytes"], f"repeat {repeat} log size mismatch")
    text = path.read_text(encoding="utf-8", errors="replace")
    expected_result = (
        "DTQI_GPU_ACCEPTANCE_A_RESULT=PASS"
        if repeat == "a"
        else "DTQI_GPU_ACCEPTANCE_B_AND_COMPARISON_RESULT=PASS"
    )
    checks = {
        "nonempty": bool(text.strip()),
        "no_traceback": "Traceback (most recent call last)" not in text,
        "no_conda_error": "ERROR conda.cli.main_run" not in text,
        "run_status_zero": "RUN_STATUS=0" in text,
        "tee_status_zero": "TEE_STATUS=0" in text,
        "pass_marker": expected_result in text,
    }
    _require(all(checks.values()), f"repeat {repeat} execution log checks failed")
    return checks


def validate_repeat(spec: Mapping[str, Any], *, repeat: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(spec["root"])
    identity_path = root / "run_identity.json"
    summary_path = root / "summary.json"
    _require(identity_path.is_file() and summary_path.is_file(), f"repeat {repeat} identity/summary missing")
    _require(file_sha256(identity_path) == spec["run_identity_sha256"], f"repeat {repeat} identity hash mismatch")
    _require(file_sha256(summary_path) == spec["summary_sha256"], f"repeat {repeat} summary hash mismatch")
    identity = load_json(identity_path)
    summary = load_json(summary_path)
    _require(summary.get("identity") == identity, f"repeat {repeat} identity linkage mismatch")
    expected_summary = (
        summary.get("schema_version") == 1,
        summary.get("status") == "pass",
        summary.get("claim_eligible") is False,
        summary.get("stage") == "gpu_acceptance",
        summary.get("expected_cases") == 3,
        summary.get("completed_cases") == 3,
        summary.get("new_cases") == 3,
        summary.get("resumed_cases") == 0,
        summary.get("failure_records") == 0,
        summary.get("scientific_fields") == list(SCIENTIFIC_FIELDS),
        summary.get("full_screen_authorized") is False,
        summary.get("holdout_accessed") is False,
        summary.get("pg19_test_accessed") is False,
    )
    _require(all(expected_summary), f"repeat {repeat} summary checks failed")
    failure_files = sorted((root / "failures").glob("*.json")) if (root / "failures").exists() else []
    _require(not failure_files, f"repeat {repeat} contains failure files")
    case_files = sorted((root / "cases").glob("*.json"))
    _require(len(case_files) == 3, f"repeat {repeat} case count mismatch")
    case_manifest = []
    scientific = []
    expected_ids = set(identity["expected_case_ids"])
    for path in case_files:
        record = load_json(path)
        _require(record.get("status") == "completed", f"repeat {repeat} incomplete case")
        _require(record.get("case_id") == path.stem, f"repeat {repeat} case filename mismatch")
        _require(record.get("identity") == identity, f"repeat {repeat} case identity mismatch")
        _require(record.get("model") == summary["model"], f"repeat {repeat} case model mismatch")
        _require(record["case_id"] in expected_ids, f"repeat {repeat} unexpected case ID")
        case_manifest.append(
            {
                "path": f"cases/{path.name}",
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
        scientific.append({field: record[field] for field in SCIENTIFIC_FIELDS})
    _require({row["case_id"] for row in scientific} == expected_ids, f"repeat {repeat} case ID set mismatch")
    scientific.sort(key=lambda row: row["case_id"])
    return {
        "root": str(root),
        "case_count": 3,
        "failure_count": 0,
        "run_identity_sha256": file_sha256(identity_path),
        "summary_sha256": file_sha256(summary_path),
        "case_file_manifest": case_manifest,
        "case_file_manifest_sha256": canonical_sha256(case_manifest),
        "scientific_payload_sha256": canonical_sha256(scientific),
        "execution_log": spec["execution_log"],
        "execution_log_sha256": spec["execution_log_sha256"],
        "execution_log_size_bytes": spec["execution_log_size_bytes"],
        "execution_log_checks": _validate_log(Path(spec["execution_log"]), spec, repeat),
    }, scientific


def build_postrun_audit(manifest_path: str | Path) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = load_json(source)
    validate_artifact_manifest(manifest)
    repeats = {}
    payloads = {}
    for name in ("a", "b"):
        repeats[name], payloads[name] = validate_repeat(manifest["repeats"][name], repeat=name)
    _require(payloads["a"] == payloads["b"], "repeat scientific payloads differ")
    comparison_spec = manifest["comparison"]
    comparison_path = Path(comparison_spec["path"])
    _require(comparison_path.is_file(), "acceptance comparison is missing")
    _require(file_sha256(comparison_path) == comparison_spec["sha256"], "comparison hash mismatch")
    _require(comparison_path.stat().st_size == comparison_spec["size_bytes"], "comparison size mismatch")
    comparison = load_json(comparison_path)
    _require(comparison.get("status") == "pass", "comparison status mismatch")
    _require(comparison.get("mismatch_case_ids") == [], "comparison contains mismatches")
    _require(comparison.get("case_count") == 3, "comparison case count mismatch")
    _require(comparison.get("fields") == list(SCIENTIFIC_FIELDS), "comparison fields mismatch")
    _require(
        comparison.get("required_consistency") == "bitwise_equal_json_numeric_payload",
        "comparison consistency policy mismatch",
    )
    _require(
        comparison.get("scientific_payload_sha256")
        == comparison_spec["scientific_payload_sha256"]
        == repeats["a"]["scientific_payload_sha256"]
        == repeats["b"]["scientific_payload_sha256"],
        "comparison scientific payload linkage mismatch",
    )
    _require(comparison.get("comparator_sha256") == comparison_spec["comparator_sha256"], "comparator hash mismatch")
    return {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "status": "pass",
        "claim_eligible": False,
        "interpretation_performed": False,
        "artifact_manifest_path": str(source),
        "artifact_manifest_sha256": file_sha256(source),
        "artifact_source_commit": manifest["artifact_source_commit"],
        "execution_sha256": manifest["execution_sha256"],
        "repeat_count": 2,
        "case_count_per_repeat": 3,
        "total_case_file_count": 6,
        "failure_count": 0,
        "scientific_fields": list(SCIENTIFIC_FIELDS),
        "scientific_payload_sha256": repeats["a"]["scientific_payload_sha256"],
        "repeat_payloads_bitwise_equal": True,
        "comparison": {
            "path": str(comparison_path),
            "sha256": file_sha256(comparison_path),
            "size_bytes": comparison_path.stat().st_size,
            "status": comparison["status"],
            "mismatch_case_ids": comparison["mismatch_case_ids"],
            "comparator_sha256": comparison["comparator_sha256"],
        },
        "repeats": repeats,
        "execution_boundary": manifest["execution_boundary"],
    }


__all__ = [
    "AUDIT_ID",
    "CageV4DTQIPostrunError",
    "MANIFEST_ID",
    "build_postrun_audit",
    "validate_artifact_manifest",
    "validate_repeat",
]
