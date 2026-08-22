from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from utils.llama2_cage_v3_transfer_quality_acceptance import (
    SCIENTIFIC_FIELDS,
    expand_acceptance_cases,
    lf_normalized_file_sha256,
    load_execution,
    load_server_input_manifest,
    validate_completed_case,
)
from utils.llama2_cage_v3_transfer_quality_protocol import load_transfer_quality_protocol
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


ARTIFACT_MANIFEST_ID = "llama2-7b-cage-v3-transfer-quality-acceptance-artifacts-v1"
AUDIT_ID = "llama2-7b-cage-v3-transfer-quality-acceptance-postrun-v1"
ARTIFACT_MANIFEST_SHA256 = "986673209bf09b479f26e55c952abdf355c9abfe1bc64b2d1d6b61f55e3d4de2"
EXPECTED_SCIENTIFIC_SHA256 = "bf0dcb6deb38bc1b6c7433f911b8abcc0ed819b92dac32ffe9c70f61b285b592"


class Llama2CageV3TransferQualityPostrunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3TransferQualityPostrunError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3TransferQualityPostrunError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _verify_server_file(path: Path, *, sha256: str, size_bytes: int | None = None, label: str) -> None:
    _require(path.is_file(), f"{label} is missing")
    _require(file_sha256(path) == sha256, f"{label} hash mismatch")
    if size_bytes is not None:
        _require(path.stat().st_size == size_bytes, f"{label} size mismatch")


def shell_case_manifest_sha256(root: Path) -> str:
    lines = [
        f"{file_sha256(path)}  cases/{path.name}\n"
        for path in sorted((root / "cases").glob("*.json"), key=lambda item: item.name)
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def validate_artifact_manifest(manifest: Mapping[str, Any], *, repo_root: Path) -> None:
    _require(manifest.get("schema_version") == 1, "artifact manifest schema changed")
    _require(manifest.get("artifact_manifest_id") == ARTIFACT_MANIFEST_ID, "artifact manifest identity changed")
    _require(
        manifest.get("status")
        == "frozen_after_two_fresh_bitwise_equal_acceptance_repeats_before_joint_postrun_audit",
        "artifact manifest status changed",
    )
    _require(manifest.get("claim_eligible") is False, "artifact claim boundary changed")
    _require(
        manifest.get("execution_sha256")
        == "af1ee4e339280b8c97057d02412eab5c2441acfd43cf4f631034396fcabd3fb8",
        "artifact execution changed",
    )
    _require(
        manifest.get("input_manifest_sha256")
        == "ef6b3e9c48e219ab09d35dadca4026f2cd99f47c3c922d8a8e1d2b5c4db19d04",
        "artifact input manifest changed",
    )
    gate = manifest.get("gate_receipt", {})
    _require(
        gate
        == {
            "path": "configs/llama2_7b_cage_v3_transfer_quality_acceptance_gate_receipt_v2.json",
            "sha256": "80e2dfe0e5f957b40dda28f3411132917399f0185bc3abfd1587d90b8ede24b5",
        },
        "artifact gate receipt changed",
    )
    _require(file_sha256(repo_root / gate["path"]) == gate["sha256"], "checked-in gate receipt hash mismatch")
    failed = manifest.get("failed_pre_metric_attempt", {})
    _require(
        failed.get("sha256") == "ed0995748cdea3f0a9d106349096cba1e962d16c7c19950cf1ed54e0ebcdef2e"
        and failed.get("completed_scientific_case_count") == 0
        and failed.get("token_nll_computed") is False,
        "failed pre-metric attempt receipt changed",
    )
    _require(file_sha256(repo_root / failed["path"]) == failed["sha256"], "failed attempt receipt hash mismatch")
    repeats = manifest.get("repeats", {})
    _require(set(repeats) == {"a", "b"}, "artifact repeat set changed")
    expected_source_commits = {
        "a": "137a2cbe6fdd48537d6218669c2eee7f3c5981ab",
        "b": "8d8df3411cb0f7dcb1fccfbb1ba6ac0757fc9e1f",
    }
    for repeat in ("a", "b"):
        spec = repeats[repeat]
        _require(spec.get("scientific_payload_sha256") == EXPECTED_SCIENTIFIC_SHA256, f"repeat {repeat} payload changed")
        _require(spec.get("execution_source_commit") == expected_source_commits[repeat], f"repeat {repeat} source changed")
        receipt = repo_root / spec["artifact_receipt_path"]
        _require(lf_normalized_file_sha256(receipt) == spec["artifact_receipt_sha256"], f"repeat {repeat} artifact receipt changed")
    comparison = manifest.get("comparison", {})
    _require(
        comparison.get("sha256") == "ee92895593bd6f3ae342982c152ed2957d93712a2a7519497a3de35cf9c04177"
        and comparison.get("size_bytes") == 565
        and comparison.get("scientific_payload_sha256") == EXPECTED_SCIENTIFIC_SHA256
        and comparison.get("comparator_sha256") == "585f7875e33b8ec38afbd332f781e0f5621916a87981bc5bb567dd9339ac6a5b",
        "comparison receipt changed",
    )
    package = manifest.get("environment_package_check", {})
    _require(
        package.get("status") == "known_dependency_metadata_incompatibilities_reported_after_successful_comparison"
        and package.get("scientific_comparator_affected") is False
        and package.get("environment_mutation_authorized") is False
        and package.get("full_execution_gate_must_revalidate_required_imports") is True
        and len(package.get("reported_lines", [])) == 4,
        "package-check disclosure changed",
    )
    _require(
        manifest.get("authorization_before_joint_postrun")
        == {
            "joint_postrun_audit": True,
            "full_600_case_execution": False,
            "candidate_tuning": False,
            "quality_interpretation": False,
            "paper_claims": False,
            "runtime_claims": False,
        },
        "pre-postrun authorization changed",
    )


def _validate_log(spec: Mapping[str, Any], *, repeat: str) -> dict[str, bool]:
    path = Path(spec["execution_log"])
    _verify_server_file(
        path,
        sha256=spec["execution_log_sha256"],
        size_bytes=spec["execution_log_size_bytes"],
        label=f"repeat {repeat} execution log",
    )
    text = path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "nonempty": bool(text.strip()),
        "no_traceback": "Traceback (most recent call last)" not in text,
        "no_explicit_execution_failure": "Repeat A execution failed" not in text and "Repeat B execution failed" not in text,
        "start_marker": spec["start_marker"] in text,
        "end_marker": spec["end_marker"] in text,
    }
    _require(all(checks.values()), f"repeat {repeat} execution log checks failed")
    return checks


def _validate_repeat(
    spec: Mapping[str, Any],
    *,
    repeat: str,
    expected_cases: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(spec["output_dir"])
    identity_path = root / "run_identity.json"
    summary_path = root / "summary.json"
    _verify_server_file(identity_path, sha256=spec["run_identity_sha256"], label=f"repeat {repeat} run identity")
    _verify_server_file(summary_path, sha256=spec["summary_sha256"], label=f"repeat {repeat} summary")
    identity = _load(identity_path)
    summary = _load(summary_path)
    _require(
        summary.get("schema_version") == 1
        and summary.get("status") == "pass"
        and summary.get("claim_eligible") is False
        and summary.get("repeat") == repeat
        and summary.get("expected_cases") == 12
        and summary.get("completed_cases") == 12
        and summary.get("failure_records") == 0
        and summary.get("new_cases") == 12
        and summary.get("resumed_cases") == 0
        and summary.get("scientific_payload_sha256") == EXPECTED_SCIENTIFIC_SHA256,
        f"repeat {repeat} summary checks failed",
    )
    _require(
        summary.get("determinism")
        == {
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
        f"repeat {repeat} determinism changed",
    )
    boundary = summary.get("execution_boundary", {})
    _require(boundary.get("acceptance_only") is True, f"repeat {repeat} acceptance boundary changed")
    for key in ("full_600_case_execution_authorized", "candidate_tuning_authorized", "paper_claims_authorized", "runtime_claims_authorized"):
        _require(boundary.get(key) is False, f"repeat {repeat} expanded authorization: {key}")
    _require(identity.get("source_state") == {"git_commit": spec["execution_source_commit"], "dirty": False}, f"repeat {repeat} source state changed")
    _require(identity.get("expected_case_ids") == [case["case_id"] for case in expected_cases], f"repeat {repeat} expected IDs changed")
    failure_files = sorted((root / "failures").glob("*.json")) if (root / "failures").exists() else []
    _require(not failure_files, f"repeat {repeat} contains failure files")
    records = []
    for case in expected_cases:
        path = root / "cases" / f"{case['case_id']}.json"
        record = _load(path)
        validate_completed_case(record, case)
        provenance = record.get("provenance", {})
        _require(
            provenance.get("repeat") == repeat
            and provenance.get("source_state") == identity["source_state"]
            and provenance.get("cublas_workspace_config") == ":4096:8",
            f"repeat {repeat} case provenance changed",
        )
        records.append(record)
    payload = [{field: record[field] for field in SCIENTIFIC_FIELDS} for record in records]
    payload_sha = canonical_sha256(payload)
    _require(payload_sha == EXPECTED_SCIENTIFIC_SHA256, f"repeat {repeat} scientific payload changed")
    _require(shell_case_manifest_sha256(root) == spec["case_file_hash_manifest_sha256"], f"repeat {repeat} case manifest changed")
    return {
        "output_dir": str(root),
        "case_count": 12,
        "failure_count": 0,
        "run_identity_sha256": spec["run_identity_sha256"],
        "summary_sha256": spec["summary_sha256"],
        "case_file_hash_manifest_sha256": spec["case_file_hash_manifest_sha256"],
        "scientific_payload_sha256": payload_sha,
        "execution_source_commit": spec["execution_source_commit"],
        "execution_log": spec["execution_log"],
        "execution_log_sha256": spec["execution_log_sha256"],
        "execution_log_checks": _validate_log(spec, repeat=repeat),
    }, payload


def _validate_comparison(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"])
    log = Path(spec["execution_log"])
    _verify_server_file(path, sha256=spec["sha256"], size_bytes=spec["size_bytes"], label="acceptance comparison")
    _verify_server_file(
        log,
        sha256=spec["execution_log_sha256"],
        size_bytes=spec["execution_log_size_bytes"],
        label="acceptance comparison log",
    )
    report = _load(path)
    _require(
        report.get("schema_version") == 1
        and report.get("comparison_id") == "llama2-7b-cage-v3-transfer-quality-acceptance-repeat-comparison-v1"
        and report.get("status") == "pass"
        and report.get("claim_eligible") is False
        and report.get("case_count") == 12
        and report.get("required_consistency") == "bitwise_equal_json_scientific_payload"
        and report.get("scientific_fields") == list(SCIENTIFIC_FIELDS)
        and report.get("mismatch_case_ids") == []
        and report.get("scientific_payload_sha256") == EXPECTED_SCIENTIFIC_SHA256
        and report.get("telemetry_excluded") is True
        and report.get("full_600_case_execution_authorized") is False,
        "acceptance comparison payload changed",
    )
    text = log.read_text(encoding="utf-8", errors="replace")
    checks = {
        "nonempty": bool(text.strip()),
        "no_traceback": "Traceback (most recent call last)" not in text,
        "start_marker": "=== LLAMA2 CAGE-V3 TRANSFER-QUALITY ACCEPTANCE A-B COMPARISON START ===" in text,
        "end_marker": "=== LLAMA2 CAGE-V3 TRANSFER-QUALITY ACCEPTANCE A-B COMPARISON END ===" in text,
    }
    _require(all(checks.values()), "acceptance comparison log checks failed")
    return {
        "path": str(path),
        "sha256": spec["sha256"],
        "size_bytes": spec["size_bytes"],
        "execution_log": str(log),
        "execution_log_sha256": spec["execution_log_sha256"],
        "execution_log_checks": checks,
        "scientific_payload_sha256": EXPECTED_SCIENTIFIC_SHA256,
        "mismatch_case_ids": [],
    }


def build_postrun_audit(manifest_path: str | Path, *, repo_root: Path) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = _load(source)
    _require(lf_normalized_file_sha256(source) == ARTIFACT_MANIFEST_SHA256, "artifact manifest file hash changed")
    validate_artifact_manifest(manifest, repo_root=repo_root)
    execution_path = repo_root / "configs" / "llama2_7b_cage_v3_transfer_quality_acceptance_execution_v1.json"
    execution, execution_sha = load_execution(execution_path, repo_root=repo_root)
    _require(execution_sha == manifest["execution_sha256"], "postrun execution linkage changed")
    protocol, _ = load_transfer_quality_protocol(repo_root / execution["protocol"]["path"], repo_root=repo_root)
    input_manifest = load_server_input_manifest(execution, protocol=protocol)
    expected_cases = expand_acceptance_cases(
        execution=execution,
        execution_sha256=execution_sha,
        protocol=protocol,
        input_manifest=input_manifest,
    )
    repeats = {}
    payloads = {}
    for repeat in ("a", "b"):
        repeats[repeat], payloads[repeat] = _validate_repeat(
            manifest["repeats"][repeat], repeat=repeat, expected_cases=expected_cases
        )
    _require(payloads["a"] == payloads["b"], "acceptance repeat scientific payloads differ")
    comparison_spec = dict(manifest["comparison"])
    comparison_spec["reported_package_lines"] = manifest["environment_package_check"]["reported_lines"]
    comparison = _validate_comparison(comparison_spec)
    comparison_log_text = Path(comparison_spec["execution_log"]).read_text(encoding="utf-8", errors="replace")
    _require(
        all(line in comparison_log_text for line in manifest["environment_package_check"]["reported_lines"]),
        "reported package-check disclosure does not match comparison log",
    )
    return {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "status": "pass",
        "claim_eligible": False,
        "interpretation_performed": False,
        "artifact_manifest_path": str(source),
        "artifact_manifest_sha256": ARTIFACT_MANIFEST_SHA256,
        "execution_sha256": execution_sha,
        "input_manifest_sha256": manifest["input_manifest_sha256"],
        "failed_pre_metric_attempt_count": 1,
        "case_count_per_repeat": 12,
        "repeat_count": 2,
        "total_case_file_count": 24,
        "failure_count": 0,
        "scientific_payload_sha256": EXPECTED_SCIENTIFIC_SHA256,
        "repeat_payloads_bitwise_equal": True,
        "repeats": repeats,
        "comparison": comparison,
        "environment_package_check": manifest["environment_package_check"],
        "full_execution_gate_candidate": {
            "status": "eligible_for_separate_checked_in_gate_review",
            "case_count": 600,
            "required_runtime_import_revalidation": True,
            "full_600_case_execution_authorized_by_this_audit": False,
        },
        "execution_boundary": {
            "acceptance_complete": True,
            "acceptance_repeats_bitwise_equal": True,
            "full_600_case_execution": False,
            "candidate_tuning": False,
            "quality_interpretation": False,
            "paper_claims": False,
            "runtime_claims": False,
        },
    }


__all__ = [
    "ARTIFACT_MANIFEST_ID",
    "AUDIT_ID",
    "Llama2CageV3TransferQualityPostrunError",
    "build_postrun_audit",
    "shell_case_manifest_sha256",
    "validate_artifact_manifest",
]
