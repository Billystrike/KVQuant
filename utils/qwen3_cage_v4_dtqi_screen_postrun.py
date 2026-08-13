from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256
from utils.qwen3_cage_v4_dtqi_acceptance import load_json
from utils.qwen3_cage_v4_dtqi_screen import SCIENTIFIC_FIELDS
from utils.qwen3_formal import formal_scoring


MANIFEST_ID = "qwen3-8b-cage-v4-dtqi-pg19-screen-artifacts-v1"
AUDIT_ID = "qwen3-8b-cage-v4-dtqi-pg19-screen-postrun-v1"


class CageV4DTQIScreenPostrunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV4DTQIScreenPostrunError(message)


def validate_artifact_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema_version") == 1, "screen artifact schema mismatch")
    _require(manifest.get("manifest_id") == MANIFEST_ID, "screen artifact identity mismatch")
    _require(manifest.get("status") == "declared_from_completed_screen_before_postrun_audit", "screen artifact status mismatch")
    _require(manifest.get("claim_eligible") is False, "screen artifact claim boundary changed")
    _require(manifest.get("execution_source_commit") == "b0c81baa6536b55787a1ded052aa55f720e5c71c", "screen source commit changed")
    _require(manifest.get("execution_sha256") == "6b8daeaea2b65efb73cc79fed3ffe8e8882ae9fecd8773ca770736fe248c13d9", "screen execution hash changed")
    _require(manifest.get("root") == "/root/autodl-tmp/qwen3_cage_v4/dtqi_screen_full_b0c81ba", "screen root changed")
    _require(manifest.get("run_identity_sha256") == "f010f9cc88e3272e4ac0ac669aafc02190a5c6e3c28da44b96cc1184258196e2", "screen identity hash changed")
    _require(manifest.get("summary_sha256") == "808c11b02e1122d906253b8cb269c100bb4f7f844eb417be7db2f535ebfce852", "screen summary hash changed")
    _require(manifest.get("case_file_manifest_sha256") == "7e4b4e4ca7d3d7cc94029ededc86dcaa087b5f57c4716bf8cd3052ff76767f03", "screen case manifest hash changed")
    _require(manifest.get("execution_log_sha256") == "05074a50d7e95540ca19e0618418555a6257b18ce5c36142db5b4d42eba58f24" and manifest.get("execution_log_size_bytes") == 38096, "screen log identity changed")
    _require(manifest.get("expected") == {
        "case_count": 120,
        "target_token_count": 7680,
        "document_count": 20,
        "anchor_count": 40,
        "length_counts": {"1024": 40, "2048": 40, "4032": 40},
        "failure_count": 0,
        "new_cases": 120,
        "resumed_cases": 0,
    }, "screen expected counts changed")
    _require(manifest.get("execution_boundary") == {
        "interpretation_performed": False,
        "holdout_accessed": False,
        "pg19_test_accessed": False,
        "runtime_claims_authorized": False,
        "paper_claims_authorized": False,
    }, "screen artifact boundary changed")


def _validate_log(manifest: Mapping[str, Any]) -> dict[str, bool]:
    path = Path(manifest["execution_log"])
    _require(path.is_file(), "screen execution log missing")
    _require(file_sha256(path) == manifest["execution_log_sha256"], "screen execution log hash mismatch")
    _require(path.stat().st_size == manifest["execution_log_size_bytes"], "screen execution log size mismatch")
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = (
        "=== CAGE-V4-DTQI 120-CASE SCREEN START ===",
        "=== DIRECT 120-CASE AUDIT ===",
        "=== CAGE-V4-DTQI 120-CASE SCREEN END ===",
    )
    checks = {
        "nonempty": bool(text.strip()),
        "no_traceback": "Traceback (most recent call last)" not in text,
        "no_conda_error": "ERROR conda.cli.main_run" not in text,
        "no_explicit_error": "ERROR:" not in text,
        "required_markers": all(marker in text for marker in markers),
    }
    _require(all(checks.values()), "screen execution log checks failed")
    return checks


def validate_screen(manifest: Mapping[str, Any], expected_cases: list[dict[str, Any]]) -> dict[str, Any]:
    root = Path(manifest["root"])
    identity_path = root / "run_identity.json"
    summary_path = root / "summary.json"
    _require(identity_path.is_file() and summary_path.is_file(), "screen identity or summary missing")
    _require(file_sha256(identity_path) == manifest["run_identity_sha256"], "screen identity hash mismatch")
    _require(file_sha256(summary_path) == manifest["summary_sha256"], "screen summary hash mismatch")
    identity = load_json(identity_path)
    summary = load_json(summary_path)
    _require(summary.get("identity") == identity, "screen summary identity linkage mismatch")
    _require(identity.get("source_state") == {"dirty": False, "git_commit": manifest["execution_source_commit"]}, "screen source state mismatch")
    _require(identity.get("execution_sha256") == manifest["execution_sha256"], "screen identity execution mismatch")
    expected_summary = (
        summary.get("status") == "pass",
        summary.get("claim_eligible") is False,
        summary.get("stage") == "screen_full",
        summary.get("expected_cases") == 120,
        summary.get("completed_cases") == 120,
        summary.get("new_cases") == 120,
        summary.get("resumed_cases") == 0,
        summary.get("failure_records") == 0,
        summary.get("baseline_results_reused") is True,
        summary.get("local_mse_computed") is False,
        summary.get("interpretation_performed") is False,
        summary.get("holdout_accessed") is False,
        summary.get("pg19_test_accessed") is False,
        summary.get("scientific_fields") == list(SCIENTIFIC_FIELDS),
    )
    _require(all(expected_summary), "screen summary checks failed")
    failure_files = list((root / "failures").glob("*.json")) if (root / "failures").exists() else []
    _require(not failure_files, "screen failure records exist")
    expected_by_id = {case["case_id"]: case for case in expected_cases}
    _require(set(identity["expected_case_ids"]) == set(expected_by_id), "screen expected case IDs mismatch")
    case_files = sorted((root / "cases").glob("*.json"))
    _require(len(case_files) == 120, "screen case file count mismatch")
    file_manifest = []
    scientific = []
    documents = set()
    anchors = set()
    length_counts: dict[str, int] = {}
    for path in case_files:
        record = load_json(path)
        case_id = record.get("case_id")
        _require(case_id == path.stem and case_id in expected_by_id, "screen case identity mismatch")
        expected = expected_by_id[case_id]
        _require(record.get("status") == "completed" and record.get("stage") == "screen_full", "screen case status mismatch")
        _require(record.get("identity") == identity and record.get("model") == summary["model"], "screen case provenance mismatch")
        _require(record.get("method") == expected["method"] and record.get("input") == expected["input"], "screen case method/input mismatch")
        _require(record.get("scoring") == formal_scoring(record.get("scoring", {}).get("token_nlls", [])), "screen scoring mismatch")
        cache, resume = record.get("cache", {}), record.get("resume", {})
        _require(cache.get("recent_window_equals_residual") is True and cache.get("key_quantization_triggered") is True and cache.get("value_quantization_triggered") is True, "screen cache mechanics mismatch")
        _require(resume.get("cache_identity_preserved") is True and resume.get("logits_finite") is True, "screen resume mismatch")
        document = record["input"]["document_id"]
        anchor = record["input"]["anchor_index"]
        length = str(record["input"]["prompt_length"])
        documents.add(document)
        anchors.add((document, anchor))
        length_counts[length] = length_counts.get(length, 0) + 1
        file_manifest.append({"path": f"cases/{path.name}", "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
        scientific.append({field: record[field] for field in SCIENTIFIC_FIELDS})
    _require(canonical_sha256(file_manifest) == manifest["case_file_manifest_sha256"], "screen case file manifest mismatch")
    scientific.sort(key=lambda row: row["case_id"])
    _require(len(documents) == 20 and len(anchors) == 40, "screen document/anchor count mismatch")
    _require(length_counts == {"1024": 40, "2048": 40, "4032": 40}, "screen length counts mismatch")
    return {
        "case_count": 120,
        "target_token_count": 7680,
        "document_count": 20,
        "anchor_count": 40,
        "length_counts": length_counts,
        "failure_count": 0,
        "run_identity_sha256": file_sha256(identity_path),
        "summary_sha256": file_sha256(summary_path),
        "case_file_manifest_sha256": canonical_sha256(file_manifest),
        "scientific_payload_sha256": canonical_sha256(scientific),
        "execution_log": manifest["execution_log"],
        "execution_log_sha256": manifest["execution_log_sha256"],
        "execution_log_size_bytes": manifest["execution_log_size_bytes"],
        "execution_log_checks": _validate_log(manifest),
    }


__all__ = ["AUDIT_ID", "CageV4DTQIScreenPostrunError", "MANIFEST_ID", "validate_artifact_manifest", "validate_screen"]
