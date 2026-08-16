from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v3_promotion_acceptance import SCIENTIFIC_FIELDS
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256
from utils.qwen3_formal import formal_scoring


ARTIFACT_MANIFEST_ID = "qwen3-8b-cage-v3-promotion-acceptance-artifacts-v1"
AUDIT_ID = "qwen3-8b-cage-v3-promotion-acceptance-postrun-v1"
SOURCE_COMMIT = "cc72bf72d206a94b39535415cf43394cc7b65dfa"
ARTIFACT_MANIFEST_CANONICAL_SHA256 = "3922263eec0841b95535e21eb969b6b4ee19a3e4225330379bfdc93be3fe231e"
PARTITION_CASE_COUNTS = {"cage_qwen3": 12, "kitty_qwen3": 3}
FULL_HOLDOUT_CASE_COUNTS = {"cage_qwen3": 480, "kitty_qwen3": 120, "total": 600}


class CageV3PromotionPostrunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3PromotionPostrunError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV3PromotionPostrunError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_artifact_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema_version") == 1, "artifact manifest schema mismatch")
    _require(manifest.get("artifact_manifest_id") == ARTIFACT_MANIFEST_ID, "artifact manifest ID mismatch")
    _require(
        manifest.get("status") == "frozen_after_two_fresh_acceptance_repeats_before_joint_postrun_audit",
        "artifact manifest status mismatch",
    )
    _require(manifest.get("claim_eligible") is False, "artifact claim boundary changed")
    _require(manifest.get("execution_sha256") == "5713e058e3df2a6b54aa162a4b99b28b18b7f8ec18ac8f23a95f6b58db826c23", "execution hash changed")
    _require(manifest.get("protocol_sha256") == "bc5d3811c0483433feef883c5511d350e8e4cbc48e12f0e4804bc2ccecf31f18", "protocol hash changed")
    _require(manifest.get("input_manifest_sha256") == "7c662aecd392f91f8d4a674af6af01496deba952820babd29045e3193dc5300d", "input manifest hash changed")
    source = manifest.get("acceptance_source", {})
    _require(source.get("git_commit") == SOURCE_COMMIT, "acceptance source commit changed")
    _require(source.get("kitty_commit") == "dfd2c07b407d6b407179359207c612ab631f3ed1", "Kitty commit changed")
    _require(source.get("transformers_commit") == "37f8b0b53512e6aae0cfd15746c133c101783178", "Transformers commit changed")
    _require(
        source.get("source_sha256")
        == {
            "runner": "8c9462ac029e14bcb40c31573f3fc0442d9629b68d0aefd5e86252e0e5bf3296",
            "comparator": "0c9c727240cb0b8a488fb3d456c0d5d53f9282ccaccbd27d6d830d7777534031",
            "acceptance_utils": "e59656cf2b368d8d9073fb1a837ed71f7e509d47559d6792c5a39109b20a42fa",
            "quality_runtime": "4991266a2c7c41e75823f1423c6de60e7ab02ec7e0a0c059afa8763940450209",
        },
        "acceptance source hashes changed",
    )
    _require(set(manifest.get("partitions", {})) == set(PARTITION_CASE_COUNTS), "artifact partitions changed")
    for partition, expected_count in PARTITION_CASE_COUNTS.items():
        spec = manifest["partitions"][partition]
        _require(spec.get("case_count") == expected_count, f"{partition} case count changed")
        _require(set(spec.get("repeat_a", {})) == {"output_dir", "run_identity_sha256", "summary_sha256", "execution_log", "execution_log_sha256", "execution_log_size_bytes"}, f"{partition} repeat A receipt fields changed")
        _require(set(spec.get("repeat_b", {})) == {"output_dir", "run_identity_sha256", "summary_sha256", "execution_log", "execution_log_sha256", "execution_log_size_bytes"}, f"{partition} repeat B receipt fields changed")
        _require(set(spec.get("comparison", {})) == {"path", "sha256", "size_bytes", "log_path", "log_sha256", "log_size_bytes"}, f"{partition} comparison receipt fields changed")
        _require(sum(spec.get("method_counts", {}).values()) == expected_count, f"{partition} method counts changed")
    attempts = manifest.get("failed_pre_case_attempts")
    _require(isinstance(attempts, list) and len(attempts) == 2, "failed pre-case attempt count changed")
    for attempt in attempts:
        _require(attempt.get("cases_executed") == 0, "failed attempt executed cases")
        _require(attempt.get("gpu_model_loaded") is False, "failed attempt loaded a GPU model")
        _require(attempt.get("holdout_method_metrics_read") is False, "failed attempt read holdout metrics")
    _require(
        manifest.get("authorization_before_joint_gate")
        == {
            "full_holdout": False,
            "holdout_interpretation": False,
            "pg19_test": False,
            "llama2_execution": False,
            "kitty_llama_port": False,
            "paper_claims": False,
            "runtime_claims": False,
        },
        "pre-gate authorization boundary changed",
    )
    _require(
        canonical_sha256(manifest) == ARTIFACT_MANIFEST_CANONICAL_SHA256,
        "artifact manifest frozen payload mismatch",
    )


def _verify_file(path: Path, *, sha256: str, size_bytes: int | None = None, label: str) -> None:
    _require(path.is_file(), f"{label} is missing")
    _require(file_sha256(path) == sha256, f"{label} hash mismatch")
    if size_bytes is not None:
        _require(path.stat().st_size == size_bytes, f"{label} size mismatch")


def shell_case_manifest_sha256(root: Path) -> str:
    lines = []
    for path in sorted((root / "cases").glob("*.json"), key=lambda item: item.name):
        lines.append(f"{file_sha256(path)}  cases/{path.name}\n")
    import hashlib

    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def scientific_payload(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payload = [{field: record[field] for field in SCIENTIFIC_FIELDS} for record in records]
    return sorted(payload, key=lambda row: row["case_id"])


def _validate_execution_log(path: Path, spec: Mapping[str, Any], *, partition: str, repeat: str) -> dict[str, bool]:
    _verify_file(path, sha256=spec["execution_log_sha256"], size_bytes=spec["execution_log_size_bytes"], label=f"{partition} repeat {repeat} log")
    text = path.read_text(encoding="utf-8", errors="replace")
    family = "CAGE" if partition == "cage_qwen3" else "KITTY"
    checks = {
        "nonempty": bool(text.strip()),
        "no_traceback": "Traceback (most recent call last)" not in text,
        "no_conda_error": "ERROR conda.cli.main_run" not in text,
        "no_explicit_error": "ERROR:" not in text,
        "start_marker": f"=== CAGE-V3 PROMOTION {family} ACCEPTANCE {repeat.upper()} START ===" in text,
        "end_marker": f"=== CAGE-V3 PROMOTION {family} ACCEPTANCE {repeat.upper()} END ===" in text,
    }
    _require(all(checks.values()), f"{partition} repeat {repeat} execution log checks failed")
    return checks


def _validate_repeat(spec: Mapping[str, Any], *, partition: str, repeat: str, expected_methods: Mapping[str, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(spec["output_dir"])
    identity_path = root / "run_identity.json"
    summary_path = root / "summary.json"
    _verify_file(identity_path, sha256=spec["run_identity_sha256"], label=f"{partition} repeat {repeat} run identity")
    _verify_file(summary_path, sha256=spec["summary_sha256"], label=f"{partition} repeat {repeat} summary")
    identity = _load(identity_path)
    summary = _load(summary_path)
    expected_count = PARTITION_CASE_COUNTS[partition]
    _require(
        summary.get("schema_version") == 1
        and summary.get("status") == "pass"
        and summary.get("claim_eligible") is False
        and summary.get("partition") == partition
        and summary.get("stage") == "promotion_acceptance"
        and summary.get("expected_cases") == expected_count
        and summary.get("completed_cases") == expected_count
        and summary.get("new_cases") == expected_count
        and summary.get("resumed_cases") == 0
        and summary.get("failure_records") == 0
        and summary.get("full_holdout_authorized") is False
        and summary.get("pg19_test_accessed") is False
        and summary.get("run_identity") == identity,
        f"{partition} repeat {repeat} summary checks failed",
    )
    source_state = identity.get("source_state", {})
    _require(identity.get("execution_sha256") == "5713e058e3df2a6b54aa162a4b99b28b18b7f8ec18ac8f23a95f6b58db826c23", f"{partition} repeat {repeat} execution linkage mismatch")
    _require(identity.get("protocol_sha256") == "bc5d3811c0483433feef883c5511d350e8e4cbc48e12f0e4804bc2ccecf31f18", f"{partition} repeat {repeat} protocol linkage mismatch")
    _require(identity.get("input_manifest_sha256") == "7c662aecd392f91f8d4a674af6af01496deba952820babd29045e3193dc5300d", f"{partition} repeat {repeat} input linkage mismatch")
    _require(source_state.get("git_commit") == SOURCE_COMMIT and source_state.get("dirty") is False, f"{partition} repeat {repeat} CAGE source mismatch")
    if partition == "kitty_qwen3":
        _require(source_state.get("kitty_commit") == "dfd2c07b407d6b407179359207c612ab631f3ed1" and source_state.get("kitty_dirty") is False, f"{partition} repeat {repeat} Kitty source mismatch")
        _require(source_state.get("transformers_commit") == "37f8b0b53512e6aae0cfd15746c133c101783178", f"{partition} repeat {repeat} Transformers source mismatch")
    failure_files = sorted((root / "failures").glob("*.json")) if (root / "failures").exists() else []
    _require(not failure_files, f"{partition} repeat {repeat} contains failure records")
    case_files = sorted((root / "cases").glob("*.json"), key=lambda item: item.name)
    _require(len(case_files) == expected_count, f"{partition} repeat {repeat} case count mismatch")
    records = [_load(path) for path in case_files]
    expected_ids = set(identity.get("expected_case_ids", []))
    _require(len(expected_ids) == expected_count, f"{partition} repeat {repeat} expected case IDs changed")
    methods: Counter[str] = Counter()
    for path, record in zip(case_files, records):
        _require(record.get("schema_version") == 1 and record.get("status") == "completed", f"{partition} repeat {repeat} incomplete case")
        _require(record.get("case_id") == path.stem and record.get("case_id") in expected_ids, f"{partition} repeat {repeat} case identity mismatch")
        _require(record.get("partition") == partition and record.get("stage") == "promotion_acceptance", f"{partition} repeat {repeat} case boundary mismatch")
        _require(record.get("identity") == identity and record.get("model") == summary.get("model"), f"{partition} repeat {repeat} model/identity mismatch")
        scoring = record.get("scoring", {})
        token_nlls = scoring.get("token_nlls", [])
        _require(len(token_nlls) == 64 and all(isinstance(value, (int, float)) and math.isfinite(value) for value in token_nlls), f"{partition} repeat {repeat} invalid token NLLs")
        _require(scoring == formal_scoring(token_nlls), f"{partition} repeat {repeat} scoring mismatch")
        method = record.get("method", {})
        methods[method.get("metric_method_id")] += 1
        memory = record.get("memory", {})
        _require(memory.get("model_total_bytes") == method.get("packed_bytes"), f"{partition} repeat {repeat} memory mismatch")
        cache = record.get("quality_cache", {})
        expected_length = record.get("input", {}).get("prompt_length") + 63
        _require(cache.get("reported_seq_length") == expected_length and cache.get("expected_seq_length") == expected_length and cache.get("layer_count") == 36 and cache.get("tensors_finite") is True, f"{partition} repeat {repeat} cache checks failed")
    _require(dict(methods) == dict(expected_methods), f"{partition} repeat {repeat} method counts mismatch")
    payload = scientific_payload(records)
    return {
        "output_dir": str(root),
        "case_count": expected_count,
        "failure_count": 0,
        "case_ids_sha256": canonical_sha256(sorted(expected_ids)),
        "case_file_manifest_sha256": shell_case_manifest_sha256(root),
        "run_identity_sha256": file_sha256(identity_path),
        "summary_sha256": file_sha256(summary_path),
        "scientific_payload_sha256": canonical_sha256(payload),
        "execution_log": str(spec["execution_log"]),
        "execution_log_sha256": spec["execution_log_sha256"],
        "execution_log_size_bytes": spec["execution_log_size_bytes"],
        "execution_log_checks": _validate_execution_log(Path(spec["execution_log"]), spec, partition=partition, repeat=repeat),
    }, payload


def _validate_comparison(spec: Mapping[str, Any], *, partition: str, expected_payload_sha256: str) -> dict[str, Any]:
    path = Path(spec["path"])
    log_path = Path(spec["log_path"])
    _verify_file(path, sha256=spec["sha256"], size_bytes=spec["size_bytes"], label=f"{partition} comparison")
    _verify_file(log_path, sha256=spec["log_sha256"], size_bytes=spec["log_size_bytes"], label=f"{partition} comparison log")
    value = _load(path)
    expected_count = PARTITION_CASE_COUNTS[partition]
    _require(value.get("status") == "pass" and value.get("claim_eligible") is False, f"{partition} comparison status mismatch")
    _require(value.get("partition") == partition and value.get("case_count") == expected_count, f"{partition} comparison identity mismatch")
    _require(value.get("scientific_fields") == list(SCIENTIFIC_FIELDS), f"{partition} comparison fields mismatch")
    _require(value.get("required_consistency") == "bitwise_equal_json_numeric_payload" and value.get("mismatch_case_ids") == [], f"{partition} comparison consistency mismatch")
    _require(value.get("scientific_payload_sha256") == expected_payload_sha256, f"{partition} comparison payload linkage mismatch")
    _require(value.get("comparator_sha256") == "0c9c727240cb0b8a488fb3d456c0d5d53f9282ccaccbd27d6d830d7777534031", f"{partition} comparator hash mismatch")
    _require(value.get("full_holdout_authorized") is False, f"{partition} comparison crossed full-holdout boundary")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    _require("Traceback (most recent call last)" not in text and "ERROR:" not in text, f"{partition} comparison log contains an error")
    family = "CAGE" if partition == "cage_qwen3" else "KITTY"
    _require(
        f"=== CAGE-V3 PROMOTION {family} A-B COMPARISON START ===" in text
        and f"=== CAGE-V3 PROMOTION {family} A-B COMPARISON END ===" in text,
        f"{partition} comparison log markers missing",
    )
    return {
        "path": str(path),
        "sha256": spec["sha256"],
        "size_bytes": spec["size_bytes"],
        "log_path": str(log_path),
        "log_sha256": spec["log_sha256"],
        "log_size_bytes": spec["log_size_bytes"],
        "status": "pass",
        "scientific_payload_sha256": expected_payload_sha256,
        "mismatch_case_ids": [],
    }


def _validate_preflight(spec: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(spec["output_path"])
    log = Path(spec["log_path"])
    _verify_file(output, sha256=spec["output_sha256"], size_bytes=spec["output_size_bytes"], label="acceptance execution preflight")
    _verify_file(log, sha256=spec["log_sha256"], size_bytes=spec["log_size_bytes"], label="acceptance execution preflight log")
    value = _load(output)
    _require(value.get("status") == "pass" and value.get("claim_eligible") is False, "acceptance execution preflight did not pass")
    _require(value.get("full_holdout_authorized") is False and value.get("pg19_test_accessed") is False and value.get("holdout_method_metrics_read") is False, "acceptance execution preflight boundary changed")
    return {"output_path": str(output), "output_sha256": spec["output_sha256"], "log_path": str(log), "log_sha256": spec["log_sha256"], "status": "pass"}


def build_postrun_audit(manifest_path: str | Path) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = _load(source)
    validate_artifact_manifest(manifest)
    preflight = _validate_preflight(manifest["execution_preflight"])
    failed_attempts = []
    for attempt in manifest["failed_pre_case_attempts"]:
        log = Path(attempt["log_path"])
        _verify_file(log, sha256=attempt["log_sha256"], size_bytes=attempt["log_size_bytes"], label=f"failed pre-case attempt {attempt['label']}")
        failed_attempts.append(dict(attempt))
    partitions = {}
    total_case_files = 0
    for partition in ("cage_qwen3", "kitty_qwen3"):
        spec = manifest["partitions"][partition]
        repeats = {}
        payloads = {}
        for repeat in ("a", "b"):
            repeats[repeat], payloads[repeat] = _validate_repeat(spec[f"repeat_{repeat}"], partition=partition, repeat=repeat, expected_methods=spec["method_counts"])
        _require(payloads["a"] == payloads["b"], f"{partition} repeat scientific payloads differ")
        payload_sha256 = canonical_sha256(payloads["a"])
        _require(payload_sha256 == spec["scientific_payload_sha256"], f"{partition} frozen payload hash mismatch")
        comparison = _validate_comparison(spec["comparison"], partition=partition, expected_payload_sha256=payload_sha256)
        total_case_files += 2 * spec["case_count"]
        partitions[partition] = {
            "case_count_per_repeat": spec["case_count"],
            "repeat_count": 2,
            "total_case_file_count": 2 * spec["case_count"],
            "failure_count": 0,
            "method_counts": spec["method_counts"],
            "scientific_payload_sha256": payload_sha256,
            "repeat_payloads_bitwise_equal": True,
            "repeats": repeats,
            "comparison": comparison,
        }
    return {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "status": "pass",
        "claim_eligible": False,
        "interpretation_performed": False,
        "artifact_manifest_path": str(source),
        "artifact_manifest_sha256": file_sha256(source),
        "acceptance_source_commit": SOURCE_COMMIT,
        "execution_sha256": manifest["execution_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "input_manifest_sha256": manifest["input_manifest_sha256"],
        "execution_preflight": preflight,
        "failed_pre_case_attempt_count": len(failed_attempts),
        "failed_pre_case_attempts": failed_attempts,
        "partition_count": 2,
        "case_count_per_joint_repeat": sum(PARTITION_CASE_COUNTS.values()),
        "repeat_count": 2,
        "total_case_file_count": total_case_files,
        "failure_count": 0,
        "partitions": partitions,
        "acceptance_gate_candidate": {
            "status": "eligible_for_checked_in_gate_review",
            "full_holdout_case_counts": FULL_HOLDOUT_CASE_COUNTS,
            "full_holdout_authorized_by_this_audit": False,
        },
        "execution_boundary": {
            **manifest["authorization_before_joint_gate"],
            "acceptance_complete": True,
            "acceptance_repeats_bitwise_equal": True,
            "full_holdout_gate_candidate_created": True,
        },
    }


__all__ = [
    "ARTIFACT_MANIFEST_ID",
    "AUDIT_ID",
    "CageV3PromotionPostrunError",
    "build_postrun_audit",
    "scientific_payload",
    "shell_case_manifest_sha256",
    "validate_artifact_manifest",
]
