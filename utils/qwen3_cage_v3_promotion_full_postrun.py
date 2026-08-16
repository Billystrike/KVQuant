from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256
from utils.qwen3_formal import formal_scoring


ARTIFACT_MANIFEST_ID = "qwen3-8b-cage-v3-promotion-full-artifacts-v1"
AUDIT_ID = "qwen3-8b-cage-v3-promotion-full-postrun-v1"
MANIFEST_CANONICAL_SHA256 = "12225c5532d7064aa3176718813918365b01fd0d4d77d9b815082cb33db3ebdc"
EXECUTION_COMMIT = "a2fe881b4531b9874ae8880c16fab6960453e68f"
EXECUTION_SHA256 = "c7da6f2389e64c4471ef876722a6c37e22276f0ce97ee5d8b0bd8fde58375805"
GATE_SHA256 = "c6c0d54abc1b0fdbd9ecaffc6d243fee9f66b391a626a56de4245eb47c091819"
PROTOCOL_SHA256 = "bc5d3811c0483433feef883c5511d350e8e4cbc48e12f0e4804bc2ccecf31f18"
INPUT_MANIFEST_SHA256 = "7c662aecd392f91f8d4a674af6af01496deba952820babd29045e3193dc5300d"
EXPECTED_SOURCE_SHA256 = {
    "runner": "84da671efab351fedd7d7d6cc8349e1605f9b433ce1247c73697757702125187",
    "full_utils": "6ea2610cc9354fe0963e39b67ddd63fb2aebcaa3f11193b217ea7ee426e1619f",
    "gate_utils": "b67c729915a9e75ccabe28d1d485ecc6a6072da076eb4f2c0184843b0b379a3a",
    "quality_runtime": "4991266a2c7c41e75823f1423c6de60e7ab02ec7e0a0c059afa8763940450209",
    "round1_runtime": "3dc1528a2cfff9a84f7e0bd8cd1f1d740c6e264295803bcfdfba8767344010b8",
}
SOURCE_PATHS = {
    "runner": "scripts/qwen3_run_cage_v3_promotion_full.py",
    "full_utils": "utils/qwen3_cage_v3_promotion_full.py",
    "gate_utils": "utils/qwen3_cage_v3_promotion_gate.py",
    "quality_runtime": "scripts/qwen3_run_cage_v4_metric_acceptance.py",
    "round1_runtime": "scripts/qwen3_run_cage_v2_round1.py",
}
SCIENTIFIC_FIELDS = ("case_id", "partition", "method", "input", "memory", "scoring", "quality_cache")
EXPECTED_PARTITIONS = {
    "kitty_qwen3": {
        "case_count": 120,
        "case_ids_sha256": "fca71e59bb93d589f26f00f1d842baa82f038e08539e1c0a9402589c7b69711f",
        "method_counts": {"kitty-pro-25pct": 120},
    },
    "cage_qwen3": {
        "case_count": 480,
        "case_ids_sha256": "d378919e78e09d0e484b953cdee3dbb91a3de347f671a9204b97483eb9bec96a",
        "method_counts": {
            "fp16": 120,
            "kivi-kittypro-matched": 120,
            "cage-v1-kittypro-matched": 120,
            "cage-v3-sr2-sink32-calibrated": 120,
        },
    },
}


class CageV3PromotionFullPostrunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3PromotionFullPostrunError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV3PromotionFullPostrunError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _verify_file(path: Path, *, sha256: str, size_bytes: int | None = None, label: str) -> None:
    _require(path.is_file(), f"{label} is missing")
    _require(file_sha256(path) == sha256, f"{label} hash mismatch")
    if size_bytes is not None:
        _require(path.stat().st_size == size_bytes, f"{label} size mismatch")


def shell_case_manifest_sha256(root: Path) -> str:
    lines = []
    for path in sorted((root / "cases").glob("*.json"), key=lambda item: item.name):
        lines.append(f"{file_sha256(path)}  cases/{path.name}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def validate_artifact_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema_version") == 1, "full artifact manifest schema mismatch")
    _require(manifest.get("artifact_manifest_id") == ARTIFACT_MANIFEST_ID, "full artifact manifest ID mismatch")
    _require(manifest.get("status") == "frozen_after_both_full_partitions_before_joint_postrun_and_interpretation", "full artifact manifest status mismatch")
    _require(manifest.get("claim_eligible") is False, "full artifact claim boundary changed")
    _require(manifest.get("execution_source_commit") == EXECUTION_COMMIT, "full execution commit changed")
    _require(manifest.get("execution_sha256") == EXECUTION_SHA256, "full execution hash changed")
    _require(manifest.get("gate_sha256") == GATE_SHA256, "full gate hash changed")
    _require(manifest.get("protocol_sha256") == PROTOCOL_SHA256, "full protocol hash changed")
    _require(manifest.get("input_manifest_sha256") == INPUT_MANIFEST_SHA256, "full input manifest hash changed")
    _require(manifest.get("source_sha256") == EXPECTED_SOURCE_SHA256, "full execution source hashes changed")
    _require(tuple(manifest.get("scientific_payload_fields", ())) == SCIENTIFIC_FIELDS, "full scientific fields changed")
    _require(set(manifest.get("partitions", {})) == set(EXPECTED_PARTITIONS), "full partitions changed")
    for partition, expected in EXPECTED_PARTITIONS.items():
        record = manifest["partitions"][partition]
        _require(record.get("case_count") == expected["case_count"], f"{partition} artifact count changed")
        _require(record.get("case_ids_sha256") == expected["case_ids_sha256"], f"{partition} artifact case IDs changed")
        _require(record.get("method_counts") == expected["method_counts"], f"{partition} artifact method counts changed")
    attempts = manifest.get("failed_pre_case_attempts")
    _require(isinstance(attempts, list) and len(attempts) == 1, "full failed pre-case attempt count changed")
    _require(attempts[0].get("cases_executed") == 0 and attempts[0].get("gpu_model_loaded") is False and attempts[0].get("holdout_method_metrics_read") is False, "failed pre-case attempt boundary changed")
    _require(
        manifest.get("postrun_boundary")
        == {
            "full_execution_completed": True,
            "interpretation_authorized_before_postrun": False,
            "analysis_authorized_by_manifest": False,
            "pg19_test_access": False,
            "llama2_execution": False,
            "kitty_llama_port": False,
            "paper_claims": False,
            "runtime_claims": False,
        },
        "full post-run boundary changed",
    )
    _require(canonical_sha256(manifest) == MANIFEST_CANONICAL_SHA256, "full artifact manifest frozen payload mismatch")


def _validate_preflight(spec: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(spec["output_path"])
    log = Path(spec["log_path"])
    _verify_file(output, sha256=spec["output_sha256"], size_bytes=spec["output_size_bytes"], label="full execution preflight")
    _verify_file(log, sha256=spec["log_sha256"], size_bytes=spec["log_size_bytes"], label="full execution preflight log")
    value = _load(output)
    _require(value.get("status") == "pass" and value.get("claim_eligible") is False, "full execution preflight did not pass")
    _require(value.get("execution_sha256") == EXECUTION_SHA256 and value.get("gate_sha256") == GATE_SHA256, "full execution preflight linkage mismatch")
    _require(value.get("holdout_method_metrics_read") is False and value.get("interpretation_performed") is False and value.get("pg19_test_accessed") is False, "full execution preflight boundary changed")
    for partition, expected in EXPECTED_PARTITIONS.items():
        record = value.get("partitions", {}).get(partition, {})
        _require(record.get("case_count") == expected["case_count"] and record.get("case_ids_sha256") == expected["case_ids_sha256"], f"{partition} execution preflight receipt mismatch")
    return {"output_path": str(output), "output_sha256": spec["output_sha256"], "log_path": str(log), "log_sha256": spec["log_sha256"], "status": "pass"}


def _validate_execution_log(path: Path, spec: Mapping[str, Any], *, partition: str) -> dict[str, bool]:
    _verify_file(path, sha256=spec["execution_log_sha256"], size_bytes=spec["execution_log_size_bytes"], label=f"{partition} execution log")
    text = path.read_text(encoding="utf-8", errors="replace")
    family = "KITTY" if partition == "kitty_qwen3" else "CAGE"
    checks = {
        "nonempty": bool(text.strip()),
        "no_traceback": "Traceback (most recent call last)" not in text,
        "no_conda_error": "ERROR conda.cli.main_run" not in text,
        "no_explicit_error": "ERROR:" not in text,
        "start_marker": f"=== CAGE-V3 PROMOTION {family} FULL START ===" in text,
        "end_marker": f"=== CAGE-V3 PROMOTION {family} FULL END ===" in text,
    }
    _require(all(checks.values()), f"{partition} execution log checks failed")
    return checks


def _validate_partition(spec: Mapping[str, Any], *, partition: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = EXPECTED_PARTITIONS[partition]
    root = Path(spec["output_dir"])
    identity_path = root / "run_identity.json"
    summary_path = root / "summary.json"
    _verify_file(identity_path, sha256=spec["run_identity_sha256"], label=f"{partition} run identity")
    _verify_file(summary_path, sha256=spec["summary_sha256"], label=f"{partition} summary")
    _require(shell_case_manifest_sha256(root) == spec["case_file_manifest_sha256"], f"{partition} case-file manifest mismatch")
    identity = _load(identity_path)
    summary = _load(summary_path)
    count = expected["case_count"]
    _require(
        summary.get("schema_version") == 1
        and summary.get("status") == "pass"
        and summary.get("claim_eligible") is False
        and summary.get("partition") == partition
        and summary.get("stage") == "promotion_full_holdout"
        and summary.get("expected_cases") == count
        and summary.get("completed_cases") == count
        and summary.get("new_cases") + summary.get("resumed_cases") == count
        and summary.get("failure_records") == 0
        and summary.get("holdout_interpretation_authorized") is False
        and summary.get("pg19_test_accessed") is False
        and summary.get("llama2_execution_authorized") is False
        and summary.get("kitty_llama_port_authorized") is False
        and summary.get("run_identity") == identity,
        f"{partition} summary checks failed",
    )
    _require(identity.get("execution_sha256") == EXECUTION_SHA256 and identity.get("gate_sha256") == GATE_SHA256 and identity.get("protocol_sha256") == PROTOCOL_SHA256 and identity.get("input_manifest_sha256") == INPUT_MANIFEST_SHA256, f"{partition} run identity linkage mismatch")
    source = identity.get("source_state", {})
    _require(source.get("git_commit") == EXECUTION_COMMIT and source.get("dirty") is False, f"{partition} source state mismatch")
    if partition == "kitty_qwen3":
        _require(source.get("kitty_commit") == "dfd2c07b407d6b407179359207c612ab631f3ed1" and source.get("kitty_dirty") is False, "Kitty source state mismatch")
        _require(source.get("transformers_commit") == "37f8b0b53512e6aae0cfd15746c133c101783178", "Kitty Transformers state mismatch")
    case_files = sorted((root / "cases").glob("*.json"), key=lambda item: item.name)
    failure_files = sorted((root / "failures").glob("*.json")) if (root / "failures").exists() else []
    _require(len(case_files) == count and not failure_files, f"{partition} case/failure count mismatch")
    expected_ids = identity.get("expected_case_ids", [])
    _require(len(expected_ids) == count and canonical_sha256(expected_ids) == expected["case_ids_sha256"], f"{partition} case-ID receipt mismatch")
    records = [_load(path) for path in case_files]
    methods: Counter[str] = Counter()
    for path, record in zip(case_files, records):
        _require(record.get("schema_version") == 1 and record.get("status") == "completed" and record.get("claim_eligible") is False, f"{partition} incomplete case")
        _require(record.get("case_id") == path.stem and record.get("case_id") in set(expected_ids), f"{partition} case identity mismatch")
        _require(record.get("partition") == partition and record.get("stage") == "promotion_full_holdout", f"{partition} case boundary mismatch")
        _require(record.get("identity") == identity and record.get("model") == summary.get("model"), f"{partition} case model/identity mismatch")
        token_nlls = record.get("scoring", {}).get("token_nlls", [])
        _require(len(token_nlls) == 64 and all(isinstance(value, (int, float)) and math.isfinite(value) for value in token_nlls), f"{partition} invalid token NLLs")
        _require(record.get("scoring") == formal_scoring(token_nlls), f"{partition} scoring mismatch")
        method = record.get("method", {})
        methods[method.get("metric_method_id")] += 1
        _require(record.get("memory", {}).get("model_total_bytes") == method.get("packed_bytes"), f"{partition} memory mismatch")
        cache = record.get("quality_cache", {})
        expected_length = record.get("input", {}).get("prompt_length") + 63
        _require(cache.get("reported_seq_length") == expected_length and cache.get("expected_seq_length") == expected_length and cache.get("layer_count") == 36 and cache.get("tensors_finite") is True, f"{partition} cache checks failed")
    _require(dict(methods) == expected["method_counts"], f"{partition} method counts mismatch")
    payload = sorted(
        ({field: record[field] for field in SCIENTIFIC_FIELDS} for record in records),
        key=lambda row: row["case_id"],
    )
    return {
        "output_dir": str(root),
        "case_count": count,
        "failure_count": 0,
        "target_token_count": count * 64,
        "method_counts": dict(methods),
        "case_ids_sha256": expected["case_ids_sha256"],
        "case_file_manifest_sha256": spec["case_file_manifest_sha256"],
        "run_identity_sha256": spec["run_identity_sha256"],
        "summary_sha256": spec["summary_sha256"],
        "scientific_payload_sha256": canonical_sha256(payload),
        "execution_log": spec["execution_log"],
        "execution_log_sha256": spec["execution_log_sha256"],
        "execution_log_size_bytes": spec["execution_log_size_bytes"],
        "execution_log_checks": _validate_execution_log(Path(spec["execution_log"]), spec, partition=partition),
    }, payload


def build_postrun_audit(manifest_path: str | Path) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = _load(source)
    validate_artifact_manifest(manifest)
    repo_root = source.parents[1]
    for name, relative_path in SOURCE_PATHS.items():
        _verify_file(
            repo_root / relative_path,
            sha256=EXPECTED_SOURCE_SHA256[name],
            label=f"frozen execution source {name}",
        )
    preflight = _validate_preflight(manifest["execution_preflight"])
    attempts = []
    for attempt in manifest["failed_pre_case_attempts"]:
        log = Path(attempt["log_path"])
        _verify_file(log, sha256=attempt["log_sha256"], size_bytes=attempt["log_size_bytes"], label="failed pre-case attempt log")
        text = log.read_text(encoding="utf-8", errors="replace")
        _require("ERROR: output already exists" in text, "failed pre-case attempt classification mismatch")
        attempts.append(dict(attempt))
    partitions = {}
    payloads = []
    for partition in ("kitty_qwen3", "cage_qwen3"):
        partitions[partition], payload = _validate_partition(manifest["partitions"][partition], partition=partition)
        payloads.extend(payload)
    combined_payload = sorted(payloads, key=lambda row: (row["partition"], row["case_id"]))
    return {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "status": "pass",
        "claim_eligible": False,
        "interpretation_performed": False,
        "artifact_manifest_path": str(source),
        "artifact_manifest_sha256": file_sha256(source),
        "execution_source_commit": EXECUTION_COMMIT,
        "execution_sha256": EXECUTION_SHA256,
        "gate_sha256": GATE_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "execution_preflight": preflight,
        "failed_pre_case_attempt_count": len(attempts),
        "failed_pre_case_attempts": attempts,
        "partition_count": 2,
        "case_count": 600,
        "target_token_count": 38400,
        "failure_count": 0,
        "scientific_payload_sha256": canonical_sha256(combined_payload),
        "partitions": partitions,
        "analysis_gate_candidate": {
            "status": "eligible_for_checked_in_analysis_receipt",
            "analysis_authorized_by_this_audit": False,
            "expected_case_count": 600,
            "expected_target_token_count": 38400,
        },
        "execution_boundary": {
            "full_execution_completed": True,
            "analysis_gate_candidate_created": True,
            "interpretation_authorized": False,
            "pg19_test_accessed": False,
            "llama2_execution_authorized": False,
            "kitty_llama_port_authorized": False,
            "paper_claims_authorized": False,
            "runtime_claims_authorized": False,
        },
    }


__all__ = [
    "AUDIT_ID",
    "ARTIFACT_MANIFEST_ID",
    "CageV3PromotionFullPostrunError",
    "build_postrun_audit",
    "shell_case_manifest_sha256",
    "validate_artifact_manifest",
]
