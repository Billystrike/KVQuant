from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v3_promotion_acceptance import (
    PARTITIONS,
    expand_promotion_methods,
)
from utils.qwen3_cage_v3_promotion_protocol import PROMPT_LENGTHS
from utils.qwen3_cage_v3_promotion_protocol import load_promotion_protocol
from utils.qwen3_cage_v3_promotion_gate import load_promotion_gate
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


FULL_EXECUTION_ID = "qwen3-8b-cage-v3-promotion-full-holdout-v1"
EXPECTED_CASE_ID_HASHES = {
    "cage_qwen3": "d378919e78e09d0e484b953cdee3dbb91a3de347f671a9204b97483eb9bec96a",
    "kitty_qwen3": "fca71e59bb93d589f26f00f1d842baa82f038e08539e1c0a9402589c7b69711f",
}


class CageV3PromotionFullError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3PromotionFullError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV3PromotionFullError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def load_full_execution(
    path: Path,
    *,
    repo_root: Path,
    verify_server_artifacts: bool,
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any], str]:
    execution = _load(path)
    _require(execution.get("schema_version") == 1, "promotion full execution schema mismatch")
    _require(execution.get("execution_id") == FULL_EXECUTION_ID, "promotion full execution ID mismatch")
    _require(
        execution.get("status") == "frozen_after_passed_gate_preflight_before_full_holdout_gpu_execution",
        "promotion full execution freeze boundary changed",
    )
    _require(execution.get("claim_eligible") is False, "promotion full execution claim boundary changed")
    _require(
        execution.get("gate_preflight")
        == {
            "output_path": "/root/autodl-tmp/kitty_setup_audit/qwen3_cage_v3_promotion_gate_preflight_505922d.json",
            "output_sha256": "72656bce038939b575140465ba848364816451500fed350ffdefd1c906696377",
            "output_size_bytes": 2578,
            "log_path": "/root/autodl-tmp/kitty_setup_audit/qwen3_cage_v3_promotion_gate_preflight_505922d_20260816T141406.log",
            "log_sha256": "00d7b4e0e9144ae9344e634fe0995fc49e7cee7d67adda0067fa60a9b52ea4f8",
            "log_size_bytes": 7950,
        },
        "promotion gate-preflight receipt changed",
    )
    _require(
        execution.get("input_manifest")
        == {
            "path": "/root/autodl-tmp/kitty_setup_audit/qwen3_cage_v4_pg19_input_manifest_1ac723c.json",
            "sha256": "7c662aecd392f91f8d4a674af6af01496deba952820babd29045e3193dc5300d",
            "size_bytes": 5880826,
            "partition": "holdout",
        },
        "promotion full input receipt changed",
    )
    protocol_path = Path(execution["protocol"]["path"])
    if not protocol_path.is_absolute():
        protocol_path = repo_root / protocol_path
    protocol, protocol_sha256 = load_promotion_protocol(protocol_path)
    _require(protocol_sha256 == execution["protocol"]["sha256"], "promotion full protocol hash mismatch")
    gate_path = Path(execution["acceptance_gate"]["path"])
    if not gate_path.is_absolute():
        gate_path = repo_root / gate_path
    gate, gate_sha256 = load_promotion_gate(
        gate_path,
        repo_root=repo_root,
        protocol_sha256=protocol_sha256,
        input_manifest_sha256=execution["input_manifest"]["sha256"],
        verify_server_artifacts=verify_server_artifacts,
    )
    _require(gate_sha256 == execution["acceptance_gate"]["sha256"], "promotion full gate hash mismatch")
    _require(execution["input_manifest"]["sha256"] == protocol["input_receipt"]["input_manifest_sha256"], "promotion full input hash mismatch")
    _require(execution["input_manifest"]["size_bytes"] == protocol["input_receipt"]["input_manifest_size_bytes"], "promotion full input size mismatch")
    _require(
        execution.get("partitions")
        == {
            "cage_qwen3": {
                "case_count": 480,
                "case_ids_sha256": EXPECTED_CASE_ID_HASHES["cage_qwen3"],
                "method_counts": {
                    "fp16": 120,
                    "kivi-kittypro-matched": 120,
                    "cage-v1-kittypro-matched": 120,
                    "cage-v3-sr2-sink32-calibrated": 120,
                },
            },
            "kitty_qwen3": {
                "case_count": 120,
                "case_ids_sha256": EXPECTED_CASE_ID_HASHES["kitty_qwen3"],
                "method_counts": {"kitty-pro-25pct": 120},
            },
        },
        "promotion full partition receipt changed",
    )
    _require(
        execution.get("authorization")
        == {
            "gpu_full_holdout_execution": True,
            "resume": True,
            "holdout_interpretation": False,
            "pg19_test": False,
            "llama2_execution": False,
            "kitty_llama_port": False,
            "paper_claims": False,
            "runtime_claims": False,
        },
        "promotion full execution authorization changed",
    )
    _require(execution.get("execution_order") == ["kitty_qwen3", "cage_qwen3"], "promotion full execution order changed")
    _require(
        execution.get("resume_policy")
        == {
            "enabled": True,
            "existing_case_must_pass_full_identity_schema_scoring_memory_and_cache_validation": True,
            "failed_case_records_block_completion": True,
            "cross_partition_output_reuse": False,
        },
        "promotion full resume policy changed",
    )
    _require(
        execution.get("source_paths")
        == {
            "runner": "scripts/qwen3_run_cage_v3_promotion_full.py",
            "full_utils": "utils/qwen3_cage_v3_promotion_full.py",
            "gate_utils": "utils/qwen3_cage_v3_promotion_gate.py",
            "quality_runtime": "scripts/qwen3_run_cage_v4_metric_acceptance.py",
            "round1_runtime": "scripts/qwen3_run_cage_v2_round1.py",
        },
        "promotion full source paths changed",
    )
    _require(
        set(execution.get("source_sha256", {})) == set(execution["source_paths"]),
        "promotion full source-hash keys changed",
    )
    _require(
        tuple(execution.get("scientific_payload_fields", ()))
        == ("case_id", "partition", "method", "input", "memory", "scoring", "quality_cache"),
        "promotion full scientific fields changed",
    )
    for name, relative in execution["source_sha256"].items():
        source_path = repo_root / execution["source_paths"][name]
        _require(file_sha256(source_path) == relative, f"promotion full source hash changed: {name}")
    if verify_server_artifacts:
        manifest_path = Path(execution["input_manifest"]["path"])
        _require(file_sha256(manifest_path) == execution["input_manifest"]["sha256"], "promotion full server manifest hash mismatch")
        _require(manifest_path.stat().st_size == execution["input_manifest"]["size_bytes"], "promotion full server manifest size mismatch")
        preflight = execution["gate_preflight"]
        output_path = Path(preflight["output_path"])
        log_path = Path(preflight["log_path"])
        _require(file_sha256(output_path) == preflight["output_sha256"] and output_path.stat().st_size == preflight["output_size_bytes"], "promotion gate-preflight output mismatch")
        _require(file_sha256(log_path) == preflight["log_sha256"] and log_path.stat().st_size == preflight["log_size_bytes"], "promotion gate-preflight log mismatch")
        receipt = _load(output_path)
        _require(receipt.get("status") == "pass" and receipt.get("claim_eligible") is False, "promotion gate-preflight did not pass")
        _require(receipt.get("gate_sha256") == gate_sha256 and receipt.get("protocol_sha256") == protocol_sha256, "promotion gate-preflight linkage mismatch")
        _require(receipt.get("holdout_method_metrics_read") is False and receipt.get("interpretation_performed") is False and receipt.get("pg19_test_accessed") is False, "promotion gate-preflight boundary changed")
        for partition, expected_hash in EXPECTED_CASE_ID_HASHES.items():
            record = receipt.get("partitions", {}).get(partition, {})
            _require(record.get("full_holdout_case_count") == execution["partitions"][partition]["case_count"], f"{partition} preflight count mismatch")
            _require(record.get("full_holdout_case_ids_sha256") == expected_hash, f"{partition} preflight case IDs mismatch")
    return execution, file_sha256(path), protocol, protocol_sha256, gate, gate_sha256


def full_holdout_inputs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    anchors = sorted(
        (row for row in manifest.get("anchors", []) if row.get("partition") == "holdout"),
        key=lambda row: (row["document_id"], row["anchor_index"]),
    )
    _require(len(anchors) == 40, "promotion holdout anchor count mismatch")
    _require(len({(row["document_id"], row["anchor_index"]) for row in anchors}) == 40, "promotion holdout anchors are not unique")
    case_by_id = {row["case_id"]: row for row in manifest.get("cases", [])}
    inputs = []
    for anchor in anchors:
        _require(anchor.get("anchor_index") in (0, 1), "promotion holdout anchor index changed")
        full_window = anchor.get("full_window_ids", [])
        _require(len(full_window) == max(PROMPT_LENGTHS) + 64, "promotion full-window length mismatch")
        compact = {row["prompt_length"]: row for row in anchor.get("cases", [])}
        _require(set(compact) == set(PROMPT_LENGTHS), "promotion anchor prompt lengths changed")
        for length in PROMPT_LENGTHS:
            source = case_by_id.get(compact[length]["case_id"])
            _require(isinstance(source, dict), "promotion compact case is missing")
            identity = source.get("identity", {})
            _require(identity.get("partition") == "holdout", "promotion input partition changed")
            _require(identity.get("document_id") == anchor["document_id"], "promotion input document changed")
            _require(identity.get("anchor_index") == anchor["anchor_index"], "promotion input anchor changed")
            _require(identity.get("prompt_length") == length, "promotion input prompt length changed")
            prompt_offset = max(PROMPT_LENGTHS) - length
            prompt_ids = list(full_window[prompt_offset : max(PROMPT_LENGTHS)])
            continuation_ids = list(full_window[-64:])
            _require(len(prompt_ids) == length and len(continuation_ids) == 64, "promotion input token length mismatch")
            inputs.append(
                {
                    "input_case_id": source["case_id"],
                    "identity": copy.deepcopy(identity),
                    "prompt_ids": prompt_ids,
                    "continuation_ids": continuation_ids,
                }
            )
    _require(len(inputs) == 120, "promotion holdout input count mismatch")
    _require(len({row["input_case_id"] for row in inputs}) == 120, "promotion holdout input IDs are not unique")
    return inputs


def expand_full_holdout_cases(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    gate_sha256: str,
    manifest: Mapping[str, Any],
    partition: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    _require(partition in PARTITIONS, "unsupported promotion partition")
    methods = expand_promotion_methods(protocol, repo_root=repo_root, partition=partition)
    methods_by_length: dict[int, list[dict[str, Any]]] = {length: [] for length in PROMPT_LENGTHS}
    for method in methods:
        methods_by_length[method["prompt_length"]].append(method)
    expected_methods_per_length = 4 if partition == "cage_qwen3" else 1
    _require(all(len(rows) == expected_methods_per_length for rows in methods_by_length.values()), "promotion full method grid mismatch")
    cases = []
    for source in full_holdout_inputs(manifest):
        length = source["identity"]["prompt_length"]
        for method in methods_by_length[length]:
            identity = {
                "gate_sha256": gate_sha256,
                "protocol_sha256": protocol_sha256,
                "partition": partition,
                "stage": "promotion_full_holdout",
                "method_id": method["id"],
                "input_case_id": source["input_case_id"],
            }
            cases.append(
                {
                    "case_id": canonical_sha256(identity)[:24],
                    "partition": partition,
                    "method": copy.deepcopy(method),
                    "input": copy.deepcopy(source["identity"]),
                    "prompt_ids": list(source["prompt_ids"]),
                    "continuation_ids": list(source["continuation_ids"]),
                }
            )
    expected = 480 if partition == "cage_qwen3" else 120
    _require(len(cases) == expected, "promotion full case count mismatch")
    _require(len({case["case_id"] for case in cases}) == expected, "promotion full case IDs are not unique")
    return cases


__all__ = [
    "CageV3PromotionFullError",
    "EXPECTED_CASE_ID_HASHES",
    "FULL_EXECUTION_ID",
    "expand_full_holdout_cases",
    "full_holdout_inputs",
    "load_full_execution",
]
