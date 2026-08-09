from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from utils.qwen3_cage_v3_execution import canonical_sha256, load_json
from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes
from utils.qwen3_cage_v3_protocol import (
    calibrated_tiered_two_bit_quotas,
    file_sha256,
)
from utils.qwen3_perturbation_protocol import (
    aggregate_layer_metrics,
    validate_aggregates,
    validate_layer_records,
)


RECEIPT_ID = "qwen3-8b-cage-v3-calibration-full-receipt-v1"
PLAN_ID = "qwen3-8b-cage-v3-calibrated-layer-quota-plan-v1"
FAMILIES = (
    "pure-sr2-sink32-uniform32",
    "pure-sr2-sink64-uniform32",
)
PROMPT_LENGTHS = (1024, 2048, 4032)
SCIENTIFIC_FIELDS = (
    "case_id",
    "method",
    "input",
    "memory",
    "layer_metrics",
    "aggregates",
    "cache",
)


class CageV3CalibrationPostrunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3CalibrationPostrunError(message)


def shell_case_manifest_sha256(paths: Sequence[Path]) -> str:
    lines = "".join(
        f"{file_sha256(path)}  cases/{path.name}\n" for path in sorted(paths)
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def validate_receipt(
    receipt: dict[str, Any],
    *,
    repo_root: Path,
    protocol_sha256: str,
    execution_sha256: str,
) -> None:
    _require(receipt.get("schema_version") == 1, "calibration receipt schema mismatch")
    _require(receipt.get("receipt_id") == RECEIPT_ID, "calibration receipt ID mismatch")
    _require(
        receipt.get("status")
        == "declared_from_execution_outputs_before_quota_derivation",
        "calibration receipt status mismatch",
    )
    _require(receipt.get("claim_eligible") is False, "calibration receipt must be claim-ineligible")
    _require(receipt.get("protocol", {}).get("sha256") == protocol_sha256, "receipt protocol mismatch")
    _require(receipt.get("execution", {}).get("sha256") == execution_sha256, "receipt execution mismatch")
    gate = receipt.get("acceptance_gate", {})
    gate_path = Path(gate.get("path", ""))
    if not gate_path.is_absolute():
        gate_path = repo_root / gate_path
    _require(file_sha256(gate_path) == gate.get("sha256"), "receipt acceptance gate mismatch")
    artifact = receipt.get("calibration_full", {})
    expected_artifact = {
        "case_count": 30,
        "layer_record_count": 1080,
        "failure_count": 0,
    }
    for field, expected in expected_artifact.items():
        _require(artifact.get(field) == expected, f"receipt calibration {field} mismatch")
    authorization = receipt.get("derivation_authorization", {})
    _require(authorization.get("anchor_indices") == list(range(5)), "receipt anchor authorization mismatch")
    _require(authorization.get("score_metric") == "joint_post_o_proj_mse", "receipt score metric mismatch")
    _require(
        authorization.get("quota_rule")
        == {
            "top_12_layers": 48,
            "middle_12_layers": 32,
            "bottom_12_layers": 16,
            "total_channels": 1152,
        },
        "receipt quota rule mismatch",
    )
    for field in ("screen_metrics", "holdout_metrics", "reserved_unseen_metrics"):
        _require(authorization.get(field) is False, f"receipt improperly authorizes {field}")


def _validate_cache(record: dict[str, Any]) -> None:
    method = record["method"]
    prompt_length = record["input"]["prompt_length"]
    config = method["config"]
    expected_length = prompt_length + 1
    sink = min(expected_length, config["sink_length"])
    non_sink = expected_length - sink
    residual = config["residual_length"]
    expected_key = sink + non_sink - non_sink % residual
    expected_value = max(sink, expected_length - residual)
    cache = record.get("cache", {})
    expected = (
        cache.get("reported_seq_length") == expected_length,
        cache.get("expected_seq_length") == expected_length,
        cache.get("layer_count") == 36,
        cache.get("tensor_dtypes") == ["torch.float16"],
        cache.get("tensors_finite") is True,
        cache.get("key_quantized_lengths") == [expected_key],
        cache.get("value_quantized_lengths") == [expected_value],
        cache.get("one_bit_channels") == [0],
        cache.get("two_bit_channels") == [32],
        cache.get("one_bit_indices_empty") is True,
        cache.get("value_adaptive") is False,
    )
    _require(all(expected), f"cache mechanics mismatch: {record.get('case_id')}")


def derive_layer_quota_plans(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group = (record["method"]["family_id"], record["input"]["prompt_length"])
        groups[group].append(record)
    plans = []
    for family_id in FAMILIES:
        for prompt_length in PROMPT_LENGTHS:
            group_records = groups.get((family_id, prompt_length), [])
            _require(len(group_records) == 5, f"calibration group count mismatch: {family_id}/{prompt_length}")
            anchor_indices = sorted(record["input"]["anchor_index"] for record in group_records)
            _require(anchor_indices == list(range(5)), f"calibration anchors mismatch: {family_id}/{prompt_length}")
            scores = []
            for layer_idx in range(36):
                values = [
                    float(record["layer_metrics"][layer_idx]["metrics"]["joint_post_o_proj_mse"])
                    for record in group_records
                ]
                _require(all(math.isfinite(value) and value >= 0 for value in values), "invalid layer score")
                scores.append(math.fsum(values) / 5)
            quotas = calibrated_tiered_two_bit_quotas(scores)
            ranking = sorted(range(36), key=lambda layer_idx: (-scores[layer_idx], layer_idx))
            _require([quotas.count(value) for value in (48, 32, 16)] == [12, 12, 12], "quota tier count mismatch")
            _require(sum(quotas) == 1152, "quota total mismatch")
            plans.append({
                "family_id": family_id,
                "sink_length": 32 if "sink32" in family_id else 64,
                "prompt_length": prompt_length,
                "anchor_indices": list(range(5)),
                "layer_scores": scores,
                "ranked_layer_indices": ranking,
                "layer_two_bit_channel_quotas": quotas,
                "quota_counts": {"48": 12, "32": 12, "16": 12},
                "quota_total": sum(quotas),
            })
    _require(set(groups) == {(family, length) for family in FAMILIES for length in PROMPT_LENGTHS}, "unexpected calibration groups")
    return plans


def _validate_output_identity(
    *,
    receipt: dict[str, Any],
    expected_cases: Sequence[dict[str, Any]],
) -> tuple[Path, dict[str, Any], dict[str, Any], list[Path]]:
    artifact = receipt["calibration_full"]
    root = Path(artifact["output_dir"])
    identity_path = root / "run_identity.json"
    summary_path = root / "summary.json"
    log_path = Path(artifact["execution_log"])
    _require(file_sha256(identity_path) == artifact["run_identity_sha256"], "run identity hash mismatch")
    _require(file_sha256(summary_path) == artifact["summary_sha256"], "summary hash mismatch")
    _require(file_sha256(log_path) == artifact["execution_log_sha256"], "execution log hash mismatch")
    _require(log_path.stat().st_size == artifact["execution_log_size_bytes"], "execution log size mismatch")
    log_text = log_path.read_text(encoding="utf-8")
    for forbidden in ("Traceback (most recent call last)", "ERROR conda.cli.main_run"):
        _require(forbidden not in log_text, f"execution log contains {forbidden}")
    _require("QWEN3 CAGE-V3 CALIBRATION FULL END" in log_text, "execution log lacks end marker")
    identity = load_json(identity_path)
    summary = load_json(summary_path)
    expected_ids = [case["case_id"] for case in expected_cases]
    _require(
        identity.get("source_state")
        == {"git_commit": artifact["execution_commit"], "dirty": False},
        "calibration execution source mismatch",
    )
    _require(identity.get("expected_case_ids") == expected_ids, "ordered calibration case IDs mismatch")
    for field, expected in (
        ("schema_version", 1),
        ("status", "pass"),
        ("claim_eligible", False),
        ("stage", "calibration_full"),
        ("expected_cases", 30),
        ("completed_cases", 30),
        ("new_cases", 30),
        ("resumed_cases", 0),
        ("failure_records", 0),
        ("case_ids_sha256", artifact["case_ids_sha256"]),
    ):
        _require(summary.get(field) == expected, f"calibration summary {field} mismatch")
    _require(summary.get("identity") == identity, "summary/run identity mismatch")
    _require(all(summary.get("model", {}).get("checks", {}).values()), "model checks did not all pass")
    paths = sorted((root / "cases").glob("*.json"))
    _require(len(paths) == 30, "calibration case file count mismatch")
    failures = sorted((root / "failures").glob("*.json")) if (root / "failures").exists() else []
    _require(not failures, "calibration failure files remain")
    _require(
        shell_case_manifest_sha256(paths) == artifact["shell_case_file_manifest_sha256"],
        "calibration shell case manifest mismatch",
    )
    return root, identity, summary, paths


def validate_and_derive_quota_plan(
    *,
    receipt: dict[str, Any],
    receipt_sha256: str,
    expected_cases: Sequence[dict[str, Any]],
    protocol_sha256: str,
    execution_sha256: str,
    gate_sha256: str,
) -> dict[str, Any]:
    _, identity, summary, paths = _validate_output_identity(
        receipt=receipt,
        expected_cases=expected_cases,
    )
    expected_by_id = {case["case_id"]: case for case in expected_cases}
    validated_records = []
    scientific_payload = []
    layer_record_count = 0
    for path in paths:
        record = load_json(path)
        case_id = record.get("case_id")
        _require(path.stem == case_id, f"case filename/ID mismatch: {path}")
        _require(case_id in expected_by_id, f"unexpected calibration case: {case_id}")
        expected = expected_by_id[case_id]
        _require(record.get("schema_version") == 1 and record.get("status") == "completed", f"case schema mismatch: {case_id}")
        _require(record.get("identity") == identity, f"case identity mismatch: {case_id}")
        _require(record.get("model") == summary["model"], f"case model mismatch: {case_id}")
        _require(record.get("method") == expected["method"], f"case method mismatch: {case_id}")
        _require(record.get("input") == expected["input"], f"case input mismatch: {case_id}")
        config = record["method"]["config"]
        memory = estimate_qwen3_cage_v2_bytes(
            seq_len=record["method"]["prompt_length"],
            residual_length=config["residual_length"],
            one_bit_channels=config["one_bit_channels"],
            two_bit_channels=config["two_bit_channels"],
            sink_length=config["sink_length"],
        )
        _require(record.get("memory") == memory, f"case memory mismatch: {case_id}")
        _require(memory["model_total_bytes"] == record["method"]["packed_bytes"], f"packed bytes mismatch: {case_id}")
        _require(memory["model_total_bytes"] <= record["method"]["target_bytes"], f"target bytes exceeded: {case_id}")
        layers = record.get("layer_metrics", [])
        validate_layer_records(layers)
        validate_aggregates(record.get("aggregates", {}), layers)
        _require(record["aggregates"] == aggregate_layer_metrics(layers), f"case aggregate mismatch: {case_id}")
        _validate_cache(record)
        validated_records.append(record)
        layer_record_count += len(layers)
        scientific_payload.append({field: record[field] for field in SCIENTIFIC_FIELDS})
    _require(layer_record_count == 1080, "calibration layer record total mismatch")
    scientific_payload.sort(key=lambda record: record["case_id"])

    plans = derive_layer_quota_plans(validated_records)
    return {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "status": "derived_from_frozen_calibration_only_before_screen_execution",
        "claim_eligible": False,
        "receipt_sha256": receipt_sha256,
        "protocol_sha256": protocol_sha256,
        "execution_sha256": execution_sha256,
        "acceptance_gate_sha256": gate_sha256,
        "calibration_case_count": 30,
        "calibration_layer_record_count": 1080,
        "calibration_scientific_payload_sha256": canonical_sha256(scientific_payload),
        "score_metric": "joint_post_o_proj_mse",
        "score_aggregation": "arithmetic_mean_over_five_calibration_anchors_separately_per_sink_family_and_prompt_length",
        "ranking_rule": "descending_score_then_ascending_layer_idx",
        "quota_rule": {"top_12_layers": 48, "middle_12_layers": 32, "bottom_12_layers": 16, "total_channels": 1152},
        "screen_metrics_consumed": False,
        "holdout_metrics_consumed": False,
        "reserved_unseen_metrics_consumed": False,
        "plans": plans,
    }


__all__ = [
    "CageV3CalibrationPostrunError",
    "FAMILIES",
    "PLAN_ID",
    "PROMPT_LENGTHS",
    "RECEIPT_ID",
    "derive_layer_quota_plans",
    "shell_case_manifest_sha256",
    "validate_and_derive_quota_plan",
    "validate_receipt",
]
