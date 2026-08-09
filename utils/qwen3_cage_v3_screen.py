from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v3_execution import canonical_sha256, load_json
from utils.qwen3_cage_v3_manifest import validate_cage_v3_manifest
from utils.qwen3_cage_v3_protocol import file_sha256, load_cage_v3_protocol
from utils.qwen3_memory import estimate_qwen3_kitty_bytes
from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes


EXECUTION_ID = "qwen3-8b-cage-v3-screen-execution-v1"
PLAN_ID = "qwen3-8b-cage-v3-calibrated-layer-quota-plan-v1"
PARTITIONS = ("cage_qwen3", "kitty_qwen3")
STAGES = ("screen_acceptance", "screen_full")
SCIENTIFIC_FIELDS = (
    "case_id",
    "method",
    "input",
    "memory",
    "layer_metrics",
    "aggregates",
    "cache",
)


class CageV3ScreenError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3ScreenError(message)


def validate_quota_plan(plan: dict[str, Any]) -> None:
    _require(plan.get("schema_version") == 1 and plan.get("plan_id") == PLAN_ID, "quota plan identity mismatch")
    _require(
        plan.get("status") == "derived_from_frozen_calibration_only_before_screen_execution",
        "quota plan status mismatch",
    )
    _require(plan.get("claim_eligible") is False, "quota plan must be claim-ineligible")
    _require(plan.get("calibration_case_count") == 30, "quota plan calibration count mismatch")
    _require(plan.get("calibration_layer_record_count") == 1080, "quota plan layer count mismatch")
    for field in ("screen_metrics_consumed", "holdout_metrics_consumed", "reserved_unseen_metrics_consumed"):
        _require(plan.get(field) is False, f"quota plan consumed {field}")
    expected_groups = [
        (family, length)
        for family in ("pure-sr2-sink32-uniform32", "pure-sr2-sink64-uniform32")
        for length in (1024, 2048, 4032)
    ]
    records = plan.get("plans", [])
    _require([(row.get("family_id"), row.get("prompt_length")) for row in records] == expected_groups, "quota plan groups mismatch")
    for row in records:
        scores = row.get("layer_scores", [])
        ranking = sorted(range(36), key=lambda index: (-scores[index], index)) if len(scores) == 36 else []
        _require(row.get("ranked_layer_indices") == ranking, "quota plan ranking mismatch")
        quotas = [32] * 36
        for layer_idx in ranking[:12]:
            quotas[layer_idx] = 48
        for layer_idx in ranking[-12:]:
            quotas[layer_idx] = 16
        _require(row.get("layer_two_bit_channel_quotas") == quotas, "quota plan allocation mismatch")
        _require(quotas.count(48) == quotas.count(32) == quotas.count(16) == 12, "quota plan tier count mismatch")
        _require(sum(quotas) == 1152, "quota plan total mismatch")


def load_screen_execution(
    path: str | Path,
    *,
    repo_root: Path,
    verify_artifacts: bool,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = Path(path).resolve()
    execution = load_json(source)
    _require(execution.get("schema_version") == 1 and execution.get("execution_id") == EXECUTION_ID, "screen execution identity mismatch")
    _require(execution.get("status") == "frozen_after_calibration_before_any_screen_metric", "screen execution status mismatch")
    _require(execution.get("claim_eligible") is False, "screen execution must be claim-ineligible")
    protocol_path = Path(execution["protocol"]["path"])
    if not protocol_path.is_absolute():
        protocol_path = repo_root / protocol_path
    protocol, protocol_sha256 = load_cage_v3_protocol(protocol_path)
    _require(protocol_sha256 == execution["protocol"]["sha256"], "screen protocol hash mismatch")
    manifest_path = Path(execution["input_manifest"]["path"])
    manifest = {}
    if verify_artifacts:
        _require(file_sha256(manifest_path) == execution["input_manifest"]["sha256"], "screen manifest hash mismatch")
        manifest = load_json(manifest_path)
        validate_cage_v3_manifest(manifest, protocol=protocol, protocol_sha256=protocol_sha256)
        _require(len(manifest["cases"]) == execution["input_manifest"]["case_count"], "screen manifest count mismatch")
    plan_path = Path(execution["quota_plan"]["path"])
    if not plan_path.is_absolute():
        plan_path = repo_root / plan_path
    _require(file_sha256(plan_path) == execution["quota_plan"]["sha256"], "screen quota plan hash mismatch")
    plan = load_json(plan_path)
    validate_quota_plan(plan)
    _require(
        plan["calibration_scientific_payload_sha256"]
        == execution["quota_plan"]["calibration_scientific_payload_sha256"],
        "screen quota evidence mismatch",
    )
    _require(tuple(execution.get("scientific_payload_fields", ())) == SCIENTIFIC_FIELDS, "screen scientific fields changed")
    _require(
        execution.get("authorization")
        == {
            "calibration_metrics": False,
            "screen_metrics": True,
            "holdout_metrics": False,
            "reserved_unseen_metrics": False,
            "end_to_end_quality": False,
        },
        "screen authorization changed",
    )
    expected_stages = {"screen_acceptance": ([5], 1), "screen_full": (list(range(5, 10)), 5)}
    for stage, (anchors, _) in expected_stages.items():
        _require(execution["stages"][stage]["anchor_indices"] == anchors, f"{stage} anchors changed")
    _require(
        execution.get("partitions", {}).get("cage_qwen3", {}).get("method_count") == 12
        and execution["partitions"]["kitty_qwen3"]["method_count"] == 3,
        "screen partition method counts changed",
    )
    return execution, file_sha256(source), protocol, manifest, plan


def _plan_quotas(plan: dict[str, Any], *, sink_length: int, prompt_length: int) -> list[int]:
    family = f"pure-sr2-sink{sink_length}-uniform32"
    matches = [
        row for row in plan["plans"]
        if row["family_id"] == family and row["prompt_length"] == prompt_length
    ]
    _require(len(matches) == 1, "matching calibrated quota plan missing")
    return list(matches[0]["layer_two_bit_channel_quotas"])


def expand_screen_methods(
    *, protocol: dict[str, Any], plan: dict[str, Any], partition: str
) -> list[dict[str, Any]]:
    _require(partition in PARTITIONS, "unsupported screen partition")
    if partition == "kitty_qwen3":
        methods = []
        for point in protocol["kitty_pro_targets"]:
            report = estimate_qwen3_kitty_bytes(
                seq_len=point["prompt_length"], boosted_channels=point["boosted_channels"]
            )
            _require(report["model_total_bytes"] == point["target_bytes"], "Kitty-Pro frozen bytes mismatch")
            methods.append({
                "id": point["method_id"],
                "name": "kitty",
                "family_id": "kitty-pro-25pct",
                "prompt_length": point["prompt_length"],
                "packed_bytes": point["target_bytes"],
                "target_bytes": point["target_bytes"],
                "config": {"boosted_channels": point["boosted_channels"]},
            })
        return methods
    methods = []
    targets = {point["prompt_length"]: point["target_bytes"] for point in protocol["kitty_pro_targets"]}
    for point in protocol["cage_v2_best_controls"]:
        methods.append({
            "id": point["method_id"],
            "name": "cage_v2",
            "family_id": "cage-v2-best-control",
            "prompt_length": point["prompt_length"],
            "packed_bytes": point["packed_bytes"],
            "target_bytes": targets[point["prompt_length"]],
            "config": {
                "residual_length": point["residual_length"],
                "sink_length": point["sink_length"],
                "one_bit_channels": point["one_bit_channels"],
                "two_bit_channels": point["two_bit_channels"],
            },
        })
    for family in protocol["candidate_families"]:
        for point in family["points"]:
            if family["layer_allocation"] == "uniform32":
                quotas: int | list[int] = 32
            else:
                quotas = _plan_quotas(
                    plan,
                    sink_length=family["sink_length"],
                    prompt_length=point["prompt_length"],
                )
            report = estimate_qwen3_cage_v2_bytes(
                seq_len=point["prompt_length"],
                residual_length=point["residual_length"],
                one_bit_channels=0,
                two_bit_channels=quotas,
                sink_length=family["sink_length"],
            )
            _require(report["model_total_bytes"] == point["packed_bytes"], "CAGE-v3 frozen bytes mismatch")
            methods.append({
                "id": point["method_id"],
                "name": "cage_v2",
                "family_id": family["family_id"],
                "prompt_length": point["prompt_length"],
                "packed_bytes": point["packed_bytes"],
                "target_bytes": point["target_bytes"],
                "layer_allocation": family["layer_allocation"],
                "config": {
                    "residual_length": point["residual_length"],
                    "sink_length": family["sink_length"],
                    "one_bit_channels": 0,
                    "two_bit_channels": quotas,
                },
            })
    _require(len(methods) == 12, "CAGE screen method count mismatch")
    return methods


def expand_screen_cases(
    *,
    execution: dict[str, Any],
    execution_sha256: str,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    plan: dict[str, Any],
    partition: str,
    stage: str,
) -> list[dict[str, Any]]:
    _require(stage in STAGES, "unsupported screen stage")
    methods = expand_screen_methods(protocol=protocol, plan=plan, partition=partition)
    base = {
        (case["identity"]["anchor_index"], case["identity"]["prompt_length"]): case
        for case in manifest["cases"]
    }
    cases = []
    for anchor_index in execution["stages"][stage]["anchor_indices"]:
        for method in methods:
            source = base[(anchor_index, method["prompt_length"])]
            identity = {
                "execution_id": execution["execution_id"],
                "execution_sha256": execution_sha256,
                "partition": partition,
                "stage": stage,
                "method_id": method["id"],
                "base_case_id": source["case_id"],
            }
            cases.append({
                "case_id": canonical_sha256(identity)[:24],
                "method": method,
                "input": source["identity"],
                "prompt_ids": source["prompt_ids"],
                "continuation_ids": source["continuation_ids"],
            })
    expected = execution["partitions"][partition]["acceptance_case_count" if stage == "screen_acceptance" else "full_case_count"]
    _require(len(cases) == expected and len({case["case_id"] for case in cases}) == expected, "screen expanded case count mismatch")
    return cases


__all__ = [
    "CageV3ScreenError",
    "EXECUTION_ID",
    "PARTITIONS",
    "SCIENTIFIC_FIELDS",
    "STAGES",
    "expand_screen_cases",
    "expand_screen_methods",
    "load_screen_execution",
    "validate_quota_plan",
]
