from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.qwen3_cage_v3_execution import canonical_sha256, load_json
from utils.qwen3_cage_v3_manifest import validate_cage_v3_manifest
from utils.qwen3_cage_v3_protocol import file_sha256, load_cage_v3_protocol
from utils.qwen3_cage_v3_screen import SCIENTIFIC_FIELDS, expand_screen_methods, validate_quota_plan


EXECUTION_ID = "qwen3-8b-cage-v3-holdout-execution-v1"
DECISION_ID = "qwen3-8b-cage-v3-development-screen-decision-v1"
EXPECTED_DECISION_SHA256 = "60e90b4176812a8a6e4d82886776d1cfa211ef2797a07a9a73f8bdbc906414c2"
SELECTED_FAMILY_ID = "pure-sr2-sink32-calibrated"
PARTITIONS = ("cage_qwen3", "kitty_qwen3")
STAGES = ("holdout_acceptance", "holdout_full")
HOLDOUT_ANCHORS = list(range(10, 20))


class CageV3HoldoutError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3HoldoutError(message)


def _resolve(path: str | Path, repo_root: Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else repo_root / result


def _candidate_summary(report: dict[str, Any]) -> dict[str, Any]:
    gates = report["gates"]
    return {
        "family_id": report["family_id"],
        "memory_all_three": gates["memory_all_three"],
        "beats_cage_v2_all_three": gates["beats_cage_v2_all_three"],
        "kitty_pro_no_worse_length_count": gates["kitty_pro_no_worse_length_count"],
        "family_pass": report["family_pass"],
        "overall_15_case_mean_delta_vs_kitty_pro": report["overall_15_case_versus_kitty_pro"]["mean_delta"],
    }


def validate_screen_decision(
    decision: dict[str, Any],
    *,
    decision_path: Path,
    repo_root: Path,
    verify_artifacts: bool,
) -> None:
    _require(file_sha256(decision_path) == EXPECTED_DECISION_SHA256, "screen decision hash mismatch")
    _require(decision.get("schema_version") == 1 and decision.get("decision_id") == DECISION_ID, "screen decision identity mismatch")
    _require(
        decision.get("status") == "frozen_after_screen_interpretation_before_any_holdout_metric",
        "screen decision status mismatch",
    )
    _require(decision.get("claim_eligible") is False, "screen decision must be claim-ineligible")
    receipt = decision.get("screen_postrun_receipt", {})
    receipt_path = _resolve(receipt.get("path", ""), repo_root)
    _require(file_sha256(receipt_path) == receipt.get("sha256"), "screen receipt changed after decision")
    frozen = decision.get("decision", {})
    _require(
        (
            frozen.get("analysis_id"),
            frozen.get("decision_status"),
            frozen.get("advanced_family_count"),
            frozen.get("selected_family_id"),
        )
        == (
            "qwen3-8b-cage-v3-development-screen-analysis-v1",
            "selected_one_candidate_for_frozen_holdout_authorization",
            1,
            SELECTED_FAMILY_ID,
        ),
        "screen selection changed",
    )
    _require(frozen.get("bootstrap_used_for_gate") is False, "bootstrap entered screen gate")
    expected_candidates = [
        ("pure-sr2-sink32-uniform", True, False, 1, False),
        (SELECTED_FAMILY_ID, True, True, 2, True),
        ("pure-sr2-sink64-calibrated", True, True, 2, True),
    ]
    observed_candidates = [
        (
            row.get("family_id"),
            row.get("memory_all_three"),
            row.get("beats_cage_v2_all_three"),
            row.get("kitty_pro_no_worse_length_count"),
            row.get("family_pass"),
        )
        for row in decision.get("all_screen_candidates", [])
    ]
    _require(observed_candidates == expected_candidates, "screen candidate reporting changed")
    authorization = decision.get("holdout_authorization", {})
    _require(
        authorization
        == {
            "authorized": True,
            "anchor_indices": HOLDOUT_ANCHORS,
            "selected_candidate_family_count": 1,
            "selected_candidate_family_id": SELECTED_FAMILY_ID,
            "cage_v2_controls_required": True,
            "kitty_pro_required": True,
            "acceptance_anchor_indices": [10],
            "fresh_acceptance_repeats_required": 2,
            "required_consistency": "bitwise_equal_json_numeric_payload",
            "full_holdout_authorized_only_after_joint_acceptance_gate": True,
            "reserved_unseen_metrics": False,
            "end_to_end_quality": False,
        },
        "holdout authorization changed",
    )
    _require(
        decision.get("consumption_state_at_freeze")
        == {"holdout_metrics_consumed": False, "reserved_unseen_metrics_consumed": False},
        "screen decision consumed protected metrics",
    )
    if not verify_artifacts:
        return
    artifacts = decision["screen_analysis"]
    paths = {
        "analysis": Path(artifacts["analysis_path"]),
        "manifest": Path(artifacts["manifest_path"]),
        "markdown": Path(artifacts["markdown_path"]),
        "execution_log": Path(artifacts["execution_log"]),
    }
    for name, path in paths.items():
        _require(file_sha256(path) == artifacts[f"{name}_sha256"], f"screen {name} artifact changed")
        _require(path.stat().st_size == artifacts[f"{name}_size_bytes"], f"screen {name} size changed")
    analysis = load_json(paths["analysis"])
    manifest = load_json(paths["manifest"])
    _require(analysis.get("status") == "pass" and analysis.get("claim_eligible") is False, "screen analysis status mismatch")
    _require(analysis.get("decision_status") == frozen["decision_status"], "screen analysis decision mismatch")
    _require(analysis.get("advanced_family_count") == 1, "screen analysis advanced count mismatch")
    _require(analysis.get("advanced_families") == [{
        "rank": 1,
        "family_id": SELECTED_FAMILY_ID,
        "overall_15_case_mean_delta_vs_kitty_pro": frozen["selected_overall_15_case_mean_delta_vs_kitty_pro"],
        "total_model_total_bytes_across_lengths": frozen["selected_total_model_total_bytes_across_lengths"],
    }], "screen analysis selected payload mismatch")
    _require([_candidate_summary(row) for row in analysis.get("family_reports", [])] == decision["all_screen_candidates"], "screen analysis candidate payload mismatch")
    _require(analysis.get("holdout_authorized_after_decision_freeze") is True, "screen analysis did not authorize freeze")
    _require(analysis.get("holdout_metrics_consumed") is False, "screen analysis consumed holdout")
    _require(analysis.get("reserved_unseen_metrics_consumed") is False, "screen analysis consumed reserved metrics")
    _require(
        manifest.get("status") == "pass"
        and manifest.get("advanced_family_count") == 1
        and manifest.get("holdout_authorized_after_decision_freeze") is True,
        "screen analysis manifest mismatch",
    )
    _require(
        manifest.get("outputs", {}).get(paths["analysis"].name, {}).get("sha256") == artifacts["analysis_sha256"]
        and manifest["outputs"].get(paths["markdown"].name, {}).get("sha256") == artifacts["markdown_sha256"],
        "screen analysis manifest output hashes mismatch",
    )


def load_holdout_execution(
    path: str | Path,
    *,
    repo_root: Path,
    verify_artifacts: bool,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = Path(path).resolve()
    execution = load_json(source)
    _require(execution.get("schema_version") == 1 and execution.get("execution_id") == EXECUTION_ID, "holdout execution identity mismatch")
    _require(execution.get("status") == "frozen_after_screen_decision_before_any_holdout_metric", "holdout execution status mismatch")
    _require(execution.get("claim_eligible") is False, "holdout execution must be claim-ineligible")
    protocol_path = _resolve(execution["protocol"]["path"], repo_root)
    protocol, protocol_sha256 = load_cage_v3_protocol(protocol_path)
    _require(protocol_sha256 == execution["protocol"]["sha256"], "holdout protocol hash mismatch")
    decision_path = _resolve(execution["screen_decision"]["path"], repo_root)
    _require(file_sha256(decision_path) == execution["screen_decision"]["sha256"], "holdout decision hash mismatch")
    decision = load_json(decision_path)
    validate_screen_decision(
        decision,
        decision_path=decision_path,
        repo_root=repo_root,
        verify_artifacts=verify_artifacts,
    )
    _require(execution["screen_decision"]["selected_family_id"] == SELECTED_FAMILY_ID, "holdout selected family mismatch")
    manifest: dict[str, Any] = {}
    manifest_path = Path(execution["input_manifest"]["path"])
    if verify_artifacts:
        _require(file_sha256(manifest_path) == execution["input_manifest"]["sha256"], "holdout manifest hash mismatch")
        manifest = load_json(manifest_path)
        validate_cage_v3_manifest(manifest, protocol=protocol, protocol_sha256=protocol_sha256)
        _require(len(manifest["cases"]) == execution["input_manifest"]["case_count"], "holdout manifest count mismatch")
    plan_path = _resolve(execution["quota_plan"]["path"], repo_root)
    _require(file_sha256(plan_path) == execution["quota_plan"]["sha256"], "holdout quota plan hash mismatch")
    plan = load_json(plan_path)
    validate_quota_plan(plan)
    _require(plan["calibration_scientific_payload_sha256"] == execution["quota_plan"]["calibration_scientific_payload_sha256"], "holdout quota evidence mismatch")
    _require(tuple(execution.get("scientific_payload_fields", ())) == SCIENTIFIC_FIELDS, "holdout scientific fields changed")
    _require(
        execution.get("authorization")
        == {
            "calibration_metrics": False,
            "screen_metrics": False,
            "holdout_metrics": True,
            "reserved_unseen_metrics": False,
            "end_to_end_quality": False,
        },
        "holdout authorization changed",
    )
    _require(execution.get("stages", {}).get("holdout_acceptance", {}).get("anchor_indices") == [10], "holdout acceptance anchor changed")
    _require(execution["stages"].get("holdout_full", {}).get("anchor_indices") == HOLDOUT_ANCHORS, "holdout full anchors changed")
    _require(execution.get("partitions", {}).get("cage_qwen3", {}).get("method_count") == 6, "holdout CAGE method count changed")
    _require(execution["partitions"].get("kitty_qwen3", {}).get("method_count") == 3, "holdout Kitty method count changed")
    return execution, file_sha256(source), protocol, manifest, plan, decision


def expand_holdout_methods(*, protocol: dict[str, Any], plan: dict[str, Any], partition: str) -> list[dict[str, Any]]:
    _require(partition in PARTITIONS, "unsupported holdout partition")
    methods = expand_screen_methods(protocol=protocol, plan=plan, partition=partition)
    if partition == "kitty_qwen3":
        _require(len(methods) == 3 and {row["family_id"] for row in methods} == {"kitty-pro-25pct"}, "holdout Kitty methods changed")
        return methods
    selected = [
        method for method in methods
        if method["family_id"] in ("cage-v2-best-control", SELECTED_FAMILY_ID)
    ]
    _require(len(selected) == 6, "holdout CAGE method count mismatch")
    _require(sum(row["family_id"] == SELECTED_FAMILY_ID for row in selected) == 3, "holdout selected family points mismatch")
    _require(sum(row["family_id"] == "cage-v2-best-control" for row in selected) == 3, "holdout control points mismatch")
    return selected


def expand_holdout_cases(
    *,
    execution: dict[str, Any],
    execution_sha256: str,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    plan: dict[str, Any],
    partition: str,
    stage: str,
) -> list[dict[str, Any]]:
    _require(stage in STAGES, "unsupported holdout stage")
    methods = expand_holdout_methods(protocol=protocol, plan=plan, partition=partition)
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
    key = "acceptance_case_count" if stage == "holdout_acceptance" else "full_case_count"
    expected = execution["partitions"][partition][key]
    _require(len(cases) == expected and len({case["case_id"] for case in cases}) == expected, "holdout expanded case count mismatch")
    return cases


__all__ = [
    "CageV3HoldoutError",
    "DECISION_ID",
    "EXECUTION_ID",
    "EXPECTED_DECISION_SHA256",
    "HOLDOUT_ANCHORS",
    "PARTITIONS",
    "SELECTED_FAMILY_ID",
    "STAGES",
    "expand_holdout_cases",
    "expand_holdout_methods",
    "load_holdout_execution",
    "validate_screen_decision",
]
