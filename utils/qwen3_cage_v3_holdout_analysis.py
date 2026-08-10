from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Any, Sequence

from utils.qwen3_cage_v3_analysis import paired_summary
from utils.qwen3_cage_v3_protocol import file_sha256


EXPECTED_HOLDOUT_RECEIPT_SHA256 = "6eeeb54c45f9ca0ca3cec015c3bd33e408a067755bb651f36a8947fa57d905ac"
HOLDOUT_ANCHORS = list(range(10, 20))
LENGTHS = [1024, 2048, 4032]


class CageV3HoldoutAnalysisError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3HoldoutAnalysisError(message)


def validate_holdout_receipt(receipt: dict[str, Any], *, receipt_path: Path) -> None:
    _require(
        file_sha256(receipt_path) == EXPECTED_HOLDOUT_RECEIPT_SHA256,
        "holdout receipt hash mismatch",
    )
    _require(receipt.get("schema_version") == 1, "holdout receipt schema mismatch")
    _require(
        receipt.get("receipt_id") == "qwen3-8b-cage-v3-development-holdout-postrun-receipt-v1",
        "holdout receipt ID mismatch",
    )
    _require(
        receipt.get("status") == "frozen_after_joint_postrun_audit_before_holdout_interpretation",
        "holdout receipt status mismatch",
    )
    _require(receipt.get("claim_eligible") is False, "holdout receipt must be claim-ineligible")
    _require(receipt.get("interpretation_performed") is False, "holdout receipt already interpreted")
    _require(
        receipt.get("audit")
        == {
            "path": "/root/autodl-tmp/qwen3_cage_v3/holdout_postrun_audit_df80cb0.json",
            "sha256": "d71f4e06eb09710996677ee0b5b62213d90821d94663c35a70dd96873b119168",
            "size_bytes": 2528,
            "execution_log": "/root/autodl-tmp/kitty_setup_audit/qwen3_cage_v3_holdout_postrun_df80cb0.log",
            "execution_log_sha256": "cf507c338006073201b3d89f4b16081000e6d0e9cc954f05716ed337fcd9e1a3",
            "execution_log_size_bytes": 11684,
            "source_commit": "df80cb07bd0710d7694839f7c2c7989fdbe88747",
            "validator_sha256": "c888810a79cf44be84ae8bde5b79136cc3cf60995a3cfaff3422665845c3dbf9",
            "postrun_utils_sha256": "16ebeaf18ef41914b4387d24fb9def415b380ee7689e273f890116a7d1fa3506",
            "tests_sha256": "deb343006c74172618cf6b10e091315cb0c831b75a559283249de6333fcb2bc8",
            "tests_passed": 52,
        },
        "holdout audit provenance mismatch",
    )
    _require(
        receipt.get("frozen_inputs")
        == {
            "execution_source_commit": "391c78103d9bd6a86e0c19f1aa84ee09bf586e67",
            "execution_sha256": "b8a39f59247346e6a893e05d79d44082e64c6a38f3e0da618bf86f4195ccc2f1",
            "protocol_sha256": "c92e452b5eac99c7a015da21a82080cf61ddd070e677cc3664ed78bf657cbfe9",
            "manifest_sha256": "e53cdec987d8206ed6c21ba3be5c04f3751c25f2fee074fa12e30916e06404d7",
            "quota_plan_sha256": "01a1063651d4565a732474b9f27f7a674f434750e7aea2f55429f619bed91914",
            "screen_decision_sha256": "60e90b4176812a8a6e4d82886776d1cfa211ef2797a07a9a73f8bdbc906414c2",
            "acceptance_gate_sha256": "e83cdccdb1faf9005ab7a5cbdbecd3c63ede04e27ed9b94862e718a5eb7cd576",
            "artifact_manifest_sha256": "25ea16b8d43d3e88b26d2585c7553eab1b3e11f7f76a6efe7270631247523a13",
        },
        "holdout frozen inputs mismatch",
    )
    _require(
        receipt.get("audit_summary")
        == {
            "status": "pass",
            "case_count": 90,
            "layer_record_count": 3240,
            "failure_count": 0,
            "joint_scientific_payload_sha256": "fc2ae7f310adf98251900d235c5e21c6769131e38b89641679c9535a4e80d5f2",
            "selected_family_id": "pure-sr2-sink32-calibrated",
            "holdout_metrics_consumed": True,
            "reserved_unseen_metrics_consumed": False,
            "end_to_end_quality_authorized": False,
            "interpretation_performed": False,
        },
        "holdout audit summary mismatch",
    )
    partitions = receipt.get("partitions", {})
    _require(tuple(partitions) == ("cage_qwen3", "kitty_qwen3"), "holdout receipt partitions mismatch")
    _require(
        (partitions["cage_qwen3"].get("case_count"), partitions["cage_qwen3"].get("layer_record_count"))
        == (60, 2160),
        "CAGE holdout receipt totals mismatch",
    )
    _require(
        (partitions["kitty_qwen3"].get("case_count"), partitions["kitty_qwen3"].get("layer_record_count"))
        == (30, 1080),
        "Kitty holdout receipt totals mismatch",
    )
    _require(
        receipt.get("next_authorization")
        == {
            "development_holdout_interpretation": True,
            "reserved_unseen_metrics": False,
            "end_to_end_quality": False,
            "new_candidate_selection": False,
            "protocol_mutation": False,
        },
        "holdout receipt authorization mismatch",
    )


def _index(records: Sequence[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        method_id = record["method"]["id"]
        anchor = int(record["input"]["anchor_index"])
        _require(anchor not in result[method_id], f"duplicate holdout record: {method_id}/{anchor}")
        result[method_id][anchor] = record
    return dict(result)


def _records(index: dict[str, dict[int, dict[str, Any]]], method_id: str) -> dict[int, dict[str, Any]]:
    records = index.get(method_id, {})
    _require(sorted(records) == HOLDOUT_ANCHORS, f"method lacks frozen holdout grid: {method_id}")
    return records


def _values(index: dict[str, dict[int, dict[str, Any]]], method_id: str) -> dict[int, float]:
    return {
        anchor: float(record["aggregates"]["joint_post_o_proj_mse"]["mean"])
        for anchor, record in _records(index, method_id).items()
    }


def _bytes(index: dict[str, dict[int, dict[str, Any]]], method_id: str) -> int:
    values = {
        int(record["memory"]["model_total_bytes"])
        for record in _records(index, method_id).values()
    }
    _require(len(values) == 1, f"method packed bytes vary across holdout anchors: {method_id}")
    return values.pop()


def _point_map(points: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(point["prompt_length"]): point for point in points}
    _require(sorted(result) == LENGTHS, "holdout point lengths mismatch")
    return result


def build_holdout_analysis(
    *,
    protocol: dict[str, Any],
    decision: dict[str, Any],
    receipt: dict[str, Any],
    receipt_sha256: str,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    _require(len(records) == 90, "holdout analysis requires exactly 90 records")
    _require(
        protocol["development_data"]["partitions"]["holdout"] == HOLDOUT_ANCHORS,
        "holdout anchors changed",
    )
    _require(
        protocol["development_data"]["partitions"]["reserved_unseen"] == list(range(20, 50)),
        "reserved-unseen partition changed",
    )
    gate = protocol.get("holdout_gate", {})
    _require(gate.get("anchors") == HOLDOUT_ANCHORS, "holdout gate anchors mismatch")
    _require(
        gate.get("versus_cage_v2")
        == "selected candidate mean paired delta must be strictly negative at all three lengths",
        "holdout CAGE-v2 gate changed",
    )
    _require(
        gate.get("versus_kitty_pro")
        == "selected candidate mean paired delta must be nonpositive at at least two of three lengths",
        "holdout Kitty-Pro gate changed",
    )
    _require(gate.get("bootstrap") == "descriptive only and forbidden for gate decisions", "bootstrap gate changed")
    _require(gate.get("quality_run_before_holdout_pass") is False, "quality-run boundary changed")
    reporting = protocol.get("reporting_requirements", {})
    _require(reporting.get("report_all_lengths") is True, "all-length reporting changed")
    _require(reporting.get("report_unfavorable_results") is True, "unfavorable-result reporting changed")

    selected_family_id = decision["decision"]["selected_family_id"]
    _require(selected_family_id == "pure-sr2-sink32-calibrated", "selected holdout family changed")
    _require(
        receipt["audit_summary"]["selected_family_id"] == selected_family_id,
        "receipt selected family mismatch",
    )
    families = {row["family_id"]: row for row in protocol["candidate_families"]}
    _require(selected_family_id in families, "selected family is absent from frozen protocol")
    candidate_points = _point_map(families[selected_family_id]["points"])
    control_points = _point_map(protocol["cage_v2_best_controls"])
    kitty_points = _point_map(protocol["kitty_pro_targets"])
    index = _index(records)

    expected_method_ids = {
        point["method_id"]
        for points in (candidate_points, control_points, kitty_points)
        for point in points.values()
    }
    _require(set(index) == expected_method_ids, "holdout method grid mismatch")

    length_reports = []
    overall_candidate: dict[tuple[int, int], float] = {}
    overall_control: dict[tuple[int, int], float] = {}
    overall_kitty: dict[tuple[int, int], float] = {}
    for length in LENGTHS:
        candidate_point = candidate_points[length]
        control_point = control_points[length]
        kitty_point = kitty_points[length]
        candidate_id = candidate_point["method_id"]
        control_id = control_point["method_id"]
        kitty_id = kitty_point["method_id"]
        candidate_records = _records(index, candidate_id)
        _require(
            all(record["method"]["family_id"] == selected_family_id for record in candidate_records.values()),
            f"selected candidate identity mismatch: {candidate_id}",
        )
        candidate = _values(index, candidate_id)
        control = _values(index, control_id)
        kitty = _values(index, kitty_id)
        candidate_bytes = _bytes(index, candidate_id)
        control_bytes = _bytes(index, control_id)
        kitty_bytes = _bytes(index, kitty_id)
        _require(candidate_bytes == candidate_point["packed_bytes"], "candidate frozen bytes mismatch")
        _require(control_bytes == control_point["packed_bytes"], "CAGE-v2 frozen bytes mismatch")
        _require(kitty_bytes == kitty_point["target_bytes"], "Kitty-Pro frozen bytes mismatch")
        for anchor in HOLDOUT_ANCHORS:
            key = (length, anchor)
            overall_candidate[key] = candidate[anchor]
            overall_control[key] = control[anchor]
            overall_kitty[key] = kitty[anchor]
        versus_control = paired_summary(candidate, control)
        versus_kitty = paired_summary(candidate, kitty)
        length_reports.append({
            "prompt_length": length,
            "candidate_method_id": candidate_id,
            "cage_v2_control_method_id": control_id,
            "kitty_pro_method_id": kitty_id,
            "candidate_model_total_bytes": candidate_bytes,
            "cage_v2_control_model_total_bytes": control_bytes,
            "kitty_pro_model_total_bytes": kitty_bytes,
            "candidate_minus_kitty_pro_bytes": candidate_bytes - kitty_bytes,
            "memory_pass": candidate_bytes <= kitty_bytes,
            "versus_cage_v2": versus_control,
            "versus_kitty_pro": versus_kitty,
            "beats_cage_v2": versus_control["mean_delta"] < 0,
            "no_worse_than_kitty_pro": versus_kitty["mean_delta"] <= 0,
        })

    kitty_pass_count = sum(row["no_worse_than_kitty_pro"] for row in length_reports)
    gates = {
        "memory_all_three": all(row["memory_pass"] for row in length_reports),
        "beats_cage_v2_all_three": all(row["beats_cage_v2"] for row in length_reports),
        "no_worse_than_kitty_pro_at_least_two": kitty_pass_count >= 2,
        "kitty_pro_no_worse_length_count": kitty_pass_count,
    }
    holdout_pass = (
        gates["memory_all_three"]
        and gates["beats_cage_v2_all_three"]
        and gates["no_worse_than_kitty_pro_at_least_two"]
    )
    return {
        "schema_version": 1,
        "analysis_id": "qwen3-8b-cage-v3-development-holdout-analysis-v1",
        "status": "pass",
        "claim_eligible": False,
        "development_only": True,
        "receipt_sha256": receipt_sha256,
        "selected_family_id": selected_family_id,
        "primary_metric": "mean joint_post_o_proj_mse over 36 layers",
        "paired_unit": "same prompt length and development-holdout anchor",
        "anchors": HOLDOUT_ANCHORS,
        "prompt_lengths": LENGTHS,
        "lengths": length_reports,
        "overall_30_case_versus_cage_v2": paired_summary(overall_candidate, overall_control),
        "overall_30_case_versus_kitty_pro": paired_summary(overall_candidate, overall_kitty),
        "gates": gates,
        "holdout_gate_pass": holdout_pass,
        "decision_status": (
            "development_holdout_passed_preregistered_local_perturbation_gate"
            if holdout_pass
            else "development_holdout_failed_preregistered_local_perturbation_gate"
        ),
        "bootstrap": {
            "used_for_gate": False,
            "descriptive_interval_computed": False,
            "reason": "seed and resample count were not preregistered; exact paired deltas are reported instead",
        },
        "report_all_lengths": True,
        "report_unfavorable_results": True,
        "reserved_unseen_metrics_consumed": False,
        "end_to_end_quality_authorized": False,
        "next_protocol_may_be_frozen": holdout_pass,
        "kitty_12_5pct_followup_required_if_advanced": holdout_pass,
        "new_candidate_selection_authorized": False,
    }


__all__ = [
    "CageV3HoldoutAnalysisError",
    "EXPECTED_HOLDOUT_RECEIPT_SHA256",
    "HOLDOUT_ANCHORS",
    "LENGTHS",
    "build_holdout_analysis",
    "validate_holdout_receipt",
]
