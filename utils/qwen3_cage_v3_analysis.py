from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.qwen3_cage_v3_protocol import file_sha256


EXPECTED_RECEIPT_SHA256 = "c8cb023245171919ba9a7b75dd809cbd963661d6117f206366b15276baa7b277"
ANCHORS = [5, 6, 7, 8, 9]
LENGTHS = [1024, 2048, 4032]


class CageV3AnalysisError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3AnalysisError(message)


def validate_screen_receipt(receipt: dict[str, Any], *, receipt_path: Path) -> None:
    _require(file_sha256(receipt_path) == EXPECTED_RECEIPT_SHA256, "screen receipt hash mismatch")
    _require(receipt.get("schema_version") == 1, "screen receipt schema mismatch")
    _require(
        receipt.get("status") == "frozen_after_joint_postrun_audit_before_screen_interpretation",
        "screen receipt status mismatch",
    )
    _require(receipt.get("claim_eligible") is False, "screen receipt must be claim-ineligible")
    _require(receipt.get("interpretation_performed") is False, "screen receipt already interpreted")
    summary = receipt.get("audit_summary", {})
    _require(
        (summary.get("status"), summary.get("case_count"), summary.get("layer_record_count"), summary.get("failure_count"))
        == ("pass", 75, 2700, 0),
        "screen receipt audit totals mismatch",
    )
    for field in ("holdout_metrics_consumed", "reserved_unseen_metrics_consumed", "interpretation_performed"):
        _require(summary.get(field) is False, f"screen receipt boundary mismatch: {field}")
    partitions = receipt.get("partitions", {})
    _require(tuple(partitions) == ("cage_qwen3", "kitty_qwen3"), "screen receipt partitions mismatch")
    _require(partitions["cage_qwen3"].get("case_count") == 60, "CAGE receipt count mismatch")
    _require(partitions["kitty_qwen3"].get("case_count") == 15, "Kitty receipt count mismatch")
    _require(
        receipt.get("next_authorization")
        == {
            "screen_interpretation": True,
            "maximum_candidates_advanced": 1,
            "holdout_metrics": False,
            "reserved_unseen_metrics": False,
            "end_to_end_quality": False,
        },
        "screen receipt authorization mismatch",
    )


def paired_summary(candidate: Mapping[Any, float], baseline: Mapping[Any, float]) -> dict[str, Any]:
    keys = sorted(candidate)
    _require(keys and keys == sorted(baseline), "paired screen contrast requires the same nonempty grid")
    deltas = [float(candidate[key]) - float(baseline[key]) for key in keys]
    _require(all(math.isfinite(value) for value in deltas), "paired screen contrast contains non-finite values")
    favor = sum(value < 0 for value in deltas)
    ties = sum(value == 0 for value in deltas)
    return {
        "paired_count": len(deltas),
        "mean_delta": math.fsum(deltas) / len(deltas),
        "median_delta": statistics.median(deltas),
        "minimum_delta": min(deltas),
        "maximum_delta": max(deltas),
        "favor_count": favor,
        "tie_count": ties,
        "oppose_count": len(deltas) - favor - ties,
        "negative_delta_favors_candidate": True,
        "paired_deltas": [
            (
                {"prompt_length": key[0], "anchor_index": key[1], "delta": delta}
                if isinstance(key, tuple)
                else {"anchor_index": key, "delta": delta}
            )
            for key, delta in zip(keys, deltas)
        ],
    }


def _index(records: Sequence[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    index: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        method_id = record["method"]["id"]
        anchor = record["input"]["anchor_index"]
        _require(anchor not in index[method_id], f"duplicate screen record: {method_id}/{anchor}")
        index[method_id][anchor] = record
    return dict(index)


def _records(index: dict[str, dict[int, dict[str, Any]]], method_id: str) -> dict[int, dict[str, Any]]:
    records = index.get(method_id, {})
    _require(sorted(records) == ANCHORS, f"method lacks frozen screen grid: {method_id}")
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
    _require(len(values) == 1, f"method packed bytes vary across anchors: {method_id}")
    return values.pop()


def _point_map(points: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(point["prompt_length"]): point for point in points}
    _require(sorted(result) == LENGTHS, "screen point lengths mismatch")
    return result


def build_screen_analysis(
    *,
    protocol: dict[str, Any],
    receipt: dict[str, Any],
    receipt_sha256: str,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    gate = protocol.get("screen_gate", {})
    _require(gate.get("maximum_candidates_advanced") == 1, "screen advancement limit mismatch")
    _require(protocol["development_data"]["partitions"]["screen"] == ANCHORS, "screen anchors changed")
    _require(len(records) == 75, "screen analysis requires exactly 75 records")
    index = _index(records)
    kitty = _point_map(protocol["kitty_pro_targets"])
    controls = _point_map(protocol["cage_v2_best_controls"])
    family_reports = []
    for family in protocol["candidate_families"]:
        points = _point_map(family["points"])
        length_reports = []
        overall_candidate: dict[tuple[int, int], float] = {}
        overall_kitty: dict[tuple[int, int], float] = {}
        overall_control: dict[tuple[int, int], float] = {}
        total_bytes = 0
        for length in LENGTHS:
            candidate_point = points[length]
            control_point = controls[length]
            kitty_point = kitty[length]
            candidate_id = candidate_point["method_id"]
            control_id = control_point["method_id"]
            kitty_id = kitty_point["method_id"]
            candidate_records = _records(index, candidate_id)
            _require(
                all(record["method"]["family_id"] == family["family_id"] for record in candidate_records.values()),
                f"candidate family identity mismatch: {candidate_id}",
            )
            candidate = _values(index, candidate_id)
            control = _values(index, control_id)
            kitty_values = _values(index, kitty_id)
            candidate_bytes = _bytes(index, candidate_id)
            control_bytes = _bytes(index, control_id)
            kitty_bytes = _bytes(index, kitty_id)
            _require(candidate_bytes == candidate_point["packed_bytes"], "candidate frozen bytes mismatch")
            _require(control_bytes == control_point["packed_bytes"], "CAGE-v2 control frozen bytes mismatch")
            _require(kitty_bytes == kitty_point["target_bytes"], "Kitty-Pro frozen bytes mismatch")
            total_bytes += candidate_bytes
            for anchor in ANCHORS:
                key = (length, anchor)
                overall_candidate[key] = candidate[anchor]
                overall_control[key] = control[anchor]
                overall_kitty[key] = kitty_values[anchor]
            versus_control = paired_summary(candidate, control)
            versus_kitty = paired_summary(candidate, kitty_values)
            length_reports.append({
                "prompt_length": length,
                "candidate_method_id": candidate_id,
                "cage_v2_control_method_id": control_id,
                "kitty_pro_method_id": kitty_id,
                "candidate_model_total_bytes": candidate_bytes,
                "cage_v2_control_model_total_bytes": control_bytes,
                "kitty_pro_model_total_bytes": kitty_bytes,
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
        family_reports.append({
            "family_id": family["family_id"],
            "sink_length": family["sink_length"],
            "layer_allocation": family["layer_allocation"],
            "lengths": length_reports,
            "total_model_total_bytes_across_lengths": total_bytes,
            "overall_15_case_versus_cage_v2": paired_summary(overall_candidate, overall_control),
            "overall_15_case_versus_kitty_pro": paired_summary(overall_candidate, overall_kitty),
            "gates": gates,
            "family_pass": gates["memory_all_three"]
            and gates["beats_cage_v2_all_three"]
            and gates["no_worse_than_kitty_pro_at_least_two"],
        })
    _require(len(family_reports) == 3, "screen analysis must report all three candidates")
    passing = [report for report in family_reports if report["family_pass"]]
    ranked = sorted(
        passing,
        key=lambda report: (
            report["overall_15_case_versus_kitty_pro"]["mean_delta"],
            report["total_model_total_bytes_across_lengths"],
            report["family_id"],
        ),
    )
    selected = ranked[:1]
    decision_status = (
        "selected_one_candidate_for_frozen_holdout_authorization"
        if selected
        else "closed_negative_development_screen_do_not_read_holdout"
    )
    return {
        "schema_version": 1,
        "analysis_id": "qwen3-8b-cage-v3-development-screen-analysis-v1",
        "status": "pass",
        "decision_status": decision_status,
        "claim_eligible": False,
        "receipt_sha256": receipt_sha256,
        "primary_metric": "mean joint_post_o_proj_mse over 36 layers",
        "paired_unit": "same prompt length and screen anchor",
        "anchors": ANCHORS,
        "prompt_lengths": LENGTHS,
        "family_reports": family_reports,
        "advanced_families": [
            {
                "rank": rank,
                "family_id": report["family_id"],
                "overall_15_case_mean_delta_vs_kitty_pro": report["overall_15_case_versus_kitty_pro"]["mean_delta"],
                "total_model_total_bytes_across_lengths": report["total_model_total_bytes_across_lengths"],
            }
            for rank, report in enumerate(selected, start=1)
        ],
        "advanced_family_count": len(selected),
        "holdout_authorized_after_decision_freeze": len(selected) == 1,
        "holdout_metrics_consumed": False,
        "reserved_unseen_metrics_consumed": False,
        "selection_tie_break": "overall_15_case_mean_delta_vs_kitty_pro_then_total_packed_bytes_then_family_id",
        "bootstrap_used_for_gate": False,
        "report_all_three_candidates": True,
    }


__all__ = [
    "ANCHORS",
    "CageV3AnalysisError",
    "EXPECTED_RECEIPT_SHA256",
    "LENGTHS",
    "build_screen_analysis",
    "paired_summary",
    "validate_screen_receipt",
]
