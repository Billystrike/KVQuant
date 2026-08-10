from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.qwen3_cage_v3_protocol import file_sha256


EXPECTED_HOLDOUT_DECISION_SHA256 = "fb82e4b2d6a65480143b178bdf5b85f1009c81f8943e620eaffda968a17b6433"


class CageV3HoldoutDecisionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3HoldoutDecisionError(message)


def validate_holdout_decision(decision: dict[str, Any], *, decision_path: Path) -> None:
    _require(
        file_sha256(decision_path) == EXPECTED_HOLDOUT_DECISION_SHA256,
        "holdout decision hash mismatch",
    )
    _require(decision.get("schema_version") == 1, "holdout decision schema mismatch")
    _require(
        decision.get("decision_id") == "qwen3-8b-cage-v3-development-holdout-decision-v1",
        "holdout decision ID mismatch",
    )
    _require(
        decision.get("status") == "frozen_negative_after_preregistered_holdout_interpretation",
        "holdout decision status mismatch",
    )
    _require(decision.get("claim_eligible") is False, "holdout decision must remain claim-ineligible")
    analysis = decision.get("analysis", {})
    _require(
        (analysis.get("analysis_sha256"), analysis.get("manifest_sha256"), analysis.get("execution_log_sha256"))
        == (
            "b6e6946cd462671034a360897ffb3bb44dc752ecd572384a2854bd071bee71fd",
            "2fc44033ff20181a2b23991b26996777f6e604918bd9c5a7da8e98f0145522c9",
            "7552f20d9bc6665025241e3963b3393804a208621c5373f8c5b98759f8df59be",
        ),
        "holdout analysis provenance mismatch",
    )
    frozen = decision.get("frozen_inputs", {})
    _require(
        frozen.get("holdout_postrun_receipt_sha256")
        == "6eeeb54c45f9ca0ca3cec015c3bd33e408a067755bb651f36a8947fa57d905ac",
        "holdout receipt identity mismatch",
    )
    _require(
        frozen.get("holdout_joint_scientific_payload_sha256")
        == "fc2ae7f310adf98251900d235c5e21c6769131e38b89641679c9535a4e80d5f2",
        "holdout scientific payload identity mismatch",
    )
    result = decision.get("decision", {})
    _require(result.get("selected_family_id") == "pure-sr2-sink32-calibrated", "selected family changed")
    _require(result.get("holdout_gate_pass") is False, "negative holdout decision was changed")
    _require(
        result.get("decision_status") == "development_holdout_failed_preregistered_local_perturbation_gate",
        "negative holdout status changed",
    )
    _require(
        result.get("gates")
        == {
            "memory_all_three": True,
            "beats_cage_v2_all_three": True,
            "no_worse_than_kitty_pro_at_least_two": False,
            "kitty_pro_no_worse_length_count": 1,
        },
        "holdout gates changed",
    )
    _require(result.get("bootstrap_used_for_gate") is False, "bootstrap entered the holdout gate")
    lengths = decision.get("all_lengths", [])
    _require([row.get("prompt_length") for row in lengths] == [1024, 2048, 4032], "length reporting changed")
    _require(all(row.get("memory_pass") is True for row in lengths), "memory results changed")
    _require(all(row.get("beats_cage_v2") is True for row in lengths), "CAGE-v2 results changed")
    _require(
        [row.get("no_worse_than_kitty_pro") for row in lengths] == [True, False, False],
        "Kitty-Pro length results changed",
    )
    _require(
        decision.get("consumption_state")
        == {
            "calibration_metrics_consumed": True,
            "screen_metrics_consumed": True,
            "holdout_metrics_consumed": True,
            "reserved_unseen_metrics_consumed": False,
        },
        "CAGE-v3 consumption boundary changed",
    )
    _require(
        decision.get("closure")
        == {
            "cage_v3_closed": True,
            "kitty_12_5pct_followup_authorized": False,
            "end_to_end_quality_authorized": False,
            "new_candidate_selection_authorized": False,
            "protocol_mutation_authorized": False,
            "reserved_unseen_execution_authorized": False,
        },
        "CAGE-v3 closure authorization changed",
    )


__all__ = [
    "CageV3HoldoutDecisionError",
    "EXPECTED_HOLDOUT_DECISION_SHA256",
    "validate_holdout_decision",
]
