from __future__ import annotations

from pathlib import Path
from typing import Any

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


__all__ = [
    "CageV3HoldoutAnalysisError",
    "EXPECTED_HOLDOUT_RECEIPT_SHA256",
    "HOLDOUT_ANCHORS",
    "LENGTHS",
    "validate_holdout_receipt",
]
