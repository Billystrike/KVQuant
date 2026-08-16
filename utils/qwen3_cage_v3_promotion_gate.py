from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


GATE_ID = "qwen3-8b-cage-v3-promotion-acceptance-gate-v1"
GATE_CANONICAL_SHA256 = "20fd85ad54753a7991e79d6775b0e0b51503492754d496dd972cd79a9fed7c2e"
EXPECTED_PROTOCOL_SHA256 = "bc5d3811c0483433feef883c5511d350e8e4cbc48e12f0e4804bc2ccecf31f18"
EXPECTED_INPUT_MANIFEST_SHA256 = "7c662aecd392f91f8d4a674af6af01496deba952820babd29045e3193dc5300d"
EXPECTED_ACCEPTANCE_COMMIT = "cc72bf72d206a94b39535415cf43394cc7b65dfa"
EXPECTED_POSTRUN_COMMIT = "7a9df420497215206b8016f794fde9b7eb181881"
EXPECTED_SOURCE_HASHES = {
    "artifact_manifest": "f10ce80bbb77a9547e1861afeed9fd8eff83e5e61c3911ea2946ea77e82fadd7",
    "postrun_validator": "033d4aafbe29a596012d3f3021efedc47cdecd5fc6bccb3929b7667451a235ce",
    "postrun_utils": "0665ef13151bc7be293123eccf37e1b941ea3b2d01e15a2c3b031c26a302c640",
    "acceptance_runner": "8c9462ac029e14bcb40c31573f3fc0442d9629b68d0aefd5e86252e0e5bf3296",
    "acceptance_comparator": "0c9c727240cb0b8a488fb3d456c0d5d53f9282ccaccbd27d6d830d7777534031",
    "acceptance_utils": "e59656cf2b368d8d9073fb1a837ed71f7e509d47559d6792c5a39109b20a42fa",
    "quality_runtime": "4991266a2c7c41e75823f1423c6de60e7ab02ec7e0a0c059afa8763940450209",
}
SOURCE_PATHS = {
    "artifact_manifest": "configs/qwen3_8b_cage_v3_promotion_acceptance_artifacts_v1.json",
    "postrun_validator": "scripts/qwen3_validate_cage_v3_promotion_postrun.py",
    "postrun_utils": "utils/qwen3_cage_v3_promotion_postrun.py",
    "acceptance_runner": "scripts/qwen3_run_cage_v3_promotion_acceptance.py",
    "acceptance_comparator": "scripts/qwen3_compare_cage_v3_promotion_acceptance.py",
    "acceptance_utils": "utils/qwen3_cage_v3_promotion_acceptance.py",
    "quality_runtime": "scripts/qwen3_run_cage_v4_metric_acceptance.py",
}
EXPECTED_PARTITIONS = {
    "cage_qwen3": {
        "acceptance_case_count_per_repeat": 12,
        "repeat_count": 2,
        "acceptance_total_case_file_count": 24,
        "scientific_payload_sha256": "d2e4e8868f504b6aac1d37e96ef548b2ed2cb36418e3cb8c8474471afa3f4e54",
        "case_ids_sha256": "19b42a51933358033f52e15c066d47dc0ec97ec9ac8632ddb7f6a242fd23e8b4",
        "comparison_sha256": "6bc4747bb524ac02dfa61f8abbcd15fd581d68e271ff2b4567665fb34b6d526f",
        "full_holdout_case_count": 480,
    },
    "kitty_qwen3": {
        "acceptance_case_count_per_repeat": 3,
        "repeat_count": 2,
        "acceptance_total_case_file_count": 6,
        "scientific_payload_sha256": "2611d43afe699b10f70b3f2c694a96b3bd93fb0cd5c0beb2cda76db84838c26d",
        "case_ids_sha256": "10da77aa70bbfe3f8f9aa77fa0cd4be5a29b2e6711e125737430beee2f448df1",
        "comparison_sha256": "1dba6226cb3fdfadf249e7988f60cce4620ac8caa60db28a6b033b5086eda336",
        "full_holdout_case_count": 120,
    },
}
EXPECTED_AUTHORIZATION = {
    "authorization_basis": "joint acceptance post-run passed after two fresh bitwise-equal scientific-payload repeats for both partitions",
    "input_partition": "holdout",
    "development_only": True,
    "document_count": 20,
    "anchors_per_document": 2,
    "anchor_count": 40,
    "prompt_lengths": [1024, 2048, 4032],
    "continuation_tokens": 64,
    "cage_qwen3_case_count": 480,
    "kitty_qwen3_case_count": 120,
    "total_case_count": 600,
    "gpu_full_holdout_execution": True,
    "resume_requires_exact_identity": True,
    "holdout_interpretation_before_complete_postrun": False,
    "pg19_test_access": False,
    "llama2_execution": False,
    "kitty_llama_port": False,
    "paper_claims": False,
    "runtime_claims": False,
}


class CageV3PromotionGateError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3PromotionGateError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV3PromotionGateError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _validate_joint_postrun(gate: Mapping[str, Any]) -> None:
    spec = gate["joint_postrun"]
    output = Path(spec["output_path"])
    log = Path(spec["log_path"])
    _require(output.is_file() and log.is_file(), "joint post-run artifact is missing")
    _require(file_sha256(output) == spec["output_sha256"] and output.stat().st_size == spec["output_size_bytes"], "joint post-run output receipt mismatch")
    _require(file_sha256(log) == spec["log_sha256"] and log.stat().st_size == spec["log_size_bytes"], "joint post-run log receipt mismatch")
    value = _load(output)
    _require(value.get("status") == "pass" and value.get("claim_eligible") is False, "joint post-run did not pass")
    _require(value.get("interpretation_performed") is False and value.get("failure_count") == 0, "joint post-run boundary changed")
    _require(value.get("acceptance_source_commit") == EXPECTED_ACCEPTANCE_COMMIT, "joint post-run acceptance commit mismatch")
    _require(value.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256 and value.get("input_manifest_sha256") == EXPECTED_INPUT_MANIFEST_SHA256, "joint post-run protocol/input mismatch")
    _require(value.get("artifact_manifest_sha256") == EXPECTED_SOURCE_HASHES["artifact_manifest"], "joint post-run artifact manifest mismatch")
    _require(value.get("case_count_per_joint_repeat") == 15 and value.get("repeat_count") == 2 and value.get("total_case_file_count") == 30, "joint post-run case count mismatch")
    _require(value.get("failed_pre_case_attempt_count") == 2, "joint post-run failed-attempt count mismatch")
    candidate = value.get("acceptance_gate_candidate", {})
    _require(candidate.get("status") == "eligible_for_checked_in_gate_review" and candidate.get("full_holdout_authorized_by_this_audit") is False, "joint post-run candidate boundary changed")
    for partition, expected in EXPECTED_PARTITIONS.items():
        record = value.get("partitions", {}).get(partition, {})
        _require(record.get("case_count_per_repeat") == expected["acceptance_case_count_per_repeat"], f"{partition} post-run case count mismatch")
        _require(record.get("repeat_count") == 2 and record.get("repeat_payloads_bitwise_equal") is True, f"{partition} repeat gate mismatch")
        _require(record.get("scientific_payload_sha256") == expected["scientific_payload_sha256"], f"{partition} scientific payload mismatch")
        _require(record.get("repeats", {}).get("a", {}).get("case_ids_sha256") == expected["case_ids_sha256"] and record.get("repeats", {}).get("b", {}).get("case_ids_sha256") == expected["case_ids_sha256"], f"{partition} acceptance case IDs mismatch")
        _require(record.get("comparison", {}).get("sha256") == expected["comparison_sha256"] and record.get("comparison", {}).get("mismatch_case_ids") == [], f"{partition} comparison receipt mismatch")
    text = log.read_text(encoding="utf-8", errors="replace")
    _require("Traceback (most recent call last)" not in text and "ERROR:" not in text, "joint post-run log contains an error")
    _require("=== CAGE-V3 PROMOTION ACCEPTANCE JOINT POSTRUN START ===" in text and "=== CAGE-V3 PROMOTION ACCEPTANCE JOINT POSTRUN END ===" in text, "joint post-run log markers missing")


def load_promotion_gate(
    path: Path,
    *,
    repo_root: Path,
    protocol_sha256: str,
    input_manifest_sha256: str,
    verify_server_artifacts: bool,
) -> tuple[dict[str, Any], str]:
    gate = _load(path)
    _require(canonical_sha256(gate) == GATE_CANONICAL_SHA256, "promotion gate frozen payload mismatch")
    _require(gate.get("schema_version") == 1 and gate.get("gate_id") == GATE_ID, "promotion gate identity mismatch")
    _require(gate.get("status") == "pass" and gate.get("claim_eligible") is False, "promotion gate status mismatch")
    _require(gate.get("protocol_sha256") == protocol_sha256 == EXPECTED_PROTOCOL_SHA256, "promotion gate protocol mismatch")
    _require(gate.get("input_manifest_sha256") == input_manifest_sha256 == EXPECTED_INPUT_MANIFEST_SHA256, "promotion gate input mismatch")
    _require(gate.get("acceptance_source_commit") == EXPECTED_ACCEPTANCE_COMMIT and gate.get("postrun_source_commit") == EXPECTED_POSTRUN_COMMIT, "promotion gate source commit mismatch")
    _require(gate.get("frozen_source_sha256") == EXPECTED_SOURCE_HASHES, "promotion gate source receipts changed")
    _require(gate.get("partitions") == EXPECTED_PARTITIONS, "promotion gate partition receipts changed")
    _require(gate.get("full_holdout_authorization") == EXPECTED_AUTHORIZATION, "promotion gate authorization changed")
    for name, relative in SOURCE_PATHS.items():
        _require(file_sha256(repo_root / relative) == EXPECTED_SOURCE_HASHES[name], f"promotion accepted source changed: {name}")
    if verify_server_artifacts:
        _validate_joint_postrun(gate)
    return gate, file_sha256(path)


__all__ = [
    "CageV3PromotionGateError",
    "EXPECTED_AUTHORIZATION",
    "GATE_ID",
    "load_promotion_gate",
]
