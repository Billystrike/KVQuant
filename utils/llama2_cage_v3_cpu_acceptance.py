from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


PROTOCOL_ID = "llama2-7b-cage-v3-cpu-implementation-acceptance-v1"
EXPECTED_PROTOCOL_SHA256 = "ea5299d08d0dbceabd701fcdd063ec3ab04b8fa5ad4c0ad6573f161c0649f6f1"


class Llama2CageV3CPUAcceptanceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3CPUAcceptanceError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3CPUAcceptanceError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _lf_normalized_sha256(path: Path) -> str:
    """Hash tracked text in its Linux/Git LF representation."""

    try:
        payload = path.read_bytes().replace(b"\r\n", b"\n")
    except OSError as error:
        raise Llama2CageV3CPUAcceptanceError(f"cannot read frozen source {path}: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def validate_cpu_protocol(protocol: Mapping[str, Any], *, repo_root: Path) -> None:
    _require(protocol.get("schema_version") == 1, "CPU protocol schema mismatch")
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "CPU protocol identity mismatch")
    _require(protocol.get("status") == "frozen_before_llama2_cage_v3_cpu_acceptance", "CPU protocol status mismatch")
    _require(protocol.get("claim_eligible") is False, "CPU protocol claim boundary changed")
    _require(
        protocol.get("source_identity_mode") == "sha256_after_deterministic_crlf_to_lf_normalization",
        "CPU source identity mode changed",
    )
    repair = protocol.get("administrative_repair", {})
    _require(
        repair
        == {
            "reason": "The first attempt exposed two Windows-worktree CRLF hashes that did not equal the Linux checkout bytes; no direct acceptance or quality computation had started.",
            "failed_attempt_receipt_path": "configs/llama2_7b_cage_v3_cpu_acceptance_failed_attempt_v1.json",
            "failed_attempt_receipt_sha256": "01d49b918ee9e662a7f06c4e5f0674dc3f3d34ad63dc03b1d3b8b70f6ea33e86",
            "candidate_algorithm_changed": False,
            "quota_plan_changed": False,
            "metric_changed": False,
            "authorization_changed": False,
        },
        "CPU administrative repair receipt changed",
    )
    repair_path = repo_root / repair["failed_attempt_receipt_path"]
    _require(file_sha256(repair_path) == repair["failed_attempt_receipt_sha256"], "failed-attempt receipt hash mismatch")
    transfer = protocol.get("transfer_boundary", {})
    _require(transfer.get("qwen3_promotion_pass") is False, "Qwen3 promotion failure changed")
    _require(transfer.get("qwen3_conclusion_reopened") is False, "Qwen3 conclusion was reopened")
    _require(transfer.get("llama2_quality_metrics_consumed") is False, "Llama quality metrics entered CPU acceptance")
    artifacts = protocol.get("static_preflight_artifacts", {})
    expected_artifacts = {
        "quota": {
            "server_path": "/root/autodl-tmp/kitty_setup_audit/llama2_cage_v3_transfer_quota_plan_13e6592.json",
            "server_sha256": "630644ab0608ad9c59909b95976c1daa4e56669152a43ed5a9a925d3605aceaa",
            "server_size_bytes": 11112,
            "canonical_sha256": "bc454ce0b00fd6adc22b36e656ff5f32b22b76a1b6d6e77f791b711dff2cd454",
            "checked_in_path": "configs/llama2_7b_cage_v3_transfer_quota_plan_v1.json",
            "checked_in_sha256": "6fc7e33a63db5ed4b1e303edd50c306cf91ab07b99dc981b1d6ea91a58f845e8",
            "checked_in_size_bytes": 11113,
            "normalization": "identical JSON payload with one terminal LF added by repository text normalization",
        },
        "preflight": {
            "server_path": "/root/autodl-tmp/kitty_setup_audit/llama2_cage_v3_transfer_static_preflight_13e6592.json",
            "server_sha256": "9b8209d4d41bb599a96e36c756fcb8b23ff6065f694f5fb3f7b8a8fad2934c85",
            "server_size_bytes": 6613,
            "canonical_sha256": "d7accd7f9b9d4279bf166693b4a51e246a8c13227411af8229aa61e519163d65",
            "checked_in_path": "configs/llama2_7b_cage_v3_transfer_static_preflight_receipt_v1.json",
            "checked_in_sha256": "b25ec8acc4d7e23195e1d9f5dd45bdeba6312e405b2e929e978136cb9dcdd417",
            "checked_in_size_bytes": 6614,
            "normalization": "identical JSON payload with one terminal LF added by repository text normalization",
        },
    }
    _require(artifacts == expected_artifacts, "static preflight artifact receipt changed")
    for spec in artifacts.values():
        checked = repo_root / spec["checked_in_path"]
        _require(file_sha256(checked) == spec["checked_in_sha256"], f"checked-in artifact hash mismatch: {checked}")
        _require(checked.stat().st_size == spec["checked_in_size_bytes"], f"checked-in artifact size mismatch: {checked}")
        _require(canonical_sha256(_load(checked)) == spec["canonical_sha256"], f"checked-in artifact canonical mismatch: {checked}")
    sources = protocol.get("frozen_sources", {})
    for name, spec in sources.items():
        path = repo_root / spec["path"]
        _require(_lf_normalized_sha256(path) == spec["sha256"], f"frozen source changed: {name}")
    fixture = protocol.get("cpu_fixture", {})
    _require(
        fixture
        == {
            "batch_size": 1,
            "prompt_length": 7,
            "continuation_tokens": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_hidden_layers": 1,
            "sink_length": 1,
            "residual_length": 4,
            "two_bit_channels": 2,
            "group_size": 4,
            "seed": 20260817,
        },
        "CPU fixture changed",
    )
    checks = protocol.get("required_checks", [])
    _require(
        checks
        == [
            "exact production quota installation and architecture rejection",
            "prefill logits/output identical with and without cache materialization",
            "persistent prompt-adaptive sparse Key indices",
            "all-channel INT2 Key backbone plus sparse INT2 refinement",
            "uniform INT2 Value path without Value adaptivity",
            "FP16 sink preservation",
            "single-token continuation and deterministic Key flush",
            "rolling Value residual",
            "old quantized prefix is not requantized",
            "finite reconstructed cache and attention output",
            "unsupported prompt length batch size and multi-token continuation fail closed",
        ],
        "CPU acceptance checks changed",
    )
    boundary = protocol.get("execution_boundary", {})
    for key in (
        "gpu_used",
        "full_model_weights_loaded",
        "llama2_quality_metrics_read",
        "gpu_acceptance_authorized",
        "full_transfer_authorized",
        "kitty_llama_port_authorized",
        "paper_main_method_change_authorized",
        "runtime_claims_authorized",
    ):
        _require(boundary.get(key) is False, f"CPU boundary changed: {key}")


def load_cpu_protocol(path: Path, *, repo_root: Path) -> tuple[dict[str, Any], str]:
    protocol = _load(path)
    validate_cpu_protocol(protocol, repo_root=repo_root)
    digest = file_sha256(path)
    _require(digest == EXPECTED_PROTOCOL_SHA256, "CPU protocol file hash mismatch")
    return protocol, digest


def validate_static_preflight_payloads(protocol: Mapping[str, Any], *, repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    quota_spec = protocol["static_preflight_artifacts"]["quota"]
    preflight_spec = protocol["static_preflight_artifacts"]["preflight"]
    quota = _load(repo_root / quota_spec["checked_in_path"])
    preflight = _load(repo_root / preflight_spec["checked_in_path"])
    _require(quota.get("llama2_metrics_consumed") is False, "quota consumed Llama metrics")
    _require(quota.get("plan_id") == "llama2-7b-cage-v3-depth-normalized-quota-plan-v1", "quota identity mismatch")
    _require(preflight.get("status") == "pass", "static preflight did not pass")
    _require(preflight.get("qwen3_promotion_pass") is False, "preflight changed Qwen3 failure")
    _require(preflight.get("qwen3_conclusion_reopened") is False, "preflight reopened Qwen3")
    _require(preflight.get("derived_quota_plan", {}).get("sha256") == quota_spec["server_sha256"], "preflight quota hash mismatch")
    _require(preflight.get("derived_quota_plan", {}).get("canonical_sha256") == quota_spec["canonical_sha256"], "preflight quota canonical mismatch")
    _require(preflight.get("packed_memory_points") == quota.get("packed_memory_preflight", {}).get("points"), "preflight packed-memory points mismatch")
    _require(all(preflight.get("checks", {}).values()), "static preflight contains a failed check")
    _require(not any(preflight.get("boundary", {}).values()), "static preflight authorization boundary changed")
    return quota, preflight


__all__ = [
    "EXPECTED_PROTOCOL_SHA256",
    "Llama2CageV3CPUAcceptanceError",
    "load_cpu_protocol",
    "validate_cpu_protocol",
    "validate_static_preflight_payloads",
]
