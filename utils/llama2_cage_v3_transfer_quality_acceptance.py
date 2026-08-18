from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.llama2_cage_v3_transfer_quality_manifest import validate_input_manifest
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


EXECUTION_ID = "llama2-7b-cage-v3-transfer-quality-acceptance-v1"
INPUT_ARTIFACT_SHA256 = "6e81ca1741cabd3268983013d99d2434099992ccd0e6f4bdb79f47dafabaac86"
INPUT_MANIFEST_SHA256 = "ef6b3e9c48e219ab09d35dadca4026f2cd99f47c3c922d8a8e1d2b5c4db19d04"
PROTOCOL_SHA256 = "355d73629dece9015288d10acdd7c95e08da0cff941a38395d54e1868380b654"
SCIENTIFIC_FIELDS = ("case_id", "method", "input", "memory", "scoring", "cache")


class Llama2CageV3TransferQualityAcceptanceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3TransferQualityAcceptanceError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3TransferQualityAcceptanceError(f"cannot load JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def lf_normalized_file_sha256(path: Path) -> str:
    """Hash tracked text in the repository's canonical Linux/Git form."""

    try:
        payload = path.read_bytes().replace(b"\r\n", b"\n")
    except OSError as error:
        raise Llama2CageV3TransferQualityAcceptanceError(
            f"cannot read frozen source {path}: {error}"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def validate_execution(execution: Mapping[str, Any], *, repo_root: Path) -> None:
    require(execution.get("schema_version") == 1 and execution.get("execution_id") == EXECUTION_ID, "acceptance execution identity changed")
    require(execution.get("status") == "frozen_before_any_llama2_v3_quality_metric", "acceptance execution status changed")
    require(execution.get("claim_eligible") is False, "acceptance claim boundary changed")
    repair = execution.get("administrative_repair", {})
    require(repair == {
        "reason": "The first gate attempt exposed Windows-worktree CRLF hashes for five tracked Python sources; static tests stopped before the gate validator, model loading, CUDA computation, or quality scoring.",
        "failed_attempt_receipt_path": "configs/llama2_7b_cage_v3_transfer_quality_acceptance_gate_failed_attempt_v1.json",
        "failed_attempt_receipt_sha256": "c119db81abfdc333c26682e6a57d9b3d86a8659554dc7da3b1ed3d0638794801",
        "source_identity_mode_after_repair": "sha256_after_deterministic_crlf_to_lf_normalization",
        "candidate_algorithm_changed": False,
        "input_manifest_changed": False,
        "method_matrix_changed": False,
        "scoring_changed": False,
        "authorization_changed": False,
    }, "acceptance administrative repair changed")
    require(file_sha256(repo_root / repair["failed_attempt_receipt_path"]) == repair["failed_attempt_receipt_sha256"], "failed gate-attempt receipt changed")
    protocol = execution.get("protocol", {})
    require(protocol == {"path": "configs/llama2_7b_cage_v3_transfer_quality_protocol_v1.json", "sha256": PROTOCOL_SHA256}, "acceptance protocol link changed")
    inputs = execution.get("input_artifacts", {})
    require(inputs == {"path": "configs/llama2_7b_cage_v3_transfer_quality_input_artifacts_v1.json", "sha256": INPUT_ARTIFACT_SHA256}, "input artifact link changed")
    manifest = execution.get("input_manifest", {})
    require(manifest == {
        "path": "/root/autodl-tmp/kitty_setup_audit/llama2_cage_v3_transfer_quality_input_manifest_68d8bef.json",
        "sha256": INPUT_MANIFEST_SHA256,
        "size_bytes": 5160675,
        "anchor_count": 50,
        "input_record_count": 150,
    }, "input manifest link changed")
    quota = execution.get("quota_plan", {})
    require(quota == {"path": "configs/llama2_7b_cage_v3_transfer_quota_plan_v1.json", "sha256": "6fc7e33a63db5ed4b1e303edd50c306cf91ab07b99dc981b1d6ea91a58f845e8"}, "quota link changed")
    model = execution.get("model", {})
    require(model.get("reference") == "/root/autodl-tmp/models/Llama-2-7b-hf", "model reference changed")
    require(model.get("config_sha256") == "9242e7db1bc2a17873e66084c3b1c6ed10883076e156b338fd6a7775748e2e3c", "model config changed")
    require(model.get("dtype") == "float16" and model.get("device") == "cuda:0", "model execution mode changed")
    require(model.get("weight_files") == [
        {"name": "model-00001-of-00002.safetensors", "sha256": "4ec71fd53e99766de38f24753b30c9e8942630e9e576a1ba27b0ec531e87be41", "size_bytes": 9976578928},
        {"name": "model-00002-of-00002.safetensors", "sha256": "41780b5dac322ac35598737e99208d90bdc632a1ba3389ebedbb46a1d8385a7f", "size_bytes": 3500297344},
    ], "model weights changed")
    require(execution.get("environment") == {"python_major_minor": "3.10", "torch": "2.4.1+cu121", "transformers": "4.43.1", "gpu": "NVIDIA GeForce RTX 4090 D"}, "environment changed")
    acceptance = execution.get("acceptance", {})
    require(acceptance.get("anchor_indices") == [0] and acceptance.get("prompt_lengths") == [1024, 2048, 4032], "acceptance inputs changed")
    require(acceptance.get("method_roles") == ["quality_reference", "transfer_candidate", "primary_predecessor", "primary_uniform_baseline"], "acceptance roles changed")
    require(acceptance.get("case_count_per_repeat") == 12 and acceptance.get("repeats") == ["a", "b"], "acceptance case/repeat count changed")
    require(acceptance.get("required_repeat_consistency") == "bitwise_equal_json_scientific_payload", "repeat consistency changed")
    require(acceptance.get("telemetry_excluded_from_repeat_comparison") is True, "telemetry comparison boundary changed")
    scoring = execution.get("scoring", {})
    require((scoring.get("boundary_target_count"), scoring.get("single_token_decode_target_count"), scoring.get("primary_target_count_per_case")) == (1, 63, 64), "scoring target counts changed")
    require(scoring.get("fp16_one_shot_reference") == "diagnostic_only", "FP16 diagnostic boundary changed")
    require(scoring.get("same_token_ids_targets_order_and_reduction_for_all_methods") is True, "scoring equality changed")
    determinism = execution.get("determinism", {})
    require(determinism == {"seed": 20260817, "torch_deterministic_algorithms": True, "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "batch_size": 1}, "determinism changed")
    boundary = execution.get("current_boundary", {})
    for key in ("quality_acceptance_execution_authorized", "full_600_case_execution_authorized", "candidate_tuning_authorized", "paper_main_method_change_authorized", "runtime_claims_authorized"):
        require(boundary.get(key) is False, f"acceptance boundary changed: {key}")
    for spec in (protocol, inputs, quota):
        require(file_sha256(repo_root / spec["path"]) == spec["sha256"], f"frozen acceptance reference changed: {spec['path']}")
    source = execution.get("source_manifest", {})
    require(source.get("path") == "configs/llama2_7b_cage_v3_transfer_quality_acceptance_sources_v1.json", "source manifest path changed")
    require(isinstance(source.get("sha256"), str) and len(source["sha256"]) == 64 and source["sha256"] != "TO_BE_FILLED", "source manifest hash is not frozen")
    require(lf_normalized_file_sha256(repo_root / source["path"]) == source["sha256"], "source manifest file changed")


def load_execution(path: Path, *, repo_root: Path) -> tuple[dict[str, Any], str]:
    execution = load_json(path)
    validate_execution(execution, repo_root=repo_root)
    return execution, file_sha256(path)


def validate_input_artifact_manifest(value: Mapping[str, Any]) -> None:
    require(value.get("artifact_id") == "llama2-7b-cage-v3-transfer-quality-input-artifacts-v1", "input artifact identity changed")
    require(value.get("status") == "pass" and value.get("claim_eligible") is False, "input artifact did not pass")
    require(value.get("execution_source_commit") == "68d8befb56ab08f64ca0b3332866f2725eaf971d", "input artifact source changed")
    spec = value.get("input_manifest", {})
    require(spec.get("sha256") == INPUT_MANIFEST_SHA256 and spec.get("size_bytes") == 5160675, "input artifact payload changed")
    decision = value.get("decision", {})
    require(decision.get("input_manifest_pass") is True and decision.get("quality_acceptance_protocol_design_authorized") is True, "acceptance design is not authorized")
    for key in ("quality_acceptance_execution_authorized", "full_600_case_execution_authorized", "paper_claims_authorized"):
        require(decision.get(key) is False, f"input artifact expanded authorization: {key}")


def load_server_input_manifest(
    execution: Mapping[str, Any], *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    spec = execution["input_manifest"]
    path = Path(spec["path"])
    require(path.is_file(), "server input manifest is missing")
    require(path.stat().st_size == spec["size_bytes"], "server input manifest size mismatch")
    require(file_sha256(path) == spec["sha256"], "server input manifest digest mismatch")
    manifest = load_json(path)
    validate_input_manifest(manifest, protocol=protocol)
    return manifest


def expand_acceptance_cases(
    *, execution: Mapping[str, Any], execution_sha256: str, protocol: Mapping[str, Any], input_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records = {
        (row["identity"]["anchor_index"], row["identity"]["prompt_length"]): row
        for row in input_manifest["inputs"]
    }
    cases = []
    for length_row in protocol["method_length_matrix"]:
        length = length_row["prompt_length"]
        input_row = records[(0, length)]
        memory = length_row["packed_memory"]
        for method in length_row["methods"]:
            family = method["method"]
            logical_bytes = {
                "fp16": length * 32 * 32 * 128 * 2 * 2,
                "cage_v3": memory["candidate_bytes"],
                "cage_v1": memory["cage_v1_bytes"],
                "kivi": memory["kivi_bytes"],
            }[family]
            identity = {
                "execution_sha256": execution_sha256,
                "input_manifest_sha256": INPUT_MANIFEST_SHA256,
                "input_id": input_row["input_id"],
                "method": method,
                "prompt_length": length,
            }
            cases.append({
                "case_id": canonical_sha256(identity)[:24],
                "method": dict(method),
                "input": input_row,
                "memory": {
                    "representation": "complete active logical packed paper estimate only",
                    "logical_packed_bytes": logical_bytes,
                    "fp16_bytes": length * 32 * 32 * 128 * 2 * 2,
                },
            })
    require(len(cases) == 12 and len({case["case_id"] for case in cases}) == 12, "acceptance case expansion changed")
    return cases


def scientific_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {field: record[field] for field in SCIENTIFIC_FIELDS}


def validate_completed_case(record: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    require(record.get("schema_version") == 1 and record.get("status") == "completed", "case status changed")
    require(record.get("case_id") == expected["case_id"], "case ID changed")
    for field in ("method", "input", "memory"):
        require(record.get(field) == expected[field], f"case {field} changed")
    scoring = record.get("scoring", {})
    nlls = scoring.get("token_nlls")
    require(isinstance(nlls, list) and len(nlls) == 64, "case must contain 64 token NLLs")
    require(all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in nlls), "case NLLs are invalid")
    require((scoring.get("boundary_target_count"), scoring.get("decode_target_count"), scoring.get("all_target_count")) == (1, 63, 64), "case scoring counts changed")
    require(scoring.get("primary_metric") == "cache_conditioned_all_64_target_mean_nll", "case primary metric changed")
    require(math.isclose(scoring.get("nll_sum"), math.fsum(nlls), rel_tol=0.0, abs_tol=1e-12), "case NLL sum changed")
    require(math.isclose(scoring.get("mean_nll"), math.fsum(nlls) / 64, rel_tol=0.0, abs_tol=1e-12), "case mean NLL changed")
    cache = record.get("cache", {})
    expected_final = expected["input"]["identity"]["prompt_length"] + 63
    require(cache.get("all_layer_lengths_equal") is True and cache.get("final_cache_length") == expected_final, "case cache length changed")
    require(cache.get("layer_count") == 32, "case cache layer count changed")
    require(cache.get("method_family") == expected["method"]["method"], "case cache family changed")
    require(cache.get("quantized_path_verified") is (expected["method"]["method"] != "fp16"), "case quantized-path check changed")


def validate_gate_receipt(
    gate: Mapping[str, Any], *, execution_sha256: str, repo_root: Path
) -> None:
    require(gate.get("gate_id") == "llama2-7b-cage-v3-transfer-quality-acceptance-execution-gate-v1", "acceptance gate identity changed")
    require(gate.get("status") == "pass" and gate.get("claim_eligible") is False, "acceptance gate did not pass")
    require(gate.get("execution_sha256") == execution_sha256, "acceptance gate execution mismatch")
    require(gate.get("input_manifest_sha256") == INPUT_MANIFEST_SHA256, "acceptance gate input mismatch")
    require(all(gate.get("checks", {}).values()), "acceptance gate contains failed checks")
    decision = gate.get("decision", {})
    require(decision.get("quality_acceptance_execution_authorized") is True, "quality acceptance is not authorized")
    for key in ("full_600_case_execution_authorized", "candidate_tuning_authorized", "paper_main_method_change_authorized", "runtime_claims_authorized"):
        require(decision.get(key) is False, f"acceptance gate expanded authorization: {key}")
    sources = gate.get("frozen_sources", {})
    require(bool(sources), "acceptance gate source list is empty")
    for spec in sources.values():
        require(lf_normalized_file_sha256(repo_root / spec["path"]) == spec["sha256"], f"acceptance gate source changed: {spec['path']}")


__all__ = [
    "INPUT_MANIFEST_SHA256",
    "Llama2CageV3TransferQualityAcceptanceError",
    "SCIENTIFIC_FIELDS",
    "expand_acceptance_cases",
    "lf_normalized_file_sha256",
    "load_execution",
    "load_json",
    "load_server_input_manifest",
    "require",
    "scientific_payload",
    "validate_completed_case",
    "validate_execution",
    "validate_gate_receipt",
    "validate_input_artifact_manifest",
]
