from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v4_data import file_sha256


class Llama2CageV3GPUExecutionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3GPUExecutionError(message)


def load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3GPUExecutionError(f"cannot load JSON {path}: {error}") from error
    require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def validate_static_preflight_receipt(receipt: Mapping[str, Any]) -> None:
    require(receipt.get("schema_version") == 1, "GPU preflight receipt schema mismatch")
    require(receipt.get("preflight_id") == "llama2-7b-cage-v3-production-gpu-acceptance-static-preflight-v1", "GPU preflight identity mismatch")
    require(receipt.get("status") == "pass" and receipt.get("failures") == [], "GPU static preflight did not pass")
    require(receipt.get("claim_eligible") is False, "GPU preflight claim boundary changed")
    require(all(receipt.get("checks", {}).values()), "GPU preflight contains a failed check")
    require(not any(receipt.get("execution_boundary", {}).values()), "GPU preflight execution boundary changed")


def validate_static_preflight_receipt_file(path: Path) -> dict[str, Any]:
    require(
        file_sha256(path) == "63f1473f02a7628ebe677e6e65b7001870514c6f81a630ffb1e3131b155b2d25",
        "GPU static preflight receipt file hash mismatch",
    )
    receipt = load_object(path)
    validate_static_preflight_receipt(receipt)
    return receipt


def validate_gate_receipt(gate: Mapping[str, Any], *, repo_root: Path) -> None:
    require(gate.get("schema_version") == 1, "GPU gate schema mismatch")
    require(gate.get("gate_id") == "llama2-7b-cage-v3-production-gpu-acceptance-execution-gate-v1", "GPU gate identity mismatch")
    require(gate.get("status") == "pass", "GPU execution gate did not pass")
    require(gate.get("claim_eligible") is False, "GPU gate claim boundary changed")
    require(
        gate.get("protocol_sha256") == "d4e3e0468f0719f00632ddc6e0b40e04cc97c459fa2352cd9fe27cb8585273ba",
        "GPU gate protocol identity mismatch",
    )
    require(
        gate.get("preflight_receipt_sha256") == "63f1473f02a7628ebe677e6e65b7001870514c6f81a630ffb1e3131b155b2d25",
        "GPU gate preflight identity mismatch",
    )
    expected_checks = {
        "protocol_frozen",
        "static_preflight_receipt_frozen_and_passed",
        "source_manifest_frozen",
        "all_execution_sources_frozen",
        "repository_clean",
        "single_expected_gpu_with_sufficient_memory",
        "production_model_config_identity",
        "production_weight_full_sha256_identity",
        "weight_identity_matches_static_preflight",
        "no_model_weights_loaded",
        "no_cuda_computation",
        "no_corpus_or_quality_metric",
    }
    checks = gate.get("checks", {})
    require(set(checks) == expected_checks and all(checks.values()), "GPU gate checks changed or failed")
    decision = gate.get("decision", {})
    require(
        set(decision)
        == {
            "gpu_acceptance_execution_authorized",
            "formal_transfer_authorized",
            "corpus_access_authorized",
            "quality_metric_authorized",
            "runtime_claims_authorized",
        },
        "GPU gate decision fields changed",
    )
    require(decision.get("gpu_acceptance_execution_authorized") is True, "GPU acceptance execution is not authorized")
    for key in ("formal_transfer_authorized", "corpus_access_authorized", "quality_metric_authorized", "runtime_claims_authorized"):
        require(decision.get(key) is False, f"GPU gate expanded authorization: {key}")
    frozen_sources = gate.get("frozen_sources", {})
    require(len(frozen_sources) == 10, "GPU gate frozen source count mismatch")
    for name, spec in frozen_sources.items():
        require(file_sha256(repo_root / spec["path"]) == spec["sha256"], f"GPU gate source changed: {name}")


def full_file_sha256(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().contiguous().cpu()
    header = json.dumps(
        {"shape": list(value.shape), "dtype": str(value.dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header)
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "Llama2CageV3GPUExecutionError",
    "full_file_sha256",
    "load_object",
    "require",
    "tensor_sha256",
    "validate_gate_receipt",
    "validate_static_preflight_receipt",
    "validate_static_preflight_receipt_file",
]
