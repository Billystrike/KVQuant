from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v4_data import file_sha256


PROTOCOL_ID = "llama2-7b-cage-v3-production-gpu-acceptance-v1"
EXPECTED_PROTOCOL_SHA256 = "d4e3e0468f0719f00632ddc6e0b40e04cc97c459fa2352cd9fe27cb8585273ba"


class Llama2CageV3GPUProtocolError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3GPUProtocolError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3GPUProtocolError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def compact_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def synthetic_token_ids() -> list[int]:
    return [3 + ((index * 7919 + 17) % 31997) for index in range(1026)]


def validate_protocol(protocol: Mapping[str, Any], *, repo_root: Path) -> None:
    _require(protocol.get("schema_version") == 1, "GPU protocol schema mismatch")
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "GPU protocol identity mismatch")
    _require(
        protocol.get("status") == "frozen_after_cpu_acceptance_before_any_production_weight_gpu_execution",
        "GPU protocol status mismatch",
    )
    _require(protocol.get("claim_eligible") is False, "GPU protocol claim boundary changed")
    cpu = protocol.get("cpu_acceptance_receipt", {})
    _require(cpu.get("sha256") == "ef92e1afe81c1837e79f0741d3006549ea6d3e93f65887f000c192f4eb7129b1", "CPU receipt changed")
    _require(file_sha256(repo_root / cpu["path"]) == cpu["sha256"], "checked-in CPU receipt hash mismatch")
    _require(cpu.get("cpu_implementation_acceptance_pass") is True, "CPU acceptance pass changed")
    _require(cpu.get("broad_environment_package_check_pass") is False, "package-check failure was hidden")
    preserved = protocol.get("preserved_scientific_boundary", {})
    _require(preserved and not any(preserved.values()), "preserved scientific boundary changed")
    candidate = protocol.get("candidate", {})
    _require(
        (candidate.get("prompt_length"), candidate.get("continuation_tokens"), candidate.get("residual_length"), candidate.get("sink_length"))
        == (1024, 2, 176, 32),
        "GPU acceptance point changed",
    )
    _require(file_sha256(repo_root / candidate["quota_plan_path"]) == candidate["quota_plan_sha256"], "quota plan changed")
    token_ids = synthetic_token_ids()
    synthetic = protocol.get("synthetic_input", {})
    _require(compact_ids_sha256(token_ids[:1024]) == synthetic.get("prompt_ids_sha256"), "prompt token hash mismatch")
    _require(compact_ids_sha256(token_ids[1024:]) == synthetic.get("continuation_ids_sha256"), "continuation token hash mismatch")
    _require(compact_ids_sha256(token_ids) == synthetic.get("all_ids_sha256"), "complete token hash mismatch")
    _require(synthetic.get("corpus_accessed") is False and synthetic.get("tokenizer_used") is False, "synthetic boundary changed")
    design = protocol.get("acceptance_design", {})
    _require(design.get("repeat_count") == 2, "GPU repeat count changed")
    _require(design.get("quality_metric_computed") is False, "GPU acceptance enabled a quality metric")
    _require(design.get("reference_baseline_run") is False, "GPU acceptance enabled a baseline")
    boundary = protocol.get("execution_boundary", {})
    _require(boundary.get("static_preflight_authorized") is True, "static preflight authorization changed")
    _require(boundary.get("gpu_acceptance_implementation_authorized_after_preflight_pass") is True, "implementation authorization changed")
    for key in (
        "gpu_acceptance_execution_authorized_by_this_protocol_alone",
        "formal_transfer_experiment_authorized",
        "corpus_access_authorized",
        "llama2_quality_metric_authorized",
        "kitty_llama_port_authorized",
        "paper_main_method_change_authorized",
        "runtime_claims_authorized",
    ):
        _require(boundary.get(key) is False, f"GPU boundary changed: {key}")


def load_protocol(path: Path, *, repo_root: Path) -> tuple[dict[str, Any], str]:
    protocol = _load(path)
    validate_protocol(protocol, repo_root=repo_root)
    digest = file_sha256(path)
    _require(digest == EXPECTED_PROTOCOL_SHA256, "GPU protocol file hash mismatch")
    return protocol, digest


def query_gpu_inventory() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise Llama2CageV3GPUProtocolError(f"cannot query GPU inventory: {error}") from error
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    _require(len(rows) == 1, f"expected exactly one visible GPU, got {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    _require(len(fields) == 3, "unexpected nvidia-smi inventory format")
    try:
        memory_mib = int(fields[1])
    except ValueError as error:
        raise Llama2CageV3GPUProtocolError("invalid GPU memory value") from error
    return {"name": fields[0], "memory_total_mib": memory_mib, "driver_version": fields[2], "visible_gpu_count": 1}


__all__ = [
    "Llama2CageV3GPUProtocolError",
    "compact_ids_sha256",
    "load_protocol",
    "query_gpu_inventory",
    "synthetic_token_ids",
    "validate_protocol",
]
