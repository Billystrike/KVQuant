#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llama2_cage_v3_transfer_protocol import (
    EXPECTED_QWEN_PLAN_SHA256,
    audit_llama2_model_metadata,
    derive_llama2_quota_plan,
    derive_packed_memory_preflight,
    load_transfer_protocol,
)
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _serialize(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _prepare_atomic(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen preflight output: {path}")
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CPU-only Llama-2 CAGE-v3 cross-architecture static preflight"
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--quota-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    model_path = args.model.resolve()
    quota_output = args.quota_output.resolve()
    output = args.output.resolve()
    if quota_output == output:
        raise ValueError("quota and preflight outputs must be different files")
    if quota_output.exists() or output.exists():
        raise FileExistsError("refusing to overwrite an existing preflight artifact")

    protocol, protocol_sha256 = load_transfer_protocol(protocol_path, repo_root=REPO_ROOT)
    qwen_plan_path = (REPO_ROOT / protocol["source_candidate"]["qwen3_quota_plan_path"]).resolve()
    if file_sha256(qwen_plan_path) != EXPECTED_QWEN_PLAN_SHA256:
        raise ValueError("frozen Qwen3 quota-plan file hash mismatch")
    qwen_plan = _load_object(qwen_plan_path)
    quota_plan = derive_llama2_quota_plan(protocol=protocol, qwen_plan=qwen_plan)
    quota_plan["packed_memory_preflight"] = derive_packed_memory_preflight(
        protocol=protocol, quota_plan=quota_plan
    )
    quota_payload = _serialize(quota_plan)
    quota_file_sha256 = hashlib.sha256(quota_payload).hexdigest()

    model_receipt = audit_llama2_model_metadata(model_path)
    report = {
        "schema_version": 1,
        "preflight_id": "llama2-7b-cage-v3-cross-architecture-transfer-static-preflight-v1",
        "status": "pass",
        "claim_eligible": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "qwen3_promotion_pass": False,
        "qwen3_conclusion_reopened": False,
        "protocol_deviation_disclosed": True,
        "qwen3_quota_plan_sha256": file_sha256(qwen_plan_path),
        "derived_quota_plan": {
            "path": str(quota_output),
            "sha256": quota_file_sha256,
            "canonical_sha256": canonical_sha256(quota_plan),
            "size_bytes": len(quota_payload),
            "plan_id": quota_plan["plan_id"],
        },
        "model_receipt": model_receipt,
        "packed_memory_points": quota_plan["packed_memory_preflight"]["points"],
        "checks": {
            "protocol": True,
            "qwen3_negative_result_preserved": True,
            "qwen3_quota_plan": True,
            "deterministic_depth_mapping": True,
            "target_quota_counts": True,
            "model_and_tokenizer_identity": True,
            "full_weight_file_hashes": True,
            "cage_v1_byte_tolerance": True,
            "kivi_memory_asymmetry_disclosed": True,
        },
        "boundary": {
            "llama2_candidate_quality_metrics_read": False,
            "llama2_baseline_quality_metrics_read": False,
            "gpu_used": False,
            "cpu_implementation_acceptance_authorized_by_this_artifact": False,
            "gpu_acceptance_authorized": False,
            "full_transfer_authorized": False,
            "kitty_llama_port_authorized": False,
            "paper_main_method_change_authorized": False,
            "runtime_claims_authorized": False,
        },
        "next_step": "check in this exact preflight receipt and quota plan before defining the CPU implementation-acceptance gate",
    }
    report_payload = _serialize(report)
    quota_temporary = _prepare_atomic(quota_output, quota_payload)
    report_temporary = None
    try:
        report_temporary = _prepare_atomic(output, report_payload)
        os.replace(quota_temporary, quota_output)
        os.replace(report_temporary, output)
    finally:
        quota_temporary.unlink(missing_ok=True)
        if report_temporary is not None:
            report_temporary.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
