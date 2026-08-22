from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from utils.llama2_cage_v3_transfer_quality_acceptance import (
    SCIENTIFIC_FIELDS,
    lf_normalized_file_sha256,
    validate_completed_case,
)
from utils.llama2_cage_v3_transfer_quality_full_design import DESIGN_SHA256, validate_design
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


FULL_RUN_ID = "llama2-7b-cage-v3-transfer-quality-full-v1"
FULL_GATE_RECEIPT_ID = "llama2-7b-cage-v3-transfer-quality-full-execution-gate-receipt-v1"


class Llama2CageV3TransferQualityFullError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3TransferQualityFullError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3TransferQualityFullError(f"cannot load JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def load_design(path: Path, *, repo_root: Path) -> tuple[dict[str, Any], str]:
    design = load_json(path)
    require(lf_normalized_file_sha256(path) == DESIGN_SHA256, "full design file changed")
    validate_design(design, repo_root=repo_root)
    return design, DESIGN_SHA256


def validate_full_gate_receipt(receipt: Mapping[str, Any], *, repo_root: Path) -> None:
    require(receipt.get("schema_version") == 1, "full gate receipt schema changed")
    require(receipt.get("receipt_id") == FULL_GATE_RECEIPT_ID, "full gate receipt identity changed")
    require(receipt.get("status") == "pass" and receipt.get("claim_eligible") is False, "full gate receipt did not pass")
    require(receipt.get("design_sha256") == DESIGN_SHA256, "full gate design linkage changed")
    require(receipt.get("case_count") == 600 and receipt.get("target_count") == 38400, "full gate case counts changed")
    checks = receipt.get("checks", {})
    require(bool(checks) and all(checks.values()), "full gate receipt contains failed checks")
    sources = receipt.get("frozen_source_sha256", {})
    require(bool(sources), "full gate source list is empty")
    for spec in sources.values():
        require(lf_normalized_file_sha256(repo_root / spec["path"]) == spec["sha256"], f"full gate source changed: {spec['path']}")
    decision = receipt.get("decision", {})
    require(decision.get("full_600_case_execution_authorized") is True, "full execution is not authorized")
    for key in ("quality_interpretation_authorized", "candidate_tuning_authorized", "paper_claims_authorized", "runtime_claims_authorized"):
        require(decision.get(key) is False, f"full gate expanded authorization: {key}")


def load_full_gate_receipt(path: Path, *, repo_root: Path) -> tuple[dict[str, Any], str]:
    receipt = load_json(path)
    validate_full_gate_receipt(receipt, repo_root=repo_root)
    return receipt, file_sha256(path)


def expand_full_cases(
    *,
    design: Mapping[str, Any],
    protocol: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    gate_receipt_sha256: str,
) -> list[dict[str, Any]]:
    records = {
        (row["identity"]["anchor_index"], row["identity"]["prompt_length"]): row
        for row in input_manifest["inputs"]
    }
    cases = []
    for length_row in protocol["method_length_matrix"]:
        length = length_row["prompt_length"]
        memory = length_row["packed_memory"]
        for method in length_row["methods"]:
            family = method["method"]
            for anchor_index in design["case_matrix"]["anchor_indices"]:
                input_row = records[(anchor_index, length)]
                logical_bytes = {
                    "fp16": length * 32 * 32 * 128 * 2 * 2,
                    "cage_v3": memory["candidate_bytes"],
                    "cage_v1": memory["cage_v1_bytes"],
                    "kivi": memory["kivi_bytes"],
                }[family]
                identity = {
                    "full_run_id": FULL_RUN_ID,
                    "design_sha256": DESIGN_SHA256,
                    "gate_receipt_sha256": gate_receipt_sha256,
                    "input_manifest_sha256": design["input_manifest"]["sha256"],
                    "input_id": input_row["input_id"],
                    "method": method,
                    "prompt_length": length,
                    "anchor_index": anchor_index,
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
    require(len(cases) == 600 and len({case["case_id"] for case in cases}) == 600, "full case expansion changed")
    return cases


def scientific_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {field: record[field] for field in SCIENTIFIC_FIELDS}


def validate_full_case(record: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    validate_completed_case(record, expected)
    provenance = record.get("provenance", {})
    require(provenance.get("full_run_id") == FULL_RUN_ID, "full case run identity changed")
    require(provenance.get("design_sha256") == DESIGN_SHA256, "full case design linkage changed")
    require(isinstance(provenance.get("gate_receipt_sha256"), str) and len(provenance["gate_receipt_sha256"]) == 64, "full case gate linkage changed")
    require(provenance.get("cublas_workspace_config") == ":4096:8", "full case CuBLAS determinism changed")


__all__ = [
    "FULL_GATE_RECEIPT_ID",
    "FULL_RUN_ID",
    "Llama2CageV3TransferQualityFullError",
    "expand_full_cases",
    "load_design",
    "load_full_gate_receipt",
    "load_json",
    "require",
    "scientific_payload",
    "validate_full_case",
    "validate_full_gate_receipt",
]
