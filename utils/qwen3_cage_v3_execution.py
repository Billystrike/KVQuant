from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes
from utils.qwen3_cage_v3_manifest import validate_cage_v3_manifest
from utils.qwen3_cage_v3_protocol import file_sha256, load_cage_v3_protocol


EXECUTION_ID = "qwen3-8b-cage-v3-calibration-execution-v1"
STAGES = ("calibration_acceptance", "calibration_full")
SCIENTIFIC_FIELDS = (
    "case_id",
    "method",
    "input",
    "memory",
    "layer_metrics",
    "aggregates",
    "cache",
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_calibration_execution(
    path: str | Path,
    *,
    repo_root: Path,
    verify_artifacts: bool,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    source = Path(path).resolve()
    execution = load_json(source)
    if execution.get("schema_version") != 1 or execution.get("execution_id") != EXECUTION_ID:
        raise ValueError("CAGE-v3 calibration execution identity mismatch")
    if execution.get("status") != "frozen_before_any_cage_v3_gpu_metric" or execution.get("claim_eligible") is not False:
        raise ValueError("CAGE-v3 calibration execution boundary mismatch")
    protocol_path = Path(execution["protocol"]["path"])
    if not protocol_path.is_absolute():
        protocol_path = repo_root / protocol_path
    protocol, protocol_sha256 = load_cage_v3_protocol(protocol_path)
    if protocol_sha256 != execution["protocol"]["sha256"]:
        raise ValueError("CAGE-v3 calibration protocol hash mismatch")
    manifest_path = Path(execution["input_manifest"]["path"])
    if verify_artifacts and file_sha256(manifest_path) != execution["input_manifest"]["sha256"]:
        raise ValueError("CAGE-v3 calibration manifest hash mismatch")
    manifest = load_json(manifest_path) if verify_artifacts else {}
    if verify_artifacts:
        validate_cage_v3_manifest(manifest, protocol=protocol, protocol_sha256=protocol_sha256)
        if len(manifest["cases"]) != execution["input_manifest"]["case_count"]:
            raise ValueError("CAGE-v3 calibration manifest count mismatch")
        cpu_path = Path(execution["cpu_acceptance"]["path"])
        if file_sha256(cpu_path) != execution["cpu_acceptance"]["sha256"]:
            raise ValueError("CAGE-v3 CPU acceptance hash mismatch")
        if load_json(cpu_path).get("status") != execution["cpu_acceptance"]["status"]:
            raise ValueError("CAGE-v3 CPU acceptance status mismatch")
    if tuple(execution.get("scientific_payload_fields", ())) != SCIENTIFIC_FIELDS:
        raise ValueError("CAGE-v3 calibration scientific fields changed")
    authorization = execution.get("authorization", {})
    if authorization != {
        "calibration_metrics": True,
        "screen_metrics": False,
        "holdout_metrics": False,
        "reserved_unseen_metrics": False,
        "end_to_end_quality": False,
    }:
        raise ValueError("CAGE-v3 calibration authorization changed")
    expected_stages = {
        "calibration_acceptance": ([0], 6),
        "calibration_full": (list(range(5)), 30),
    }
    for stage, (anchors, count) in expected_stages.items():
        record = execution["stages"][stage]
        if record["anchor_indices"] != anchors or record["case_count"] != count:
            raise ValueError(f"CAGE-v3 {stage} anchors/count changed")
    families = execution.get("probe_families", [])
    if tuple(row.get("family_id") for row in families) != (
        "pure-sr2-sink32-uniform32",
        "pure-sr2-sink64-uniform32",
    ):
        raise ValueError("CAGE-v3 calibration probe families changed")
    for family in families:
        if family.get("two_bit_channels") != 32:
            raise ValueError("CAGE-v3 calibration probes must use uniform32")
        for point in family["points"]:
            report = estimate_qwen3_cage_v2_bytes(
                seq_len=point["prompt_length"],
                residual_length=point["residual_length"],
                one_bit_channels=0,
                two_bit_channels=32,
                sink_length=family["sink_length"],
            )
            if report["model_total_bytes"] != point["packed_bytes"] or point["packed_bytes"] > point["target_bytes"]:
                raise ValueError("CAGE-v3 calibration probe bytes changed")
    return execution, file_sha256(source), protocol, manifest


def expand_calibration_cases(
    *,
    execution: dict[str, Any],
    execution_sha256: str,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    stage: str,
) -> list[dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"unsupported CAGE-v3 calibration stage: {stage}")
    allowed_anchors = execution["stages"][stage]["anchor_indices"]
    base_cases = {
        (case["identity"]["anchor_index"], case["identity"]["prompt_length"]): case
        for case in manifest["cases"]
    }
    expanded = []
    for anchor_index in allowed_anchors:
        for family in execution["probe_families"]:
            for point in family["points"]:
                base = base_cases[(anchor_index, point["prompt_length"])]
                method = {
                    "id": point["method_id"],
                    "name": "cage_v3_calibration_probe",
                    "family_id": family["family_id"],
                    "prompt_length": point["prompt_length"],
                    "target_bytes": point["target_bytes"],
                    "packed_bytes": point["packed_bytes"],
                    "config": {
                        "one_bit_channels": 0,
                        "two_bit_channels": 32,
                        "sink_length": family["sink_length"],
                        "residual_length": point["residual_length"],
                    },
                }
                identity = {
                    "execution_id": execution["execution_id"],
                    "execution_sha256": execution_sha256,
                    "protocol_id": protocol["protocol_id"],
                    "stage": stage,
                    "base_case_id": base["case_id"],
                    "method_id": method["id"],
                }
                expanded.append({
                    "case_id": canonical_sha256(identity)[:24],
                    "identity": identity,
                    "method": method,
                    "input": {
                        "base_case_id": base["case_id"],
                        "anchor_index": anchor_index,
                        "prompt_length": point["prompt_length"],
                        "continuation_start": base["identity"]["continuation_start"],
                        "prompt_ids_sha256": base["identity"]["prompt_ids_sha256"],
                        "continuation_ids_sha256": base["identity"]["continuation_ids_sha256"],
                    },
                    "prompt_ids": base["prompt_ids"],
                    "continuation_ids": base["continuation_ids"],
                })
    if len(expanded) != execution["stages"][stage]["case_count"]:
        raise ValueError("CAGE-v3 expanded calibration case count mismatch")
    if len({case["case_id"] for case in expanded}) != len(expanded):
        raise ValueError("CAGE-v3 calibration case IDs are not unique")
    return expanded


__all__ = [
    "EXECUTION_ID",
    "SCIENTIFIC_FIELDS",
    "STAGES",
    "canonical_sha256",
    "expand_calibration_cases",
    "load_calibration_execution",
    "load_json",
]
