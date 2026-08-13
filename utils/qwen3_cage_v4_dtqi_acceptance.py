from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v3_screen import validate_quota_plan
from utils.qwen3_cage_v4_acceptance import acceptance_input_cases
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256
from utils.qwen3_cage_v4_data import load_data_protocol
from utils.qwen3_cage_v4_dtqi_protocol import load_dtqi_protocol
from utils.qwen3_cage_v4_metric_protocol import load_metric_protocol


EXECUTION_ID = "qwen3-8b-cage-v4-dtqi-gpu-acceptance-v1"
CPU_RECEIPT_ID = "qwen3-8b-cage-v4-dtqi-cpu-acceptance-receipt-v1"
STAGE = "gpu_acceptance"
SCIENTIFIC_FIELDS = (
    "case_id",
    "method",
    "input",
    "memory",
    "scoring",
    "cache",
    "resume",
)


class CageV4DTQIAcceptanceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV4DTQIAcceptanceError(message)


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_cpu_receipt(receipt: Mapping[str, Any], *, verify_artifacts: bool) -> None:
    _require(receipt.get("schema_version") == 1, "CPU receipt schema mismatch")
    _require(receipt.get("receipt_id") == CPU_RECEIPT_ID, "CPU receipt identity mismatch")
    _require(receipt.get("status") == "pass", "CPU receipt status mismatch")
    _require(receipt.get("claim_eligible") is False, "CPU receipt claim boundary changed")
    _require(
        receipt.get("source_commit") == "127a1bc8bde0af79344c4a561877ba303316bf59",
        "CPU receipt source commit changed",
    )
    _require(
        receipt.get("protocol_sha256")
        == "8e5478aa75dd2a8685d1b3839b4260583f44949022df1365678d1f0ee476e549",
        "CPU receipt protocol changed",
    )
    verified = receipt.get("verified", {})
    expected_verified = {
        "unit_test_count": 24,
        "direct_check_count": 13,
        "memory_point_count": 3,
        "all_checks_pass": True,
        "all_memory_points_exact": True,
        "dtqi_changes_synthetic_ranking": True,
        "recent_window_bound_to_residual": True,
        "weights_fixed_half_half": True,
        "one_bit_path_empty": True,
        "value_path_uniform": True,
        "gpu_execution_blocked_during_cpu_acceptance": True,
        "holdout_and_test_blocked": True,
    }
    _require(verified == expected_verified, "CPU receipt verified payload changed")
    authorization = receipt.get("authorization", {})
    _require(
        authorization
        == {
            "freeze_gpu_acceptance_execution": True,
            "gpu_acceptance_repeats": 2,
            "gpu_full_screen": False,
            "holdout_method_metrics": False,
            "pg19_test_access": False,
            "runtime_claims": False,
            "paper_claims": False,
        },
        "CPU receipt authorization changed",
    )
    if verify_artifacts:
        for key in ("artifact", "execution_log"):
            spec = receipt[key]
            path = Path(spec["path"])
            _require(path.is_file(), f"CPU receipt {key} is missing")
            _require(file_sha256(path) == spec["sha256"], f"CPU receipt {key} hash mismatch")
            _require(path.stat().st_size == spec["size_bytes"], f"CPU receipt {key} size mismatch")


def load_acceptance_execution(
    path: str | Path,
    *,
    repo_root: str | Path,
    verify_artifacts: bool,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(repo_root)
    source = Path(path)
    execution = load_json(source)
    _require(
        execution.get("schema_version") == 1
        and execution.get("execution_id") == EXECUTION_ID,
        "DTQI acceptance execution identity mismatch",
    )
    _require(
        execution.get("status") == "frozen_after_cpu_acceptance_before_gpu_acceptance",
        "DTQI acceptance execution status mismatch",
    )
    _require(execution.get("claim_eligible") is False, "DTQI acceptance claim boundary changed")

    protocol_spec = execution["dtqi_protocol"]
    protocol_path = root / protocol_spec["path"]
    protocol, protocol_sha256 = load_dtqi_protocol(protocol_path, repo_root=root)
    _require(protocol_sha256 == protocol_spec["sha256"], "DTQI protocol hash mismatch")

    receipt_spec = execution["cpu_acceptance_receipt"]
    receipt_path = root / receipt_spec["path"]
    _require(file_sha256(receipt_path) == receipt_spec["sha256"], "CPU receipt file hash mismatch")
    receipt = load_json(receipt_path)
    validate_cpu_receipt(receipt, verify_artifacts=verify_artifacts)

    metric_spec = execution["metric_protocol"]
    metric_path = root / metric_spec["path"]
    metric_protocol, metric_sha256 = load_metric_protocol(metric_path)
    _require(metric_sha256 == metric_spec["sha256"], "metric protocol hash mismatch")

    data_spec = execution["data_protocol"]
    data_path = root / data_spec["path"]
    _, data_sha256 = load_data_protocol(data_path)
    _require(data_sha256 == data_spec["sha256"], "data protocol hash mismatch")

    quota_spec = execution["quota_plan"]
    quota_path = root / quota_spec["path"]
    _require(file_sha256(quota_path) == quota_spec["sha256"], "quota plan hash mismatch")
    quota_plan = load_json(quota_path)
    validate_quota_plan(quota_plan)

    input_spec = execution["input_manifest"]
    if verify_artifacts:
        input_path = Path(input_spec["path"])
        _require(input_path.is_file(), "DTQI acceptance input manifest is missing")
        _require(file_sha256(input_path) == input_spec["sha256"], "input manifest hash mismatch")

    _require(
        execution.get("model")
        == {
            "path": "/root/autodl-tmp/models/Qwen3-8B",
            "hf_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "parameter_count": 8190735360,
            "dtype": "torch.float16",
            "device": "cuda:0",
            "attention_implementation": "flash_attention_2",
            "config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
            "model_index_sha256": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
        },
        "DTQI acceptance model identity changed",
    )
    _require(
        execution.get("environment")
        == {
            "conda_prefix": "/root/autodl-tmp/conda-envs/cage-qwen3",
            "python": "3.10.20",
            "torch": "2.4.1+cu121",
            "transformers": "4.53.2",
            "cuda": "12.1",
            "gpu": "NVIDIA GeForce RTX 4090 D",
        },
        "DTQI acceptance environment identity changed",
    )

    acceptance = execution.get("acceptance", {})
    _require(
        acceptance
        == {
            "partition": "screen",
            "document_id": "pg19-validation-27-23a9deb97ca58bf3",
            "anchor_index": 0,
            "prompt_lengths": [1024, 2048, 4032],
            "continuation_tokens": 64,
            "case_count_per_repeat": 3,
            "fresh_repeat_count": 2,
            "required_consistency": "bitwise_equal_json_numeric_payload",
            "local_mse_used_for_selection": False,
        },
        "DTQI acceptance case policy changed",
    )
    boundary = execution.get("execution_boundary", {})
    _require(boundary.get("gpu_acceptance_authorized") is True, "GPU acceptance is not authorized")
    for field in (
        "gpu_full_screen_authorized",
        "holdout_method_metrics_authorized",
        "pg19_test_access_authorized",
        "runtime_claims_authorized",
        "paper_claims_authorized",
    ):
        _require(boundary.get(field) is False, f"DTQI acceptance boundary changed: {field}")
    if verify_artifacts:
        for name, spec in execution.get("source_files", {}).items():
            source_path = root / spec["path"]
            _require(source_path.is_file(), f"DTQI acceptance source missing: {name}")
            _require(file_sha256(source_path) == spec["sha256"], f"DTQI acceptance source changed: {name}")
    return execution, file_sha256(source), protocol, metric_protocol, quota_plan


def _quotas_for_length(plan: Mapping[str, Any], prompt_length: int) -> list[int]:
    matches = [
        row
        for row in plan["plans"]
        if row["family_id"] == "pure-sr2-sink32-uniform32"
        and row["prompt_length"] == prompt_length
    ]
    _require(len(matches) == 1, "matching DTQI quota plan is missing")
    return list(matches[0]["layer_two_bit_channel_quotas"])


def expand_acceptance_cases(
    *,
    execution: Mapping[str, Any],
    execution_sha256: str,
    dtqi_protocol: Mapping[str, Any],
    metric_protocol: Mapping[str, Any],
    quota_plan: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    inputs = acceptance_input_cases(input_manifest, protocol=dict(metric_protocol))
    cases = []
    for point in dtqi_protocol["candidate"]["points"]:
        length = point["prompt_length"]
        source = inputs[length]
        method = {
            "id": f"cage-v4-dtqi-l{length}",
            "candidate_id": "cage-v4-dtqi",
            "prompt_length": length,
            "packed_bytes": point["packed_bytes"],
            "kitty_pro_target_bytes": point["kitty_pro_target_bytes"],
            "config": {
                "residual_length": point["residual_length"],
                "recent_query_window": point["recent_query_window"],
                "sink_length": 32,
                "one_bit_channels": 0,
                "two_bit_channels": _quotas_for_length(quota_plan, length),
                "global_query_weight": 0.5,
                "recent_query_weight": 0.5,
            },
        }
        identity = {
            "execution_sha256": execution_sha256,
            "stage": STAGE,
            "method_id": method["id"],
            "input_case_id": source["input_case_id"],
        }
        cases.append(
            {
                "case_id": canonical_sha256(identity)[:24],
                "method": method,
                "input": copy.deepcopy(source["identity"]),
                "prompt_ids": source["prompt_ids"],
                "continuation_ids": source["continuation_ids"],
            }
        )
    _require(len(cases) == 3, "DTQI acceptance must contain three cases")
    _require(len({case["case_id"] for case in cases}) == 3, "DTQI case IDs are not unique")
    return cases


__all__ = [
    "CPU_RECEIPT_ID",
    "CageV4DTQIAcceptanceError",
    "EXECUTION_ID",
    "SCIENTIFIC_FIELDS",
    "STAGE",
    "expand_acceptance_cases",
    "load_acceptance_execution",
    "load_json",
    "validate_cpu_receipt",
]
