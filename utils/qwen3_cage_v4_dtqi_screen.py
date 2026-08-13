from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v3_screen import validate_quota_plan
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256, load_data_protocol
from utils.qwen3_cage_v4_dtqi_acceptance import load_json
from utils.qwen3_cage_v4_dtqi_protocol import load_dtqi_protocol
from utils.qwen3_cage_v4_metric_protocol import load_metric_protocol
from utils.qwen3_cage_v4_screen import screen_input_cases


EXECUTION_ID = "qwen3-8b-cage-v4-dtqi-pg19-screen-execution-v1"
RECEIPT_ID = "qwen3-8b-cage-v4-dtqi-gpu-acceptance-receipt-v1"
STAGE = "screen_full"
SCIENTIFIC_FIELDS = ("case_id", "method", "input", "memory", "scoring", "cache", "resume")


class CageV4DTQIScreenError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV4DTQIScreenError(message)


def validate_gpu_receipt(receipt: Mapping[str, Any], *, verify_artifacts: bool) -> None:
    _require(receipt.get("schema_version") == 1, "GPU receipt schema mismatch")
    _require(receipt.get("receipt_id") == RECEIPT_ID, "GPU receipt identity mismatch")
    _require(receipt.get("status") == "frozen_after_gpu_acceptance_postrun_before_full_screen", "GPU receipt status mismatch")
    _require(receipt.get("claim_eligible") is False, "GPU receipt claim boundary changed")
    _require(receipt.get("acceptance_source_commit") == "5a2cceb95e8975e86246f63a8c555e1da2ce68cb", "GPU receipt source changed")
    _require(receipt.get("acceptance_execution_sha256") == "da87398a5870aa5bc49321995901f85e635f558a3c06551bdcd76c316bd8b972", "GPU receipt execution changed")
    _require(receipt.get("verified") == {
        "repeat_count": 2,
        "case_count_per_repeat": 3,
        "total_case_file_count": 6,
        "failure_count": 0,
        "failed_postrun_validation_attempt_count": 1,
        "repeat_payloads_bitwise_equal": True,
        "scientific_payload_sha256": "e94f84b8446ab5eda22ef4e0fe9c637cbe69cc5caf7ca65b1a311a529ba2cb73",
        "comparison_sha256": "a5d2630d285ce8c0bea5ba3a2d143550ff3446740469c5e718b8087c4a506021",
    }, "GPU receipt verified payload changed")
    _require(receipt.get("authorization") == {
        "gpu_full_screen": True,
        "screen_case_count": 120,
        "reuse_frozen_baseline_results": True,
        "holdout_method_metrics": False,
        "pg19_test_access": False,
        "interpret_before_full_postrun": False,
        "runtime_claims": False,
        "paper_claims": False,
    }, "GPU receipt authorization changed")
    if verify_artifacts:
        spec = receipt["postrun_audit"]
        for path_key, sha_key, size_key in (
            ("path", "sha256", "size_bytes"),
            ("execution_log", "execution_log_sha256", "execution_log_size_bytes"),
        ):
            path = Path(spec[path_key])
            _require(path.is_file(), f"GPU receipt artifact missing: {path_key}")
            _require(file_sha256(path) == spec[sha_key], f"GPU receipt artifact hash mismatch: {path_key}")
            _require(path.stat().st_size == spec[size_key], f"GPU receipt artifact size mismatch: {path_key}")


def _quotas(plan: Mapping[str, Any], length: int) -> list[int]:
    rows = [row for row in plan["plans"] if row["family_id"] == "pure-sr2-sink32-uniform32" and row["prompt_length"] == length]
    _require(len(rows) == 1, "DTQI screen quota row is not unique")
    return list(rows[0]["layer_two_bit_channel_quotas"])


def load_screen_execution(path: str | Path, *, repo_root: str | Path, verify_artifacts: bool):
    root = Path(repo_root)
    source = Path(path)
    execution = load_json(source)
    _require(execution.get("schema_version") == 1 and execution.get("execution_id") == EXECUTION_ID, "DTQI screen execution identity mismatch")
    _require(execution.get("status") == "frozen_after_gpu_acceptance_before_full_screen", "DTQI screen execution status mismatch")
    _require(execution.get("claim_eligible") is False, "DTQI screen claim boundary changed")
    receipt_spec = execution["gpu_acceptance_receipt"]
    receipt_path = root / receipt_spec["path"]
    _require(file_sha256(receipt_path) == receipt_spec["sha256"], "GPU receipt file hash mismatch")
    receipt = load_json(receipt_path)
    validate_gpu_receipt(receipt, verify_artifacts=verify_artifacts)
    protocol, protocol_sha = load_dtqi_protocol(root / execution["dtqi_protocol"]["path"], repo_root=root)
    _require(protocol_sha == execution["dtqi_protocol"]["sha256"], "DTQI protocol hash mismatch")
    metric, metric_sha = load_metric_protocol(root / execution["metric_protocol"]["path"])
    _require(metric_sha == execution["metric_protocol"]["sha256"], "metric protocol hash mismatch")
    _, data_sha = load_data_protocol(root / execution["data_protocol"]["path"])
    _require(data_sha == execution["data_protocol"]["sha256"], "data protocol hash mismatch")
    quota_path = root / execution["quota_plan"]["path"]
    _require(file_sha256(quota_path) == execution["quota_plan"]["sha256"], "quota plan hash mismatch")
    quota = load_json(quota_path)
    validate_quota_plan(quota)
    if verify_artifacts:
        manifest_path = Path(execution["input_manifest"]["path"])
        _require(manifest_path.is_file(), "DTQI screen input manifest missing")
        _require(file_sha256(manifest_path) == execution["input_manifest"]["sha256"], "DTQI screen input manifest hash mismatch")
        for name, spec in execution["source_files"].items():
            file = root / spec["path"]
            _require(file.is_file() and file_sha256(file) == spec["sha256"], f"DTQI screen source changed: {name}")
    _require(execution.get("model") == {
        "path": "/root/autodl-tmp/models/Qwen3-8B",
        "hf_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "parameter_count": 8190735360,
        "dtype": "torch.float16",
        "device": "cuda:0",
        "attention_implementation": "flash_attention_2",
        "config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
        "model_index_sha256": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    }, "DTQI screen model identity changed")
    _require(execution.get("environment") == {
        "conda_prefix": "/root/autodl-tmp/conda-envs/cage-qwen3",
        "python": "3.10.20",
        "torch": "2.4.1+cu121",
        "transformers": "4.53.2",
        "cuda": "12.1",
        "gpu": "NVIDIA GeForce RTX 4090 D",
    }, "DTQI screen environment identity changed")
    _require(execution.get("screen") == {
        "partition": "screen",
        "documents": 20,
        "anchors_per_document": 2,
        "prompt_lengths": [1024, 2048, 4032],
        "continuation_tokens": 64,
        "case_count": 120,
        "candidate_count": 1,
        "reuse_frozen_baseline_results": True,
        "local_mse_computed": False,
        "resume_completed_cases": True,
    }, "DTQI screen case policy changed")
    _require(execution.get("execution_boundary") == {
        "gpu_full_screen_authorized": True,
        "holdout_method_metrics_authorized": False,
        "pg19_test_access_authorized": False,
        "interpret_before_full_postrun_authorized": False,
        "runtime_claims_authorized": False,
        "paper_claims_authorized": False,
    }, "DTQI screen execution boundary changed")
    return execution, file_sha256(source), protocol, metric, quota


def expand_screen_cases(*, execution: Mapping[str, Any], execution_sha256: str, protocol: Mapping[str, Any], quota_plan: Mapping[str, Any], input_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = screen_input_cases(input_manifest)
    points = {point["prompt_length"]: point for point in protocol["candidate"]["points"]}
    cases = []
    for source in inputs:
        length = source["identity"]["prompt_length"]
        point = points[length]
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
                "two_bit_channels": _quotas(quota_plan, length),
                "global_query_weight": 0.5,
                "recent_query_weight": 0.5,
            },
        }
        identity = {"execution_sha256": execution_sha256, "stage": STAGE, "method_id": method["id"], "input_case_id": source["input_case_id"]}
        cases.append({
            "case_id": canonical_sha256(identity)[:24],
            "method": method,
            "input": copy.deepcopy(source["identity"]),
            "prompt_ids": list(source["prompt_ids"]),
            "continuation_ids": list(source["continuation_ids"]),
        })
    _require(len(cases) == 120 and len({case["case_id"] for case in cases}) == 120, "DTQI screen case count/identity mismatch")
    return cases


__all__ = ["EXECUTION_ID", "SCIENTIFIC_FIELDS", "STAGE", "CageV4DTQIScreenError", "expand_screen_cases", "load_screen_execution", "validate_gpu_receipt"]
