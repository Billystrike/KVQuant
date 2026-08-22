from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from utils.llama2_cage_v3_transfer_quality_acceptance import lf_normalized_file_sha256
from utils.llama2_cage_v3_transfer_quality_manifest import validate_input_manifest
from utils.llama2_cage_v3_transfer_quality_protocol import load_transfer_quality_protocol
from utils.qwen3_cage_v4_data import file_sha256


DESIGN_ID = "llama2-7b-cage-v3-transfer-quality-full-design-v1"
DESIGN_SHA256 = "bda724619ad986703ca359c6a7fa66a44d0d762151689895ce18cd6574483e7a"
POSTRUN_RECEIPT_SHA256 = "913a09ca5d7f743a8ad3eb9147ff4193424de89903ecd593afd579ca9559aba4"


class Llama2CageV3TransferQualityFullDesignError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3TransferQualityFullDesignError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3TransferQualityFullDesignError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_postrun_receipt(receipt: Mapping[str, Any], *, repo_root: Path) -> None:
    _require(receipt.get("schema_version") == 1, "postrun receipt schema changed")
    _require(
        receipt.get("receipt_id") == "llama2-7b-cage-v3-transfer-quality-acceptance-postrun-receipt-v1",
        "postrun receipt identity changed",
    )
    _require(
        receipt.get("status") == "frozen_after_successful_joint_acceptance_postrun_before_full_execution_design"
        and receipt.get("claim_eligible") is False,
        "postrun receipt status changed",
    )
    acceptance = receipt.get("acceptance", {})
    _require(
        acceptance.get("case_count_per_repeat") == 12
        and acceptance.get("repeat_count") == 2
        and acceptance.get("total_case_file_count") == 24
        and acceptance.get("failure_count") == 0
        and acceptance.get("scientific_payload_sha256")
        == "bf0dcb6deb38bc1b6c7433f911b8abcc0ed819b92dac32ffe9c70f61b285b592"
        and acceptance.get("repeat_payloads_bitwise_equal") is True
        and acceptance.get("comparison_mismatch_case_ids") == []
        and acceptance.get("interpretation_performed") is False,
        "postrun acceptance evidence changed",
    )
    history = receipt.get("preserved_history", {})
    _require(
        history.get("failed_pre_metric_attempt_count") == 1
        and history.get("failed_postrun_attempt_count") == 1
        and history.get("failed_postrun_receipt_sha256")
        == "a07c70105b120073d46ec2ce26e8d4f0d0cc89d5e7b7dd6080bcb4b37c457006",
        "postrun failure history changed",
    )
    _require(
        lf_normalized_file_sha256(repo_root / history["failed_postrun_receipt_path"])
        == history["failed_postrun_receipt_sha256"],
        "postrun failure receipt file changed",
    )
    package = receipt.get("environment_package_check", {})
    _require(
        package.get("status") == "known_dependency_metadata_incompatibilities_reported_after_successful_comparison"
        and package.get("scientific_acceptance_affected") is False
        and package.get("environment_mutation_authorized") is False
        and package.get("full_execution_gate_must_revalidate_required_imports") is True,
        "postrun package disclosure changed",
    )
    _require(
        receipt.get("decision")
        == {
            "acceptance_complete": True,
            "full_execution_design_preflight_authorized": True,
            "full_runner_implementation_authorized": False,
            "full_execution_gate_authorized": False,
            "full_600_case_execution_authorized": False,
            "candidate_tuning_authorized": False,
            "quality_interpretation_authorized": False,
            "paper_claims_authorized": False,
            "runtime_claims_authorized": False,
        },
        "postrun decision boundary changed",
    )


def validate_design(design: Mapping[str, Any], *, repo_root: Path) -> None:
    _require(design.get("schema_version") == 1 and design.get("design_id") == DESIGN_ID, "full design identity changed")
    _require(
        design.get("status") == "frozen_after_acceptance_postrun_before_full_runner_or_gate_implementation"
        and design.get("claim_eligible") is False,
        "full design status changed",
    )
    protocol_spec = design.get("protocol", {})
    _require(
        protocol_spec
        == {
            "path": "configs/llama2_7b_cage_v3_transfer_quality_protocol_v1.json",
            "sha256": "355d73629dece9015288d10acdd7c95e08da0cff941a38395d54e1868380b654",
        },
        "full design protocol link changed",
    )
    protocol, protocol_sha = load_transfer_quality_protocol(repo_root / protocol_spec["path"], repo_root=repo_root)
    _require(protocol_sha == protocol_spec["sha256"], "full design protocol file changed")
    postrun = design.get("acceptance_postrun_receipt", {})
    _require(
        postrun
        == {
            "path": "configs/llama2_7b_cage_v3_transfer_quality_acceptance_postrun_receipt_v1.json",
            "sha256": POSTRUN_RECEIPT_SHA256,
        },
        "full design postrun link changed",
    )
    receipt_path = repo_root / postrun["path"]
    _require(lf_normalized_file_sha256(receipt_path) == postrun["sha256"], "postrun receipt file changed")
    validate_postrun_receipt(_load(receipt_path), repo_root=repo_root)
    input_spec = design.get("input_manifest", {})
    _require(
        input_spec.get("sha256") == "ef6b3e9c48e219ab09d35dadca4026f2cd99f47c3c922d8a8e1d2b5c4db19d04"
        and input_spec.get("size_bytes") == 5160675
        and input_spec.get("anchor_count") == 50
        and input_spec.get("input_record_count") == 150,
        "full design input manifest changed",
    )
    quota = design.get("quota_plan", {})
    _require(
        quota
        == {
            "path": "configs/llama2_7b_cage_v3_transfer_quota_plan_v1.json",
            "sha256": "6fc7e33a63db5ed4b1e303edd50c306cf91ab07b99dc981b1d6ea91a58f845e8",
        }
        and file_sha256(repo_root / quota["path"]) == quota["sha256"],
        "full design quota changed",
    )
    matrix = design.get("case_matrix", {})
    _require(matrix.get("prompt_lengths") == [1024, 2048, 4032], "full design lengths changed")
    _require(
        matrix.get("method_roles")
        == ["quality_reference", "transfer_candidate", "primary_predecessor", "primary_uniform_baseline"],
        "full design method roles changed",
    )
    _require(matrix.get("anchor_indices") == list(range(50)), "full design anchors changed")
    _require(
        matrix.get("method_length_point_count") == 12
        and matrix.get("anchors_per_point") == 50
        and matrix.get("full_case_count") == 600
        and matrix.get("target_count_per_case") == 64
        and matrix.get("total_target_count") == 38400,
        "full design counts changed",
    )
    _require(
        matrix.get("execution_order") == "prompt_length_then_protocol_method_order_then_anchor_index",
        "full design execution order changed",
    )
    _require(
        len(protocol["method_length_matrix"]) == 3
        and sum(len(row["methods"]) for row in protocol["method_length_matrix"]) == 12,
        "protocol method-length matrix changed",
    )
    scoring = design.get("scoring", {})
    _require(
        scoring.get("primary_metric") == "cache_conditioned_all_64_target_mean_nll"
        and scoring.get("boundary_target_count") == 1
        and scoring.get("single_token_decode_target_count") == 63
        and scoring.get("target_count_per_case") == 64
        and scoring.get("fp16_one_shot_reference") == "diagnostic_only"
        and scoring.get("same_token_ids_targets_order_and_reduction_for_all_methods") is True,
        "full design scoring changed",
    )
    determinism = design.get("determinism", {})
    _require(
        determinism
        == {
            "seed": 20260817,
            "cublas_workspace_config": ":4096:8",
            "torch_deterministic_algorithms": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "batch_size": 1,
            "per_case_seed_reset_required": True,
        },
        "full design determinism changed",
    )
    output = design.get("output_and_resume", {})
    _require(
        output.get("output_dir") == "/root/autodl-tmp/llama2_cage_v3/transfer_quality_full_v1"
        and output.get("fresh_before_first_attempt") is True
        and output.get("summary_write") == "only_after_all_600_cases_validate"
        and output.get("existing_failure_record_removed_only_after_same_case_completes") is True,
        "full design output/resume boundary changed",
    )
    environment = design.get("environment", {})
    _require(
        environment.get("python_major_minor") == "3.10"
        and environment.get("torch") == "2.4.1+cu121"
        and environment.get("transformers") == "4.43.1"
        and environment.get("gpu") == "NVIDIA GeForce RTX 4090 D"
        and len(environment.get("required_imports", [])) == 6
        and len(environment.get("known_pip_check_issues", [])) == 4
        and environment.get("do_not_mutate_frozen_environment_for_metadata_only_warnings") is True
        and environment.get("full_gate_must_import_every_required_runtime_module") is True,
        "full design environment policy changed",
    )
    _require(
        design.get("analysis_boundary")
        == {
            "read_or_interpret_partial_results": False,
            "formal_analysis_only_after_600_case_postrun_pass": True,
            "report_all_lengths_and_unfavorable_results": True,
            "candidate_or_threshold_change_after_full_execution_starts": False,
            "kitty_llama_included": False,
        },
        "full design analysis boundary changed",
    )
    _require(
        design.get("authorization")
        == {
            "static_design_preflight": True,
            "full_runner_and_gate_implementation_after_design_preflight_pass": True,
            "full_execution_gate_run": False,
            "full_600_case_execution": False,
            "quality_interpretation": False,
            "candidate_tuning": False,
            "paper_main_method_change": False,
            "runtime_claims": False,
        },
        "full design authorization changed",
    )


def validate_server_artifacts(design: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    receipt = _load(repo_root / design["acceptance_postrun_receipt"]["path"])
    output_spec = receipt["postrun_output"]
    output_path = Path(output_spec["path"])
    _require(output_path.is_file(), "server acceptance postrun output is missing")
    _require(output_path.stat().st_size == output_spec["size_bytes"], "server acceptance postrun output size changed")
    _require(file_sha256(output_path) == output_spec["sha256"], "server acceptance postrun output hash changed")
    output = _load(output_path)
    _require(
        output.get("status") == "pass"
        and output.get("interpretation_performed") is False
        and output.get("total_case_file_count") == 24
        and output.get("failure_count") == 0
        and output.get("repeat_payloads_bitwise_equal") is True,
        "server acceptance postrun output changed",
    )
    log_spec = receipt["execution_log"]
    log_path = Path(log_spec["path"])
    _require(log_path.is_file(), "server acceptance postrun log is missing")
    _require(log_path.stat().st_size == log_spec["size_bytes"], "server acceptance postrun log size changed")
    _require(file_sha256(log_path) == log_spec["sha256"], "server acceptance postrun log hash changed")
    input_spec = design["input_manifest"]
    input_path = Path(input_spec["path"])
    _require(input_path.is_file(), "server input manifest is missing")
    _require(input_path.stat().st_size == input_spec["size_bytes"], "server input manifest size changed")
    _require(file_sha256(input_path) == input_spec["sha256"], "server input manifest hash changed")
    protocol, _ = load_transfer_quality_protocol(repo_root / design["protocol"]["path"], repo_root=repo_root)
    input_manifest = _load(input_path)
    validate_input_manifest(input_manifest, protocol=protocol)
    output_dir = Path(design["output_and_resume"]["output_dir"])
    _require(not output_dir.exists(), "full output directory already exists before first authorized attempt")
    return {
        "acceptance_postrun_output_sha256": output_spec["sha256"],
        "acceptance_postrun_log_sha256": log_spec["sha256"],
        "input_manifest_sha256": input_spec["sha256"],
        "input_record_count": 150,
        "expanded_full_case_count": 600,
        "full_output_exists": False,
    }


__all__ = [
    "DESIGN_ID",
    "DESIGN_SHA256",
    "Llama2CageV3TransferQualityFullDesignError",
    "validate_design",
    "validate_postrun_receipt",
    "validate_server_artifacts",
]
