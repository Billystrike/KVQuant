from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


PROTOCOL_ID = "llama2-7b-cage-v3-cross-architecture-transfer-quality-v1"
EXPECTED_PROTOCOL_SHA256 = "355d73629dece9015288d10acdd7c95e08da0cff941a38395d54e1868380b654"
EXPECTED_PROTOCOL_CANONICAL_SHA256 = "fd14b4681a98e24d78d904dc5a7a5c4f2eab4379972e53e3a73a2def946adba3"
EXPECTED_PARENT_SHA256 = "f0007dde3967ba1495a14602df6791a60eab6e5fb82b5742078aa1e3037ded07"
EXPECTED_POSTRUN_SHA256 = "e76dcab6ee54e8e28fd5a552707248208897e65ace5f869384f8e8bfc9d9653f"
EXPECTED_QUOTA_SHA256 = "6fc7e33a63db5ed4b1e303edd50c306cf91ab07b99dc981b1d6ea91a58f845e8"
PROMPT_LENGTHS = (1024, 2048, 4032)


class Llama2CageV3TransferQualityProtocolError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3TransferQualityProtocolError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3TransferQualityProtocolError(
            f"cannot load JSON {path}: {error}"
        ) from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _validate_reference(
    spec: Mapping[str, Any], *, repo_root: Path, expected_path: str, expected_sha: str
) -> dict[str, Any]:
    _require(spec.get("path") == expected_path, f"frozen reference path changed: {expected_path}")
    _require(spec.get("sha256") == expected_sha, f"frozen reference digest changed: {expected_path}")
    path = repo_root / expected_path
    _require(path.is_file(), f"frozen reference is missing: {expected_path}")
    _require(file_sha256(path) == expected_sha, f"frozen reference file changed: {expected_path}")
    return _load(path)


def validate_transfer_quality_protocol(
    protocol: Mapping[str, Any], *, repo_root: Path
) -> None:
    _require(
        canonical_sha256(protocol) == EXPECTED_PROTOCOL_CANONICAL_SHA256,
        "transfer-quality protocol frozen payload mismatch",
    )
    _require(
        protocol.get("schema_version") == 1 and protocol.get("protocol_id") == PROTOCOL_ID,
        "transfer-quality protocol identity mismatch",
    )
    _require(
        protocol.get("status")
        == "frozen_after_production_gpu_acceptance_before_any_llama2_v3_quality_metric",
        "transfer-quality protocol status changed",
    )
    _require(protocol.get("claim_eligible") is False, "claim boundary changed")

    parent = _validate_reference(
        protocol.get("parent_transfer_protocol", {}),
        repo_root=repo_root,
        expected_path="configs/llama2_7b_cage_v3_cross_arch_transfer_protocol_v1.json",
        expected_sha=EXPECTED_PARENT_SHA256,
    )
    _require(
        parent.get("original_promotion_outcome", {}).get("promotion_pass") is False,
        "Qwen3 negative promotion outcome was changed",
    )
    _require(
        parent.get("original_promotion_outcome", {}).get("qwen3_conclusion_is_not_reopened")
        is True,
        "Qwen3 negative conclusion was reopened",
    )
    _require(
        parent.get("staged_execution", {}).get("stage_3_full_transfer", {}).get("current_authorized")
        is False,
        "parent full-transfer boundary changed",
    )

    receipt_spec = protocol.get("gpu_acceptance_postrun_receipt", {})
    receipt = _validate_reference(
        receipt_spec,
        repo_root=repo_root,
        expected_path="configs/llama2_7b_cage_v3_gpu_acceptance_postrun_receipt_v1.json",
        expected_sha=EXPECTED_POSTRUN_SHA256,
    )
    _require(receipt_spec.get("production_gpu_acceptance_pass") is True, "GPU pass changed")
    _require(
        receipt_spec.get("scientific_payload_sha256")
        == "55c72774342885664f7fa388f7a1c0b996cbb864d7cb9f2678eb20fb3c6cce23",
        "GPU scientific payload changed",
    )
    _require(receipt.get("status") == "pass", "GPU post-run receipt is not a pass")
    decision = receipt.get("decision", {})
    _require(decision.get("production_gpu_acceptance_pass") is True, "GPU acceptance did not pass")
    _require(decision.get("quality_protocol_design_authorized") is True, "quality design is not authorized")
    _require(decision.get("formal_transfer_execution_authorized") is False, "quality execution was prematurely authorized")

    quota = _validate_reference(
        protocol.get("quota_plan", {}),
        repo_root=repo_root,
        expected_path="configs/llama2_7b_cage_v3_transfer_quota_plan_v1.json",
        expected_sha=EXPECTED_QUOTA_SHA256,
    )
    _require(quota.get("plan_id") == "llama2-7b-cage-v3-depth-normalized-quota-plan-v1", "quota plan changed")
    _require(quota.get("llama2_metrics_consumed") is False, "quota plan consumed Llama metrics")

    model = protocol.get("model", {})
    _require(
        model
        == {
            "reference": "/root/autodl-tmp/models/Llama-2-7b-hf",
            "dtype": "float16",
            "device": "cuda:0",
            "config_sha256": "9242e7db1bc2a17873e66084c3b1c6ed10883076e156b338fd6a7775748e2e3c",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 128,
            "max_position_embeddings": 4096,
        },
        "model identity changed",
    )

    input_spec = protocol.get("input", {})
    _require(input_spec.get("corpus_id") == "Salesforce/wikitext", "corpus changed")
    _require(input_spec.get("corpus_config") == "wikitext-2-raw-v1", "corpus config changed")
    _require(input_spec.get("split") == "test", "corpus split changed")
    _require(input_spec.get("revision") == "00aa25585682d4957f9e86edc73f59be7419af99", "corpus revision changed")
    _require(input_spec.get("expected_token_count") == 341468, "token count changed")
    _require(
        input_spec.get("expected_token_ids_sha256")
        == "8163e5b39c668be8eec1e4d82eaecb1ee2a09d958f81873773e93bf4a4484f10",
        "token stream changed",
    )
    _require(input_spec.get("anchor_indices") == list(range(50)), "anchor indices changed")
    _require(input_spec.get("prompt_lengths") == list(PROMPT_LENGTHS), "prompt lengths changed")
    _require(input_spec.get("continuation_tokens") == 64, "continuation length changed")
    _require(input_spec.get("all_windows_must_fit_native_context") is True, "context boundary changed")
    _require(input_spec.get("all_windows_must_be_nonoverlapping") is True, "overlap boundary changed")
    _require(max(PROMPT_LENGTHS) + 64 == 4096, "frozen longest window no longer fits native context")

    scoring = protocol.get("scoring", {})
    _require(scoring.get("boundary_target_count") == 1, "boundary target count changed")
    _require(scoring.get("single_token_decode_target_count") == 63, "decode target count changed")
    _require(scoring.get("primary_target_count_per_case") == 64, "primary target count changed")
    _require(scoring.get("paired_unit") == "anchor", "paired unit changed")
    _require(scoring.get("fp16_incremental_cache_path_is_primary") is True, "FP16 path changed")
    _require(scoring.get("fp16_one_shot_reference_is_diagnostic_only") is True, "diagnostic boundary changed")
    _require(scoring.get("same_token_ids_targets_order_and_reduction_for_all_methods") is True, "scoring equality changed")

    expected_matrix = {
        1024: {
            "bytes": (163020800, 163106816, 168820736),
            "ids": ("fp16-l1024", "cage-v3-r176-l1024", "cage-v1-r237-l1024", "kivi-g64-r320-l1024"),
            "candidate": (176, 32),
            "v1_residual": 237,
            "kivi": (64, 320),
        },
        2048: {
            "bytes": (249069568, 248725504, 255852544),
            "ids": ("fp16-l2048", "cage-v3-r288-l2048", "cage-v1-r253-l2048", "kivi-g32-r224-l2048"),
            "candidate": (288, 32),
            "v1_residual": 253,
            "kivi": (32, 224),
        },
        4032: {
            "bytes": (391348224, 391254016, 398196736),
            "ids": ("fp16-l4032", "cage-v3-r112-l4032", "cage-v1-r96-l4032", "kivi-g128-r256-l4032"),
            "candidate": (112, 32),
            "v1_residual": 96,
            "kivi": (128, 256),
        },
    }
    rows = protocol.get("method_length_matrix", [])
    _require(isinstance(rows, list) and len(rows) == 3, "method-length matrix shape changed")
    all_ids: list[str] = []
    for row in rows:
        length = row.get("prompt_length")
        _require(length in expected_matrix, "unexpected prompt length in method matrix")
        expected = expected_matrix[length]
        memory = row.get("packed_memory", {})
        _require(
            (memory.get("candidate_bytes"), memory.get("cage_v1_bytes"), memory.get("kivi_bytes"))
            == expected["bytes"],
            f"packed bytes changed at {length}",
        )
        methods = row.get("methods", [])
        _require(isinstance(methods, list) and len(methods) == 4, f"method count changed at {length}")
        _require(tuple(method.get("id") for method in methods) == expected["ids"], f"method IDs changed at {length}")
        _require(
            tuple(method.get("role") for method in methods)
            == ("quality_reference", "transfer_candidate", "primary_predecessor", "primary_uniform_baseline"),
            f"method roles changed at {length}",
        )
        _require(tuple(method.get("method") for method in methods) == ("fp16", "cage_v3", "cage_v1", "kivi"), f"method families changed at {length}")
        _require(
            (methods[1].get("residual_length"), methods[1].get("sink_length"))
            == expected["candidate"],
            f"candidate point changed at {length}",
        )
        _require(methods[2].get("residual_length") == expected["v1_residual"], f"CAGE-v1 point changed at {length}")
        _require(methods[2].get("key_group_sizes") == [32, 64, 128], f"CAGE-v1 key groups changed at {length}")
        _require(methods[2].get("value_group_sizes") == [32, 64, 128], f"CAGE-v1 value groups changed at {length}")
        _require(methods[2].get("key_clip_percentiles") == [0.999, 0.995, 0.99], f"CAGE-v1 key clips changed at {length}")
        _require(methods[2].get("value_clip_percentiles") == [0.999, 0.995, 0.99], f"CAGE-v1 value clips changed at {length}")
        group_size, residual = expected["kivi"]
        _require(
            (methods[3].get("k_bits"), methods[3].get("v_bits"), methods[3].get("group_size"), methods[3].get("residual_length"))
            == (2, 2, group_size, residual),
            f"KIVI point changed at {length}",
        )
        _require(residual % group_size == 0, f"KIVI residual is not divisible at {length}")
        all_ids.extend(method.get("id") for method in methods)
    _require([row.get("prompt_length") for row in rows] == list(PROMPT_LENGTHS), "method matrix order changed")
    _require(len(all_ids) == len(set(all_ids)) == 12, "method IDs are not unique")

    cases = protocol.get("case_matrix", {})
    _require(
        (cases.get("method_length_points"), cases.get("anchors_per_point"), cases.get("full_case_count"))
        == (12, 50, 600),
        "full case matrix changed",
    )
    _require(cases.get("acceptance_anchor_indices") == [0], "acceptance anchors changed")
    _require(
        (cases.get("acceptance_case_count_per_repeat"), cases.get("acceptance_repeat_count"), cases.get("full_execution_repeat_count"))
        == (12, 2, 1),
        "execution repeat matrix changed",
    )

    gate = protocol.get("primary_transfer_gate", {})
    _require(
        gate.get("comparisons")
        == ["transfer_candidate_vs_primary_predecessor", "transfer_candidate_vs_primary_uniform_baseline"],
        "primary comparisons changed",
    )
    _require(gate.get("per_length_noninferiority_relative_ppl_percent_at_most") == 0.5, "noninferiority threshold changed")
    _require(gate.get("overall_material_superiority_relative_ppl_percent_at_most") == -1.0, "material threshold changed")
    _require((gate.get("bootstrap_resamples"), gate.get("bootstrap_seed")) == (10000, 20260817), "bootstrap changed")
    for key in (
        "ci95_upper_bound_must_be_below_zero",
        "both_primary_comparisons_must_pass",
        "decimal_direction_alone_is_never_a_pass",
        "no_threshold_or_candidate_change_after_any_quality_result",
    ):
        _require(gate.get(key) is True, f"transfer gate changed: {key}")

    reporting = protocol.get("reporting", {})
    _require(reporting.get("report_qwen3_promotion_failure") is True, "Qwen3 failure reporting changed")
    _require(reporting.get("report_all_llama2_lengths_and_unfavorable_results") is True, "complete reporting changed")
    _require(reporting.get("canonical_full_corpus_perplexity_claim") is False, "canonical PPL claim authorized")
    _require(reporting.get("latency_throughput_or_real_cuda_memory_claim") is False, "runtime claim authorized")
    _require(reporting.get("kitty_llama_included") is False, "Kitty was silently included")
    _require(reporting.get("kitty_requires_separate_nonofficial_port_fidelity_protocol") is True, "Kitty fidelity boundary changed")

    stages = protocol.get("staged_execution", {})
    _require(stages.get("static_protocol_preflight_authorized") is True, "static preflight authorization changed")
    _require(stages.get("input_manifest_build_authorized_only_after_static_preflight_pass") is True, "manifest gate changed")
    _require(stages.get("two_repeat_quality_acceptance_authorized_only_after_input_manifest_freeze") is True, "acceptance gate changed")
    _require(stages.get("full_600_case_execution_authorized_only_after_acceptance_postrun_gate") is True, "full gate changed")
    boundary = protocol.get("current_boundary", {})
    for key in (
        "corpus_access_authorized",
        "quality_acceptance_execution_authorized",
        "full_transfer_execution_authorized",
        "quality_metrics_read",
        "candidate_tuning_authorized",
        "kitty_llama_port_authorized",
        "paper_main_method_change_authorized",
        "runtime_claims_authorized",
    ):
        _require(boundary.get(key) is False, f"current boundary changed: {key}")


def load_transfer_quality_protocol(
    path: Path, *, repo_root: Path
) -> tuple[dict[str, Any], str]:
    protocol = _load(path)
    validate_transfer_quality_protocol(protocol, repo_root=repo_root)
    digest = file_sha256(path)
    _require(digest == EXPECTED_PROTOCOL_SHA256, "transfer-quality protocol file hash mismatch")
    return protocol, digest


__all__ = [
    "EXPECTED_POSTRUN_SHA256",
    "EXPECTED_PROTOCOL_SHA256",
    "EXPECTED_QUOTA_SHA256",
    "Llama2CageV3TransferQualityProtocolError",
    "PROTOCOL_ID",
    "load_transfer_quality_protocol",
    "validate_transfer_quality_protocol",
]
