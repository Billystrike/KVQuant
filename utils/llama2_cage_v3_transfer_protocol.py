from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256
from utils.qwen3_memory import estimate_qwen3_cage_bytes, estimate_qwen3_kivi_bytes


PROTOCOL_ID = "llama2-7b-cage-v3-cross-architecture-transfer-v1"
PROTOCOL_CANONICAL_SHA256 = "73ad2a81e887ab2e94937a0f1ebe88736ab310bd98fbeb9c42c77a0fdc7346a9"
EXPECTED_PROTOCOL_SHA256 = "f0007dde3967ba1495a14602df6791a60eab6e5fb82b5742078aa1e3037ded07"
EXPECTED_QWEN_PLAN_SHA256 = "01a1063651d4565a732474b9f27f7a674f434750e7aea2f55429f619bed91914"
PROMPT_LENGTHS = (1024, 2048, 4032)


class Llama2CageV3TransferProtocolError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3TransferProtocolError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3TransferProtocolError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_transfer_protocol(protocol: Mapping[str, Any], *, repo_root: Path) -> None:
    _require(canonical_sha256(protocol) == PROTOCOL_CANONICAL_SHA256, "transfer protocol frozen payload mismatch")
    _require(protocol.get("schema_version") == 1 and protocol.get("protocol_id") == PROTOCOL_ID, "transfer protocol identity mismatch")
    _require(protocol.get("status") == "prospectively_frozen_after_qwen3_promotion_failure_before_any_llama2_cage_v3_metric", "transfer protocol status mismatch")
    _require(protocol.get("claim_eligible") is False, "transfer protocol claim boundary changed")
    outcome = protocol.get("original_promotion_outcome", {})
    _require(outcome.get("promotion_pass") is False, "Qwen3 promotion failure was changed")
    _require(outcome.get("kitty_pro_gate_pass") is False, "Qwen3 Kitty-Pro failure was changed")
    _require(math.isclose(outcome.get("kitty_pro_4032_relative_ppl_percent"), 1.1904321873860653, rel_tol=0.0, abs_tol=0.0), "Qwen3 4032 result was changed")
    _require(outcome.get("qwen3_conclusion_is_not_reopened") is True, "Qwen3 conclusion reopening is forbidden")
    disclosure = protocol.get("protocol_deviation_disclosure", {})
    for key in (
        "original_stop_rule_acknowledged",
        "original_stop_rule_would_not_authorize_llama2_v3",
        "old_protocol_is_not_edited_or_reinterpreted",
        "followup_was_defined_after_observing_qwen3_promotion_results",
        "followup_is_frozen_before_any_llama2_v3_metric",
    ):
        _require(disclosure.get(key) is True, f"transfer deviation disclosure changed: {key}")
    source = protocol.get("source_candidate", {})
    _require(source.get("method_id") == "cage-v3-sr2-sink32-calibrated", "transfer candidate changed")
    _require(source.get("sink_length") == 32 and source.get("one_bit_channels") == 0, "transfer candidate policy changed")
    _require(source.get("qwen3_residual_lengths") == {"1024": 176, "2048": 288, "4032": 112}, "transfer residual lengths changed")
    _require(source.get("no_llama_specific_candidate_search") is True and source.get("no_llama_metric_calibration") is True, "target-specific tuning was authorized")
    target = protocol.get("target_model_preflight", {})
    _require(
        (
            target.get("expected_num_hidden_layers"),
            target.get("expected_num_attention_heads"),
            target.get("expected_num_key_value_heads"),
            target.get("expected_head_dim"),
            target.get("expected_native_context"),
        )
        == (32, 32, 32, 128, 4096),
        "target architecture changed",
    )
    normalization = protocol.get("architecture_normalization", {})
    _require(normalization.get("no_target_metric_used_by_mapping") is True, "target metrics entered architecture mapping")
    _require(
        normalization.get("target_quota_rule")
        == {
            "top_ranked_11_layers": 48,
            "middle_ranked_10_layers": 32,
            "bottom_ranked_11_layers": 16,
            "total_two_bit_channels_across_32_layers_per_kv_head": 1024,
            "mean_two_bit_channels_per_layer_per_kv_head": 32,
        },
        "target quota rule changed",
    )
    scoring = protocol.get("input_and_scoring", {})
    _require(scoring.get("token_ids_sha256") == "8163e5b39c668be8eec1e4d82eaecb1ee2a09d958f81873773e93bf4a4484f10", "Llama token stream changed")
    _require(scoring.get("anchor_count") == 50 and scoring.get("prompt_lengths") == list(PROMPT_LENGTHS) and scoring.get("continuation_tokens") == 64, "transfer input grid changed")
    _require(scoring.get("known_baseline_results_do_not_authorize_candidate_or_threshold_changes") is True, "known baselines entered tuning")
    roles = protocol.get("method_roles", {})
    _require(roles.get("transfer_candidate") == "cage-v3-llama2-depth-normalized", "transfer method changed")
    _require(
        roles.get("primary_predecessor") == "cage-v1-closest-packed-byte"
        and roles.get("primary_uniform_baseline") == "kivi-closest-feasible-int2",
        "primary baselines changed",
    )
    _require(roles.get("cage_v1_match_tolerance_relative_to_candidate") == 0.0025, "CAGE-v1 memory tolerance changed")
    _require("do not call the comparison equal-memory" in roles.get("kivi_feasibility_boundary", ""), "KIVI memory asymmetry disclosure changed")
    memory = protocol.get("frozen_packed_memory_preflight", {})
    _require(memory.get("quality_metrics_consumed") is False and memory.get("grid_change_after_preflight") is False, "memory grid boundary changed")
    _require(
        memory.get("model_shape")
        == {
            "batch_size": 1,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "head_dim": 128,
            "bytes_per_meta": 2,
            "bytes_per_full_precision": 2,
        },
        "memory model shape changed",
    )
    _require(
        memory.get("cage_v1_grid")
        == {
            "residual_length_min": 1,
            "residual_length_max": 512,
            "residual_length_step": 1,
            "key_bucket_sizes": [42, 43, 43],
            "value_bucket_sizes": [42, 43, 43],
            "key_group_sizes": [32, 64, 128],
            "value_group_sizes": [32, 64, 128],
            "bits": 2,
            "bytes_per_bucket_index": 8,
        },
        "CAGE-v1 memory grid changed",
    )
    _require(
        memory.get("kivi_grid")
        == {
            "bits": 2,
            "group_sizes": [32, 64, 128],
            "residual_length_rule": "positive multiples of group_size at most 512",
            "sink_length": 0,
        },
        "KIVI memory grid changed",
    )
    gate = protocol.get("primary_transfer_gate", {})
    _require(gate.get("comparisons") == ["cage-v1-closest-packed-byte", "kivi-closest-feasible-int2"], "transfer comparisons changed")
    _require(gate.get("per_length_noninferiority_relative_ppl_percent_at_most") == 0.5, "per-length threshold changed")
    _require(gate.get("overall_material_superiority_relative_ppl_percent_at_most") == -1.0, "material threshold changed")
    _require(gate.get("bootstrap_resamples") == 10_000 and gate.get("bootstrap_seed") == 20_260_817, "bootstrap changed")
    _require(gate.get("both_primary_comparisons_must_pass") is True and gate.get("decimal_direction_alone_is_never_a_pass") is True, "transfer gate changed")
    statistics = protocol.get("statistics", {})
    _require(statistics.get("paired_unit") == "anchor", "paired statistical unit changed")
    _require("not document-clustered or population-level significance" in statistics.get("inference_scope", ""), "inference scope changed")
    _require(statistics.get("canonical_full_corpus_perplexity_claim") is False, "canonical PPL claim was authorized")
    kitty = protocol.get("kitty_secondary_track", {})
    _require(kitty.get("included_in_primary_transfer_decision") is False and kitty.get("current_execution_authorized") is False, "Kitty secondary boundary changed")
    _require(kitty.get("must_report_qwen3_4032_failure_alongside_any_llama_result") is True and kitty.get("cannot_reopen_qwen3_promotion") is True, "Kitty reporting boundary changed")
    stages = protocol.get("staged_execution", {})
    _require(stages.get("stage_0_static_preflight", {}).get("authorized") is True, "static preflight authorization changed")
    _require(stages.get("stage_0_static_preflight", {}).get("gpu") is False, "static preflight cannot use GPU")
    _require(stages.get("stage_2_gpu_acceptance", {}).get("full_transfer_execution_authorized") is False, "full transfer was prematurely authorized")
    _require(stages.get("stage_3_full_transfer", {}).get("current_authorized") is False, "full transfer was prematurely authorized")
    boundary = protocol.get("reporting_and_boundaries", {})
    for key in (
        "additional_v3_tuning_after_any_llama_metric",
        "pg19_test_access",
        "kitty_llama_port_currently_authorized",
        "llama2_full_gpu_execution_currently_authorized",
        "paper_main_method_change_currently_authorized",
        "runtime_latency_throughput_or_real_cuda_memory_claim",
    ):
        _require(boundary.get(key) is False, f"transfer boundary changed: {key}")
    references = (
        (outcome["protocol_path"], outcome["protocol_sha256"]),
        (outcome["results_receipt_path"], outcome["results_receipt_sha256"]),
        (source["qwen3_quota_plan_path"], source["qwen3_quota_plan_sha256"]),
        (target["existing_paired_ppl_config_path"], target["existing_paired_ppl_config_sha256"]),
        (target["existing_acceptance_config_path"], target["existing_acceptance_config_sha256"]),
    )
    for relative, expected_sha in references:
        _require(file_sha256(repo_root / relative) == expected_sha, f"transfer frozen reference changed: {relative}")


def load_transfer_protocol(path: Path, *, repo_root: Path) -> tuple[dict[str, Any], str]:
    protocol = _load(path)
    validate_transfer_protocol(protocol, repo_root=repo_root)
    digest = file_sha256(path)
    _require(digest == EXPECTED_PROTOCOL_SHA256, "transfer protocol file hash mismatch")
    return protocol, digest


def derive_llama2_quota_plan(
    *, protocol: Mapping[str, Any], qwen_plan: Mapping[str, Any]
) -> dict[str, Any]:
    _require(qwen_plan.get("plan_id") == "qwen3-8b-cage-v3-calibrated-layer-quota-plan-v1", "Qwen quota plan identity mismatch")
    selected = {}
    for plan in qwen_plan.get("plans", []):
        if plan.get("sink_length") == 32 and plan.get("prompt_length") in PROMPT_LENGTHS:
            selected[int(plan["prompt_length"])] = plan
    _require(tuple(sorted(selected)) == PROMPT_LENGTHS, "Qwen sink32 quota plans are incomplete")
    derived = []
    mapping = protocol["architecture_normalization"]["layer_score_transfer"]
    _require(mapping.get("source_layers") == 36 and mapping.get("target_layers") == 32, "layer mapping shape changed")
    for length in PROMPT_LENGTHS:
        source = selected[length]
        scores = source["layer_scores"]
        _require(len(scores) == 36 and all(math.isfinite(float(value)) for value in scores), "Qwen layer scores are invalid")
        source_indices = [min(35, math.floor(36 * ((layer + 0.5) / 32))) for layer in range(32)]
        mapped_scores = [float(scores[index]) for index in source_indices]
        ranked = sorted(range(32), key=lambda layer: (-mapped_scores[layer], layer))
        quotas = [16] * 32
        for layer in ranked[:11]:
            quotas[layer] = 48
        for layer in ranked[11:21]:
            quotas[layer] = 32
        counts = Counter(quotas)
        _require(counts == Counter({16: 11, 32: 10, 48: 11}), "derived Llama quota counts mismatch")
        _require(sum(quotas) == 1024, "derived Llama quota total mismatch")
        derived.append(
            {
                "prompt_length": length,
                "residual_length": protocol["source_candidate"]["qwen3_residual_lengths"][str(length)],
                "sink_length": 32,
                "source_layer_indices": source_indices,
                "mapped_qwen_layer_scores": mapped_scores,
                "ranked_target_layer_indices": ranked,
                "layer_two_bit_channel_quotas": quotas,
                "quota_counts": {"16": 11, "32": 10, "48": 11},
                "quota_total": 1024,
            }
        )
    return {
        "schema_version": 1,
        "plan_id": "llama2-7b-cage-v3-depth-normalized-quota-plan-v1",
        "status": "derived_without_llama2_candidate_metrics",
        "claim_eligible": False,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "qwen3_quota_plan_sha256": EXPECTED_QWEN_PLAN_SHA256,
        "source_layers": 36,
        "target_layers": 32,
        "target_head_dim": 128,
        "target_quota_mean_per_layer_per_kv_head": 32,
        "plans": derived,
        "llama2_metrics_consumed": False,
    }


def derive_packed_memory_preflight(
    *, protocol: Mapping[str, Any], quota_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze candidate bytes and byte-only baseline choices without quality data."""

    _require(quota_plan.get("llama2_metrics_consumed") is False, "quota plan consumed Llama metrics")
    plans = {int(row["prompt_length"]): row for row in quota_plan.get("plans", [])}
    _require(tuple(sorted(plans)) == PROMPT_LENGTHS, "derived Llama quota plans are incomplete")
    frozen = protocol["frozen_packed_memory_preflight"]
    shape = frozen["model_shape"]
    v1_grid = frozen["cage_v1_grid"]
    kivi_grid = frozen["kivi_grid"]
    points = []
    for length in PROMPT_LENGTHS:
        plan = plans[length]
        candidate_report = estimate_qwen3_cage_v2_bytes(
            seq_len=length,
            residual_length=int(plan["residual_length"]),
            one_bit_channels=0,
            two_bit_channels=plan["layer_two_bit_channel_quotas"],
            sink_length=int(plan["sink_length"]),
            key_base_group_size=128,
            key_refinement_group_size=128,
            value_group_size=128,
            **shape,
            bytes_per_channel_index=1,
        )
        candidate_bytes = int(candidate_report["model_total_bytes"])
        cage_options = []
        for residual in range(
            int(v1_grid["residual_length_min"]),
            int(v1_grid["residual_length_max"]) + 1,
            int(v1_grid["residual_length_step"]),
        ):
            report = estimate_qwen3_cage_bytes(
                seq_len=length,
                residual_length=residual,
                key_bucket_sizes=v1_grid["key_bucket_sizes"],
                value_bucket_sizes=v1_grid["value_bucket_sizes"],
                key_group_sizes=v1_grid["key_group_sizes"],
                value_group_sizes=v1_grid["value_group_sizes"],
                bits=int(v1_grid["bits"]),
                bytes_per_bucket_index=int(v1_grid["bytes_per_bucket_index"]),
                **shape,
            )
            cage_options.append(
                {
                    "point_id": f"cage-r{residual}",
                    "residual_length": residual,
                    "packed_bytes": int(report["model_total_bytes"]),
                }
            )
        kivi_options = []
        for group_size in kivi_grid["group_sizes"]:
            for residual in range(int(group_size), 513, int(group_size)):
                report = estimate_qwen3_kivi_bytes(
                    seq_len=length,
                    group_size=int(group_size),
                    residual_length=residual,
                    sink_length=int(kivi_grid["sink_length"]),
                    bits=int(kivi_grid["bits"]),
                    **shape,
                )
                kivi_options.append(
                    {
                        "point_id": f"kivi-g{group_size}-r{residual}",
                        "group_size": int(group_size),
                        "residual_length": residual,
                        "packed_bytes": int(report["model_total_bytes"]),
                    }
                )

        def selection_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
            return (
                abs(int(row["packed_bytes"]) - candidate_bytes),
                int(row["packed_bytes"]),
                str(row["point_id"]),
            )

        cage = min(cage_options, key=selection_key)
        kivi = min(kivi_options, key=selection_key)
        cage_delta = (int(cage["packed_bytes"]) - candidate_bytes) / candidate_bytes
        kivi_delta = (int(kivi["packed_bytes"]) - candidate_bytes) / candidate_bytes
        _require(abs(cage_delta) <= 0.0025, f"CAGE-v1 byte match exceeds tolerance at {length}")
        _require(kivi_delta >= 0.0, f"closest KIVI gives candidate an unfair larger-memory budget at {length}")
        points.append(
            {
                "prompt_length": length,
                "candidate": {
                    "point_id": f"cage-v3-llama2-depth-normalized-r{plan['residual_length']}",
                    "residual_length": int(plan["residual_length"]),
                    "sink_length": int(plan["sink_length"]),
                    "packed_bytes": candidate_bytes,
                },
                "cage_v1_closest": {
                    **cage,
                    "relative_byte_delta_vs_candidate": cage_delta,
                    "within_frozen_tolerance": True,
                },
                "kivi_closest_feasible": {
                    **kivi,
                    "relative_byte_delta_vs_candidate": kivi_delta,
                    "candidate_uses_no_more_bytes": True,
                    "equal_memory_claim_allowed": False,
                },
            }
        )
    return {
        "schema_version": 1,
        "preflight_id": "llama2-7b-cage-v3-packed-memory-preflight-v1",
        "status": "selected_without_llama2_quality_metrics",
        "claim_eligible": False,
        "representation": frozen["representation"],
        "selection_rule": frozen["selection_rule"],
        "quality_metrics_consumed": False,
        "points": points,
    }


def audit_llama2_model_metadata(model_path: Path) -> dict[str, Any]:
    root = model_path.resolve()
    config_path = root / "config.json"
    tokenizer_config_path = root / "tokenizer_config.json"
    _require(config_path.is_file(), "Llama config.json is missing")
    _require(tokenizer_config_path.is_file(), "Llama tokenizer_config.json is missing")
    config = _load(config_path)
    hidden_size = config.get("hidden_size")
    attention_heads = config.get("num_attention_heads")
    _require(type(hidden_size) is int and type(attention_heads) is int and attention_heads > 0, "Llama hidden/head metadata is invalid")
    head_dim = config.get("head_dim") or hidden_size // attention_heads
    checks = {
        "model_type": config.get("model_type") == "llama",
        "num_hidden_layers": config.get("num_hidden_layers") == 32,
        "num_attention_heads": config.get("num_attention_heads") == 32,
        "num_key_value_heads": config.get("num_key_value_heads", config.get("num_attention_heads")) == 32,
        "head_dim": head_dim == 128,
        "max_position_embeddings": config.get("max_position_embeddings") == 4096,
    }
    _require(all(checks.values()), "Llama architecture preflight failed")
    metadata_names = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    metadata = {
        name: {"sha256": file_sha256(root / name), "size_bytes": (root / name).stat().st_size}
        for name in metadata_names
        if (root / name).is_file()
    }
    _require("config.json" in metadata and "tokenizer_config.json" in metadata, "Llama metadata receipt is incomplete")
    _require("tokenizer.json" in metadata or "tokenizer.model" in metadata, "Llama tokenizer payload is missing")
    index_name = next(
        (name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json") if (root / name).is_file()),
        None,
    )
    weight_files = []
    if index_name is not None:
        index = _load(root / index_name)
        filenames = sorted(set(index.get("weight_map", {}).values()))
        _require(bool(filenames), "Llama weight index is empty")
        for name in filenames:
            path = root / name
            _require(path.is_file(), f"Llama weight shard is missing: {name}")
            weight_files.append(
                {"name": name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            )
    else:
        paths = sorted((*root.glob("*.safetensors"), *root.glob("pytorch_model*.bin")))
        _require(bool(paths), "Llama weight files are missing")
        weight_files = [
            {"name": path.name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in paths
        ]
    return {
        "reference": str(root),
        "architecture": {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads", config.get("num_attention_heads")),
            "head_dim": head_dim,
            "max_position_embeddings": config.get("max_position_embeddings"),
            "torch_dtype": config.get("torch_dtype"),
        },
        "checks": checks,
        "metadata": metadata,
        "weight_files": weight_files,
        "weight_file_count": len(weight_files),
        "weight_total_size_bytes": sum(row["size_bytes"] for row in weight_files),
    }


__all__ = [
    "EXPECTED_PROTOCOL_SHA256",
    "Llama2CageV3TransferProtocolError",
    "PROTOCOL_ID",
    "audit_llama2_model_metadata",
    "derive_packed_memory_preflight",
    "derive_llama2_quota_plan",
    "load_transfer_protocol",
    "validate_transfer_protocol",
]
