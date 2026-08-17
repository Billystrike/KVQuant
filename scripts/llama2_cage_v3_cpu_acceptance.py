#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoConfig, LlamaConfig

from models.llama_cage_v3 import (
    LlamaCageV3Config,
    LlamaCageV3Plan,
    append_decode_token,
    build_prefill_cache,
    install_llama_cage_v3_config,
    reconstruct_key,
    reconstruct_value,
    unpack_cache,
)
from models.llama_kivi import LlamaFlashAttention_KIVI
from utils.llama2_cage_v3_cpu_acceptance import (
    load_cpu_protocol,
    validate_static_preflight_payloads,
)
from utils.qwen3_cage_v4_data import file_sha256


def _serialize(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen CPU-acceptance output: {path}")
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(_serialize(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tiny_policy(fixture: dict) -> LlamaCageV3Config:
    return LlamaCageV3Config(
        plans=(
            LlamaCageV3Plan(
                prompt_length=fixture["prompt_length"],
                residual_length=fixture["residual_length"],
                sink_length=fixture["sink_length"],
                layer_two_bit_channel_quotas=(fixture["two_bit_channels"],),
            ),
        ),
        key_base_group_size=fixture["group_size"],
        key_refinement_group_size=fixture["group_size"],
        value_group_size=fixture["group_size"],
    )


def _tiny_model_config(fixture: dict) -> LlamaConfig:
    hidden_size = fixture["num_attention_heads"] * fixture["head_dim"]
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_attention_heads=fixture["num_attention_heads"],
        num_key_value_heads=fixture["num_key_value_heads"],
        num_hidden_layers=fixture["num_hidden_layers"],
        max_position_embeddings=64,
        vocab_size=64,
        attention_dropout=0.0,
    )
    config.use_flash = True
    config.k_bits = 2
    config.v_bits = 2
    config.group_size = fixture["group_size"]
    config.residual_length = fixture["residual_length"]
    config.cage_enable = False
    config.cage_v3_enable = True
    config.cage_v3_plans = [
        {
            "prompt_length": fixture["prompt_length"],
            "residual_length": fixture["residual_length"],
            "sink_length": fixture["sink_length"],
            "layer_two_bit_channel_quotas": [fixture["two_bit_channels"]],
        }
    ]
    config.cage_v3_key_base_group_size = fixture["group_size"]
    config.cage_v3_key_refinement_group_size = fixture["group_size"]
    config.cage_v3_value_group_size = fixture["group_size"]
    return config


def _production_config_checks(model_path: Path, quota: dict) -> tuple[dict, dict]:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    installed = install_llama_cage_v3_config(config, quota)
    architecture = {
        "class": f"{config.__class__.__module__}.{config.__class__.__name__}",
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.hidden_size // config.num_attention_heads,
        "max_position_embeddings": config.max_position_embeddings,
    }
    rejection = AutoConfig.from_pretrained(model_path, local_files_only=True)
    rejection.num_hidden_layers -= 1
    rejected = False
    try:
        install_llama_cage_v3_config(rejection, quota)
    except ValueError:
        rejected = True
    checks = {
        "production_architecture_exact": architecture
        == {
            "class": "transformers.models.llama.configuration_llama.LlamaConfig",
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 128,
            "max_position_embeddings": 4096,
        },
        "production_quota_installation_exact": (
            installed.num_hidden_layers == 32
            and [plan.prompt_length for plan in installed.plans] == [1024, 2048, 4032]
            and all(sum(plan.layer_two_bit_channel_quotas) == 1024 for plan in installed.plans)
            and config.cage_v3_enable is True
            and config.cage_enable is False
        ),
        "wrong_architecture_rejected": rejected,
    }
    return architecture, checks


def _core_cache_checks(fixture: dict) -> tuple[dict, dict]:
    policy = _tiny_policy(fixture)
    batch = fixture["batch_size"]
    prompt = fixture["prompt_length"]
    query = torch.randn(batch, fixture["num_attention_heads"], prompt, fixture["head_dim"])
    key = torch.randn(batch, fixture["num_key_value_heads"], prompt, fixture["head_dim"])
    value = torch.randn_like(key)
    cache = build_prefill_cache(
        policy,
        query_states=query,
        key_states=key,
        value_states=value,
        layer_idx=0,
    )
    sink = fixture["sink_length"]
    initial_key_quantized = cache.key_quantized.clone()
    initial_value_quantized = cache.value_quantized.clone()
    indices = cache.two_bit_indices.clone()
    prefill = {
        "two_bit_index_shape": list(cache.two_bit_indices.shape),
        "key_quantized_length": cache.key_quantized.shape[-2],
        "key_residual_length": cache.key_residual.shape[-2],
        "value_quantized_length": cache.value_quantized.shape[-2],
        "value_residual_length": cache.value_residual.shape[-2],
        "kv_seq_len": cache.kv_seq_len,
    }
    checks = {
        "persistent_prompt_adaptive_sparse_key_indices": tuple(cache.two_bit_indices.shape)
        == (fixture["num_key_value_heads"], fixture["two_bit_channels"]),
        "all_channel_int2_key_backbone_with_sparse_refinement": (
            cache.key_quantized is not None
            and not torch.equal(cache.key_quantized, key[:, :, sink : sink + cache.key_quantized.shape[-2], :])
        ),
        "uniform_int2_value_without_adaptive_state": (
            cache.value_quantized is not None
            and not hasattr(cache, "value_bucket_indices")
            and not torch.equal(cache.value_quantized, value[:, :, sink : sink + cache.value_quantized.shape[-2], :])
        ),
        "fp16_sink_preserved": (
            torch.equal(cache.key_sink, key[:, :, :sink, :])
            and torch.equal(cache.value_sink, value[:, :, :sink, :])
        ),
    }
    for _ in range(fixture["continuation_tokens"]):
        cache = append_decode_token(
            policy,
            cache,
            key_states=torch.randn(batch, fixture["num_key_value_heads"], 1, fixture["head_dim"]),
            value_states=torch.randn(batch, fixture["num_key_value_heads"], 1, fixture["head_dim"]),
        )
    reconstructed_key = reconstruct_key(cache)
    reconstructed_value = reconstruct_value(cache)
    checks.update(
        {
            "single_token_continuation_and_key_flush": (
                cache.key_residual is None
                and cache.key_quantized.shape[-2] == 8
                and cache.kv_seq_len == prompt + fixture["continuation_tokens"]
            ),
            "rolling_value_residual": (
                cache.value_quantized.shape[-2] == 4 and cache.value_residual.shape[-2] == fixture["residual_length"]
            ),
            "old_quantized_prefix_not_requantized": (
                torch.equal(cache.key_quantized[:, :, : initial_key_quantized.shape[-2], :], initial_key_quantized)
                and torch.equal(cache.value_quantized[:, :, : initial_value_quantized.shape[-2], :], initial_value_quantized)
            ),
            "sparse_indices_persist_across_decode": torch.equal(cache.two_bit_indices, indices),
            "reconstructed_cache_finite": (
                bool(torch.isfinite(reconstructed_key).all())
                and bool(torch.isfinite(reconstructed_value).all())
                and reconstructed_key.shape[-2] == cache.kv_seq_len
                and reconstructed_value.shape[-2] == cache.kv_seq_len
            ),
        }
    )
    continuation = {
        "key_quantized_length": cache.key_quantized.shape[-2],
        "key_residual_length": 0 if cache.key_residual is None else cache.key_residual.shape[-2],
        "value_quantized_length": cache.value_quantized.shape[-2],
        "value_residual_length": cache.value_residual.shape[-2],
        "kv_seq_len": cache.kv_seq_len,
    }
    return {"prefill": prefill, "continuation": continuation}, checks


def _attention_checks(fixture: dict) -> tuple[dict, dict]:
    attention = LlamaFlashAttention_KIVI(_tiny_model_config(fixture), layer_idx=0).eval()
    hidden_size = fixture["num_attention_heads"] * fixture["head_dim"]
    hidden = torch.randn(fixture["batch_size"], fixture["prompt_length"], hidden_size)
    positions = torch.arange(fixture["prompt_length"]).unsqueeze(0)
    mask = torch.zeros(1, 1, fixture["prompt_length"], fixture["prompt_length"])
    with torch.inference_mode():
        reference, _, _ = attention(hidden, attention_mask=mask, position_ids=positions, use_cache=False)
        actual, _, packed = attention(hidden, attention_mask=mask, position_ids=positions, use_cache=True)
        decode, _, updated = attention(
            torch.randn(1, 1, hidden_size),
            attention_mask=torch.zeros(1, 1, 1, fixture["prompt_length"] + 1),
            position_ids=torch.tensor([[fixture["prompt_length"]]]),
            past_key_value=packed,
            use_cache=True,
        )
    checks = {
        "prefill_attention_exact": torch.equal(reference, actual),
        "attention_output_finite": bool(torch.isfinite(decode).all()),
        "attention_cache_length_exact": unpack_cache(updated).kv_seq_len == fixture["prompt_length"] + 1,
    }
    details = {
        "prefill_max_abs_delta": float((reference - actual).abs().max().item()),
        "decode_output_shape": list(decode.shape),
        "decode_output_dtype": str(decode.dtype),
        "updated_cache_length": unpack_cache(updated).kv_seq_len,
    }
    return details, checks


def _fail_closed_checks(fixture: dict) -> dict:
    policy = _tiny_policy(fixture)
    prompt_rejected = batch_rejected = multitoken_rejected = False
    try:
        build_prefill_cache(
            policy,
            query_states=torch.randn(1, fixture["num_attention_heads"], fixture["prompt_length"] - 1, fixture["head_dim"]),
            key_states=torch.randn(1, fixture["num_key_value_heads"], fixture["prompt_length"] - 1, fixture["head_dim"]),
            value_states=torch.randn(1, fixture["num_key_value_heads"], fixture["prompt_length"] - 1, fixture["head_dim"]),
            layer_idx=0,
        )
    except ValueError:
        prompt_rejected = True
    try:
        build_prefill_cache(
            policy,
            query_states=torch.randn(2, fixture["num_attention_heads"], fixture["prompt_length"], fixture["head_dim"]),
            key_states=torch.randn(2, fixture["num_key_value_heads"], fixture["prompt_length"], fixture["head_dim"]),
            value_states=torch.randn(2, fixture["num_key_value_heads"], fixture["prompt_length"], fixture["head_dim"]),
            layer_idx=0,
        )
    except ValueError:
        batch_rejected = True
    cache = build_prefill_cache(
        policy,
        query_states=torch.randn(1, fixture["num_attention_heads"], fixture["prompt_length"], fixture["head_dim"]),
        key_states=torch.randn(1, fixture["num_key_value_heads"], fixture["prompt_length"], fixture["head_dim"]),
        value_states=torch.randn(1, fixture["num_key_value_heads"], fixture["prompt_length"], fixture["head_dim"]),
        layer_idx=0,
    )
    try:
        append_decode_token(
            policy,
            cache,
            key_states=torch.randn(1, fixture["num_key_value_heads"], 2, fixture["head_dim"]),
            value_states=torch.randn(1, fixture["num_key_value_heads"], 2, fixture["head_dim"]),
        )
    except ValueError:
        multitoken_rejected = True
    return {
        "unsupported_prompt_length_rejected": prompt_rejected,
        "unsupported_batch_size_rejected": batch_rejected,
        "multitoken_continuation_rejected": multitoken_rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU-only Llama-2 CAGE-v3 implementation acceptance")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    protocol, protocol_sha256 = load_cpu_protocol(protocol_path, repo_root=REPO_ROOT)
    quota, preflight = validate_static_preflight_payloads(protocol, repo_root=REPO_ROOT)
    torch.manual_seed(protocol["cpu_fixture"]["seed"])

    architecture, production_checks = _production_config_checks(model_path, quota)
    cache_details, cache_checks = _core_cache_checks(protocol["cpu_fixture"])
    attention_details, attention_checks = _attention_checks(protocol["cpu_fixture"])
    fail_closed_checks = _fail_closed_checks(protocol["cpu_fixture"])
    checks = {**production_checks, **cache_checks, **attention_checks, **fail_closed_checks}
    failures = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "schema_version": 1,
        "acceptance_id": "llama2-7b-cage-v3-cpu-implementation-acceptance-v1",
        "status": "pass" if not failures else "fail",
        "claim_eligible": False,
        "failures": failures,
        "checks": checks,
        "protocol_sha256": protocol_sha256,
        "static_preflight_receipt_sha256": protocol["static_preflight_artifacts"]["preflight"]["server_sha256"],
        "quota_plan_sha256": protocol["static_preflight_artifacts"]["quota"]["server_sha256"],
        "qwen3_promotion_pass": False,
        "qwen3_conclusion_reopened": False,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_available_observed_only": bool(torch.cuda.is_available()),
        "device_used": "cpu",
        "production_model": {
            "reference": str(model_path),
            "config_sha256": file_sha256(model_path / "config.json"),
            "architecture": architecture,
            "full_model_weights_loaded": False,
        },
        "fixture": protocol["cpu_fixture"],
        "cache_details": cache_details,
        "attention_details": attention_details,
        "source_sha256": {
            "runner": file_sha256(Path(__file__).resolve()),
            **{name: file_sha256(REPO_ROOT / spec["path"]) for name, spec in protocol["frozen_sources"].items()},
        },
        "execution_boundary": protocol["execution_boundary"],
        "static_preflight_checks_preserved": all(preflight["checks"].values()),
        "next_step": "freeze this exact passing CPU receipt before defining a separate GPU acceptance gate",
    }
    _write_atomic(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if failures:
        raise RuntimeError(f"Llama-2 CAGE-v3 CPU acceptance failed: {failures}")


if __name__ == "__main__":
    main()
