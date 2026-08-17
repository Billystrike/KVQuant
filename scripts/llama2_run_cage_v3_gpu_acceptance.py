#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoConfig

from models.llama_cage_v3 import install_llama_cage_v3_config, unpack_cache
from models.llama_kivi import LlamaForCausalLM_KIVI
from utils.llama2_cage_v3_gpu_acceptance_protocol import load_protocol, synthetic_token_ids
from utils.llama2_cage_v3_gpu_execution import (
    full_file_sha256,
    load_object,
    require,
    tensor_sha256,
    validate_gate_receipt,
    validate_static_preflight_receipt_file,
)
from utils.qwen3_cage_v4_data import file_sha256


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite GPU acceptance output: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_records(past_key_values: tuple, expected_quotas: list[int]) -> tuple[list[dict], list[dict]]:
    require(len(past_key_values) == 32, "GPU cache must contain 32 layers")
    records = []
    probes = []
    for layer_idx, (packed, quota) in enumerate(zip(past_key_values, expected_quotas)):
        cache = unpack_cache(packed)
        key_probe = cache.key_quantized[:, :, :2, :8]
        value_probe = cache.value_quantized[:, :, :2, :8]
        records.append(
            {
                "layer_idx": layer_idx,
                "quota": quota,
                "two_bit_indices_shape": list(cache.two_bit_indices.shape),
                "two_bit_indices_sha256": tensor_sha256(cache.two_bit_indices),
                "key_sink_shape": list(cache.key_sink.shape),
                "key_quantized_shape": list(cache.key_quantized.shape),
                "key_residual_shape": list(cache.key_residual.shape),
                "value_sink_shape": list(cache.value_sink.shape),
                "value_quantized_shape": list(cache.value_quantized.shape),
                "value_residual_shape": list(cache.value_residual.shape),
                "kv_seq_len": cache.kv_seq_len,
                "key_probe_sha256": tensor_sha256(key_probe),
                "value_probe_sha256": tensor_sha256(value_probe),
            }
        )
        probes.append(
            {
                "indices": cache.two_bit_indices.detach().clone(),
                "key": key_probe.detach().clone(),
                "value": value_probe.detach().clone(),
            }
        )
    return records, probes


def _validate_updated_cache(past_key_values: tuple, probes: list[dict], expected_length: int) -> bool:
    require(len(past_key_values) == len(probes), "updated cache layer count changed")
    unchanged = True
    for packed, probe in zip(past_key_values, probes):
        cache = unpack_cache(packed)
        require(cache.kv_seq_len == expected_length, "updated cache sequence length mismatch")
        unchanged = unchanged and torch.equal(cache.two_bit_indices, probe["indices"])
        unchanged = unchanged and torch.equal(cache.key_quantized[:, :, :2, :8], probe["key"])
        unchanged = unchanged and torch.equal(cache.value_quantized[:, :, :2, :8], probe["value"])
    return bool(unchanged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen Llama-2 CAGE-v3 production GPU acceptance repeat")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--gate-receipt", type=Path, required=True)
    parser.add_argument("--repeat", choices=("a", "b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol, protocol_sha256 = load_protocol(args.protocol.resolve(), repo_root=REPO_ROOT)
    validate_static_preflight_receipt_file(args.preflight_receipt.resolve())
    gate = load_object(args.gate_receipt.resolve())
    validate_gate_receipt(gate, repo_root=REPO_ROOT)
    model_spec = protocol["production_model"]
    model_path = Path(model_spec["reference"])
    weight_receipts = []
    for spec in model_spec["weight_files"]:
        path = model_path / spec["name"]
        digest = full_file_sha256(path)
        require(digest == spec["sha256"], f"weight hash mismatch: {path}")
        weight_receipts.append({"name": spec["name"], "sha256": digest, "size_bytes": path.stat().st_size})

    torch.manual_seed(20260817)
    torch.cuda.manual_seed_all(20260817)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    quota = load_object(REPO_ROOT / protocol["candidate"]["quota_plan_path"])
    installed = install_llama_cage_v3_config(config, quota)
    config.k_bits = 2
    config.v_bits = 2
    config.group_size = 128
    config.residual_length = 176
    config.use_flash = True
    config._flash_attn_2_enabled = False
    config.use_cache = True

    load_started = time.perf_counter()
    model = LlamaForCausalLM_KIVI.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda:0").eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    require(parameter_devices == ["cuda:0"], "model parameters are not exclusively on cuda:0")
    require(parameter_dtypes == ["torch.float16"], "model parameters are not exclusively float16")
    require(len(model.model.layers) == 32, "loaded model layer count mismatch")
    require(all(layer.self_attn.cage_v3_config is not None for layer in model.model.layers), "CAGE-v3 was not installed in every layer")

    ids = synthetic_token_ids()
    prompt = torch.tensor([ids[:1024]], dtype=torch.long, device="cuda:0")
    continuation = torch.tensor([ids[1024:]], dtype=torch.long, device="cuda:0")
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    with torch.inference_mode():
        prefill_output = model(input_ids=prompt, use_cache=True, return_dict=True)
    require(bool(torch.isfinite(prefill_output.logits[:, -1, :]).all()), "prefill logits are non-finite")
    prefill_logits_sha256 = tensor_sha256(prefill_output.logits[:, -1, :])
    prefill_past = prefill_output.past_key_values
    del prefill_output
    expected_plan = installed.plan_for_prompt(1024)
    expected_quotas = list(expected_plan.layer_two_bit_channel_quotas)
    cache_records, probes = _cache_records(prefill_past, expected_quotas)

    with torch.inference_mode():
        first = model(input_ids=continuation[:, :1], past_key_values=prefill_past, use_cache=True, return_dict=True)
    require(bool(torch.isfinite(first.logits[:, -1, :]).all()), "first continuation logits are non-finite")
    first_logits_sha256 = tensor_sha256(first.logits[:, -1, :])
    first_past = first.past_key_values
    first_unchanged = _validate_updated_cache(first_past, probes, 1025)
    del first
    with torch.inference_mode():
        second = model(input_ids=continuation[:, 1:], past_key_values=first_past, use_cache=True, return_dict=True)
    require(bool(torch.isfinite(second.logits[:, -1, :]).all()), "second continuation logits are non-finite")
    second_logits_sha256 = tensor_sha256(second.logits[:, -1, :])
    second_past = second.past_key_values
    second_unchanged = _validate_updated_cache(second_past, probes, 1026)
    del second

    multitoken_rejected = False
    try:
        with torch.inference_mode():
            model(input_ids=continuation, past_key_values=prefill_past, use_cache=True, return_dict=True)
    except ValueError:
        multitoken_rejected = True
    torch.cuda.synchronize()
    run_seconds = time.perf_counter() - run_started
    peak_bytes = int(torch.cuda.max_memory_allocated())
    quota_counts = {str(value): expected_quotas.count(value) for value in sorted(set(expected_quotas))}
    checks = {
        "model_loaded_on_cuda_float16": parameter_devices == ["cuda:0"] and parameter_dtypes == ["torch.float16"],
        "all_layers_cage_v3_installed": len(cache_records) == 32,
        "prefill_logits_finite": True,
        "cache_materialized_all_layers": all(
            row["key_sink_shape"][2] == 32
            and row["key_quantized_shape"][2] == 880
            and row["key_residual_shape"][2] == 112
            and row["value_quantized_shape"][2] == 816
            and row["value_residual_shape"][2] == 176
            for row in cache_records
        ),
        "frozen_quota_counts": quota_counts == {"16": 11, "32": 10, "48": 11},
        "first_decode_cache_length": all(unpack_cache(item).kv_seq_len == 1025 for item in first_past),
        "second_decode_cache_length": all(unpack_cache(item).kv_seq_len == 1026 for item in second_past),
        "old_prefix_and_indices_unchanged": first_unchanged and second_unchanged,
        "multitoken_continuation_rejected": multitoken_rejected,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    scientific = {
        "method_id": protocol["candidate"]["method_id"],
        "synthetic_input_sha256": protocol["synthetic_input"]["all_ids_sha256"],
        "prompt_length": 1024,
        "continuation_tokens": 2,
        "checks": checks,
        "failures": failures,
        "parameter_devices": parameter_devices,
        "parameter_dtypes": parameter_dtypes,
        "quota_counts": quota_counts,
        "prefill_logits_sha256": prefill_logits_sha256,
        "first_decode_logits_sha256": first_logits_sha256,
        "second_decode_logits_sha256": second_logits_sha256,
        "cache_records": cache_records,
        "reported_cache_lengths": [1024, 1025, 1026],
    }
    payload = {
        "schema_version": 1,
        "acceptance_id": "llama2-7b-cage-v3-production-gpu-acceptance-v1",
        "status": "pass" if not failures else "fail",
        "claim_eligible": False,
        "repeat": args.repeat,
        "protocol_sha256": protocol_sha256,
        "preflight_receipt_sha256": file_sha256(args.preflight_receipt.resolve()),
        "gate_receipt_sha256": file_sha256(args.gate_receipt.resolve()),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(0),
        "weight_receipts": weight_receipts,
        "scientific_payload": scientific,
        "telemetry": {
            "model_load_seconds": load_seconds,
            "acceptance_run_seconds": run_seconds,
            "peak_cuda_allocator_bytes": peak_bytes,
        },
        "execution_boundary": {
            "synthetic_acceptance_only": True,
            "corpus_accessed": False,
            "quality_metric_computed": False,
            "formal_transfer_authorized": False,
            "runtime_claims_authorized": False,
            "real_packed_cuda_cache_allocated": False,
        },
    }
    _write_atomic(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if failures:
        raise RuntimeError(f"GPU acceptance failed: {failures}")


if __name__ == "__main__":
    main()
