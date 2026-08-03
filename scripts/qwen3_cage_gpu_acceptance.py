#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import cache_utils as transformers_cache_utils
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3 import modeling_qwen3

from models.cage_config import CageConfig
from models.qwen3_cage import Qwen3CageCache, install_qwen3_cage_attention


EXPECTED = {
    "python": "3.10.20",
    "torch": "2.4.1+cu121",
    "transformers": "4.53.2",
    "transformers_cache_utils_sha256": "529e858b9bb0ba59a830713bbd5ab594e29f262bbad1bba860ff7de3248e5a5a",
    "transformers_qwen3_modeling_sha256": "c12b4c4a34a06e887f3df399b1b09e9966c12d2a19769c258b7379066e585ea4",
    "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "generation_config_sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "model_index_sha256": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "parameter_count": 8190735360,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "first_answer_token_id": 20,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-8B CAGE GPU acceptance")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, default=320)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--residual-length", type=int, default=128)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _build_prompt(tokenizer, target_length: int) -> tuple[str, torch.Tensor, int]:
    if target_length <= 0:
        raise ValueError("prompt length must be positive")
    question = "Ignore the neutral context above. What is 2 plus 3? Reply with only the digit."
    for filler_count in range(target_length + 1):
        content = " context" * filler_count + "\n" + question
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids
        token_count = int(input_ids.shape[1])
        if token_count == target_length:
            return rendered, input_ids, filler_count
        if token_count > target_length:
            break
    raise ValueError(f"could not construct an exact {target_length}-token prompt")


def _cage_config() -> CageConfig:
    return CageConfig(
        cage_enable=True,
        cage_mode="fake",
        cage_k_enable=True,
        cage_v_enable=True,
        cage_k_importance="q2_var",
        cage_k_group_sizes=[32, 64, 128],
        cage_k_clip_percentiles=[0.999, 0.995, 0.99],
        cage_k_num_buckets=3,
        cage_v_importance="wo_var",
        cage_v_group_sizes=[32, 64, 128],
        cage_v_clip_percentiles=[0.999, 0.995, 0.99],
        cage_v_num_buckets=3,
    )


def main() -> None:
    args = _parse_args()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-8B GPU acceptance")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")

    torch.manual_seed(20260803)
    torch.cuda.manual_seed_all(20260803)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total_bytes = torch.cuda.mem_get_info()

    metadata_hashes = {
        "config_sha256": _sha256(model_path / "config.json"),
        "generation_config_sha256": _sha256(model_path / "generation_config.json"),
        "tokenizer_config_sha256": _sha256(model_path / "tokenizer_config.json"),
        "model_index_sha256": _sha256(model_path / "model.safetensors.index.json"),
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    rendered_prompt, input_ids_cpu, filler_count = _build_prompt(tokenizer, args.prompt_length)
    attention_mask_cpu = torch.ones_like(input_ids_cpu)

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()
    load_seconds = time.perf_counter() - load_started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})

    input_ids = input_ids_cpu.to("cuda:0")
    attention_mask = attention_mask_cpu.to("cuda:0")
    with torch.inference_mode():
        reference_cache = DynamicCache()
        reference = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=reference_cache,
            use_cache=True,
            logits_to_keep=1,
        )
        reference_last_logits = reference.logits[:, -1, :].detach().cpu()
    del reference, reference_cache
    torch.cuda.empty_cache()

    installed_attention_modules = install_qwen3_cage_attention(model)
    with torch.inference_mode():
        prefill_cache = Qwen3CageCache(
            _cage_config(),
            residual_length=args.residual_length,
        )
        candidate = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=prefill_cache,
            use_cache=True,
            logits_to_keep=1,
        )
        candidate_last_logits = candidate.logits[:, -1, :].detach().cpu()
    prefill_logits_exact = torch.equal(candidate_last_logits, reference_last_logits)
    prefill_logits_max_abs_delta = float(
        (candidate_last_logits - reference_last_logits).abs().max().item()
    )
    prefill_key_quantized_lengths = list(prefill_cache.key_quantized_lengths)
    prefill_value_quantized_lengths = list(prefill_cache.value_quantized_lengths)
    del candidate, prefill_cache, candidate_last_logits, reference_last_logits
    torch.cuda.empty_cache()

    generation_cache = Qwen3CageCache(
        _cage_config(),
        residual_length=args.residual_length,
    )
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=generation_cache,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    generation_seconds = time.perf_counter() - generation_started
    generated_ids = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
    generated_text_raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
    generated_text_clean = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    cache_length_after_generate = generation_cache.get_seq_length()

    resume_length_before = generation_cache.get_seq_length()
    with torch.inference_mode():
        resume = model(
            input_ids=generated[:, -1:],
            attention_mask=torch.ones(
                (1, resume_length_before + 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
            past_key_values=generation_cache,
            use_cache=True,
            logits_to_keep=1,
        )
    resume_length_after = generation_cache.get_seq_length()
    resume_logits_finite = bool(torch.isfinite(resume.logits).all().item())
    resume_next_token_id = int(resume.logits[:, -1, :].argmax(dim=-1).item())
    resume_next_token_text = tokenizer.decode([resume_next_token_id], skip_special_tokens=False)

    key_bucket_shapes = sorted(
        {
            tuple(tuple(index.shape) for index in policy.key_bucket_indices)
            for policy in generation_cache.layer_policies
        }
    )
    value_bucket_shapes = sorted(
        {
            tuple(tuple(index.shape) for index in policy.value_bucket_indices)
            for policy in generation_cache.layer_policies
        }
    )
    key_index_dtypes = sorted(
        {str(index.dtype) for policy in generation_cache.layer_policies for index in policy.key_bucket_indices}
    )
    value_index_dtypes = sorted(
        {str(index.dtype) for policy in generation_cache.layer_policies for index in policy.value_bucket_indices}
    )
    cache_tensor_dtypes = sorted(
        {str(tensor.dtype) for tensor in generation_cache.key_cache + generation_cache.value_cache}
    )
    cache_tensors_finite = all(
        bool(torch.isfinite(tensor).all().item())
        for tensor in generation_cache.key_cache + generation_cache.value_cache
    )
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    cache_utils_path = Path(transformers_cache_utils.__file__).resolve()
    qwen3_modeling_path = Path(modeling_qwen3.__file__).resolve()
    source_hashes = {
        "transformers_cache_utils_sha256": _sha256(cache_utils_path),
        "transformers_qwen3_modeling_sha256": _sha256(qwen3_modeling_path),
        "qwen3_cage_source_sha256": _sha256(REPO_ROOT / "models" / "qwen3_cage.py"),
        "qwen3_cage_gpu_acceptance_source_sha256": _sha256(Path(__file__).resolve()),
    }
    think_match = re.search(r"<think>(.*?)</think>", rendered_prompt, flags=re.DOTALL)
    expected_generate_cache_length = args.prompt_length + args.max_new_tokens - 1
    expected_resume_cache_length = args.prompt_length + args.max_new_tokens
    expected_key_quantized_length = (
        expected_resume_cache_length
        - expected_resume_cache_length % args.residual_length
    )
    expected_value_quantized_length = max(
        0,
        expected_resume_cache_length - args.residual_length,
    )

    checks = {
        "python_version": platform.python_version() == EXPECTED["python"],
        "torch_version": str(torch.__version__) == EXPECTED["torch"],
        "transformers_version": transformers.__version__ == EXPECTED["transformers"],
        "transformers_cache_utils_sha256": source_hashes["transformers_cache_utils_sha256"]
        == EXPECTED["transformers_cache_utils_sha256"],
        "transformers_qwen3_modeling_sha256": source_hashes["transformers_qwen3_modeling_sha256"]
        == EXPECTED["transformers_qwen3_modeling_sha256"],
        "model_metadata_hashes": all(metadata_hashes[name] == EXPECTED[name] for name in metadata_hashes),
        "model_class": model.__class__.__name__ == "Qwen3ForCausalLM",
        "model_module": model.__class__.__module__ == "transformers.models.qwen3.modeling_qwen3",
        "parameter_count": parameter_count == EXPECTED["parameter_count"],
        "parameter_device": parameter_devices == ["cuda:0"],
        "parameter_dtype": parameter_dtypes == ["torch.float16"],
        "attention_implementation": model.config._attn_implementation == "flash_attention_2",
        "installed_attention_modules": installed_attention_modules == EXPECTED["num_hidden_layers"],
        "prefill_logits_exact": prefill_logits_exact,
        "prefill_key_quantized_lengths": set(prefill_key_quantized_lengths)
        == {args.prompt_length - args.prompt_length % args.residual_length},
        "prefill_value_quantized_lengths": set(prefill_value_quantized_lengths)
        == {max(0, args.prompt_length - args.residual_length)},
        "generated_token_count": len(generated_ids) == args.max_new_tokens,
        "first_answer_token": bool(generated_ids) and generated_ids[0] == EXPECTED["first_answer_token_id"],
        "semantic_contains_digit_5": "5" in generated_text_clean,
        "cache_length_after_generate": cache_length_after_generate == expected_generate_cache_length,
        "resume_length_after": resume_length_after == expected_resume_cache_length,
        "resume_logits_finite": resume_logits_finite,
        "layer_policy_count": len(generation_cache.layer_policies) == EXPECTED["num_hidden_layers"],
        "key_bucket_shapes": key_bucket_shapes == [((8, 42), (8, 43), (8, 43))],
        "value_bucket_shapes": value_bucket_shapes == [((8, 42), (8, 43), (8, 43))],
        "index_dtypes": key_index_dtypes == ["torch.int64"] and value_index_dtypes == ["torch.int64"],
        "cache_tensor_dtype": cache_tensor_dtypes == ["torch.float16"],
        "cache_tensors_finite": cache_tensors_finite,
        "final_key_quantized_lengths": set(generation_cache.key_quantized_lengths)
        == {expected_key_quantized_length},
        "final_value_quantized_lengths": set(generation_cache.value_quantized_lengths)
        == {expected_value_quantized_length},
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_revision": EXPECTED["model_revision"],
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_free_before_bytes": free_before,
        "cuda_total_bytes": total_bytes,
        "model_class": model.__class__.__name__,
        "model_module": model.__class__.__module__,
        "parameter_count": parameter_count,
        "parameter_devices": parameter_devices,
        "parameter_dtypes": parameter_dtypes,
        "attention_implementation": model.config._attn_implementation,
        "model_load_seconds": load_seconds,
        "metadata_hashes": metadata_hashes,
        "source_paths": {
            "transformers_cache_utils": str(cache_utils_path),
            "transformers_qwen3_modeling": str(qwen3_modeling_path),
        },
        "source_hashes": source_hashes,
        "prompt_token_count": int(input_ids.shape[1]),
        "filler_count": filler_count,
        "thinking_tags_present": think_match is not None,
        "thinking_body_non_whitespace": bool(think_match and think_match.group(1).strip()),
        "residual_length": args.residual_length,
        "installed_attention_modules": installed_attention_modules,
        "prefill_logits_exact": prefill_logits_exact,
        "prefill_logits_max_abs_delta": prefill_logits_max_abs_delta,
        "prefill_key_quantized_lengths": sorted(set(prefill_key_quantized_lengths)),
        "prefill_value_quantized_lengths": sorted(set(prefill_value_quantized_lengths)),
        "generated_token_ids": generated_ids,
        "generated_text_raw": generated_text_raw,
        "generated_text_clean": generated_text_clean,
        "generation_seconds": generation_seconds,
        "semantic_contains_digit_5": "5" in generated_text_clean,
        "cache_length_after_generate": cache_length_after_generate,
        "resume_length_before": resume_length_before,
        "resume_length_after": resume_length_after,
        "resume_logits_finite": resume_logits_finite,
        "resume_next_token_id": resume_next_token_id,
        "resume_next_token_text": resume_next_token_text,
        "layer_policy_count": len(generation_cache.layer_policies),
        "key_bucket_shapes": [[list(shape) for shape in entry] for entry in key_bucket_shapes],
        "value_bucket_shapes": [[list(shape) for shape in entry] for entry in value_bucket_shapes],
        "key_index_dtypes": key_index_dtypes,
        "value_index_dtypes": value_index_dtypes,
        "cache_tensor_dtypes": cache_tensor_dtypes,
        "cache_tensors_finite": cache_tensors_finite,
        "final_key_quantized_lengths": sorted(set(generation_cache.key_quantized_lengths)),
        "final_value_quantized_lengths": sorted(set(generation_cache.value_quantized_lengths)),
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "representation_note": "fake quantization: internal cache remains FP16 tensors; no packed-runtime memory claim",
        "checks": checks,
    }
    _write_json_atomic(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"acceptance_json: {output_path}")
    print(f"acceptance_json_sha256: {_sha256(output_path)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
