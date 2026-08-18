#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, REQUIRED_CUBLAS_WORKSPACE_CONFIG):
    raise RuntimeError(
        "CUBLAS_WORKSPACE_CONFIG must equal :4096:8 for the frozen deterministic acceptance"
    )
os.environ["CUBLAS_WORKSPACE_CONFIG"] = REQUIRED_CUBLAS_WORKSPACE_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoConfig, LlamaForCausalLM

from models.llama_cage_v3 import install_llama_cage_v3_config, unpack_cache
from models.llama_kivi import LlamaForCausalLM_KIVI
from models.cage_cache import is_cage_past_key_value
from utils.cage_experiment_config import apply_method_config, resolve_method
from utils.llama2_cage_v3_transfer_quality_acceptance import (
    expand_acceptance_cases,
    load_execution,
    load_json,
    load_server_input_manifest,
    require,
    validate_completed_case,
    validate_gate_receipt,
)
from utils.llama2_cage_v3_transfer_quality_protocol import load_transfer_quality_protocol
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout)
    require(not dirty, "quality acceptance requires a clean repository")
    return {"git_commit": commit, "dirty": False}


def _token_nll(logits: torch.Tensor, target: torch.Tensor) -> float:
    value = float(F.cross_entropy(logits.float(), target.reshape(1), reduction="sum").detach().cpu())
    require(math.isfinite(value) and value >= 0, "token NLL is invalid")
    return value


def _one_shot_nlls(model, prompt: torch.Tensor, continuation: torch.Tensor) -> list[float]:
    full = torch.cat((prompt, continuation), dim=-1)
    with torch.inference_mode():
        output = model(input_ids=full, use_cache=False, return_dict=True)
    start = prompt.shape[-1] - 1
    logits = output.logits[:, start : start + 64, :]
    require(logits.shape[1] == 64, "FP16 one-shot target coverage changed")
    values = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), continuation.reshape(-1), reduction="none")
    result = [float(value) for value in values.detach().cpu().tolist()]
    require(len(result) == 64 and all(math.isfinite(value) and value >= 0 for value in result), "FP16 one-shot NLLs are invalid")
    return result


def _configure_model(native_config, case: dict, quota: dict):
    method = case["method"]
    family = method["method"]
    config = copy.deepcopy(native_config)
    config.use_cache = True
    if family == "fp16":
        return config, LlamaForCausalLM
    if family == "cage_v3":
        install_llama_cage_v3_config(config, quota)
        config.k_bits = 2
        config.v_bits = 2
        config.group_size = 128
        config.residual_length = method["residual_length"]
        config.use_flash = True
        config._flash_attn_2_enabled = False
        return config, LlamaForCausalLM_KIVI
    if family == "cage_v1":
        raw = {"id": method["id"], "method": "cage", "residual_length": method["residual_length"]}
        resolved = resolve_method(raw)
        require(resolved["method_config"]["cage_k_group_sizes"] == method["key_group_sizes"], "CAGE-v1 key groups changed")
        require(resolved["method_config"]["cage_v_group_sizes"] == method["value_group_sizes"], "CAGE-v1 value groups changed")
        require(resolved["method_config"]["cage_k_clip_percentiles"] == method["key_clip_percentiles"], "CAGE-v1 key clips changed")
        require(resolved["method_config"]["cage_v_clip_percentiles"] == method["value_clip_percentiles"], "CAGE-v1 value clips changed")
        apply_method_config(config, "cage", resolved["method_config"])
        config._flash_attn_2_enabled = False
        return config, LlamaForCausalLM_KIVI
    if family == "kivi":
        raw = {"id": method["id"], "method": "kivi", "k_bits": 2, "v_bits": 2, "group_size": method["group_size"], "residual_length": method["residual_length"]}
        resolved = resolve_method(raw)
        apply_method_config(config, "kivi", resolved["method_config"])
        config._flash_attn_2_enabled = False
        return config, LlamaForCausalLM_KIVI
    raise ValueError(f"unsupported acceptance method {family}")


def _cache_lengths(past_key_values, family: str) -> list[int]:
    require(len(past_key_values) == 32, "cache layer count changed")
    lengths = []
    for layer in past_key_values:
        if family == "fp16":
            require(isinstance(layer, tuple) and len(layer) == 2, "FP16 cache structure changed")
            lengths.append(int(layer[0].shape[-2]))
        elif family == "cage_v3":
            lengths.append(int(unpack_cache(layer).kv_seq_len))
        elif family == "cage_v1":
            require(is_cage_past_key_value(layer), "CAGE-v1 cache path was not triggered")
            lengths.append(layer[-1])
        else:
            require(isinstance(layer, tuple) and len(layer) == 9 and type(layer[-1]) is int, "KIVI cache path was not triggered")
            lengths.append(layer[-1])
    return lengths


def _score_case(model, case: dict) -> tuple[dict, dict, dict]:
    prompt = torch.tensor([case["input"]["prompt_ids"]], dtype=torch.long, device="cuda:0")
    continuation = torch.tensor([case["input"]["continuation_ids"]], dtype=torch.long, device="cuda:0")
    family = case["method"]["method"]
    reference = _one_shot_nlls(model, prompt, continuation) if family == "fp16" else None
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=prompt, use_cache=True, return_dict=True)
    nlls = [_token_nll(output.logits[:, -1, :], continuation[:, 0])]
    past = output.past_key_values
    del output
    for index in range(63):
        with torch.inference_mode():
            output = model(input_ids=continuation[:, index : index + 1], past_key_values=past, use_cache=True, return_dict=True)
        nlls.append(_token_nll(output.logits[:, -1, :], continuation[:, index + 1]))
        past = output.past_key_values
        del output
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    lengths = _cache_lengths(past, family)
    expected_length = case["input"]["identity"]["prompt_length"] + 63
    require(all(length == expected_length for length in lengths), "final cache lengths changed")
    nll_sum = math.fsum(nlls)
    scoring = {
        "primary_metric": "cache_conditioned_all_64_target_mean_nll",
        "boundary_target_count": 1,
        "decode_target_count": 63,
        "all_target_count": 64,
        "token_nlls": nlls,
        "nll_sum": nll_sum,
        "mean_nll": nll_sum / 64,
        "perplexity": math.exp(nll_sum / 64),
        "fp16_one_shot_reference": None,
    }
    if reference is not None:
        deltas = [abs(left - right) for left, right in zip(nlls, reference)]
        scoring["fp16_one_shot_reference"] = {
            "token_nlls": reference,
            "mean_nll": math.fsum(reference) / 64,
            "mean_absolute_token_nll_delta": math.fsum(deltas) / 64,
            "max_absolute_token_nll_delta": max(deltas),
        }
    cache = {
        "layer_count": 32,
        "final_cache_length": expected_length,
        "all_layer_lengths_equal": len(set(lengths)) == 1,
        "method_family": family,
        "quantized_path_verified": family != "fp16",
    }
    telemetry = {
        "score_seconds": elapsed,
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    del past, prompt, continuation
    return scoring, cache, telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen Llama-2 CAGE-v3 transfer-quality acceptance repeat")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--gate-receipt", type=Path, required=True)
    parser.add_argument("--repeat", choices=("a", "b"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    execution, execution_sha = load_execution(args.execution.resolve(), repo_root=REPO_ROOT)
    gate = load_json(args.gate_receipt.resolve())
    validate_gate_receipt(gate, execution_sha256=execution_sha, repo_root=REPO_ROOT)
    protocol, protocol_sha = load_transfer_quality_protocol(REPO_ROOT / execution["protocol"]["path"], repo_root=REPO_ROOT)
    manifest = load_server_input_manifest(execution, protocol=protocol)
    cases = expand_acceptance_cases(execution=execution, execution_sha256=execution_sha, protocol=protocol, input_manifest=manifest)
    source_state = _source_state()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    failures_dir = output_dir / "failures"
    cases_dir.mkdir(exist_ok=True)
    failures_dir.mkdir(exist_ok=True)

    torch.manual_seed(execution["determinism"]["seed"])
    require(
        execution["determinism"]["cublas_workspace_config"] == REQUIRED_CUBLAS_WORKSPACE_CONFIG
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == REQUIRED_CUBLAS_WORKSPACE_CONFIG,
        "deterministic CuBLAS workspace configuration changed",
    )
    torch.cuda.manual_seed_all(execution["determinism"]["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.get_device_name(0) == execution["environment"]["gpu"], "GPU identity changed")
    require(str(torch.__version__) == execution["environment"]["torch"], "torch version changed")
    require(__import__("transformers").__version__ == execution["environment"]["transformers"], "Transformers version changed")
    native_config = AutoConfig.from_pretrained(execution["model"]["reference"], local_files_only=True)
    quota = load_json(REPO_ROOT / execution["quota_plan"]["path"])

    expected_ids = [case["case_id"] for case in cases]
    resumed = 0
    new = 0
    for index, case in enumerate(cases, start=1):
        path = cases_dir / f"{case['case_id']}.json"
        if path.is_file():
            existing = load_json(path)
            validate_completed_case(existing, case)
            resumed += 1
            print(f"[{index}/12] resumed {case['case_id']} {case['method']['id']}")
            continue
        print(f"[{index}/12] running {case['method']['id']}")
        config, model_class = _configure_model(native_config, case, quota)
        load_started = time.perf_counter()
        model = model_class.from_pretrained(
            execution["model"]["reference"], config=config, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, local_files_only=True,
        ).to("cuda:0").eval()
        require(sorted({str(parameter.device) for parameter in model.parameters()}) == ["cuda:0"], "model parameters are not exclusively on cuda:0")
        require(sorted({str(parameter.dtype) for parameter in model.parameters()}) == ["torch.float16"], "model parameters are not exclusively float16")
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        try:
            scoring, cache, telemetry = _score_case(model, case)
            record = {
                "schema_version": 1,
                "status": "completed",
                "case_id": case["case_id"],
                "method": case["method"],
                "input": case["input"],
                "memory": case["memory"],
                "scoring": scoring,
                "cache": cache,
                "runtime_diagnostics": {"model_load_seconds": load_seconds, **telemetry},
                "provenance": {
                    "repeat": args.repeat,
                    "source_state": source_state,
                    "execution_sha256": execution_sha,
                    "protocol_sha256": protocol_sha,
                    "input_manifest_sha256": execution["input_manifest"]["sha256"],
                    "gate_receipt_sha256": file_sha256(args.gate_receipt.resolve()),
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "transformers": __import__("transformers").__version__,
                    "gpu": torch.cuda.get_device_name(0),
                    "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                },
            }
            validate_completed_case(record, case)
            _atomic_json(path, record)
            (failures_dir / f"{case['case_id']}.json").unlink(missing_ok=True)
            new += 1
            print(f"[{index}/12] completed {case['case_id']} mean_nll={scoring['mean_nll']:.9g}")
        except Exception as error:
            _atomic_json(failures_dir / f"{case['case_id']}.json", {"case_id": case["case_id"], "status": "failed", "error": str(error)})
            raise
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    records = [load_json(cases_dir / f"{case_id}.json") for case_id in expected_ids]
    for record, case in zip(records, cases):
        validate_completed_case(record, case)
    scientific = [{field: record[field] for field in ("case_id", "method", "input", "memory", "scoring", "cache")} for record in records]
    identity = {
        "execution_sha256": execution_sha,
        "protocol_sha256": protocol_sha,
        "input_manifest_sha256": execution["input_manifest"]["sha256"],
        "gate_receipt_sha256": file_sha256(args.gate_receipt.resolve()),
        "source_state": source_state,
        "expected_case_ids": expected_ids,
    }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "repeat": args.repeat,
        "expected_cases": 12,
        "completed_cases": len(records),
        "failure_records": len(list(failures_dir.glob("*.json"))),
        "new_cases": new,
        "resumed_cases": resumed,
        "scientific_payload_sha256": canonical_sha256(scientific),
        "execution_boundary": {
            "acceptance_only": True,
            "full_600_case_execution_authorized": False,
            "candidate_tuning_authorized": False,
            "paper_claims_authorized": False,
            "runtime_claims_authorized": False,
        },
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }
    require(summary["completed_cases"] == 12 and summary["failure_records"] == 0, "acceptance completion failed")
    _atomic_json(output_dir / "run_identity.json", identity)
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
