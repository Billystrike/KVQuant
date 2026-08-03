#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers import cache_utils as transformers_cache_utils
from transformers.models.qwen3 import modeling_qwen3

from models.cage_config import CageConfig
from models.qwen3_cage import Qwen3CageCache, install_qwen3_cage_attention
from models.qwen3_kivi import Qwen3KiviCache, Qwen3KiviCacheConfig
from utils.qwen3_formal import (
    Qwen3FormalError,
    atomic_write_json,
    expand_partition_cases,
    file_sha256,
    formal_scoring,
    load_execution_config,
    load_formal_protocol,
    validate_completed_result,
    validate_input_manifest,
)


EXPECTED = {
    "python": "3.10.20",
    "torch": "2.4.1+cu121",
    "transformers": "4.53.2",
    "parameter_count": 8190735360,
    "config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "generation_config_sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "model_index_sha256": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "transformers_cache_utils_sha256": "529e858b9bb0ba59a830713bbd5ab594e29f262bbad1bba860ff7de3248e5a5a",
    "transformers_qwen3_modeling_sha256": "c12b4c4a34a06e887f3df399b1b09e9966c12d2a19769c258b7379066e585ea4",
}
PARTITION = "cage_qwen3"
SEED = 20260803


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Qwen3 FP16/CAGE/KIVI partition")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("acceptance", "full"), required=True)
    return parser.parse_args()


def _source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    ).stdout
    return {"git_commit": commit, "dirty": bool(status)}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3FormalError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Qwen3FormalError(f"JSON root must be an object: {path}")
    return value


def _token_nll(logits: torch.Tensor, target: torch.Tensor) -> float:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise Qwen3FormalError(f"expected logits [1, vocab], got {tuple(logits.shape)}")
    loss = F.cross_entropy(logits.float(), target.reshape(1), reduction="sum")
    value = float(loss.detach().cpu().item())
    if not math.isfinite(value) or value < 0:
        raise Qwen3FormalError(f"invalid token NLL {value!r}")
    return value


def _cage_cache(config: dict[str, Any]) -> Qwen3CageCache:
    cage_config = CageConfig(
        cage_enable=True,
        cage_mode="fake",
        cage_k_enable=True,
        cage_v_enable=True,
        cage_k_importance=config["key_importance"],
        cage_k_group_sizes=list(config["key_group_sizes"]),
        cage_k_clip_percentiles=list(config["key_clip_percentiles"]),
        cage_k_num_buckets=config["key_num_buckets"],
        cage_v_importance=config["value_importance"],
        cage_v_group_sizes=list(config["value_group_sizes"]),
        cage_v_clip_percentiles=list(config["value_clip_percentiles"]),
        cage_v_num_buckets=config["value_num_buckets"],
        cage_ablation=config["ablation"],
        cage_assignment_seed=config["assignment_seed"],
    )
    return Qwen3CageCache(
        cage_config,
        residual_length=config["residual_length"],
        bits=config["bits"],
    )


def _make_cache(method: dict[str, Any]):
    if method["name"] == "fp16":
        return DynamicCache()
    if method["name"] == "kivi":
        return Qwen3KiviCache(Qwen3KiviCacheConfig(**method["config"]))
    if method["name"] == "cage":
        return _cage_cache(method["config"])
    raise Qwen3FormalError(f"unsupported CAGE-partition method {method['name']!r}")


def _cache_diagnostics(cache: Any, method: dict[str, Any], expected_length: int) -> dict[str, Any]:
    reported_length = int(cache.get_seq_length())
    tensors = list(cache.key_cache) + list(cache.value_cache)
    finite = all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)
    record: dict[str, Any] = {
        "cache_class": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "reported_seq_length": reported_length,
        "expected_seq_length": expected_length,
        "layer_count": len(cache.key_cache),
        "tensor_dtypes": sorted({str(tensor.dtype) for tensor in tensors}),
        "tensors_finite": finite,
        "key_quantized_lengths": None,
        "value_quantized_lengths": None,
        "layer_policy_count": None,
    }
    if method["name"] in {"cage", "kivi"}:
        record["key_quantized_lengths"] = sorted(set(cache.key_quantized_lengths))
        record["value_quantized_lengths"] = sorted(set(cache.value_quantized_lengths))
        residual = method["config"]["residual_length"]
        expected_key = expected_length - expected_length % residual
        expected_value = max(0, expected_length - residual)
        if record["key_quantized_lengths"] != [expected_key]:
            raise Qwen3FormalError("final Key quantized length differs from protocol mechanics")
        if record["value_quantized_lengths"] != [expected_value]:
            raise Qwen3FormalError("final Value quantized length differs from protocol mechanics")
    if method["name"] == "cage":
        record["layer_policy_count"] = len(cache.layer_policies)
    if reported_length != expected_length:
        raise Qwen3FormalError("final cache sequence length differs from scoring protocol")
    if len(cache.key_cache) != 36 or len(cache.value_cache) != 36:
        raise Qwen3FormalError("final cache must contain all 36 layers")
    if record["tensor_dtypes"] != ["torch.float16"] or not finite:
        raise Qwen3FormalError("final cache tensors must be finite FP16")
    return record


def _run_case(model: Any, case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = torch.tensor([case["prompt_ids"]], dtype=torch.long, device="cuda:0")
    continuation = torch.tensor(
        [case["continuation_ids"]], dtype=torch.long, device="cuda:0"
    )
    prompt_length = case["input"]["prompt_length"]
    if prompt.shape[-1] != prompt_length or continuation.shape[-1] != 64:
        raise Qwen3FormalError("prepared case tensor lengths differ from frozen input")
    cache = _make_cache(case["method"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started_total = time.perf_counter()

    attention_mask = torch.ones_like(prompt)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=prompt,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    boundary = _token_nll(output.logits[:, -1, :], continuation[:, 0])
    past_key_values = output.past_key_values
    if past_key_values is not cache:
        raise Qwen3FormalError("prefill did not preserve the cache object identity")
    del output
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started

    token_nlls = [boundary]
    started = time.perf_counter()
    for index in range(63):
        attention_mask = torch.ones(
            (1, prompt_length + index + 1),
            dtype=torch.long,
            device="cuda:0",
        )
        with torch.inference_mode():
            output = model(
                input_ids=continuation[:, index : index + 1],
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        if output.past_key_values is not cache:
            raise Qwen3FormalError("decode did not preserve the cache object identity")
        token_nlls.append(
            _token_nll(output.logits[:, -1, :], continuation[:, index + 1])
        )
        del output
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - started
    elapsed_seconds = time.perf_counter() - started_total
    expected_cache_length = prompt_length + 63
    cache_record = _cache_diagnostics(cache, case["method"], expected_cache_length)
    runtime = {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "elapsed_seconds": elapsed_seconds,
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    scoring = formal_scoring(token_nlls)
    del past_key_values, cache, prompt, continuation, attention_mask
    gc.collect()
    torch.cuda.empty_cache()
    return scoring, {"cache": cache_record, "runtime": runtime}


def _model_identity(model: Any) -> dict[str, Any]:
    cache_utils_path = Path(transformers_cache_utils.__file__).resolve()
    qwen_path = Path(modeling_qwen3.__file__).resolve()
    model_path = Path(model.config._name_or_path).resolve()
    metadata_hashes = {
        "config_sha256": file_sha256(model_path / "config.json"),
        "generation_config_sha256": file_sha256(model_path / "generation_config.json"),
        "model_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        "tokenizer_config_sha256": file_sha256(model_path / "tokenizer_config.json"),
    }
    identity = {
        "reference": str(Path(model.config._name_or_path).resolve()),
        "revision": getattr(model.config, "_commit_hash", None),
        "model_type": getattr(model.config, "model_type", None),
        "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_devices": sorted({str(parameter.device) for parameter in model.parameters()}),
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
        "attention_implementation": model.config._attn_implementation,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "transformers_cache_utils_sha256": file_sha256(cache_utils_path),
        "transformers_qwen3_modeling_sha256": file_sha256(qwen_path),
        "qwen3_cage_sha256": file_sha256(REPO_ROOT / "models" / "qwen3_cage.py"),
        "qwen3_kivi_sha256": file_sha256(REPO_ROOT / "models" / "qwen3_kivi.py"),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "metadata_hashes": metadata_hashes,
    }
    checks = {
        "python": identity["python"] == EXPECTED["python"],
        "torch": identity["torch"] == EXPECTED["torch"],
        "transformers": identity["transformers"] == EXPECTED["transformers"],
        "parameter_count": identity["parameter_count"] == EXPECTED["parameter_count"],
        "parameter_device": identity["parameter_devices"] == ["cuda:0"],
        "parameter_dtype": identity["parameter_dtypes"] == ["torch.float16"],
        "attention_implementation": identity["attention_implementation"] == "flash_attention_2",
        "cache_utils": identity["transformers_cache_utils_sha256"]
        == EXPECTED["transformers_cache_utils_sha256"],
        "qwen_modeling": identity["transformers_qwen3_modeling_sha256"]
        == EXPECTED["transformers_qwen3_modeling_sha256"],
        "model_type": identity["model_type"] == "qwen3",
        "native_context": identity["max_position_embeddings"] == 40960,
        "metadata_hashes": all(
            metadata_hashes[name] == EXPECTED[name] for name in metadata_hashes
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise Qwen3FormalError(f"model/dependency identity checks failed: {failures}")
    identity["checks"] = checks
    return identity


def main() -> None:
    args = _parse_args()
    if args.stage == "full":
        raise Qwen3FormalError(
            "formal full execution is locked until both repeated acceptance gates pass"
        )
    if not torch.cuda.is_available():
        raise Qwen3FormalError("formal CAGE partition requires CUDA")
    protocol, protocol_sha256 = load_formal_protocol(args.protocol.resolve())
    execution, execution_sha256 = load_execution_config(
        args.execution_config.resolve(),
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    input_manifest_path = args.input_manifest.resolve()
    input_manifest_sha256 = file_sha256(input_manifest_path)
    if input_manifest_sha256 != execution["input_manifest"]["sha256"]:
        raise Qwen3FormalError("input manifest SHA-256 differs from execution freeze")
    input_manifest = _load_json(input_manifest_path)
    validate_input_manifest(
        input_manifest,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    source_state = _source_state()
    if source_state["dirty"]:
        raise Qwen3FormalError("formal execution requires a clean source tree")
    cases = expand_partition_cases(
        protocol=protocol,
        execution=execution,
        execution_sha256=execution_sha256,
        input_manifest=input_manifest,
        partition=PARTITION,
        stage=args.stage,
    )
    output_dir = args.output_dir.resolve()
    run_identity = {
        "schema_version": 1,
        "partition": PARTITION,
        "stage": args.stage,
        "execution_id": execution["execution_id"],
        "execution_sha256": execution_sha256,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "source_state": source_state,
        "expected_case_ids": [case["case_id"] for case in cases],
    }
    lock_path = output_dir / "run_identity.json"
    if lock_path.exists():
        if _load_json(lock_path) != run_identity:
            raise Qwen3FormalError("existing output run identity differs; refusing mixed resume")
    else:
        atomic_write_json(lock_path, run_identity)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        protocol["model"]["reference"],
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()
    load_seconds = time.perf_counter() - load_started
    installed = install_qwen3_cage_attention(model)
    if installed != 36:
        raise Qwen3FormalError(f"expected to install 36 Qwen3 attention adapters, got {installed}")
    model_identity = _model_identity(model)
    model_identity["load_seconds"] = load_seconds

    completed = 0
    resumed = 0
    identity_fields = {
        "execution_id": execution["execution_id"],
        "execution_sha256": execution_sha256,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "source_state": source_state,
    }
    for index, case in enumerate(cases, 1):
        result_path = output_dir / "cases" / f"{case['case_id']}.json"
        if result_path.exists():
            record = _load_json(result_path)
            validate_completed_result(
                record,
                expected_case=case,
                **identity_fields,
            )
            failure_path = output_dir / "failures" / f"{case['case_id']}.json"
            if failure_path.exists():
                failure_path.unlink()
            resumed += 1
            print(f"[{index}/{len(cases)}] resume-valid {case['case_id']} {case['method']['id']}", flush=True)
            continue
        print(
            f"[{index}/{len(cases)}] running {case['case_id']} "
            f"{case['method']['id']} l={case['input']['prompt_length']} "
            f"a={case['input']['anchor_index']}",
            flush=True,
        )
        try:
            scoring, diagnostics = _run_case(model, case)
            record = {
                "schema_version": 1,
                "status": "completed",
                "case_id": case["case_id"],
                "partition": PARTITION,
                "stage": args.stage,
                "identity": identity_fields,
                "model": model_identity,
                "method": case["method"],
                "input": case["input"],
                "scoring": scoring,
                "cache": diagnostics["cache"],
                "runtime": diagnostics["runtime"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "representation_note": "fake-quantized FP16 cache; packed paper-estimate quality path only",
            }
            validate_completed_result(
                record,
                expected_case=case,
                **identity_fields,
            )
            atomic_write_json(result_path, record)
            failure_path = output_dir / "failures" / f"{case['case_id']}.json"
            if failure_path.exists():
                failure_path.unlink()
            completed += 1
            print(
                f"[{index}/{len(cases)}] completed {case['case_id']} "
                f"elapsed={record['runtime']['elapsed_seconds']:.3f}s",
                flush=True,
            )
        except Exception as error:
            failure = {
                "schema_version": 1,
                "status": "failed",
                "case_id": case["case_id"],
                "partition": PARTITION,
                "stage": args.stage,
                "identity": identity_fields,
                "method": case["method"],
                "input": case["input"],
                "error_type": type(error).__name__,
                "message": str(error),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(output_dir / "failures" / f"{case['case_id']}.json", failure)
            raise

    records = []
    for case in cases:
        record = _load_json(output_dir / "cases" / f"{case['case_id']}.json")
        validate_completed_result(record, expected_case=case, **identity_fields)
        records.append(record)
    summary = {
        "schema_version": 1,
        "status": "pass",
        "partition": PARTITION,
        "stage": args.stage,
        "expected_cases": len(cases),
        "completed_cases": len(records),
        "new_cases": completed,
        "resumed_cases": resumed,
        "failure_records": len(list((output_dir / "failures").glob("*.json")))
        if (output_dir / "failures").exists()
        else 0,
        "identity": identity_fields,
        "model": model_identity,
        "case_ids_sha256": hashlib.sha256(
            json.dumps([case["case_id"] for case in cases], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    if summary["failure_records"] != 0:
        raise Qwen3FormalError("failure records remain in the formal output directory")
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
