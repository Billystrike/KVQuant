#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
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
from transformers import AutoModelForCausalLM

import scripts.qwen3_run_cage_v2_round1 as round1_runtime
from models.qwen3_cage_v4 import CageV4DTQIConfig, Qwen3CageV4DTQICache
from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes
from utils.qwen3_cage_v4_data import (
    file_sha256,
    load_data_protocol,
    validate_input_manifest,
)
from utils.qwen3_cage_v4_dtqi_acceptance import (
    SCIENTIFIC_FIELDS,
    STAGE,
    expand_acceptance_cases,
    load_acceptance_execution,
    load_json,
)
from utils.qwen3_formal import formal_scoring


SEED = 20260813


class CageV4DTQIRuntimeError(RuntimeError):
    pass


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _token_nll(logits: torch.Tensor, target: torch.Tensor) -> float:
    value = float(
        F.cross_entropy(logits.float(), target.reshape(1), reduction="sum")
        .detach()
        .cpu()
        .item()
    )
    if not math.isfinite(value) or value < 0:
        raise CageV4DTQIRuntimeError(f"invalid token NLL: {value}")
    return value


def _make_cache(method: dict[str, Any]) -> Qwen3CageV4DTQICache:
    config = method["config"]
    if config["recent_query_window"] != config["residual_length"]:
        raise CageV4DTQIRuntimeError("DTQI recent window differs from residual length")
    return Qwen3CageV4DTQICache(
        CageV4DTQIConfig(
            one_bit_channels=config["one_bit_channels"],
            two_bit_channels=config["two_bit_channels"],
            key_base_group_size=128,
            key_refinement_group_size=128,
            value_group_size=128,
            sink_length=config["sink_length"],
        ),
        residual_length=config["residual_length"],
    )


def _memory_report(method: dict[str, Any]) -> dict[str, Any]:
    config = method["config"]
    report = estimate_qwen3_cage_v2_bytes(
        seq_len=method["prompt_length"],
        residual_length=config["residual_length"],
        one_bit_channels=config["one_bit_channels"],
        two_bit_channels=config["two_bit_channels"],
        sink_length=config["sink_length"],
    )
    if report["model_total_bytes"] != method["packed_bytes"]:
        raise CageV4DTQIRuntimeError("DTQI memory differs from frozen CAGE-v3 bytes")
    if report["model_total_bytes"] > method["kitty_pro_target_bytes"]:
        raise CageV4DTQIRuntimeError("DTQI memory exceeds Kitty-Pro target")
    return report


def _cache_diagnostics(
    cache: Qwen3CageV4DTQICache,
    method: dict[str, Any],
    *,
    expected_length: int,
) -> dict[str, Any]:
    config = method["config"]
    residual = config["residual_length"]
    sink = min(expected_length, config["sink_length"])
    non_sink = expected_length - sink
    expected_key = sink + non_sink - non_sink % residual
    expected_value = max(sink, expected_length - residual)
    tensors = list(cache.key_cache) + list(cache.value_cache)
    policies = []
    for layer_idx, policy in enumerate(cache.layer_policies):
        expected_two = config["two_bit_channels"][layer_idx]
        policies.append(
            {
                "layer_idx": layer_idx,
                "importance_policy": policy.importance_policy,
                "global_query_weight": policy.global_query_weight,
                "recent_query_weight": policy.recent_query_weight,
                "recent_query_window": residual,
                "one_bit_channels": policy.one_bit_channels,
                "two_bit_channels": policy.two_bit_channels,
                "one_bit_index_shape": list(policy.one_bit_indices.shape),
                "two_bit_index_shape": list(policy.two_bit_indices.shape),
                "expected_two_bit_channels": expected_two,
                "matches_frozen_quota": policy.two_bit_channels == expected_two,
            }
        )
    record = {
        "cache_class": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "reported_seq_length": int(cache.get_seq_length()),
        "expected_seq_length": expected_length,
        "layer_count": len(cache.key_cache),
        "tensor_dtypes": sorted({str(tensor.dtype) for tensor in tensors}),
        "tensors_finite": all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors),
        "key_quantized_lengths": sorted(set(cache.key_quantized_lengths)),
        "value_quantized_lengths": sorted(set(cache.value_quantized_lengths)),
        "expected_key_quantized_length": expected_key,
        "expected_value_quantized_length": expected_value,
        "key_quantization_triggered": expected_key > sink,
        "value_quantization_triggered": expected_value > sink,
        "recent_query_window": residual,
        "residual_length": residual,
        "recent_window_equals_residual": config["recent_query_window"] == residual,
        "one_bit_channels": 0,
        "two_bit_quota_total": sum(config["two_bit_channels"]),
        "two_bit_quota_counts": {
            "48": config["two_bit_channels"].count(48),
            "32": config["two_bit_channels"].count(32),
            "16": config["two_bit_channels"].count(16),
        },
        "value_adaptive": False,
        "policy_checks": policies,
    }
    checks = (
        record["reported_seq_length"] == expected_length,
        record["layer_count"] == 36,
        record["tensor_dtypes"] == ["torch.float16"],
        record["tensors_finite"],
        record["key_quantized_lengths"] == [expected_key],
        record["value_quantized_lengths"] == [expected_value],
        record["key_quantization_triggered"],
        record["value_quantization_triggered"],
        record["recent_window_equals_residual"],
        record["two_bit_quota_total"] == 1152,
        record["two_bit_quota_counts"] == {"48": 12, "32": 12, "16": 12},
        len(policies) == 36,
        all(
            row["importance_policy"]
            == "dual_timescale_query_energy_times_key_variance"
            and row["global_query_weight"] == 0.5
            and row["recent_query_weight"] == 0.5
            and row["recent_query_window"] == residual
            and row["one_bit_channels"] == 0
            and row["one_bit_index_shape"] == [8, 0]
            and row["two_bit_index_shape"] == [8, row["expected_two_bit_channels"]]
            and row["matches_frozen_quota"]
            for row in policies
        ),
    )
    if not all(checks):
        raise CageV4DTQIRuntimeError("DTQI cache diagnostics failed")
    return record


def _run_case(model: Any, case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prompt_length = case["input"]["prompt_length"]
    prompt = torch.tensor([case["prompt_ids"]], dtype=torch.long, device="cuda:0")
    continuation = torch.tensor(
        [case["continuation_ids"]], dtype=torch.long, device="cuda:0"
    )
    if prompt.shape != (1, prompt_length) or continuation.shape != (1, 64):
        raise CageV4DTQIRuntimeError("DTQI acceptance input shape mismatch")
    cache = _make_cache(case["method"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started_total = time.perf_counter()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    if output.past_key_values is not cache:
        raise CageV4DTQIRuntimeError("DTQI prefill did not preserve cache identity")
    token_nlls = [_token_nll(output.logits[:, -1, :], continuation[:, 0])]
    del output
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for index in range(63):
        attention_mask = torch.ones(
            (1, prompt_length + index + 1), dtype=torch.long, device="cuda:0"
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
            raise CageV4DTQIRuntimeError("DTQI decode did not preserve cache identity")
        token_nlls.append(_token_nll(output.logits[:, -1, :], continuation[:, index + 1]))
        del output
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - started
    length_before_resume = int(cache.get_seq_length())
    with torch.inference_mode():
        resumed = model(
            input_ids=continuation[:, -1:],
            attention_mask=torch.ones(
                (1, prompt_length + 64), dtype=torch.long, device="cuda:0"
            ),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    resume = {
        "length_before": length_before_resume,
        "length_after": int(cache.get_seq_length()),
        "expected_length_before": prompt_length + 63,
        "expected_length_after": prompt_length + 64,
        "cache_identity_preserved": resumed.past_key_values is cache,
        "logits_finite": bool(torch.isfinite(resumed.logits).all().item()),
        "next_token_id": int(resumed.logits[:, -1, :].argmax(dim=-1).item()),
    }
    del resumed
    if (
        resume["length_before"] != resume["expected_length_before"]
        or resume["length_after"] != resume["expected_length_after"]
        or not resume["cache_identity_preserved"]
        or not resume["logits_finite"]
    ):
        raise CageV4DTQIRuntimeError("DTQI resume validation failed")
    cache_record = _cache_diagnostics(
        cache,
        case["method"],
        expected_length=prompt_length + 64,
    )
    scoring = formal_scoring(token_nlls)
    runtime = {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "elapsed_seconds": time.perf_counter() - started_total,
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    del cache, prompt, continuation, attention_mask
    gc.collect()
    torch.cuda.empty_cache()
    return scoring, cache_record, {"resume": resume, "runtime": runtime}


def _validate_record(
    record: dict[str, Any],
    *,
    case: dict[str, Any],
    identity: dict[str, Any],
    model_identity: dict[str, Any],
) -> None:
    checks = (
        record.get("schema_version") == 1,
        record.get("status") == "completed",
        record.get("stage") == STAGE,
        record.get("case_id") == case["case_id"],
        record.get("identity") == identity,
        record.get("model") == model_identity,
        record.get("method") == case["method"],
        record.get("input") == case["input"],
        record.get("memory") == _memory_report(case["method"]),
        record.get("scoring")
        == formal_scoring(record.get("scoring", {}).get("token_nlls", [])),
    )
    if not all(checks):
        raise CageV4DTQIRuntimeError("DTQI acceptance record identity/schema mismatch")
    cache = record.get("cache", {})
    resume = record.get("resume", {})
    expected_length = case["input"]["prompt_length"] + 64
    if (
        cache.get("reported_seq_length") != expected_length
        or cache.get("tensors_finite") is not True
        or cache.get("recent_window_equals_residual") is not True
        or cache.get("key_quantization_triggered") is not True
        or cache.get("value_quantization_triggered") is not True
        or resume.get("length_after") != expected_length
        or resume.get("cache_identity_preserved") is not True
        or resume.get("logits_finite") is not True
    ):
        raise CageV4DTQIRuntimeError("DTQI acceptance cache/resume record mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen CAGE-v4-DTQI GPU acceptance")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise CageV4DTQIRuntimeError("CAGE-v4-DTQI GPU acceptance requires CUDA")
    execution_path = args.execution.resolve()
    execution, execution_sha256, protocol, metric_protocol, quota_plan = (
        load_acceptance_execution(
            execution_path,
            repo_root=REPO_ROOT,
            verify_artifacts=True,
        )
    )
    manifest_path = Path(execution["input_manifest"]["path"])
    manifest = load_json(manifest_path)
    data_path = REPO_ROOT / execution["data_protocol"]["path"]
    data_protocol, data_sha256 = load_data_protocol(data_path)
    if data_sha256 != execution["data_protocol"]["sha256"]:
        raise CageV4DTQIRuntimeError("DTQI data protocol hash mismatch")
    validate_input_manifest(manifest, protocol=data_protocol, protocol_sha256=data_sha256)
    cases = expand_acceptance_cases(
        execution=execution,
        execution_sha256=execution_sha256,
        dtqi_protocol=protocol,
        metric_protocol=metric_protocol,
        quota_plan=quota_plan,
        input_manifest=manifest,
    )
    source_state = round1_runtime._source_state("cage_qwen3")
    if source_state["dirty"] or source_state.get("kitty_dirty", False):
        raise CageV4DTQIRuntimeError("DTQI acceptance requires clean sources")
    identity = {
        "schema_version": 1,
        "experiment": "qwen3_cage_v4_dtqi_gpu_acceptance",
        "claim_eligible": False,
        "stage": STAGE,
        "execution_sha256": execution_sha256,
        "dtqi_protocol_sha256": execution["dtqi_protocol"]["sha256"],
        "cpu_acceptance_receipt_sha256": execution["cpu_acceptance_receipt"]["sha256"],
        "input_manifest_sha256": execution["input_manifest"]["sha256"],
        "source_state": source_state,
        "expected_case_ids": [case["case_id"] for case in cases],
    }
    output_dir = args.output_dir.resolve()
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if load_json(identity_path) != identity:
            raise CageV4DTQIRuntimeError("existing DTQI acceptance identity differs")
    else:
        _write_atomic(identity_path, identity)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    environment = execution["environment"]
    if str(Path(sys.prefix).resolve()) != environment["conda_prefix"]:
        raise CageV4DTQIRuntimeError("DTQI acceptance Conda environment mismatch")
    if torch.version.cuda != environment["cuda"]:
        raise CageV4DTQIRuntimeError("DTQI acceptance CUDA version mismatch")
    if torch.cuda.get_device_name(0) != environment["gpu"]:
        raise CageV4DTQIRuntimeError("DTQI acceptance GPU identity mismatch")
    model = AutoModelForCausalLM.from_pretrained(
        execution["model"]["path"],
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()
    if round1_runtime.install_qwen3_cage_attention(model) != 36:
        raise CageV4DTQIRuntimeError("expected 36 Qwen3 attention adapters")
    model_identity = round1_runtime._model_identity(model, "cage_qwen3")
    expected_model = execution["model"]
    if (
        model_identity["reference"] != expected_model["path"]
        or model_identity["parameter_count"] != expected_model["parameter_count"]
        or model_identity["parameter_devices"] != [expected_model["device"]]
        or model_identity["parameter_dtypes"] != [expected_model["dtype"]]
        or model_identity["attention_implementation"]
        != expected_model["attention_implementation"]
        or model_identity["metadata_hashes"]["config_sha256"]
        != expected_model["config_sha256"]
        or model_identity["metadata_hashes"]["model_index_sha256"]
        != expected_model["model_index_sha256"]
    ):
        raise CageV4DTQIRuntimeError("DTQI acceptance model identity mismatch")
    model_identity["dtqi_acceptance_execution_sha256"] = execution_sha256
    model_identity["dtqi_runtime_sha256"] = execution["source_files"]["dtqi_runtime"]["sha256"]
    completed = 0
    resumed_count = 0
    for index, case in enumerate(cases, 1):
        path = output_dir / "cases" / f"{case['case_id']}.json"
        if path.exists():
            _validate_record(
                load_json(path),
                case=case,
                identity=identity,
                model_identity=model_identity,
            )
            resumed_count += 1
            print(f"[{index}/3] resume-valid {case['case_id']}", flush=True)
            continue
        print(
            f"[{index}/3] running cage-v4-dtqi l={case['input']['prompt_length']}",
            flush=True,
        )
        try:
            scoring, cache_record, details = _run_case(model, case)
            record = {
                "schema_version": 1,
                "status": "completed",
                "stage": STAGE,
                "case_id": case["case_id"],
                "identity": identity,
                "model": model_identity,
                "method": case["method"],
                "input": case["input"],
                "memory": _memory_report(case["method"]),
                "scoring": scoring,
                "cache": cache_record,
                "resume": details["resume"],
                "runtime": details["runtime"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "representation_note": "fake-quant accuracy simulation with packed paper-estimate memory",
            }
            _validate_record(
                record,
                case=case,
                identity=identity,
                model_identity=model_identity,
            )
            _write_atomic(path, record)
            completed += 1
            print(
                f"[{index}/3] completed {case['case_id']} "
                f"mean_nll={scoring['mean_nll']:.9g}",
                flush=True,
            )
        except Exception as error:
            _write_atomic(
                output_dir / "failures" / f"{case['case_id']}.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "case_id": case["case_id"],
                    "method": case["method"],
                    "input": case["input"],
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise
    records = []
    for case in cases:
        record = load_json(output_dir / "cases" / f"{case['case_id']}.json")
        _validate_record(
            record,
            case=case,
            identity=identity,
            model_identity=model_identity,
        )
        records.append(record)
    failure_files = (
        list((output_dir / "failures").glob("*.json"))
        if (output_dir / "failures").exists()
        else []
    )
    if failure_files:
        raise CageV4DTQIRuntimeError("DTQI acceptance failure records remain")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "stage": STAGE,
        "expected_cases": 3,
        "completed_cases": len(records),
        "new_cases": completed,
        "resumed_cases": resumed_count,
        "failure_records": 0,
        "identity": identity,
        "model": model_identity,
        "scientific_fields": list(SCIENTIFIC_FIELDS),
        "full_screen_authorized": False,
        "holdout_accessed": False,
        "pg19_test_accessed": False,
    }
    _write_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
