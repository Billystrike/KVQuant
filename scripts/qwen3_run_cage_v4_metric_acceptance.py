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

import scripts.qwen3_run_cage_partition as formal_runtime
import scripts.qwen3_run_cage_v2_round1 as round1_runtime
import scripts.qwen3_run_perturbation_partition as perturbation_runtime
from utils.qwen3_cage_v4_acceptance import PARTITIONS, expand_acceptance_cases
from utils.qwen3_cage_v4_data import file_sha256, load_data_protocol, validate_input_manifest
from utils.qwen3_cage_v4_metric_protocol import load_metric_protocol
from utils.qwen3_formal import formal_scoring
from utils.qwen3_perturbation_protocol import (
    aggregate_layer_metrics,
    validate_aggregates,
    validate_layer_records,
)


SEED = 20260810


class CageV4MetricRuntimeError(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PG-19 metric-validity acceptance partition")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CageV4MetricRuntimeError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _token_nll(logits: torch.Tensor, target: torch.Tensor) -> float:
    loss = F.cross_entropy(logits.float(), target.reshape(1), reduction="sum")
    value = float(loss.detach().cpu().item())
    if not math.isfinite(value) or value < 0:
        raise CageV4MetricRuntimeError(f"invalid token NLL: {value}")
    return value


def _make_cache(method: dict[str, Any]):
    if method["runtime_family"] == "formal":
        return formal_runtime._make_cache(method)
    return round1_runtime._make_cache(method)


def _cache_diagnostics(cache: Any, method: dict[str, Any], expected_length: int) -> dict[str, Any]:
    if method["runtime_family"] == "formal":
        return formal_runtime._cache_diagnostics(cache, method, expected_length)
    return round1_runtime._cache_diagnostics(cache, method, expected_length)


def _memory_report(method: dict[str, Any]) -> dict[str, Any]:
    if method["runtime_family"] == "formal":
        report = perturbation_runtime._memory_report(method, method["prompt_length"])
    else:
        report = round1_runtime._memory_report(method)
    if report.get("model_total_bytes") != method["packed_bytes"]:
        raise CageV4MetricRuntimeError("runtime memory report differs from frozen metric protocol")
    return report


def _quality_case(model: Any, case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = torch.tensor([case["prompt_ids"]], dtype=torch.long, device="cuda:0")
    continuation = torch.tensor([case["continuation_ids"]], dtype=torch.long, device="cuda:0")
    prompt_length = case["input"]["prompt_length"]
    if prompt.shape != (1, prompt_length) or continuation.shape != (1, 64):
        raise CageV4MetricRuntimeError("acceptance quality tensor shape mismatch")
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
        raise CageV4MetricRuntimeError("quality prefill did not preserve cache identity")
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
            raise CageV4MetricRuntimeError("quality decode did not preserve cache identity")
        token_nlls.append(_token_nll(output.logits[:, -1, :], continuation[:, index + 1]))
        del output
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - started
    diagnostics = _cache_diagnostics(cache, case["method"], prompt_length + 63)
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
    return scoring, {"cache": diagnostics, "runtime": runtime}


def _local_case(
    model: Any,
    recorder: Any,
    case: dict[str, Any],
    partition: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if case["method"]["metric_method_id"] == "fp16":
        return None, {"skipped": True, "reason": "FP16 is the zero-perturbation reference"}
    if case["method"]["runtime_family"] == "formal":
        layers, runtime = perturbation_runtime._run_case(model, recorder, case, partition)
    else:
        layers, runtime = round1_runtime._run_case(model, recorder, case)
    validate_layer_records(layers)
    aggregates = aggregate_layer_metrics(layers)
    validate_aggregates(aggregates, layers)
    if aggregates["relative_k_reconstruction_error"]["maximum"] <= 0:
        raise CageV4MetricRuntimeError("compressed method did not perturb Key cache")
    if aggregates["relative_v_reconstruction_error"]["maximum"] <= 0:
        raise CageV4MetricRuntimeError("compressed method did not perturb Value cache")
    return {
        "metric": "joint_post_o_proj_mse",
        "layer_metrics": layers,
        "aggregates": aggregates,
        "cache": runtime.pop("cache"),
    }, runtime


def _validate_record(
    record: dict[str, Any],
    *,
    case: dict[str, Any],
    run_identity: dict[str, Any],
    model_identity: dict[str, Any],
) -> None:
    if record.get("schema_version") != 1 or record.get("status") != "completed":
        raise CageV4MetricRuntimeError("acceptance record schema mismatch")
    if record.get("case_id") != case["case_id"]:
        raise CageV4MetricRuntimeError("acceptance case ID mismatch")
    if record.get("partition") != run_identity["partition"] or record.get("stage") != "acceptance":
        raise CageV4MetricRuntimeError("acceptance partition/stage mismatch")
    if record.get("identity") != run_identity or record.get("model") != model_identity:
        raise CageV4MetricRuntimeError("acceptance identity mismatch")
    if record.get("method") != case["method"] or record.get("input") != case["input"]:
        raise CageV4MetricRuntimeError("acceptance method/input mismatch")
    expected_scoring = formal_scoring(record.get("scoring", {}).get("token_nlls", []))
    if record.get("scoring") != expected_scoring:
        raise CageV4MetricRuntimeError("acceptance NLL scoring mismatch")
    if record.get("memory") != _memory_report(case["method"]):
        raise CageV4MetricRuntimeError("acceptance memory report mismatch")
    quality_cache = record.get("quality_cache")
    expected_length = case["input"]["prompt_length"] + 63
    if (
        not isinstance(quality_cache, dict)
        or quality_cache.get("reported_seq_length") != expected_length
        or quality_cache.get("expected_seq_length") != expected_length
        or quality_cache.get("layer_count") != 36
        or quality_cache.get("tensors_finite") is not True
    ):
        raise CageV4MetricRuntimeError("acceptance quality-cache diagnostics mismatch")
    local = record.get("local_perturbation")
    if case["method"]["metric_method_id"] == "fp16":
        if local is not None:
            raise CageV4MetricRuntimeError("FP16 acceptance must not carry candidate perturbation")
    else:
        if not isinstance(local, dict) or local.get("metric") != "joint_post_o_proj_mse":
            raise CageV4MetricRuntimeError("compressed acceptance lacks local perturbation")
        validate_layer_records(local.get("layer_metrics", []))
        validate_aggregates(local.get("aggregates", {}), local["layer_metrics"])


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise CageV4MetricRuntimeError("metric-validity acceptance requires CUDA")
    protocol_path = args.protocol.resolve()
    manifest_path = args.input_manifest.resolve()
    protocol, protocol_sha256 = load_metric_protocol(protocol_path)
    if file_sha256(manifest_path) != protocol["input_receipt"]["input_manifest_sha256"]:
        raise CageV4MetricRuntimeError("acceptance input manifest SHA mismatch")
    manifest = _load_json(manifest_path)
    data_protocol_path = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_pg19_data_protocol_v1.json"
    data_protocol, data_protocol_sha256 = load_data_protocol(data_protocol_path)
    if data_protocol_sha256 != protocol["input_receipt"]["data_protocol_sha256"]:
        raise CageV4MetricRuntimeError("acceptance data protocol SHA mismatch")
    validate_input_manifest(manifest, protocol=data_protocol, protocol_sha256=data_protocol_sha256)
    source_state = round1_runtime._source_state(args.partition)
    if source_state["dirty"] or source_state.get("kitty_dirty", False):
        raise CageV4MetricRuntimeError("acceptance requires clean frozen sources")
    cases = expand_acceptance_cases(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        manifest=manifest,
        partition=args.partition,
        repo_root=REPO_ROOT,
    )
    runner_sha256 = file_sha256(Path(__file__).resolve())
    utils_sha256 = file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v4_acceptance.py")
    run_identity = {
        "schema_version": 1,
        "experiment": "qwen3_cage_v4_pg19_metric_validity_acceptance",
        "claim_eligible": False,
        "partition": args.partition,
        "stage": "acceptance",
        "metric_protocol_sha256": protocol_sha256,
        "input_manifest_sha256": file_sha256(manifest_path),
        "source_state": source_state,
        "runner_sha256": runner_sha256,
        "acceptance_utils_sha256": utils_sha256,
        "expected_case_ids": [case["case_id"] for case in cases],
    }
    output_dir = args.output_dir.resolve()
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if _load_json(identity_path) != run_identity:
            raise CageV4MetricRuntimeError("existing acceptance run identity differs")
    else:
        _write_atomic(identity_path, run_identity)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/models/Qwen3-8B",
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()
    if round1_runtime.install_qwen3_cage_attention(model) != 36:
        raise CageV4MetricRuntimeError("expected 36 Qwen3 attention adapters")
    recorder = round1_runtime.Qwen3PerturbationRecorder(expected_layers=36, top_k=10)
    if recorder.install(model) != 36:
        raise CageV4MetricRuntimeError("expected 36 perturbation callbacks")
    model_identity = round1_runtime._model_identity(model, args.partition)
    model_identity["metric_acceptance_runner_sha256"] = runner_sha256

    completed = 0
    resumed = 0
    for index, case in enumerate(cases, 1):
        path = output_dir / "cases" / f"{case['case_id']}.json"
        if path.exists():
            _validate_record(
                _load_json(path), case=case, run_identity=run_identity, model_identity=model_identity
            )
            resumed += 1
            print(f"[{index}/{len(cases)}] resume-valid {case['case_id']}", flush=True)
            continue
        print(
            f"[{index}/{len(cases)}] running {case['method']['id']} "
            f"l={case['input']['prompt_length']}",
            flush=True,
        )
        try:
            local, local_runtime = _local_case(model, recorder, case, args.partition)
            scoring, quality_runtime = _quality_case(model, case)
            record = {
                "schema_version": 1,
                "status": "completed",
                "case_id": case["case_id"],
                "partition": args.partition,
                "stage": "acceptance",
                "identity": run_identity,
                "model": model_identity,
                "method": case["method"],
                "input": case["input"],
                "memory": _memory_report(case["method"]),
                "scoring": scoring,
                "local_perturbation": local,
                "quality_cache": quality_runtime.pop("cache"),
                "runtime": {
                    "local": local_runtime,
                    "quality": quality_runtime["runtime"],
                },
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "representation_note": "accuracy simulation with packed paper-estimate memory",
            }
            _validate_record(
                record, case=case, run_identity=run_identity, model_identity=model_identity
            )
            _write_atomic(path, record)
            completed += 1
            local_value = (
                "fp16-reference"
                if local is None
                else f"{local['aggregates']['joint_post_o_proj_mse']['mean']:.9g}"
            )
            print(
                f"[{index}/{len(cases)}] completed {case['case_id']} "
                f"mean_nll={scoring['mean_nll']:.9g} local_mse={local_value}",
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
        record = _load_json(output_dir / "cases" / f"{case['case_id']}.json")
        _validate_record(record, case=case, run_identity=run_identity, model_identity=model_identity)
        records.append(record)
    failures = list((output_dir / "failures").glob("*.json")) if (output_dir / "failures").exists() else []
    if failures:
        raise CageV4MetricRuntimeError("acceptance failure records remain")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "partition": args.partition,
        "stage": "acceptance",
        "expected_cases": len(cases),
        "completed_cases": len(records),
        "new_cases": completed,
        "resumed_cases": resumed,
        "failure_records": 0,
        "run_identity": run_identity,
        "model": model_identity,
    }
    _write_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
