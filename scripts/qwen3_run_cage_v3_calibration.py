#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
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
import transformers
from transformers import AutoModelForCausalLM
from transformers import cache_utils as transformers_cache_utils
from transformers.models.qwen3 import modeling_qwen3

from models.qwen3_cage import install_qwen3_cage_attention
from models.qwen3_cage_v2 import CageV2Config, Qwen3CageV2Cache
from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes
from utils.qwen3_cage_v3_execution import (
    canonical_sha256,
    expand_calibration_cases,
    load_calibration_execution,
    load_json,
)
from utils.qwen3_cage_v3_protocol import file_sha256
from utils.qwen3_perturbation_protocol import (
    aggregate_layer_metrics,
    validate_aggregates,
    validate_layer_records,
)
from utils.qwen3_perturbation_runtime import Qwen3PerturbationRecorder


EXPECTED = {
    "python": "3.10.20",
    "torch": "2.4.1+cu121",
    "transformers": "4.53.2",
    "parameter_count": 8190735360,
    "cache_utils_sha256": "529e858b9bb0ba59a830713bbd5ab594e29f262bbad1bba860ff7de3248e5a5a",
    "qwen3_modeling_sha256": "c12b4c4a34a06e887f3df399b1b09e9966c12d2a19769c258b7379066e585ea4",
    "config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "generation_config_sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "model_index_sha256": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
}
SEED = 20260809


class CageV3CalibrationError(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen CAGE-v3 calibration probes")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibration_acceptance", "calibration_full"), required=True)
    parser.add_argument("--acceptance-gate", type=Path)
    return parser.parse_args()


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_state() -> dict[str, Any]:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=all")),
    }


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
        raise CageV3CalibrationError("calibration runtime memory differs from frozen bytes")
    if report["model_total_bytes"] > method["target_bytes"]:
        raise CageV3CalibrationError("calibration point exceeds Kitty-Pro byte ceiling")
    return report


def _make_cache(method: dict[str, Any]) -> Qwen3CageV2Cache:
    config = method["config"]
    return Qwen3CageV2Cache(
        CageV2Config(
            one_bit_channels=0,
            two_bit_channels=32,
            key_base_group_size=128,
            key_refinement_group_size=128,
            value_group_size=128,
            sink_length=config["sink_length"],
        ),
        residual_length=config["residual_length"],
    )


def _cache_diagnostics(cache: Qwen3CageV2Cache, method: dict[str, Any], expected_length: int) -> dict[str, Any]:
    tensors = list(cache.key_cache) + list(cache.value_cache)
    residual = method["config"]["residual_length"]
    sink = min(expected_length, method["config"]["sink_length"])
    non_sink = expected_length - sink
    expected_key = sink + non_sink - non_sink % residual
    expected_value = max(sink, expected_length - residual)
    record = {
        "cache_class": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "reported_seq_length": int(cache.get_seq_length()),
        "expected_seq_length": expected_length,
        "layer_count": len(cache.key_cache),
        "tensor_dtypes": sorted({str(tensor.dtype) for tensor in tensors}),
        "tensors_finite": all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors),
        "key_quantized_lengths": sorted(set(cache.key_quantized_lengths)),
        "value_quantized_lengths": sorted(set(cache.value_quantized_lengths)),
        "one_bit_channels": sorted({policy.one_bit_channels for policy in cache.layer_policies}),
        "two_bit_channels": sorted({policy.two_bit_channels for policy in cache.layer_policies}),
        "one_bit_indices_empty": all(policy.one_bit_indices.shape[-1] == 0 for policy in cache.layer_policies),
        "value_adaptive": False,
    }
    checks = (
        record["reported_seq_length"] == expected_length,
        record["layer_count"] == 36,
        record["tensor_dtypes"] == ["torch.float16"],
        record["tensors_finite"],
        record["key_quantized_lengths"] == [expected_key],
        record["value_quantized_lengths"] == [expected_value],
        record["one_bit_channels"] == [0],
        record["two_bit_channels"] == [32],
        record["one_bit_indices_empty"],
    )
    if not all(checks):
        raise CageV3CalibrationError("CAGE-v3 calibration cache mechanics mismatch")
    return record


def _run_case(model: Any, recorder: Qwen3PerturbationRecorder, case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_length = case["input"]["prompt_length"]
    prompt = torch.tensor([case["prompt_ids"]], dtype=torch.long, device="cuda:0")
    decode_token = torch.tensor([[case["continuation_ids"][0]]], dtype=torch.long, device="cuda:0")
    full_reference = torch.cat((prompt, decode_token), dim=1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    recorder.begin_reference(prompt_length=prompt_length)
    with torch.inference_mode():
        output = model(
            input_ids=full_reference,
            attention_mask=torch.ones_like(full_reference),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    del output
    cache = _make_cache(case["method"])
    recorder.begin_candidate_prefill()
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
        raise CageV3CalibrationError("calibration prefill did not preserve cache identity")
    del output
    recorder.begin_candidate_decode()
    with torch.inference_mode():
        output = model(
            input_ids=decode_token,
            attention_mask=torch.ones((1, prompt_length + 1), dtype=torch.long, device="cuda:0"),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    if output.past_key_values is not cache:
        raise CageV3CalibrationError("calibration decode did not preserve cache identity")
    del output
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    layers = recorder.finish()
    runtime = {
        "elapsed_seconds": elapsed,
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "cache": _cache_diagnostics(cache, case["method"], prompt_length + 1),
    }
    del cache, prompt, decode_token, full_reference
    recorder.reset_case()
    gc.collect()
    torch.cuda.empty_cache()
    return layers, runtime


def _model_identity(model: Any) -> dict[str, Any]:
    cache_path = Path(transformers_cache_utils.__file__).resolve()
    qwen_path = Path(modeling_qwen3.__file__).resolve()
    model_path = Path(model.config._name_or_path).resolve()
    metadata = {
        "config_sha256": file_sha256(model_path / "config.json"),
        "generation_config_sha256": file_sha256(model_path / "generation_config.json"),
        "model_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        "tokenizer_config_sha256": file_sha256(model_path / "tokenizer_config.json"),
    }
    identity = {
        "reference": str(model_path),
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_devices": sorted({str(parameter.device) for parameter in model.parameters()}),
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
        "attention_implementation": model.config._attn_implementation,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cache_utils_sha256": file_sha256(cache_path),
        "qwen3_modeling_sha256": file_sha256(qwen_path),
        "metadata_hashes": metadata,
        "source_sha256": {
            "runner": file_sha256(Path(__file__).resolve()),
            "execution_utils": file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v3_execution.py"),
            "protocol_utils": file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v3_protocol.py"),
            "qwen3_cage": file_sha256(REPO_ROOT / "models" / "qwen3_cage.py"),
            "qwen3_cage_v2": file_sha256(REPO_ROOT / "models" / "qwen3_cage_v2.py"),
            "cage_v2_quant": file_sha256(REPO_ROOT / "models" / "cage_v2_quant.py"),
            "cage_v2_memory": file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v2.py"),
            "recorder": file_sha256(REPO_ROOT / "utils" / "qwen3_perturbation_runtime.py"),
        },
    }
    checks = {
        "python": identity["python"] == EXPECTED["python"],
        "torch": identity["torch"] == EXPECTED["torch"],
        "transformers": identity["transformers"] == EXPECTED["transformers"],
        "parameters": identity["parameter_count"] == EXPECTED["parameter_count"],
        "device": identity["parameter_devices"] == ["cuda:0"],
        "dtype": identity["parameter_dtypes"] == ["torch.float16"],
        "attention": identity["attention_implementation"] == "flash_attention_2",
        "cache_utils": identity["cache_utils_sha256"] == EXPECTED["cache_utils_sha256"],
        "qwen3_modeling": identity["qwen3_modeling_sha256"] == EXPECTED["qwen3_modeling_sha256"],
        "metadata": all(metadata[name] == EXPECTED[name] for name in metadata),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise CageV3CalibrationError(f"calibration model identity checks failed: {failures}")
    identity["checks"] = checks
    return identity


def _validate_record(record: dict[str, Any], *, case: dict[str, Any], run_identity: dict[str, Any], model_identity: dict[str, Any]) -> None:
    if record.get("schema_version") != 1 or record.get("status") != "completed":
        raise CageV3CalibrationError("calibration result schema mismatch")
    if record.get("case_id") != case["case_id"] or record.get("identity") != run_identity:
        raise CageV3CalibrationError("calibration result identity mismatch")
    if record.get("model") != model_identity or record.get("method") != case["method"] or record.get("input") != case["input"]:
        raise CageV3CalibrationError("calibration model/method/input mismatch")
    validate_layer_records(record.get("layer_metrics", []))
    validate_aggregates(record.get("aggregates", {}), record["layer_metrics"])
    if record.get("memory") != _memory_report(case["method"]):
        raise CageV3CalibrationError("calibration memory report mismatch")
    if record["aggregates"]["relative_k_reconstruction_error"]["maximum"] <= 0:
        raise CageV3CalibrationError("calibration probe did not perturb Key")
    if record["aggregates"]["relative_v_reconstruction_error"]["maximum"] <= 0:
        raise CageV3CalibrationError("calibration probe did not perturb Value")


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise CageV3CalibrationError("CAGE-v3 calibration requires CUDA")
    execution_path = args.execution.resolve()
    execution, execution_sha256, protocol, manifest = load_calibration_execution(
        execution_path, repo_root=REPO_ROOT, verify_artifacts=True
    )
    source_state = _source_state()
    if source_state["dirty"]:
        raise CageV3CalibrationError("CAGE-v3 calibration requires a clean source tree")
    if args.stage == "calibration_full":
        if args.acceptance_gate is None:
            raise CageV3CalibrationError("calibration_full requires a frozen acceptance gate")
        gate = load_json(args.acceptance_gate.resolve())
        gate_checks = (
            gate.get("schema_version") == 1,
            gate.get("gate_id") == "qwen3-8b-cage-v3-calibration-acceptance-gate-v1",
            gate.get("status") == "pass",
            gate.get("claim_eligible") is False,
            gate.get("execution_sha256") == execution_sha256,
            gate.get("acceptance_source", {}).get("git_commit") == source_state["git_commit"],
            gate.get("calibration_full_authorization", {}).get("anchor_indices") == list(range(5)),
            gate.get("calibration_full_authorization", {}).get("case_count") == 30,
        )
        if not all(gate_checks):
            raise CageV3CalibrationError("calibration acceptance gate is invalid for this execution")
    elif args.acceptance_gate is not None:
        raise CageV3CalibrationError("calibration_acceptance must not consume a gate")
    cases = expand_calibration_cases(
        execution=execution,
        execution_sha256=execution_sha256,
        protocol=protocol,
        manifest=manifest,
        stage=args.stage,
    )
    run_identity = {
        "schema_version": 1,
        "experiment": "qwen3_cage_v3_calibration_development_only",
        "claim_eligible": False,
        "stage": args.stage,
        "execution_sha256": execution_sha256,
        "protocol_sha256": execution["protocol"]["sha256"],
        "manifest_sha256": execution["input_manifest"]["sha256"],
        "source_state": source_state,
        "expected_case_ids": [case["case_id"] for case in cases],
    }
    output_dir = args.output_dir.resolve()
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if load_json(identity_path) != run_identity:
            raise CageV3CalibrationError("existing calibration run identity differs")
    else:
        _write_atomic(identity_path, run_identity)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        protocol["model"]["reference"],
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()
    if install_qwen3_cage_attention(model) != 36:
        raise CageV3CalibrationError("expected 36 Qwen3 attention adapters")
    recorder = Qwen3PerturbationRecorder(expected_layers=36, top_k=10)
    if recorder.install(model) != 36:
        raise CageV3CalibrationError("expected 36 calibration perturbation callbacks")
    model_identity = _model_identity(model)

    completed = 0
    resumed = 0
    for index, case in enumerate(cases, 1):
        path = output_dir / "cases" / f"{case['case_id']}.json"
        if path.exists():
            _validate_record(load_json(path), case=case, run_identity=run_identity, model_identity=model_identity)
            resumed += 1
            print(f"[{index}/{len(cases)}] resume-valid {case['case_id']}", flush=True)
            continue
        print(f"[{index}/{len(cases)}] running {case['method']['id']} l={case['input']['prompt_length']} a={case['input']['anchor_index']}", flush=True)
        try:
            layers, runtime = _run_case(model, recorder, case)
            aggregates = aggregate_layer_metrics(layers)
            record = {
                "schema_version": 1,
                "status": "completed",
                "case_id": case["case_id"],
                "identity": run_identity,
                "model": model_identity,
                "method": case["method"],
                "input": case["input"],
                "memory": _memory_report(case["method"]),
                "layer_metrics": layers,
                "aggregates": aggregates,
                "cache": runtime.pop("cache"),
                "runtime": runtime,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "representation_note": "claim-ineligible CAGE-v3 calibration fake quantization plus packed paper estimate",
            }
            _validate_record(record, case=case, run_identity=run_identity, model_identity=model_identity)
            _write_atomic(path, record)
            completed += 1
            print(f"[{index}/{len(cases)}] completed {case['case_id']} joint_post_o_proj_mse={aggregates['joint_post_o_proj_mse']['mean']:.9g} elapsed={runtime['elapsed_seconds']:.3f}s", flush=True)
        except Exception as error:
            _write_atomic(output_dir / "failures" / f"{case['case_id']}.json", {
                "schema_version": 1,
                "status": "failed",
                "case_id": case["case_id"],
                "method": case["method"],
                "input": case["input"],
                "error_type": type(error).__name__,
                "message": str(error),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            raise

    records = []
    for case in cases:
        record = load_json(output_dir / "cases" / f"{case['case_id']}.json")
        _validate_record(record, case=case, run_identity=run_identity, model_identity=model_identity)
        records.append(record)
    failure_count = len(list((output_dir / "failures").glob("*.json"))) if (output_dir / "failures").exists() else 0
    if failure_count:
        raise CageV3CalibrationError("calibration failure records remain")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "stage": args.stage,
        "expected_cases": len(cases),
        "completed_cases": len(records),
        "new_cases": completed,
        "resumed_cases": resumed,
        "failure_records": failure_count,
        "case_ids_sha256": canonical_sha256([case["case_id"] for case in cases]),
        "identity": run_identity,
        "model": model_identity,
    }
    _write_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
