#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM

import scripts.qwen3_run_cage_v2_round1 as round1_runtime
import scripts.qwen3_run_cage_v4_dtqi_acceptance as accepted_runtime
from utils.qwen3_cage_v4_data import file_sha256, load_data_protocol, validate_input_manifest
from utils.qwen3_cage_v4_dtqi_acceptance import load_json
from utils.qwen3_cage_v4_dtqi_screen import SCIENTIFIC_FIELDS, STAGE, expand_screen_cases, load_screen_execution
from utils.qwen3_formal import formal_scoring


SEED = 20260813


class CageV4DTQIScreenRuntimeError(RuntimeError):
    pass


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_record(record: dict[str, Any], *, case: dict[str, Any], identity: dict[str, Any], model_identity: dict[str, Any]) -> None:
    checks = (
        record.get("schema_version") == 1,
        record.get("status") == "completed",
        record.get("stage") == STAGE,
        record.get("case_id") == case["case_id"],
        record.get("identity") == identity,
        record.get("model") == model_identity,
        record.get("method") == case["method"],
        record.get("input") == case["input"],
        record.get("memory") == accepted_runtime._memory_report(case["method"]),
        record.get("scoring") == formal_scoring(record.get("scoring", {}).get("token_nlls", [])),
    )
    if not all(checks):
        raise CageV4DTQIScreenRuntimeError("DTQI screen record identity/schema mismatch")
    expected_length = case["input"]["prompt_length"] + 64
    cache = record.get("cache", {})
    resume = record.get("resume", {})
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
        raise CageV4DTQIScreenRuntimeError("DTQI screen cache/resume mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen 120-case CAGE-v4-DTQI PG-19 screen")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise CageV4DTQIScreenRuntimeError("DTQI screen requires CUDA")
    execution, execution_sha, protocol, metric, quota = load_screen_execution(args.execution.resolve(), repo_root=REPO_ROOT, verify_artifacts=True)
    manifest = load_json(execution["input_manifest"]["path"])
    data_protocol, data_sha = load_data_protocol(REPO_ROOT / execution["data_protocol"]["path"])
    if data_sha != execution["data_protocol"]["sha256"]:
        raise CageV4DTQIScreenRuntimeError("DTQI screen data protocol hash mismatch")
    validate_input_manifest(manifest, protocol=data_protocol, protocol_sha256=data_sha)
    cases = expand_screen_cases(execution=execution, execution_sha256=execution_sha, protocol=protocol, quota_plan=quota, input_manifest=manifest)
    source_state = round1_runtime._source_state("cage_qwen3")
    if source_state["dirty"]:
        raise CageV4DTQIScreenRuntimeError("DTQI screen requires clean source state")
    identity = {
        "schema_version": 1,
        "experiment": "qwen3_cage_v4_dtqi_pg19_screen",
        "claim_eligible": False,
        "stage": STAGE,
        "execution_sha256": execution_sha,
        "dtqi_protocol_sha256": execution["dtqi_protocol"]["sha256"],
        "gpu_acceptance_receipt_sha256": execution["gpu_acceptance_receipt"]["sha256"],
        "input_manifest_sha256": execution["input_manifest"]["sha256"],
        "source_state": source_state,
        "expected_case_ids": [case["case_id"] for case in cases],
    }
    output_dir = args.output_dir.resolve()
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if load_json(identity_path) != identity:
            raise CageV4DTQIScreenRuntimeError("existing DTQI screen identity differs")
    else:
        _write_atomic(identity_path, identity)
    environment = execution["environment"]
    if str(Path(sys.prefix).resolve()) != environment["conda_prefix"] or torch.version.cuda != environment["cuda"] or torch.cuda.get_device_name(0) != environment["gpu"]:
        raise CageV4DTQIScreenRuntimeError("DTQI screen environment identity mismatch")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        execution["model"]["path"],
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()
    if round1_runtime.install_qwen3_cage_attention(model) != 36:
        raise CageV4DTQIScreenRuntimeError("expected 36 Qwen3 attention adapters")
    model_identity = round1_runtime._model_identity(model, "cage_qwen3")
    model_identity["dtqi_screen_execution_sha256"] = execution_sha
    model_identity["dtqi_runtime_sha256"] = execution["source_files"]["dtqi_runtime"]["sha256"]
    model_identity["dtqi_screen_runner_sha256"] = file_sha256(Path(__file__).resolve())
    completed = 0
    resumed = 0
    for index, case in enumerate(cases, 1):
        path = output_dir / "cases" / f"{case['case_id']}.json"
        if path.exists():
            _validate_record(load_json(path), case=case, identity=identity, model_identity=model_identity)
            resumed += 1
            print(f"[{index}/120] resume-valid {case['case_id']}", flush=True)
            continue
        print(f"[{index}/120] running l={case['input']['prompt_length']} document={case['input']['document_id']} anchor={case['input']['anchor_index']}", flush=True)
        try:
            scoring, cache, details = accepted_runtime._run_case(model, case)
            record = {
                "schema_version": 1,
                "status": "completed",
                "stage": STAGE,
                "case_id": case["case_id"],
                "identity": identity,
                "model": model_identity,
                "method": case["method"],
                "input": case["input"],
                "memory": accepted_runtime._memory_report(case["method"]),
                "scoring": scoring,
                "cache": cache,
                "resume": details["resume"],
                "runtime": details["runtime"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "representation_note": "fake-quant accuracy simulation with packed paper-estimate memory",
            }
            _validate_record(record, case=case, identity=identity, model_identity=model_identity)
            _write_atomic(path, record)
            completed += 1
            print(f"[{index}/120] completed {case['case_id']} mean_nll={scoring['mean_nll']:.9g}", flush=True)
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
    for case in cases:
        _validate_record(load_json(output_dir / "cases" / f"{case['case_id']}.json"), case=case, identity=identity, model_identity=model_identity)
    failures = list((output_dir / "failures").glob("*.json")) if (output_dir / "failures").exists() else []
    if failures:
        raise CageV4DTQIScreenRuntimeError("DTQI screen failure records remain")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "stage": STAGE,
        "expected_cases": 120,
        "completed_cases": 120,
        "new_cases": completed,
        "resumed_cases": resumed,
        "failure_records": 0,
        "identity": identity,
        "model": model_identity,
        "scientific_fields": list(SCIENTIFIC_FIELDS),
        "baseline_results_reused": True,
        "local_mse_computed": False,
        "interpretation_performed": False,
        "holdout_accessed": False,
        "pg19_test_accessed": False,
    }
    _write_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
