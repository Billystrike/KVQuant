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
import scripts.qwen3_run_cage_v4_metric_acceptance as accepted_runtime
from utils.qwen3_cage_v4_acceptance import call_with_recorder_uninstalled
from utils.qwen3_cage_v4_data import file_sha256
from utils.qwen3_cage_v4_execution import load_screen_execution
from utils.qwen3_cage_v4_gate import EXPECTED_KITTY_COMMIT, EXPECTED_TRANSFORMERS_COMMIT
from utils.qwen3_formal import formal_scoring
from utils.qwen3_perturbation_protocol import validate_aggregates, validate_layer_records


SEED = 20260810
PARTITIONS = ("cage_qwen3", "kitty_qwen3")


class CageV4ScreenRuntimeError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CageV4ScreenRuntimeError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_record(
    record: dict[str, Any],
    *,
    case: dict[str, Any],
    run_identity: dict[str, Any],
    model_identity: dict[str, Any],
) -> None:
    checks = (
        record.get("schema_version") == 1,
        record.get("status") == "completed",
        record.get("case_id") == case["case_id"],
        record.get("partition") == run_identity["partition"],
        record.get("stage") == "screen_full",
        record.get("identity") == run_identity,
        record.get("model") == model_identity,
        record.get("method") == case["method"],
        record.get("input") == case["input"],
    )
    if not all(checks):
        raise CageV4ScreenRuntimeError("screen result identity/schema mismatch")
    expected_scoring = formal_scoring(record.get("scoring", {}).get("token_nlls", []))
    if record.get("scoring") != expected_scoring:
        raise CageV4ScreenRuntimeError("screen NLL scoring mismatch")
    if record.get("memory") != accepted_runtime._memory_report(case["method"]):
        raise CageV4ScreenRuntimeError("screen memory report mismatch")
    expected_length = case["input"]["prompt_length"] + 63
    quality_cache = record.get("quality_cache")
    quality_checks = (
        isinstance(quality_cache, dict),
        quality_cache.get("reported_seq_length") == expected_length
        if isinstance(quality_cache, dict)
        else False,
        quality_cache.get("expected_seq_length") == expected_length
        if isinstance(quality_cache, dict)
        else False,
        quality_cache.get("layer_count") == 36 if isinstance(quality_cache, dict) else False,
        quality_cache.get("tensors_finite") is True if isinstance(quality_cache, dict) else False,
    )
    if not all(quality_checks):
        raise CageV4ScreenRuntimeError("screen quality-cache diagnostics mismatch")
    local = record.get("local_perturbation")
    if case["method"]["metric_method_id"] == "fp16":
        if local is not None:
            raise CageV4ScreenRuntimeError("FP16 screen result must not carry perturbation")
    else:
        if not isinstance(local, dict) or local.get("metric") != "joint_post_o_proj_mse":
            raise CageV4ScreenRuntimeError("compressed screen result lacks local perturbation")
        validate_layer_records(local.get("layer_metrics", []))
        validate_aggregates(local.get("aggregates", {}), local["layer_metrics"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen PG-19 metric screen partition")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise CageV4ScreenRuntimeError("metric screen requires CUDA")
    execution_path = args.execution.resolve()
    execution, execution_sha256, protocol, protocol_sha256, _, expanded = load_screen_execution(
        execution_path,
        repo_root=REPO_ROOT,
        verify_artifacts=True,
    )
    if expanded is None:
        raise CageV4ScreenRuntimeError("screen inputs were not expanded")
    cases = expanded[args.partition]
    source_state = round1_runtime._source_state(args.partition)
    if source_state["dirty"] or source_state.get("kitty_dirty", False):
        raise CageV4ScreenRuntimeError("metric screen requires clean sources")
    runtime_dependency_state = {
        "kitty_commit": round1_runtime._git(round1_runtime.KITTY_ROOT, "rev-parse", "HEAD"),
        "kitty_dirty": bool(
            round1_runtime._git(
                round1_runtime.KITTY_ROOT,
                "status",
                "--porcelain",
                "--untracked-files=all",
            )
        ),
        "transformers_commit": round1_runtime._git(
            round1_runtime.KITTY_ROOT / "third_party" / "transformers",
            "rev-parse",
            "HEAD",
        ),
    }
    if (
        runtime_dependency_state["kitty_commit"] != EXPECTED_KITTY_COMMIT
        or runtime_dependency_state["transformers_commit"] != EXPECTED_TRANSFORMERS_COMMIT
        or runtime_dependency_state["kitty_dirty"]
    ):
        raise CageV4ScreenRuntimeError("frozen Kitty/Transformers runtime dependency changed")
    runner_sha256 = file_sha256(Path(__file__).resolve())
    accepted_runner_sha256 = file_sha256(
        REPO_ROOT / "scripts" / "qwen3_run_cage_v4_metric_acceptance.py"
    )
    run_identity = {
        "schema_version": 1,
        "experiment": "qwen3_cage_v4_pg19_metric_validity_screen",
        "claim_eligible": False,
        "partition": args.partition,
        "stage": "screen_full",
        "execution_sha256": execution_sha256,
        "protocol_sha256": protocol_sha256,
        "acceptance_gate_sha256": execution["acceptance_gate"]["sha256"],
        "input_manifest_sha256": execution["input_manifest"]["sha256"],
        "source_state": source_state,
        "runtime_dependency_state": runtime_dependency_state,
        "runner_sha256": runner_sha256,
        "accepted_runtime_sha256": accepted_runner_sha256,
        "expected_case_ids": [case["case_id"] for case in cases],
    }
    output_dir = args.output_dir.resolve()
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if _load(identity_path) != run_identity:
            raise CageV4ScreenRuntimeError("existing screen identity differs")
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
        raise CageV4ScreenRuntimeError("expected 36 Qwen3 attention adapters")
    recorder = round1_runtime.Qwen3PerturbationRecorder(expected_layers=36, top_k=10)
    if recorder.install(model) != 36:
        raise CageV4ScreenRuntimeError("expected 36 perturbation callbacks")
    model_identity = round1_runtime._model_identity(model, args.partition)
    model_identity["metric_screen_runner_sha256"] = runner_sha256
    model_identity["accepted_runtime_sha256"] = accepted_runner_sha256
    completed = 0
    resumed = 0
    for index, case in enumerate(cases, 1):
        path = output_dir / "cases" / f"{case['case_id']}.json"
        if path.exists():
            _validate_record(
                _load(path), case=case, run_identity=run_identity, model_identity=model_identity
            )
            resumed += 1
            print(f"[{index}/{len(cases)}] resume-valid {case['case_id']}", flush=True)
            continue
        print(
            f"[{index}/{len(cases)}] running {case['method']['id']} "
            f"document={case['input']['document_id']} anchor={case['input']['anchor_index']}",
            flush=True,
        )
        try:
            local, local_runtime = accepted_runtime._local_case(
                model, recorder, case, args.partition
            )
            scoring, quality_runtime = call_with_recorder_uninstalled(
                recorder=recorder,
                model=model,
                callback=lambda: accepted_runtime._quality_case(model, case),
                expected_modules=36,
            )
            record = {
                "schema_version": 1,
                "status": "completed",
                "case_id": case["case_id"],
                "partition": args.partition,
                "stage": "screen_full",
                "identity": run_identity,
                "model": model_identity,
                "method": case["method"],
                "input": case["input"],
                "memory": accepted_runtime._memory_report(case["method"]),
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
        record = _load(output_dir / "cases" / f"{case['case_id']}.json")
        _validate_record(record, case=case, run_identity=run_identity, model_identity=model_identity)
        records.append(record)
    failures = list((output_dir / "failures").glob("*.json")) if (output_dir / "failures").exists() else []
    if failures:
        raise CageV4ScreenRuntimeError("screen failure records remain")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "partition": args.partition,
        "stage": "screen_full",
        "expected_cases": len(cases),
        "completed_cases": len(records),
        "new_cases": completed,
        "resumed_cases": resumed,
        "failure_records": 0,
        "run_identity": run_identity,
        "model": model_identity,
    }
    _write_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
    del model, recorder
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
