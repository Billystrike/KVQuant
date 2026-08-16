#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM

import scripts.qwen3_run_cage_v2_round1 as round1_runtime
import scripts.qwen3_run_cage_v4_metric_acceptance as quality_runtime
from utils.qwen3_cage_v3_promotion_acceptance import PARTITIONS
from utils.qwen3_cage_v3_promotion_full import (
    EXPECTED_CASE_ID_HASHES,
    expand_full_holdout_cases,
    load_full_execution,
)
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256, load_data_protocol, validate_input_manifest
from utils.qwen3_formal import formal_scoring


SEED = 20260816
SCIENTIFIC_FIELDS = ("case_id", "partition", "method", "input", "memory", "scoring", "quality_cache")


class CageV3PromotionFullRuntimeError(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen CAGE-v3 promotion full development holdout")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CageV3PromotionFullRuntimeError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_hashes() -> dict[str, str]:
    return {
        "runner": file_sha256(Path(__file__).resolve()),
        "full_utils": file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v3_promotion_full.py"),
        "gate_utils": file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v3_promotion_gate.py"),
        "quality_runtime": file_sha256(REPO_ROOT / "scripts" / "qwen3_run_cage_v4_metric_acceptance.py"),
        "round1_runtime": file_sha256(REPO_ROOT / "scripts" / "qwen3_run_cage_v2_round1.py"),
    }


def _validate_record(record: dict[str, Any], *, case: dict[str, Any], run_identity: dict[str, Any], model_identity: dict[str, Any]) -> None:
    checks = (
        record.get("schema_version") == 1,
        record.get("status") == "completed",
        record.get("case_id") == case["case_id"],
        record.get("partition") == run_identity["partition"],
        record.get("stage") == "promotion_full_holdout",
        record.get("claim_eligible") is False,
        record.get("identity") == run_identity,
        record.get("model") == model_identity,
        record.get("method") == case["method"],
        record.get("input") == case["input"],
    )
    if not all(checks):
        raise CageV3PromotionFullRuntimeError("promotion full identity/schema mismatch")
    token_nlls = record.get("scoring", {}).get("token_nlls", [])
    if len(token_nlls) != 64 or record.get("scoring") != formal_scoring(token_nlls):
        raise CageV3PromotionFullRuntimeError("promotion full NLL scoring mismatch")
    if record.get("memory") != quality_runtime._memory_report(case["method"]):
        raise CageV3PromotionFullRuntimeError("promotion full memory report mismatch")
    diagnostics = record.get("quality_cache", {})
    expected_length = case["input"]["prompt_length"] + 63
    if not (
        diagnostics.get("reported_seq_length") == expected_length
        and diagnostics.get("expected_seq_length") == expected_length
        and diagnostics.get("layer_count") == 36
        and diagnostics.get("tensors_finite") is True
    ):
        raise CageV3PromotionFullRuntimeError("promotion full cache diagnostics mismatch")


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise CageV3PromotionFullRuntimeError("promotion full execution requires CUDA")
    execution_path = args.execution.resolve()
    execution, execution_sha256, protocol, protocol_sha256, gate, gate_sha256 = load_full_execution(
        execution_path,
        repo_root=REPO_ROOT,
        verify_server_artifacts=True,
    )
    manifest_path = args.input_manifest.resolve()
    if str(manifest_path) != execution["input_manifest"]["path"] or file_sha256(manifest_path) != execution["input_manifest"]["sha256"]:
        raise CageV3PromotionFullRuntimeError("promotion full input manifest mismatch")
    manifest = _load(manifest_path)
    data_path = REPO_ROOT / protocol["input_receipt"]["data_protocol_path"]
    data_protocol, data_sha256 = load_data_protocol(data_path)
    if data_sha256 != protocol["input_receipt"]["data_protocol_sha256"]:
        raise CageV3PromotionFullRuntimeError("promotion full data protocol mismatch")
    validate_input_manifest(manifest, protocol=data_protocol, protocol_sha256=data_sha256)
    source_state = round1_runtime._source_state(args.partition)
    if source_state["dirty"] or source_state.get("kitty_dirty", False):
        raise CageV3PromotionFullRuntimeError("promotion full execution requires clean sources")
    cases = expand_full_holdout_cases(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        gate_sha256=gate_sha256,
        manifest=manifest,
        partition=args.partition,
        repo_root=REPO_ROOT,
    )
    if canonical_sha256([case["case_id"] for case in cases]) != EXPECTED_CASE_ID_HASHES[args.partition]:
        raise CageV3PromotionFullRuntimeError("promotion full case-ID receipt mismatch")
    if Counter(case["method"]["metric_method_id"] for case in cases) != Counter(execution["partitions"][args.partition]["method_counts"]):
        raise CageV3PromotionFullRuntimeError("promotion full method-count receipt mismatch")
    run_identity = {
        "schema_version": 1,
        "experiment": "qwen3_cage_v3_promotion_full_holdout",
        "claim_eligible": False,
        "partition": args.partition,
        "stage": "promotion_full_holdout",
        "execution_sha256": execution_sha256,
        "gate_sha256": gate_sha256,
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": file_sha256(manifest_path),
        "source_state": source_state,
        "source_sha256": _source_hashes(),
        "expected_case_ids": [case["case_id"] for case in cases],
        "holdout_interpretation_authorized": False,
        "pg19_test_accessed": False,
        "llama2_execution_authorized": False,
        "kitty_llama_port_authorized": False,
    }
    output_dir = args.output_dir.resolve()
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if _load(identity_path) != run_identity:
            raise CageV3PromotionFullRuntimeError("existing promotion full run identity differs")
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
    if round1_runtime.install_qwen3_cage_attention(model) != 36:
        raise CageV3PromotionFullRuntimeError("expected 36 Qwen3 attention adapters")
    model_identity = round1_runtime._model_identity(model, args.partition)
    model_identity["promotion_full_source_sha256"] = _source_hashes()
    completed = 0
    resumed = 0
    for index, case in enumerate(cases, 1):
        path = output_dir / "cases" / f"{case['case_id']}.json"
        if path.exists():
            _validate_record(_load(path), case=case, run_identity=run_identity, model_identity=model_identity)
            resumed += 1
            print(f"[{index}/{len(cases)}] resume-valid {case['case_id']}", flush=True)
            continue
        print(f"[{index}/{len(cases)}] running {case['method']['id']} input={case['input']['case_id'] if 'case_id' in case['input'] else case['case_id']}", flush=True)
        try:
            scoring, result = quality_runtime._quality_case(model, case)
            record = {
                "schema_version": 1,
                "status": "completed",
                "claim_eligible": False,
                "case_id": case["case_id"],
                "partition": args.partition,
                "stage": "promotion_full_holdout",
                "identity": run_identity,
                "model": model_identity,
                "method": case["method"],
                "input": case["input"],
                "memory": quality_runtime._memory_report(case["method"]),
                "scoring": scoring,
                "quality_cache": result.pop("cache"),
                "runtime": result["runtime"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "representation_note": "accuracy simulation with complete logical packed paper-estimate memory",
            }
            _validate_record(record, case=case, run_identity=run_identity, model_identity=model_identity)
            _write_atomic(path, record)
            completed += 1
            print(f"[{index}/{len(cases)}] completed {case['case_id']} mean_nll={scoring['mean_nll']:.9g}", flush=True)
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
        raise CageV3PromotionFullRuntimeError("promotion full failure records remain")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "partition": args.partition,
        "stage": "promotion_full_holdout",
        "expected_cases": len(cases),
        "completed_cases": len(records),
        "new_cases": completed,
        "resumed_cases": resumed,
        "failure_records": 0,
        "holdout_interpretation_authorized": False,
        "pg19_test_accessed": False,
        "llama2_execution_authorized": False,
        "kitty_llama_port_authorized": False,
        "scientific_fields": list(SCIENTIFIC_FIELDS),
        "run_identity": run_identity,
        "model": model_identity,
    }
    _write_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
