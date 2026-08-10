#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
from utils.qwen3_cage_v3_execution import canonical_sha256, load_json
from utils.qwen3_cage_v3_holdout import (
    HOLDOUT_ANCHORS,
    expand_holdout_cases,
    load_holdout_execution,
)
from utils.qwen3_cage_v3_protocol import file_sha256
from utils.qwen3_cage_v3_screen import SCIENTIFIC_FIELDS


SEED = 20260810
EXPECTED_KITTY_COMMIT = "dfd2c07b407d6b407179359207c612ab631f3ed1"
EXPECTED_TRANSFORMERS_COMMIT = "37f8b0b53512e6aae0cfd15746c133c101783178"


class CageV3HoldoutRuntimeError(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen CAGE-v3 local holdout partition")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("cage_qwen3", "kitty_qwen3"), required=True)
    parser.add_argument("--stage", choices=("holdout_acceptance", "holdout_full"), required=True)
    parser.add_argument("--acceptance-gate", type=Path)
    return parser.parse_args()


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_hashes() -> dict[str, str]:
    return {
        "runner": file_sha256(Path(__file__).resolve()),
        "comparator": file_sha256(REPO_ROOT / "scripts" / "qwen3_compare_cage_v3_holdout_acceptance.py"),
        "holdout_utils": file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v3_holdout.py"),
        "round1_runtime": file_sha256(REPO_ROOT / "scripts" / "qwen3_run_cage_v2_round1.py"),
        "qwen3_cage": file_sha256(REPO_ROOT / "models" / "qwen3_cage.py"),
        "qwen3_cage_v2": file_sha256(REPO_ROOT / "models" / "qwen3_cage_v2.py"),
        "cage_v2_quant": file_sha256(REPO_ROOT / "models" / "cage_v2_quant.py"),
        "cage_v2_memory": file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v2.py"),
        "recorder": file_sha256(REPO_ROOT / "utils" / "qwen3_perturbation_runtime.py"),
    }


def _shell_case_manifest_sha256(root: Path) -> str:
    lines = "".join(
        f"{file_sha256(path)}  cases/{path.name}\n"
        for path in sorted((root / "cases").glob("*.json"))
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _scientific_payload(root: Path, expected: int) -> tuple[str, list[dict[str, Any]]]:
    records = []
    for path in sorted((root / "cases").glob("*.json")):
        value = load_json(path)
        records.append({field: value[field] for field in SCIENTIFIC_FIELDS})
    if len(records) != expected:
        raise CageV3HoldoutRuntimeError("holdout acceptance artifact case count mismatch")
    records.sort(key=lambda record: record["case_id"])
    return canonical_sha256(records), records


def _validate_full_gate(
    gate_path: Path,
    *,
    execution: dict[str, Any],
    execution_sha256: str,
) -> dict[str, Any]:
    gate = load_json(gate_path)
    static = (
        gate.get("schema_version") == 1,
        gate.get("gate_id") == "qwen3-8b-cage-v3-holdout-acceptance-gate-v1",
        gate.get("status") == "pass",
        gate.get("claim_eligible") is False,
        gate.get("execution_sha256") == execution_sha256,
        gate.get("protocol_sha256") == execution["protocol"]["sha256"],
        gate.get("manifest_sha256") == execution["input_manifest"]["sha256"],
        gate.get("quota_plan_sha256") == execution["quota_plan"]["sha256"],
        gate.get("screen_decision_sha256") == execution["screen_decision"]["sha256"],
        gate.get("acceptance_source", {}).get("source_sha256") == _source_hashes(),
        gate.get("holdout_full_authorization", {}).get("anchor_indices") == HOLDOUT_ANCHORS,
        gate.get("holdout_full_authorization", {}).get("cage_case_count") == 60,
        gate.get("holdout_full_authorization", {}).get("kitty_case_count") == 30,
        gate.get("holdout_full_authorization", {}).get("selected_family_id") == "pure-sr2-sink32-calibrated",
        gate.get("holdout_full_authorization", {}).get("reserved_unseen_metrics") is False,
        gate.get("holdout_full_authorization", {}).get("end_to_end_quality") is False,
    )
    if not all(static):
        raise CageV3HoldoutRuntimeError("holdout acceptance gate static identity is invalid")
    accepted_commit = gate["acceptance_source"].get("git_commit")
    for partition, count in (("cage_qwen3", 6), ("kitty_qwen3", 3)):
        partition_gate = gate["partitions"][partition]
        repeats = []
        for name in ("repeat_a", "repeat_b"):
            artifact = partition_gate[name]
            root = Path(artifact["output_dir"])
            identity_path = root / "run_identity.json"
            summary_path = root / "summary.json"
            log_path = Path(artifact["execution_log"])
            checks = (
                file_sha256(identity_path) == artifact["run_identity_sha256"],
                file_sha256(summary_path) == artifact["summary_sha256"],
                file_sha256(log_path) == artifact["execution_log_sha256"],
                log_path.stat().st_size == artifact["execution_log_size_bytes"],
                _shell_case_manifest_sha256(root) == artifact["case_file_manifest_sha256"],
            )
            if not all(checks):
                raise CageV3HoldoutRuntimeError(f"holdout gate artifact mismatch: {partition}/{name}")
            identity = load_json(identity_path)
            summary = load_json(summary_path)
            source = identity.get("source_state", {})
            if source.get("git_commit") != accepted_commit or source.get("dirty") is not False:
                raise CageV3HoldoutRuntimeError(f"holdout gate source mismatch: {partition}/{name}")
            if partition == "kitty_qwen3" and (
                source.get("kitty_commit") != EXPECTED_KITTY_COMMIT
                or source.get("transformers_commit") != EXPECTED_TRANSFORMERS_COMMIT
                or source.get("kitty_dirty") is not False
            ):
                raise CageV3HoldoutRuntimeError(f"holdout Kitty source mismatch: {name}")
            expected_summary = (
                summary.get("status") == "pass",
                summary.get("claim_eligible") is False,
                summary.get("partition") == partition,
                summary.get("stage") == "holdout_acceptance",
                summary.get("expected_cases") == count,
                summary.get("completed_cases") == count,
                summary.get("new_cases") == count,
                summary.get("resumed_cases") == 0,
                summary.get("failure_records") == 0,
                summary.get("identity") == identity,
                summary.get("model", {}).get("source_sha256") == _source_hashes(),
            )
            if not all(expected_summary):
                raise CageV3HoldoutRuntimeError(f"holdout gate summary mismatch: {partition}/{name}")
            payload_sha, payload = _scientific_payload(root, count)
            if payload_sha != partition_gate["scientific_payload_sha256"]:
                raise CageV3HoldoutRuntimeError(f"holdout gate payload mismatch: {partition}/{name}")
            repeats.append(payload)
        if repeats[0] != repeats[1]:
            raise CageV3HoldoutRuntimeError(f"holdout repeats differ: {partition}")
        comparison = partition_gate["comparison"]
        comparison_path = Path(comparison["path"])
        if file_sha256(comparison_path) != comparison["sha256"]:
            raise CageV3HoldoutRuntimeError(f"holdout comparison hash mismatch: {partition}")
        value = load_json(comparison_path)
        if (
            value.get("status") != "pass"
            or value.get("partition") != partition
            or value.get("mismatch_case_ids") != []
            or value.get("scientific_payload_sha256") != partition_gate["scientific_payload_sha256"]
            or value.get("comparator_sha256") != _source_hashes()["comparator"]
        ):
            raise CageV3HoldoutRuntimeError(f"holdout comparison mismatch: {partition}")
    return gate


def _model_identity(model: Any, partition: str) -> dict[str, Any]:
    identity = round1_runtime._model_identity(model, partition)
    identity["source_sha256"] = _source_hashes()
    return identity


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise CageV3HoldoutRuntimeError("CAGE-v3 holdout requires CUDA")
    execution, execution_sha256, protocol, manifest, plan, decision = load_holdout_execution(
        args.execution.resolve(), repo_root=REPO_ROOT, verify_artifacts=True
    )
    source_state = round1_runtime._source_state(args.partition)
    if source_state["dirty"] or source_state.get("kitty_dirty", False):
        raise CageV3HoldoutRuntimeError("CAGE-v3 holdout requires clean sources")
    if args.partition == "kitty_qwen3" and (
        source_state.get("kitty_commit") != EXPECTED_KITTY_COMMIT
        or source_state.get("transformers_commit") != EXPECTED_TRANSFORMERS_COMMIT
    ):
        raise CageV3HoldoutRuntimeError("Kitty provenance differs from holdout freeze")
    if args.stage == "holdout_full":
        if args.acceptance_gate is None:
            raise CageV3HoldoutRuntimeError("holdout_full requires joint acceptance gate")
        _validate_full_gate(args.acceptance_gate.resolve(), execution=execution, execution_sha256=execution_sha256)
    elif args.acceptance_gate is not None:
        raise CageV3HoldoutRuntimeError("holdout_acceptance must not consume a gate")
    cases = expand_holdout_cases(
        execution=execution,
        execution_sha256=execution_sha256,
        protocol=protocol,
        manifest=manifest,
        plan=plan,
        partition=args.partition,
        stage=args.stage,
    )
    run_identity = {
        "schema_version": 1,
        "experiment": "qwen3_cage_v3_holdout_development_only",
        "claim_eligible": False,
        "partition": args.partition,
        "stage": args.stage,
        "execution_sha256": execution_sha256,
        "protocol_sha256": execution["protocol"]["sha256"],
        "manifest_sha256": execution["input_manifest"]["sha256"],
        "quota_plan_sha256": execution["quota_plan"]["sha256"],
        "screen_decision_sha256": execution["screen_decision"]["sha256"],
        "selected_family_id": decision["decision"]["selected_family_id"],
        "source_state": source_state,
        "expected_case_ids": [case["case_id"] for case in cases],
    }
    output_dir = args.output_dir.resolve()
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if load_json(identity_path) != run_identity:
            raise CageV3HoldoutRuntimeError("existing holdout run identity differs")
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
        raise CageV3HoldoutRuntimeError("expected 36 Qwen3 attention adapters")
    recorder = round1_runtime.Qwen3PerturbationRecorder(expected_layers=36, top_k=10)
    if recorder.install(model) != 36:
        raise CageV3HoldoutRuntimeError("expected 36 holdout perturbation callbacks")
    model_identity = _model_identity(model, args.partition)
    completed = 0
    resumed = 0
    for index, case in enumerate(cases, 1):
        path = output_dir / "cases" / f"{case['case_id']}.json"
        if path.exists():
            round1_runtime._validate_record(load_json(path), case=case, run_identity=run_identity, model_identity=model_identity)
            resumed += 1
            print(f"[{index}/{len(cases)}] resume-valid {case['case_id']}", flush=True)
            continue
        print(f"[{index}/{len(cases)}] running {case['method']['id']} l={case['input']['prompt_length']} a={case['input']['anchor_index']}", flush=True)
        try:
            layers, runtime = round1_runtime._run_case(model, recorder, case)
            aggregates = round1_runtime.aggregate_layer_metrics(layers)
            record = {
                "schema_version": 1,
                "status": "completed",
                "case_id": case["case_id"],
                "identity": run_identity,
                "model": model_identity,
                "method": case["method"],
                "input": case["input"],
                "memory": round1_runtime._memory_report(case["method"]),
                "layer_metrics": layers,
                "aggregates": aggregates,
                "cache": runtime.pop("cache"),
                "runtime": runtime,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "representation_note": "claim-ineligible CAGE-v3 holdout fake quantization plus packed paper estimate",
            }
            round1_runtime._validate_record(record, case=case, run_identity=run_identity, model_identity=model_identity)
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
        round1_runtime._validate_record(record, case=case, run_identity=run_identity, model_identity=model_identity)
        records.append(record)
    failure_count = len(list((output_dir / "failures").glob("*.json"))) if (output_dir / "failures").exists() else 0
    if failure_count:
        raise CageV3HoldoutRuntimeError("holdout failure records remain")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "partition": args.partition,
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
