#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_formal import (
    atomic_write_json,
    expand_partition_cases,
    load_execution_config,
    load_formal_protocol,
    validate_input_manifest,
)
from utils.qwen3_memory import (
    estimate_qwen3_cage_bytes,
    estimate_qwen3_fp16_bytes,
    estimate_qwen3_kivi_bytes,
    estimate_qwen3_kitty_bytes,
)
from utils.qwen3_perturbation_protocol import (
    Qwen3PerturbationError,
    aggregate_layer_metrics,
    file_sha256,
    load_perturbation_acceptance_gate,
    load_perturbation_protocol,
    perturbation_case_id,
    scientific_payload_sha256,
    validate_aggregates,
    validate_layer_records,
)


ARTIFACT_SET_ID = "qwen3-8b-cage-kitty-memory-perturbation-full-artifacts-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently validate both Qwen3 full perturbation partitions"
    )
    parser.add_argument("--perturbation-protocol", type=Path, required=True)
    parser.add_argument("--acceptance-gate", type=Path, required=True)
    parser.add_argument("--quality-protocol", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3PerturbationError(f"cannot load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise Qwen3PerturbationError(f"{label} must be a JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_artifact_manifest(
    manifest: dict[str, Any], *, protocol_sha256: str, gate_sha256: str
) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("artifact_set_id") != ARTIFACT_SET_ID:
        raise Qwen3PerturbationError("full artifact manifest identity mismatch")
    if manifest.get("status") != "declared_from_execution_outputs_before_joint_postrun_audit":
        raise Qwen3PerturbationError("full artifact manifest status mismatch")
    if manifest.get("source_commit") != "420a1f53abbaa2ca49df7ba64650ea90e50e9983":
        raise Qwen3PerturbationError("full artifact source commit mismatch")
    if manifest.get("perturbation_protocol_sha256") != protocol_sha256:
        raise Qwen3PerturbationError("full artifact protocol hash mismatch")
    if manifest.get("acceptance_gate_sha256") != gate_sha256:
        raise Qwen3PerturbationError("full artifact gate hash mismatch")
    partitions = manifest.get("partitions", {})
    if set(partitions) != {"cage_qwen3", "kitty_qwen3"}:
        raise Qwen3PerturbationError("full artifact partitions mismatch")
    for partition, count in (("cage_qwen3", 1000), ("kitty_qwen3", 300)):
        record = partitions[partition]
        if record.get("expected_cases") != count:
            raise Qwen3PerturbationError(f"{partition} artifact case count mismatch")
        for name in (
            "run_identity_sha256",
            "summary_sha256",
            "shell_case_file_manifest_sha256",
            "execution_log_sha256",
        ):
            value = record.get(name)
            if not isinstance(value, str) or len(value) != 64:
                raise Qwen3PerturbationError(f"{partition}.{name} is not a SHA-256")
        if not isinstance(record.get("execution_log_size_bytes"), int):
            raise Qwen3PerturbationError(f"{partition} log size is invalid")
    expected = manifest.get("joint_expected", {})
    if expected != {
        "case_count": 1300,
        "layer_records": 46800,
        "failure_count": 0,
        "interpretation_allowed_before_audit_pass": False,
    }:
        raise Qwen3PerturbationError("joint full artifact expectations mismatch")


def _expected_memory(method: dict[str, Any], prompt_length: int) -> dict[str, Any]:
    name = method["name"]
    config = method["config"]
    if name == "fp16":
        return estimate_qwen3_fp16_bytes(seq_len=prompt_length)
    if name == "cage":
        return estimate_qwen3_cage_bytes(
            seq_len=prompt_length,
            residual_length=config["residual_length"],
            key_group_sizes=config["key_group_sizes"],
            value_group_sizes=config["value_group_sizes"],
            bits=config["bits"],
        )
    if name == "kivi":
        return estimate_qwen3_kivi_bytes(
            seq_len=prompt_length,
            group_size=config["group_size"],
            residual_length=config["residual_length"],
            bits=config["bits"],
        )
    if name == "kitty":
        return estimate_qwen3_kitty_bytes(
            seq_len=prompt_length,
            boosted_channels=config["boosted_channels"],
        )
    raise Qwen3PerturbationError(f"unsupported method for memory validation: {name}")


def _validate_cache(record: dict[str, Any], case: dict[str, Any]) -> None:
    cache = record.get("cache")
    if not isinstance(cache, dict):
        raise Qwen3PerturbationError("completed case lacks cache diagnostics")
    expected_length = case["input"]["prompt_length"] + 1
    if (
        cache.get("reported_seq_length") != expected_length
        or cache.get("expected_seq_length") != expected_length
        or cache.get("layer_count") != 36
        or cache.get("tensor_dtypes") != ["torch.float16"]
        or cache.get("tensors_finite") is not True
    ):
        raise Qwen3PerturbationError("completed case cache diagnostics mismatch")
    method = case["method"]
    if method["name"] in {"cage", "kivi"}:
        residual = method["config"]["residual_length"]
        expected_key = expected_length - expected_length % residual
        expected_value = max(0, expected_length - residual)
        if cache.get("key_quantized_lengths") != [expected_key]:
            raise Qwen3PerturbationError("completed case Key cache mechanics mismatch")
        if cache.get("value_quantized_lengths") != [expected_value]:
            raise Qwen3PerturbationError("completed case Value cache mechanics mismatch")
    elif method["name"] == "kitty":
        boosted = method["config"]["boosted_channels"]
        expected_config = {
            "sink_length": 32,
            "buffer_length": 128,
            "group_size": 128,
            "kbits": 2,
            "vbits": 2,
            "promote_ratio": boosted / 128,
            "promote_bit": 4,
            "channel_selection": 1,
            "VCache_BitDecoding": False,
            "PostQuant": True,
        }
        if cache.get("official_config") != expected_config:
            raise Qwen3PerturbationError("completed Kitty config mismatch")


def _validate_case(
    record: dict[str, Any],
    *,
    case: dict[str, Any],
    run_identity: dict[str, Any],
    model_identity: dict[str, Any],
    measurement: dict[str, Any],
) -> None:
    if record.get("schema_version") != 1 or record.get("status") != "completed":
        raise Qwen3PerturbationError("completed case schema/status mismatch")
    if record.get("case_id") != case["perturbation_case_id"]:
        raise Qwen3PerturbationError("completed case ID mismatch")
    if record.get("base_quality_case_id") != case["case_id"]:
        raise Qwen3PerturbationError("completed base quality case ID mismatch")
    if record.get("partition") != case["partition"] or record.get("stage") != "full":
        raise Qwen3PerturbationError("completed case partition/stage mismatch")
    if record.get("identity") != run_identity or record.get("model") != model_identity:
        raise Qwen3PerturbationError("completed case identity mismatch")
    if record.get("method") != case["method"] or record.get("input") != case["input"]:
        raise Qwen3PerturbationError("completed case method/input mismatch")
    if record.get("measurement") != measurement:
        raise Qwen3PerturbationError("completed case measurement mismatch")
    if record.get("memory") != _expected_memory(
        case["method"], case["input"]["prompt_length"]
    ):
        raise Qwen3PerturbationError("completed case packed-memory report mismatch")
    layers = record.get("layer_metrics", [])
    validate_layer_records(layers)
    validate_aggregates(record.get("aggregates", {}), layers)
    if aggregate_layer_metrics(layers) != record["aggregates"]:
        raise Qwen3PerturbationError("completed case aggregate exact recomputation mismatch")
    if case["method"]["name"] == "fp16":
        for layer in layers:
            for name, value in layer["metrics"].items():
                expected = 1.0 if name == "topk_attention_overlap" else 0.0
                if float(value) != expected:
                    raise Qwen3PerturbationError("FP16 local identity mismatch")
    else:
        if record["aggregates"]["relative_k_reconstruction_error"]["maximum"] <= 0:
            raise Qwen3PerturbationError("quantized case has no Key perturbation")
        if record["aggregates"]["relative_v_reconstruction_error"]["maximum"] <= 0:
            raise Qwen3PerturbationError("quantized case has no Value perturbation")
    _validate_cache(record, case)
    runtime = record.get("runtime", {})
    for name in ("elapsed_seconds", "cuda_max_allocated_bytes", "cuda_max_reserved_bytes"):
        value = runtime.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise Qwen3PerturbationError(f"completed case runtime {name} is invalid")


def _case_manifest(case_paths: list[Path]) -> tuple[str, str]:
    records = [
        {"filename": path.name, "sha256": file_sha256(path)} for path in case_paths
    ]
    canonical = _canonical_sha256(records)
    shell_lines = "".join(
        f"{record['sha256']}  {path}\n" for record, path in zip(records, case_paths)
    ).encode("utf-8")
    shell = hashlib.sha256(shell_lines).hexdigest()
    return canonical, shell


def _validate_log(path: Path, *, expected_count: int, partition: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    running = len(re.findall(r"^\[\d+/\d+\] running ", text, flags=re.MULTILINE))
    completed = len(re.findall(r"^\[\d+/\d+\] completed ", text, flags=re.MULTILINE))
    resumed = len(re.findall(r"^\[\d+/\d+\] resume-valid ", text, flags=re.MULTILINE))
    end_marker = (
        "=== QWEN3 PERTURBATION CAGE FULL END ==="
        if partition == "cage_qwen3"
        else "=== QWEN3 PERTURBATION KITTY FULL END ==="
    )
    checks = {
        "nonempty": bool(text),
        "no_traceback": "Traceback (most recent call last)" not in text,
        "no_conda_error": "ERROR conda.cli" not in text,
        "end_marker": end_marker in text,
        "running_lines": running == expected_count,
        "completed_lines": completed == expected_count,
        "resume_lines": resumed == 0,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise Qwen3PerturbationError(
            f"{partition} execution log checks failed: {failures}"
        )
    return {"checks": checks, "running": running, "completed": completed, "resumed": resumed}


def _validate_partition(
    *,
    partition: str,
    artifact: dict[str, Any],
    quality: dict[str, Any],
    execution: dict[str, Any],
    execution_sha256: str,
    input_manifest: dict[str, Any],
    perturbation: dict[str, Any],
    perturbation_sha256: str,
    gate_sha256: str,
    input_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    output_dir = Path(artifact["output_dir"])
    identity_path = output_dir / "run_identity.json"
    summary_path = output_dir / "summary.json"
    log_path = Path(artifact["execution_log"])
    for label, path, expected_sha in (
        ("run identity", identity_path, artifact["run_identity_sha256"]),
        ("summary", summary_path, artifact["summary_sha256"]),
        ("execution log", log_path, artifact["execution_log_sha256"]),
    ):
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise Qwen3PerturbationError(f"{partition} {label} hash mismatch")
    if log_path.stat().st_size != artifact["execution_log_size_bytes"]:
        raise Qwen3PerturbationError(f"{partition} execution log size mismatch")

    base_cases = expand_partition_cases(
        protocol=quality,
        execution=execution,
        execution_sha256=execution_sha256,
        input_manifest=input_manifest,
        partition=partition,
        stage="full",
    )
    cases = []
    for base in base_cases:
        case = dict(base)
        case["perturbation_case_id"] = perturbation_case_id(
            base_case_id=case["case_id"],
            perturbation_protocol_sha256=perturbation_sha256,
        )
        cases.append(case)
    if len(cases) != artifact["expected_cases"]:
        raise Qwen3PerturbationError(f"{partition} expanded case count mismatch")

    expected_source = {"git_commit": source_commit, "dirty": False}
    if partition == "kitty_qwen3":
        expected_source.update(
            kitty_commit="dfd2c07b407d6b407179359207c612ab631f3ed1",
            kitty_dirty=False,
            transformers_commit="37f8b0b53512e6aae0cfd15746c133c101783178",
        )
    expected_identity = {
        "schema_version": 1,
        "partition": partition,
        "stage": "full",
        "perturbation_protocol_id": perturbation["protocol_id"],
        "perturbation_protocol_sha256": perturbation_sha256,
        "quality_protocol_sha256": perturbation["inherited_quality_protocol"]["sha256"],
        "quality_execution_sha256": execution_sha256,
        "input_manifest_sha256": input_sha256,
        "source_state": expected_source,
        "expected_case_ids": [case["perturbation_case_id"] for case in cases],
        "acceptance_gate_sha256": gate_sha256,
    }
    run_identity = _load_json(identity_path, f"{partition} run identity")
    if run_identity != expected_identity:
        raise Qwen3PerturbationError(f"{partition} run identity content mismatch")
    summary = _load_json(summary_path, f"{partition} summary")
    expected_case_ids_sha = hashlib.sha256(
        json.dumps(
            expected_identity["expected_case_ids"], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "pass"
        or summary.get("partition") != partition
        or summary.get("stage") != "full"
        or summary.get("expected_cases") != len(cases)
        or summary.get("completed_cases") != len(cases)
        or summary.get("new_cases") != len(cases)
        or summary.get("resumed_cases") != 0
        or summary.get("failure_records") != 0
        or summary.get("identity") != run_identity
        or summary.get("case_ids_sha256") != expected_case_ids_sha
    ):
        raise Qwen3PerturbationError(f"{partition} summary content mismatch")
    model_identity = summary.get("model")
    model_checks = model_identity.get("checks", {}) if isinstance(model_identity, dict) else {}
    if not model_checks or not all(model_checks.values()):
        raise Qwen3PerturbationError(f"{partition} model identity checks failed")

    case_dir = output_dir / "cases"
    case_paths = sorted(case_dir.glob("*.json"))
    expected_paths = sorted(
        case_dir / f"{case['perturbation_case_id']}.json" for case in cases
    )
    if case_paths != expected_paths:
        raise Qwen3PerturbationError(f"{partition} case file set mismatch")
    case_lookup = {case["perturbation_case_id"]: case for case in cases}
    for path in case_paths:
        record = _load_json(path, f"{partition} completed case")
        _validate_case(
            record,
            case=case_lookup[path.stem],
            run_identity=run_identity,
            model_identity=model_identity,
            measurement=perturbation["measurement"],
        )

    canonical_manifest_sha, shell_manifest_sha = _case_manifest(case_paths)
    if shell_manifest_sha != artifact["shell_case_file_manifest_sha256"]:
        raise Qwen3PerturbationError(f"{partition} shell case-file manifest mismatch")
    scientific_sha, _ = scientific_payload_sha256(
        output_dir, expected_case_count=len(cases)
    )
    log_report = _validate_log(
        log_path, expected_count=len(cases), partition=partition
    )
    return {
        "case_count": len(cases),
        "layer_record_count": len(cases) * 36,
        "failure_count": 0,
        "run_identity_sha256": artifact["run_identity_sha256"],
        "summary_sha256": artifact["summary_sha256"],
        "execution_log_sha256": artifact["execution_log_sha256"],
        "execution_log_size_bytes": artifact["execution_log_size_bytes"],
        "case_ids_sha256": expected_case_ids_sha,
        "case_file_hash_manifest_sha256": canonical_manifest_sha,
        "shell_case_file_manifest_sha256": shell_manifest_sha,
        "scientific_payload_sha256": scientific_sha,
        "model_identity_sha256": _canonical_sha256(model_identity),
        "all_case_schema_checks_pass": True,
        "all_aggregate_recomputations_pass": True,
        "all_memory_checks_pass": True,
        "all_cache_mechanics_checks_pass": True,
        "all_model_identity_checks_pass": True,
        "execution_log": log_report,
    }


def main() -> None:
    args = _parse_args()
    perturbation, perturbation_sha256 = load_perturbation_protocol(
        args.perturbation_protocol.resolve()
    )
    _, gate_sha256 = load_perturbation_acceptance_gate(
        args.acceptance_gate.resolve(),
        perturbation_protocol_sha256=perturbation_sha256,
        verify_artifacts=True,
    )
    quality, quality_sha256 = load_formal_protocol(args.quality_protocol.resolve())
    if quality_sha256 != perturbation["inherited_quality_protocol"]["sha256"]:
        raise Qwen3PerturbationError("quality protocol differs from perturbation freeze")
    execution, execution_sha256 = load_execution_config(
        args.execution_config.resolve(),
        protocol=quality,
        protocol_sha256=quality_sha256,
    )
    if execution_sha256 != perturbation["inherited_execution"]["sha256"]:
        raise Qwen3PerturbationError("quality execution differs from perturbation freeze")
    input_path = args.input_manifest.resolve()
    input_sha256 = file_sha256(input_path)
    if input_sha256 != perturbation["inherited_input_manifest"]["sha256"]:
        raise Qwen3PerturbationError("input manifest differs from perturbation freeze")
    input_manifest = _load_json(input_path, "input manifest")
    validate_input_manifest(
        input_manifest, protocol=quality, protocol_sha256=quality_sha256
    )
    artifact_path = args.artifact_manifest.resolve()
    artifacts = _load_json(artifact_path, "full artifact manifest")
    _validate_artifact_manifest(
        artifacts,
        protocol_sha256=perturbation_sha256,
        gate_sha256=gate_sha256,
    )

    partitions = {}
    for partition in ("cage_qwen3", "kitty_qwen3"):
        partitions[partition] = _validate_partition(
            partition=partition,
            artifact=artifacts["partitions"][partition],
            quality=quality,
            execution=execution,
            execution_sha256=execution_sha256,
            input_manifest=input_manifest,
            perturbation=perturbation,
            perturbation_sha256=perturbation_sha256,
            gate_sha256=gate_sha256,
            input_sha256=input_sha256,
            source_commit=artifacts["source_commit"],
        )
    report = {
        "schema_version": 1,
        "status": "pass",
        "stage": "joint_full_postrun_validation",
        "artifact_set_id": artifacts["artifact_set_id"],
        "artifact_manifest_sha256": file_sha256(artifact_path),
        "perturbation_protocol_sha256": perturbation_sha256,
        "acceptance_gate_sha256": gate_sha256,
        "quality_protocol_sha256": quality_sha256,
        "quality_execution_sha256": execution_sha256,
        "input_manifest_sha256": input_sha256,
        "source_commit": artifacts["source_commit"],
        "case_count": sum(record["case_count"] for record in partitions.values()),
        "layer_record_count": sum(
            record["layer_record_count"] for record in partitions.values()
        ),
        "failure_count": 0,
        "partitions": partitions,
        "joint_scientific_payload_sha256": _canonical_sha256(
            {
                name: record["scientific_payload_sha256"]
                for name, record in partitions.items()
            }
        ),
        "interpretation_performed": False,
    }
    if report["case_count"] != 1300 or report["layer_record_count"] != 46800:
        raise Qwen3PerturbationError("joint post-run counts mismatch")
    atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"audit_json: {args.output.resolve()}")
    print(f"audit_json_sha256: {file_sha256(args.output.resolve())}")


if __name__ == "__main__":
    main()
