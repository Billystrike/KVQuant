#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
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

from models.cage_config import CageConfig
from models.qwen3_cage import Qwen3CageCache, install_qwen3_cage_attention
from models.qwen3_cage_v2 import CageV2Config, Qwen3CageV2Cache
from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes
from utils.qwen3_cage_v2_protocol import validate_cage_v2_dev_manifest
from utils.qwen3_memory import estimate_qwen3_cage_bytes, estimate_qwen3_kitty_bytes
from utils.qwen3_perturbation_protocol import (
    aggregate_layer_metrics,
    file_sha256,
    validate_aggregates,
    validate_layer_records,
)
from utils.qwen3_perturbation_runtime import Qwen3PerturbationRecorder


EXPECTED = {
    "python": "3.10.20",
    "torch": "2.4.1+cu121",
    "transformers": "4.53.2",
    "parameter_count": 8190735360,
    "kitty_commit": "dfd2c07b407d6b407179359207c612ab631f3ed1",
    "transformers_commit": "37f8b0b53512e6aae0cfd15746c133c101783178",
    "cache_utils_sha256": "529e858b9bb0ba59a830713bbd5ab594e29f262bbad1bba860ff7de3248e5a5a",
    "qwen3_modeling_sha256": "c12b4c4a34a06e887f3df399b1b09e9966c12d2a19769c258b7379066e585ea4",
}
KITTY_ROOT = Path("/root/autodl-tmp/Kitty")
SEED = 20260809


class CageV2Round1Error(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run claim-ineligible CAGE-v2 round1 local screening")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("cage_qwen3", "kitty_qwen3"), required=True)
    parser.add_argument("--stage", choices=("acceptance", "screen"), required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV2Round1Error(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CageV2Round1Error(f"JSON root must be an object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_state(partition: str) -> dict[str, Any]:
    state = {
        "git_commit": _git(REPO_ROOT, "rev-parse", "HEAD"),
        "dirty": bool(_git(REPO_ROOT, "status", "--porcelain", "--untracked-files=all")),
    }
    if partition == "kitty_qwen3":
        state.update(
            kitty_commit=_git(KITTY_ROOT, "rev-parse", "HEAD"),
            kitty_dirty=bool(
                _git(KITTY_ROOT, "status", "--porcelain", "--untracked-files=all")
            ),
            transformers_commit=_git(
                KITTY_ROOT / "third_party" / "transformers", "rev-parse", "HEAD"
            ),
        )
    return state


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = _load_json(path)
    if protocol.get("schema_version") != 1:
        raise CageV2Round1Error("round1 protocol schema mismatch")
    if protocol.get("status") != "development_protocol_not_frozen_for_final_claims":
        raise CageV2Round1Error("round1 protocol must remain development-only")
    if protocol.get("round1", {}).get("status") != "frozen_before_round1_gpu_results":
        raise CageV2Round1Error("round1 point grid is not frozen")
    return protocol, file_sha256(path)


def _expand_methods(protocol: dict[str, Any], partition: str) -> list[dict[str, Any]]:
    if partition == "kitty_qwen3":
        return [
            {
                "id": point["method_id"],
                "name": "kitty",
                "config": {"boosted_channels": point["boosted_channels"]},
                "prompt_length": point["prompt_length"],
                "packed_bytes": point["packed_bytes"],
            }
            for point in protocol["round1"]["kitty_points"]
        ]
    methods = []
    for point in protocol["round1"]["cage_v1_controls"]:
        methods.append(
            {
                "id": point["method_id"],
                "name": "cage_v1",
                "config": {"residual_length": point["residual_length"]},
                "prompt_length": point["prompt_length"],
            }
        )
    for point in protocol["round1"]["cage_v2_points"]:
        methods.append(
            {
                "id": point["method_id"],
                "name": "cage_v2",
                "config": {
                    "residual_length": point["residual_length"],
                    "sink_length": point["sink_length"],
                    "one_bit_channels": point["one_bit_channels"],
                    "two_bit_channels": point["two_bit_channels"],
                },
                "prompt_length": point["prompt_length"],
                "target_method": point["target_method"],
                "target_bytes": point["target_bytes"],
                "packed_bytes": point["packed_bytes"],
            }
        )
    return methods


def _expand_cases(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
    partition: str,
    stage: str,
) -> list[dict[str, Any]]:
    data = protocol["development_data"]
    anchor_key = "acceptance_anchor_indices" if stage == "acceptance" else "round1_screening_anchor_indices"
    anchors = set(data[anchor_key])
    inputs = [case for case in manifest["cases"] if case["identity"]["anchor_index"] in anchors]
    cases = []
    for method in _expand_methods(protocol, partition):
        matching = [
            record for record in inputs if record["identity"]["prompt_length"] == method["prompt_length"]
        ]
        for record in matching:
            scientific_identity = {
                "protocol_sha256": protocol_sha256,
                "manifest_sha256": manifest_sha256,
                "partition": partition,
                "method_id": method["id"],
                "input_case_id": record["case_id"],
            }
            cases.append(
                {
                    "case_id": _canonical_sha256(scientific_identity)[:24],
                    "method": method,
                    "input": record["identity"],
                    "prompt_ids": record["prompt_ids"],
                    "continuation_ids": record["continuation_ids"],
                }
            )
    expected_multiplier = 1 if stage == "acceptance" else len(data["round1_screening_anchor_indices"])
    if len(cases) != len(_expand_methods(protocol, partition)) * expected_multiplier:
        raise CageV2Round1Error("round1 expanded case count mismatch")
    return cases


def _v1_cache(config: dict[str, Any]) -> Qwen3CageCache:
    cage_config = CageConfig(cage_enable=True, cage_mode="fake")
    return Qwen3CageCache(cage_config, residual_length=config["residual_length"], bits=2)


def _v2_cache(config: dict[str, Any]) -> Qwen3CageV2Cache:
    return Qwen3CageV2Cache(
        CageV2Config(
            one_bit_channels=config["one_bit_channels"],
            two_bit_channels=config["two_bit_channels"],
            key_base_group_size=128,
            key_refinement_group_size=128,
            value_group_size=128,
            sink_length=config["sink_length"],
        ),
        residual_length=config["residual_length"],
    )


def _make_cache(method: dict[str, Any]):
    if method["name"] == "cage_v1":
        return _v1_cache(method["config"])
    if method["name"] == "cage_v2":
        return _v2_cache(method["config"])
    if method["name"] == "kitty":
        from kitty_sim import KittyKVCache, KittyKVCacheConfig

        boosted = method["config"]["boosted_channels"]
        return KittyKVCache(
            KittyKVCacheConfig(
                sink_length=32,
                buffer_length=128,
                group_size=128,
                kbits=2,
                vbits=2,
                promote_ratio=boosted / 128,
                promote_bit=4,
                channel_selection=1,
                VCache_BitDecoding=False,
                PostQuant=True,
            )
        )
    raise CageV2Round1Error(f"unsupported round1 method {method['name']!r}")


def _memory_report(method: dict[str, Any]) -> dict[str, Any]:
    seq_len = method["prompt_length"]
    config = method["config"]
    if method["name"] == "cage_v1":
        return estimate_qwen3_cage_bytes(
            seq_len=seq_len,
            residual_length=config["residual_length"],
        )
    if method["name"] == "cage_v2":
        report = estimate_qwen3_cage_v2_bytes(
            seq_len=seq_len,
            residual_length=config["residual_length"],
            one_bit_channels=config["one_bit_channels"],
            two_bit_channels=config["two_bit_channels"],
            sink_length=config["sink_length"],
        )
        if report["model_total_bytes"] != method["packed_bytes"]:
            raise CageV2Round1Error("CAGE-v2 runtime memory differs from frozen round1 bytes")
        if report["model_total_bytes"] > method["target_bytes"]:
            raise CageV2Round1Error("CAGE-v2 round1 point exceeds its target byte ceiling")
        return report
    report = estimate_qwen3_kitty_bytes(
        seq_len=seq_len,
        boosted_channels=config["boosted_channels"],
    )
    if report["model_total_bytes"] != method["packed_bytes"]:
        raise CageV2Round1Error("Kitty runtime memory differs from frozen round1 bytes")
    return report


def _cache_diagnostics(cache: Any, method: dict[str, Any], expected_length: int) -> dict[str, Any]:
    tensors = list(cache.key_cache) + list(cache.value_cache)
    record = {
        "cache_class": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "reported_seq_length": int(cache.get_seq_length()),
        "expected_seq_length": expected_length,
        "layer_count": len(cache.key_cache),
        "tensor_dtypes": sorted({str(tensor.dtype) for tensor in tensors}),
        "tensors_finite": all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors),
    }
    if record["reported_seq_length"] != expected_length or record["layer_count"] != 36:
        raise CageV2Round1Error("round1 cache shape mechanics mismatch")
    if record["tensor_dtypes"] != ["torch.float16"] or not record["tensors_finite"]:
        raise CageV2Round1Error("round1 cache tensors must be finite FP16 simulation storage")

    if method["name"] == "cage_v1":
        residual = method["config"]["residual_length"]
        expected_key = expected_length - expected_length % residual
        expected_value = max(0, expected_length - residual)
    elif method["name"] == "cage_v2":
        residual = method["config"]["residual_length"]
        sink = min(expected_length, method["config"]["sink_length"])
        non_sink = expected_length - sink
        expected_key = sink + non_sink - non_sink % residual
        expected_value = max(sink, expected_length - residual)
    else:
        boosted = method["config"]["boosted_channels"]
        official = {
            "sink_length": cache.sink_length,
            "buffer_length": cache.buffer_length,
            "group_size": cache.group_size,
            "kbits": cache.kbits,
            "vbits": cache.vbits,
            "promote_ratio": cache.promote_ratio,
            "promote_bit": cache.promote_bit,
            "channel_selection": cache.channel_selection,
            "VCache_BitDecoding": cache.VCache_BitDecoding,
            "PostQuant": cache.PostQuant,
        }
        expected_official = {
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
        if official != expected_official:
            raise CageV2Round1Error("Kitty official config changed during round1")
        record["official_config"] = official
        return record

    record["key_quantized_lengths"] = sorted(set(cache.key_quantized_lengths))
    record["value_quantized_lengths"] = sorted(set(cache.value_quantized_lengths))
    if record["key_quantized_lengths"] != [expected_key]:
        raise CageV2Round1Error("round1 Key quantized length mismatch")
    if record["value_quantized_lengths"] != [expected_value]:
        raise CageV2Round1Error("round1 Value quantized length mismatch")
    if method["name"] == "cage_v2":
        record["value_adaptive"] = False
        record["one_bit_channels"] = method["config"]["one_bit_channels"]
        record["two_bit_channels"] = method["config"]["two_bit_channels"]
    return record


def _run_case(
    model: Any, recorder: Qwen3PerturbationRecorder, case: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
        raise CageV2Round1Error("round1 prefill did not preserve cache identity")
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
        raise CageV2Round1Error("round1 decode did not preserve cache identity")
    del output
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    layer_records = recorder.finish()
    diagnostics = {
        "elapsed_seconds": elapsed,
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "cache": _cache_diagnostics(cache, case["method"], prompt_length + 1),
    }
    del cache, prompt, decode_token, full_reference
    recorder.reset_case()
    gc.collect()
    torch.cuda.empty_cache()
    return layer_records, diagnostics


def _model_identity(model: Any, partition: str) -> dict[str, Any]:
    cache_path = Path(transformers_cache_utils.__file__).resolve()
    qwen_path = Path(modeling_qwen3.__file__).resolve()
    identity = {
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
        "partition": partition,
        "source_sha256": {
            "qwen3_cage_v1": file_sha256(REPO_ROOT / "models" / "qwen3_cage.py"),
            "cage_v2_quant": file_sha256(REPO_ROOT / "models" / "cage_v2_quant.py"),
            "qwen3_cage_v2": file_sha256(REPO_ROOT / "models" / "qwen3_cage_v2.py"),
            "cage_v2_memory": file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v2.py"),
            "recorder": file_sha256(REPO_ROOT / "utils" / "qwen3_perturbation_runtime.py"),
            "runner": file_sha256(Path(__file__).resolve()),
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
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise CageV2Round1Error(f"round1 model identity checks failed: {failures}")
    identity["checks"] = checks
    return identity


def _validate_record(
    record: dict[str, Any],
    *,
    case: dict[str, Any],
    run_identity: dict[str, Any],
    model_identity: dict[str, Any],
) -> None:
    if record.get("schema_version") != 1 or record.get("status") != "completed":
        raise CageV2Round1Error("round1 result schema mismatch")
    if record.get("case_id") != case["case_id"]:
        raise CageV2Round1Error("round1 result case ID mismatch")
    if record.get("identity") != run_identity or record.get("model") != model_identity:
        raise CageV2Round1Error("round1 result identity mismatch")
    if record.get("method") != case["method"] or record.get("input") != case["input"]:
        raise CageV2Round1Error("round1 result method/input mismatch")
    validate_layer_records(record.get("layer_metrics", []))
    validate_aggregates(record.get("aggregates", {}), record["layer_metrics"])
    if record.get("memory") != _memory_report(case["method"]):
        raise CageV2Round1Error("round1 result memory report mismatch")
    if record["aggregates"]["relative_k_reconstruction_error"]["maximum"] <= 0:
        raise CageV2Round1Error("round1 method did not perturb Key cache")
    if record["aggregates"]["relative_v_reconstruction_error"]["maximum"] <= 0:
        raise CageV2Round1Error("round1 method did not perturb Value cache")


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise CageV2Round1Error("round1 screening requires CUDA")
    protocol_path = args.protocol.resolve()
    manifest_path = args.input_manifest.resolve()
    protocol, protocol_sha256 = _load_protocol(protocol_path)
    manifest = _load_json(manifest_path)
    manifest_sha256 = file_sha256(manifest_path)
    validate_cage_v2_dev_manifest(manifest, protocol=protocol, protocol_sha256=protocol_sha256)
    source_state = _source_state(args.partition)
    if source_state["dirty"] or source_state.get("kitty_dirty", False):
        raise CageV2Round1Error("round1 execution requires clean sources")
    if manifest["source_state"] != {
        "git_commit": source_state["git_commit"],
        "dirty": False,
    }:
        raise CageV2Round1Error("development manifest was not built from the execution commit")
    if args.partition == "kitty_qwen3":
        if source_state["kitty_commit"] != EXPECTED["kitty_commit"]:
            raise CageV2Round1Error("Kitty commit differs from the frozen provenance")
        if source_state["transformers_commit"] != EXPECTED["transformers_commit"]:
            raise CageV2Round1Error("Transformers commit differs from the frozen provenance")

    cases = _expand_cases(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        partition=args.partition,
        stage=args.stage,
    )
    run_identity = {
        "schema_version": 1,
        "experiment": "qwen3_cage_v2_round1_development_only",
        "claim_eligible": False,
        "partition": args.partition,
        "stage": args.stage,
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "source_state": source_state,
        "expected_case_ids": [case["case_id"] for case in cases],
    }
    output_dir = args.output_dir.resolve()
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if _load_json(identity_path) != run_identity:
            raise CageV2Round1Error("existing round1 run identity differs")
    else:
        _atomic_write_json(identity_path, run_identity)

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
        raise CageV2Round1Error("expected 36 Qwen3 attention adapters")
    recorder = Qwen3PerturbationRecorder(expected_layers=36, top_k=10)
    if recorder.install(model) != 36:
        raise CageV2Round1Error("expected 36 round1 perturbation callbacks")
    model_identity = _model_identity(model, args.partition)

    completed = 0
    resumed = 0
    for index, case in enumerate(cases, 1):
        path = output_dir / "cases" / f"{case['case_id']}.json"
        if path.exists():
            _validate_record(
                _load_json(path),
                case=case,
                run_identity=run_identity,
                model_identity=model_identity,
            )
            resumed += 1
            print(f"[{index}/{len(cases)}] resume-valid {case['case_id']}", flush=True)
            continue
        print(
            f"[{index}/{len(cases)}] running {case['method']['id']} "
            f"l={case['input']['prompt_length']} a={case['input']['anchor_index']}",
            flush=True,
        )
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
                "representation_note": "development-only fake quantization plus packed paper estimate",
            }
            _validate_record(
                record,
                case=case,
                run_identity=run_identity,
                model_identity=model_identity,
            )
            _atomic_write_json(path, record)
            completed += 1
            print(
                f"[{index}/{len(cases)}] completed {case['case_id']} "
                f"joint_post_o_proj_mse={aggregates['joint_post_o_proj_mse']['mean']:.9g} "
                f"elapsed={runtime['elapsed_seconds']:.3f}s",
                flush=True,
            )
        except Exception as error:
            _atomic_write_json(
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
        _validate_record(
            record,
            case=case,
            run_identity=run_identity,
            model_identity=model_identity,
        )
        records.append(record)
    failure_count = len(list((output_dir / "failures").glob("*.json"))) if (output_dir / "failures").exists() else 0
    if failure_count:
        raise CageV2Round1Error("round1 failure records remain")
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
        "case_ids_sha256": _canonical_sha256([case["case_id"] for case in cases]),
        "identity": run_identity,
        "model": model_identity,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
