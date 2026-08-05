#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
    load_perturbation_protocol,
    perturbation_case_id,
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
    "config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "generation_config_sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "model_index_sha256": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "transformers_cache_utils_sha256": "529e858b9bb0ba59a830713bbd5ab594e29f262bbad1bba860ff7de3248e5a5a",
    "transformers_qwen3_modeling_sha256": "c12b4c4a34a06e887f3df399b1b09e9966c12d2a19769c258b7379066e585ea4",
}
KITTY_ROOT = Path("/root/autodl-tmp/Kitty")
SEED = 20260805


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one isolated Qwen3 memory--perturbation partition"
    )
    parser.add_argument("--perturbation-protocol", type=Path, required=True)
    parser.add_argument("--quality-protocol", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("cage_qwen3", "kitty_qwen3"), required=True)
    parser.add_argument("--stage", choices=("acceptance", "full"), required=True)
    parser.add_argument("--acceptance-gate", type=Path)
    return parser.parse_args()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3PerturbationError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Qwen3PerturbationError(f"JSON root must be an object: {path}")
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


def _make_cache(method: dict[str, Any], partition: str):
    if partition == "cage_qwen3":
        if method["name"] == "fp16":
            return DynamicCache()
        if method["name"] == "kivi":
            return Qwen3KiviCache(Qwen3KiviCacheConfig(**method["config"]))
        if method["name"] == "cage":
            return _cage_cache(method["config"])
    elif partition == "kitty_qwen3" and method["name"] == "kitty":
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
    raise Qwen3PerturbationError(
        f"unsupported method {method['name']!r} in partition {partition!r}"
    )


def _memory_report(method: dict[str, Any], prompt_length: int) -> dict[str, Any]:
    name = method["name"]
    config = method["config"]
    if name == "fp16":
        return estimate_qwen3_fp16_bytes(seq_len=prompt_length)
    if name == "kivi":
        return estimate_qwen3_kivi_bytes(
            seq_len=prompt_length,
            group_size=config["group_size"],
            residual_length=config["residual_length"],
            bits=config["bits"],
        )
    if name == "cage":
        return estimate_qwen3_cage_bytes(
            seq_len=prompt_length,
            residual_length=config["residual_length"],
            key_group_sizes=config["key_group_sizes"],
            value_group_sizes=config["value_group_sizes"],
            bits=config["bits"],
        )
    if name == "kitty":
        return estimate_qwen3_kitty_bytes(
            seq_len=prompt_length,
            boosted_channels=config["boosted_channels"],
        )
    raise Qwen3PerturbationError(f"cannot estimate memory for method {name!r}")


def _cache_diagnostics(
    cache: Any, method: dict[str, Any], *, expected_length: int
) -> dict[str, Any]:
    tensors = list(cache.key_cache) + list(cache.value_cache)
    record: dict[str, Any] = {
        "cache_class": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "reported_seq_length": int(cache.get_seq_length()),
        "expected_seq_length": expected_length,
        "layer_count": len(cache.key_cache),
        "tensor_dtypes": sorted({str(tensor.dtype) for tensor in tensors}),
        "tensors_finite": all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors),
    }
    if record["reported_seq_length"] != expected_length:
        raise Qwen3PerturbationError("cache sequence length differs from the measurement")
    if record["layer_count"] != 36 or len(cache.value_cache) != 36:
        raise Qwen3PerturbationError("candidate cache must contain all 36 layers")
    if record["tensor_dtypes"] != ["torch.float16"] or not record["tensors_finite"]:
        raise Qwen3PerturbationError("candidate cache tensors must be finite FP16 simulation storage")

    if method["name"] in {"cage", "kivi"}:
        residual = method["config"]["residual_length"]
        expected_key = expected_length - expected_length % residual
        expected_value = max(0, expected_length - residual)
        record["key_quantized_lengths"] = sorted(set(cache.key_quantized_lengths))
        record["value_quantized_lengths"] = sorted(set(cache.value_quantized_lengths))
        if record["key_quantized_lengths"] != [expected_key]:
            raise Qwen3PerturbationError("Key quantized length differs from cache mechanics")
        if record["value_quantized_lengths"] != [expected_value]:
            raise Qwen3PerturbationError("Value quantized length differs from cache mechanics")
    elif method["name"] == "kitty":
        expected_boosted = method["config"]["boosted_channels"]
        record["official_config"] = {
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
        expected_config = {
            "sink_length": 32,
            "buffer_length": 128,
            "group_size": 128,
            "kbits": 2,
            "vbits": 2,
            "promote_ratio": expected_boosted / 128,
            "promote_bit": 4,
            "channel_selection": 1,
            "VCache_BitDecoding": False,
            "PostQuant": True,
        }
        if record["official_config"] != expected_config:
            raise Qwen3PerturbationError("Kitty cache config differs from the official frozen path")
    return record


def _run_case(
    model: Any,
    recorder: Qwen3PerturbationRecorder,
    case: dict[str, Any],
    partition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_length = case["input"]["prompt_length"]
    prompt = torch.tensor([case["prompt_ids"]], dtype=torch.long, device="cuda:0")
    decode_token = torch.tensor(
        [[case["continuation_ids"][0]]], dtype=torch.long, device="cuda:0"
    )
    if prompt.shape != (1, prompt_length):
        raise Qwen3PerturbationError("prompt tensor differs from frozen input")
    full_reference = torch.cat((prompt, decode_token), dim=1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    recorder.begin_reference(prompt_length=prompt_length)
    with torch.inference_mode():
        reference_output = model(
            input_ids=full_reference,
            attention_mask=torch.ones_like(full_reference),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    del reference_output

    cache = _make_cache(case["method"], partition)
    recorder.begin_candidate_prefill()
    with torch.inference_mode():
        prefill_output = model(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    if prefill_output.past_key_values is not cache:
        raise Qwen3PerturbationError("candidate prefill did not preserve cache identity")
    del prefill_output

    recorder.begin_candidate_decode()
    with torch.inference_mode():
        decode_output = model(
            input_ids=decode_token,
            attention_mask=torch.ones((1, prompt_length + 1), dtype=torch.long, device="cuda:0"),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    if decode_output.past_key_values is not cache:
        raise Qwen3PerturbationError("candidate decode did not preserve cache identity")
    if int(cache.get_seq_length()) != prompt_length + 1:
        raise Qwen3PerturbationError("candidate cache length differs from prompt_length + 1")
    del decode_output
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    layer_records = recorder.finish()
    diagnostics = {
        "elapsed_seconds": elapsed,
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "cache": _cache_diagnostics(
            cache,
            case["method"],
            expected_length=prompt_length + 1,
        ),
    }
    del cache, prompt, decode_token, full_reference
    recorder.reset_case()
    gc.collect()
    torch.cuda.empty_cache()
    return layer_records, diagnostics


def _model_identity(model: Any, partition: str) -> dict[str, Any]:
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
        "transformers_cache_utils_sha256": file_sha256(Path(transformers_cache_utils.__file__).resolve()),
        "transformers_qwen3_modeling_sha256": file_sha256(Path(modeling_qwen3.__file__).resolve()),
        "qwen3_cage_sha256": file_sha256(REPO_ROOT / "models" / "qwen3_cage.py"),
        "recorder_sha256": file_sha256(REPO_ROOT / "utils" / "qwen3_perturbation_runtime.py"),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "metadata_hashes": metadata,
        "partition": partition,
    }
    checks = {
        "python": identity["python"] == EXPECTED["python"],
        "torch": identity["torch"] == EXPECTED["torch"],
        "transformers": identity["transformers"] == EXPECTED["transformers"],
        "parameters": identity["parameter_count"] == EXPECTED["parameter_count"],
        "device": identity["parameter_devices"] == ["cuda:0"],
        "dtype": identity["parameter_dtypes"] == ["torch.float16"],
        "attention": identity["attention_implementation"] == "flash_attention_2",
        "cache_utils": identity["transformers_cache_utils_sha256"] == EXPECTED["transformers_cache_utils_sha256"],
        "qwen_modeling": identity["transformers_qwen3_modeling_sha256"] == EXPECTED["transformers_qwen3_modeling_sha256"],
        "metadata": all(metadata[name] == EXPECTED[name] for name in metadata),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise Qwen3PerturbationError(f"model identity checks failed: {failures}")
    identity["checks"] = checks
    return identity


def _validate_record(
    record: dict[str, Any],
    case: dict[str, Any],
    identity: dict[str, Any],
    model_identity: dict[str, Any],
) -> None:
    if record.get("schema_version") != 1 or record.get("status") != "completed":
        raise Qwen3PerturbationError("completed perturbation record schema mismatch")
    if record.get("case_id") != case["perturbation_case_id"]:
        raise Qwen3PerturbationError("completed perturbation case ID mismatch")
    if record.get("base_quality_case_id") != case["case_id"]:
        raise Qwen3PerturbationError("base quality case ID mismatch")
    if record.get("method") != case["method"] or record.get("input") != case["input"]:
        raise Qwen3PerturbationError("completed perturbation method/input mismatch")
    if record.get("identity") != identity:
        raise Qwen3PerturbationError("completed perturbation identity mismatch")
    if record.get("model") != model_identity:
        raise Qwen3PerturbationError("completed perturbation model identity mismatch")
    validate_layer_records(record.get("layer_metrics", []))
    validate_aggregates(record.get("aggregates", {}), record["layer_metrics"])
    memory = record.get("memory")
    if not isinstance(memory, dict) or memory.get("seq_len") != case["input"]["prompt_length"]:
        raise Qwen3PerturbationError("completed perturbation memory report mismatch")
    if case["method"]["name"] == "fp16":
        for layer in record["layer_metrics"]:
            for name, value in layer["metrics"].items():
                expected = 1.0 if name == "topk_attention_overlap" else 0.0
                if float(value) != expected:
                    raise Qwen3PerturbationError("FP16 perturbation identity check failed")
    else:
        if record["aggregates"]["relative_k_reconstruction_error"]["maximum"] <= 0:
            raise Qwen3PerturbationError("quantized method did not perturb any Key cache state")
        if record["aggregates"]["relative_v_reconstruction_error"]["maximum"] <= 0:
            raise Qwen3PerturbationError("quantized method did not perturb any Value cache state")


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise Qwen3PerturbationError("Qwen3 perturbation execution requires CUDA")
    if args.stage == "full":
        raise Qwen3PerturbationError(
            "full perturbation execution is locked until the two-repeat acceptance gate is frozen"
        )
    perturbation, perturbation_sha256 = load_perturbation_protocol(
        args.perturbation_protocol.resolve()
    )
    receipt_path = REPO_ROOT / perturbation["quality_results_receipt"]["path"]
    if file_sha256(receipt_path) != perturbation["quality_results_receipt"]["sha256"]:
        raise Qwen3PerturbationError("formal quality results receipt differs from the post-quality freeze")
    quality, quality_sha256 = load_formal_protocol(args.quality_protocol.resolve())
    if quality_sha256 != perturbation["inherited_quality_protocol"]["sha256"]:
        raise Qwen3PerturbationError("quality protocol hash differs from perturbation freeze")
    execution, execution_sha256 = load_execution_config(
        args.execution_config.resolve(), protocol=quality, protocol_sha256=quality_sha256
    )
    if execution_sha256 != perturbation["inherited_execution"]["sha256"]:
        raise Qwen3PerturbationError("execution config hash differs from perturbation freeze")
    input_path = args.input_manifest.resolve()
    input_sha256 = file_sha256(input_path)
    if input_sha256 != perturbation["inherited_input_manifest"]["sha256"]:
        raise Qwen3PerturbationError("input manifest hash differs from perturbation freeze")
    input_manifest = _load_json(input_path)
    validate_input_manifest(input_manifest, protocol=quality, protocol_sha256=quality_sha256)
    source_state = _source_state(args.partition)
    if source_state["dirty"] or source_state.get("kitty_dirty", False):
        raise Qwen3PerturbationError("perturbation execution requires clean frozen sources")
    if args.partition == "kitty_qwen3":
        if source_state["kitty_commit"] != EXPECTED["kitty_commit"]:
            raise Qwen3PerturbationError("Kitty commit differs from the freeze")
        if source_state["transformers_commit"] != EXPECTED["transformers_commit"]:
            raise Qwen3PerturbationError("Transformers submodule commit differs from the freeze")

    base_cases = expand_partition_cases(
        protocol=quality,
        execution=execution,
        execution_sha256=execution_sha256,
        input_manifest=input_manifest,
        partition=args.partition,
        stage="acceptance",
    )
    cases = []
    for case in base_cases:
        case = dict(case)
        case["perturbation_case_id"] = perturbation_case_id(
            base_case_id=case["case_id"],
            perturbation_protocol_sha256=perturbation_sha256,
        )
        cases.append(case)

    output_dir = args.output_dir.resolve()
    run_identity = {
        "schema_version": 1,
        "partition": args.partition,
        "stage": args.stage,
        "perturbation_protocol_id": perturbation["protocol_id"],
        "perturbation_protocol_sha256": perturbation_sha256,
        "quality_protocol_sha256": quality_sha256,
        "quality_execution_sha256": execution_sha256,
        "input_manifest_sha256": input_sha256,
        "source_state": source_state,
        "expected_case_ids": [case["perturbation_case_id"] for case in cases],
    }
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if _load_json(identity_path) != run_identity:
            raise Qwen3PerturbationError("existing run identity differs; refusing mixed resume")
    else:
        atomic_write_json(identity_path, run_identity)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        quality["model"]["reference"],
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()
    installed = install_qwen3_cage_attention(model)
    if installed != 36:
        raise Qwen3PerturbationError("expected 36 Qwen3 attention adapters")
    recorder = Qwen3PerturbationRecorder(expected_layers=36, top_k=10)
    if recorder.install(model) != 36:
        raise Qwen3PerturbationError("expected 36 perturbation callbacks")
    model_identity = _model_identity(model, args.partition)

    completed = 0
    resumed = 0
    for index, case in enumerate(cases, 1):
        path = output_dir / "cases" / f"{case['perturbation_case_id']}.json"
        if path.exists():
            _validate_record(_load_json(path), case, run_identity, model_identity)
            resumed += 1
            print(f"[{index}/{len(cases)}] resume-valid {case['perturbation_case_id']}", flush=True)
            continue
        print(
            f"[{index}/{len(cases)}] running {case['method']['id']} "
            f"l={case['input']['prompt_length']} a={case['input']['anchor_index']}",
            flush=True,
        )
        try:
            layer_records, runtime = _run_case(model, recorder, case, args.partition)
            validate_layer_records(layer_records)
            aggregates = aggregate_layer_metrics(layer_records)
            record = {
                "schema_version": 1,
                "status": "completed",
                "case_id": case["perturbation_case_id"],
                "base_quality_case_id": case["case_id"],
                "partition": args.partition,
                "stage": args.stage,
                "identity": run_identity,
                "model": model_identity,
                "method": case["method"],
                "input": case["input"],
                "memory": _memory_report(case["method"], case["input"]["prompt_length"]),
                "measurement": perturbation["measurement"],
                "layer_metrics": layer_records,
                "aggregates": aggregates,
                "cache": runtime.pop("cache"),
                "runtime": runtime,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "representation_note": "accuracy simulation plus packed paper-estimate memory",
            }
            _validate_record(record, case, run_identity, model_identity)
            atomic_write_json(path, record)
            completed += 1
            print(
                f"[{index}/{len(cases)}] completed {case['perturbation_case_id']} "
                f"joint_post_o_proj_mse={aggregates['joint_post_o_proj_mse']['mean']:.9g} "
                f"elapsed={runtime['elapsed_seconds']:.3f}s",
                flush=True,
            )
        except Exception as error:
            atomic_write_json(
                output_dir / "failures" / f"{case['perturbation_case_id']}.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "case_id": case["perturbation_case_id"],
                    "base_quality_case_id": case["case_id"],
                    "identity": run_identity,
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
        record = _load_json(output_dir / "cases" / f"{case['perturbation_case_id']}.json")
        _validate_record(record, case, run_identity, model_identity)
        records.append(record)
    failure_count = len(list((output_dir / "failures").glob("*.json"))) if (output_dir / "failures").exists() else 0
    if failure_count:
        raise Qwen3PerturbationError("failure records remain in the output directory")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "partition": args.partition,
        "stage": args.stage,
        "expected_cases": len(cases),
        "completed_cases": len(records),
        "new_cases": completed,
        "resumed_cases": resumed,
        "failure_records": failure_count,
        "identity": run_identity,
        "model": model_identity,
        "case_ids_sha256": hashlib.sha256(
            json.dumps([case["perturbation_case_id"] for case in cases], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Qwen3FormalError as error:
        raise Qwen3PerturbationError(str(error)) from error
