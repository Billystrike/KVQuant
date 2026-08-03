#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM
from transformers import cache_utils as transformers_cache_utils
from transformers.models.qwen3 import modeling_qwen3

import kitty_sim.kitty_simulate as kitty_simulate
from kitty_sim import KittyKVCache, KittyKVCacheConfig

from utils.qwen3_formal import (
    Qwen3FormalError,
    atomic_write_json,
    expand_partition_cases,
    file_sha256,
    formal_scoring,
    load_execution_config,
    load_formal_protocol,
    validate_completed_result,
    validate_input_manifest,
)


EXPECTED = {
    "python": "3.10.20",
    "torch": "2.4.1+cu121",
    "transformers": "4.53.2",
    "parameter_count": 8190735360,
    "kitty_commit": "dfd2c07b407d6b407179359207c612ab631f3ed1",
    "transformers_commit": "37f8b0b53512e6aae0cfd15746c133c101783178",
    "kitty_simulate_blob": "1149778f674ba49aa2dcb5d6d7b62fcff3e29fa9",
    "kitty_utils_quant_blob": "907fd6b1951b869c33216d8eb658df21e8bf9c29",
    "config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "generation_config_sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "model_index_sha256": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "transformers_cache_utils_sha256": "529e858b9bb0ba59a830713bbd5ab594e29f262bbad1bba860ff7de3248e5a5a",
    "transformers_qwen3_modeling_sha256": "c12b4c4a34a06e887f3df399b1b09e9966c12d2a19769c258b7379066e585ea4",
}
PARTITION = "kitty_qwen3"
KITTY_ROOT = Path("/root/autodl-tmp/Kitty")
SEED = 20260803


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Qwen3 Kitty accuracy-simulation partition")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("acceptance", "full"), required=True)
    return parser.parse_args()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def _source_state() -> dict[str, Any]:
    cage_commit = _git(REPO_ROOT, "rev-parse", "HEAD")
    cage_dirty = bool(_git(REPO_ROOT, "status", "--porcelain", "--untracked-files=all"))
    kitty_commit = _git(KITTY_ROOT, "rev-parse", "HEAD")
    kitty_dirty = bool(_git(KITTY_ROOT, "status", "--porcelain", "--untracked-files=all"))
    transformers_commit = _git(KITTY_ROOT / "third_party" / "transformers", "rev-parse", "HEAD")
    return {
        "git_commit": cage_commit,
        "dirty": cage_dirty,
        "kitty_commit": kitty_commit,
        "kitty_dirty": kitty_dirty,
        "transformers_commit": transformers_commit,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3FormalError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Qwen3FormalError(f"JSON root must be an object: {path}")
    return value


def _token_nll(logits: torch.Tensor, target: torch.Tensor) -> float:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise Qwen3FormalError(f"expected logits [1, vocab], got {tuple(logits.shape)}")
    loss = F.cross_entropy(logits.float(), target.reshape(1), reduction="sum")
    value = float(loss.detach().cpu().item())
    if not math.isfinite(value) or value < 0:
        raise Qwen3FormalError(f"invalid token NLL {value!r}")
    return value


class QuantizationTracker:
    def __init__(self) -> None:
        self.original_build = kitty_simulate.build_promote_mask
        self.original_quant = kitty_simulate.fake_quant_groupwise_lastdim
        self.reset()

    def reset(self) -> None:
        self.key_fake_quant_calls = 0
        self.value_fake_quant_calls = 0
        self.promoted_channels_per_head: set[int] = set()

    def build_promote_mask(self, key_states, promote_ratio, channel_selection):
        mask = self.original_build(key_states, promote_ratio, channel_selection)
        counts = mask.sum(dim=-1).detach().cpu().reshape(-1).tolist()
        self.promoted_channels_per_head.update(int(value) for value in counts)
        return mask

    def fake_quant_groupwise_lastdim(
        self, data, group_size, bit, promote_mask=None, promote_bit=4
    ):
        if promote_mask is None:
            self.value_fake_quant_calls += 1
        else:
            self.key_fake_quant_calls += 1
        return self.original_quant(
            data,
            group_size,
            bit,
            promote_mask=promote_mask,
            promote_bit=promote_bit,
        )

    def install(self) -> None:
        kitty_simulate.build_promote_mask = self.build_promote_mask
        kitty_simulate.fake_quant_groupwise_lastdim = self.fake_quant_groupwise_lastdim


def _make_cache(method: dict[str, Any]) -> KittyKVCache:
    boosted = method["config"]["boosted_channels"]
    if boosted not in {16, 32}:
        raise Qwen3FormalError("Kitty boosted channel count must be 16 or 32")
    config = KittyKVCacheConfig(
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
    return KittyKVCache(config)


def _cache_diagnostics(
    cache: KittyKVCache,
    method: dict[str, Any],
    tracker: QuantizationTracker,
    expected_length: int,
) -> dict[str, Any]:
    reported_length = int(cache.get_seq_length())
    tensors = list(cache.key_cache) + list(cache.value_cache)
    finite = all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)
    expected_promoted = method["config"]["boosted_channels"]
    expected_key_calls = 36 * ((expected_length - 32) // 128)
    expected_value_calls = 36 * 64
    record = {
        "cache_class": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "reported_seq_length": reported_length,
        "expected_seq_length": expected_length,
        "layer_count": len(cache.key_cache),
        "tensor_dtypes": sorted({str(tensor.dtype) for tensor in tensors}),
        "tensors_finite": finite,
        "sink_length": cache.sink_length,
        "buffer_length": cache.buffer_length,
        "group_size": cache.group_size,
        "kbits": cache.kbits,
        "vbits": cache.vbits,
        "promote_ratio": cache.promote_ratio,
        "promote_bit": cache.promote_bit,
        "channel_selection": cache.channel_selection,
        "post_quant": cache.PostQuant,
        "key_fake_quant_calls": tracker.key_fake_quant_calls,
        "value_fake_quant_calls": tracker.value_fake_quant_calls,
        "promoted_channels_per_head": sorted(tracker.promoted_channels_per_head),
    }
    checks = {
        "reported_length": reported_length == expected_length,
        "layer_count": len(cache.key_cache) == 36 and len(cache.value_cache) == 36,
        "tensor_dtype": record["tensor_dtypes"] == ["torch.float16"],
        "tensors_finite": finite,
        "official_config": (
            cache.sink_length == 32
            and cache.buffer_length == 128
            and cache.group_size == 128
            and cache.kbits == 2
            and cache.vbits == 2
            and cache.promote_bit == 4
            and cache.channel_selection == 1
            and cache.VCache_BitDecoding is False
            and cache.PostQuant is True
        ),
        "promoted_channels": record["promoted_channels_per_head"] == [expected_promoted],
        "key_quantization_triggered": tracker.key_fake_quant_calls == expected_key_calls,
        "value_quantization_triggered": tracker.value_fake_quant_calls == expected_value_calls,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise Qwen3FormalError(f"Kitty cache diagnostics failed: {failures}")
    record["checks"] = checks
    return record


def _run_case(
    model: Any, case: dict[str, Any], tracker: QuantizationTracker
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = torch.tensor([case["prompt_ids"]], dtype=torch.long, device="cuda:0")
    continuation = torch.tensor(
        [case["continuation_ids"]], dtype=torch.long, device="cuda:0"
    )
    prompt_length = case["input"]["prompt_length"]
    if prompt.shape[-1] != prompt_length or continuation.shape[-1] != 64:
        raise Qwen3FormalError("prepared Kitty case lengths differ from frozen input")
    tracker.reset()
    cache = _make_cache(case["method"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    total_started = time.perf_counter()

    attention_mask = torch.ones_like(prompt)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=prompt,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    token_nlls = [_token_nll(output.logits[:, -1, :], continuation[:, 0])]
    if output.past_key_values is not cache:
        raise Qwen3FormalError("Kitty prefill did not preserve cache identity")
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
            raise Qwen3FormalError("Kitty decode did not preserve cache identity")
        token_nlls.append(
            _token_nll(output.logits[:, -1, :], continuation[:, index + 1])
        )
        del output
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - started
    elapsed_seconds = time.perf_counter() - total_started
    expected_length = prompt_length + 63
    cache_record = _cache_diagnostics(cache, case["method"], tracker, expected_length)
    runtime = {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "elapsed_seconds": elapsed_seconds,
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    scoring = formal_scoring(token_nlls)
    del cache, prompt, continuation, attention_mask
    gc.collect()
    torch.cuda.empty_cache()
    return scoring, {"cache": cache_record, "runtime": runtime}


def _model_identity(model: Any, source_state: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(model.config._name_or_path).resolve()
    cache_utils_path = Path(transformers_cache_utils.__file__).resolve()
    qwen_path = Path(modeling_qwen3.__file__).resolve()
    simulate_path = Path(kitty_simulate.__file__).resolve()
    expected_simulate_path = (KITTY_ROOT / "src" / "kitty_sim" / "kitty_simulate.py").resolve()
    metadata_hashes = {
        "config_sha256": file_sha256(model_path / "config.json"),
        "generation_config_sha256": file_sha256(model_path / "generation_config.json"),
        "model_index_sha256": file_sha256(model_path / "model.safetensors.index.json"),
        "tokenizer_config_sha256": file_sha256(model_path / "tokenizer_config.json"),
    }
    simulate_blob = _git(KITTY_ROOT, "hash-object", "src/kitty_sim/kitty_simulate.py")
    utils_blob = _git(KITTY_ROOT, "hash-object", "src/kitty_sim/utils_quant.py")
    identity = {
        "reference": str(model_path),
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_type": getattr(model.config, "model_type", None),
        "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_devices": sorted({str(parameter.device) for parameter in model.parameters()}),
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
        "attention_implementation": model.config._attn_implementation,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "source_state": source_state,
        "kitty_simulate_path": str(simulate_path),
        "kitty_simulate_blob": simulate_blob,
        "kitty_utils_quant_blob": utils_blob,
        "transformers_cache_utils_sha256": file_sha256(cache_utils_path),
        "transformers_qwen3_modeling_sha256": file_sha256(qwen_path),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "metadata_hashes": metadata_hashes,
    }
    checks = {
        "python": identity["python"] == EXPECTED["python"],
        "torch": identity["torch"] == EXPECTED["torch"],
        "transformers": identity["transformers"] == EXPECTED["transformers"],
        "kitty_commit": source_state["kitty_commit"] == EXPECTED["kitty_commit"],
        "transformers_commit": source_state["transformers_commit"] == EXPECTED["transformers_commit"],
        "kitty_clean": source_state["kitty_dirty"] is False,
        "kitty_simulate_path": simulate_path == expected_simulate_path,
        "kitty_simulate_blob": simulate_blob == EXPECTED["kitty_simulate_blob"],
        "kitty_utils_quant_blob": utils_blob == EXPECTED["kitty_utils_quant_blob"],
        "cache_utils": identity["transformers_cache_utils_sha256"]
        == EXPECTED["transformers_cache_utils_sha256"],
        "qwen_modeling": identity["transformers_qwen3_modeling_sha256"]
        == EXPECTED["transformers_qwen3_modeling_sha256"],
        "metadata_hashes": all(metadata_hashes[name] == EXPECTED[name] for name in metadata_hashes),
        "model_type": identity["model_type"] == "qwen3",
        "native_context": identity["max_position_embeddings"] == 40960,
        "parameter_count": identity["parameter_count"] == EXPECTED["parameter_count"],
        "parameter_device": identity["parameter_devices"] == ["cuda:0"],
        "parameter_dtype": identity["parameter_dtypes"] == ["torch.float16"],
        "attention_implementation": identity["attention_implementation"] == "flash_attention_2",
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise Qwen3FormalError(f"Kitty model/source identity checks failed: {failures}")
    identity["checks"] = checks
    return identity


def main() -> None:
    args = _parse_args()
    if args.stage == "full":
        raise Qwen3FormalError("formal full execution is locked until both repeated partition gates pass")
    if not torch.cuda.is_available():
        raise Qwen3FormalError("formal Kitty partition requires CUDA")
    protocol, protocol_sha256 = load_formal_protocol(args.protocol.resolve())
    execution, execution_sha256 = load_execution_config(
        args.execution_config.resolve(), protocol=protocol, protocol_sha256=protocol_sha256
    )
    input_manifest_path = args.input_manifest.resolve()
    input_manifest_sha256 = file_sha256(input_manifest_path)
    if input_manifest_sha256 != execution["input_manifest"]["sha256"]:
        raise Qwen3FormalError("input manifest SHA-256 differs from execution freeze")
    input_manifest = _load_json(input_manifest_path)
    validate_input_manifest(input_manifest, protocol=protocol, protocol_sha256=protocol_sha256)
    source_state = _source_state()
    if source_state["dirty"] or source_state["kitty_dirty"]:
        raise Qwen3FormalError("formal Kitty execution requires both source trees to be clean")
    cases = expand_partition_cases(
        protocol=protocol,
        execution=execution,
        execution_sha256=execution_sha256,
        input_manifest=input_manifest,
        partition=PARTITION,
        stage=args.stage,
    )
    output_dir = args.output_dir.resolve()
    run_identity = {
        "schema_version": 1,
        "partition": PARTITION,
        "stage": args.stage,
        "execution_id": execution["execution_id"],
        "execution_sha256": execution_sha256,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "source_state": source_state,
        "expected_case_ids": [case["case_id"] for case in cases],
    }
    lock_path = output_dir / "run_identity.json"
    if lock_path.exists():
        if _load_json(lock_path) != run_identity:
            raise Qwen3FormalError("existing Kitty output identity differs; refusing mixed resume")
    else:
        atomic_write_json(lock_path, run_identity)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tracker = QuantizationTracker()
    tracker.install()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        protocol["model"]["reference"],
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    ).eval()
    load_seconds = time.perf_counter() - load_started
    model_identity = _model_identity(model, source_state)
    model_identity["load_seconds"] = load_seconds
    identity_fields = {
        "execution_id": execution["execution_id"],
        "execution_sha256": execution_sha256,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "source_state": source_state,
    }

    completed = resumed = 0
    for index, case in enumerate(cases, 1):
        result_path = output_dir / "cases" / f"{case['case_id']}.json"
        failure_path = output_dir / "failures" / f"{case['case_id']}.json"
        if result_path.exists():
            record = _load_json(result_path)
            validate_completed_result(record, expected_case=case, **identity_fields)
            if failure_path.exists():
                failure_path.unlink()
            resumed += 1
            print(f"[{index}/{len(cases)}] resume-valid {case['case_id']} {case['method']['id']}", flush=True)
            continue
        print(
            f"[{index}/{len(cases)}] running {case['case_id']} "
            f"{case['method']['id']} l={case['input']['prompt_length']} "
            f"a={case['input']['anchor_index']}",
            flush=True,
        )
        try:
            scoring, diagnostics = _run_case(model, case, tracker)
            record = {
                "schema_version": 1,
                "status": "completed",
                "case_id": case["case_id"],
                "partition": PARTITION,
                "stage": args.stage,
                "identity": identity_fields,
                "model": model_identity,
                "method": case["method"],
                "input": case["input"],
                "scoring": scoring,
                "cache": diagnostics["cache"],
                "runtime": diagnostics["runtime"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "representation_note": "official Kitty fake-quant accuracy simulation; packed paper-estimate only",
            }
            validate_completed_result(record, expected_case=case, **identity_fields)
            atomic_write_json(result_path, record)
            if failure_path.exists():
                failure_path.unlink()
            completed += 1
            print(
                f"[{index}/{len(cases)}] completed {case['case_id']} "
                f"elapsed={record['runtime']['elapsed_seconds']:.3f}s",
                flush=True,
            )
        except Exception as error:
            atomic_write_json(
                failure_path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "case_id": case["case_id"],
                    "partition": PARTITION,
                    "stage": args.stage,
                    "identity": identity_fields,
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
        validate_completed_result(record, expected_case=case, **identity_fields)
        records.append(record)
    failure_count = (
        len(list((output_dir / "failures").glob("*.json")))
        if (output_dir / "failures").exists()
        else 0
    )
    if failure_count:
        raise Qwen3FormalError("failure records remain in Kitty acceptance output")
    summary = {
        "schema_version": 1,
        "status": "pass",
        "partition": PARTITION,
        "stage": args.stage,
        "expected_cases": len(cases),
        "completed_cases": len(records),
        "new_cases": completed,
        "resumed_cases": resumed,
        "failure_records": failure_count,
        "identity": identity_fields,
        "model": model_identity,
        "case_ids_sha256": hashlib.sha256(
            json.dumps([case["case_id"] for case in cases], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
