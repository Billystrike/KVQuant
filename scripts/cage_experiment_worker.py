"""Run one resolved CAGE experiment configuration, one point at a time."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cage_experiment_config import expand_jobs, load_and_resolve_manifest
from utils.cage_experiment_hooks import (
    begin_candidate_capture, begin_reference_capture, collect_layer_metrics,
    reset_experiment_capture,
)
from utils.cage_experiment_io import (
    atomic_write_json, atomic_write_jsonl, collect_provenance,
    load_prompt_records, prepare_prompt, stable_run_id,
)
from utils.cage_memory import (
    build_memory_namespace, sum_cache_summaries, summarize_cache_bytes,
    summarize_fp16_cache_bytes, summarize_runtime_cache_bytes,
)


METRIC_NAMES = (
    "relative_k_reconstruction_error", "attention_logit_mse",
    "attention_score_kl", "topk_attention_overlap", "weighted_key_error",
    "relative_v_reconstruction_error", "attention_output_mse",
    "post_o_proj_mse", "weighted_value_error",
    "joint_attention_output_mse", "joint_post_o_proj_mse",
    "joint_attention_output_relative_error",
)

WORKER_EXIT_CODES = {0, 2, 3, 4, 5}


def classify_worker_failure(error: BaseException, stage: str, *, unusable: bool = False) -> int:
    """Map worker failures onto the documented orchestration contract."""
    if stage == "input":
        return 2
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return 3
    if stage == "load":
        return 4
    if unusable or stage in {"runtime", "capture_reset"}:
        return 5
    return 5


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("job index must be nonnegative")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--job-index", required=True, type=_nonnegative)
    return parser.parse_args(argv)


def apply_method_config(config: Any, method: str, values: dict[str, Any]) -> Any:
    if getattr(config, "model_type", None) != "llama":
        raise ValueError("experiment worker requires model_type == 'llama'")
    for name, value in values.items():
        setattr(config, name, value)
    if method == "kivi":
        config.cage_enable = False
        config.use_flash = True
    elif method == "cage":
        config.cage_enable = True
        config.cage_mode = "fake"
        config.use_flash = True
        # The shared KIVI attention constructor still requires this legacy
        # field; CAGE fake quantization uses its resolved per-bucket sizes.
        config.group_size = int(values["cage_k_group_sizes"][0]) if "cage_k_group_sizes" in values else 32
    elif method != "fp16":
        raise ValueError(f"unsupported method {method!r}")
    return config


def aggregate_layer_metrics(layer_records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for name in METRIC_NAMES:
        values = [float(record[name]) for record in layer_records
                  if isinstance(record.get(name), (int, float)) and not isinstance(record.get(name), bool)]
        if not values:
            continue
        result[name] = {"mean": float(statistics.fmean(values)),
                        "median": float(statistics.median(values)), "max": float(max(values))}
    return result


def attach_layer_context(layer_records: Sequence[dict[str, Any]], layer_memory: Sequence[dict[str, Any]],
                         run_id: str, method: str, prompt_length: int) -> None:
    if len(layer_records) != len(layer_memory):
        raise ValueError(f"layer metric count {len(layer_records)} does not equal "
                         f"layer memory count {len(layer_memory)}")
    for record, memory in zip(layer_records, layer_memory):
        record.update({"run_id": run_id, "method": method,
                       "prompt_length": prompt_length, "memory": memory})


def release_reference_output(reference_output: Any, cleanup=gc.collect) -> None:
    del reference_output
    cleanup()
    return None


def remove_stale_failure(output_dir: Path, run_id: str) -> None:
    try:
        (output_dir / "failures" / f"{run_id}.json").unlink()
    except FileNotFoundError:
        pass


def fp16_zero_layer_records(layer_count: int) -> list[dict[str, Any]]:
    return [{"layer_index": index, "phase": "teacher_forced_decode",
             "query_source": "fp16_reference_final_position",
             **{name: 0.0 for name in METRIC_NAMES}} for index in range(layer_count)]


def completed_run_exists(output_dir: Path, run_id: str) -> bool:
    path = output_dir / "runs" / f"{run_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def _load_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("methods") and "method_config" in raw["methods"][0]:
        return raw
    return load_and_resolve_manifest(path)


def _resolved_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _load_inputs_and_model(job: dict[str, Any], prompts: dict[str, Any]):
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

    reference = job["model"]["reference"]
    tokenizer = AutoTokenizer.from_pretrained(reference, use_fast=False)
    prepared = {(sample_id, length): prepare_prompt(tokenizer, prompts[sample_id].text, length)
                for sample_id in job["sample_ids"] for length in job["prompt_lengths"]}
    config = AutoConfig.from_pretrained(reference)
    apply_method_config(config, job["method"], job["method_config"])
    if job["method"] == "fp16":
        model_class = LlamaForCausalLM
    else:
        from models.llama_kivi import LlamaForCausalLM_KIVI
        model_class = LlamaForCausalLM_KIVI
    started = time.perf_counter()
    model = model_class.from_pretrained(reference, config=config,
                                        torch_dtype=_dtype(job["model"]["dtype"]),
                                        low_cpu_mem_usage=True)
    model.to(job["model"]["device"])
    model.eval()
    return tokenizer, prepared, model, time.perf_counter() - started


def _memory(method: str, past_key_values: Any, device: str):
    if method == "fp16":
        paper_layers = [summarize_fp16_cache_bytes(layer) for layer in past_key_values]
        runtime_layers = paper_layers
    else:
        paper_layers = [summarize_cache_bytes(layer) for layer in past_key_values]
        runtime_layers = [summarize_runtime_cache_bytes(layer) for layer in past_key_values]
    allocated = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    reserved = torch.cuda.max_memory_reserved() if device == "cuda" else 0
    total = build_memory_namespace(sum_cache_summaries(paper_layers),
                                   sum_cache_summaries(runtime_layers),
                                   max_allocated_bytes=allocated, max_reserved_bytes=reserved)
    layers = [build_memory_namespace(paper, runtime, max_allocated_bytes=0,
                                     max_reserved_bytes=0)
              for paper, runtime in zip(paper_layers, runtime_layers)]
    return total, layers


def _point_id(job: dict[str, Any], sample: Any, prompt_length: int, source: dict[str, Any]) -> str:
    return stable_run_id({"job": job, "sample_id": sample.sample_id,
                          "text_sha256": __import__("hashlib").sha256(sample.text.encode()).hexdigest(),
                          "prompt_length": prompt_length, "model": job["model"],
                          "source_state": source})


def run_job(manifest_path: str | Path, job_index: int) -> int:
    manifest_path = Path(manifest_path).resolve()
    manifest = _load_manifest(manifest_path)
    jobs = expand_jobs(manifest)
    if job_index < 0 or job_index >= len(jobs):
        raise ValueError(f"job-index {job_index} out of range for {len(jobs)} jobs")
    job = jobs[job_index]
    output_dir = _resolved_path(job["output_dir"], manifest_path)
    prompts = load_prompt_records(_resolved_path(job["prompts_file"], manifest_path))
    missing = sorted(set(job["sample_ids"]) - set(prompts))
    if missing:
        raise ValueError(f"selected sample_ids missing from prompts: {missing}")
    provenance = collect_provenance(ROOT)
    pending = []
    for sample_id in job["sample_ids"]:
        for prompt_length in job["prompt_lengths"]:
            run_id = _point_id(job, prompts[sample_id], prompt_length, provenance["source_state"])
            if completed_run_exists(output_dir, run_id):
                remove_stale_failure(output_dir, run_id)
            else:
                pending.append((sample_id, prompt_length, run_id))
    if not pending:
        return 0
    torch.manual_seed(job["measurement"]["seed"])
    try:
        tokenizer, prepared, model, load_duration = _load_inputs_and_model(job, prompts)
    except Exception as error:
        exit_code = classify_worker_failure(error, "load")
        category = "cuda_out_of_memory" if exit_code == 3 else type(error).__name__
        for _, _, run_id in pending:
            atomic_write_json(output_dir / "failures" / f"{run_id}.json",
                              {"schema_version": 1, "run_id": run_id, "status": "failed",
                               "category": category, "stage": "load",
                               "retryable": False, "message": str(error)})
        gc.collect()
        if job["model"]["device"] == "cuda":
            torch.cuda.empty_cache()
        return exit_code
    device = job["model"]["device"]
    model_record = {"reference": job["model"]["reference"], "dtype": job["model"]["dtype"],
                    "device": device, "model_type": getattr(model.config, "model_type", None)}
    outcome = 0
    for sample_id, prompt_length, run_id in pending:
        unusable = False
        point = prepared[(sample_id, prompt_length)]
        stage = "reference"
        prompt_ids = continuation_ids = full_ids = None
        prefill = reference_output = candidate_output = None
        memory_record = layer_memory = layer_records = None
        timing = {"load_seconds": float(load_duration)}
        try:
                prompt_ids = point["prompt_ids"].to(device)
                continuation_ids = point["continuation_ids"].to(device)
                full_ids = torch.cat((prompt_ids, continuation_ids), dim=-1)
                if job["method"] != "fp16":
                    begin_reference_capture(model)
                started = time.perf_counter()
                with torch.inference_mode():
                    reference_output = model(input_ids=full_ids, use_cache=False, return_dict=True)
                timing["reference_seconds"] = time.perf_counter() - started
                if job["method"] != "fp16":
                    begin_candidate_capture(model)
                reference_output = release_reference_output(reference_output)
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                stage = "prefill"
                started = time.perf_counter()
                with torch.inference_mode():
                    prefill = model(input_ids=prompt_ids, use_cache=True, return_dict=True)
                timing["prefill_seconds"] = time.perf_counter() - started
                memory_record, layer_memory = _memory(job["method"], prefill.past_key_values, device)
                stage = "decode"
                started = time.perf_counter()
                with torch.inference_mode():
                    candidate_output = model(input_ids=continuation_ids,
                                             past_key_values=prefill.past_key_values,
                                             use_cache=True, return_dict=True)
                timing["decode_seconds"] = time.perf_counter() - started
                stage = "metrics"
                started = time.perf_counter()
                layer_count = len(model.model.layers)
                layer_records = (fp16_zero_layer_records(layer_count) if job["method"] == "fp16"
                                 else collect_layer_metrics(model))
                attach_layer_context(layer_records, layer_memory, run_id,
                                     job["method"], prompt_length)
                timing["metric_seconds"] = time.perf_counter() - started
                input_record = {"sample_id": sample_id, "text_sha256": point["text_sha256"],
                                "prompt_length": prompt_length, "continuation_tokens": 1}
                quantization_record = {"method": job["method"], **job["method_config"]}
                run_record = {"schema_version": 1, "run_id": run_id, "status": "completed",
                              "model": model_record,
                              "method": {"name": job["method"], "resolved_config": job["method_config"]},
                              "input": input_record, "quantization": quantization_record,
                              "measurement": {"phase": "teacher_forced_decode",
                                              "query_source": "fp16_reference_final_position",
                                              "query_count": 1, "layer_count": len(layer_records)},
                              "memory": memory_record,
                              "metrics_aggregate": aggregate_layer_metrics(layer_records),
                              "runtime_diagnostics": timing, "provenance": provenance}
                atomic_write_jsonl(output_dir / "layers" / f"{run_id}.jsonl", layer_records)
                atomic_write_json(output_dir / "runs" / f"{run_id}.json", run_record)
                remove_stale_failure(output_dir, run_id)
        except Exception as error:
                category = "cuda_out_of_memory" if isinstance(error, torch.cuda.OutOfMemoryError) else type(error).__name__
                exit_code = classify_worker_failure(error, stage)
                atomic_write_json(output_dir / "failures" / f"{run_id}.json",
                                  {"schema_version": 1, "run_id": run_id, "status": "failed",
                                   "category": category, "stage": stage, "retryable": False,
                                   "message": str(error)})
                if outcome == 0:
                    outcome = exit_code
        finally:
                try:
                    reset_experiment_capture(model)
                except Exception as reset_error:
                    if job["method"] != "fp16":
                        outcome = classify_worker_failure(
                            reset_error, "capture_reset", unusable=True
                        )
                        unusable = True
                del prompt_ids, continuation_ids, full_ids, prefill
                del reference_output, candidate_output, memory_record, layer_memory, layer_records
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        if unusable:
            break
    del tokenizer, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return outcome


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_job(args.manifest, args.job_index)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"manifest/input error: {error}", file=sys.stderr)
        return classify_worker_failure(error, "input")


if __name__ == "__main__":
    raise SystemExit(main())
