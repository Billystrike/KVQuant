"""Run one resolved CAGE experiment configuration, one point at a time."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import posixpath
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cage_experiment_config import (
    expand_jobs, load_and_resolve_manifest, validate_resolved_manifest,
)
from utils.cage_experiment_hooks import (
    begin_candidate_capture, begin_reference_capture, collect_layer_metrics,
    reset_experiment_capture,
)
from utils.cage_experiment_io import (
    atomic_write_json, atomic_write_jsonl, collect_provenance,
    load_prompt_records, prepare_prompt, stable_run_id,
)
from utils.cage_experiment_schema import (
    ExperimentPointError,
    METRIC_NAMES,
    SCHEMA_VERSION,
    aggregate_layer_metrics,
    is_valid_completed_point,
    validate_completed_artifacts,
)
from utils.cage_memory import (
    build_memory_namespace, sum_cache_summaries, summarize_cache_bytes,
    summarize_cache_structure, summarize_fp16_cache_bytes, summarize_runtime_cache_bytes,
)


WORKER_EXIT_CODES = {0, 2, 3, 4, 5, 6}


class FailureClassification(NamedTuple):
    category: str
    stage: str
    exit_code: int
    retryable: bool


def classify_worker_failure(
    error: BaseException, stage: str, *, unusable: bool = False,
) -> FailureClassification:
    """Map worker failures onto the documented orchestration contract."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return FailureClassification("cuda_out_of_memory", stage, 3, False)
    if stage == "input":
        return FailureClassification("input_error", stage, 2, False)
    if stage == "load":
        return FailureClassification("model_load_error", stage, 4, False)
    if unusable and stage == "capture_reset":
        return FailureClassification("transient_model_state", stage, 5, True)
    return FailureClassification("experiment_point_error", stage, 6, False)


def failure_record(
    run_id: str, error: BaseException, classification: FailureClassification,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "failed",
        "category": classification.category,
        "stage": classification.stage,
        "retryable": classification.retryable,
        "message": str(error),
    }


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


def attach_layer_context(layer_records: Sequence[dict[str, Any]], layer_memory: Sequence[dict[str, Any]],
                         run_id: str, method: str, prompt_length: int) -> None:
    if len(layer_records) != len(layer_memory):
        raise ValueError(f"layer metric count {len(layer_records)} does not equal "
                         f"layer memory count {len(layer_memory)}")
    for record, memory in zip(layer_records, layer_memory):
        record.update({"schema_version": SCHEMA_VERSION, "run_id": run_id, "method": method,
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
             **{
                 name: (1.0 if name == "topk_attention_overlap" else 0.0)
                 for name in METRIC_NAMES
             }} for index in range(layer_count)]


def completed_run_exists(output_dir: Path, run_id: str) -> bool:
    return is_valid_completed_point(output_dir, run_id)


def _load_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    methods = raw.get("methods") if isinstance(raw, dict) else None
    if isinstance(methods, list) and any(
        isinstance(method, dict) and "method_config" in method
        for method in methods
    ):
        return validate_resolved_manifest(raw)
    return load_and_resolve_manifest(path)


def _resolved_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _prepare_inputs(job: dict[str, Any], prompts: dict[str, Any]):
    from transformers import AutoTokenizer

    reference = job["model"]["reference"]
    tokenizer = AutoTokenizer.from_pretrained(reference, use_fast=False)
    prepared = {(sample_id, length): prepare_prompt(tokenizer, prompts[sample_id].text, length)
                for sample_id in job["sample_ids"] for length in job["prompt_lengths"]}
    return tokenizer, prepared


def validate_native_model_context(config: Any, model: dict[str, Any]) -> None:
    """Reject resolved/native context drift and explicit RoPE extension."""

    actual = getattr(config, "max_position_embeddings", None)
    expected = model["max_position_embeddings"]
    if type(actual) is not int or actual != expected:
        raise ValueError(
            f"actual config.max_position_embeddings {actual!r} does not equal resolved "
            f"manifest model.max_position_embeddings {expected}"
        )
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None:
        raise ValueError(
            f"native-context pilot requires config.rope_scaling to be null, got "
            f"{rope_scaling!r}"
        )


def _load_native_config(job: dict[str, Any]) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(job["model"]["reference"])


def _load_model(job: dict[str, Any], config: Any):
    from transformers import LlamaForCausalLM

    reference = job["model"]["reference"]
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
    return model, time.perf_counter() - started


def _memory(
    method: str,
    past_key_values: Any,
    *,
    prompt_length: int,
    residual_length: int | None,
):
    if method == "fp16":
        paper_layers = [summarize_fp16_cache_bytes(layer) for layer in past_key_values]
        runtime_layers = paper_layers
    else:
        paper_layers = [summarize_cache_bytes(layer) for layer in past_key_values]
        runtime_layers = [summarize_runtime_cache_bytes(layer) for layer in past_key_values]
    total = build_memory_namespace(sum_cache_summaries(paper_layers),
                                   sum_cache_summaries(runtime_layers),
                                   max_allocated_bytes=0, max_reserved_bytes=0)
    layers = [build_memory_namespace(paper, runtime, max_allocated_bytes=0,
                                     max_reserved_bytes=0)
              for paper, runtime in zip(paper_layers, runtime_layers)]
    for layer, cache in zip(layers, past_key_values):
        layer["cache_structure"] = summarize_cache_structure(
            method,
            cache,
            prompt_length=prompt_length,
            residual_length=residual_length,
        )
    return total, layers


def update_point_cuda_diagnostic(memory: dict[str, Any], device: str) -> None:
    if device == "cuda":
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
    else:
        allocated = reserved = 0
    memory["cuda_peak_diagnostic"] = {
        "max_allocated_bytes": allocated,
        "max_reserved_bytes": reserved,
    }


def require_finite_tensor_tree(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ExperimentPointError(f"{name} contains non-finite values")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite_tensor_tree(f"{name}.{key}", child)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite_tensor_tree(f"{name}[{index}]", child)


def _normalized_model_identity(model: dict[str, Any]) -> dict[str, Any]:
    identity = dict(model)
    reference = identity.get("reference")
    if isinstance(reference, str):
        identity["reference"] = posixpath.normpath(reference.replace("\\", "/"))
    return identity


def point_identity(
    job: dict[str, Any], sample: Any, prompt_length: int, source: dict[str, Any],
) -> dict[str, Any]:
    """Return only the scientific identity local to one experiment point."""

    return {
        "model": _normalized_model_identity(job["model"]),
        "method": {
            "name": job["method"],
            "resolved_config": job["method_config"],
        },
        "measurement": job["measurement"],
        "input": {
            "sample_id": sample.sample_id,
            "text_sha256": hashlib.sha256(sample.text.encode("utf-8")).hexdigest(),
            "prompt_length": prompt_length,
        },
        "source_state": source,
    }


def _point_id(job: dict[str, Any], sample: Any, prompt_length: int, source: dict[str, Any]) -> str:
    return stable_run_id(point_identity(job, sample, prompt_length, source))


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
    seed = job["measurement"]["seed"]
    torch.manual_seed(seed)
    provenance = collect_provenance(ROOT, deterministic_seed=seed)
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
    try:
        config = _load_native_config(job)
    except Exception as error:
        classification = classify_worker_failure(error, "load")
        for _, _, run_id in pending:
            atomic_write_json(
                output_dir / "failures" / f"{run_id}.json",
                failure_record(run_id, error, classification),
            )
        return classification.exit_code
    try:
        validate_native_model_context(config, job["model"])
        apply_method_config(config, job["method"], job["method_config"])
    except Exception as error:
        classification = classify_worker_failure(error, "input")
        for _, _, run_id in pending:
            atomic_write_json(
                output_dir / "failures" / f"{run_id}.json",
                failure_record(run_id, error, classification),
            )
        return classification.exit_code
    try:
        tokenizer, prepared = _prepare_inputs(job, prompts)
    except Exception as error:
        classification = classify_worker_failure(error, "input")
        for _, _, run_id in pending:
            atomic_write_json(
                output_dir / "failures" / f"{run_id}.json",
                failure_record(run_id, error, classification),
            )
        return classification.exit_code
    try:
        model, load_duration = _load_model(job, config)
    except Exception as error:
        classification = classify_worker_failure(error, "load")
        for _, _, run_id in pending:
            atomic_write_json(output_dir / "failures" / f"{run_id}.json",
                              failure_record(run_id, error, classification))
        gc.collect()
        if job["model"]["device"] == "cuda":
            torch.cuda.empty_cache()
        del tokenizer
        return classification.exit_code
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
            require_finite_tensor_tree("reference logits", getattr(reference_output, "logits", None))
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
            require_finite_tensor_tree("prefill cache", prefill.past_key_values)
            timing["prefill_seconds"] = time.perf_counter() - started
            memory_record, layer_memory = _memory(
                job["method"],
                prefill.past_key_values,
                prompt_length=prompt_length,
                residual_length=job["method_config"].get("residual_length"),
            )
            stage = "decode"
            started = time.perf_counter()
            with torch.inference_mode():
                candidate_output = model(input_ids=continuation_ids,
                                         past_key_values=prefill.past_key_values,
                                         use_cache=True, return_dict=True)
            require_finite_tensor_tree("candidate logits", getattr(candidate_output, "logits", None))
            require_finite_tensor_tree(
                "candidate cache", getattr(candidate_output, "past_key_values", None)
            )
            timing["decode_seconds"] = time.perf_counter() - started
            stage = "metrics"
            started = time.perf_counter()
            layer_count = len(model.model.layers)
            layer_records = (fp16_zero_layer_records(layer_count) if job["method"] == "fp16"
                             else collect_layer_metrics(model))
            attach_layer_context(layer_records, layer_memory, run_id,
                                 job["method"], prompt_length)
            timing["metric_seconds"] = time.perf_counter() - started
            update_point_cuda_diagnostic(memory_record, device)
            input_record = {"sample_id": sample_id, "text_sha256": point["text_sha256"],
                            "prompt_length": prompt_length, "continuation_tokens": 1}
            quantization_record = {"method": job["method"], **job["method_config"]}
            run_record = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "status": "completed",
                          "model": model_record,
                          "method": {"name": job["method"], "resolved_config": job["method_config"]},
                          "input": input_record, "quantization": quantization_record,
                          "measurement": {"phase": "teacher_forced_decode",
                                          "query_source": "fp16_reference_final_position",
                                          "query_count": 1, "layer_count": len(layer_records)},
                          "memory": memory_record,
                          "metrics_aggregate": aggregate_layer_metrics(layer_records),
                          "runtime_diagnostics": timing, "provenance": provenance}
            validate_completed_artifacts(run_record, layer_records, expected_run_id=run_id)
            atomic_write_jsonl(output_dir / "layers" / f"{run_id}.jsonl", layer_records)
            atomic_write_json(output_dir / "runs" / f"{run_id}.json", run_record)
            remove_stale_failure(output_dir, run_id)
        except Exception as error:
            classification = classify_worker_failure(error, stage)
            atomic_write_json(output_dir / "failures" / f"{run_id}.json",
                              failure_record(run_id, error, classification))
            if outcome == 0:
                outcome = classification.exit_code
        finally:
            try:
                reset_experiment_capture(model)
            except Exception as reset_error:
                if job["method"] != "fp16":
                    classification = classify_worker_failure(
                        reset_error, "capture_reset", unusable=True
                    )
                    atomic_write_json(
                        output_dir / "failures" / f"{run_id}.json",
                        failure_record(run_id, reset_error, classification),
                    )
                    outcome = classification.exit_code
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
        return classify_worker_failure(error, "input").exit_code


if __name__ == "__main__":
    raise SystemExit(main())
