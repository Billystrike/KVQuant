"""Run deterministic native-context cache-conditioned continuation PPL cases."""

from __future__ import annotations

import argparse
import copy
import gc
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cage_experiment_config import apply_method_config
from utils.cage_experiment_io import (
    atomic_write_json,
    collect_provenance,
    source_state_identity,
)
from utils.cage_ppl import (
    PPL_CONTINUATION_TOKENS,
    PPL_SCHEMA_VERSION,
    PPLError,
    aggregate_ppl_cases,
    expand_ppl_cases,
    is_valid_completed_ppl_case,
    load_ppl_manifest,
    resolved_ppl_manifest,
    validate_completed_ppl_case,
    validate_corpus_snapshot,
)


EXIT_SUCCESS = 0
EXIT_PREFLIGHT = 2
EXIT_OOM = 3
EXIT_MODEL_LOAD = 4
EXIT_CASE_ERROR = 6
SEED = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args(argv)


def _resolved_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def validate_native_model_context(config: Any, model: dict[str, Any]) -> None:
    if getattr(config, "model_type", None) != "llama":
        raise PPLError("PPL evaluation requires model_type == 'llama'")
    actual = getattr(config, "max_position_embeddings", None)
    if actual != model["max_position_embeddings"]:
        raise PPLError(
            f"actual max_position_embeddings {actual!r} differs from manifest"
        )
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None:
        raise PPLError(f"native-context PPL requires rope_scaling null, got {rope_scaling!r}")


def _load_preflight(manifest: dict[str, Any]):
    try:
        import datasets
        from datasets import load_dataset
        from transformers import AutoConfig, AutoTokenizer
    except Exception as error:  # pragma: no cover - environment dependent.
        raise PPLError(f"cannot import PPL preflight dependencies: {error}") from error

    model_config = AutoConfig.from_pretrained(manifest["model"]["reference"])
    validate_native_model_context(model_config, manifest["model"])
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"]["reference"], use_fast=False
    )
    corpus = manifest["corpus"]
    dataset = load_dataset(
        corpus["id"],
        corpus["config"],
        split=corpus["split"],
        revision=corpus["revision"],
    )
    snapshot, token_ids = validate_corpus_snapshot(
        manifest,
        list(dataset["text"]),
        tokenizer,
        dataset_fingerprint=getattr(dataset, "_fingerprint", None),
        datasets_version=datasets.__version__,
    )
    return model_config, tokenizer, snapshot, token_ids


def _load_model(model_config: dict[str, Any], config: Any, method: str):
    started = time.perf_counter()
    if method == "fp16":
        from transformers import LlamaForCausalLM

        model_class = LlamaForCausalLM
    else:
        from models.llama_kivi import LlamaForCausalLM_KIVI

        model_class = LlamaForCausalLM_KIVI
    model = model_class.from_pretrained(
        model_config["reference"],
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(model_config["device"])
    model.eval()
    return model, time.perf_counter() - started


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _token_nll(logits: torch.Tensor, target: torch.Tensor) -> float:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise PPLError(f"expected one-token logits [1, vocab], got {tuple(logits.shape)}")
    value = F.cross_entropy(logits.float(), target.reshape(1), reduction="sum")
    result = float(value.detach().cpu())
    if not math.isfinite(result) or result < 0:
        raise PPLError(f"non-finite token NLL {result!r}")
    return result


def _one_shot_nlls(
    model: Any, prompt: torch.Tensor, continuation: torch.Tensor
) -> list[float]:
    full = torch.cat((prompt, continuation), dim=-1)
    with torch.inference_mode():
        output = model(input_ids=full, use_cache=False, return_dict=True)
    start = prompt.shape[-1] - 1
    stop = start + continuation.shape[-1]
    logits = output.logits[:, start:stop, :]
    if logits.shape[1] != PPL_CONTINUATION_TOKENS:
        raise PPLError("FP16 one-shot logits do not cover all continuation targets")
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        continuation.reshape(-1),
        reduction="none",
    )
    values = [float(value) for value in losses.detach().cpu().tolist()]
    if len(values) != PPL_CONTINUATION_TOKENS or any(
        not math.isfinite(value) or value < 0 for value in values
    ):
        raise PPLError("FP16 one-shot reference contains invalid NLL values")
    del output, logits, losses, full
    return values


def _score_from_nlls(
    token_nlls: Sequence[float], reference_nlls: Sequence[float] | None
) -> dict[str, Any]:
    if len(token_nlls) != PPL_CONTINUATION_TOKENS:
        raise PPLError("incremental scoring must produce 64 token NLLs")
    all_sum = math.fsum(token_nlls)
    decode = token_nlls[1:]
    decode_sum = math.fsum(decode)
    all_mean = all_sum / len(token_nlls)
    decode_mean = decode_sum / len(decode)
    reference = None
    if reference_nlls is not None:
        if len(reference_nlls) != PPL_CONTINUATION_TOKENS:
            raise PPLError("FP16 reference must contain 64 token NLLs")
        reference_sum = math.fsum(reference_nlls)
        deltas = [
            abs(left - right)
            for left, right in zip(token_nlls, reference_nlls)
        ]
        reference = {
            "token_nlls": list(reference_nlls),
            "nll_sum": reference_sum,
            "mean_nll": reference_sum / len(reference_nlls),
            "perplexity": math.exp(reference_sum / len(reference_nlls)),
            "mean_absolute_token_nll_delta": math.fsum(deltas) / len(deltas),
            "max_absolute_token_nll_delta": max(deltas),
        }
    result = {
        "primary_metric": "cache_dependent_decode_nll",
        "boundary_target_count": 1,
        "decode_target_count": len(decode),
        "all_target_count": len(token_nlls),
        "token_nlls": list(token_nlls),
        "boundary_nll": token_nlls[0],
        "decode_nll_sum": decode_sum,
        "decode_mean_nll": decode_mean,
        "decode_perplexity": math.exp(decode_mean),
        "all_nll_sum": all_sum,
        "all_mean_nll": all_mean,
        "all_perplexity": math.exp(all_mean),
        "fp16_one_shot_reference": reference,
    }
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise PPLError(f"non-finite aggregate scoring value {key}")
    return result


def run_case(
    *, case: dict[str, Any], manifest: dict[str, Any], model: Any,
    provenance: dict[str, Any], load_seconds: float,
) -> dict[str, Any]:
    device = manifest["model"]["device"]
    prompt = torch.tensor([case["prompt_ids"]], dtype=torch.long, device=device)
    continuation = torch.tensor(
        [case["continuation_ids"]], dtype=torch.long, device=device
    )
    if prompt.shape[-1] != case["input"]["prompt_length"]:
        raise PPLError("prepared prompt length differs from case identity")
    if continuation.shape[-1] != PPL_CONTINUATION_TOKENS:
        raise PPLError("prepared continuation length differs from protocol")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    total_started = time.perf_counter()
    reference_nlls = None
    reference_seconds = 0.0
    if case["method"]["name"] == "fp16":
        _sync(device)
        started = time.perf_counter()
        reference_nlls = _one_shot_nlls(model, prompt, continuation)
        _sync(device)
        reference_seconds = time.perf_counter() - started

    _sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=prompt, use_cache=True, return_dict=True)
    boundary_nll = _token_nll(output.logits[:, -1, :], continuation[:, 0])
    past_key_values = output.past_key_values
    if past_key_values is None:
        raise PPLError("prefill did not return past_key_values")
    del output
    _sync(device)
    prefill_seconds = time.perf_counter() - started

    token_nlls = [boundary_nll]
    _sync(device)
    started = time.perf_counter()
    for index in range(PPL_CONTINUATION_TOKENS - 1):
        with torch.inference_mode():
            output = model(
                input_ids=continuation[:, index:index + 1],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        token_nlls.append(
            _token_nll(output.logits[:, -1, :], continuation[:, index + 1])
        )
        past_key_values = output.past_key_values
        if past_key_values is None:
            raise PPLError("decode step did not return past_key_values")
        del output
    _sync(device)
    decode_seconds = time.perf_counter() - started
    scoring = _score_from_nlls(token_nlls, reference_nlls)
    elapsed = time.perf_counter() - total_started
    allocated = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    reserved = int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0

    record = {
        "schema_version": PPL_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "status": "completed",
        "model": {
            **manifest["model"],
            "model_type": getattr(model.config, "model_type", None),
        },
        "method": case["method"],
        "input": case["input"],
        "scoring": scoring,
        "runtime_diagnostics": {
            "load_seconds": float(load_seconds),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "reference_seconds": reference_seconds,
            "elapsed_seconds": elapsed,
            "cuda_max_allocated_bytes": allocated,
            "cuda_max_reserved_bytes": reserved,
        },
        "provenance": provenance,
    }
    del past_key_values, prompt, continuation
    return record


def _remove_stale_failure(output_dir: Path, case_id: str) -> None:
    try:
        (output_dir / "failures" / f"{case_id}.json").unlink()
    except FileNotFoundError:
        pass


def _failure(case_id: str, category: str, stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": PPL_SCHEMA_VERSION,
        "case_id": case_id,
        "status": "failed",
        "category": category,
        "stage": stage,
        "retryable": False,
        "message": str(error),
    }


def _record_failures(
    output_dir: Path, cases: Sequence[dict[str, Any]], category: str,
    stage: str, error: BaseException,
) -> None:
    for case in cases:
        atomic_write_json(
            output_dir / "failures" / f"{case['case_id']}.json",
            _failure(case["case_id"], category, stage, error),
        )


def _quality_summary(
    manifest: dict[str, Any], records: Sequence[dict[str, Any]],
    failure_records: int, expected_cases: int,
) -> dict[str, Any]:
    methods = []
    lengths = []
    for method in manifest["methods"]:
        subset = [record for record in records if record["method"]["id"] == method["id"]]
        token_count = sum(record["scoring"]["decode_target_count"] for record in subset)
        nll_sum = math.fsum(record["scoring"]["decode_nll_sum"] for record in subset)
        methods.append({
            "method_id": method["id"],
            "completed_cases": len(subset),
            "expected_cases": len(manifest["prompt_lengths"]) * len(manifest["anchor_indices"]),
            "decode_target_count": token_count,
            "decode_nll_sum": nll_sum,
            "decode_mean_nll": nll_sum / token_count if token_count else None,
            "decode_perplexity": math.exp(nll_sum / token_count) if token_count else None,
        })
        for prompt_length in manifest["prompt_lengths"]:
            length_subset = [
                record for record in subset
                if record["input"]["prompt_length"] == prompt_length
            ]
            length_tokens = sum(
                record["scoring"]["decode_target_count"] for record in length_subset
            )
            length_nll = math.fsum(
                record["scoring"]["decode_nll_sum"] for record in length_subset
            )
            lengths.append({
                "method_id": method["id"],
                "prompt_length": prompt_length,
                "completed_cases": len(length_subset),
                "expected_cases": len(manifest["anchor_indices"]),
                "decode_target_count": length_tokens,
                "decode_nll_sum": length_nll,
                "decode_mean_nll": length_nll / length_tokens if length_tokens else None,
                "decode_perplexity": math.exp(length_nll / length_tokens) if length_tokens else None,
            })
    fp16_deltas = [
        record["scoring"]["fp16_one_shot_reference"]
        for record in records if record["method"]["name"] == "fp16"
    ]
    complete = len(records) == expected_cases and failure_records == 0
    return {
        "schema_version": PPL_SCHEMA_VERSION,
        "protocol_stage": manifest["protocol_stage"],
        "expected_cases": expected_cases,
        "completed_cases": len(records),
        "failure_records": failure_records,
        "completion_gate": "PASS" if complete else "INCOMPLETE",
        "quality_gate": "NOT_APPLICABLE",
        "method_quality": methods,
        "length_quality": lengths,
        "fp16_calibration_cases": len(fp16_deltas),
        "fp16_calibration_max_mean_abs_token_nll_delta": (
            max(value["mean_absolute_token_nll_delta"] for value in fp16_deltas)
            if fp16_deltas else None
        ),
        "fp16_calibration_max_abs_token_nll_delta": (
            max(value["max_absolute_token_nll_delta"] for value in fp16_deltas)
            if fp16_deltas else None
        ),
    }


def _refresh(
    result: dict[str, Any], manifest: dict[str, Any], output_dir: Path,
    expected_ids: Sequence[str],
) -> None:
    records = aggregate_ppl_cases(output_dir, expected_case_ids=expected_ids)
    failures = sum(
        (output_dir / "failures" / f"{case_id}.json").is_file()
        for case_id in expected_ids
    )
    quality = _quality_summary(manifest, records, failures, len(expected_ids))
    atomic_write_json(output_dir / "summary" / "quality.json", quality)
    result.update(quality)


def run_manifest(manifest_path: str | Path) -> tuple[int, dict[str, Any]]:
    path = Path(manifest_path).resolve()
    try:
        manifest = load_ppl_manifest(path)
        source_state = source_state_identity(ROOT)
        if source_state["dirty"]:
            raise PPLError("formal PPL evaluation requires a clean tracked source state")
        native_config, tokenizer, snapshot, token_ids = _load_preflight(manifest)
        cases = expand_ppl_cases(manifest, token_ids, source_state)
        output_dir = _resolved_path(manifest["output_dir"], path)
        atomic_write_json(
            output_dir / "manifest.resolved.json",
            resolved_ppl_manifest(manifest, source_state, snapshot, cases),
        )
    except Exception as error:
        return EXIT_PREFLIGHT, {"error": str(error), "stage": "preflight"}

    expected_ids = [case["case_id"] for case in cases]
    reusable = [
        case_id for case_id in expected_ids
        if is_valid_completed_ppl_case(output_dir, case_id)
    ]
    for case_id in reusable:
        _remove_stale_failure(output_dir, case_id)
    reusable_set = set(reusable)
    pending = [case for case in cases if case["case_id"] not in reusable_set]
    result: dict[str, Any] = {
        "git_commit": source_state["git_commit"],
        "source_dirty": source_state["dirty"],
        "protocol_stage": manifest["protocol_stage"],
        "corpus_token_count": snapshot["token_count"],
        "corpus_token_ids_sha256": snapshot["token_ids_sha256"],
        "native_max_position_embeddings": native_config.max_position_embeddings,
        "native_rope_scaling": getattr(native_config, "rope_scaling", None),
        "expanded_cases": len(cases),
        "valid_reusable_cases": len(reusable),
        "remaining_cases": len(pending),
    }
    _refresh(result, manifest, output_dir, expected_ids)
    if not pending:
        return EXIT_SUCCESS, result

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    provenance = collect_provenance(ROOT, deterministic_seed=SEED)
    case_failed = False
    device = manifest["model"]["device"]
    for method in manifest["methods"]:
        method_pending = [
            case for case in pending if case["method"]["id"] == method["id"]
        ]
        if not method_pending:
            continue
        method_config = copy.deepcopy(native_config)
        try:
            apply_method_config(method_config, method["method"], method["method_config"])
        except Exception as error:
            _record_failures(output_dir, method_pending, "ppl_method_config_error", "configuration", error)
            _refresh(result, manifest, output_dir, expected_ids)
            return EXIT_PREFLIGHT, {**result, "error": str(error), "stage": "configuration"}
        try:
            model, load_seconds = _load_model(manifest["model"], method_config, method["method"])
        except torch.cuda.OutOfMemoryError as error:  # pragma: no cover
            _record_failures(output_dir, method_pending, "cuda_out_of_memory", "load", error)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            _refresh(result, manifest, output_dir, expected_ids)
            return EXIT_OOM, {**result, "error": str(error), "stage": "load"}
        except Exception as error:  # pragma: no cover
            _record_failures(output_dir, method_pending, "model_load_error", "load", error)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            _refresh(result, manifest, output_dir, expected_ids)
            return EXIT_MODEL_LOAD, {**result, "error": str(error), "stage": "load"}

        try:
            for case in method_pending:
                try:
                    record = run_case(
                        case=case, manifest=manifest, model=model,
                        provenance=provenance, load_seconds=load_seconds,
                    )
                    atomic_write_json(output_dir / "cases" / f"{case['case_id']}.json", record)
                    validate_completed_ppl_case(output_dir, case["case_id"])
                    _remove_stale_failure(output_dir, case["case_id"])
                except torch.cuda.OutOfMemoryError as error:
                    atomic_write_json(
                        output_dir / "failures" / f"{case['case_id']}.json",
                        _failure(case["case_id"], "cuda_out_of_memory", "scoring", error),
                    )
                    _refresh(result, manifest, output_dir, expected_ids)
                    return EXIT_OOM, {**result, "error": str(error), "stage": "scoring"}
                except Exception as error:
                    case_failed = True
                    atomic_write_json(
                        output_dir / "failures" / f"{case['case_id']}.json",
                        _failure(case["case_id"], "ppl_case_error", "scoring", error),
                    )
        finally:
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    _refresh(result, manifest, output_dir, expected_ids)
    return (EXIT_CASE_ERROR if case_failed else EXIT_SUCCESS), result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        exit_code, result = run_manifest(args.manifest)
    except Exception as error:
        print(f"PPL error [runner]: {error}", file=sys.stderr)
        return EXIT_CASE_ERROR
    for key in (
        "git_commit", "source_dirty", "protocol_stage", "corpus_token_count",
        "corpus_token_ids_sha256", "native_max_position_embeddings",
        "native_rope_scaling", "expanded_cases", "valid_reusable_cases",
        "remaining_cases", "completed_cases", "failure_records",
        "completion_gate", "quality_gate", "fp16_calibration_cases",
        "fp16_calibration_max_mean_abs_token_nll_delta",
        "fp16_calibration_max_abs_token_nll_delta",
    ):
        if key in result:
            print(f"{key}={result[key]}")
    for method in result.get("method_quality", []):
        print(
            f"method_ppl[{method['method_id']}]={method['decode_perplexity']} "
            f"cases={method['completed_cases']}/{method['expected_cases']} "
            f"tokens={method['decode_target_count']}"
        )
    for length in result.get("length_quality", []):
        print(
            f"length_ppl[{length['method_id']},{length['prompt_length']}]="
            f"{length['decode_perplexity']} "
            f"cases={length['completed_cases']}/{length['expected_cases']}"
        )
    if "completion_gate" in result:
        print(f"PPL_COMPLETION_GATE={result['completion_gate']}")
    if "quality_gate" in result:
        print(f"PPL_QUALITY_GATE={result['quality_gate']}")
    if exit_code == EXIT_SUCCESS:
        print("PPL_RESULT=PASS")
    elif "error" in result:
        print(f"PPL error [{result.get('stage', 'unknown')}]: {result['error']}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
