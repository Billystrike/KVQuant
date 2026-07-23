"""Run deterministic native-context FP16/KIVI/CAGE passkey cases."""

from __future__ import annotations

import argparse
import copy
import gc
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cage_experiment_io import (
    atomic_write_json,
    collect_provenance,
    source_state_identity,
)
from utils.cage_experiment_config import apply_method_config
from utils.cage_passkey import (
    PASSKEY_SCHEMA_VERSION,
    PasskeyError,
    aggregate_passkey_cases,
    expand_passkey_cases,
    first_five_digit,
    is_valid_completed_passkey_case,
    load_passkey_manifest,
    resolved_passkey_manifest,
)


EXIT_SUCCESS = 0
EXIT_PREFLIGHT = 2
EXIT_OOM = 3
EXIT_MODEL_LOAD = 4
EXIT_CASE_ERROR = 6


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args(argv)


def _resolved_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def validate_native_model_context(config: Any, model: dict[str, Any]) -> None:
    if getattr(config, "model_type", None) != "llama":
        raise PasskeyError("passkey evaluation requires model_type == 'llama'")
    actual = getattr(config, "max_position_embeddings", None)
    expected = model["max_position_embeddings"]
    if type(actual) is not int or actual != expected:
        raise PasskeyError(
            f"actual config.max_position_embeddings {actual!r} does not equal "
            f"manifest value {expected}"
        )
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None:
        raise PasskeyError(
            f"native-context passkey calibration requires rope_scaling null, got {rope_scaling!r}"
        )


def _load_preflight(model: dict[str, Any]):
    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception as error:  # pragma: no cover - environment dependent.
        raise PasskeyError(f"cannot import transformers model helpers: {error}") from error
    reference = model["reference"]
    config = AutoConfig.from_pretrained(reference)
    validate_native_model_context(config, model)
    tokenizer = AutoTokenizer.from_pretrained(reference, use_fast=False)
    return config, tokenizer


def _load_model(model_config: dict[str, Any], config: Any, method: str):
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
    return model


def _remove_stale_failure(output_dir: Path, case_id: str) -> None:
    try:
        (output_dir / "failures" / f"{case_id}.json").unlink()
    except FileNotFoundError:
        pass


def _failure_record(case_id: str, category: str, stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": PASSKEY_SCHEMA_VERSION,
        "case_id": case_id,
        "status": "failed",
        "category": category,
        "stage": stage,
        "retryable": False,
        "message": str(error),
    }


def _cuda_peaks(device: str) -> tuple[int, int]:
    if device != "cuda":
        return 0, 0
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def _attach_quality_counts(result: dict[str, Any], records: Sequence[dict[str, Any]]) -> None:
    exact_matches = sum(record["generation"]["exact_match"] for record in records)
    contains_matches = sum(record["generation"]["contains_target"] for record in records)
    result["exact_matches"] = exact_matches
    result["contains_matches"] = contains_matches
    result["exact_match_accuracy"] = exact_matches / len(records) if records else 0.0


def _attach_quality_summary(
    result: dict[str, Any], manifest: dict[str, Any], records: Sequence[dict[str, Any]]
) -> None:
    methods = []
    all_cells = []
    expected_per_method = (
        len(manifest["prompt_lengths"])
        * len(manifest["passkey_positions_percent"])
        * manifest["key_generation"]["count"]
    )
    for method in manifest["methods"]:
        method_id = method["id"]
        method_records = [
            record for record in records if record["method"]["id"] == method_id
        ]
        cells = []
        for prompt_length in manifest["prompt_lengths"]:
            for position_percent in manifest["passkey_positions_percent"]:
                cell = [
                    record
                    for record in method_records
                    if record["input"]["prompt_length"] == prompt_length
                    and record["input"]["position_percent"] == position_percent
                ]
                cell_summary = {
                    "method_id": method_id,
                    "prompt_length": prompt_length,
                    "position_percent": position_percent,
                    "exact_matches": sum(
                        record["generation"]["exact_match"] for record in cell
                    ),
                    "contains_matches": sum(
                        record["generation"]["contains_target"] for record in cell
                    ),
                    "completed_cases": len(cell),
                    "expected_cases": manifest["key_generation"]["count"],
                }
                cells.append(cell_summary)
                all_cells.append(cell_summary)
        lengths = []
        for prompt_length in manifest["prompt_lengths"]:
            length_records = [
                record
                for record in method_records
                if record["input"]["prompt_length"] == prompt_length
            ]
            lengths.append({
                "prompt_length": prompt_length,
                "exact_matches": sum(
                    record["generation"]["exact_match"]
                    for record in length_records
                ),
                "contains_matches": sum(
                    record["generation"]["contains_target"]
                    for record in length_records
                ),
                "completed_cases": len(length_records),
                "expected_cases": (
                    len(manifest["passkey_positions_percent"])
                    * manifest["key_generation"]["count"]
                ),
            })
        positions = []
        for position_percent in manifest["passkey_positions_percent"]:
            position_records = [
                record
                for record in method_records
                if record["input"]["position_percent"] == position_percent
            ]
            positions.append({
                "position_percent": position_percent,
                "exact_matches": sum(
                    record["generation"]["exact_match"]
                    for record in position_records
                ),
                "contains_matches": sum(
                    record["generation"]["contains_target"]
                    for record in position_records
                ),
                "completed_cases": len(position_records),
                "expected_cases": (
                    len(manifest["prompt_lengths"])
                    * manifest["key_generation"]["count"]
                ),
            })
        exact_matches = sum(
            record["generation"]["exact_match"] for record in method_records
        )
        contains_matches = sum(
            record["generation"]["contains_target"] for record in method_records
        )
        methods.append({
            "method_id": method_id,
            "method": method["method"],
            "resolved_config": method["method_config"],
            "completed_cases": len(method_records),
            "expected_cases": expected_per_method,
            "exact_matches": exact_matches,
            "contains_matches": contains_matches,
            "exact_match_accuracy": (
                exact_matches / len(method_records) if method_records else 0.0
            ),
            "lengths": lengths,
            "positions": positions,
            "cells": cells,
        })
    result["method_quality"] = methods
    result["quality_cells"] = all_cells
    expected_total = expected_per_method * len(manifest["methods"])
    complete = (
        len(records) == expected_total
        and result.get("failure_records", 0) == 0
        and all(
            method["completed_cases"] == method["expected_cases"]
            for method in methods
        )
    )
    result["completion_gate"] = "PASS" if complete else "INCOMPLETE"
    if not complete:
        result["quality_gate"] = "INCOMPLETE"
        return
    if manifest["protocol_stage"] == "stage_b":
        result["quality_gate"] = "NOT_APPLICABLE"
        return
    fp16 = methods[0]
    if manifest["key_generation"]["count"] == 1:
        passed = all(cell["exact_matches"] == 1 for cell in fp16["cells"])
    else:
        passed = (
            fp16["exact_match_accuracy"] >= 0.9
            and all(cell["exact_matches"] >= 4 for cell in fp16["cells"])
        )
    result["quality_gate"] = "PASS" if passed else "FAIL"


def _write_quality_summary(output_dir: Path, result: dict[str, Any]) -> None:
    atomic_write_json(
        output_dir / "summary" / "quality.json",
        {
            "schema_version": PASSKEY_SCHEMA_VERSION,
            "protocol_stage": result["protocol_stage"],
            "expanded_cases": result["expanded_cases"],
            "completed_cases": result["completed_cases"],
            "failure_records": result["failure_records"],
            "exact_matches": result["exact_matches"],
            "contains_matches": result["contains_matches"],
            "exact_match_accuracy": result["exact_match_accuracy"],
            "completion_gate": result["completion_gate"],
            "quality_gate": result["quality_gate"],
            "methods": result["method_quality"],
            "cells": result["quality_cells"],
        },
    )


def run_case(
    *,
    case: dict[str, Any],
    manifest: dict[str, Any],
    tokenizer: Any,
    model: Any,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    device = manifest["model"]["device"]
    generation_config = manifest["generation"]
    input_ids = torch.tensor([case["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=generation_config["max_new_tokens"],
            do_sample=generation_config["do_sample"],
            num_beams=generation_config["num_beams"],
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    continuation = generated[0, input_ids.shape[1]:]
    generated_count = int(continuation.shape[0])
    response_text = tokenizer.decode(continuation.tolist(), skip_special_tokens=True)
    parsed = first_five_digit(response_text)
    target = case["input"]["target"]
    max_allocated, max_reserved = _cuda_peaks(device)
    return {
        "schema_version": PASSKEY_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "status": "completed",
        "model": {
            **manifest["model"],
            "model_type": str(getattr(model.config, "model_type", "")),
        },
        "method": case["method"],
        "input": case["input"],
        "generation": {
            **generation_config,
            "generated_token_count": generated_count,
            "response_text": response_text,
            "first_five_digit": parsed,
            "exact_match": parsed == target,
            "contains_target": target in response_text,
            "stopped_early": generated_count < generation_config["max_new_tokens"],
        },
        "runtime_diagnostics": {
            "elapsed_seconds": elapsed,
            "cuda_max_allocated_bytes": max_allocated,
            "cuda_max_reserved_bytes": max_reserved,
        },
        "provenance": provenance,
    }


def _refresh_result(
    result: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
    expected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    completed = aggregate_passkey_cases(output_dir, expected_case_ids=expected_ids)
    result["completed_cases"] = len(completed)
    result["failure_records"] = sum(
        (output_dir / "failures" / f"{case_id}.json").is_file()
        for case_id in expected_ids
    )
    _attach_quality_counts(result, completed)
    _attach_quality_summary(result, manifest, completed)
    _write_quality_summary(output_dir, result)
    return completed


def _record_method_failures(
    output_dir: Path,
    cases: Sequence[dict[str, Any]],
    category: str,
    stage: str,
    error: BaseException,
) -> None:
    for case in cases:
        atomic_write_json(
            output_dir / "failures" / f"{case['case_id']}.json",
            _failure_record(case["case_id"], category, stage, error),
        )


def run_manifest(manifest_path: str | Path) -> tuple[int, dict[str, Any]]:
    path = Path(manifest_path).resolve()
    try:
        manifest = load_passkey_manifest(path)
        source_state = source_state_identity(ROOT)
        if source_state["dirty"]:
            raise PasskeyError(
                "formal passkey evaluation requires a clean tracked source state"
            )
        native_config, tokenizer = _load_preflight(manifest["model"])
        generated_keys, cases = expand_passkey_cases(manifest, tokenizer, source_state)
        output_dir = _resolved_path(manifest["output_dir"], path)
        resolved = resolved_passkey_manifest(
            manifest,
            generated_keys=generated_keys,
            cases=cases,
            source_state=source_state,
        )
        atomic_write_json(output_dir / "manifest.resolved.json", resolved)
    except Exception as error:
        return EXIT_PREFLIGHT, {"error": str(error), "stage": "preflight"}

    expected_ids = [case["case_id"] for case in cases]
    reusable = [
        case_id
        for case_id in expected_ids
        if is_valid_completed_passkey_case(output_dir, case_id)
    ]
    for case_id in reusable:
        _remove_stale_failure(output_dir, case_id)
    reusable_set = set(reusable)
    pending = [case for case in cases if case["case_id"] not in reusable_set]
    result: dict[str, Any] = {
        "git_commit": source_state["git_commit"],
        "source_dirty": source_state["dirty"],
        "protocol_stage": manifest["protocol_stage"],
        "generated_keys": ",".join(generated_keys),
        "native_max_position_embeddings": native_config.max_position_embeddings,
        "native_rope_scaling": getattr(native_config, "rope_scaling", None),
        "expanded_cases": len(cases),
        "valid_reusable_cases": len(reusable),
        "remaining_cases": len(pending),
    }
    _refresh_result(result, manifest, output_dir, expected_ids)
    if not pending:
        return EXIT_SUCCESS, result

    torch.manual_seed(manifest["generation"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manifest["generation"]["seed"])
    provenance = collect_provenance(
        ROOT, deterministic_seed=manifest["generation"]["seed"]
    )
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
            apply_method_config(
                method_config, method["method"], method["method_config"]
            )
        except Exception as error:
            _record_method_failures(
                output_dir,
                method_pending,
                "passkey_method_config_error",
                "configuration",
                error,
            )
            _refresh_result(result, manifest, output_dir, expected_ids)
            return EXIT_PREFLIGHT, {
                **result,
                "error": str(error),
                "stage": "configuration",
            }

        try:
            model = _load_model(
                manifest["model"], method_config, method["method"]
            )
        except torch.cuda.OutOfMemoryError as error:  # pragma: no cover
            _record_method_failures(
                output_dir, method_pending, "cuda_out_of_memory", "load", error
            )
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            _refresh_result(result, manifest, output_dir, expected_ids)
            return EXIT_OOM, {**result, "error": str(error), "stage": "load"}
        except Exception as error:  # pragma: no cover - environment dependent.
            _record_method_failures(
                output_dir, method_pending, "model_load_error", "load", error
            )
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            _refresh_result(result, manifest, output_dir, expected_ids)
            return EXIT_MODEL_LOAD, {
                **result,
                "error": str(error),
                "stage": "load",
            }

        try:
            for case in method_pending:
                try:
                    record = run_case(
                        case=case,
                        manifest=manifest,
                        tokenizer=tokenizer,
                        model=model,
                        provenance=provenance,
                    )
                    atomic_write_json(
                        output_dir / "cases" / f"{case['case_id']}.json", record
                    )
                    _remove_stale_failure(output_dir, case["case_id"])
                except torch.cuda.OutOfMemoryError as error:
                    atomic_write_json(
                        output_dir / "failures" / f"{case['case_id']}.json",
                        _failure_record(
                            case["case_id"],
                            "cuda_out_of_memory",
                            "generation",
                            error,
                        ),
                    )
                    _refresh_result(result, manifest, output_dir, expected_ids)
                    return EXIT_OOM, {
                        **result,
                        "error": str(error),
                        "stage": "generation",
                    }
                except Exception as error:
                    case_failed = True
                    atomic_write_json(
                        output_dir / "failures" / f"{case['case_id']}.json",
                        _failure_record(
                            case["case_id"],
                            "passkey_case_error",
                            "generation",
                            error,
                        ),
                    )
        finally:
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    _refresh_result(result, manifest, output_dir, expected_ids)
    return (EXIT_CASE_ERROR if case_failed else EXIT_SUCCESS), result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        exit_code, result = run_manifest(args.manifest)
    except Exception as error:  # Last-resort classification for output/process failures.
        print(f"passkey error [runner]: {error}", file=sys.stderr)
        return EXIT_CASE_ERROR
    for key in (
        "git_commit",
        "source_dirty",
        "protocol_stage",
        "generated_keys",
        "native_max_position_embeddings",
        "native_rope_scaling",
        "expanded_cases",
        "valid_reusable_cases",
        "remaining_cases",
        "completed_cases",
        "failure_records",
        "exact_matches",
        "contains_matches",
        "exact_match_accuracy",
        "completion_gate",
    ):
        if key in result:
            print(f"{key}={result[key]}")
    if "error" in result:
        print(f"passkey error [{result.get('stage', 'unknown')}]: {result['error']}", file=sys.stderr)
    for method in result.get("method_quality", []):
        print(
            f"method_exact[{method['method_id']}]="
            f"{method['exact_matches']}/{method['expected_cases']}"
        )
        print(
            f"method_contains[{method['method_id']}]="
            f"{method['contains_matches']}/{method['expected_cases']}"
        )
        print(
            f"method_exact_accuracy[{method['method_id']}]="
            f"{method['exact_match_accuracy']}"
        )
        for length in method["lengths"]:
            print(
                f"length_exact[{method['method_id']},{length['prompt_length']}]="
                f"{length['exact_matches']}/{length['expected_cases']}"
            )
        for position in method["positions"]:
            print(
                f"position_exact[{method['method_id']},"
                f"{position['position_percent']}]="
                f"{position['exact_matches']}/{position['expected_cases']}"
            )
    for cell in result.get("quality_cells", []):
        print(
            f"cell_exact[{cell['method_id']},{cell['prompt_length']},"
            f"{cell['position_percent']}]="
            f"{cell['exact_matches']}/{cell['expected_cases']}"
        )
    if "completion_gate" in result:
        print(f"PASSKEY_COMPLETION_GATE={result['completion_gate']}")
    if "quality_gate" in result:
        print(f"PASSKEY_QUALITY_GATE={result['quality_gate']}")
    if exit_code == EXIT_SUCCESS:
        print("PASSKEY_RESULT=PASS")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
