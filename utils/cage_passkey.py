"""Deterministic native-context passkey inputs, identities, and result IO."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import posixpath
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils.cage_experiment_io import (
    _atomic_write,
    atomic_write_jsonl,
    stable_run_id,
)
from utils.cage_experiment_config import resolve_method


PASSKEY_SCHEMA_VERSION = 2
PROMPT_TEMPLATE_ID = "passkey-v1"
FILLER_ID = "neutral-facts-v1"
PASSKEY_PROMPT_LENGTHS = (512, 1024, 2048, 4032)
PASSKEY_POSITIONS_PERCENT = (10, 50, 90)
PASSKEY_KEY_SEED = 20260723
PASSKEY_MAX_NEW_TOKENS = 8
PASSKEY_GENERATION_SEED = 0
FIRST_FIVE_DIGIT_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")

STAGE_A_RAW_METHODS = (
    {"id": "fp16", "method": "fp16"},
)
STAGE_B_RAW_METHODS = (
    {"id": "fp16", "method": "fp16"},
    {
        "id": "kivi-g32-r32",
        "method": "kivi",
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 32,
        "residual_length": 32,
    },
    {
        "id": "kivi-g64-r64",
        "method": "kivi",
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 64,
        "residual_length": 64,
    },
    {"id": "cage-r32", "method": "cage", "residual_length": 32},
    {"id": "cage-r64", "method": "cage", "residual_length": 64},
)

PROMPT_PREFIX = (
    "There is a five-digit pass key hidden in the text below. Remember it and "
    "answer the final question with only the five digits.\n"
)
PASSKEY_STATEMENT = "\nThe pass key is {target}. Remember it. {target} is the pass key.\n"
PROMPT_QUERY = (
    "\nWhat is the pass key? Answer with only the five digits.\n"
    "The pass key is"
)
FILLER_TEXT = (
    " The grass is green. The sky is blue. The sun is yellow. "
    "A quiet river crosses the valley. Small clouds move above the hills."
)

RAW_MANIFEST_FIELDS = frozenset({
    "model",
    "methods",
    "prompt_template_id",
    "filler_id",
    "prompt_lengths",
    "passkey_positions_percent",
    "key_generation",
    "generation",
    "output_dir",
})
MODEL_FIELDS = frozenset({
    "reference", "dtype", "device", "max_position_embeddings",
})
KEY_GENERATION_FIELDS = frozenset({"seed", "count"})
GENERATION_CONFIG_FIELDS = frozenset({
    "max_new_tokens", "do_sample", "num_beams", "seed",
})
CASE_FIELDS = frozenset({
    "schema_version",
    "case_id",
    "status",
    "model",
    "method",
    "input",
    "generation",
    "runtime_diagnostics",
    "provenance",
})
CASE_MODEL_FIELDS = frozenset({
    "reference", "dtype", "device", "max_position_embeddings", "model_type",
})
CASE_METHOD_FIELDS = frozenset({"id", "name", "resolved_config"})
CASE_INPUT_FIELDS = frozenset({
    "prompt_template_id",
    "filler_id",
    "prompt_length",
    "position_percent",
    "actual_statement_position_fraction",
    "key_index",
    "target",
    "statement_token_start",
    "statement_token_end",
    "prompt_ids_sha256",
})
CASE_GENERATION_FIELDS = frozenset({
    "max_new_tokens",
    "do_sample",
    "num_beams",
    "seed",
    "generated_token_count",
    "response_text",
    "first_five_digit",
    "exact_match",
    "contains_target",
    "stopped_early",
})
RUNTIME_FIELDS = frozenset({
    "elapsed_seconds", "cuda_max_allocated_bytes", "cuda_max_reserved_bytes",
})


class PasskeyError(ValueError):
    """Raised for deterministic passkey manifest, input, or artifact failures."""


def _canonical_methods(raw_methods: Any) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(raw_methods, list) or not raw_methods:
        raise PasskeyError("methods must be a non-empty list")
    try:
        resolved = [
            resolve_method(value, index)
            for index, value in enumerate(raw_methods)
        ]
        stage_a = [
            resolve_method(value, index)
            for index, value in enumerate(STAGE_A_RAW_METHODS)
        ]
        stage_b = [
            resolve_method(value, index)
            for index, value in enumerate(STAGE_B_RAW_METHODS)
        ]
    except ValueError as error:
        raise PasskeyError(f"invalid passkey method configuration: {error}") from error
    if resolved == stage_a:
        return "stage_a", resolved
    if resolved == stage_b:
        return "stage_b", resolved
    raise PasskeyError(
        "passkey methods must equal the declared Stage-A FP16 method or the "
        "declared Stage-B FP16/KIVI/CAGE matrix"
    )


def _case_method(method: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": method["id"],
        "name": method["method"],
        "resolved_config": json.loads(json.dumps(method["method_config"])),
    }


def _allowed_case_methods() -> tuple[dict[str, Any], ...]:
    return tuple(
        _case_method(resolve_method(value, index))
        for index, value in enumerate(STAGE_B_RAW_METHODS)
    )

def _require_exact_fields(name: str, value: dict[str, Any], expected: frozenset[str]) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise PasskeyError(f"{name} missing required fields {missing}")
    if unknown:
        raise PasskeyError(f"{name} has unknown fields {unknown}")


def _require_object(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PasskeyError(f"{name} must be an object")
    return value


def _require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PasskeyError(f"{name} must be an integer >= {minimum}")
    return value


def _require_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise PasskeyError(f"{name} must be a boolean")
    return value


def _require_nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PasskeyError(f"{name} must be a non-empty string")
    return value


def _require_finite_real(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PasskeyError(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise PasskeyError(f"{name} must be a finite real number >= {minimum}")
    return number


def load_passkey_manifest(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a declared Stage-A or Stage-B manifest."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PasskeyError(f"cannot load passkey manifest {source}: {error}") from error
    manifest = _require_object("passkey manifest", raw)
    _require_exact_fields("passkey manifest", manifest, RAW_MANIFEST_FIELDS)

    model = _require_object("model", manifest["model"])
    _require_exact_fields("model", model, MODEL_FIELDS)
    _require_nonempty_string("model.reference", model["reference"])
    if model["dtype"] != "float16":
        raise PasskeyError("passkey model.dtype must equal 'float16'")
    if model["device"] != "cuda":
        raise PasskeyError("passkey model.device must equal 'cuda'")
    max_positions = _require_int(
        "model.max_position_embeddings", model["max_position_embeddings"], minimum=1
    )
    if max_positions != 4096:
        raise PasskeyError("passkey model.max_position_embeddings must equal 4096")

    protocol_stage, methods = _canonical_methods(manifest["methods"])
    if manifest["prompt_template_id"] != PROMPT_TEMPLATE_ID:
        raise PasskeyError(f"prompt_template_id must equal {PROMPT_TEMPLATE_ID!r}")
    if manifest["filler_id"] != FILLER_ID:
        raise PasskeyError(f"filler_id must equal {FILLER_ID!r}")

    lengths = manifest["prompt_lengths"]
    if not isinstance(lengths, list) or not lengths:
        raise PasskeyError("prompt_lengths must be a non-empty list")
    for index, length in enumerate(lengths):
        _require_int(f"prompt_lengths[{index}]", length, minimum=1)
    if len(lengths) != len(set(lengths)):
        raise PasskeyError("prompt_lengths must be unique")
    if tuple(lengths) != PASSKEY_PROMPT_LENGTHS:
        raise PasskeyError(
            f"passkey prompt_lengths must equal {list(PASSKEY_PROMPT_LENGTHS)}"
        )

    positions = manifest["passkey_positions_percent"]
    if not isinstance(positions, list) or not positions:
        raise PasskeyError("passkey_positions_percent must be a non-empty list")
    for index, position in enumerate(positions):
        _require_int(f"passkey_positions_percent[{index}]", position, minimum=1)
        if position >= 100:
            raise PasskeyError("passkey positions must be integers from 1 through 99")
    if len(positions) != len(set(positions)):
        raise PasskeyError("passkey_positions_percent must be unique")
    if tuple(positions) != PASSKEY_POSITIONS_PERCENT:
        raise PasskeyError(
            "passkey_positions_percent must equal "
            f"{list(PASSKEY_POSITIONS_PERCENT)}"
        )

    key_generation = _require_object("key_generation", manifest["key_generation"])
    _require_exact_fields("key_generation", key_generation, KEY_GENERATION_FIELDS)
    key_seed = _require_int("key_generation.seed", key_generation["seed"])
    if key_seed != PASSKEY_KEY_SEED:
        raise PasskeyError(f"passkey key_generation.seed must equal {PASSKEY_KEY_SEED}")
    key_count = _require_int("key_generation.count", key_generation["count"], minimum=1)
    if key_count not in {1, 5}:
        raise PasskeyError(
            "passkey key_generation.count must equal 1 (smoke) or 5 (full)"
        )

    generation = _require_object("generation", manifest["generation"])
    _require_exact_fields("generation", generation, GENERATION_CONFIG_FIELDS)
    max_new_tokens = _require_int(
        "generation.max_new_tokens", generation["max_new_tokens"], minimum=1
    )
    if max_new_tokens != PASSKEY_MAX_NEW_TOKENS:
        raise PasskeyError(
            f"passkey generation.max_new_tokens must equal {PASSKEY_MAX_NEW_TOKENS}"
        )
    if _require_bool("generation.do_sample", generation["do_sample"]):
        raise PasskeyError("passkey generation.do_sample must be false")
    if _require_int("generation.num_beams", generation["num_beams"], minimum=1) != 1:
        raise PasskeyError("passkey generation.num_beams must equal 1")
    generation_seed = _require_int("generation.seed", generation["seed"])
    if generation_seed != PASSKEY_GENERATION_SEED:
        raise PasskeyError(
            f"passkey generation.seed must equal {PASSKEY_GENERATION_SEED}"
        )
    if max(lengths) + max_new_tokens > max_positions:
        raise PasskeyError(
            "largest prompt plus max_new_tokens exceeds model.max_position_embeddings"
        )
    _require_nonempty_string("output_dir", manifest["output_dir"])
    resolved = json.loads(json.dumps(manifest))
    resolved["protocol_stage"] = protocol_stage
    resolved["methods"] = methods
    return resolved


def generate_passkeys(seed: int, count: int) -> list[str]:
    """Generate stable, unique five-digit keys without RNG-version dependence."""

    _require_int("seed", seed)
    _require_int("count", count, minimum=1)
    if count > 90000:
        raise PasskeyError("count cannot exceed the 90000 unique five-digit values")
    keys: list[str] = []
    seen: set[str] = set()
    nonce = 0
    while len(keys) < count:
        digest = hashlib.sha256(f"{seed}:{nonce}".encode("ascii")).digest()
        candidate = str(10000 + int.from_bytes(digest[:8], "big") % 90000)
        if candidate not in seen:
            seen.add(candidate)
            keys.append(candidate)
        nonce += 1
    return keys


def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise PasskeyError("tokenizer output must contain input_ids")
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list) or not ids or any(type(item) is not int for item in ids):
        raise PasskeyError("tokenizer input_ids must be a non-empty integer list")
    return ids


def _repeat_to_length(values: Sequence[int], length: int) -> list[int]:
    if not values:
        raise PasskeyError("filler token sequence must be non-empty")
    return [values[index % len(values)] for index in range(length)]


def prompt_ids_sha256(input_ids: Sequence[int]) -> str:
    canonical = json.dumps(list(input_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def build_passkey_prompt(
    tokenizer: Any,
    *,
    target: str,
    prompt_length: int,
    position_percent: int,
) -> dict[str, Any]:
    """Build exact-length token IDs with the statement near the requested position."""

    if not isinstance(target, str) or FIRST_FIVE_DIGIT_RE.fullmatch(target) is None:
        raise PasskeyError("target must be exactly five digits")
    _require_int("prompt_length", prompt_length, minimum=1)
    _require_int("position_percent", position_percent, minimum=1)
    if position_percent >= 100:
        raise PasskeyError("position_percent must be from 1 through 99")

    prefix_ids = _token_ids(tokenizer, PROMPT_PREFIX, add_special_tokens=True)
    statement_ids = _token_ids(
        tokenizer,
        PASSKEY_STATEMENT.format(target=target),
        add_special_tokens=False,
    )
    query_ids = _token_ids(tokenizer, PROMPT_QUERY, add_special_tokens=False)
    filler_ids = _token_ids(tokenizer, FILLER_TEXT, add_special_tokens=False)
    fixed_length = len(prefix_ids) + len(statement_ids) + len(query_ids)
    filler_length = prompt_length - fixed_length
    if filler_length < 0:
        raise PasskeyError(
            f"prompt_length {prompt_length} is shorter than fixed template length {fixed_length}"
        )

    minimum_start = len(prefix_ids)
    maximum_start = prompt_length - len(statement_ids) - len(query_ids)
    requested_start = round(prompt_length * position_percent / 100)
    statement_start = min(max(requested_start, minimum_start), maximum_start)
    before_length = statement_start - len(prefix_ids)
    after_length = filler_length - before_length
    if before_length < 0 or after_length < 0:
        raise PasskeyError("cannot place passkey statement within the requested prompt length")
    filler_stream = _repeat_to_length(filler_ids, filler_length)
    input_ids = (
        prefix_ids
        + filler_stream[:before_length]
        + statement_ids
        + filler_stream[before_length:before_length + after_length]
        + query_ids
    )
    if len(input_ids) != prompt_length:
        raise PasskeyError(
            f"constructed prompt has {len(input_ids)} tokens; expected {prompt_length}"
        )
    statement_end = statement_start + len(statement_ids)
    return {
        "input_ids": input_ids,
        "prompt_ids_sha256": prompt_ids_sha256(input_ids),
        "statement_token_start": statement_start,
        "statement_token_end": statement_end,
        "actual_statement_position_fraction": statement_start / prompt_length,
    }


def first_five_digit(response_text: str) -> str | None:
    if not isinstance(response_text, str):
        raise PasskeyError("response_text must be a string")
    match = FIRST_FIVE_DIGIT_RE.search(response_text)
    return match.group(1) if match else None


def normalized_model_identity(model: dict[str, Any]) -> dict[str, Any]:
    value = dict(model)
    value["reference"] = posixpath.normpath(value["reference"].replace("\\", "/"))
    return value


def passkey_case_identity(
    *,
    model: dict[str, Any],
    method: dict[str, Any],
    input_record: dict[str, Any],
    generation: dict[str, Any],
    source_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": normalized_model_identity({
            key: model[key]
            for key in ("reference", "dtype", "device", "max_position_embeddings")
        }),
        "method": method,
        "input": {
            key: input_record[key]
            for key in (
                "prompt_template_id",
                "filler_id",
                "prompt_length",
                "position_percent",
                "actual_statement_position_fraction",
                "key_index",
                "target",
                "statement_token_start",
                "statement_token_end",
                "prompt_ids_sha256",
            )
        },
        "generation": {
            key: generation[key]
            for key in ("max_new_tokens", "do_sample", "num_beams", "seed")
        },
        "source_state": source_state,
    }


def passkey_case_id(**kwargs: Any) -> str:
    return stable_run_id(passkey_case_identity(**kwargs))


def expand_passkey_cases(
    manifest: dict[str, Any], tokenizer: Any, source_state: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    keys = generate_passkeys(
        manifest["key_generation"]["seed"], manifest["key_generation"]["count"]
    )
    model = manifest["model"]
    prepared_inputs: dict[tuple[int, int, int], dict[str, Any]] = {}
    for prompt_length in manifest["prompt_lengths"]:
        for position_percent in manifest["passkey_positions_percent"]:
            for key_index, target in enumerate(keys):
                prepared_inputs[(prompt_length, position_percent, key_index)] = (
                    build_passkey_prompt(
                        tokenizer,
                        target=target,
                        prompt_length=prompt_length,
                        position_percent=position_percent,
                    )
                )
    cases = []
    seen_ids: set[str] = set()
    for resolved_method in manifest["methods"]:
        method = _case_method(resolved_method)
        for prompt_length in manifest["prompt_lengths"]:
            for position_percent in manifest["passkey_positions_percent"]:
                for key_index, target in enumerate(keys):
                    prepared = prepared_inputs[
                        (prompt_length, position_percent, key_index)
                    ]
                    input_record = {
                        "prompt_template_id": manifest["prompt_template_id"],
                        "filler_id": manifest["filler_id"],
                        "prompt_length": prompt_length,
                        "position_percent": position_percent,
                        "actual_statement_position_fraction": prepared[
                            "actual_statement_position_fraction"
                        ],
                        "key_index": key_index,
                        "target": target,
                        "statement_token_start": prepared["statement_token_start"],
                        "statement_token_end": prepared["statement_token_end"],
                        "prompt_ids_sha256": prepared["prompt_ids_sha256"],
                    }
                    case_id = passkey_case_id(
                        model=model,
                        method=method,
                        input_record=input_record,
                        generation=manifest["generation"],
                        source_state=source_state,
                    )
                    if case_id in seen_ids:
                        raise PasskeyError(f"duplicate expanded case ID {case_id}")
                    seen_ids.add(case_id)
                    cases.append({
                        "case_id": case_id,
                        "method": method,
                        "input": input_record,
                        "input_ids": prepared["input_ids"],
                    })
    return keys, cases


def resolved_passkey_manifest(
    manifest: dict[str, Any],
    *,
    generated_keys: Sequence[str],
    cases: Sequence[dict[str, Any]],
    source_state: dict[str, Any],
) -> dict[str, Any]:
    resolved = json.loads(json.dumps(manifest))
    resolved["generated_keys"] = list(generated_keys)
    resolved["source_state"] = source_state
    resolved["expanded_cases"] = [
        {
            "case_id": case["case_id"],
            "method": case["method"],
            "input": case["input"],
        }
        for case in cases
    ]
    return resolved


def validate_completed_passkey_case(
    output_dir: str | Path, case_id: str
) -> dict[str, Any]:
    path = Path(output_dir) / "cases" / f"{case_id}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PasskeyError(f"cannot load completed passkey case {path}: {error}") from error
    case = _require_object(f"completed passkey case {path}", record)
    _require_exact_fields(f"completed passkey case {path}", case, CASE_FIELDS)
    if type(case["schema_version"]) is not int or case["schema_version"] != PASSKEY_SCHEMA_VERSION:
        raise PasskeyError(
            f"completed passkey case schema_version must equal {PASSKEY_SCHEMA_VERSION}"
        )
    if case["case_id"] != case_id or path.stem != case_id:
        raise PasskeyError("completed passkey case ID must match filename and requested ID")
    if case["status"] != "completed":
        raise PasskeyError("completed passkey case status must equal 'completed'")

    model = _require_object("case.model", case["model"])
    _require_exact_fields("case.model", model, CASE_MODEL_FIELDS)
    for name in ("reference", "dtype", "device", "model_type"):
        _require_nonempty_string(f"case.model.{name}", model[name])
    if model["dtype"] != "float16" or model["device"] != "cuda" or model["model_type"] != "llama":
        raise PasskeyError("completed passkey case model must be float16 CUDA Llama")
    max_positions = _require_int(
        "case.model.max_position_embeddings", model["max_position_embeddings"], minimum=1
    )
    if max_positions != 4096:
        raise PasskeyError("completed passkey case max_position_embeddings must equal 4096")

    method = _require_object("case.method", case["method"])
    _require_exact_fields("case.method", method, CASE_METHOD_FIELDS)
    if method not in _allowed_case_methods():
        raise PasskeyError("completed passkey case method is outside the declared protocols")

    input_record = _require_object("case.input", case["input"])
    _require_exact_fields("case.input", input_record, CASE_INPUT_FIELDS)
    if input_record["prompt_template_id"] != PROMPT_TEMPLATE_ID:
        raise PasskeyError("case.input.prompt_template_id is invalid")
    if input_record["filler_id"] != FILLER_ID:
        raise PasskeyError("case.input.filler_id is invalid")
    prompt_length = _require_int("case.input.prompt_length", input_record["prompt_length"], minimum=1)
    if prompt_length not in PASSKEY_PROMPT_LENGTHS:
        raise PasskeyError("case.input.prompt_length is outside the declared passkey lengths")
    position = _require_int("case.input.position_percent", input_record["position_percent"], minimum=1)
    if position not in PASSKEY_POSITIONS_PERCENT:
        raise PasskeyError("case.input.position_percent is outside the declared passkey positions")
    actual_fraction = _require_finite_real(
        "case.input.actual_statement_position_fraction",
        input_record["actual_statement_position_fraction"],
        minimum=0.0,
    )
    if actual_fraction >= 1:
        raise PasskeyError("case.input.actual_statement_position_fraction must be below 1")
    key_index = _require_int("case.input.key_index", input_record["key_index"])
    if key_index >= 5:
        raise PasskeyError("case.input.key_index must be from 0 through 4")
    target = _require_nonempty_string("case.input.target", input_record["target"])
    if FIRST_FIVE_DIGIT_RE.fullmatch(target) is None:
        raise PasskeyError("case.input.target must be exactly five digits")
    statement_start = _require_int(
        "case.input.statement_token_start", input_record["statement_token_start"]
    )
    statement_end = _require_int(
        "case.input.statement_token_end", input_record["statement_token_end"], minimum=1
    )
    if not (statement_start < statement_end <= prompt_length):
        raise PasskeyError("case.input statement token bounds are invalid")
    if not math.isclose(actual_fraction, statement_start / prompt_length, rel_tol=0, abs_tol=1e-15):
        raise PasskeyError("case.input actual statement position fraction is inconsistent")
    if abs(actual_fraction - position / 100) > 1 / prompt_length:
        raise PasskeyError("case.input statement position differs from the requested position")
    digest = _require_nonempty_string(
        "case.input.prompt_ids_sha256", input_record["prompt_ids_sha256"]
    )
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise PasskeyError("case.input.prompt_ids_sha256 must be lowercase SHA-256")

    generation = _require_object("case.generation", case["generation"])
    _require_exact_fields("case.generation", generation, CASE_GENERATION_FIELDS)
    max_new_tokens = _require_int(
        "case.generation.max_new_tokens", generation["max_new_tokens"], minimum=1
    )
    if max_new_tokens != PASSKEY_MAX_NEW_TOKENS:
        raise PasskeyError("case.generation.max_new_tokens is outside the passkey protocol")
    _require_bool("case.generation.do_sample", generation["do_sample"])
    if generation["do_sample"]:
        raise PasskeyError("case.generation.do_sample must be false")
    if _require_int("case.generation.num_beams", generation["num_beams"], minimum=1) != 1:
        raise PasskeyError("case.generation.num_beams must equal 1")
    if _require_int("case.generation.seed", generation["seed"]) != PASSKEY_GENERATION_SEED:
        raise PasskeyError("case.generation.seed is outside the passkey protocol")
    if prompt_length + max_new_tokens > max_positions:
        raise PasskeyError("completed case prompt plus generation exceeds native context")
    generated_count = _require_int(
        "case.generation.generated_token_count", generation["generated_token_count"]
    )
    if generated_count > max_new_tokens:
        raise PasskeyError("generated_token_count exceeds max_new_tokens")
    response_text = generation["response_text"]
    if not isinstance(response_text, str):
        raise PasskeyError("case.generation.response_text must be a string")
    parsed = first_five_digit(response_text)
    if generation["first_five_digit"] != parsed:
        raise PasskeyError("case.generation.first_five_digit is inconsistent with response_text")
    expected_exact = parsed == target
    expected_contains = target in response_text
    if _require_bool("case.generation.exact_match", generation["exact_match"]) != expected_exact:
        raise PasskeyError("case.generation.exact_match is inconsistent")
    if _require_bool("case.generation.contains_target", generation["contains_target"]) != expected_contains:
        raise PasskeyError("case.generation.contains_target is inconsistent")
    if _require_bool("case.generation.stopped_early", generation["stopped_early"]) != (
        generated_count < max_new_tokens
    ):
        raise PasskeyError("case.generation.stopped_early is inconsistent")

    runtime = _require_object("case.runtime_diagnostics", case["runtime_diagnostics"])
    _require_exact_fields("case.runtime_diagnostics", runtime, RUNTIME_FIELDS)
    _require_finite_real("case.runtime_diagnostics.elapsed_seconds", runtime["elapsed_seconds"], minimum=0)
    _require_int(
        "case.runtime_diagnostics.cuda_max_allocated_bytes",
        runtime["cuda_max_allocated_bytes"],
    )
    _require_int(
        "case.runtime_diagnostics.cuda_max_reserved_bytes",
        runtime["cuda_max_reserved_bytes"],
    )
    provenance = _require_object("case.provenance", case["provenance"])
    source_state = _require_object("case.provenance.source_state", provenance.get("source_state"))
    if source_state.get("dirty") is not False:
        raise PasskeyError("completed passkey case source state must be clean")

    expected_id = passkey_case_id(
        model=model,
        method=method,
        input_record=input_record,
        generation=generation,
        source_state=source_state,
    )
    if expected_id != case_id:
        raise PasskeyError(
            f"completed passkey case ID mismatch: expected {expected_id}, got {case_id}"
        )
    return case


def is_valid_completed_passkey_case(output_dir: str | Path, case_id: str) -> bool:
    try:
        validate_completed_passkey_case(output_dir, case_id)
    except PasskeyError:
        return False
    return True


def aggregate_passkey_cases(
    output_dir: str | Path, *, expected_case_ids: Iterable[str]
) -> list[dict[str, Any]]:
    root = Path(output_dir)
    records = []
    for case_id in sorted(set(expected_case_ids)):
        path = root / "cases" / f"{case_id}.json"
        if not path.is_file():
            continue
        try:
            records.append(validate_completed_passkey_case(root, case_id))
        except PasskeyError:
            # Invalid stale artifacts are not resumable and remain available
            # for diagnosis until a successful atomic replacement is written.
            continue
    records.sort(key=lambda record: record["case_id"])
    atomic_write_jsonl(root / "summary" / "cases.jsonl", records)

    fields = [
        "case_id", "method", "prompt_length", "position_percent", "key_index",
        "target", "generated_token_count", "first_five_digit", "exact_match",
        "contains_target", "stopped_early", "response_text", "elapsed_seconds",
    ]
    rows = [{
        "case_id": record["case_id"],
        "method": record["method"]["id"],
        "prompt_length": record["input"]["prompt_length"],
        "position_percent": record["input"]["position_percent"],
        "key_index": record["input"]["key_index"],
        "target": record["input"]["target"],
        "generated_token_count": record["generation"]["generated_token_count"],
        "first_five_digit": record["generation"]["first_five_digit"],
        "exact_match": record["generation"]["exact_match"],
        "contains_target": record["generation"]["contains_target"],
        "stopped_early": record["generation"]["stopped_early"],
        "response_text": record["generation"]["response_text"],
        "elapsed_seconds": record["runtime_diagnostics"]["elapsed_seconds"],
    } for record in records]

    def write_csv(handle: Any) -> None:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    _atomic_write(root / "summary" / "cases.csv", write_csv)
    return records


__all__ = [
    "FILLER_ID",
    "PASSKEY_SCHEMA_VERSION",
    "PROMPT_TEMPLATE_ID",
    "PasskeyError",
    "aggregate_passkey_cases",
    "build_passkey_prompt",
    "expand_passkey_cases",
    "first_five_digit",
    "generate_passkeys",
    "is_valid_completed_passkey_case",
    "load_passkey_manifest",
    "passkey_case_id",
    "passkey_case_identity",
    "prompt_ids_sha256",
    "resolved_passkey_manifest",
    "validate_completed_passkey_case",
]
