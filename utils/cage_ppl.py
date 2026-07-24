"""Strict inputs, identities, schema, and IO for cache-conditioned PPL."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import posixpath
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils.cage_experiment_config import resolve_method
from utils.cage_experiment_io import _atomic_write, atomic_write_jsonl, stable_run_id


PPL_SCHEMA_VERSION = 1
PPL_PROMPT_LENGTHS = (512, 1024, 2048, 4032)
PPL_CONTINUATION_TOKENS = 64
PPL_DECODE_TARGETS = 63
PPL_ANCHOR_INDICES = (0, 1, 2, 3, 4)
PPL_SELECTION_ID = "interior-sixths-v1"

WIKITEXT_ID = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_SPLIT = "test"
WIKITEXT_REVISION = "00aa25585682d4957f9e86edc73f59be7419af99"
WIKITEXT_JOIN_SEPARATOR = "\n\n"
WIKITEXT_ROWS = 4358
WIKITEXT_NONEMPTY_ROWS = 2891
WIKITEXT_UTF8_BYTES = 1296370
WIKITEXT_TEXT_SHA256 = (
    "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83"
)
WIKITEXT_TOKEN_COUNT = 341468
WIKITEXT_TOKEN_IDS_SHA256 = (
    "8163e5b39c668be8eec1e4d82eaecb1ee2a09d958f81873773e93bf4a4484f10"
)

PPL_ACCEPTANCE_RAW_METHODS = (
    {"id": "fp16", "method": "fp16"},
    {
        "id": "kivi-g32-r32",
        "method": "kivi",
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 32,
        "residual_length": 32,
    },
    {"id": "cage-r32", "method": "cage", "residual_length": 32},
)

PPL_FULL_RAW_METHODS = (
    {"id": "fp16", "method": "fp16"},
    {
        "id": "kivi-g32-r32", "method": "kivi", "k_bits": 2,
        "v_bits": 2, "group_size": 32, "residual_length": 32,
    },
    {
        "id": "kivi-g32-r64", "method": "kivi", "k_bits": 2,
        "v_bits": 2, "group_size": 32, "residual_length": 64,
    },
    {
        "id": "kivi-g32-r128", "method": "kivi", "k_bits": 2,
        "v_bits": 2, "group_size": 32, "residual_length": 128,
    },
    {
        "id": "kivi-g64-r64", "method": "kivi", "k_bits": 2,
        "v_bits": 2, "group_size": 64, "residual_length": 64,
    },
    {
        "id": "kivi-g64-r128", "method": "kivi", "k_bits": 2,
        "v_bits": 2, "group_size": 64, "residual_length": 128,
    },
    {
        "id": "kivi-g128-r128", "method": "kivi", "k_bits": 2,
        "v_bits": 2, "group_size": 128, "residual_length": 128,
    },
    {"id": "cage-r32", "method": "cage", "residual_length": 32},
    {"id": "cage-r64", "method": "cage", "residual_length": 64},
    {"id": "cage-r128", "method": "cage", "residual_length": 128},
)

RAW_FIELDS = frozenset({
    "model", "corpus", "methods", "prompt_lengths", "anchor_indices",
    "measurement", "output_dir",
})
MODEL_FIELDS = frozenset({
    "reference", "dtype", "device", "max_position_embeddings",
})
CORPUS_FIELDS = frozenset({
    "id", "config", "split", "revision", "join_separator",
    "add_special_tokens", "expected_rows", "expected_nonempty_rows",
    "expected_joined_utf8_bytes", "expected_joined_text_sha256",
    "expected_token_count", "expected_token_ids_sha256",
})
MEASUREMENT_FIELDS = frozenset({"selection_id", "continuation_tokens"})
CASE_FIELDS = frozenset({
    "schema_version", "case_id", "status", "model", "method", "input",
    "scoring", "runtime_diagnostics", "provenance",
})
CASE_MODEL_FIELDS = frozenset({
    "reference", "dtype", "device", "max_position_embeddings", "model_type",
})
CASE_METHOD_FIELDS = frozenset({"id", "name", "resolved_config"})
CASE_INPUT_FIELDS = frozenset({
    "corpus_id", "corpus_config", "corpus_split", "corpus_revision",
    "corpus_token_ids_sha256", "selection_id", "anchor_index",
    "continuation_start", "prompt_length", "continuation_tokens",
    "prompt_ids_sha256", "continuation_ids_sha256", "full_ids_sha256",
})
SCORING_FIELDS = frozenset({
    "primary_metric", "boundary_target_count", "decode_target_count",
    "all_target_count", "token_nlls", "boundary_nll", "decode_nll_sum",
    "decode_mean_nll", "decode_perplexity", "all_nll_sum", "all_mean_nll",
    "all_perplexity", "fp16_one_shot_reference",
})
FP16_REFERENCE_FIELDS = frozenset({
    "token_nlls", "nll_sum", "mean_nll", "perplexity",
    "mean_absolute_token_nll_delta", "max_absolute_token_nll_delta",
})
RUNTIME_FIELDS = frozenset({
    "load_seconds", "prefill_seconds", "decode_seconds",
    "reference_seconds", "elapsed_seconds", "cuda_max_allocated_bytes",
    "cuda_max_reserved_bytes",
})


class PPLError(ValueError):
    """Raised for PPL manifest, corpus, identity, or artifact failures."""


def _require_object(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PPLError(f"{name} must be an object")
    return value


def _require_fields(name: str, value: dict[str, Any], fields: frozenset[str]) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing:
        raise PPLError(f"{name} missing required fields {missing}")
    if unknown:
        raise PPLError(f"{name} has unknown fields {unknown}")


def _require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PPLError(f"{name} must be an integer >= {minimum}")
    return value


def _require_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise PPLError(f"{name} must be a non-empty string")
    return value


def _finite(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PPLError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise PPLError(f"{name} must be a finite real number >= {minimum}")
    return result


def _resolved_methods(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        return [resolve_method(value, index) for index, value in enumerate(values)]
    except ValueError as error:
        raise PPLError(f"invalid PPL method configuration: {error}") from error


def _case_method(method: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": method["id"],
        "name": method["method"],
        "resolved_config": json.loads(json.dumps(method["method_config"])),
    }


def load_ppl_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PPLError(f"cannot load PPL manifest {source}: {error}") from error
    manifest = _require_object("PPL manifest", manifest)
    _require_fields("PPL manifest", manifest, RAW_FIELDS)

    model = _require_object("model", manifest["model"])
    _require_fields("model", model, MODEL_FIELDS)
    _require_string("model.reference", model["reference"])
    if model["dtype"] != "float16" or model["device"] != "cuda":
        raise PPLError("PPL model requires dtype='float16' and device='cuda'")
    if _require_int("model.max_position_embeddings", model["max_position_embeddings"], minimum=1) != 4096:
        raise PPLError("PPL model.max_position_embeddings must equal 4096")

    corpus = _require_object("corpus", manifest["corpus"])
    _require_fields("corpus", corpus, CORPUS_FIELDS)
    expected_corpus = {
        "id": WIKITEXT_ID,
        "config": WIKITEXT_CONFIG,
        "split": WIKITEXT_SPLIT,
        "revision": WIKITEXT_REVISION,
        "join_separator": WIKITEXT_JOIN_SEPARATOR,
        "add_special_tokens": False,
        "expected_rows": WIKITEXT_ROWS,
        "expected_nonempty_rows": WIKITEXT_NONEMPTY_ROWS,
        "expected_joined_utf8_bytes": WIKITEXT_UTF8_BYTES,
        "expected_joined_text_sha256": WIKITEXT_TEXT_SHA256,
        "expected_token_count": WIKITEXT_TOKEN_COUNT,
        "expected_token_ids_sha256": WIKITEXT_TOKEN_IDS_SHA256,
    }
    if corpus != expected_corpus:
        raise PPLError("PPL corpus must equal the frozen WikiText-2 raw test snapshot")

    methods = _resolved_methods(manifest["methods"])
    acceptance = _resolved_methods(PPL_ACCEPTANCE_RAW_METHODS)
    full = _resolved_methods(PPL_FULL_RAW_METHODS)
    if methods == acceptance:
        protocol_stage = "acceptance"
    elif methods == full:
        protocol_stage = "full"
    else:
        raise PPLError("PPL methods must equal the acceptance or full matrix")

    lengths = manifest["prompt_lengths"]
    anchors = manifest["anchor_indices"]
    if protocol_stage == "acceptance":
        if lengths != [512, 4032] or anchors != [0]:
            raise PPLError("PPL acceptance requires lengths [512, 4032] and anchor [0]")
    else:
        if tuple(lengths) != PPL_PROMPT_LENGTHS or tuple(anchors) != PPL_ANCHOR_INDICES:
            raise PPLError("PPL full matrix requires all declared lengths and anchors")

    measurement = _require_object("measurement", manifest["measurement"])
    _require_fields("measurement", measurement, MEASUREMENT_FIELDS)
    if measurement != {
        "selection_id": PPL_SELECTION_ID,
        "continuation_tokens": PPL_CONTINUATION_TOKENS,
    }:
        raise PPLError("PPL measurement differs from the frozen scoring protocol")
    if max(lengths) + PPL_CONTINUATION_TOKENS > 4096:
        raise PPLError("PPL prompt plus continuation exceeds native context")
    _require_string("output_dir", manifest["output_dir"])

    resolved = json.loads(json.dumps(manifest))
    resolved["protocol_stage"] = protocol_stage
    resolved["methods"] = methods
    return resolved


def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise PPLError("tokenizer output must contain input_ids")
    values = encoded["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    if not isinstance(values, list) or any(type(item) is not int for item in values):
        raise PPLError("tokenizer input_ids must be an integer list")
    return values


def token_ids_sha256(values: Sequence[int]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_corpus_snapshot(
    manifest: dict[str, Any],
    texts: Sequence[str],
    tokenizer: Any,
    *,
    dataset_fingerprint: str | None,
    datasets_version: str,
) -> tuple[dict[str, Any], list[int]]:
    corpus = manifest["corpus"]
    if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
        raise PPLError("WikiText rows must be a sequence of strings")
    if any(not isinstance(text, str) for text in texts):
        raise PPLError("WikiText rows must all be strings")
    joined = corpus["join_separator"].join(texts)
    joined_bytes = joined.encode("utf-8")
    token_ids = _token_ids(
        tokenizer, joined, add_special_tokens=corpus["add_special_tokens"]
    )
    actual = {
        "rows": len(texts),
        "nonempty_rows": sum(bool(text.strip()) for text in texts),
        "joined_utf8_bytes": len(joined_bytes),
        "joined_text_sha256": hashlib.sha256(joined_bytes).hexdigest(),
        "token_count": len(token_ids),
        "token_ids_sha256": token_ids_sha256(token_ids),
    }
    expected = {
        "rows": corpus["expected_rows"],
        "nonempty_rows": corpus["expected_nonempty_rows"],
        "joined_utf8_bytes": corpus["expected_joined_utf8_bytes"],
        "joined_text_sha256": corpus["expected_joined_text_sha256"],
        "token_count": corpus["expected_token_count"],
        "token_ids_sha256": corpus["expected_token_ids_sha256"],
    }
    if actual != expected:
        raise PPLError(f"WikiText corpus/token identity mismatch: {actual!r}")
    snapshot = {
        **actual,
        "dataset_fingerprint": dataset_fingerprint,
        "datasets_version": _require_string("datasets_version", datasets_version),
        "tokenizer_type": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    return snapshot, token_ids


def continuation_anchor(token_count: int, anchor_index: int) -> int:
    _require_int("token_count", token_count, minimum=1)
    _require_int("anchor_index", anchor_index)
    if anchor_index not in PPL_ANCHOR_INDICES:
        raise PPLError(f"anchor_index must be one of {list(PPL_ANCHOR_INDICES)}")
    minimum = max(PPL_PROMPT_LENGTHS)
    maximum = token_count - PPL_CONTINUATION_TOKENS
    if maximum < minimum:
        raise PPLError("token stream is too short for the PPL protocol")
    return minimum + ((maximum - minimum) * (anchor_index + 1) // 6)


def normalized_model_identity(model: dict[str, Any]) -> dict[str, Any]:
    value = dict(model)
    value["reference"] = posixpath.normpath(value["reference"].replace("\\", "/"))
    return value


def ppl_case_identity(
    *, model: dict[str, Any], method: dict[str, Any], input_record: dict[str, Any],
    source_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": normalized_model_identity(model),
        "method": method,
        "input": input_record,
        "scoring_protocol": {
            "primary_metric": "cache_dependent_decode_nll",
            "boundary_target_count": 1,
            "decode_target_count": PPL_DECODE_TARGETS,
            "all_target_count": PPL_CONTINUATION_TOKENS,
        },
        "source_state": source_state,
    }


def expand_ppl_cases(
    manifest: dict[str, Any], token_ids: Sequence[int], source_state: dict[str, Any]
) -> list[dict[str, Any]]:
    if len(token_ids) != manifest["corpus"]["expected_token_count"]:
        raise PPLError("expanded PPL token count differs from manifest")
    model = manifest["model"]
    cases = []
    seen: set[str] = set()
    for resolved_method in manifest["methods"]:
        method = _case_method(resolved_method)
        for prompt_length in manifest["prompt_lengths"]:
            for anchor_index in manifest["anchor_indices"]:
                start = continuation_anchor(len(token_ids), anchor_index)
                prompt = token_ids[start - prompt_length:start]
                continuation = token_ids[start:start + PPL_CONTINUATION_TOKENS]
                full = [*prompt, *continuation]
                input_record = {
                    "corpus_id": manifest["corpus"]["id"],
                    "corpus_config": manifest["corpus"]["config"],
                    "corpus_split": manifest["corpus"]["split"],
                    "corpus_revision": manifest["corpus"]["revision"],
                    "corpus_token_ids_sha256": manifest["corpus"]["expected_token_ids_sha256"],
                    "selection_id": manifest["measurement"]["selection_id"],
                    "anchor_index": anchor_index,
                    "continuation_start": start,
                    "prompt_length": prompt_length,
                    "continuation_tokens": PPL_CONTINUATION_TOKENS,
                    "prompt_ids_sha256": token_ids_sha256(prompt),
                    "continuation_ids_sha256": token_ids_sha256(continuation),
                    "full_ids_sha256": token_ids_sha256(full),
                }
                case_id = stable_run_id(ppl_case_identity(
                    model=model,
                    method=method,
                    input_record=input_record,
                    source_state=source_state,
                ))
                if case_id in seen:
                    raise PPLError(f"duplicate PPL case ID {case_id}")
                seen.add(case_id)
                cases.append({
                    "case_id": case_id,
                    "method": method,
                    "input": input_record,
                    "prompt_ids": list(prompt),
                    "continuation_ids": list(continuation),
                })
    return cases


def resolved_ppl_manifest(
    manifest: dict[str, Any], source_state: dict[str, Any],
    corpus_snapshot: dict[str, Any], cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    resolved = json.loads(json.dumps(manifest))
    resolved["source_state"] = source_state
    resolved["corpus_snapshot"] = corpus_snapshot
    resolved["expanded_cases"] = [
        {"case_id": case["case_id"], "method": case["method"], "input": case["input"]}
        for case in cases
    ]
    return resolved


def validate_completed_ppl_case(root: str | Path, case_id: str) -> dict[str, Any]:
    path = Path(root) / "cases" / f"{case_id}.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PPLError(f"cannot load completed PPL case {path}: {error}") from error
    record = _require_object("PPL case", record)
    _require_fields("PPL case", record, CASE_FIELDS)
    if record["schema_version"] != PPL_SCHEMA_VERSION:
        raise PPLError("PPL case schema_version mismatch")
    if record["case_id"] != case_id or record["status"] != "completed":
        raise PPLError("PPL case identity or status mismatch")
    model = _require_object("case.model", record["model"])
    method = _require_object("case.method", record["method"])
    input_record = _require_object("case.input", record["input"])
    scoring = _require_object("case.scoring", record["scoring"])
    runtime = _require_object("case.runtime_diagnostics", record["runtime_diagnostics"])
    _require_fields("case.model", model, CASE_MODEL_FIELDS)
    _require_fields("case.method", method, CASE_METHOD_FIELDS)
    _require_fields("case.input", input_record, CASE_INPUT_FIELDS)
    _require_fields("case.scoring", scoring, SCORING_FIELDS)
    _require_fields("case.runtime_diagnostics", runtime, RUNTIME_FIELDS)

    if method["name"] not in {"fp16", "kivi", "cage"}:
        raise PPLError("PPL case method name is invalid")
    if input_record["continuation_tokens"] != PPL_CONTINUATION_TOKENS:
        raise PPLError("PPL case continuation token count mismatch")
    if scoring["primary_metric"] != "cache_dependent_decode_nll":
        raise PPLError("PPL case primary metric mismatch")
    expected_counts = (1, PPL_DECODE_TARGETS, PPL_CONTINUATION_TOKENS)
    actual_counts = (
        scoring["boundary_target_count"], scoring["decode_target_count"],
        scoring["all_target_count"],
    )
    if actual_counts != expected_counts:
        raise PPLError("PPL case target accounting mismatch")
    nlls = scoring["token_nlls"]
    if not isinstance(nlls, list) or len(nlls) != PPL_CONTINUATION_TOKENS:
        raise PPLError("PPL case token_nlls must contain 64 values")
    nlls = [_finite(f"token_nlls[{index}]", value, minimum=0.0)
            for index, value in enumerate(nlls)]
    decode = nlls[1:]
    expected_values = {
        "boundary_nll": nlls[0],
        "decode_nll_sum": sum(decode),
        "decode_mean_nll": sum(decode) / len(decode),
        "decode_perplexity": math.exp(sum(decode) / len(decode)),
        "all_nll_sum": sum(nlls),
        "all_mean_nll": sum(nlls) / len(nlls),
        "all_perplexity": math.exp(sum(nlls) / len(nlls)),
    }
    for name, expected in expected_values.items():
        actual = _finite(f"case.scoring.{name}", scoring[name], minimum=0.0)
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise PPLError(f"PPL case {name} is inconsistent with token_nlls")

    reference = scoring["fp16_one_shot_reference"]
    if method["name"] == "fp16":
        reference = _require_object("fp16_one_shot_reference", reference)
        _require_fields("fp16_one_shot_reference", reference, FP16_REFERENCE_FIELDS)
        reference_nlls = reference["token_nlls"]
        if not isinstance(reference_nlls, list) or len(reference_nlls) != 64:
            raise PPLError("FP16 one-shot reference must contain 64 token NLLs")
        reference_nlls = [
            _finite(f"reference.token_nlls[{index}]", value, minimum=0.0)
            for index, value in enumerate(reference_nlls)
        ]
        deltas = [abs(left - right) for left, right in zip(nlls, reference_nlls)]
        reference_expected = {
            "nll_sum": sum(reference_nlls),
            "mean_nll": sum(reference_nlls) / 64,
            "perplexity": math.exp(sum(reference_nlls) / 64),
            "mean_absolute_token_nll_delta": sum(deltas) / 64,
            "max_absolute_token_nll_delta": max(deltas),
        }
        for name, expected in reference_expected.items():
            actual = _finite(f"reference.{name}", reference[name], minimum=0.0)
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                raise PPLError(f"FP16 reference {name} is inconsistent")
    elif reference is not None:
        raise PPLError("non-FP16 PPL case must not contain one-shot reference")

    for name in RUNTIME_FIELDS:
        _finite(f"case.runtime_diagnostics.{name}", runtime[name], minimum=0.0)
    provenance = _require_object("case.provenance", record["provenance"])
    source_state = _require_object("case.provenance.source_state", provenance.get("source_state"))
    if source_state.get("dirty") is not False:
        raise PPLError("completed PPL case source state must be clean")
    identity_model = {
        key: model[key]
        for key in ("reference", "dtype", "device", "max_position_embeddings")
    }
    expected_case_id = stable_run_id(ppl_case_identity(
        model=identity_model,
        method=method,
        input_record=input_record,
        source_state=source_state,
    ))
    if expected_case_id != case_id:
        raise PPLError("completed PPL case ID is inconsistent with its identity")
    return record


def is_valid_completed_ppl_case(root: str | Path, case_id: str) -> bool:
    try:
        validate_completed_ppl_case(root, case_id)
    except (OSError, PPLError, ValueError, OverflowError):
        return False
    return True


def aggregate_ppl_cases(
    root: str | Path, *, expected_case_ids: Iterable[str]
) -> list[dict[str, Any]]:
    destination = Path(root)
    records = []
    for case_id in expected_case_ids:
        try:
            records.append(validate_completed_ppl_case(destination, case_id))
        except (OSError, PPLError, ValueError, OverflowError):
            continue
    atomic_write_jsonl(destination / "summary" / "cases.jsonl", records)
    csv_path = destination / "summary" / "cases.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id", "method_id", "method", "prompt_length", "anchor_index",
        "continuation_start", "decode_target_count", "decode_nll_sum",
        "decode_mean_nll", "decode_perplexity", "all_nll_sum",
        "all_mean_nll", "all_perplexity", "elapsed_seconds",
        "cuda_max_allocated_bytes", "cuda_max_reserved_bytes",
    ]

    def write(handle: Any) -> None:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({
                "case_id": record["case_id"],
                "method_id": record["method"]["id"],
                "method": record["method"]["name"],
                "prompt_length": record["input"]["prompt_length"],
                "anchor_index": record["input"]["anchor_index"],
                "continuation_start": record["input"]["continuation_start"],
                "decode_target_count": record["scoring"]["decode_target_count"],
                "decode_nll_sum": record["scoring"]["decode_nll_sum"],
                "decode_mean_nll": record["scoring"]["decode_mean_nll"],
                "decode_perplexity": record["scoring"]["decode_perplexity"],
                "all_nll_sum": record["scoring"]["all_nll_sum"],
                "all_mean_nll": record["scoring"]["all_mean_nll"],
                "all_perplexity": record["scoring"]["all_perplexity"],
                "elapsed_seconds": record["runtime_diagnostics"]["elapsed_seconds"],
                "cuda_max_allocated_bytes": record["runtime_diagnostics"]["cuda_max_allocated_bytes"],
                "cuda_max_reserved_bytes": record["runtime_diagnostics"]["cuda_max_reserved_bytes"],
            })

    _atomic_write(csv_path, write)
    return records


__all__ = [
    "PPL_ACCEPTANCE_RAW_METHODS", "PPL_ANCHOR_INDICES", "PPL_CONTINUATION_TOKENS",
    "PPL_DECODE_TARGETS", "PPL_FULL_RAW_METHODS", "PPL_PROMPT_LENGTHS",
    "PPL_SCHEMA_VERSION", "PPLError", "aggregate_ppl_cases", "continuation_anchor",
    "expand_ppl_cases", "is_valid_completed_ppl_case", "load_ppl_manifest",
    "resolved_ppl_manifest", "token_ids_sha256", "validate_completed_ppl_case",
    "validate_corpus_snapshot",
]
