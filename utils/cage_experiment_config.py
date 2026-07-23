"""Strict loading, resolution, and job expansion for CAGE experiments."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


_TOP_FIELDS = {
    "model", "prompts_file", "sample_ids", "prompt_lengths", "methods",
    "measurement", "output_dir",
}
_RESOLVED_METHOD_FIELDS = {"id", "method", "method_config"}
_MODEL_FIELDS = {"reference", "dtype", "device", "max_position_embeddings"}
_MEASUREMENT_FIELDS = {"decode_tokens", "seed"}
_COMMON_METHOD_FIELDS = {"id", "method"}
_METHOD_FIELDS = {
    "fp16": _COMMON_METHOD_FIELDS,
    "kivi": _COMMON_METHOD_FIELDS | {"k_bits", "v_bits", "group_size", "residual_length"},
    "cage": _COMMON_METHOD_FIELDS | {
        "k_bits", "v_bits", "residual_length", "cage_mode", "cage_k_enable",
        "cage_v_enable", "cage_k_importance", "cage_k_group_sizes",
        "cage_k_clip_percentiles", "cage_k_num_buckets", "cage_v_importance",
        "cage_v_group_sizes", "cage_v_clip_percentiles", "cage_v_num_buckets",
    },
}
_CAGE_DEFAULTS = {
    "k_bits": 2,
    "v_bits": 2,
    "residual_length": 32,
    "cage_mode": "fake",
    "cage_k_enable": True,
    "cage_v_enable": True,
    "cage_k_importance": "q2_var",
    "cage_k_group_sizes": [32, 64, 128],
    "cage_k_clip_percentiles": [0.999, 0.995, 0.99],
    "cage_k_num_buckets": 3,
    "cage_v_importance": "wo_var",
    "cage_v_group_sizes": [32, 64, 128],
    "cage_v_clip_percentiles": [0.999, 0.995, 0.99],
    "cage_v_num_buckets": 3,
}
_SUPPORTED_DTYPES = {"float16", "bfloat16"}
_SUPPORTED_DEVICES = {"cpu", "cuda"}
_SUPPORTED_BITS = {2, 4}


def load_and_resolve_manifest(path: str | Path) -> dict:
    """Load a JSON manifest, reject ambiguity, and return explicit settings."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load manifest {manifest_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("manifest must be a JSON object")
    return _resolve_manifest_object(raw)


def _resolve_manifest_object(raw: dict[str, Any]) -> dict:
    """Resolve and validate one raw manifest object."""

    _require_exact_fields("manifest", raw, _TOP_FIELDS, _TOP_FIELDS)

    model = _object("model", raw["model"])
    _require_exact_fields("model", model, _MODEL_FIELDS, _MODEL_FIELDS)
    _nonempty_string("model.reference", model["reference"])
    _nonempty_string("model.dtype", model["dtype"])
    if model["dtype"] not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {model['dtype']!r}")
    _nonempty_string("model.device", model["device"])
    if model["device"] not in _SUPPORTED_DEVICES:
        raise ValueError(f"unsupported device {model['device']!r}")
    _positive_int("model.max_position_embeddings", model["max_position_embeddings"])

    sample_ids = _unique_nonempty_strings("sample_ids", raw["sample_ids"])
    prompt_lengths = _positive_int_list("prompt_lengths", raw["prompt_lengths"])
    if len(prompt_lengths) != len(set(prompt_lengths)):
        raise ValueError("prompt_lengths entries must be unique")
    _nonempty_string("prompts_file", raw["prompts_file"])
    _nonempty_string("output_dir", raw["output_dir"])

    measurement = _object("measurement", raw["measurement"])
    _require_exact_fields("measurement", measurement, _MEASUREMENT_FIELDS, _MEASUREMENT_FIELDS)
    if measurement["decode_tokens"] != 1 or isinstance(measurement["decode_tokens"], bool):
        raise ValueError("measurement.decode_tokens must equal 1")
    if not isinstance(measurement["seed"], int) or isinstance(measurement["seed"], bool):
        raise ValueError("measurement.seed must be an integer")
    if max(prompt_lengths) + measurement["decode_tokens"] > model["max_position_embeddings"]:
        raise ValueError(
            "context limit exceeded: prompt_length + measurement.decode_tokens must not exceed "
            "model.max_position_embeddings"
        )

    methods = raw["methods"]
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a non-empty list")
    resolved_methods = [_resolve_method(value, index) for index, value in enumerate(methods)]
    ids = [method["id"] for method in resolved_methods]
    if len(ids) != len(set(ids)):
        raise ValueError("method ids must be unique")
    scientific_configs = [
        (
            method["method"],
            json.dumps(
                method["method_config"],
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        for method in resolved_methods
    ]
    if len(scientific_configs) != len(set(scientific_configs)):
        raise ValueError(
            "methods contain scientifically duplicate resolved configurations"
        )

    return {
        "model": copy.deepcopy(model),
        "prompts_file": raw["prompts_file"],
        "sample_ids": sample_ids,
        "prompt_lengths": prompt_lengths,
        "methods": resolved_methods,
        "measurement": copy.deepcopy(measurement),
        "output_dir": raw["output_dir"],
    }


def validate_resolved_manifest(raw: Any) -> dict:
    """Strictly validate a canonical worker-facing resolved manifest."""

    manifest = _object("resolved manifest", raw)
    allowed_top_fields = _TOP_FIELDS | {"jobs"}
    _require_exact_fields("resolved manifest", manifest, allowed_top_fields, _TOP_FIELDS)
    methods = manifest["methods"]
    if not isinstance(methods, list) or not methods:
        raise ValueError("resolved manifest methods must be a non-empty list")

    raw_methods = []
    for index, value in enumerate(methods):
        item = _object(f"resolved methods[{index}]", value)
        _require_exact_fields(
            f"resolved methods[{index}]",
            item,
            _RESOLVED_METHOD_FIELDS,
            _RESOLVED_METHOD_FIELDS,
        )
        method = item["method"]
        _nonempty_string(f"resolved methods[{index}].method", method)
        if method not in _METHOD_FIELDS:
            raise ValueError(f"unsupported method {method!r}")
        config = _object(f"resolved methods[{index}].method_config", item["method_config"])
        expected_config_fields = _METHOD_FIELDS[method] - _COMMON_METHOD_FIELDS
        _require_exact_fields(
            f"resolved methods[{index}].method_config",
            config,
            expected_config_fields,
            expected_config_fields,
        )
        raw_methods.append({
            "id": item["id"],
            "method": method,
            **copy.deepcopy(config),
        })

    raw_manifest = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "jobs"
    }
    raw_manifest["methods"] = raw_methods
    resolved = _resolve_manifest_object(raw_manifest)
    expected_jobs = expand_jobs(resolved)
    if "jobs" in manifest:
        if manifest["jobs"] != expected_jobs:
            raise ValueError(
                "resolved manifest jobs must exactly match canonical expanded methods"
            )
        resolved["jobs"] = expected_jobs
    return resolved


def expand_jobs(manifest: dict) -> list[dict]:
    """Expand resolved methods in manifest order into independent job records."""
    jobs = []
    for method in manifest["methods"]:
        jobs.append({
            "job_id": method["id"],
            "method": method["method"],
            "model": copy.deepcopy(manifest["model"]),
            "method_config": copy.deepcopy(method["method_config"]),
            "sample_ids": list(manifest["sample_ids"]),
            "prompt_lengths": list(manifest["prompt_lengths"]),
            "measurement": copy.deepcopy(manifest["measurement"]),
            "prompts_file": manifest["prompts_file"],
            "output_dir": manifest["output_dir"],
        })
    return jobs


def _resolve_method(value: Any, index: int) -> dict:
    item = _object(f"methods[{index}]", value)
    method = item.get("method")
    _nonempty_string(f"methods[{index}].method", method)
    if method not in _METHOD_FIELDS:
        raise ValueError(f"unsupported method {method!r}")
    _require_exact_fields(f"methods[{index}]", item, _METHOD_FIELDS[method], _COMMON_METHOD_FIELDS)
    _nonempty_string(f"methods[{index}].id", item["id"])

    if method == "fp16":
        config = {}
    elif method == "kivi":
        for name in ("k_bits", "v_bits", "group_size", "residual_length"):
            if name not in item:
                raise ValueError(f"methods[{index}] missing required field {name!r}")
        _bits("k_bits", item["k_bits"]); _bits("v_bits", item["v_bits"])
        _positive_int("group_size", item["group_size"])
        _positive_int("residual_length", item["residual_length"])
        if item["residual_length"] % item["group_size"]:
            raise ValueError("KIVI residual_length must be divisible by group_size")
        config = {name: item[name] for name in ("k_bits", "v_bits", "group_size", "residual_length")}
    else:
        config = copy.deepcopy(_CAGE_DEFAULTS)
        config.update({key: copy.deepcopy(value) for key, value in item.items() if key not in _COMMON_METHOD_FIELDS})
        _validate_cage(config)
    return {"id": item["id"], "method": method, "method_config": config}


def _validate_cage(config: dict) -> None:
    if config["k_bits"] != 2 or isinstance(config["k_bits"], bool):
        raise ValueError("CAGE k_bits must equal 2 for the scoped core pilot")
    if config["v_bits"] != 2 or isinstance(config["v_bits"], bool):
        raise ValueError("CAGE v_bits must equal 2 for the scoped core pilot")
    _positive_int("residual_length", config["residual_length"])
    if config["cage_mode"] != "fake":
        raise ValueError("cage_mode must be 'fake'")
    for name in ("cage_k_enable", "cage_v_enable"):
        if config[name] is not True:
            raise ValueError(f"{name} must be true for the scoped core pilot")
    if config["cage_k_importance"] != "q2_var":
        raise ValueError("cage_k_importance must equal 'q2_var' for the scoped core pilot")
    if config["cage_v_importance"] != "wo_var":
        raise ValueError("cage_v_importance must equal 'wo_var' for the scoped core pilot")
    for prefix in ("cage_k", "cage_v"):
        _nonempty_string(f"{prefix}_importance", config[f"{prefix}_importance"])
        count = config[f"{prefix}_num_buckets"]
        _positive_int(f"{prefix}_num_buckets", count)
        groups = _positive_int_list(f"{prefix}_group_sizes", config[f"{prefix}_group_sizes"])
        clips = config[f"{prefix}_clip_percentiles"]
        if not isinstance(clips, list) or not clips or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 < x <= 1 for x in clips
        ):
            raise ValueError(f"{prefix}_clip_percentiles must contain values in (0, 1]")
        if len(groups) != count or len(clips) != count:
            raise ValueError(f"{prefix} bucket lists must match {prefix}_num_buckets")


def _require_exact_fields(name: str, value: dict, allowed: set[str], required: set[str]) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{name} missing required fields: {sorted(missing)}")


def _object(name: str, value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _nonempty_string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _bits(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in _SUPPORTED_BITS:
        raise ValueError(f"{name} must be one of {sorted(_SUPPORTED_BITS)}")


def _unique_nonempty_strings(name: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    for item in value:
        _nonempty_string(name, item)
    if len(value) != len(set(value)):
        raise ValueError(f"{name} entries must be unique")
    return list(value)


def _positive_int_list(name: str, value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    for item in value:
        _positive_int(name, item)
    return list(value)


__all__ = ["expand_jobs", "load_and_resolve_manifest", "validate_resolved_manifest"]
