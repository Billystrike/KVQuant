from __future__ import annotations

import hashlib
import json
import copy
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from utils.qwen3_cases import (
    QWEN3_ANCHOR_COUNT,
    QWEN3_CONTINUATION_TOKENS,
    QWEN3_PROMPT_LENGTHS,
    QWEN3_SELECTION_ID,
    continuation_anchor,
    token_ids_sha256,
)


FORMAL_PROTOCOL_ID = "qwen3-8b-cage-kitty-formal-quality-v1"
FORMAL_EXECUTION_ID = "qwen3-8b-cage-kitty-formal-execution-v1"
FORMAL_ACCEPTANCE_GATE_ID = "qwen3-8b-cage-kitty-formal-acceptance-gate-v1"
INPUT_MANIFEST_SCHEMA_VERSION = 1


class Qwen3FormalError(ValueError):
    pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_formal_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    try:
        protocol = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3FormalError(f"cannot load formal protocol {source}: {error}") from error
    if not isinstance(protocol, dict):
        raise Qwen3FormalError("formal protocol must be a JSON object")
    _validate_protocol(protocol)
    return protocol, file_sha256(source)


def load_execution_config(
    path: str | Path,
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
) -> tuple[dict[str, Any], str]:
    source = Path(path)
    try:
        execution = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3FormalError(f"cannot load formal execution config {source}: {error}") from error
    if not isinstance(execution, dict):
        raise Qwen3FormalError("formal execution config must be a JSON object")
    _validate_execution_config(
        execution,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    return execution, file_sha256(source)


def load_acceptance_gate(
    path: str | Path,
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    execution: dict[str, Any],
    execution_sha256: str,
    input_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    source = Path(path)
    gate = _load_json_object(source, "formal acceptance gate")
    _validate_acceptance_gate(
        gate,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        execution=execution,
        execution_sha256=execution_sha256,
        input_manifest_sha256=input_manifest_sha256,
    )
    _verify_acceptance_gate_artifacts(gate)
    return gate, file_sha256(source)


def read_corpus_snapshot(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    source = Path(path)
    try:
        snapshot = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3FormalError(f"cannot load corpus snapshot {source}: {error}") from error
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "schema_version",
        "corpus",
        "texts",
    }:
        raise Qwen3FormalError("corpus snapshot has an invalid top-level schema")
    if snapshot["schema_version"] != 1 or not isinstance(snapshot["corpus"], dict):
        raise Qwen3FormalError("corpus snapshot version or identity is invalid")
    texts = snapshot["texts"]
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise Qwen3FormalError("corpus snapshot texts must be a list of strings")
    return snapshot["corpus"], texts


def build_input_manifest(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    token_ids: Sequence[int],
    corpus_snapshot_sha256: str,
    tokenizer_identity: dict[str, Any],
    source_state: dict[str, Any],
) -> dict[str, Any]:
    _validate_protocol(protocol)
    _require_sha256("protocol_sha256", protocol_sha256)
    _require_sha256("corpus_snapshot_sha256", corpus_snapshot_sha256)
    if token_ids_sha256(token_ids) != protocol["input"]["token_ids_sha256"]:
        raise Qwen3FormalError("token stream SHA-256 differs from the frozen protocol")
    if len(token_ids) != protocol["input"]["token_count"]:
        raise Qwen3FormalError("token stream length differs from the frozen protocol")
    if corpus_snapshot_sha256 != protocol["input"]["corpus_snapshot_sha256"]:
        raise Qwen3FormalError("corpus snapshot SHA-256 differs from the frozen protocol")
    if not isinstance(tokenizer_identity, dict) or not tokenizer_identity:
        raise Qwen3FormalError("tokenizer_identity must be a nonempty object")
    _validate_source_state(source_state)

    cases: list[dict[str, Any]] = []
    for anchor_index in protocol["input"]["anchor_indices"]:
        start = continuation_anchor(len(token_ids), anchor_index)
        continuation = list(token_ids[start : start + QWEN3_CONTINUATION_TOKENS])
        for prompt_length in protocol["input"]["prompt_lengths"]:
            prompt = list(token_ids[start - prompt_length : start])
            full = [*prompt, *continuation]
            cases.append(
                {
                    "input_case_id": f"qwen3-a{anchor_index:02d}-l{prompt_length}",
                    "anchor_index": anchor_index,
                    "continuation_start": start,
                    "prompt_length": prompt_length,
                    "continuation_tokens": QWEN3_CONTINUATION_TOKENS,
                    "prompt_ids_sha256": token_ids_sha256(prompt),
                    "continuation_ids_sha256": token_ids_sha256(continuation),
                    "full_ids_sha256": token_ids_sha256(full),
                    "prompt_ids": prompt,
                    "continuation_ids": continuation,
                }
            )

    manifest = {
        "schema_version": INPUT_MANIFEST_SCHEMA_VERSION,
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha256,
        },
        "model": protocol["model"],
        "input_identity": {
            key: protocol["input"][key]
            for key in (
                "corpus_id",
                "corpus_config",
                "corpus_split",
                "corpus_revision",
                "joined_text_sha256",
                "add_special_tokens",
                "token_count",
                "token_ids_sha256",
                "selection_id",
                "anchor_count",
                "prompt_lengths",
                "continuation_tokens",
                "minimum_anchor_gap",
            )
        },
        "corpus_snapshot_sha256": corpus_snapshot_sha256,
        "tokenizer_identity": tokenizer_identity,
        "source_state": source_state,
        "cases": cases,
    }
    validate_input_manifest(manifest, protocol=protocol, protocol_sha256=protocol_sha256)
    return manifest


def validate_input_manifest(
    manifest: dict[str, Any],
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
) -> None:
    _validate_protocol(protocol)
    if not isinstance(manifest, dict):
        raise Qwen3FormalError("input manifest must be an object")
    required = {
        "schema_version",
        "protocol",
        "model",
        "input_identity",
        "corpus_snapshot_sha256",
        "tokenizer_identity",
        "source_state",
        "cases",
    }
    if set(manifest) != required:
        raise Qwen3FormalError("input manifest top-level fields differ from the schema")
    if manifest["schema_version"] != INPUT_MANIFEST_SCHEMA_VERSION:
        raise Qwen3FormalError("input manifest schema version mismatch")
    if manifest["protocol"] != {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
    }:
        raise Qwen3FormalError("input manifest protocol identity mismatch")
    if manifest["model"] != protocol["model"]:
        raise Qwen3FormalError("input manifest model identity mismatch")
    expected_input_identity = {
        key: protocol["input"][key]
        for key in (
            "corpus_id",
            "corpus_config",
            "corpus_split",
            "corpus_revision",
            "joined_text_sha256",
            "add_special_tokens",
            "token_count",
            "token_ids_sha256",
            "selection_id",
            "anchor_count",
            "prompt_lengths",
            "continuation_tokens",
            "minimum_anchor_gap",
        )
    }
    if manifest["input_identity"] != expected_input_identity:
        raise Qwen3FormalError("input manifest frozen input identity mismatch")
    if manifest["corpus_snapshot_sha256"] != protocol["input"]["corpus_snapshot_sha256"]:
        raise Qwen3FormalError("input manifest corpus snapshot mismatch")
    _validate_source_state(manifest["source_state"])

    expected_count = QWEN3_ANCHOR_COUNT * len(QWEN3_PROMPT_LENGTHS)
    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) != expected_count:
        raise Qwen3FormalError(f"input manifest must contain {expected_count} cases")
    seen: set[str] = set()
    cursor = 0
    for anchor_index in protocol["input"]["anchor_indices"]:
        expected_start = continuation_anchor(protocol["input"]["token_count"], anchor_index)
        for prompt_length in protocol["input"]["prompt_lengths"]:
            record = cases[cursor]
            cursor += 1
            if not isinstance(record, dict):
                raise Qwen3FormalError("input case must be an object")
            expected_id = f"qwen3-a{anchor_index:02d}-l{prompt_length}"
            if record.get("input_case_id") != expected_id or expected_id in seen:
                raise Qwen3FormalError("input case ID/order mismatch or duplicate")
            seen.add(expected_id)
            if record.get("anchor_index") != anchor_index:
                raise Qwen3FormalError("input case anchor mismatch")
            if record.get("continuation_start") != expected_start:
                raise Qwen3FormalError("input case continuation start mismatch")
            if record.get("prompt_length") != prompt_length:
                raise Qwen3FormalError("input case prompt length mismatch")
            if record.get("continuation_tokens") != QWEN3_CONTINUATION_TOKENS:
                raise Qwen3FormalError("input case continuation length declaration mismatch")
            prompt = record.get("prompt_ids")
            continuation = record.get("continuation_ids")
            if not isinstance(prompt, list) or len(prompt) != prompt_length:
                raise Qwen3FormalError("input case prompt IDs have the wrong length")
            if not isinstance(continuation, list) or len(continuation) != QWEN3_CONTINUATION_TOKENS:
                raise Qwen3FormalError("input case continuation IDs have the wrong length")
            if any(type(value) is not int or value < 0 for value in [*prompt, *continuation]):
                raise Qwen3FormalError("input case token IDs must be nonnegative integers")
            expected_hashes = {
                "prompt_ids_sha256": token_ids_sha256(prompt),
                "continuation_ids_sha256": token_ids_sha256(continuation),
                "full_ids_sha256": token_ids_sha256([*prompt, *continuation]),
            }
            if any(record.get(name) != value for name, value in expected_hashes.items()):
                raise Qwen3FormalError("input case token hash mismatch")


def formal_method_length_points(
    protocol: dict[str, Any], *, partition: str
) -> list[dict[str, Any]]:
    _validate_protocol(protocol)
    if partition not in {"cage_qwen3", "kitty_qwen3"}:
        raise Qwen3FormalError(f"unknown formal execution partition {partition!r}")
    cage_defaults = protocol["cage_defaults"]
    points: list[dict[str, Any]] = []
    for record in protocol["base_method_length_points"]:
        family = record["method"]
        belongs = family == "kitty" if partition == "kitty_qwen3" else family != "kitty"
        if not belongs:
            continue
        for prompt_length in record["prompt_lengths"]:
            config: dict[str, Any]
            if family == "fp16":
                config = {}
            elif family == "kivi":
                config = {
                    "bits": 2,
                    "group_size": record["group_size"],
                    "residual_length": record["residual_length"],
                }
            elif family == "cage":
                config = _resolved_cage_method_config(
                    cage_defaults,
                    residual_length=record["residual_length"],
                    key_importance=cage_defaults["key_importance"],
                    value_importance=cage_defaults["value_importance"],
                    assignment_seed=1729,
                )
            elif family == "kitty":
                config = {"boosted_channels": record["boosted_channels"]}
            else:
                raise Qwen3FormalError(f"unsupported formal method family {family!r}")
            points.append(
                {
                    "method_id": record["method_id"],
                    "method": family,
                    "prompt_length": prompt_length,
                    "config": config,
                }
            )

    if partition == "cage_qwen3":
        for mechanism_point in protocol["mechanism_points"]:
            for variant in protocol["mechanism_variants"]:
                method_id = f"{mechanism_point['full_method_id']}-{variant['suffix']}"
                points.append(
                    {
                        "method_id": method_id,
                        "method": "cage",
                        "prompt_length": mechanism_point["prompt_length"],
                        "config": _resolved_cage_method_config(
                            cage_defaults,
                            residual_length=mechanism_point["residual_length"],
                            key_importance=variant["key_importance"],
                            value_importance=variant["value_importance"],
                            assignment_seed=variant.get("assignment_seed", 1729),
                        ),
                    }
                )
    identities = [(point["method_id"], point["prompt_length"]) for point in points]
    if len(identities) != len(set(identities)):
        raise Qwen3FormalError("formal method-length points contain duplicate identities")
    expected_count = 20 if partition == "cage_qwen3" else 6
    if len(points) != expected_count:
        raise Qwen3FormalError(
            f"formal partition {partition} must contain {expected_count} method-length points"
        )
    return points


def expand_partition_cases(
    *,
    protocol: dict[str, Any],
    execution: dict[str, Any],
    execution_sha256: str,
    input_manifest: dict[str, Any],
    partition: str,
    stage: str,
) -> list[dict[str, Any]]:
    _require_sha256("execution_sha256", execution_sha256)
    if stage not in {"acceptance", "full"}:
        raise Qwen3FormalError("formal execution stage must be 'acceptance' or 'full'")
    points = formal_method_length_points(protocol, partition=partition)
    partition_config = execution["partitions"][partition]
    if stage == "acceptance":
        point_lookup = {
            (point["method_id"], point["prompt_length"]): point for point in points
        }
        selected = []
        for identity in partition_config["acceptance_points"]:
            key = (identity["method_id"], identity["prompt_length"])
            if key not in point_lookup:
                raise Qwen3FormalError(f"acceptance point is not executable: {key!r}")
            selected.append(point_lookup[key])
        points = selected
        anchors = partition_config["acceptance_anchor_indices"]
    else:
        anchors = protocol["input"]["anchor_indices"]

    input_lookup = {
        (record["anchor_index"], record["prompt_length"]): record
        for record in input_manifest["cases"]
    }
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for point in points:
        for anchor_index in anchors:
            input_record = input_lookup.get((anchor_index, point["prompt_length"]))
            if input_record is None:
                raise Qwen3FormalError("execution point has no matching frozen input case")
            method = {
                "id": point["method_id"],
                "name": point["method"],
                "config": copy.deepcopy(point["config"]),
            }
            input_identity = {
                key: input_record[key]
                for key in (
                    "input_case_id",
                    "anchor_index",
                    "continuation_start",
                    "prompt_length",
                    "continuation_tokens",
                    "prompt_ids_sha256",
                    "continuation_ids_sha256",
                    "full_ids_sha256",
                )
            }
            case_identity = {
                "execution_id": execution["execution_id"],
                "execution_sha256": execution_sha256,
                "partition": partition,
                "method": method,
                "input": input_identity,
            }
            case_id = _stable_id(case_identity)
            if case_id in seen:
                raise Qwen3FormalError("expanded formal execution contains duplicate case IDs")
            seen.add(case_id)
            cases.append(
                {
                    "case_id": case_id,
                    "partition": partition,
                    "method": method,
                    "input": input_identity,
                    "prompt_ids": copy.deepcopy(input_record["prompt_ids"]),
                    "continuation_ids": copy.deepcopy(input_record["continuation_ids"]),
                }
            )
    expected = (
        partition_config["acceptance_cases"]
        if stage == "acceptance"
        else partition_config["full_cases"]
    )
    if len(cases) != expected:
        raise Qwen3FormalError(
            f"expanded {partition} {stage} cases={len(cases)} differs from expected {expected}"
        )
    return cases


def formal_scoring(token_nlls: Sequence[float]) -> dict[str, Any]:
    if len(token_nlls) != QWEN3_CONTINUATION_TOKENS:
        raise Qwen3FormalError("formal scoring requires exactly 64 token NLL values")
    values = []
    for index, value in enumerate(token_nlls):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Qwen3FormalError(f"token_nlls[{index}] must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise Qwen3FormalError(f"token_nlls[{index}] must be finite and nonnegative")
        values.append(numeric)
    total = math.fsum(values)
    decode = values[1:]
    decode_sum = math.fsum(decode)
    mean = total / len(values)
    decode_mean = decode_sum / len(decode)
    return {
        "metric": "cache_conditioned_continuation_nll",
        "target_count": len(values),
        "boundary_target_count": 1,
        "decode_target_count": len(decode),
        "token_nlls": values,
        "boundary_nll": values[0],
        "nll_sum": total,
        "mean_nll": mean,
        "perplexity": math.exp(mean),
        "decode_nll_sum": decode_sum,
        "decode_mean_nll": decode_mean,
        "decode_perplexity": math.exp(decode_mean),
    }


def validate_completed_result(
    record: dict[str, Any],
    *,
    expected_case: dict[str, Any],
    execution_id: str,
    execution_sha256: str,
    protocol_id: str,
    protocol_sha256: str,
    input_manifest_sha256: str,
    source_state: dict[str, Any],
    acceptance_gate_sha256: str | None = None,
) -> None:
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise Qwen3FormalError("completed result schema mismatch")
    if record.get("status") != "completed":
        raise Qwen3FormalError("completed result status mismatch")
    if record.get("case_id") != expected_case["case_id"]:
        raise Qwen3FormalError("completed result case ID mismatch")
    if record.get("partition") != expected_case["partition"]:
        raise Qwen3FormalError("completed result partition mismatch")
    if record.get("method") != expected_case["method"]:
        raise Qwen3FormalError("completed result method mismatch")
    if record.get("input") != expected_case["input"]:
        raise Qwen3FormalError("completed result input mismatch")
    expected_identity = {
        "execution_id": execution_id,
        "execution_sha256": execution_sha256,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "source_state": source_state,
    }
    if acceptance_gate_sha256 is not None:
        _require_sha256("acceptance_gate_sha256", acceptance_gate_sha256)
        expected_identity["acceptance_gate_sha256"] = acceptance_gate_sha256
    if record.get("identity") != expected_identity:
        raise Qwen3FormalError("completed result execution identity mismatch")
    scoring = record.get("scoring")
    if not isinstance(scoring, dict):
        raise Qwen3FormalError("completed result scoring must be an object")
    expected_scoring = formal_scoring(scoring.get("token_nlls", []))
    if set(scoring) != set(expected_scoring):
        raise Qwen3FormalError("completed result scoring fields mismatch")
    for name, expected in expected_scoring.items():
        actual = scoring[name]
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise Qwen3FormalError(f"completed result scoring {name} is inconsistent")
        elif actual != expected:
            raise Qwen3FormalError(f"completed result scoring {name} is inconsistent")
    runtime = record.get("runtime")
    cache = record.get("cache")
    model = record.get("model")
    if not isinstance(runtime, dict) or not isinstance(cache, dict) or not isinstance(model, dict):
        raise Qwen3FormalError("completed result runtime/cache/model records are required")
    for name in ("prefill_seconds", "decode_seconds", "elapsed_seconds"):
        value = runtime.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise Qwen3FormalError(f"completed result runtime {name} is invalid")
    if cache.get("reported_seq_length") != (
        expected_case["input"]["prompt_length"] + QWEN3_CONTINUATION_TOKENS - 1
    ):
        raise Qwen3FormalError("completed result cache sequence length mismatch")


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3FormalError(f"cannot load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise Qwen3FormalError(f"{label} must be a JSON object: {path}")
    return value


def _validate_acceptance_gate(
    gate: dict[str, Any],
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    execution: dict[str, Any],
    execution_sha256: str,
    input_manifest_sha256: str,
) -> None:
    required = {
        "schema_version",
        "gate_id",
        "status",
        "closed_at_utc",
        "protocol",
        "execution",
        "input_manifest",
        "partitions",
        "failed_pre_case_attempts",
    }
    if set(gate) != required:
        raise Qwen3FormalError("formal acceptance gate top-level schema mismatch")
    if gate["schema_version"] != 1 or gate["gate_id"] != FORMAL_ACCEPTANCE_GATE_ID:
        raise Qwen3FormalError("formal acceptance gate identity mismatch")
    if gate["status"] != "passed":
        raise Qwen3FormalError("formal acceptance gate is not passed")
    if not isinstance(gate["closed_at_utc"], str) or not gate["closed_at_utc"]:
        raise Qwen3FormalError("formal acceptance gate close time is invalid")
    if gate["protocol"] != {
        "protocol_id": protocol["protocol_id"],
        "sha256": protocol_sha256,
    }:
        raise Qwen3FormalError("formal acceptance gate protocol identity mismatch")
    if gate["execution"] != {
        "execution_id": execution["execution_id"],
        "sha256": execution_sha256,
    }:
        raise Qwen3FormalError("formal acceptance gate execution identity mismatch")
    if gate["input_manifest"] != {"sha256": input_manifest_sha256}:
        raise Qwen3FormalError("formal acceptance gate input manifest mismatch")

    partitions = gate["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != {
        "cage_qwen3",
        "kitty_qwen3",
    }:
        raise Qwen3FormalError("formal acceptance gate partitions mismatch")
    for partition, expected_cases in (("cage_qwen3", 11), ("kitty_qwen3", 3)):
        record = partitions[partition]
        if not isinstance(record, dict) or set(record) != {
            "expected_cases",
            "source_state",
            "runs",
            "comparison",
        }:
            raise Qwen3FormalError(f"{partition} acceptance gate schema mismatch")
        if record["expected_cases"] != expected_cases:
            raise Qwen3FormalError(f"{partition} acceptance gate case count mismatch")
        source_state = record["source_state"]
        _validate_source_state(source_state)
        expected_source_fields = (
            {"git_commit", "dirty"}
            if partition == "cage_qwen3"
            else {
                "git_commit",
                "dirty",
                "kitty_commit",
                "kitty_dirty",
                "transformers_commit",
            }
        )
        if set(source_state) != expected_source_fields:
            raise Qwen3FormalError(f"{partition} acceptance source-state schema mismatch")
        if partition == "kitty_qwen3":
            for name in ("kitty_commit", "transformers_commit"):
                value = source_state[name]
                if not isinstance(value, str) or len(value) != 40:
                    raise Qwen3FormalError(f"Kitty acceptance {name} is invalid")
            if source_state["kitty_dirty"] is not False:
                raise Qwen3FormalError("Kitty acceptance source tree must be clean")

        runs = record["runs"]
        if not isinstance(runs, dict) or set(runs) != {"a", "b"}:
            raise Qwen3FormalError(f"{partition} acceptance runs mismatch")
        for label, run in runs.items():
            if not isinstance(run, dict) or set(run) != {
                "directory",
                "run_identity_sha256",
                "summary_sha256",
                "archive_path",
                "archive_sha256",
                "log_path",
                "log_sha256",
            }:
                raise Qwen3FormalError(f"{partition} acceptance run {label} schema mismatch")
            for name in ("directory", "archive_path", "log_path"):
                _require_absolute_path(f"{partition}.{label}.{name}", run[name])
            for name in (
                "run_identity_sha256",
                "summary_sha256",
                "archive_sha256",
                "log_sha256",
            ):
                _require_sha256(f"{partition}.{label}.{name}", run[name])

        comparison = record["comparison"]
        if not isinstance(comparison, dict) or set(comparison) != {
            "path",
            "sha256",
            "scientific_payload_sha256",
            "case_count",
        }:
            raise Qwen3FormalError(f"{partition} acceptance comparison schema mismatch")
        _require_absolute_path(f"{partition}.comparison.path", comparison["path"])
        _require_sha256(f"{partition}.comparison.sha256", comparison["sha256"])
        _require_sha256(
            f"{partition}.comparison.scientific_payload_sha256",
            comparison["scientific_payload_sha256"],
        )
        if comparison["case_count"] != expected_cases:
            raise Qwen3FormalError(f"{partition} acceptance comparison count mismatch")

    failures = gate["failed_pre_case_attempts"]
    if not isinstance(failures, list):
        raise Qwen3FormalError("failed pre-case attempts must be a list")
    for record in failures:
        if not isinstance(record, dict) or set(record) != {
            "label",
            "source_state",
            "output_directory",
            "run_identity_sha256",
            "log_path",
            "log_sha256",
            "cases_executed",
        }:
            raise Qwen3FormalError("failed pre-case attempt schema mismatch")
        if not isinstance(record["label"], str) or not record["label"]:
            raise Qwen3FormalError("failed pre-case attempt label is invalid")
        _validate_source_state(record["source_state"])
        _require_absolute_path("failed output_directory", record["output_directory"])
        _require_absolute_path("failed log_path", record["log_path"])
        _require_sha256("failed run_identity_sha256", record["run_identity_sha256"])
        _require_sha256("failed log_sha256", record["log_sha256"])
        if record["cases_executed"] != 0:
            raise Qwen3FormalError("recorded pre-case failure must have zero executed cases")


def _verify_acceptance_gate_artifacts(gate: dict[str, Any]) -> None:
    protocol = gate["protocol"]
    execution = gate["execution"]
    input_manifest_sha256 = gate["input_manifest"]["sha256"]
    for partition, record in gate["partitions"].items():
        expected_cases = record["expected_cases"]
        expected_identity = {
            "execution_id": execution["execution_id"],
            "execution_sha256": execution["sha256"],
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol["sha256"],
            "input_manifest_sha256": input_manifest_sha256,
            "source_state": record["source_state"],
        }
        locks: dict[str, dict[str, Any]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        for label, run in record["runs"].items():
            directory = Path(run["directory"])
            lock_path = directory / "run_identity.json"
            summary_path = directory / "summary.json"
            _verify_file_sha256(lock_path, run["run_identity_sha256"])
            _verify_file_sha256(summary_path, run["summary_sha256"])
            _verify_file_sha256(Path(run["archive_path"]), run["archive_sha256"])
            _verify_file_sha256(Path(run["log_path"]), run["log_sha256"])
            lock = _load_json_object(lock_path, f"{partition} acceptance run lock")
            summary = _load_json_object(summary_path, f"{partition} acceptance summary")
            case_ids = lock.get("expected_case_ids")
            if (
                lock.get("schema_version") != 1
                or lock.get("partition") != partition
                or lock.get("stage") != "acceptance"
                or lock.get("execution_id") != execution["execution_id"]
                or lock.get("execution_sha256") != execution["sha256"]
                or lock.get("protocol_id") != protocol["protocol_id"]
                or lock.get("protocol_sha256") != protocol["sha256"]
                or lock.get("input_manifest_sha256") != input_manifest_sha256
                or lock.get("source_state") != record["source_state"]
                or not isinstance(case_ids, list)
                or len(case_ids) != expected_cases
                or len(case_ids) != len(set(case_ids))
            ):
                raise Qwen3FormalError(f"{partition} acceptance run {label} lock failed")
            if (
                summary.get("schema_version") != 1
                or summary.get("status") != "pass"
                or summary.get("partition") != partition
                or summary.get("stage") != "acceptance"
                or summary.get("expected_cases") != expected_cases
                or summary.get("completed_cases") != expected_cases
                or summary.get("new_cases") != expected_cases
                or summary.get("resumed_cases") != 0
                or summary.get("failure_records") != 0
                or summary.get("identity") != expected_identity
            ):
                raise Qwen3FormalError(f"{partition} acceptance run {label} summary failed")
            locks[label] = lock
            summaries[label] = summary
        if locks["a"] != locks["b"]:
            raise Qwen3FormalError(f"{partition} acceptance run identities differ")

        comparison_record = record["comparison"]
        comparison_path = Path(comparison_record["path"])
        _verify_file_sha256(comparison_path, comparison_record["sha256"])
        comparison = _load_json_object(
            comparison_path, f"{partition} acceptance comparison"
        )
        if (
            comparison.get("schema_version") != 1
            or comparison.get("status") != "pass"
            or comparison.get("partition") != partition
            or comparison.get("stage") != "acceptance_repeat_comparison"
            or comparison.get("case_count") != expected_cases
            or comparison.get("bitwise_equal_scientific_cases") != expected_cases
            or comparison.get("mismatch_count") != 0
            or comparison.get("mismatches") != []
            or comparison.get("scientific_payload_sha256")
            != comparison_record["scientific_payload_sha256"]
            or comparison.get("execution_identity") != expected_identity
            or comparison.get("left") != record["runs"]["a"]["directory"]
            or comparison.get("right") != record["runs"]["b"]["directory"]
            or comparison.get("left_summary_sha256")
            != record["runs"]["a"]["summary_sha256"]
            or comparison.get("right_summary_sha256")
            != record["runs"]["b"]["summary_sha256"]
        ):
            raise Qwen3FormalError(f"{partition} acceptance comparison failed")

    for record in gate["failed_pre_case_attempts"]:
        output_directory = Path(record["output_directory"])
        _verify_file_sha256(
            output_directory / "run_identity.json", record["run_identity_sha256"]
        )
        _verify_file_sha256(Path(record["log_path"]), record["log_sha256"])
        cases_path = output_directory / "cases"
        if cases_path.exists() and any(cases_path.glob("*.json")):
            raise Qwen3FormalError("pre-case failure unexpectedly contains case results")


def _verify_file_sha256(path: Path, expected_sha256: str) -> None:
    try:
        actual = file_sha256(path)
    except OSError as error:
        raise Qwen3FormalError(f"cannot hash acceptance artifact {path}: {error}") from error
    if actual != expected_sha256:
        raise Qwen3FormalError(f"acceptance artifact SHA-256 mismatch: {path}")


def _require_absolute_path(name: str, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise Qwen3FormalError(f"{name} must be a nonempty path")
    path = Path(value)
    if not path.is_absolute() and not PurePosixPath(value).is_absolute():
        raise Qwen3FormalError(f"{name} must be an absolute path")
    return path


def _resolved_cage_method_config(
    defaults: dict[str, Any],
    *,
    residual_length: int,
    key_importance: str,
    value_importance: str,
    assignment_seed: int,
) -> dict[str, Any]:
    return {
        "bits": defaults["k_bits"],
        "residual_length": residual_length,
        "key_importance": key_importance,
        "value_importance": value_importance,
        "key_group_sizes": copy.deepcopy(defaults["key_group_sizes"]),
        "value_group_sizes": copy.deepcopy(defaults["value_group_sizes"]),
        "key_clip_percentiles": copy.deepcopy(defaults["key_clip_percentiles"]),
        "value_clip_percentiles": copy.deepcopy(defaults["value_clip_percentiles"]),
        "key_num_buckets": len(defaults["key_bucket_sizes"]),
        "value_num_buckets": len(defaults["value_bucket_sizes"]),
        "ablation": (
            key_importance != defaults["key_importance"]
            or value_importance != defaults["value_importance"]
        ),
        "assignment_seed": assignment_seed,
    }


def _validate_execution_config(
    execution: dict[str, Any],
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
) -> None:
    if execution.get("schema_version") != 1:
        raise Qwen3FormalError("formal execution schema version mismatch")
    if execution.get("execution_id") != FORMAL_EXECUTION_ID:
        raise Qwen3FormalError("formal execution ID mismatch")
    if execution.get("status") != "frozen_before_formal_qwen3_quality_execution":
        raise Qwen3FormalError("formal execution config is not frozen before quality execution")
    if execution.get("formal_quality_results_seen_before_freeze") is not False:
        raise Qwen3FormalError("formal execution pre-result declaration mismatch")
    expected_protocol = {
        "path": "configs/qwen3_8b_formal_quality_protocol_v1.json",
        "protocol_id": protocol["protocol_id"],
        "sha256": protocol_sha256,
    }
    if execution.get("protocol") != expected_protocol:
        raise Qwen3FormalError("formal execution protocol identity mismatch")
    input_manifest = execution.get("input_manifest")
    if not isinstance(input_manifest, dict):
        raise Qwen3FormalError("formal execution input_manifest must be an object")
    _require_sha256("execution.input_manifest.sha256", input_manifest.get("sha256"))
    if input_manifest.get("input_case_count") != 150:
        raise Qwen3FormalError("formal execution input manifest case count mismatch")
    source_commit = input_manifest.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise Qwen3FormalError("formal execution input manifest source commit mismatch")
    partitions = execution.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"cage_qwen3", "kitty_qwen3"}:
        raise Qwen3FormalError("formal execution partitions mismatch")
    for partition, expected_points, expected_full, expected_acceptance in (
        ("cage_qwen3", 20, 1000, 11),
        ("kitty_qwen3", 6, 300, 3),
    ):
        record = partitions[partition]
        if record.get("full_method_length_points") != expected_points:
            raise Qwen3FormalError(f"{partition} method-length point count mismatch")
        if record.get("full_cases") != expected_full:
            raise Qwen3FormalError(f"{partition} full case count mismatch")
        if record.get("acceptance_cases") != expected_acceptance:
            raise Qwen3FormalError(f"{partition} acceptance case count mismatch")
        if record.get("acceptance_anchor_indices") != [0]:
            raise Qwen3FormalError(f"{partition} acceptance anchors mismatch")
    for partition in partitions:
        points = formal_method_length_points(protocol, partition=partition)
        executable = {(point["method_id"], point["prompt_length"]) for point in points}
        acceptance = [
            (item.get("method_id"), item.get("prompt_length"))
            for item in partitions[partition].get("acceptance_points", [])
        ]
        if len(acceptance) != len(set(acceptance)) or any(
            identity not in executable for identity in acceptance
        ):
            raise Qwen3FormalError(f"{partition} acceptance points are invalid")


def _stable_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise Qwen3FormalError("formal protocol schema version mismatch")
    if protocol.get("protocol_id") != FORMAL_PROTOCOL_ID:
        raise Qwen3FormalError("formal protocol ID mismatch")
    if protocol.get("status") != "frozen_before_formal_qwen3_quality_results":
        raise Qwen3FormalError("formal protocol is not in the frozen pre-result state")
    input_record = protocol.get("input")
    if not isinstance(input_record, dict):
        raise Qwen3FormalError("formal protocol input must be an object")
    if input_record.get("selection_id") != QWEN3_SELECTION_ID:
        raise Qwen3FormalError("formal protocol selection ID mismatch")
    if input_record.get("anchor_count") != QWEN3_ANCHOR_COUNT:
        raise Qwen3FormalError("formal protocol anchor count mismatch")
    if input_record.get("anchor_indices") != list(range(QWEN3_ANCHOR_COUNT)):
        raise Qwen3FormalError("formal protocol anchor indices mismatch")
    if input_record.get("prompt_lengths") != list(QWEN3_PROMPT_LENGTHS):
        raise Qwen3FormalError("formal protocol prompt lengths mismatch")
    if input_record.get("continuation_tokens") != QWEN3_CONTINUATION_TOKENS:
        raise Qwen3FormalError("formal protocol continuation length mismatch")
    counts = protocol.get("case_counts")
    if not isinstance(counts, dict) or counts.get("total_unique_method_length_cases") != 1300:
        raise Qwen3FormalError("formal protocol case count mismatch")
    _require_sha256("input.token_ids_sha256", input_record.get("token_ids_sha256"))
    _require_sha256(
        "input.corpus_snapshot_sha256", input_record.get("corpus_snapshot_sha256")
    )


def _validate_source_state(source_state: Any) -> None:
    if not isinstance(source_state, dict):
        raise Qwen3FormalError("source_state must be an object")
    commit = source_state.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise Qwen3FormalError("source_state git commit is invalid")
    if source_state.get("dirty") is not False:
        raise Qwen3FormalError("source_state must be clean")


def _require_sha256(name: str, value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise Qwen3FormalError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise Qwen3FormalError(f"{name} must be a SHA-256 hex digest") from error


__all__ = [
    "FORMAL_PROTOCOL_ID",
    "FORMAL_EXECUTION_ID",
    "FORMAL_ACCEPTANCE_GATE_ID",
    "INPUT_MANIFEST_SCHEMA_VERSION",
    "Qwen3FormalError",
    "build_input_manifest",
    "atomic_write_json",
    "expand_partition_cases",
    "file_sha256",
    "load_formal_protocol",
    "load_execution_config",
    "load_acceptance_gate",
    "formal_method_length_points",
    "formal_scoring",
    "read_corpus_snapshot",
    "validate_input_manifest",
    "validate_completed_result",
]
