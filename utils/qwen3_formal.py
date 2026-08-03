from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
    "INPUT_MANIFEST_SCHEMA_VERSION",
    "Qwen3FormalError",
    "build_input_manifest",
    "file_sha256",
    "load_formal_protocol",
    "read_corpus_snapshot",
    "validate_input_manifest",
]
