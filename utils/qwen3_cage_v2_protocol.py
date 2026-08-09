from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from utils.qwen3_cases import token_ids_sha256


def build_cage_v2_dev_manifest(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    token_ids: Sequence[int],
    corpus_snapshot_sha256: str,
    corpus_identity: dict[str, Any],
    tokenizer_identity: dict[str, Any],
    source_state: dict[str, Any],
) -> dict[str, Any]:
    data = protocol["development_data"]
    _validate_token_ids(token_ids)
    anchor_count = _positive_int("anchor_count", data["anchor_count"])
    prompt_lengths = tuple(_positive_int("prompt_length", x) for x in data["prompt_lengths"])
    continuation_tokens = _positive_int("continuation_tokens", data["continuation_tokens"])
    minimum = max(prompt_lengths)
    maximum = len(token_ids) - continuation_tokens
    if maximum < minimum:
        raise ValueError("development token stream is too short")

    cases = []
    starts = []
    for anchor_index in range(anchor_count):
        start = minimum + (maximum - minimum) * (anchor_index + 1) // (anchor_count + 1)
        starts.append(start)
        continuation = list(token_ids[start : start + continuation_tokens])
        for prompt_length in prompt_lengths:
            prompt = list(token_ids[start - prompt_length : start])
            identity = {
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha256,
                "corpus_split": data["corpus_split"],
                "anchor_index": anchor_index,
                "continuation_start": start,
                "prompt_length": prompt_length,
                "prompt_ids_sha256": token_ids_sha256(prompt),
                "continuation_ids_sha256": token_ids_sha256(continuation),
            }
            cases.append(
                {
                    "case_id": _canonical_sha256(identity)[:24],
                    "identity": identity,
                    "prompt_ids": prompt,
                    "continuation_ids": continuation,
                }
            )

    maximum_window = minimum + continuation_tokens
    gaps = [right - left for left, right in zip(starts, starts[1:])]
    if gaps and min(gaps) < maximum_window:
        raise ValueError("development maximum-context windows overlap")
    manifest = {
        "schema_version": 1,
        "manifest_id": "qwen3-8b-cage-v2-development-inputs-v1",
        "status": "development_only_not_eligible_for_final_claims",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "corpus_snapshot_sha256": corpus_snapshot_sha256,
        "corpus_identity": corpus_identity,
        "tokenizer_identity": tokenizer_identity,
        "source_state": source_state,
        "token_count": len(token_ids),
        "token_ids_sha256": token_ids_sha256(token_ids),
        "selection": {
            "selection_id": data["selection_id"],
            "anchor_count": anchor_count,
            "continuation_starts": starts,
            "minimum_anchor_gap": min(gaps) if gaps else None,
            "maximum_window_tokens": maximum_window,
        },
        "cases": cases,
    }
    validate_cage_v2_dev_manifest(manifest, protocol=protocol, protocol_sha256=protocol_sha256)
    return manifest


def validate_cage_v2_dev_manifest(
    manifest: dict[str, Any], *, protocol: dict[str, Any], protocol_sha256: str
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("development manifest schema mismatch")
    if manifest.get("status") != "development_only_not_eligible_for_final_claims":
        raise ValueError("development manifest must be claim-ineligible")
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("development manifest protocol ID mismatch")
    if manifest.get("protocol_sha256") != protocol_sha256:
        raise ValueError("development manifest protocol hash mismatch")
    data = protocol["development_data"]
    if manifest.get("corpus_identity", {}).get("split") != data["corpus_split"]:
        raise ValueError("development manifest corpus split mismatch")
    cases = manifest.get("cases")
    expected_count = data["anchor_count"] * len(data["prompt_lengths"])
    if not isinstance(cases, list) or len(cases) != expected_count:
        raise ValueError("development manifest case count mismatch")
    case_ids = [case.get("case_id") for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("development manifest case IDs are not unique")
    for case in cases:
        identity = case.get("identity", {})
        prompt = case.get("prompt_ids")
        continuation = case.get("continuation_ids")
        if len(prompt) != identity.get("prompt_length"):
            raise ValueError("development prompt length mismatch")
        if len(continuation) != data["continuation_tokens"]:
            raise ValueError("development continuation length mismatch")
        if token_ids_sha256(prompt) != identity.get("prompt_ids_sha256"):
            raise ValueError("development prompt hash mismatch")
        if token_ids_sha256(continuation) != identity.get("continuation_ids_sha256"):
            raise ValueError("development continuation hash mismatch")
        if _canonical_sha256(identity)[:24] != case["case_id"]:
            raise ValueError("development case ID mismatch")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_token_ids(values: Sequence[int]) -> None:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("token_ids must be a sequence")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("token_ids must contain nonnegative integers")


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


__all__ = ["build_cage_v2_dev_manifest", "validate_cage_v2_dev_manifest"]
