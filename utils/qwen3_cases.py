from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


QWEN3_PROMPT_LENGTHS = (1024, 2048, 4032)
QWEN3_CONTINUATION_TOKENS = 64
QWEN3_ANCHOR_COUNT = 50
QWEN3_SELECTION_ID = "interior-51sts-v1"


def token_ids_sha256(values: Sequence[int]) -> str:
    _validate_token_ids(values)
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def continuation_anchor(token_count: int, anchor_index: int) -> int:
    _require_int("token_count", token_count, minimum=1)
    _require_int("anchor_index", anchor_index, minimum=0)
    if anchor_index >= QWEN3_ANCHOR_COUNT:
        raise ValueError(f"anchor_index must be smaller than {QWEN3_ANCHOR_COUNT}")
    minimum = max(QWEN3_PROMPT_LENGTHS)
    maximum = token_count - QWEN3_CONTINUATION_TOKENS
    if maximum < minimum:
        raise ValueError("token stream is too short for the Qwen3 protocol")
    return minimum + (
        (maximum - minimum) * (anchor_index + 1) // (QWEN3_ANCHOR_COUNT + 1)
    )


def build_anchor_records(token_ids: Sequence[int]) -> list[dict[str, Any]]:
    _validate_token_ids(token_ids)
    anchors: list[dict[str, Any]] = []
    for anchor_index in range(QWEN3_ANCHOR_COUNT):
        start = continuation_anchor(len(token_ids), anchor_index)
        continuation = token_ids[start : start + QWEN3_CONTINUATION_TOKENS]
        prompts = []
        for prompt_length in QWEN3_PROMPT_LENGTHS:
            prompt = token_ids[start - prompt_length : start]
            full = [*prompt, *continuation]
            if len(prompt) != prompt_length:
                raise ValueError("expanded prompt length differs from the protocol")
            prompts.append(
                {
                    "prompt_length": prompt_length,
                    "prompt_ids_sha256": token_ids_sha256(prompt),
                    "full_ids_sha256": token_ids_sha256(full),
                }
            )
        anchors.append(
            {
                "anchor_index": anchor_index,
                "continuation_start": start,
                "continuation_ids_sha256": token_ids_sha256(continuation),
                "prompts": prompts,
            }
        )
    validate_anchor_records(anchors, token_count=len(token_ids))
    return anchors


def validate_anchor_records(
    anchors: Sequence[dict[str, Any]], *, token_count: int
) -> None:
    _require_int("token_count", token_count, minimum=1)
    if len(anchors) != QWEN3_ANCHOR_COUNT:
        raise ValueError(f"anchor records must contain {QWEN3_ANCHOR_COUNT} entries")
    expected_indices = list(range(QWEN3_ANCHOR_COUNT))
    actual_indices = [record.get("anchor_index") for record in anchors]
    if actual_indices != expected_indices:
        raise ValueError("anchor indices are not the frozen ordered range")
    starts = [record.get("continuation_start") for record in anchors]
    if any(type(value) is not int for value in starts):
        raise ValueError("continuation starts must be integers")
    expected_starts = [continuation_anchor(token_count, index) for index in expected_indices]
    if starts != expected_starts:
        raise ValueError("continuation starts differ from the frozen selection rule")
    full_window = max(QWEN3_PROMPT_LENGTHS) + QWEN3_CONTINUATION_TOKENS
    gaps = [right - left for left, right in zip(starts, starts[1:])]
    if gaps and min(gaps) < full_window:
        raise ValueError("maximum-context anchor windows overlap")


def minimum_anchor_gap(anchors: Sequence[dict[str, Any]]) -> int | None:
    starts = [record["continuation_start"] for record in anchors]
    return min((right - left for left, right in zip(starts, starts[1:])), default=None)


def _validate_token_ids(values: Sequence[int]) -> None:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("token_ids must be a sequence")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("token_ids must contain nonnegative integers")


def _require_int(name: str, value: int, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


__all__ = [
    "QWEN3_ANCHOR_COUNT",
    "QWEN3_CONTINUATION_TOKENS",
    "QWEN3_PROMPT_LENGTHS",
    "QWEN3_SELECTION_ID",
    "build_anchor_records",
    "continuation_anchor",
    "minimum_anchor_gap",
    "token_ids_sha256",
    "validate_anchor_records",
]
