from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


PROMPT_LENGTHS = (1024, 2048, 4032)
CONTINUATION_TOKENS = 64
CANDIDATE_ANCHOR_COUNTS = (15, 30, 50)


def token_ids_sha256(values: Sequence[int]) -> str:
    _validate_token_ids(values)
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def interior_anchor_starts(token_count: int, anchor_count: int) -> list[int]:
    if type(token_count) is not int or token_count < 1:
        raise ValueError("token_count must be a positive integer")
    if type(anchor_count) is not int or anchor_count < 2:
        raise ValueError("anchor_count must be an integer >= 2")
    minimum = max(PROMPT_LENGTHS)
    maximum = token_count - CONTINUATION_TOKENS
    if maximum < minimum:
        raise ValueError("token stream is too short for the CAGE-v3 audit window")
    return [
        minimum + ((maximum - minimum) * (index + 1) // (anchor_count + 1))
        for index in range(anchor_count)
    ]


def candidate_selection_audit(token_ids: Sequence[int], anchor_count: int) -> dict[str, Any]:
    _validate_token_ids(token_ids)
    starts = interior_anchor_starts(len(token_ids), anchor_count)
    full_window = max(PROMPT_LENGTHS) + CONTINUATION_TOKENS
    gaps = [right - left for left, right in zip(starts, starts[1:])]
    window_hashes = []
    for start in starts:
        window = token_ids[start - max(PROMPT_LENGTHS) : start + CONTINUATION_TOKENS]
        if len(window) != full_window:
            raise ValueError("candidate audit window has the wrong length")
        window_hashes.append(token_ids_sha256(window))
    return {
        "selection_id": f"interior-{anchor_count + 1}sts-audit-v1",
        "anchor_count": anchor_count,
        "prompt_lengths": list(PROMPT_LENGTHS),
        "continuation_tokens": CONTINUATION_TOKENS,
        "full_window_tokens": full_window,
        "minimum_anchor_gap": min(gaps),
        "windows_nonoverlapping": min(gaps) >= full_window,
        "continuation_starts": starts,
        "full_window_ids_sha256": window_hashes,
    }


def _validate_token_ids(values: Sequence[int]) -> None:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("token_ids must be a sequence")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("token_ids must contain nonnegative integers")


__all__ = [
    "CANDIDATE_ANCHOR_COUNTS",
    "CONTINUATION_TOKENS",
    "PROMPT_LENGTHS",
    "candidate_selection_audit",
    "interior_anchor_starts",
    "token_ids_sha256",
]
