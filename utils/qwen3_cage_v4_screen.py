from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v4_acceptance import CageV4AcceptanceError, expand_metric_methods
from utils.qwen3_cage_v4_data import CONTINUATION_TOKENS, PROMPT_LENGTHS, canonical_sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV4AcceptanceError(message)


def screen_input_cases(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    case_by_id = {row["case_id"]: row for row in manifest["cases"]}
    result = []
    for anchor in manifest["anchors"]:
        if anchor["partition"] != "screen":
            continue
        full_window = anchor["full_window_ids"]
        for compact in anchor["cases"]:
            prompt_length = compact["prompt_length"]
            source = case_by_id[compact["case_id"]]
            prompt_offset = max(PROMPT_LENGTHS) - prompt_length
            result.append(
                {
                    "input_case_id": source["case_id"],
                    "identity": copy.deepcopy(source["identity"]),
                    "prompt_ids": list(full_window[prompt_offset : max(PROMPT_LENGTHS)]),
                    "continuation_ids": list(full_window[-CONTINUATION_TOKENS:]),
                }
            )
    _require(len(result) == 120, "PG-19 screen input count mismatch")
    _require(
        len({row["input_case_id"] for row in result}) == 120,
        "PG-19 screen input IDs are not unique",
    )
    return result


def expand_full_screen_cases(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    manifest: dict[str, Any],
    partition: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    methods = expand_metric_methods(protocol, repo_root=repo_root, partition=partition)
    inputs = screen_input_cases(manifest)
    cases = []
    for method in methods:
        for source in inputs:
            if source["identity"]["prompt_length"] != method["prompt_length"]:
                continue
            identity = {
                "metric_protocol_sha256": protocol_sha256,
                "partition": partition,
                "stage": "screen_full",
                "method_id": method["id"],
                "input_case_id": source["input_case_id"],
            }
            cases.append(
                {
                    "case_id": canonical_sha256(identity)[:24],
                    "partition": partition,
                    "method": copy.deepcopy(method),
                    "input": copy.deepcopy(source["identity"]),
                    "prompt_ids": list(source["prompt_ids"]),
                    "continuation_ids": list(source["continuation_ids"]),
                }
            )
    expected = 600 if partition == "cage_qwen3" else 120
    _require(len(cases) == expected, "metric-validity full-screen case count mismatch")
    _require(
        len({case["case_id"] for case in cases}) == expected,
        "full-screen case IDs are not unique",
    )
    return cases


__all__ = ["expand_full_screen_cases", "screen_input_cases"]
