from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v3_promotion_acceptance import (
    PARTITIONS,
    expand_promotion_methods,
)
from utils.qwen3_cage_v3_promotion_protocol import PROMPT_LENGTHS
from utils.qwen3_cage_v4_data import canonical_sha256


class CageV3PromotionFullError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3PromotionFullError(message)


def full_holdout_inputs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    anchors = sorted(
        (row for row in manifest.get("anchors", []) if row.get("partition") == "holdout"),
        key=lambda row: (row["document_id"], row["anchor_index"]),
    )
    _require(len(anchors) == 40, "promotion holdout anchor count mismatch")
    _require(len({(row["document_id"], row["anchor_index"]) for row in anchors}) == 40, "promotion holdout anchors are not unique")
    case_by_id = {row["case_id"]: row for row in manifest.get("cases", [])}
    inputs = []
    for anchor in anchors:
        _require(anchor.get("anchor_index") in (0, 1), "promotion holdout anchor index changed")
        full_window = anchor.get("full_window_ids", [])
        _require(len(full_window) == max(PROMPT_LENGTHS) + 64, "promotion full-window length mismatch")
        compact = {row["prompt_length"]: row for row in anchor.get("cases", [])}
        _require(set(compact) == set(PROMPT_LENGTHS), "promotion anchor prompt lengths changed")
        for length in PROMPT_LENGTHS:
            source = case_by_id.get(compact[length]["case_id"])
            _require(isinstance(source, dict), "promotion compact case is missing")
            identity = source.get("identity", {})
            _require(identity.get("partition") == "holdout", "promotion input partition changed")
            _require(identity.get("document_id") == anchor["document_id"], "promotion input document changed")
            _require(identity.get("anchor_index") == anchor["anchor_index"], "promotion input anchor changed")
            _require(identity.get("prompt_length") == length, "promotion input prompt length changed")
            prompt_offset = max(PROMPT_LENGTHS) - length
            prompt_ids = list(full_window[prompt_offset : max(PROMPT_LENGTHS)])
            continuation_ids = list(full_window[-64:])
            _require(len(prompt_ids) == length and len(continuation_ids) == 64, "promotion input token length mismatch")
            inputs.append(
                {
                    "input_case_id": source["case_id"],
                    "identity": copy.deepcopy(identity),
                    "prompt_ids": prompt_ids,
                    "continuation_ids": continuation_ids,
                }
            )
    _require(len(inputs) == 120, "promotion holdout input count mismatch")
    _require(len({row["input_case_id"] for row in inputs}) == 120, "promotion holdout input IDs are not unique")
    return inputs


def expand_full_holdout_cases(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    gate_sha256: str,
    manifest: Mapping[str, Any],
    partition: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    _require(partition in PARTITIONS, "unsupported promotion partition")
    methods = expand_promotion_methods(protocol, repo_root=repo_root, partition=partition)
    methods_by_length: dict[int, list[dict[str, Any]]] = {length: [] for length in PROMPT_LENGTHS}
    for method in methods:
        methods_by_length[method["prompt_length"]].append(method)
    expected_methods_per_length = 4 if partition == "cage_qwen3" else 1
    _require(all(len(rows) == expected_methods_per_length for rows in methods_by_length.values()), "promotion full method grid mismatch")
    cases = []
    for source in full_holdout_inputs(manifest):
        length = source["identity"]["prompt_length"]
        for method in methods_by_length[length]:
            identity = {
                "gate_sha256": gate_sha256,
                "protocol_sha256": protocol_sha256,
                "partition": partition,
                "stage": "promotion_full_holdout",
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
    expected = 480 if partition == "cage_qwen3" else 120
    _require(len(cases) == expected, "promotion full case count mismatch")
    _require(len({case["case_id"] for case in cases}) == expected, "promotion full case IDs are not unique")
    return cases


__all__ = [
    "CageV3PromotionFullError",
    "expand_full_holdout_cases",
    "full_holdout_inputs",
]
