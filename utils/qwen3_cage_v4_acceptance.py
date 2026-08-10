from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, TypeVar

from utils.qwen3_cage_v3_screen import expand_screen_methods, validate_quota_plan
from utils.qwen3_cage_v4_data import CONTINUATION_TOKENS, PROMPT_LENGTHS, canonical_sha256
from utils.qwen3_cage_v4_metric_protocol import COMPRESSED_METHODS
from utils.qwen3_formal import formal_method_length_points, load_formal_protocol


PARTITIONS = ("cage_qwen3", "kitty_qwen3")
T = TypeVar("T")
SCIENTIFIC_FIELDS = (
    "case_id",
    "partition",
    "method",
    "input",
    "memory",
    "scoring",
    "local_perturbation",
    "quality_cache",
)


class CageV4AcceptanceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV4AcceptanceError(message)


def call_with_recorder_uninstalled(
    *, recorder: Any, model: Any, callback: Callable[[], T], expected_modules: int
) -> T:
    """Run quality scoring without allowing the local-metric callback to fire."""
    removed = recorder.uninstall()
    _require(removed == expected_modules, "perturbation recorder uninstall count mismatch")
    try:
        result = callback()
    except BaseException as error:
        installed = recorder.install(model)
        if installed != expected_modules:
            raise CageV4AcceptanceError(
                "perturbation recorder restore failed after quality-scoring error"
            ) from error
        raise
    installed = recorder.install(model)
    _require(installed == expected_modules, "perturbation recorder restore count mismatch")
    return result


def expand_metric_methods(
    protocol: dict[str, Any], *, repo_root: Path, partition: str
) -> list[dict[str, Any]]:
    _require(partition in PARTITIONS, "unsupported metric-validity partition")
    grid = {method["method_id"]: method for method in protocol["method_grid"]}
    formal_source = repo_root / protocol["frozen_method_sources"]["formal_quality_protocol"]["path"]
    formal, _ = load_formal_protocol(formal_source)
    formal_points = {
        (point["method_id"], point["prompt_length"]): point
        for point in formal_method_length_points(formal, partition="cage_qwen3")
    }
    v3_source = repo_root / protocol["frozen_method_sources"]["cage_v3_protocol"]["path"]
    v3_protocol = json.loads(v3_source.read_text(encoding="utf-8"))
    plan_source = repo_root / protocol["frozen_method_sources"]["cage_v3_quota_plan"]["path"]
    plan = json.loads(plan_source.read_text(encoding="utf-8"))
    validate_quota_plan(plan)

    if partition == "kitty_qwen3":
        sources = expand_screen_methods(protocol=v3_protocol, plan=plan, partition=partition)
        selected = {row["prompt_length"]: row for row in sources}
        methods = []
        for point in grid["kitty-pro-25pct"]["points"]:
            source = copy.deepcopy(selected[point["prompt_length"]])
            _require(source["packed_bytes"] == point["packed_bytes"], "Kitty-Pro bytes changed")
            source.update(
                id=f"kitty-pro-25pct-l{point['prompt_length']}",
                metric_method_id="kitty-pro-25pct",
                runtime_family="round1",
            )
            methods.append(source)
        return methods

    methods = []
    formal_map = {
        "fp16": {1024: "fp16", 2048: "fp16", 4032: "fp16"},
        "kivi-kittypro-matched": {
            1024: "kivi-g64-r320",
            2048: "kivi-g32-r224",
            4032: "kivi-g128-r256",
        },
        "cage-v1-kittypro-matched": {
            1024: "cage-r224",
            2048: "cage-r288",
            4032: "cage-r96",
        },
    }
    for metric_method_id, mapping in formal_map.items():
        for point in grid[metric_method_id]["points"]:
            prompt_length = point["prompt_length"]
            source = formal_points[(mapping[prompt_length], prompt_length)]
            methods.append(
                {
                    "id": f"{metric_method_id}-l{prompt_length}",
                    "metric_method_id": metric_method_id,
                    "name": source["method"],
                    "prompt_length": prompt_length,
                    "packed_bytes": point["packed_bytes"],
                    "config": copy.deepcopy(source["config"]),
                    "runtime_family": "formal",
                }
            )
    screen_sources = expand_screen_methods(protocol=v3_protocol, plan=plan, partition=partition)
    families = {
        "cage-v2-mixed-sink32-kittypro": "cage-v2-best-control",
        "cage-v3-sr2-sink32-calibrated": "pure-sr2-sink32-calibrated",
    }
    for metric_method_id, family_id in families.items():
        selected = {
            row["prompt_length"]: row
            for row in screen_sources
            if row["family_id"] == family_id
        }
        _require(len(selected) == 3, f"missing frozen {family_id} points")
        for point in grid[metric_method_id]["points"]:
            source = copy.deepcopy(selected[point["prompt_length"]])
            _require(source["packed_bytes"] == point["packed_bytes"], f"{family_id} bytes changed")
            source.update(
                id=f"{metric_method_id}-l{point['prompt_length']}",
                metric_method_id=metric_method_id,
                runtime_family="round1",
            )
            methods.append(source)
    expected = ["fp16", *COMPRESSED_METHODS]
    observed = []
    for method_id in expected:
        observed.extend([method for method in methods if method["metric_method_id"] == method_id])
    _require(len(observed) == 15, "CAGE metric-validity method count mismatch")
    return observed


def acceptance_input_cases(
    manifest: Mapping[str, Any], *, protocol: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    policy = protocol["acceptance_policy"]
    matches = [
        anchor
        for anchor in manifest.get("anchors", [])
        if anchor.get("document_id") == policy["document_id"]
        and anchor.get("anchor_index") == policy["anchor_index"]
        and anchor.get("partition") == "screen"
    ]
    _require(len(matches) == 1, "acceptance input anchor is not unique")
    anchor = matches[0]
    full_window = anchor["full_window_ids"]
    compact_by_length = {row["prompt_length"]: row for row in anchor["cases"]}
    case_by_id = {row["case_id"]: row for row in manifest["cases"]}
    result = {}
    for prompt_length in PROMPT_LENGTHS:
        compact = compact_by_length[prompt_length]
        source = case_by_id[compact["case_id"]]
        prompt_offset = max(PROMPT_LENGTHS) - prompt_length
        result[prompt_length] = {
            "input_case_id": source["case_id"],
            "identity": copy.deepcopy(source["identity"]),
            "prompt_ids": list(full_window[prompt_offset : max(PROMPT_LENGTHS)]),
            "continuation_ids": list(full_window[-CONTINUATION_TOKENS:]),
        }
    return result


def expand_acceptance_cases(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    manifest: dict[str, Any],
    partition: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    methods = expand_metric_methods(protocol, repo_root=repo_root, partition=partition)
    inputs = acceptance_input_cases(manifest, protocol=protocol)
    cases = []
    for method in methods:
        source = inputs[method["prompt_length"]]
        identity = {
            "metric_protocol_sha256": protocol_sha256,
            "partition": partition,
            "stage": "acceptance",
            "method_id": method["id"],
            "input_case_id": source["input_case_id"],
        }
        cases.append(
            {
                "case_id": canonical_sha256(identity)[:24],
                "partition": partition,
                "method": method,
                "input": copy.deepcopy(source["identity"]),
                "prompt_ids": source["prompt_ids"],
                "continuation_ids": source["continuation_ids"],
            }
        )
    expected = 15 if partition == "cage_qwen3" else 3
    _require(len(cases) == expected, "metric-validity acceptance case count mismatch")
    _require(len({case["case_id"] for case in cases}) == expected, "acceptance case IDs are not unique")
    return cases


__all__ = [
    "CageV4AcceptanceError",
    "PARTITIONS",
    "SCIENTIFIC_FIELDS",
    "acceptance_input_cases",
    "call_with_recorder_uninstalled",
    "expand_acceptance_cases",
    "expand_metric_methods",
]
