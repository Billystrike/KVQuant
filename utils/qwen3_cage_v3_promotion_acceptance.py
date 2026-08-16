from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v3_promotion_protocol import (
    PROMPT_LENGTHS,
    load_promotion_protocol,
)
from utils.qwen3_cage_v3_screen import expand_screen_methods, validate_quota_plan
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256
from utils.qwen3_formal import formal_method_length_points, load_formal_protocol
from utils.qwen3_memory import (
    estimate_qwen3_cage_bytes,
    estimate_qwen3_fp16_bytes,
    estimate_qwen3_kivi_bytes,
)


EXECUTION_ID = "qwen3-8b-cage-v3-promotion-acceptance-v1"
PARTITIONS = ("cage_qwen3", "kitty_qwen3")
SCIENTIFIC_FIELDS = (
    "case_id",
    "partition",
    "method",
    "input",
    "memory",
    "scoring",
    "quality_cache",
)


class CageV3PromotionAcceptanceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3PromotionAcceptanceError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV3PromotionAcceptanceError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_static_preflight_receipt(
    value: Mapping[str, Any], *, execution: Mapping[str, Any], protocol_sha256: str
) -> None:
    _require(value.get("schema_version") == 1, "static preflight schema mismatch")
    _require(value.get("status") == "pass", "static preflight did not pass")
    _require(value.get("claim_eligible") is False, "static preflight claim boundary changed")
    _require(value.get("protocol_sha256") == protocol_sha256, "static preflight protocol mismatch")
    _require(
        value.get("input_manifest_sha256") == execution["input_manifest"]["sha256"],
        "static preflight manifest mismatch",
    )
    frozen_input = execution["acceptance_input"]
    expected_input = {
        key: frozen_input[key]
        for key in ("anchor_id", "anchor_index", "document_id", "input_case_ids")
    }
    _require(value.get("acceptance_input") == expected_input, "static preflight input changed")
    _require(
        value.get("full_case_counts") == {
            "cage_qwen3": 480,
            "kitty_qwen3": 120,
            "total": 600,
        },
        "static preflight full-case boundary changed",
    )
    boundary = value.get("boundary", {})
    _require(
        boundary
        == {
            "full_holdout_authorized": False,
            "gpu_acceptance_authorized_by_this_preflight": False,
            "kitty_llama_port_authorized": False,
            "llama2_execution_authorized": False,
            "pg19_test_access_authorized": False,
            "reads_holdout_method_metrics": False,
        },
        "static preflight authorization boundary changed",
    )


def load_acceptance_execution(
    path: Path,
    *,
    repo_root: Path,
    verify_server_artifacts: bool,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    execution = _load(path)
    _require(execution.get("schema_version") == 1, "acceptance execution schema mismatch")
    _require(execution.get("execution_id") == EXECUTION_ID, "acceptance execution ID mismatch")
    _require(
        execution.get("status")
        == "frozen_after_static_preflight_before_any_promotion_holdout_metric",
        "acceptance execution freeze boundary changed",
    )
    _require(execution.get("claim_eligible") is False, "acceptance must be claim-ineligible")

    protocol_path = Path(execution["protocol"]["path"])
    if not protocol_path.is_absolute():
        protocol_path = repo_root / protocol_path
    protocol, protocol_sha256 = load_promotion_protocol(protocol_path)
    _require(protocol_sha256 == execution["protocol"]["sha256"], "promotion protocol hash mismatch")
    _require(
        execution["input_manifest"]["sha256"]
        == protocol["input_receipt"]["input_manifest_sha256"],
        "promotion manifest receipt mismatch",
    )
    _require(
        execution["input_manifest"]["size_bytes"]
        == protocol["input_receipt"]["input_manifest_size_bytes"],
        "promotion manifest size receipt mismatch",
    )
    acceptance = protocol["execution_stages"]["gpu_acceptance"]
    expected_input = execution["acceptance_input"]
    _require(expected_input["document_id"] == acceptance["document_id"], "acceptance document changed")
    _require(expected_input["anchor_index"] == acceptance["anchor_index"], "acceptance anchor changed")
    _require(expected_input["prompt_lengths"] == list(PROMPT_LENGTHS), "acceptance lengths changed")
    _require(expected_input["continuation_tokens"] == 64, "acceptance continuation changed")
    _require(
        execution["partitions"]["cage_qwen3"]["case_count_per_repeat"]
        == acceptance["cage_partition_cases_per_repeat"]
        == 12,
        "CAGE acceptance count changed",
    )
    _require(
        execution["partitions"]["kitty_qwen3"]["case_count_per_repeat"]
        == acceptance["kitty_partition_cases_per_repeat"]
        == 3,
        "Kitty acceptance count changed",
    )
    _require(tuple(execution["scientific_payload_fields"]) == SCIENTIFIC_FIELDS, "scientific fields changed")
    _require(
        execution.get("authorization")
        == {
            "gpu_acceptance": True,
            "full_holdout": False,
            "holdout_interpretation": False,
            "pg19_test": False,
            "llama2_execution": False,
            "kitty_llama_port": False,
            "paper_claims": False,
            "runtime_claims": False,
        },
        "acceptance authorization boundary changed",
    )
    if verify_server_artifacts:
        manifest_path = Path(execution["input_manifest"]["path"])
        preflight = execution["static_preflight"]
        checks = (
            file_sha256(manifest_path) == execution["input_manifest"]["sha256"],
            manifest_path.stat().st_size == execution["input_manifest"]["size_bytes"],
            file_sha256(Path(preflight["output_path"])) == preflight["output_sha256"],
            Path(preflight["output_path"]).stat().st_size == preflight["output_size_bytes"],
            file_sha256(Path(preflight["log_path"])) == preflight["log_sha256"],
            Path(preflight["log_path"]).stat().st_size == preflight["log_size_bytes"],
        )
        _require(all(checks), "server-side preflight or manifest artifact mismatch")
        preflight_value = _load(Path(preflight["output_path"]))
        validate_static_preflight_receipt(
            preflight_value,
            execution=execution,
            protocol_sha256=protocol_sha256,
        )
    return execution, file_sha256(path), protocol, protocol_sha256


def _formal_point_bytes(point: dict[str, Any]) -> int:
    name = point["method"]
    config = point["config"]
    length = point["prompt_length"]
    if name == "fp16":
        report = estimate_qwen3_fp16_bytes(seq_len=length)
    elif name == "kivi":
        report = estimate_qwen3_kivi_bytes(
            seq_len=length,
            group_size=config["group_size"],
            residual_length=config["residual_length"],
            bits=config["bits"],
        )
    elif name == "cage":
        report = estimate_qwen3_cage_bytes(
            seq_len=length,
            residual_length=config["residual_length"],
            key_group_sizes=config["key_group_sizes"],
            value_group_sizes=config["value_group_sizes"],
            bits=config["bits"],
        )
    else:
        raise CageV3PromotionAcceptanceError(f"unsupported formal method: {name}")
    return int(report["model_total_bytes"])


def expand_promotion_methods(
    protocol: dict[str, Any], *, repo_root: Path, partition: str
) -> list[dict[str, Any]]:
    _require(partition in PARTITIONS, "unsupported promotion partition")
    grid = {row["method_id"]: row for row in protocol["method_grid"]}
    formal_path = repo_root / protocol["frozen_sources"]["formal_quality_protocol"]["path"]
    formal, formal_sha256 = load_formal_protocol(formal_path)
    _require(
        formal_sha256 == protocol["frozen_sources"]["formal_quality_protocol"]["sha256"],
        "formal quality protocol hash mismatch",
    )
    formal_points = {
        (point["method_id"], point["prompt_length"]): point
        for point in formal_method_length_points(formal, partition="cage_qwen3")
    }
    v3_path = repo_root / protocol["frozen_sources"]["cage_v3_protocol"]["path"]
    _require(file_sha256(v3_path) == protocol["frozen_sources"]["cage_v3_protocol"]["sha256"], "CAGE-v3 protocol hash mismatch")
    v3_protocol = _load(v3_path)
    plan_path = repo_root / protocol["frozen_sources"]["cage_v3_quota_plan"]["path"]
    _require(file_sha256(plan_path) == protocol["frozen_sources"]["cage_v3_quota_plan"]["sha256"], "CAGE-v3 quota hash mismatch")
    plan = _load(plan_path)
    validate_quota_plan(plan)

    if partition == "kitty_qwen3":
        source_by_length = {
            row["prompt_length"]: row
            for row in expand_screen_methods(protocol=v3_protocol, plan=plan, partition=partition)
        }
        methods = []
        for point in grid["kitty-pro-25pct"]["points"]:
            source = copy.deepcopy(source_by_length[point["prompt_length"]])
            _require(source["packed_bytes"] == point["packed_bytes"], "Kitty-Pro bytes changed")
            source.update(
                id=f"kitty-pro-25pct-l{point['prompt_length']}",
                metric_method_id="kitty-pro-25pct",
                runtime_family="round1",
            )
            methods.append(source)
        return methods

    methods: list[dict[str, Any]] = []
    formal_map = {
        "fp16": {length: "fp16" for length in PROMPT_LENGTHS},
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
    for method_id, mapping in formal_map.items():
        for point in grid[method_id]["points"]:
            length = point["prompt_length"]
            source = formal_points[(mapping[length], length)]
            _require(_formal_point_bytes(source) == point["packed_bytes"], f"{method_id} bytes changed")
            methods.append(
                {
                    "id": f"{method_id}-l{length}",
                    "metric_method_id": method_id,
                    "name": source["method"],
                    "prompt_length": length,
                    "packed_bytes": point["packed_bytes"],
                    "config": copy.deepcopy(source["config"]),
                    "runtime_family": "formal",
                }
            )
    v3_sources = {
        row["prompt_length"]: row
        for row in expand_screen_methods(protocol=v3_protocol, plan=plan, partition=partition)
        if row["family_id"] == "pure-sr2-sink32-calibrated"
    }
    _require(len(v3_sources) == 3, "frozen calibrated CAGE-v3 points are missing")
    for point in grid["cage-v3-sr2-sink32-calibrated"]["points"]:
        source = copy.deepcopy(v3_sources[point["prompt_length"]])
        _require(source["packed_bytes"] == point["packed_bytes"], "CAGE-v3 bytes changed")
        source.update(
            id=f"cage-v3-sr2-sink32-calibrated-l{point['prompt_length']}",
            metric_method_id="cage-v3-sr2-sink32-calibrated",
            runtime_family="round1",
        )
        methods.append(source)
    ordered_ids = execution_method_ids(protocol, partition=partition)
    ordered = [method for method_id in ordered_ids for method in methods if method["metric_method_id"] == method_id]
    _require(len(ordered) == 12, "CAGE promotion method count mismatch")
    return ordered


def execution_method_ids(protocol: dict[str, Any], *, partition: str) -> list[str]:
    return [row["method_id"] for row in protocol["method_grid"] if row["partition"] == partition]


def acceptance_inputs(
    manifest: Mapping[str, Any], *, execution: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    frozen = execution["acceptance_input"]
    matches = [
        anchor
        for anchor in manifest.get("anchors", [])
        if anchor.get("document_id") == frozen["document_id"]
        and anchor.get("anchor_index") == frozen["anchor_index"]
        and anchor.get("partition") == "holdout"
    ]
    _require(len(matches) == 1, "promotion acceptance anchor is not unique")
    anchor = matches[0]
    _require(anchor.get("anchor_id") == frozen["anchor_id"], "promotion acceptance anchor ID changed")
    full_window = anchor["full_window_ids"]
    compact_by_length = {row["prompt_length"]: row for row in anchor["cases"]}
    case_by_id = {row["case_id"]: row for row in manifest["cases"]}
    result = {}
    observed_ids = []
    for length in PROMPT_LENGTHS:
        compact = compact_by_length[length]
        source = case_by_id[compact["case_id"]]
        observed_ids.append(source["case_id"])
        prompt_offset = max(PROMPT_LENGTHS) - length
        result[length] = {
            "input_case_id": source["case_id"],
            "identity": copy.deepcopy(source["identity"]),
            "prompt_ids": list(full_window[prompt_offset : max(PROMPT_LENGTHS)]),
            "continuation_ids": list(full_window[-64:]),
        }
        _require(len(result[length]["prompt_ids"]) == length, "promotion prompt length mismatch")
        _require(len(result[length]["continuation_ids"]) == 64, "promotion continuation length mismatch")
    _require(observed_ids == frozen["input_case_ids"], "promotion acceptance input IDs changed")
    return result


def expand_acceptance_cases(
    *,
    execution: dict[str, Any],
    execution_sha256: str,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    partition: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    methods = expand_promotion_methods(protocol, repo_root=repo_root, partition=partition)
    inputs = acceptance_inputs(manifest, execution=execution)
    cases = []
    for method in methods:
        source = inputs[method["prompt_length"]]
        identity = {
            "execution_sha256": execution_sha256,
            "partition": partition,
            "stage": "promotion_acceptance",
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
    expected = execution["partitions"][partition]["case_count_per_repeat"]
    _require(len(cases) == expected, "promotion acceptance case count mismatch")
    _require(len({case["case_id"] for case in cases}) == expected, "promotion case IDs are not unique")
    return cases


__all__ = [
    "CageV3PromotionAcceptanceError",
    "EXECUTION_ID",
    "PARTITIONS",
    "SCIENTIFIC_FIELDS",
    "acceptance_inputs",
    "expand_acceptance_cases",
    "expand_promotion_methods",
    "load_acceptance_execution",
    "validate_static_preflight_receipt",
]
