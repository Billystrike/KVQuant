from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes
from utils.qwen3_memory import estimate_qwen3_kitty_bytes


PROTOCOL_ID = "qwen3-8b-cage-v3-pure-sr2-development-v1"
PROMPT_LENGTHS = (1024, 2048, 4032)
ANCHOR_COUNT = 50
LAYER_COUNT = 36
TOTAL_TWO_BIT_CHANNELS = 1152


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cage_v3_protocol(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    protocol = json.loads(source.read_text(encoding="utf-8"))
    validate_cage_v3_protocol(protocol)
    return protocol, file_sha256(source)


def calibrated_tiered_two_bit_quotas(layer_scores: Sequence[float]) -> list[int]:
    if isinstance(layer_scores, (str, bytes)) or len(layer_scores) != LAYER_COUNT:
        raise ValueError(f"layer_scores must contain {LAYER_COUNT} values")
    normalized = []
    for layer_idx, score in enumerate(layer_scores):
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"layer_scores[{layer_idx}] must be finite")
        if float(score) < 0:
            raise ValueError(f"layer_scores[{layer_idx}] must be nonnegative")
        normalized.append(float(score))
    ranked = sorted(range(LAYER_COUNT), key=lambda index: (-normalized[index], index))
    quotas = [32] * LAYER_COUNT
    for layer_idx in ranked[:12]:
        quotas[layer_idx] = 48
    for layer_idx in ranked[-12:]:
        quotas[layer_idx] = 16
    if sum(quotas) != TOTAL_TWO_BIT_CHANNELS:
        raise AssertionError("calibrated quota total changed")
    return quotas


def validate_cage_v3_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("CAGE-v3 protocol identity mismatch")
    if protocol.get("status") != "frozen_before_any_cage_v3_gpu_metric":
        raise ValueError("CAGE-v3 protocol is not frozen before GPU metrics")
    if protocol.get("claim_eligible") is not False:
        raise ValueError("CAGE-v3 development protocol must be claim-ineligible")
    data = protocol["development_data"]
    if data.get("corpus_split") != "train" or data.get("anchor_count") != ANCHOR_COUNT:
        raise ValueError("CAGE-v3 train/anchor identity mismatch")
    if tuple(data.get("prompt_lengths", ())) != PROMPT_LENGTHS:
        raise ValueError("CAGE-v3 prompt lengths changed")
    partitions = data.get("partitions", {})
    expected_partitions = {
        "calibration": list(range(0, 5)),
        "screen": list(range(5, 10)),
        "holdout": list(range(10, 20)),
        "reserved_unseen": list(range(20, 50)),
    }
    if partitions != expected_partitions:
        raise ValueError("CAGE-v3 anchor partitions changed")
    flattened = [index for values in partitions.values() for index in values]
    if sorted(flattened) != list(range(ANCHOR_COUNT)) or len(set(flattened)) != ANCHOR_COUNT:
        raise ValueError("CAGE-v3 anchor partitions overlap or are incomplete")
    fixed = protocol["fixed_quantization"]
    if fixed.get("total_two_bit_channels_across_36_layers_per_kv_head") != TOTAL_TWO_BIT_CHANNELS:
        raise ValueError("CAGE-v3 total refinement quota changed")
    if fixed.get("value_adaptive") is not False or protocol["method_boundary"].get("one_bit_refinement_enabled") is not False:
        raise ValueError("CAGE-v3 must not enable one-bit or adaptive Value paths")
    quota = protocol["calibration"]["quota_rule"]
    if quota != {
        "top_12_layers": 48,
        "middle_12_layers": 32,
        "bottom_12_layers": 16,
        "total_channels": 1152,
        "counts_are_per_kv_head": True,
    }:
        raise ValueError("CAGE-v3 calibrated quota rule changed")

    targets = {row["prompt_length"]: row for row in protocol["kitty_pro_targets"]}
    if tuple(sorted(targets)) != PROMPT_LENGTHS:
        raise ValueError("CAGE-v3 Kitty-Pro target lengths changed")
    for length, target in targets.items():
        kitty = estimate_qwen3_kitty_bytes(seq_len=length, boosted_channels=32)
        if kitty["model_total_bytes"] != target["target_bytes"]:
            raise ValueError("CAGE-v3 Kitty-Pro target bytes changed")

    families = protocol.get("candidate_families", [])
    expected_ids = (
        "pure-sr2-sink32-uniform",
        "pure-sr2-sink32-calibrated",
        "pure-sr2-sink64-calibrated",
    )
    if tuple(row.get("family_id") for row in families) != expected_ids:
        raise ValueError("CAGE-v3 candidate family order changed")
    for family in families:
        points = family.get("points", [])
        if tuple(point.get("prompt_length") for point in points) != PROMPT_LENGTHS:
            raise ValueError("CAGE-v3 candidate point lengths changed")
        for point in points:
            report = estimate_qwen3_cage_v2_bytes(
                seq_len=point["prompt_length"],
                residual_length=point["residual_length"],
                one_bit_channels=0,
                two_bit_channels=32,
                sink_length=family["sink_length"],
            )
            if report["model_total_bytes"] != point["packed_bytes"]:
                raise ValueError("CAGE-v3 candidate packed bytes changed")
            if point["packed_bytes"] > point["target_bytes"] or point["target_bytes"] != targets[point["prompt_length"]]["target_bytes"]:
                raise ValueError("CAGE-v3 candidate exceeds or changes its target")
            _validate_byte_only_residual_selection(point, family, target_bytes=point["target_bytes"])

    controls = protocol.get("cage_v2_best_controls", [])
    if tuple(row.get("prompt_length") for row in controls) != PROMPT_LENGTHS:
        raise ValueError("CAGE-v2 controls changed")
    for control in controls:
        report = estimate_qwen3_cage_v2_bytes(
            seq_len=control["prompt_length"],
            residual_length=control["residual_length"],
            one_bit_channels=control["one_bit_channels"],
            two_bit_channels=control["two_bit_channels"],
            sink_length=control["sink_length"],
        )
        if report["model_total_bytes"] != control["packed_bytes"]:
            raise ValueError("frozen CAGE-v2 control bytes changed")
    if protocol["screen_gate"].get("maximum_candidates_advanced") != 1:
        raise ValueError("CAGE-v3 screen must advance at most one candidate")
    if data["partition_boundary"].get("reserved_unseen_execution_authorized") is not False:
        raise ValueError("reserved CAGE-v3 anchors must remain unauthorized")


def _validate_byte_only_residual_selection(point: dict[str, Any], family: dict[str, Any], *, target_bytes: int) -> None:
    feasible = []
    for residual in range(16, 513, 16):
        report = estimate_qwen3_cage_v2_bytes(
            seq_len=point["prompt_length"],
            residual_length=residual,
            one_bit_channels=0,
            two_bit_channels=32,
            sink_length=family["sink_length"],
        )
        if report["model_total_bytes"] <= target_bytes:
            feasible.append((report["model_total_bytes"], -residual, residual))
    if not feasible:
        raise ValueError("CAGE-v3 candidate has no byte-feasible residual")
    selected = max(feasible)
    if point["packed_bytes"] != selected[0] or point["residual_length"] != selected[2]:
        raise ValueError("CAGE-v3 residual does not follow the frozen byte-only rule")


__all__ = [
    "ANCHOR_COUNT",
    "LAYER_COUNT",
    "PROMPT_LENGTHS",
    "PROTOCOL_ID",
    "TOTAL_TWO_BIT_CHANNELS",
    "calibrated_tiered_two_bit_quotas",
    "file_sha256",
    "load_cage_v3_protocol",
    "validate_cage_v3_protocol",
]
