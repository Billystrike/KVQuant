from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from utils.qwen3_cage_v2_gate import file_sha256


EXPECTED_RECEIPT_SHA256 = "1ce9a6b3df36726426c5f9f1118f4a62fdd1a1512295c458f65ec90ccb5e7661"


class CageV2AnalysisError(RuntimeError):
    pass


def validate_results_receipt(receipt: dict[str, Any], *, receipt_path: Any) -> None:
    if file_sha256(receipt_path) != EXPECTED_RECEIPT_SHA256:
        raise CageV2AnalysisError("results receipt hash differs from the frozen v1 receipt")
    if receipt.get("schema_version") != 1:
        raise CageV2AnalysisError("results receipt schema mismatch")
    if receipt.get("status") != "frozen_after_joint_postrun_audit_before_round1_interpretation":
        raise CageV2AnalysisError("results receipt status mismatch")
    if receipt.get("claim_eligible") is not False:
        raise CageV2AnalysisError("round1 receipt must remain claim-ineligible")
    freeze = receipt.get("analysis_freeze", {})
    if freeze.get("anchor_indices") != [5, 6, 7, 8, 9]:
        raise CageV2AnalysisError("analysis anchor grid mismatch")
    if freeze.get("maximum_distinct_families_advanced") != 3:
        raise CageV2AnalysisError("family advancement limit mismatch")
    if len(freeze.get("family_tracks", [])) != 6:
        raise CageV2AnalysisError("analysis must freeze six budget tracks")
    if freeze.get("bootstrap_role") != "descriptive_only_not_used_for_any_gate":
        raise CageV2AnalysisError("bootstrap role mismatch")


def _type7(values: Sequence[float], probability: float) -> float:
    if not values:
        raise CageV2AnalysisError("quantile requires values")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def paired_statistics(
    candidate: Mapping[Any, float],
    baseline: Mapping[Any, float],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    keys = sorted(candidate)
    if keys != sorted(baseline) or not keys:
        raise CageV2AnalysisError("paired contrasts require the same nonempty key set")
    deltas = [float(candidate[key]) - float(baseline[key]) for key in keys]
    if not all(math.isfinite(value) for value in deltas):
        raise CageV2AnalysisError("paired contrast contains non-finite values")
    rng = random.Random(bootstrap_seed)
    boot = [
        math.fsum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(bootstrap_resamples)
    ]
    boot.sort()
    favor = sum(value < 0 for value in deltas)
    tie = sum(value == 0 for value in deltas)
    return {
        "paired_count": len(deltas),
        "mean_delta": math.fsum(deltas) / len(deltas),
        "median_delta": statistics.median(deltas),
        "population_standard_deviation": statistics.pstdev(deltas),
        "minimum_delta": min(deltas),
        "maximum_delta": max(deltas),
        "favor_count": favor,
        "tie_count": tie,
        "oppose_count": len(deltas) - favor - tie,
        "bootstrap_95_interval": {
            "low": _type7(boot, 0.025),
            "high": _type7(boot, 0.975),
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "role": "descriptive_only_not_used_for_gate",
        },
        "paired_deltas": [
            {"key": str(key), "delta": delta} for key, delta in zip(keys, deltas)
        ],
    }


def _method_index(records: Sequence[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    index: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        method_id = record["method"]["id"]
        anchor = record["input"]["anchor_index"]
        if anchor in index[method_id]:
            raise CageV2AnalysisError(f"duplicate method/anchor record: {method_id}/{anchor}")
        index[method_id][anchor] = record
    return dict(index)


def _values(index: dict[str, dict[int, dict[str, Any]]], method_id: str) -> dict[int, float]:
    records = index.get(method_id, {})
    if sorted(records) != [5, 6, 7, 8, 9]:
        raise CageV2AnalysisError(f"method lacks frozen five-anchor grid: {method_id}")
    return {
        anchor: float(record["aggregates"]["joint_post_o_proj_mse"]["mean"])
        for anchor, record in records.items()
    }


def _bytes(index: dict[str, dict[int, dict[str, Any]]], method_id: str) -> int:
    values = {
        int(record["memory"]["model_total_bytes"])
        for record in index.get(method_id, {}).values()
    }
    if len(values) != 1:
        raise CageV2AnalysisError(f"method packed bytes are not constant: {method_id}")
    return values.pop()


def build_round1_analysis(
    *,
    receipt: dict[str, Any],
    records: Sequence[dict[str, Any]],
    receipt_sha256: str,
) -> dict[str, Any]:
    freeze = receipt["analysis_freeze"]
    index = _method_index(records)
    lengths = freeze["prompt_lengths"]
    bootstrap_resamples = freeze["bootstrap_resamples"]
    base_seed = freeze["bootstrap_seed"]
    if bootstrap_resamples <= 0:
        raise CageV2AnalysisError("bootstrap resamples must be positive")
    for target in freeze["kitty_mapping"]:
        for length in lengths:
            kitty_id = freeze["kitty_mapping"][target][str(length)]
            target_bytes = _bytes(index, kitty_id)
            controls = [
                method_id
                for method_id in index
                if method_id.startswith("cage-v1-")
                and next(iter(index[method_id].values()))["method"]["prompt_length"] == length
            ]
            if not controls:
                raise CageV2AnalysisError(f"no CAGE-v1 control at length {length}")
            selected = min(
                controls,
                key=lambda method_id: (abs(_bytes(index, method_id) - target_bytes), method_id),
            )
            if freeze["cage_v1_mapping"][target][str(length)] != selected:
                raise CageV2AnalysisError(f"frozen CAGE-v1 mapping is not byte-nearest: {target}/{length}")
    track_reports = []
    for order, track in enumerate(freeze["family_tracks"]):
        target = track["target"]
        length_reports = []
        overall_candidate = {}
        overall_kitty = {}
        overall_v1 = {}
        total_bytes = 0
        for length_order, length in enumerate(lengths):
            candidate_id = f"{track['method_prefix']}{length}"
            kitty_id = freeze["kitty_mapping"][target][str(length)]
            v1_id = freeze["cage_v1_mapping"][target][str(length)]
            candidate = _values(index, candidate_id)
            kitty = _values(index, kitty_id)
            v1 = _values(index, v1_id)
            candidate_bytes = _bytes(index, candidate_id)
            target_bytes = _bytes(index, kitty_id)
            total_bytes += candidate_bytes
            for anchor in freeze["anchor_indices"]:
                key = (length, anchor)
                overall_candidate[key] = candidate[anchor]
                overall_kitty[key] = kitty[anchor]
                overall_v1[key] = v1[anchor]
            versus_v1 = paired_statistics(
                candidate,
                v1,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=base_seed + order * 100 + length_order * 10 + 1,
            )
            versus_kitty = paired_statistics(
                candidate,
                kitty,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=base_seed + order * 100 + length_order * 10 + 2,
            )
            length_reports.append(
                {
                    "prompt_length": length,
                    "candidate_method_id": candidate_id,
                    "cage_v1_method_id": v1_id,
                    "kitty_method_id": kitty_id,
                    "candidate_model_total_bytes": candidate_bytes,
                    "target_model_total_bytes": target_bytes,
                    "memory_pass": candidate_bytes <= target_bytes,
                    "versus_cage_v1": versus_v1,
                    "versus_kitty": versus_kitty,
                    "cage_v1_pass": versus_v1["mean_delta"] < 0,
                    "kitty_no_worse": versus_kitty["mean_delta"] <= 0,
                }
            )
        kitty_count = sum(row["kitty_no_worse"] for row in length_reports)
        overall_kitty_stats = paired_statistics(
            overall_candidate,
            overall_kitty,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=base_seed + order * 100 + 91,
        )
        overall_v1_stats = paired_statistics(
            overall_candidate,
            overall_v1,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=base_seed + order * 100 + 92,
        )
        gates = {
            "memory_all_three": all(row["memory_pass"] for row in length_reports),
            "beats_cage_v1_all_three": all(row["cage_v1_pass"] for row in length_reports),
            "no_worse_than_kitty_at_least_two": kitty_count >= 2,
            "kitty_no_worse_length_count": kitty_count,
        }
        track_reports.append(
            {
                "family_id": track["family_id"],
                "target": target,
                "lengths": length_reports,
                "total_model_total_bytes_across_lengths": total_bytes,
                "overall_15_case_versus_kitty": overall_kitty_stats,
                "overall_15_case_versus_cage_v1": overall_v1_stats,
                "gates": gates,
                "track_pass": all(value for key, value in gates.items() if key != "kitty_no_worse_length_count"),
            }
        )

    family_reports = []
    for family_id in sorted({track["family_id"] for track in track_reports}):
        tracks = [track for track in track_reports if track["family_id"] == family_id]
        eligible = [track for track in tracks if track["track_pass"]]
        selected = None
        if eligible:
            selected = min(
                eligible,
                key=lambda track: (
                    track["overall_15_case_versus_kitty"]["mean_delta"],
                    track["total_model_total_bytes_across_lengths"],
                    track["target"],
                ),
            )
        family_reports.append(
            {
                "family_id": family_id,
                "eligible_track_count": len(eligible),
                "eligible_targets": [track["target"] for track in eligible],
                "selected_target": None if selected is None else selected["target"],
                "selected_overall_kitty_mean_delta": None
                if selected is None
                else selected["overall_15_case_versus_kitty"]["mean_delta"],
                "selected_total_model_total_bytes": None
                if selected is None
                else selected["total_model_total_bytes_across_lengths"],
                "family_eligible": selected is not None,
            }
        )
    ranked = sorted(
        (family for family in family_reports if family["family_eligible"]),
        key=lambda family: (
            family["selected_overall_kitty_mean_delta"],
            family["selected_total_model_total_bytes"],
            family["family_id"],
        ),
    )
    advanced = ranked[: freeze["maximum_distinct_families_advanced"]]
    return {
        "schema_version": 1,
        "analysis_id": "qwen3-8b-cage-v2-round1-development-analysis-v1",
        "status": "pass",
        "claim_eligible": False,
        "receipt_sha256": receipt_sha256,
        "metric": freeze["primary_metric_path"],
        "track_reports": track_reports,
        "family_reports": family_reports,
        "advanced_families": [
            {"rank": rank, **family} for rank, family in enumerate(advanced, start=1)
        ],
        "advanced_family_count": len(advanced),
        "interpretation_boundary": freeze["interpretation_scope"],
    }


__all__ = [
    "CageV2AnalysisError",
    "EXPECTED_RECEIPT_SHA256",
    "build_round1_analysis",
    "paired_statistics",
    "validate_results_receipt",
]
