from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


MANIFEST_ID = "qwen3-8b-cage-v3-promotion-analysis-artifacts-v1"
AUDIT_ID = "qwen3-8b-cage-v3-promotion-negative-closeout-v1"
MANIFEST_CANONICAL_SHA256 = "fd2ea0236cf7c38d4b7cb241284e5359710e2cc0edc3e12e2c95fe5f4bc4277d"
BASELINES = (
    "cage-v1-kittypro-matched",
    "kivi-kittypro-matched",
    "kitty-pro-25pct",
)


class CageV3PromotionCloseoutError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV3PromotionCloseoutError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV3PromotionCloseoutError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_manifest(manifest: Mapping[str, Any], *, verify_artifacts: bool) -> None:
    _require(canonical_sha256(manifest) == MANIFEST_CANONICAL_SHA256, "promotion closeout manifest frozen payload mismatch")
    _require(manifest.get("schema_version") == 1 and manifest.get("manifest_id") == MANIFEST_ID, "promotion closeout manifest identity mismatch")
    _require(manifest.get("status") == "declared_from_completed_frozen_analysis_before_negative_closeout", "promotion closeout manifest status mismatch")
    _require(manifest.get("claim_eligible") is False, "promotion closeout claim boundary changed")
    _require(manifest.get("analysis_source_commit") == "14b77677c78effee20b100a3910ed055839374c6", "promotion analysis commit changed")
    _require(
        manifest.get("frozen_decision")
        == {
            "all_three_comparison_gates_pass": False,
            "candidate_outcome": "retain_original_cage_and_close_cage_v3_promotion",
            "promotion_pass": False,
            "pg19_test_accessed": False,
            "additional_v3_tuning_performed": False,
        },
        "promotion frozen decision changed",
    )
    _require(tuple(manifest.get("frozen_comparisons", {})) == BASELINES, "promotion closeout baselines changed")
    _require(
        manifest.get("frozen_memory")
        == {
            "candidate_total_bytes": 225_967_104,
            "predecessor_total_bytes": 230_812_416,
            "candidate_memory_reduction_vs_predecessor": 0.02099242356182429,
            "candidate_not_exceed_kitty_all_lengths": True,
        },
        "promotion frozen memory changed",
    )
    _require(
        manifest.get("closeout_boundary")
        == {
            "additional_v3_tuning": False,
            "architecture_normalized_v3_design_freeze": False,
            "staged_llama2_v3_acceptance": False,
            "llama2_v3_full_experiments": False,
            "kitty_llama_port": False,
            "pg19_test_access": False,
            "paper_superiority_claims": False,
            "runtime_claims": False,
        },
        "promotion closeout boundary changed",
    )
    if verify_artifacts:
        specs = [manifest["analysis_manifest"], *manifest["outputs"].values(), manifest["execution_log"]]
        for spec in specs:
            path = Path(spec["path"])
            _require(path.is_file(), f"promotion closeout artifact missing: {path}")
            _require(file_sha256(path) == spec["sha256"], f"promotion closeout artifact hash mismatch: {path}")
            if "size_bytes" in spec:
                _require(path.stat().st_size == spec["size_bytes"], f"promotion closeout artifact size mismatch: {path}")


def _comparison_by_baseline(analysis: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    gates = analysis["promotion_gates"]
    gate_names = {
        BASELINES[0]: "versus_predecessor",
        BASELINES[1]: "versus_uniform_baseline",
        BASELINES[2]: "versus_external_baseline",
    }
    result = {}
    for baseline in BASELINES:
        source = analysis["comparison_lookup"][baseline]
        gate = gates[gate_names[baseline]]
        row = {
            "overall_relative_ppl_percent": source["overall"]["relative_ppl_percent"],
            "overall_mean_nll_delta": source["overall"]["mean_nll_delta"],
            "overall_ci95": source["overall"]["bootstrap_mean_nll_delta_ci95"],
            "per_length_relative_ppl_percent": {
                length: source["per_length"][length]["relative_ppl_percent"]
                for length in ("1024", "2048", "4032")
            },
            "all_lengths_noninferiority_pass": gate["all_lengths_noninferiority_pass"],
        }
        if baseline == BASELINES[0]:
            row["quality_superiority_track_pass"] = gate["quality_superiority_track_pass"]
            row["memory_quality_pareto_track_pass"] = gate["memory_quality_pareto_track_pass"]
        if baseline == BASELINES[2]:
            row["memory_pass"] = gate["memory_pass"]
        row["baseline_pass"] = gate["pass"]
        result[baseline] = row
    return result


def build_closeout(manifest_path: str | Path) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = _load(source)
    validate_manifest(manifest, verify_artifacts=True)
    analysis = _load(Path(manifest["outputs"]["analysis"]["path"]))
    analysis_manifest = _load(Path(manifest["analysis_manifest"]["path"]))
    _require(analysis.get("status") == "pass" and analysis.get("claim_eligible") is False, "promotion analysis status mismatch")
    _require(analysis.get("promotion_pass") is False, "promotion analysis unexpectedly passed")
    _require(analysis.get("candidate_outcome") == "retain_original_cage_and_close_cage_v3_promotion", "promotion analysis decision mismatch")
    _require(analysis_manifest.get("status") == "pass" and analysis_manifest.get("promotion_pass") is False, "promotion analysis manifest decision mismatch")
    comparisons = _comparison_by_baseline(analysis)
    for baseline, observed in comparisons.items():
        expected = manifest["frozen_comparisons"][baseline]
        _require(observed.keys() == expected.keys(), f"promotion comparison fields changed: {baseline}")
        for key, value in observed.items():
            expected_value = expected[key]
            if isinstance(value, float):
                _require(math.isclose(value, expected_value, rel_tol=0.0, abs_tol=0.0), f"promotion comparison changed: {baseline}.{key}")
            else:
                _require(value == expected_value, f"promotion comparison changed: {baseline}.{key}")
    memory = analysis["memory"]
    frozen_memory = manifest["frozen_memory"]
    for key, expected in frozen_memory.items():
        observed = memory[key]
        if isinstance(observed, float):
            _require(math.isclose(observed, expected, rel_tol=0.0, abs_tol=0.0), f"promotion memory changed: {key}")
        else:
            _require(observed == expected, f"promotion memory changed: {key}")
    log = Path(manifest["execution_log"]["path"]).read_text(encoding="utf-8", errors="replace")
    log_checks = {
        "nonempty": bool(log.strip()),
        "no_traceback": "Traceback (most recent call last)" not in log,
        "no_explicit_error": "ERROR:" not in log,
        "direct_audit_present": "=== DIRECT PROMOTION DECISION AUDIT ===" in log,
        "analysis_end_present": "=== CAGE-V3 PROMOTION FROZEN ANALYSIS END ===" in log,
        "result_pass_present": "CAGE_V3_PROMOTION_ANALYSIS_RESULT=PASS" in log,
    }
    _require(all(log_checks.values()), "promotion analysis log checks failed")
    return {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "status": "pass",
        "claim_eligible": False,
        "analysis_artifact_manifest_path": str(source),
        "analysis_artifact_manifest_sha256": file_sha256(source),
        "analysis_manifest_sha256": manifest["analysis_manifest"]["sha256"],
        "analysis_sha256": manifest["outputs"]["analysis"]["sha256"],
        "analysis_log_sha256": manifest["execution_log"]["sha256"],
        "analysis_log_checks": log_checks,
        "case_count": analysis["case_count"],
        "document_count": analysis["document_count"],
        "target_token_count": analysis["target_token_count"],
        "comparisons": comparisons,
        "memory": frozen_memory,
        "decision": manifest["frozen_decision"],
        "closeout_boundary": manifest["closeout_boundary"],
        "scientific_conclusion": (
            "CAGE-v3 materially improves over original CAGE and matched KIVI while reducing "
            "packed memory versus original CAGE, but fails the preregistered Kitty-Pro gate "
            "because 4032-token relative PPL exceeds the per-length noninferiority limit and "
            "the overall uncertainty interval does not establish noninferiority"
        ),
        "paper_method_status": {
            "retained_main_method": "original_cage",
            "cage_v3_role": "closed_development_candidate_with_reportable_negative_result",
            "llama2_v3_rerun": False,
        },
    }


__all__ = [
    "AUDIT_ID",
    "BASELINES",
    "CageV3PromotionCloseoutError",
    "MANIFEST_ID",
    "build_closeout",
    "validate_manifest",
]
