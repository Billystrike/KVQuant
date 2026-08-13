from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v4_data import file_sha256
from utils.qwen3_cage_v4_dtqi_acceptance import load_json


MANIFEST_ID = "qwen3-8b-cage-v4-dtqi-screen-analysis-artifacts-v1"
AUDIT_ID = "qwen3-8b-cage-v4-dtqi-negative-closeout-v1"


class CageV4DTQICloseoutError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV4DTQICloseoutError(message)


def validate_manifest(manifest: Mapping[str, Any], *, verify_artifacts: bool) -> None:
    _require(manifest.get("schema_version") == 1, "closeout manifest schema mismatch")
    _require(manifest.get("manifest_id") == MANIFEST_ID, "closeout manifest identity mismatch")
    _require(manifest.get("status") == "declared_from_completed_frozen_analysis_before_closeout_audit", "closeout manifest status mismatch")
    _require(manifest.get("claim_eligible") is False, "closeout claim boundary changed")
    _require(manifest.get("analysis_source_commit") == "6eb6752cfb43e40ef88c2fe33db20795f4be1379", "closeout analysis commit changed")
    _require(manifest.get("frozen_decision") == {
        "all_two_baselines_pass": False,
        "candidate_outcome": "close_cage_v4_as_negative",
        "holdout_accessed": False,
        "pg19_test_accessed": False,
    }, "closeout frozen decision changed")
    _require(manifest.get("closeout_boundary") == {
        "additional_tuning_on_same_screen": False,
        "holdout_metrics_authorized": False,
        "pg19_test_access_authorized": False,
        "runtime_claims_authorized": False,
        "paper_superiority_claims_authorized": False,
    }, "closeout boundary changed")
    _require(set(manifest.get("frozen_comparisons", {})) == {"cage-v3-sr2-sink32-calibrated", "kitty-pro-25pct"}, "closeout baselines changed")
    if verify_artifacts:
        specs = [manifest["analysis_manifest"], *manifest["outputs"].values(), manifest["execution_log"]]
        for spec in specs:
            path = Path(spec["path"])
            _require(path.is_file(), f"closeout artifact missing: {path}")
            _require(file_sha256(path) == spec["sha256"], f"closeout artifact hash mismatch: {path}")
            if "size_bytes" in spec:
                _require(path.stat().st_size == spec["size_bytes"], f"closeout artifact size mismatch: {path}")


def _comparison_by_baseline(analysis: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for baseline in ("cage-v3-sr2-sink32-calibrated", "kitty-pro-25pct"):
        rows = [row for row in analysis["paired_comparisons"] if row["baseline_method"] == baseline]
        overall = [row for row in rows if row["prompt_length"] == "overall_equal_length_weight"]
        lengths = [row for row in rows if row["prompt_length"] in (1024, 2048, 4032)]
        _require(len(overall) == 1 and len(lengths) == 3, "closeout comparison grid mismatch")
        overall = overall[0]
        result[baseline] = {
            "overall_relative_ppl_percent": overall["relative_ppl_percent"],
            "overall_mean_nll_delta": overall["mean_nll_delta"],
            "overall_ci95": overall["bootstrap_mean_nll_delta_ci95"],
            "material_superiority_pass": overall["material_superiority_pass"],
            "uncertainty_pass": overall["uncertainty_pass"],
            "per_length_relative_ppl_percent": {str(row["prompt_length"]): row["relative_ppl_percent"] for row in lengths},
            "all_lengths_noninferiority_pass": all(row["noninferiority_pass"] for row in lengths),
            "baseline_pass": overall["material_superiority_pass"] and overall["uncertainty_pass"] and all(row["noninferiority_pass"] for row in lengths),
        }
    return result


def build_closeout(manifest_path: str | Path) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = load_json(source)
    validate_manifest(manifest, verify_artifacts=True)
    analysis = load_json(manifest["outputs"]["analysis"]["path"])
    analysis_manifest = load_json(manifest["analysis_manifest"]["path"])
    _require(analysis.get("status") == "pass" and analysis.get("claim_eligible") is False, "closeout analysis status mismatch")
    _require(analysis_manifest.get("status") == "pass" and analysis_manifest.get("candidate_outcome") == "close_cage_v4_as_negative", "closeout analysis manifest decision mismatch")
    _require(analysis.get("success_decision") == {
        "all_two_baselines_pass": False,
        "candidate_outcome": "close_cage_v4_as_negative",
        "per_baseline": analysis["success_decision"]["per_baseline"],
    }, "closeout analysis decision structure mismatch")
    comparisons = _comparison_by_baseline(analysis)
    frozen = manifest["frozen_comparisons"]
    for baseline, observed in comparisons.items():
        expected = frozen[baseline]
        _require(observed.keys() == expected.keys(), f"closeout comparison fields changed: {baseline}")
        for key, value in observed.items():
            expected_value = expected[key]
            if isinstance(value, float):
                _require(math.isclose(value, expected_value, rel_tol=0.0, abs_tol=0.0), f"closeout comparison changed: {baseline}.{key}")
            else:
                _require(value == expected_value, f"closeout comparison changed: {baseline}.{key}")
    log = Path(manifest["execution_log"]["path"]).read_text(encoding="utf-8", errors="replace")
    log_checks = {
        "nonempty": bool(log.strip()),
        "no_traceback": "Traceback (most recent call last)" not in log,
        "no_explicit_error": "ERROR:" not in log,
        "direct_audit_present": "=== DIRECT ANALYSIS AUDIT ===" in log,
        "analysis_end_present": "=== CAGE-V4-DTQI FROZEN SCREEN ANALYSIS END ===" in log,
    }
    _require(all(log_checks.values()), "closeout analysis log checks failed")
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
        "decision": manifest["frozen_decision"],
        "closeout_boundary": manifest["closeout_boundary"],
        "scientific_conclusion": "DTQI partially changes the length profile but does not meet preregistered material-superiority and uncertainty requirements against both baselines",
    }


__all__ = ["AUDIT_ID", "CageV4DTQICloseoutError", "MANIFEST_ID", "build_closeout", "validate_manifest"]
