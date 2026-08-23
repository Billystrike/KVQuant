from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from utils.qwen3_cage_v4_data import file_sha256


MANIFEST_ID = "llama2-7b-cage-v3-transfer-quality-analysis-artifacts-v1"
AUDIT_ID = "llama2-7b-cage-v3-transfer-quality-closeout-v1"
MANIFEST_SHA256 = "769fc3d632a4767097fcee42aafcfd7148b9c81b56d1ea011006201a2d9aa0f6"


class Llama2CageV3TransferQualityCloseoutError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3TransferQualityCloseoutError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3TransferQualityCloseoutError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _same_float(observed: float, expected: float, label: str) -> None:
    _require(math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=0.0), f"closeout value changed: {label}")


def validate_manifest(manifest: Mapping[str, Any], *, repo_root: Path, verify_artifacts: bool) -> None:
    _require(manifest.get("schema_version") == 1, "closeout manifest schema changed")
    _require(manifest.get("manifest_id") == MANIFEST_ID, "closeout manifest identity changed")
    _require(manifest.get("status") == "declared_from_completed_frozen_analysis_before_transfer_closeout", "closeout manifest status changed")
    _require(manifest.get("claim_eligible") is False, "closeout claim boundary changed")
    _require(manifest.get("analysis_source_commit") == "d95691d9f5944b0b04682a279da7f15358e49285", "analysis source commit changed")
    receipt = manifest.get("results_receipt", {})
    _require(
        receipt == {
            "path": "configs/llama2_7b_cage_v3_transfer_quality_results_receipt_v1.json",
            "sha256": "5bcdccd46565268fc91f98bc7e4351045719618b624787e39ffad1c89fa9905f",
        },
        "results receipt changed",
    )
    _require(file_sha256(repo_root / receipt["path"]) == receipt["sha256"], "checked-in results receipt changed")
    _require(
        manifest.get("frozen_decision")
        == {
            "promotion_pass": False,
            "candidate_outcome": "do_not_promote_cage_v3_under_frozen_transfer_gate",
            "llama2_main_method": "cage_v1",
            "cage_v3_role": "secondary_length_dependent_cross_architecture_refinement",
        },
        "frozen transfer decision changed",
    )
    _require(
        manifest.get("closeout_boundary")
        == {
            "additional_tuning_on_same_anchors": False,
            "relax_frozen_promotion_gate": False,
            "promote_cage_v3_as_universal_main_method": False,
            "retain_cage_v1_as_llama2_main_method": True,
            "report_cage_v3_as_secondary_refinement": True,
            "report_all_lengths_and_unfavorable_results": True,
            "paper_evidence_packaging_authorized_after_closeout": True,
            "direct_paper_claims_authorized": False,
            "runtime_claims_authorized": False,
            "kitty_llama_port_authorized": False,
        },
        "closeout boundary changed",
    )
    _require(set(manifest.get("frozen_comparisons", {})) == {"cage_v1", "kivi", "fp16"}, "frozen comparison set changed")
    if verify_artifacts:
        specs = [manifest["analysis_manifest"], *manifest["outputs"].values(), manifest["execution_log"]]
        for spec in specs:
            path = Path(spec["path"])
            _require(path.is_file(), f"analysis artifact missing: {path}")
            _require(file_sha256(path) == spec["sha256"], f"analysis artifact hash changed: {path}")
            if "size_bytes" in spec:
                _require(path.stat().st_size == spec["size_bytes"], f"analysis artifact size changed: {path}")


def _observed_comparison(analysis: Mapping[str, Any], baseline: str) -> dict[str, Any]:
    source = analysis["comparison_lookup"][baseline]
    overall = source["overall"]
    result = {
        "overall_mean_nll_delta": overall["mean_nll_delta"],
        "overall_relative_ppl_percent": overall["relative_ppl_percent"],
        "overall_ci95": overall["bootstrap_mean_nll_delta_ci95"],
        "per_length_relative_ppl_percent": {
            length: source["per_length"][length]["relative_ppl_percent"]
            for length in ("1024", "2048", "4032")
        },
    }
    if baseline in ("cage_v1", "kivi"):
        gate = analysis["primary_transfer_gates"][baseline]
        result.update({
            "all_lengths_noninferiority_pass": gate["all_lengths_noninferiority_pass"],
            "overall_material_superiority_pass": gate["overall_material_superiority_pass"],
            "overall_uncertainty_pass": gate["overall_uncertainty_pass"],
            "baseline_pass": gate["pass"],
        })
    return result


def _validate_comparison(observed: Mapping[str, Any], expected: Mapping[str, Any], baseline: str) -> None:
    _require(observed.keys() == expected.keys(), f"comparison fields changed: {baseline}")
    for key, value in observed.items():
        expected_value = expected[key]
        if isinstance(value, float):
            _same_float(value, expected_value, f"{baseline}.{key}")
        elif isinstance(value, list):
            _require(len(value) == len(expected_value), f"comparison list changed: {baseline}.{key}")
            for index, item in enumerate(value):
                _same_float(item, expected_value[index], f"{baseline}.{key}[{index}]")
        elif isinstance(value, dict):
            _require(value.keys() == expected_value.keys(), f"comparison map changed: {baseline}.{key}")
            for name, item in value.items():
                _same_float(item, expected_value[name], f"{baseline}.{key}.{name}")
        else:
            _require(value == expected_value, f"comparison value changed: {baseline}.{key}")


def build_closeout(manifest_path: str | Path, *, repo_root: Path) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = _load(source)
    _require(file_sha256(source) == MANIFEST_SHA256, "closeout manifest file hash changed")
    validate_manifest(manifest, repo_root=repo_root, verify_artifacts=True)
    analysis = _load(Path(manifest["outputs"]["analysis"]["path"]))
    analysis_manifest = _load(Path(manifest["analysis_manifest"]["path"]))
    _require(
        analysis.get("status") == "pass"
        and analysis.get("claim_eligible") is False
        and analysis.get("interpretation_performed") is True
        and analysis.get("case_count") == 600
        and analysis.get("target_token_count") == 38_400,
        "analysis status or totals changed",
    )
    _require(
        analysis.get("promotion_pass") is False
        and analysis.get("candidate_outcome") == "do_not_promote_cage_v3_under_frozen_transfer_gate"
        and analysis.get("primary_transfer_gates", {}).get("both_primary_comparisons_pass") is False,
        "analysis promotion decision changed",
    )
    _require(
        analysis_manifest.get("status") == "pass"
        and analysis_manifest.get("promotion_pass") is False
        and analysis_manifest.get("candidate_outcome") == analysis["candidate_outcome"],
        "analysis manifest decision changed",
    )
    comparisons = {}
    for baseline in ("cage_v1", "kivi", "fp16"):
        comparisons[baseline] = _observed_comparison(analysis, baseline)
        _validate_comparison(comparisons[baseline], manifest["frozen_comparisons"][baseline], baseline)
    memory = {
        "candidate_relative_bytes_percent_vs_cage_v1": {},
        "candidate_relative_bytes_percent_vs_kivi": {},
    }
    for row in analysis["memory_comparisons"]:
        key = f"candidate_relative_bytes_percent_vs_{row['baseline']}"
        memory[key][str(row["prompt_length"])] = row["candidate_relative_bytes_percent"]
    _require(memory.keys() == manifest["frozen_memory"].keys(), "memory comparison set changed")
    for comparison, values in memory.items():
        _require(values.keys() == manifest["frozen_memory"][comparison].keys(), f"memory lengths changed: {comparison}")
        for length, value in values.items():
            _same_float(value, manifest["frozen_memory"][comparison][length], f"{comparison}.{length}")
    log_text = Path(manifest["execution_log"]["path"]).read_text(encoding="utf-8", errors="replace")
    log_checks = {
        "nonempty": bool(log_text.strip()),
        "no_traceback": "Traceback (most recent call last)" not in log_text,
        "no_explicit_error": "ERROR:" not in log_text,
        "direct_analysis_audit": "=== DIRECT ANALYSIS AUDIT AND RESULTS ===" in log_text,
        "analysis_end_marker": "=== LLAMA2 CAGE-V3 FROZEN TRANSFER-QUALITY ANALYSIS END ===" in log_text,
        "analysis_pass": "ANALYSIS_STATUS=0" in log_text,
        "direct_audit_pass": "DIRECT_STATUS=0" in log_text,
    }
    _require(all(log_checks.values()), "analysis execution log checks failed")
    return {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "status": "pass",
        "claim_eligible": False,
        "analysis_artifact_manifest_path": str(source),
        "analysis_artifact_manifest_sha256": MANIFEST_SHA256,
        "analysis_manifest_sha256": manifest["analysis_manifest"]["sha256"],
        "analysis_sha256": manifest["outputs"]["analysis"]["sha256"],
        "analysis_log_sha256": manifest["execution_log"]["sha256"],
        "analysis_log_checks": log_checks,
        "case_count": 600,
        "target_token_count": 38_400,
        "comparisons": comparisons,
        "memory": memory,
        "decision": manifest["frozen_decision"],
        "closeout_boundary": manifest["closeout_boundary"],
        "scientific_conclusion": (
            "CAGE-v3 is not promoted as a universal replacement for CAGE-v1; it is retained as a "
            "secondary length-dependent cross-architecture refinement with material overall and "
            "long-context gains versus KIVI but a frozen 2048-token regression"
        ),
    }


__all__ = [
    "AUDIT_ID",
    "Llama2CageV3TransferQualityCloseoutError",
    "MANIFEST_ID",
    "MANIFEST_SHA256",
    "build_closeout",
    "validate_manifest",
]
