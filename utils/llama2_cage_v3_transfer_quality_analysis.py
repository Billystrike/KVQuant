from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.llama2_cage_v3_transfer_quality_acceptance import SCIENTIFIC_FIELDS, load_server_input_manifest
from utils.llama2_cage_v3_transfer_quality_full import expand_full_cases, load_design, load_full_gate_receipt, validate_full_case
from utils.llama2_cage_v3_transfer_quality_full_postrun import shell_case_manifest_sha256
from utils.llama2_cage_v3_transfer_quality_protocol import load_transfer_quality_protocol
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


EXPECTED_RECEIPT_SHA256 = "5bcdccd46565268fc91f98bc7e4351045719618b624787e39ffad1c89fa9905f"
CANDIDATE = "cage_v3"
PREDECESSOR = "cage_v1"
UNIFORM_BASELINE = "kivi"
QUALITY_REFERENCE = "fp16"
METHODS = (QUALITY_REFERENCE, CANDIDATE, PREDECESSOR, UNIFORM_BASELINE)
LENGTHS = (1024, 2048, 4032)
PRIMARY_BASELINES = (PREDECESSOR, UNIFORM_BASELINE)


class Llama2CageV3TransferQualityAnalysisError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3TransferQualityAnalysisError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3TransferQualityAnalysisError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _mean(values: Sequence[float]) -> float:
    converted = [float(value) for value in values]
    _require(bool(converted) and all(math.isfinite(value) for value in converted), "invalid analysis values")
    return math.fsum(converted) / len(converted)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered) and 0.0 <= probability <= 1.0, "invalid percentile request")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_draws(unit_count: int, *, resamples: int = 10_000, seed: int = 20_260_817) -> list[list[int]]:
    _require(unit_count == 50 and resamples == 10_000 and seed == 20_260_817, "bootstrap contract changed")
    generator = random.Random(seed)
    return [[generator.randrange(unit_count) for _ in range(unit_count)] for _ in range(resamples)]


def validate_results_receipt(receipt: Mapping[str, Any], *, receipt_path: Path) -> None:
    _require(file_sha256(receipt_path) == EXPECTED_RECEIPT_SHA256, "results receipt hash mismatch")
    frozen = _load(receipt_path)
    _require(receipt == frozen, "results receipt differs from frozen file")
    _require(receipt.get("schema_version") == 1, "results receipt schema changed")
    _require(receipt.get("receipt_id") == "llama2-7b-cage-v3-transfer-quality-results-receipt-v1", "results receipt identity changed")
    _require(receipt.get("status") == "frozen_after_full_postrun_before_transfer_quality_interpretation", "results receipt status changed")
    _require(receipt.get("claim_eligible") is False and receipt.get("interpretation_performed") is False, "receipt interpretation boundary changed")
    postrun = receipt.get("postrun_audit", {})
    _require(
        postrun.get("sha256") == "1003d4d50d10f1695970db762dea8e115e648d24c2c5322640f11d2ac7864590"
        and postrun.get("size_bytes") == 2678
        and postrun.get("execution_log_sha256") == "1d683db848bc4b312670540b05fd52755e0bc9a6853f84147b9fd0c9388578b5"
        and postrun.get("execution_log_size_bytes") == 8527,
        "postrun receipt linkage changed",
    )
    results = receipt.get("frozen_results", {})
    _require(
        results.get("case_count") == 600
        and results.get("target_token_count") == 38_400
        and results.get("failure_count") == 0
        and results.get("scientific_payload_sha256") == "e5058f62b3a70f539e540576a9b5eff37a360ad96fb0a0d63685dd5c2f98003b",
        "frozen result totals changed",
    )
    _require(
        receipt.get("analysis_freeze")
        == {
            "candidate": CANDIDATE,
            "primary_predecessor": PREDECESSOR,
            "primary_uniform_baseline": UNIFORM_BASELINE,
            "quality_reference": QUALITY_REFERENCE,
            "prompt_lengths": list(LENGTHS),
            "anchors_per_method_length": 50,
            "statistical_unit": "paired_anchor",
            "overall_aggregation": "within_each_anchor_equal_weight_mean_over_three_prompt_lengths",
            "lengths_clustered_within_anchor_for_overall_bootstrap": True,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 20_260_817,
            "bootstrap_rng": "python_random_Random_reinitialized_once",
            "bootstrap_interval": "two_sided_95_percentile_type7_linear",
            "bootstrap_draws_reused_for_every_comparison": True,
            "relative_ppl_percent": "100 * (exp(candidate_mean_nll_minus_baseline_mean_nll) - 1)",
            "per_length_noninferiority_relative_ppl_percent_at_most": 0.5,
            "overall_material_superiority_relative_ppl_percent_at_most": -1.0,
            "overall_ci95_upper_bound_must_be_below_zero": True,
            "both_primary_comparisons_must_pass": True,
            "decimal_direction_alone_is_never_a_pass": True,
            "report_all_methods_lengths_comparisons_and_unfavorable_results": True,
            "memory_representation": "complete active logical packed paper estimate only",
            "canonical_full_corpus_perplexity_claim": False,
        },
        "analysis freeze changed",
    )
    _require(
        receipt.get("authorization")
        == {
            "frozen_transfer_quality_interpretation": True,
            "transfer_promotion_decision": True,
            "candidate_tuning": False,
            "paper_main_method_change": False,
            "paper_claims": False,
            "runtime_claims": False,
            "kitty_llama_port": False,
        },
        "analysis authorization changed",
    )


def load_frozen_records(receipt: Mapping[str, Any], *, repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = receipt["frozen_results"]
    root = Path(spec["output_dir"])
    _require(file_sha256(root / "run_identity.json") == spec["run_identity_sha256"], "run identity changed")
    _require(file_sha256(root / "summary.json") == spec["summary_sha256"], "summary changed")
    _require(shell_case_manifest_sha256(root) == spec["case_file_hash_manifest_sha256"], "case manifest changed")
    design, design_sha = load_design(repo_root / "configs/llama2_7b_cage_v3_transfer_quality_full_design_v1.json", repo_root=repo_root)
    _, gate_sha = load_full_gate_receipt(repo_root / "configs/llama2_7b_cage_v3_transfer_quality_full_execution_gate_receipt_v1.json", repo_root=repo_root)
    protocol, protocol_sha = load_transfer_quality_protocol(repo_root / design["protocol"]["path"], repo_root=repo_root)
    input_manifest = load_server_input_manifest({"input_manifest": design["input_manifest"]}, protocol=protocol)
    expected_cases = expand_full_cases(design=design, protocol=protocol, input_manifest=input_manifest, gate_receipt_sha256=gate_sha)
    records = []
    for case in expected_cases:
        record = _load(root / "cases" / f"{case['case_id']}.json")
        validate_full_case(record, case)
        records.append(record)
    payload = [{field: record[field] for field in SCIENTIFIC_FIELDS} for record in records]
    _require(canonical_sha256(payload) == spec["scientific_payload_sha256"], "scientific payload changed")
    _require(design_sha == spec["design_sha256"] and gate_sha == spec["gate_receipt_sha256"], "design or gate linkage changed")
    _require(protocol_sha == spec["protocol_sha256"], "protocol linkage changed")
    return records, protocol


def _protocol_bytes(protocol: Mapping[str, Any]) -> dict[tuple[str, int], int]:
    result = {}
    for row in protocol["method_length_matrix"]:
        length = int(row["prompt_length"])
        for method in row["methods"]:
            family = method["method"]
            result[(family, length)] = {
                "fp16": length * 32 * 32 * 128 * 2 * 2,
                "cage_v3": row["packed_memory"]["candidate_bytes"],
                "cage_v1": row["packed_memory"]["cage_v1_bytes"],
                "kivi": row["packed_memory"]["kivi_bytes"],
            }[family]
    _require(set(result) == {(method, length) for method in METHODS for length in LENGTHS}, "protocol memory grid changed")
    return result


def _index_records(records: Sequence[dict[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    nll: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    bytes_seen: dict[tuple[str, int], set[int]] = defaultdict(set)
    seen = set()
    for record in records:
        case_id = record["case_id"]
        _require(case_id not in seen, "duplicate case record")
        seen.add(case_id)
        method = record["method"]["method"]
        length = int(record["input"]["identity"]["prompt_length"])
        anchor = int(record["input"]["identity"]["anchor_index"])
        _require(method in METHODS and length in LENGTHS and 0 <= anchor < 50, "unexpected method/length/anchor")
        _require(anchor not in nll[method][length], "duplicate paired cell")
        value = float(record["scoring"]["mean_nll"])
        _require(math.isfinite(value), "non-finite case mean NLL")
        nll[method][length][anchor] = value
        bytes_seen[(method, length)].add(int(record["memory"]["logical_packed_bytes"]))
    _require(len(seen) == 600, "analysis requires exactly 600 unique records")
    frozen_bytes = _protocol_bytes(protocol)
    for method in METHODS:
        for length in LENGTHS:
            _require(set(nll[method][length]) == set(range(50)), f"{method}/{length} anchor grid changed")
            _require(bytes_seen[(method, length)] == {frozen_bytes[(method, length)]}, f"{method}/{length} packed bytes changed")
    return {"nll": nll, "bytes": frozen_bytes}


def _overall_by_anchor(index: Mapping[str, Any], method: str) -> dict[int, float]:
    return {anchor: _mean([index["nll"][method][length][anchor] for length in LENGTHS]) for anchor in range(50)}


def paired_anchor_comparison(candidate: Mapping[int, float], baseline: Mapping[int, float], *, draws: Sequence[Sequence[int]]) -> dict[str, Any]:
    anchors = sorted(candidate)
    _require(anchors == list(range(50)) and anchors == sorted(baseline), "paired anchor grid changed")
    deltas = [float(candidate[anchor]) - float(baseline[anchor]) for anchor in anchors]
    mean_delta = _mean(deltas)
    bootstrap = [_mean([deltas[index] for index in draw]) for draw in draws]
    lower = _percentile(bootstrap, 0.025)
    upper = _percentile(bootstrap, 0.975)
    favor = sum(delta < 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    relative = 100.0 * math.expm1(mean_delta)
    magnitude = abs(relative)
    practical = "negligible" if magnitude < 0.5 else "small" if magnitude < 1.0 else "material"
    return {
        "paired_anchor_count": 50,
        "mean_nll_delta": mean_delta,
        "median_anchor_nll_delta": statistics.median(deltas),
        "minimum_anchor_nll_delta": min(deltas),
        "maximum_anchor_nll_delta": max(deltas),
        "candidate_favor_count": favor,
        "tie_count": ties,
        "baseline_favor_count": 50 - favor - ties,
        "relative_ppl_percent": relative,
        "practical_effect_magnitude": practical,
        "bootstrap_resamples": len(draws),
        "bootstrap_mean_nll_delta_ci95": [lower, upper],
        "bootstrap_favorable": upper < 0.0,
        "bootstrap_unfavorable": lower > 0.0,
        "negative_delta_favors_candidate": True,
    }


def build_analysis(*, receipt_sha256: str, protocol: Mapping[str, Any], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    index = _index_records(records, protocol)
    draws = bootstrap_draws(50)
    summaries = []
    for method in METHODS:
        for length in LENGTHS:
            values = [index["nll"][method][length][anchor] for anchor in range(50)]
            mean_nll = _mean(values)
            summaries.append({
                "method": method,
                "scope": "prompt_length",
                "prompt_length": length,
                "case_count": 50,
                "target_token_count": 3_200,
                "logical_packed_bytes": index["bytes"][(method, length)],
                "mean_nll": mean_nll,
                "perplexity": math.exp(mean_nll),
            })
        overall = list(_overall_by_anchor(index, method).values())
        overall_mean = _mean(overall)
        summaries.append({
            "method": method,
            "scope": "overall_equal_length_weight",
            "prompt_length": "overall",
            "case_count": 150,
            "target_token_count": 9_600,
            "logical_packed_bytes": None,
            "mean_nll": overall_mean,
            "perplexity": math.exp(overall_mean),
        })
    comparisons = []
    lookup = {}
    for baseline in (QUALITY_REFERENCE, PREDECESSOR, UNIFORM_BASELINE):
        by_length = {}
        for length in LENGTHS:
            report = paired_anchor_comparison(index["nll"][CANDIDATE][length], index["nll"][baseline][length], draws=draws)
            report["noninferiority_pass"] = report["relative_ppl_percent"] <= 0.5
            comparisons.append({"candidate": CANDIDATE, "baseline": baseline, "scope": "prompt_length", "prompt_length": length, **report})
            by_length[str(length)] = report
        overall = paired_anchor_comparison(_overall_by_anchor(index, CANDIDATE), _overall_by_anchor(index, baseline), draws=draws)
        overall["material_superiority_pass"] = overall["relative_ppl_percent"] <= -1.0
        overall["uncertainty_pass"] = overall["bootstrap_mean_nll_delta_ci95"][1] < 0.0
        comparisons.append({"candidate": CANDIDATE, "baseline": baseline, "scope": "overall_equal_length_weight", "prompt_length": "overall", **overall})
        lookup[baseline] = {"per_length": by_length, "overall": overall}
    gates = {}
    for baseline in PRIMARY_BASELINES:
        all_lengths = all(lookup[baseline]["per_length"][str(length)]["noninferiority_pass"] for length in LENGTHS)
        overall = lookup[baseline]["overall"]
        passed = all_lengths and overall["material_superiority_pass"] and overall["uncertainty_pass"]
        gates[baseline] = {
            "all_lengths_noninferiority_pass": all_lengths,
            "overall_material_superiority_pass": overall["material_superiority_pass"],
            "overall_uncertainty_pass": overall["uncertainty_pass"],
            "pass": passed,
        }
    promotion_pass = all(gates[baseline]["pass"] for baseline in PRIMARY_BASELINES)
    memory = []
    for length in LENGTHS:
        candidate_bytes = index["bytes"][(CANDIDATE, length)]
        for baseline in (PREDECESSOR, UNIFORM_BASELINE):
            baseline_bytes = index["bytes"][(baseline, length)]
            memory.append({
                "candidate": CANDIDATE,
                "baseline": baseline,
                "prompt_length": length,
                "candidate_bytes": candidate_bytes,
                "baseline_bytes": baseline_bytes,
                "candidate_relative_bytes_percent": 100.0 * (candidate_bytes / baseline_bytes - 1.0),
            })
    return {
        "schema_version": 1,
        "analysis_id": "llama2-7b-cage-v3-transfer-quality-analysis-v1",
        "status": "pass",
        "claim_eligible": False,
        "interpretation_performed": True,
        "receipt_sha256": receipt_sha256,
        "case_count": 600,
        "target_token_count": 38_400,
        "method_length_summaries": summaries,
        "paired_comparisons": comparisons,
        "comparison_lookup": lookup,
        "memory_comparisons": memory,
        "primary_transfer_gates": {**gates, "both_primary_comparisons_pass": promotion_pass},
        "promotion_pass": promotion_pass,
        "candidate_outcome": "promote_cage_v3_as_llama2_main_method" if promotion_pass else "do_not_promote_cage_v3_under_frozen_transfer_gate",
        "bootstrap": {
            "unit": "paired_anchor_with_lengths_clustered_for_overall",
            "resamples": 10_000,
            "seed": 20_260_817,
            "interval": "two_sided_95_percentile_type7_linear",
            "draws_reused_for_every_comparison": True,
        },
        "reporting": {
            "all_methods_lengths_comparisons_and_unfavorable_results_retained": True,
            "canonical_full_corpus_perplexity_claim": False,
            "memory_representation": "complete active logical packed paper estimate only",
            "kitty_llama_included": False,
            "runtime_claims_authorized": False,
        },
        "next_authorization": {
            "paper_main_method_change": False,
            "paper_claims": False,
            "candidate_tuning": False,
            "kitty_llama_port": False,
            "runtime_claims": False,
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def write_analysis_outputs(analysis: Mapping[str, Any], destination: Path) -> dict[str, dict[str, Any]]:
    _require(not destination.exists(), f"analysis destination already exists: {destination}")
    destination.mkdir(parents=True)
    analysis_path = destination / "llama2_cage_v3_transfer_quality_analysis_v1.json"
    summary_path = destination / "llama2_cage_v3_method_length_summary_v1.csv"
    paired_path = destination / "llama2_cage_v3_paired_comparisons_v1.csv"
    decision_path = destination / "llama2_cage_v3_transfer_decision_v1.md"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(summary_path, analysis["method_length_summaries"])
    _write_csv(paired_path, analysis["paired_comparisons"])
    lines = [
        "# Llama-2 CAGE-v3 Cross-Architecture Transfer Decision",
        "",
        f"- Frozen promotion decision: `{analysis['candidate_outcome']}`",
        f"- Both primary comparison gates pass: `{analysis['promotion_pass']}`",
        "- Negative paired NLL and relative-PPL deltas favor CAGE-v3.",
        "- This is a 50-anchor paired transfer study, not canonical full-corpus perplexity.",
        "- Memory is a complete active logical packed paper estimate; no runtime claim is authorized.",
        "- Kitty is not included in this Llama-2 protocol.",
        "",
        "## Frozen primary comparisons",
        "",
    ]
    for baseline in PRIMARY_BASELINES:
        overall = analysis["comparison_lookup"][baseline]["overall"]
        gate = analysis["primary_transfer_gates"][baseline]
        lines.append(
            f"- CAGE-v3 vs {baseline}: overall relative PPL {overall['relative_ppl_percent']:.6f}%; "
            f"95% NLL CI [{overall['bootstrap_mean_nll_delta_ci95'][0]:.9g}, "
            f"{overall['bootstrap_mean_nll_delta_ci95'][1]:.9g}]; gate `{gate['pass']}`."
        )
    lines.extend(["", "All lengths and unfavorable outcomes are retained in the JSON and CSV outputs.", ""])
    decision_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        path.name: {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
        for path in (analysis_path, summary_path, paired_path, decision_path)
    }


__all__ = [
    "CANDIDATE",
    "EXPECTED_RECEIPT_SHA256",
    "LENGTHS",
    "Llama2CageV3TransferQualityAnalysisError",
    "METHODS",
    "PREDECESSOR",
    "PRIMARY_BASELINES",
    "QUALITY_REFERENCE",
    "UNIFORM_BASELINE",
    "bootstrap_draws",
    "build_analysis",
    "load_frozen_records",
    "paired_anchor_comparison",
    "validate_results_receipt",
    "write_analysis_outputs",
]
