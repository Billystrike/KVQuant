"""Final cross-experiment paper package for the frozen Llama-2-7B studies."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


PACKAGE_SCHEMA_VERSION = 1
MODEL_SCOPE = "Llama-2-7B-hf"
RESIDUALS = (64, 128)
BASELINE_ROLES = ("fixed-random", "uniform", "k-adaptive", "v-adaptive")
LOCAL_LENGTHS = (512, 1024, 2048, 4095)
PPL_LENGTHS = (512, 1024, 2048, 4032)
EXACT_JOIN_LENGTHS = (512, 1024, 2048)
TABLE_NAMES = (
    "operating_point_evidence",
    "comparison_summary",
    "mechanism_length_evidence",
    "mechanism_summary",
    "factorial_evidence",
    "claim_register",
)


class Llama2PaperPackageError(ValueError):
    """Raised when a frozen input cannot support the final package."""


def load_final_paper_inputs(
    evidence_dir: str | Path,
    local_ablation_dir: str | Path,
    ppl_mechanism_dir: str | Path,
) -> dict[str, Any]:
    """Strictly load the three frozen analysis packages."""

    evidence_root = Path(evidence_dir)
    local_root = Path(local_ablation_dir)
    ppl_root = Path(ppl_mechanism_dir)

    evidence_protocol = _read_json(evidence_root / "evidence_protocol.json")
    operating_rows = _read_jsonl(evidence_root / "operating_point_evidence.jsonl")
    comparison_rows = _read_jsonl(evidence_root / "comparison_summary.jsonl")
    if (
        evidence_protocol.get("schema_version") != 1
        or evidence_protocol.get("model_scope") != MODEL_SCOPE
        or evidence_protocol.get("operating_point_row_count") != 8
        or len(operating_rows) != 8
        or len(comparison_rows) != 2
    ):
        raise Llama2PaperPackageError("prior paper evidence differs from the frozen protocol")
    calibration_deviation = evidence_protocol.get("paired_ppl_protocol_deviation")
    if (
        not isinstance(calibration_deviation, dict)
        or calibration_deviation.get("overall_gate") != "FAIL"
        or calibration_deviation.get("protocol_status") != "RECORDED_DEVIATION"
    ):
        raise Llama2PaperPackageError(
            "prior paper evidence does not preserve the paired-PPL calibration deviation"
        )
    if len({
        (row.get("comparison_id"), row.get("pareto_prompt_length"))
        for row in operating_rows
    }) != 8:
        raise Llama2PaperPackageError("prior operating-point evidence has duplicate keys")

    local_protocol = _read_json(local_root / "analysis_protocol.json")
    local_contrasts = _read_jsonl(local_root / "paired_contrasts.jsonl")
    local_factorial = _read_jsonl(local_root / "factorial_decomposition.jsonl")
    if (
        local_protocol.get("schema_version") != 1
        or local_protocol.get("input_run_count") != 144
        or local_protocol.get("paired_contrast_count") != 32
        or local_protocol.get("factorial_row_count") != 8
        or local_protocol.get("primary_error_metric") != "joint_post_o_proj_mse"
        or len(local_contrasts) != 32
        or len(local_factorial) != 8
    ):
        raise Llama2PaperPackageError("local mechanism analysis differs from the frozen protocol")
    expected_local = {
        (residual, role, length)
        for residual in RESIDUALS
        for role in BASELINE_ROLES
        for length in LOCAL_LENGTHS
    }
    actual_local = {
        (row.get("residual_length"), _contrast_role(row), row.get("prompt_length"))
        for row in local_contrasts
    }
    if actual_local != expected_local or len(actual_local) != len(local_contrasts):
        raise Llama2PaperPackageError("local mechanism contrast coverage is invalid")

    ppl_protocol = _read_json(ppl_root / "analysis_protocol.json")
    ppl_pairs = _read_jsonl(ppl_root / "paired_comparisons.jsonl")
    ppl_factorial = _read_jsonl(ppl_root / "factorial_summary.jsonl")
    if (
        ppl_protocol.get("schema_version") != 1
        or ppl_protocol.get("protocol_stage") != "mechanism_ablation_full"
        or ppl_protocol.get("case_count") != 2000
        or ppl_protocol.get("anchor_count") != 50
        or ppl_protocol.get("bootstrap", {}).get("resamples") != 10_000
        or len(ppl_pairs) != 40
        or len(ppl_factorial) != 10
    ):
        raise Llama2PaperPackageError("PPL mechanism analysis differs from the frozen protocol")
    expected_ppl = {
        (residual, role, length)
        for residual in RESIDUALS
        for role in BASELINE_ROLES
        for length in (None, *PPL_LENGTHS)
    }
    actual_ppl = {
        (row.get("residual_length"), row.get("baseline_role"), row.get("prompt_length"))
        for row in ppl_pairs
    }
    if actual_ppl != expected_ppl or len(actual_ppl) != len(ppl_pairs):
        raise Llama2PaperPackageError("PPL mechanism comparison coverage is invalid")

    expected_factorial = {
        (residual, length)
        for residual in RESIDUALS
        for length in (None, *PPL_LENGTHS)
    }
    actual_factorial = {
        (row.get("residual_length"), row.get("prompt_length"))
        for row in ppl_factorial
    }
    if actual_factorial != expected_factorial or len(actual_factorial) != len(ppl_factorial):
        raise Llama2PaperPackageError("PPL factorial coverage is invalid")

    _validate_clean_states(evidence_protocol, local_protocol, ppl_protocol)
    _validate_numeric_rows("operating points", operating_rows)
    _validate_numeric_rows("comparison summary", comparison_rows)
    _validate_numeric_rows("local contrasts", local_contrasts)
    _validate_numeric_rows("local factorial", local_factorial)
    _validate_numeric_rows("PPL comparisons", ppl_pairs)
    _validate_numeric_rows("PPL factorial", ppl_factorial)

    return {
        "evidence_protocol": evidence_protocol,
        "operating_point_evidence": operating_rows,
        "comparison_summary": comparison_rows,
        "local_protocol": local_protocol,
        "local_contrasts": local_contrasts,
        "local_factorial": local_factorial,
        "ppl_protocol": ppl_protocol,
        "ppl_comparisons": ppl_pairs,
        "ppl_factorial": ppl_factorial,
        "input_directories": {
            "prior_paper_evidence": str(evidence_root.resolve()),
            "local_mechanism": str(local_root.resolve()),
            "ppl_mechanism": str(ppl_root.resolve()),
        },
    }


def build_final_paper_tables(inputs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Build exact-join, summary, factorial, and claim tables."""

    local_lookup = {
        (row["residual_length"], _contrast_role(row), row["prompt_length"]): row
        for row in inputs["local_contrasts"]
    }
    ppl_lookup = {
        (row["residual_length"], row["baseline_role"], row["prompt_length"]): row
        for row in inputs["ppl_comparisons"]
    }
    mechanism_lengths = _build_mechanism_length_rows(local_lookup, ppl_lookup)
    mechanism_summary = _build_mechanism_summary(local_lookup, ppl_lookup)
    factorial = _build_factorial_evidence(inputs["local_factorial"], inputs["ppl_factorial"])
    claims = _build_claim_register(
        inputs["comparison_summary"], mechanism_summary, factorial
    )
    return {
        "operating_point_evidence": list(inputs["operating_point_evidence"]),
        "comparison_summary": list(inputs["comparison_summary"]),
        "mechanism_length_evidence": mechanism_lengths,
        "mechanism_summary": mechanism_summary,
        "factorial_evidence": factorial,
        "claim_register": claims,
    }


def write_final_paper_package(
    output_dir: str | Path,
    inputs: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    *,
    make_plots: bool = True,
) -> list[Path]:
    """Write the final auditable Llama-2 evidence package."""

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise Llama2PaperPackageError(f"paper-package directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name in TABLE_NAMES:
        rows = tables[name]
        outputs.append(_write_jsonl(destination / f"{name}.jsonl", rows))
        outputs.append(_write_csv(destination / f"{name}.csv", rows))

    protocol = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "model_scope": MODEL_SCOPE,
        "input_packages": inputs["input_directories"],
        "input_protocol_sha256": {
            "prior_paper_evidence": _json_sha256(inputs["evidence_protocol"]),
            "local_mechanism": _json_sha256(inputs["local_protocol"]),
            "ppl_mechanism": _json_sha256(inputs["ppl_protocol"]),
        },
        "input_source_states": {
            "prior_paper_evidence": inputs["evidence_protocol"]["input_source_states"],
            "local_mechanism": [inputs["local_protocol"]["source_state"]],
            "ppl_mechanism": [inputs["ppl_protocol"]["source_state"]],
        },
        "paired_ppl_protocol_deviation": inputs["evidence_protocol"][
            "paired_ppl_protocol_deviation"
        ],
        "table_rows": {name: len(tables[name]) for name in TABLE_NAMES},
        "exact_cross_metric_join_lengths": list(EXACT_JOIN_LENGTHS),
        "unmatched_native_context_rows": {
            "local_prompt_length": 4095,
            "ppl_prompt_length": 4032,
            "policy": "retain as separate local-only and PPL-only rows",
        },
        "metric_directions": {
            "paper_memory": "lower is better",
            "local_error_delta_full_minus_baseline": "negative favors full",
            "paired_nll_delta_full_minus_baseline": "negative favors full",
            "passkey_exact_accuracy": "higher is better",
        },
        "inference_boundary": (
            "descriptive evidence over three deterministic documents and fifty "
            "deterministic WikiText-2 anchors; bootstrap intervals are not "
            "population-level significance tests"
        ),
        "implementation_boundary": (
            "fake-quant prototype and packed paper-estimate memory only; no fused-kernel "
            "latency, throughput, CUDA-peak, or realized-allocation claim"
        ),
        "model_boundary": "Llama-2-7B only; cross-model generality is not established here",
    }
    protocol_path = destination / "paper_package_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    outputs.append(protocol_path)

    ledger = destination / "claim_ledger.md"
    ledger.write_text(_render_claim_ledger(tables), encoding="utf-8")
    outputs.append(ledger)
    draft = destination / "paper_results_draft.md"
    draft.write_text(_render_results_draft(tables), encoding="utf-8")
    outputs.append(draft)
    latex = destination / "paper_tables.tex"
    latex.write_text(_render_latex_tables(tables), encoding="utf-8")
    outputs.append(latex)
    if make_plots:
        outputs.extend(_plot_cross_metric(destination, tables["mechanism_summary"]))
    return outputs


def _build_mechanism_length_rows(
    local: dict[tuple[int, str, int], dict[str, Any]],
    ppl: dict[tuple[int, str, int | None], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for residual in RESIDUALS:
        for role in BASELINE_ROLES:
            comparison_id = f"r{residual}_full_vs_{role}"
            for length in LOCAL_LENGTHS:
                local_row = local[(residual, role, length)]
                ppl_row = ppl.get((residual, role, length)) if length in EXACT_JOIN_LENGTHS else None
                rows.append(_joined_mechanism_row(
                    comparison_id, residual, role, length, length if ppl_row else None,
                    "exact_prompt_length" if ppl_row else "local_only_4095",
                    local_row, ppl_row,
                ))
            ppl_native = ppl[(residual, role, 4032)]
            rows.append(_joined_mechanism_row(
                comparison_id, residual, role, None, 4032, "ppl_only_4032", None, ppl_native,
            ))
    return rows


def _joined_mechanism_row(
    comparison_id: str,
    residual: int,
    role: str,
    local_length: int | None,
    ppl_length: int | None,
    status: str,
    local: dict[str, Any] | None,
    ppl: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "comparison_id": comparison_id,
        "residual_length": residual,
        "baseline_role": role,
        "local_prompt_length": local_length,
        "ppl_prompt_length": ppl_length,
        "cross_metric_join_status": status,
        "local_paper_memory_change_percent": (
            local["paper_memory_change_vs_baseline_percent"] if local else None
        ),
        "local_error_change_percent": (
            local["candidate_error_change_vs_baseline_percent"] if local else None
        ),
        "local_baseline_error_penalty_percent": (
            local["baseline_error_penalty_vs_full_percent"] if local else None
        ),
        "local_full_better_samples": local["full_better_samples"] if local else None,
        "local_sample_count": local["sample_count"] if local else None,
        "same_paper_and_runtime_budget": (
            local["same_paper_and_runtime_budget"] if local else None
        ),
        "ppl_delta_nll": ppl["mean_paired_delta_nll"] if ppl else None,
        "ppl_ratio_full_to_baseline": ppl["candidate_ppl_ratio_to_baseline"] if ppl else None,
        "ppl_improvement_percent": (
            100.0 * (1.0 - ppl["candidate_ppl_ratio_to_baseline"]) if ppl else None
        ),
        "ppl_bootstrap_nll_ci95_low": (
            ppl["bootstrap_mean_delta_nll_ci95_low"] if ppl else None
        ),
        "ppl_bootstrap_nll_ci95_high": (
            ppl["bootstrap_mean_delta_nll_ci95_high"] if ppl else None
        ),
        "ppl_full_better_anchors": ppl["candidate_better_anchors"] if ppl else None,
        "ppl_full_worse_anchors": ppl["candidate_worse_anchors"] if ppl else None,
    }


def _build_mechanism_summary(
    local: dict[tuple[int, str, int], dict[str, Any]],
    ppl: dict[tuple[int, str, int | None], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for residual in RESIDUALS:
        for role in BASELINE_ROLES:
            local_rows = [local[(residual, role, length)] for length in LOCAL_LENGTHS]
            ppl_overall = ppl[(residual, role, None)]
            ppl_lengths = [ppl[(residual, role, length)] for length in PPL_LENGTHS]
            local_changes = [row["candidate_error_change_vs_baseline_percent"] for row in local_rows]
            memory_changes = [row["paper_memory_change_vs_baseline_percent"] for row in local_rows]
            summary = {
                "comparison_id": f"r{residual}_full_vs_{role}",
                "residual_length": residual,
                "baseline_role": role,
                "local_error_change_percent_min": min(local_changes),
                "local_error_change_percent_max": max(local_changes),
                "local_full_better_lengths": sum(value < 0 for value in local_changes),
                "local_full_better_sample_cells": sum(row["full_better_samples"] for row in local_rows),
                "local_total_sample_cells": sum(row["sample_count"] for row in local_rows),
                "paper_memory_change_percent_min": min(memory_changes),
                "paper_memory_change_percent_max": max(memory_changes),
                "same_budget_all_local_lengths": all(
                    row["same_paper_and_runtime_budget"] for row in local_rows
                ),
                "overall_ppl_delta_nll": ppl_overall["mean_paired_delta_nll"],
                "overall_ppl_ratio_full_to_baseline": ppl_overall[
                    "candidate_ppl_ratio_to_baseline"
                ],
                "overall_ppl_improvement_percent": 100.0 * (
                    1.0 - ppl_overall["candidate_ppl_ratio_to_baseline"]
                ),
                "overall_ppl_bootstrap_nll_ci95_low": ppl_overall[
                    "bootstrap_mean_delta_nll_ci95_low"
                ],
                "overall_ppl_bootstrap_nll_ci95_high": ppl_overall[
                    "bootstrap_mean_delta_nll_ci95_high"
                ],
                "overall_ppl_full_better_anchors": ppl_overall["candidate_better_anchors"],
                "overall_ppl_full_worse_anchors": ppl_overall["candidate_worse_anchors"],
                "ppl_point_favors_full_lengths": sum(
                    row["mean_paired_delta_nll"] < 0 for row in ppl_lengths
                ),
                "ppl_ci_favors_full_lengths": sum(
                    row["bootstrap_mean_delta_nll_ci95_high"] < 0 for row in ppl_lengths
                ),
                "ppl_ci_favors_baseline_lengths": sum(
                    row["bootstrap_mean_delta_nll_ci95_low"] > 0 for row in ppl_lengths
                ),
            }
            summary["claim_classification"] = _mechanism_classification(summary)
            rows.append(summary)
    return rows


def _mechanism_classification(row: dict[str, Any]) -> str:
    role = row["baseline_role"]
    local_supported = row["local_full_better_lengths"] == 4
    ppl_supported = row["overall_ppl_bootstrap_nll_ci95_high"] < 0
    if role == "fixed-random" and row["same_budget_all_local_lengths"]:
        return "same_budget_importance_ordering_supported" if local_supported and ppl_supported else "mixed"
    if role == "uniform":
        return "adaptive_schedule_supported_with_memory_tradeoff" if local_supported and ppl_supported else "mixed"
    if role == "k-adaptive":
        if ppl_supported:
            return "full_increment_supported"
        if row["overall_ppl_bootstrap_nll_ci95_low"] > 0:
            return "key_only_increment_supported"
        return "full_vs_key_only_descriptive_parity"
    if role == "v-adaptive":
        return "key_adaptation_increment_supported" if local_supported and ppl_supported else "mixed"
    raise Llama2PaperPackageError(f"unknown baseline role {role}")


def _build_factorial_evidence(
    local_rows: Sequence[dict[str, Any]],
    ppl_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    ppl_overall = {
        row["residual_length"]: row
        for row in ppl_rows if row["scope"] == "overall_anchor_clustered"
    }
    rows = []
    for residual in RESIDUALS:
        local = [row for row in local_rows if row["residual_length"] == residual]
        ppl = ppl_overall[residual]
        full = [row["full_error_reduction_vs_uniform_percent"] for row in local]
        key = [row["key_only_error_reduction_vs_uniform_percent"] for row in local]
        value = [row["value_only_error_reduction_vs_uniform_percent"] for row in local]
        interactions = [
            row["error_interaction_full_minus_key_minus_value_plus_uniform"]
            for row in local
        ]
        row = {
            "residual_length": residual,
            "local_full_error_reduction_percent_min": min(full),
            "local_full_error_reduction_percent_max": max(full),
            "local_key_error_reduction_percent_min": min(key),
            "local_key_error_reduction_percent_max": max(key),
            "local_value_error_reduction_percent_min": min(value),
            "local_value_error_reduction_percent_max": max(value),
            "local_key_fraction_of_full_reduction_min": min(k / f for k, f in zip(key, full)),
            "local_key_fraction_of_full_reduction_max": max(k / f for k, f in zip(key, full)),
            "local_value_fraction_of_full_reduction_min": min(v / f for v, f in zip(value, full)),
            "local_value_fraction_of_full_reduction_max": max(v / f for v, f in zip(value, full)),
            "local_interaction_min": min(interactions),
            "local_interaction_max": max(interactions),
        }
        for effect in (
            "full_minus_uniform",
            "key_adaptive_minus_uniform",
            "value_adaptive_minus_uniform",
            "interaction_full_minus_key_minus_value_plus_uniform",
        ):
            row[f"ppl_mean_{effect}"] = ppl[f"mean_{effect}"]
            row[f"ppl_bootstrap_{effect}_ci95_low"] = ppl[
                f"bootstrap_{effect}_ci95_low"
            ]
            row[f"ppl_bootstrap_{effect}_ci95_high"] = ppl[
                f"bootstrap_{effect}_ci95_high"
            ]
        row["ppl_key_fraction_of_full_effect"] = (
            row["ppl_mean_key_adaptive_minus_uniform"]
            / row["ppl_mean_full_minus_uniform"]
        )
        row["ppl_value_fraction_of_full_effect"] = (
            row["ppl_mean_value_adaptive_minus_uniform"]
            / row["ppl_mean_full_minus_uniform"]
        )
        row["mechanism_interpretation"] = (
            "key_dominant_ppl;value_local_contribution;ppl_value_and_interaction_uncertain"
        )
        rows.append(row)
    return rows


def _build_claim_register(
    comparisons: Sequence[dict[str, Any]],
    mechanisms: Sequence[dict[str, Any]],
    factorial: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparison_lookup = {row["comparison_id"]: row for row in comparisons}
    mechanism_lookup = {
        (row["residual_length"], row["baseline_role"]): row for row in mechanisms
    }
    r128 = comparison_lookup["cage-r128_vs_kivi-g32-r128"]
    r64 = comparison_lookup["cage-r64_vs_kivi-g64-r64"]
    random_rows = [mechanism_lookup[(residual, "fixed-random")] for residual in RESIDUALS]
    uniform_rows = [mechanism_lookup[(residual, "uniform")] for residual in RESIDUALS]
    return [
        {
            "claim_id": "operating_point_memory_efficiency",
            "status": "supported_with_tradeoff",
            "claim": (
                "At r128, CAGE uses less packed paper-estimate cache than KIVI g32-r128 "
                "with descriptive paired-PPL parity, while local perturbation is higher."
            ),
            "evidence": (
                f"memory savings {r128['memory_savings_percent_min']:+.3f}% to "
                f"{r128['memory_savings_percent_max']:+.3f}%; overall PPL change "
                f"{r128['overall_candidate_ppl_change_percent_vs_baseline']:+.3f}%"
            ),
            "prohibited_extension": "Do not claim simultaneous dominance in local perturbation.",
        },
        {
            "claim_id": "operating_point_quality",
            "status": "supported_with_memory_cost",
            "claim": (
                "At r64, CAGE improves local perturbation and paired PPL versus KIVI g64-r64 "
                "but uses more packed paper-estimate cache."
            ),
            "evidence": (
                f"memory savings field {r64['memory_savings_percent_min']:+.3f}% to "
                f"{r64['memory_savings_percent_max']:+.3f}%; overall PPL change "
                f"{r64['overall_candidate_ppl_change_percent_vs_baseline']:+.3f}%"
            ),
            "prohibited_extension": "Do not call this a memory win or universal KIVI dominance.",
        },
        {
            "claim_id": "importance_ordering",
            "status": "supported",
            "claim": "Importance-ordered variable grouping outperforms a same-budget fixed-random assignment.",
            "evidence": (
                "local full-better cells "
                f"{sum(row['local_full_better_sample_cells'] for row in random_rows)}/"
                f"{sum(row['local_total_sample_cells'] for row in random_rows)}; "
                "paired-PPL bootstrap intervals exclude zero for both residuals"
            ),
            "prohibited_extension": "Do not attribute this result to a fused implementation.",
        },
        {
            "claim_id": "adaptive_vs_uniform",
            "status": "supported_with_memory_tradeoff",
            "claim": "Full adaptive grouping improves local error and paired PPL over uniform grouping.",
            "evidence": (
                f"overall PPL improvement {min(row['overall_ppl_improvement_percent'] for row in uniform_rows):.3f}% "
                f"to {max(row['overall_ppl_improvement_percent'] for row in uniform_rows):.3f}%"
            ),
            "prohibited_extension": "Report the additional packed-memory cost of the adaptive schedule.",
        },
        {
            "claim_id": "key_dominant_mechanism",
            "status": "supported_descriptively",
            "claim": "Key adaptation accounts for most of the measured functional PPL benefit.",
            "evidence": (
                f"PPL key/full effect fraction {min(row['ppl_key_fraction_of_full_effect'] for row in factorial):.3f} "
                f"to {max(row['ppl_key_fraction_of_full_effect'] for row in factorial):.3f}"
            ),
            "prohibited_extension": "Do not claim that full and Key-only are statistically equivalent.",
        },
        {
            "claim_id": "value_adaptation_nuance",
            "status": "mixed",
            "claim": (
                "Value adaptation consistently improves local reconstruction, but its isolated "
                "paired-PPL effect is small and its descriptive interval crosses zero."
            ),
            "evidence": "local four-of-four direction at both residuals; overall PPL Value-effect intervals cross zero",
            "prohibited_extension": "Do not claim an independently established Value-side PPL gain.",
        },
        {
            "claim_id": "passkey_boundary",
            "status": "inconclusive_saturation",
            "claim": "The native-context passkey pilot is saturated and does not distinguish methods reliably.",
            "evidence": "all methods retain target containment; exact scores are 59/60 or 60/60",
            "prohibited_extension": "Do not claim passkey improvement over KIVI.",
        },
        {
            "claim_id": "paired_ppl_calibration_deviation",
            "status": "recorded_deviation",
            "claim": (
                "The paired-PPL FP16 diagnostic exceeded the pre-registered maximum "
                "single-token calibration tolerance in 3 of 12,800 reference values."
            ),
            "evidence": (
                "maximum absolute token-NLL delta 0.0308375 versus 0.03; "
                "mean-case gate passed and post-full repeat was bitwise reproducible"
            ),
            "prohibited_extension": "Do not erase or relabel the recorded protocol deviation.",
        },
        {
            "claim_id": "implementation_boundary",
            "status": "not_measured",
            "claim": "Kernel-level speed and realized-memory benefits remain unmeasured.",
            "evidence": "current CAGE path is a fake-quant prototype with paper-estimate packed memory",
            "prohibited_extension": "No latency, throughput, CUDA-peak, or fused-kernel performance claim.",
        },
    ]


def _render_claim_ledger(tables: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# CAGE-KV Llama-2-7B final claim ledger",
        "",
        "| Claim | Status | Evidence | Prohibited extension |",
        "|---|---|---|---|",
    ]
    for row in tables["claim_register"]:
        lines.append(
            f"| {row['claim']} | {row['status']} | {row['evidence']} | "
            f"{row['prohibited_extension']} |"
        )
    lines.extend([
        "",
        "## Scope",
        "",
        "All claims are limited to Llama-2-7B, the frozen natural-text and WikiText-2 "
        "grids, fake quantization, and packed paper-estimate memory. Bootstrap intervals "
        "describe stability over deterministic anchors rather than population inference.",
    ])
    return "\n".join(lines) + "\n"


def _render_results_draft(tables: dict[str, list[dict[str, Any]]]) -> str:
    comparisons = {row["comparison_id"]: row for row in tables["comparison_summary"]}
    mechanisms = {
        (row["residual_length"], row["baseline_role"]): row
        for row in tables["mechanism_summary"]
    }
    factorial = {row["residual_length"]: row for row in tables["factorial_evidence"]}
    r128 = comparisons["cage-r128_vs_kivi-g32-r128"]
    r64 = comparisons["cage-r64_vs_kivi-g64-r64"]
    uniform64 = mechanisms[(64, "uniform")]
    uniform128 = mechanisms[(128, "uniform")]
    random64 = mechanisms[(64, "fixed-random")]
    random128 = mechanisms[(128, "fixed-random")]
    r64_extra_memory_low = min(
        abs(r64["memory_savings_percent_min"]),
        abs(r64["memory_savings_percent_max"]),
    )
    r64_extra_memory_high = max(
        abs(r64["memory_savings_percent_min"]),
        abs(r64["memory_savings_percent_max"]),
    )
    return f"""# Llama-2-7B results draft

## Memory--quality operating points

The CAGE design exposes two distinct operating regimes rather than uniformly
dominating KIVI. At residual length 128, CAGE reduces the packed KV-cache
estimate by {r128['memory_savings_percent_min']:.2f}%--{r128['memory_savings_percent_max']:.2f}%
relative to KIVI g32-r128. Its overall paired PPL change is
{r128['overall_candidate_ppl_change_percent_vs_baseline']:+.3f}%, with a descriptive
bootstrap interval crossing zero, while its local post-output-projection
perturbation is higher. At residual length 64, CAGE instead uses approximately
{r64_extra_memory_low:.2f}%--{r64_extra_memory_high:.2f}%
more packed cache than KIVI g64-r64, but reduces both local perturbation and
paired PPL; the overall PPL change is
{r64['overall_candidate_ppl_change_percent_vs_baseline']:+.3f}%.

## Mechanism evidence

Importance ordering is not explained by bucket counts alone. Against the
same-budget fixed-random control, full CAGE is better in
{random64['local_full_better_sample_cells'] + random128['local_full_better_sample_cells']}/{random64['local_total_sample_cells'] + random128['local_total_sample_cells']}
local document--length cells. The paired PPL of full CAGE is
{random64['overall_ppl_improvement_percent']:.2f}% and
{random128['overall_ppl_improvement_percent']:.2f}% lower at residual lengths 64
and 128, respectively, and all length-stratified descriptive intervals favor
the importance-ordered assignment.

Relative to uniform grouping, full CAGE reduces local perturbation at all four
lengths and lowers paired PPL by {uniform64['overall_ppl_improvement_percent']:.2f}%
(r64) and {uniform128['overall_ppl_improvement_percent']:.2f}% (r128). This gain
requires a packed-memory increase whose exact magnitude depends on residual
length and prompt length.

## Key and Value contributions

The factorial decomposition identifies Key adaptation as the main functional
driver. Key-only adaptation accounts for
{100.0 * factorial[64]['ppl_key_fraction_of_full_effect']:.1f}% (r64) and
{100.0 * factorial[128]['ppl_key_fraction_of_full_effect']:.1f}% (r128) of the
full-versus-uniform paired-NLL effect. The isolated Value-side PPL effect and
the Key--Value interaction both have descriptive intervals crossing zero.
Value adaptation nevertheless reduces local reconstruction error at every
tested length, so the evidence supports a local Value-side contribution but
not an independently established Value-side PPL gain.

## Task diagnostic and limitations

The native-context passkey pilot is saturated: all methods retain target
containment and exact accuracy is either 59/60 or 60/60. It therefore serves as
a regression check rather than evidence of task improvement. All memory values
above are packed paper estimates from a fake-quant prototype. The experiments
do not establish fused-kernel latency, throughput, CUDA-peak, realized-memory,
population-level significance, or cross-model generality. The paired-PPL FP16
diagnostic also retains its recorded calibration deviation: the mean-case gate
passed, while three single-token reference deltas marginally exceeded the
pre-registered 0.03 maximum tolerance.
"""


def _render_latex_tables(tables: dict[str, list[dict[str, Any]]]) -> str:
    comparisons = sorted(tables["comparison_summary"], key=lambda row: row["comparison_order"])
    mechanisms = sorted(
        tables["mechanism_summary"],
        key=lambda row: (row["residual_length"], BASELINE_ROLES.index(row["baseline_role"])),
    )
    lines = [
        "% Auto-generated from frozen CAGE-KV Llama-2-7B evidence.",
        "\\begin{tabular}{lrrl}",
        "\\toprule",
        "Comparison & Memory savings (\\%) & PPL change (\\%) & Interpretation \\\\",
        "\\midrule",
    ]
    for row in comparisons:
        label = row["comparison_id"].replace("_", "\\_")
        memory = (
            f"{row['memory_savings_percent_min']:+.2f} to "
            f"{row['memory_savings_percent_max']:+.2f}"
        )
        lines.append(
            f"{label} & {memory} & {row['overall_candidate_ppl_change_percent_vs_baseline']:+.3f} "
            f"& {row['ppl_status'].replace('_', ' ')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "", "\\begin{tabular}{rrrll}", "\\toprule"])
    lines.append("Residual & Baseline & PPL improvement (\\%) & Local direction & Classification \\\\")
    lines.append("\\midrule")
    for row in mechanisms:
        role = row["baseline_role"].replace("-", "--")
        direction = f"{row['local_full_better_lengths']}/4 lengths"
        classification = row["claim_classification"].replace("_", " ")
        lines.append(
            f"{row['residual_length']} & {role} & {row['overall_ppl_improvement_percent']:+.3f} "
            f"& {direction} & {classification} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _plot_cross_metric(destination: Path, rows: Sequence[dict[str, Any]]) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise Llama2PaperPackageError("matplotlib is required for package plots") from error

    colors = {64: "#2f7d32", 128: "#76b852"}
    role_labels = {
        "fixed-random": "Fixed random",
        "uniform": "Uniform",
        "k-adaptive": "Key-only",
        "v-adaptive": "Value-only",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.5))
    x = list(range(len(BASELINE_ROLES)))
    offsets = {64: -0.10, 128: 0.10}
    for residual in RESIDUALS:
        selected = {
            row["baseline_role"]: row for row in rows if row["residual_length"] == residual
        }
        means = []
        lower = []
        upper = []
        ppl_improvements = []
        ppl_lower = []
        ppl_upper = []
        for role in BASELINE_ROLES:
            row = selected[role]
            local_low = -row["local_error_change_percent_max"]
            local_high = -row["local_error_change_percent_min"]
            means.append((local_low + local_high) / 2.0)
            lower.append((local_high - local_low) / 2.0)
            upper.append((local_high - local_low) / 2.0)
            improvement = row["overall_ppl_improvement_percent"]
            ci_low = 100.0 * (1.0 - math.exp(row["overall_ppl_bootstrap_nll_ci95_high"]))
            ci_high = 100.0 * (1.0 - math.exp(row["overall_ppl_bootstrap_nll_ci95_low"]))
            ppl_improvements.append(improvement)
            ppl_lower.append(improvement - ci_low)
            ppl_upper.append(ci_high - improvement)
        positions = [value + offsets[residual] for value in x]
        axes[0].errorbar(
            positions, means, yerr=[lower, upper], fmt="o", capsize=4,
            color=colors[residual], label=f"r={residual}",
        )
        axes[1].errorbar(
            positions, ppl_improvements, yerr=[ppl_lower, ppl_upper], fmt="o",
            capsize=4, color=colors[residual], label=f"r={residual}",
        )
    for axis in axes:
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_xticks(x, [role_labels[role] for role in BASELINE_ROLES], rotation=18)
        axis.grid(True, axis="y", alpha=0.22)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Full-CAGE local-error improvement (%)")
    axes[0].set_title("Range across four prompt lengths")
    axes[1].set_ylabel("Full-CAGE paired-PPL improvement (%)")
    axes[1].set_title("Overall paired bootstrap 95% intervals")
    figure.suptitle("CAGE-KV Llama-2-7B cross-metric mechanism evidence")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    png = destination / "mechanism_cross_metric.png"
    pdf = destination / "mechanism_cross_metric.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return [png, pdf]


def _validate_clean_states(
    evidence: dict[str, Any], local: dict[str, Any], ppl: dict[str, Any]
) -> None:
    evidence_states = evidence.get("input_source_states")
    if not isinstance(evidence_states, dict) or not evidence_states:
        raise Llama2PaperPackageError("prior evidence lacks source states")
    states = [
        state
        for group in evidence_states.values()
        for state in group
    ] + [local.get("source_state"), ppl.get("source_state")]
    if any(not isinstance(state, dict) or state.get("dirty") is not False for state in states):
        raise Llama2PaperPackageError("an input analysis lacks a clean source state")


def _contrast_role(row: dict[str, Any]) -> str:
    contrast = row.get("contrast_id")
    if not isinstance(contrast, str) or not contrast.startswith("full_vs_"):
        raise Llama2PaperPackageError(f"invalid local contrast ID {contrast!r}")
    return contrast.removeprefix("full_vs_")


def _validate_numeric_rows(name: str, rows: Sequence[dict[str, Any]]) -> None:
    def visit(value: Any, path: str) -> None:
        if value is None or isinstance(value, (str, bool)):
            return
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise Llama2PaperPackageError(f"non-finite {name} value at {path}")
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
    visit(list(rows), name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2PaperPackageError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Llama2PaperPackageError(f"JSON {path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                raise Llama2PaperPackageError(f"blank line {line_number} in {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Llama2PaperPackageError(f"row {line_number} in {path} is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2PaperPackageError(f"cannot read JSONL {path}: {error}") from error
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ))
            handle.write("\n")
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "Llama2PaperPackageError", "PACKAGE_SCHEMA_VERSION",
    "build_final_paper_tables", "load_final_paper_inputs",
    "write_final_paper_package",
]
