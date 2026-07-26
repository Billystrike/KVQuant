"""Cross-experiment evidence synthesis for the Llama-2-7B CAGE-KV paper."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


EVIDENCE_SCHEMA_VERSION = 1
MODEL_SCOPE = "Llama-2-7B-hf"
PARETO_LENGTHS = (512, 1024, 2048, 4095)
PAIRED_PPL_LENGTHS = (512, 1024, 2048, 4032)
EXACT_JOIN_LENGTHS = (512, 1024, 2048)
COMPARISONS = (
    {
        "comparison_id": "cage-r128_vs_kivi-g32-r128",
        "candidate_method_id": "cage-r128",
        "baseline_method_id": "kivi-g32-r128",
    },
    {
        "comparison_id": "cage-r64_vs_kivi-g64-r64",
        "candidate_method_id": "cage-r64",
        "baseline_method_id": "kivi-g64-r64",
    },
)


class PaperEvidenceError(ValueError):
    """Raised when frozen analyses cannot support the declared evidence join."""


def load_evidence_inputs(
    pareto_analysis_dir: str | Path,
    paired_ppl_analysis_dir: str | Path,
    passkey_analysis_dir: str | Path,
) -> dict[str, Any]:
    """Strictly load the three frozen Llama-2-7B analysis packages."""

    pareto_root = Path(pareto_analysis_dir)
    paired_root = Path(paired_ppl_analysis_dir)
    passkey_root = Path(passkey_analysis_dir)

    pareto_protocol = _read_json(pareto_root / "analysis_protocol.json")
    pareto_rows = _read_jsonl(pareto_root / "aggregate_points.jsonl")
    if (
        pareto_protocol.get("schema_version") != 2
        or pareto_protocol.get("aggregate_point_count") != 40
        or len(pareto_rows) != 40
        or pareto_protocol.get("primary_error_metric") != "joint_post_o_proj_mse"
        or pareto_protocol.get("memory_axis") != "memory.paper_estimate.total_bytes"
    ):
        raise PaperEvidenceError("Pareto analysis differs from the frozen 40-point protocol")
    pareto_keys = [(row.get("config_id"), row.get("prompt_length")) for row in pareto_rows]
    if len(set(pareto_keys)) != 40 or {key[1] for key in pareto_keys} != set(PARETO_LENGTHS):
        raise PaperEvidenceError("Pareto method/length coverage is invalid")
    pareto_required = {
        "config_id", "method", "prompt_length", "paper_total_bytes",
        "paper_total_mib", "compression_ratio_vs_fp16", "primary_error",
        "primary_error_sample_pstdev", "is_pareto_global",
    }
    for row in pareto_rows:
        if not pareto_required.issubset(row):
            raise PaperEvidenceError(f"Pareto row {row.get('config_id')!r} lacks fields")
        for field in ("paper_total_bytes", "paper_total_mib", "primary_error"):
            _finite(f"Pareto {field}", row[field], minimum=0.0)

    paired_protocol = _read_json(paired_root / "analysis_protocol.json")
    paired_rows = _read_jsonl(paired_root / "paired_comparisons.jsonl")
    calibration = _read_jsonl(paired_root / "fp16_calibration_audit.jsonl")
    if (
        paired_protocol.get("schema_version") != 2
        or paired_protocol.get("case_count") != 1000
        or paired_protocol.get("primary_analysis_scope")
        != "pre_registered_cage_vs_kivi_paired_comparisons_only"
        or len(paired_rows) != 10
        or len(calibration) != 1
    ):
        raise PaperEvidenceError("paired-PPL analysis differs from the frozen protocol")
    if (
        calibration[0].get("overall_gate") != "FAIL"
        or calibration[0].get("protocol_status") != "RECORDED_DEVIATION"
        or paired_protocol.get("post_run_protocol_record", {}).get("status")
        != "RECORDED_DEVIATION"
    ):
        raise PaperEvidenceError("paired-PPL calibration deviation is not recorded")
    paired_keys = [(row.get("comparison_id"), row.get("prompt_length")) for row in paired_rows]
    expected_paired_keys = {
        (comparison["comparison_id"], length)
        for comparison in COMPARISONS
        for length in (None, *PAIRED_PPL_LENGTHS)
    }
    if set(paired_keys) != expected_paired_keys or len(set(paired_keys)) != 10:
        raise PaperEvidenceError("paired-PPL comparison coverage is invalid")

    passkey_protocol = _read_json(passkey_root / "analysis_protocol.json")
    passkey_methods = _read_jsonl(passkey_root / "method_summary.jsonl")
    passkey_pairs = _read_jsonl(passkey_root / "paired_comparisons.jsonl")
    if (
        passkey_protocol.get("schema_version") != 1
        or passkey_protocol.get("protocol_stage") != "stage_b"
        or passkey_protocol.get("case_count") != 300
        or len(passkey_methods) != 5
        or len(passkey_pairs) != 10
    ):
        raise PaperEvidenceError("passkey analysis differs from the frozen Stage-B protocol")
    if len({row.get("method_id") for row in passkey_methods}) != 5:
        raise PaperEvidenceError("passkey method summary contains duplicate IDs")

    for name, protocol in (
        ("Pareto", pareto_protocol),
        ("paired-PPL", paired_protocol),
        ("passkey", passkey_protocol),
    ):
        states = protocol.get("source_states") if name == "Pareto" else [protocol.get("source_state")]
        if not isinstance(states, list) or not states or any(
            not isinstance(state, dict) or state.get("dirty") is not False
            for state in states
        ):
            raise PaperEvidenceError(f"{name} analysis lacks a clean frozen source state")

    return {
        "pareto_protocol": pareto_protocol,
        "pareto_rows": pareto_rows,
        "paired_ppl_protocol": paired_protocol,
        "paired_ppl_rows": paired_rows,
        "paired_ppl_calibration": calibration[0],
        "passkey_protocol": passkey_protocol,
        "passkey_method_rows": passkey_methods,
        "passkey_pair_rows": passkey_pairs,
        "input_directories": {
            "pareto": str(pareto_root.resolve()),
            "paired_ppl": str(paired_root.resolve()),
            "passkey": str(passkey_root.resolve()),
        },
    }


def build_paper_evidence(inputs: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Build exact-length operating-point rows and two comparison summaries."""

    pareto = {
        (row["config_id"], row["prompt_length"]): row
        for row in inputs["pareto_rows"]
    }
    paired = {
        (row["comparison_id"], row["prompt_length"]): row
        for row in inputs["paired_ppl_rows"]
    }
    passkey_methods = {
        row["method_id"]: row for row in inputs["passkey_method_rows"]
    }

    evidence_rows = []
    summary_rows = []
    for order, comparison in enumerate(COMPARISONS):
        comparison_id = comparison["comparison_id"]
        candidate_id = comparison["candidate_method_id"]
        baseline_id = comparison["baseline_method_id"]
        comparison_evidence = []
        for prompt_length in PARETO_LENGTHS:
            candidate = pareto.get((candidate_id, prompt_length))
            baseline = pareto.get((baseline_id, prompt_length))
            if candidate is None or baseline is None:
                raise PaperEvidenceError(
                    f"Pareto analysis lacks {comparison_id} at length {prompt_length}"
                )
            exact_join = prompt_length in EXACT_JOIN_LENGTHS
            ppl = paired.get((comparison_id, prompt_length)) if exact_join else None
            if exact_join and ppl is None:
                raise PaperEvidenceError(
                    f"paired-PPL analysis lacks exact join {comparison_id}/{prompt_length}"
                )
            row = {
                "comparison_id": comparison_id,
                "comparison_order": order,
                "candidate_method_id": candidate_id,
                "baseline_method_id": baseline_id,
                "pareto_prompt_length": prompt_length,
                "ppl_prompt_length": prompt_length if exact_join else None,
                "ppl_join_status": (
                    "exact_prompt_length" if exact_join else "unmatched_4095_vs_4032"
                ),
                "candidate_paper_total_bytes": candidate["paper_total_bytes"],
                "baseline_paper_total_bytes": baseline["paper_total_bytes"],
                "candidate_paper_total_mib": candidate["paper_total_mib"],
                "baseline_paper_total_mib": baseline["paper_total_mib"],
                "candidate_memory_fraction_of_baseline": (
                    candidate["paper_total_bytes"] / baseline["paper_total_bytes"]
                ),
                "candidate_memory_savings_percent_vs_baseline": 100.0 * (
                    1.0 - candidate["paper_total_bytes"] / baseline["paper_total_bytes"]
                ),
                "candidate_primary_error": candidate["primary_error"],
                "baseline_primary_error": baseline["primary_error"],
                "candidate_primary_error_ratio_to_baseline": (
                    candidate["primary_error"] / baseline["primary_error"]
                ),
                "candidate_primary_error_change_percent_vs_baseline": 100.0 * (
                    candidate["primary_error"] / baseline["primary_error"] - 1.0
                ),
                "candidate_is_perturbation_pareto": candidate["is_pareto_global"],
                "baseline_is_perturbation_pareto": baseline["is_pareto_global"],
                "paired_delta_nll_candidate_minus_baseline": (
                    ppl["mean_paired_delta_nll"] if ppl else None
                ),
                "candidate_ppl_ratio_to_baseline": (
                    ppl["candidate_ppl_ratio_to_baseline"] if ppl else None
                ),
                "candidate_ppl_change_percent_vs_baseline": (
                    100.0 * (ppl["candidate_ppl_ratio_to_baseline"] - 1.0)
                    if ppl else None
                ),
                "paired_bootstrap_nll_ci95_low": (
                    ppl["bootstrap_mean_delta_nll_ci95_low"] if ppl else None
                ),
                "paired_bootstrap_nll_ci95_high": (
                    ppl["bootstrap_mean_delta_nll_ci95_high"] if ppl else None
                ),
                "candidate_better_ppl_anchors": (
                    ppl["candidate_better_anchors"] if ppl else None
                ),
                "candidate_worse_ppl_anchors": (
                    ppl["candidate_worse_anchors"] if ppl else None
                ),
            }
            comparison_evidence.append(row)
            evidence_rows.append(row)

        overall = paired[(comparison_id, None)]
        memory_savings = [
            row["candidate_memory_savings_percent_vs_baseline"]
            for row in comparison_evidence
        ]
        exact_rows = [
            row for row in comparison_evidence
            if row["ppl_join_status"] == "exact_prompt_length"
        ]
        passkey = _normalized_passkey(
            candidate_id, baseline_id,
            passkey_methods, inputs["passkey_pair_rows"],
        )
        memory_status = (
            "candidate_lower_all_lengths"
            if all(value > 0 for value in memory_savings)
            else "candidate_higher_all_lengths"
            if all(value < 0 for value in memory_savings)
            else "mixed_by_length"
        )
        if overall["bootstrap_mean_delta_nll_ci95_high"] < 0:
            ppl_status = "descriptive_improvement"
        elif overall["bootstrap_mean_delta_nll_ci95_low"] > 0:
            ppl_status = "descriptive_regression"
        else:
            ppl_status = "descriptive_parity"
        summary_rows.append({
            "comparison_id": comparison_id,
            "comparison_order": order,
            "candidate_method_id": candidate_id,
            "baseline_method_id": baseline_id,
            "memory_status": memory_status,
            "memory_savings_percent_min": min(memory_savings),
            "memory_savings_percent_max": max(memory_savings),
            "memory_better_length_count": sum(value > 0 for value in memory_savings),
            "perturbation_better_length_count": sum(
                row["candidate_primary_error"] < row["baseline_primary_error"]
                for row in comparison_evidence
            ),
            "exact_memory_ppl_join_length_count": len(exact_rows),
            "exact_join_win_win_count": sum(
                row["candidate_memory_savings_percent_vs_baseline"] > 0
                and row["candidate_ppl_change_percent_vs_baseline"] < 0
                for row in exact_rows
            ),
            "overall_paired_delta_nll": overall["mean_paired_delta_nll"],
            "overall_candidate_ppl_ratio_to_baseline": overall[
                "candidate_ppl_ratio_to_baseline"
            ],
            "overall_candidate_ppl_change_percent_vs_baseline": 100.0 * (
                overall["candidate_ppl_ratio_to_baseline"] - 1.0
            ),
            "overall_bootstrap_nll_ci95_low": overall[
                "bootstrap_mean_delta_nll_ci95_low"
            ],
            "overall_bootstrap_nll_ci95_high": overall[
                "bootstrap_mean_delta_nll_ci95_high"
            ],
            "overall_candidate_better_anchors": overall["candidate_better_anchors"],
            "overall_candidate_worse_anchors": overall["candidate_worse_anchors"],
            "ppl_status": ppl_status,
            **passkey,
            "claim_classification": f"{memory_status};{ppl_status}",
        })

    return {
        "operating_point_evidence": evidence_rows,
        "comparison_summary": summary_rows,
    }


def write_paper_evidence_outputs(
    output_dir: str | Path,
    inputs: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    *,
    make_plots: bool = True,
) -> list[Path]:
    """Write auditable evidence tables, claim ledger, protocol, and optional plot."""

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise PaperEvidenceError(f"evidence directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name in ("operating_point_evidence", "comparison_summary"):
        rows = tables[name]
        outputs.append(_write_jsonl(destination / f"{name}.jsonl", rows))
        outputs.append(_write_csv(destination / f"{name}.csv", rows))

    protocol = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "model_scope": MODEL_SCOPE,
        "comparison_count": len(COMPARISONS),
        "operating_point_row_count": len(tables["operating_point_evidence"]),
        "exact_join_lengths": list(EXACT_JOIN_LENGTHS),
        "unmatched_native_context_lengths": {
            "pareto_prompt_length": 4095,
            "paired_ppl_prompt_length": 4032,
            "policy": "retain both as unmatched; never silently equate them",
        },
        "comparisons": list(COMPARISONS),
        "metric_directions": {
            "paper_estimate_bytes": "lower is better",
            "joint_post_o_proj_mse": "lower is better",
            "paired_delta_nll_candidate_minus_baseline": "negative favors CAGE",
            "passkey_exact_accuracy": "higher is better",
        },
        "claim_boundary": (
            "Llama-2-7B only; paper-estimate memory and fake-quant quality evidence. "
            "No fused-kernel latency, throughput, realized-memory, population-significance, "
            "or canonical full-corpus PPL claim."
        ),
        "paired_ppl_protocol_deviation": inputs["paired_ppl_calibration"],
        "input_protocol_sha256": {
            "pareto": _json_sha256(inputs["pareto_protocol"]),
            "paired_ppl": _json_sha256(inputs["paired_ppl_protocol"]),
            "passkey": _json_sha256(inputs["passkey_protocol"]),
        },
        "input_directories": inputs["input_directories"],
        "input_source_states": {
            "pareto": inputs["pareto_protocol"]["source_states"],
            "paired_ppl": [inputs["paired_ppl_protocol"]["source_state"]],
            "passkey": [inputs["passkey_protocol"]["source_state"]],
        },
    }
    protocol_path = destination / "evidence_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    outputs.append(protocol_path)
    ledger_path = destination / "claim_ledger.md"
    ledger_path.write_text(_render_claim_ledger(tables), encoding="utf-8")
    outputs.append(ledger_path)
    if make_plots:
        outputs.extend(_plot_direct_tradeoff(destination, tables["operating_point_evidence"]))
    return outputs


def _normalized_passkey(
    candidate_id: str,
    baseline_id: str,
    methods: dict[str, dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if candidate_id not in methods or baseline_id not in methods:
        return {
            "passkey_join_status": "method_not_in_stage_b",
            "passkey_trials": None,
            "candidate_passkey_exact_accuracy": None,
            "baseline_passkey_exact_accuracy": None,
            "candidate_passkey_exact_accuracy_delta": None,
            "candidate_passkey_contains_accuracy": None,
            "baseline_passkey_contains_accuracy": None,
            "passkey_candidate_only_exact": None,
            "passkey_baseline_only_exact": None,
            "passkey_neither_exact": None,
        }
    candidate = methods[candidate_id]
    baseline = methods[baseline_id]
    matches = [
        row for row in pair_rows
        if {row["method_a_id"], row["method_b_id"]} == {candidate_id, baseline_id}
    ]
    if len(matches) != 1:
        raise PaperEvidenceError(
            f"passkey analysis requires one pair for {candidate_id}/{baseline_id}"
        )
    pair = matches[0]
    if pair["method_a_id"] == baseline_id:
        candidate_only = pair["method_b_only_exact"]
        baseline_only = pair["method_a_only_exact"]
    else:
        candidate_only = pair["method_a_only_exact"]
        baseline_only = pair["method_b_only_exact"]
    return {
        "passkey_join_status": "exact_methods",
        "passkey_trials": candidate["trials"],
        "candidate_passkey_exact_accuracy": candidate["exact_accuracy"],
        "baseline_passkey_exact_accuracy": baseline["exact_accuracy"],
        "candidate_passkey_exact_accuracy_delta": (
            candidate["exact_accuracy"] - baseline["exact_accuracy"]
        ),
        "candidate_passkey_contains_accuracy": candidate["contains_accuracy"],
        "baseline_passkey_contains_accuracy": baseline["contains_accuracy"],
        "passkey_candidate_only_exact": candidate_only,
        "passkey_baseline_only_exact": baseline_only,
        "passkey_neither_exact": pair["neither_exact"],
    }


def _render_claim_ledger(tables: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# CAGE-KV Llama-2-7B evidence ledger",
        "",
        "> This ledger joins only exact prompt lengths 512, 1024, and 2048. "
        "Pareto length 4095 and paired-PPL length 4032 remain explicitly unmatched.",
        "",
        "## Declared operating-point comparisons",
        "",
        "| Comparison | Memory range vs KIVI | Overall PPL change | PPL status | Passkey | Claim class |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in tables["comparison_summary"]:
        passkey = (
            "not tested"
            if row["passkey_join_status"] != "exact_methods"
            else f"{row['candidate_passkey_exact_accuracy']:.3f} vs "
            f"{row['baseline_passkey_exact_accuracy']:.3f}"
        )
        lines.append(
            f"| {row['comparison_id']} | "
            f"[{row['memory_savings_percent_min']:+.3f}%, "
            f"{row['memory_savings_percent_max']:+.3f}%] | "
            f"{row['overall_candidate_ppl_change_percent_vs_baseline']:+.3f}% | "
            f"{row['ppl_status']} | {passkey} | {row['claim_classification']} |"
        )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "These are Llama-2-7B fake-quant prototype results. Memory is the packed "
        "paper estimate, not runtime tensor allocation or CUDA peak. Paired-PPL "
        "bootstrap intervals are descriptive over deterministic anchors. The recorded "
        "FP16 one-shot calibration deviation remains part of the evidence record.",
        "",
        "Mechanism ablations and task-level evaluation remain required before closing "
        "the Llama-2-7B paper claims.",
    ])
    return "\n".join(lines) + "\n"


def _plot_direct_tradeoff(destination: Path, rows: Sequence[dict[str, Any]]) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise PaperEvidenceError("matplotlib is required for evidence plots") from error

    exact = [row for row in rows if row["ppl_join_status"] == "exact_prompt_length"]
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.5), sharex=False, sharey=True)
    colors = {512: "#4e79a7", 1024: "#f28e2b", 2048: "#59a14f"}
    for axis, comparison in zip(axes, COMPARISONS):
        selected = [
            row for row in exact if row["comparison_id"] == comparison["comparison_id"]
        ]
        for row in selected:
            length = row["pareto_prompt_length"]
            axis.scatter(
                row["candidate_memory_savings_percent_vs_baseline"],
                row["candidate_ppl_change_percent_vs_baseline"],
                s=70, color=colors[length], edgecolor="#333333", linewidth=0.7,
                label=str(length), zorder=3,
            )
            axis.annotate(
                str(length),
                (
                    row["candidate_memory_savings_percent_vs_baseline"],
                    row["candidate_ppl_change_percent_vs_baseline"],
                ),
                xytext=(5, 5), textcoords="offset points", fontsize=9,
            )
        axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
        axis.set_xlabel("CAGE paper-memory savings vs KIVI (%)")
        axis.set_ylabel("CAGE PPL change vs KIVI (%)")
        axis.set_title(comparison["comparison_id"].replace("_", "\n"))
        axis.grid(True, alpha=0.22)
    figure.suptitle(
        "Llama-2-7B exact-length direct operating-point evidence\n"
        "Right is less paper-estimate memory; down is lower paired PPL",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    png = destination / "direct_memory_ppl_tradeoff.png"
    pdf = destination / "direct_memory_ppl_tradeoff.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return [png, pdf]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PaperEvidenceError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PaperEvidenceError(f"JSON {path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise PaperEvidenceError(f"blank line {line_number} in {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PaperEvidenceError(f"row {line_number} in {path} is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise PaperEvidenceError(f"cannot read JSONL {path}: {error}") from error
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


def _finite(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperEvidenceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise PaperEvidenceError(f"{name} must be finite and >= {minimum}")
    return result


__all__ = [
    "COMPARISONS", "EVIDENCE_SCHEMA_VERSION", "PaperEvidenceError",
    "build_paper_evidence", "load_evidence_inputs", "write_paper_evidence_outputs",
]
