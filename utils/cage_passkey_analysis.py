"""Read-only validation and descriptive analysis for Stage-B passkey results."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from utils.cage_experiment_config import resolve_method
from utils.cage_passkey import (
    PASSKEY_GENERATION_SEED,
    PASSKEY_KEY_SEED,
    PASSKEY_MAX_NEW_TOKENS,
    PASSKEY_POSITIONS_PERCENT,
    PASSKEY_PROMPT_LENGTHS,
    PASSKEY_SCHEMA_VERSION,
    STAGE_B_RAW_METHODS,
    generate_passkeys,
    validate_completed_passkey_case,
)


ANALYSIS_SCHEMA_VERSION = 1
WILSON_Z_95 = 1.959963984540054
STAGE_B_METHOD_IDS = tuple(method["id"] for method in STAGE_B_RAW_METHODS)


class PasskeyAnalysisError(ValueError):
    """Raised when frozen passkey artifacts cannot support the analysis."""


def load_completed_passkey_matrix(
    results_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Strictly validate the frozen 300-case Stage-B output."""

    root = Path(results_dir)
    resolved = _read_json(root / "manifest.resolved.json", "resolved manifest")
    quality = _read_json(root / "summary" / "quality.json", "quality summary")
    _validate_resolved_manifest(resolved)

    expanded = resolved["expanded_cases"]
    expected_ids = [item.get("case_id") for item in expanded]
    if any(not isinstance(case_id, str) or not case_id for case_id in expected_ids):
        raise PasskeyAnalysisError("resolved expanded_cases contains an invalid case_id")
    if len(expected_ids) != 300 or len(set(expected_ids)) != 300:
        raise PasskeyAnalysisError(
            f"resolved manifest requires 300 unique cases, got {len(expected_ids)} rows "
            f"and {len(set(expected_ids))} unique IDs"
        )
    expected_id_set = set(expected_ids)

    case_ids = {path.stem for path in (root / "cases").glob("*.json")}
    failure_ids = {path.stem for path in (root / "failures").glob("*.json")}
    if case_ids != expected_id_set:
        raise PasskeyAnalysisError(_id_set_error("case artifacts", expected_id_set, case_ids))
    if failure_ids:
        raise PasskeyAnalysisError(f"failure artifacts remain: {sorted(failure_ids)}")

    records = []
    for case_id in expected_ids:
        try:
            records.append(validate_completed_passkey_case(root, case_id))
        except ValueError as error:
            raise PasskeyAnalysisError(f"invalid completed case {case_id}: {error}") from error

    summary_rows = _read_jsonl(root / "summary" / "cases.jsonl")
    summary_ids = [row.get("case_id") for row in summary_rows]
    if len(summary_ids) != 300 or set(summary_ids) != expected_id_set:
        raise PasskeyAnalysisError(
            _id_set_error("summary/cases.jsonl", expected_id_set, set(summary_ids))
        )
    if len(summary_ids) != len(set(summary_ids)):
        raise PasskeyAnalysisError("summary/cases.jsonl contains duplicate case IDs")
    record_by_id = {record["case_id"]: record for record in records}
    if any(row != record_by_id[row["case_id"]] for row in summary_rows):
        raise PasskeyAnalysisError("summary/cases.jsonl differs from completed case artifacts")

    csv_ids = _read_csv_ids(root / "summary" / "cases.csv")
    if len(csv_ids) != 300 or set(csv_ids) != expected_id_set:
        raise PasskeyAnalysisError(
            _id_set_error("summary/cases.csv", expected_id_set, set(csv_ids))
        )
    if len(csv_ids) != len(set(csv_ids)):
        raise PasskeyAnalysisError("summary/cases.csv contains duplicate case IDs")

    expanded_by_id = {item["case_id"]: item for item in expanded}
    for record in records:
        declared = expanded_by_id[record["case_id"]]
        if set(declared) != {"case_id", "method", "input"}:
            raise PasskeyAnalysisError(
                f"expanded case {record['case_id']} has unexpected fields"
            )
        if declared["method"] != record["method"] or declared["input"] != record["input"]:
            raise PasskeyAnalysisError(
                f"expanded case {record['case_id']} differs from its completed artifact"
            )

    source_state_json = _canonical_json(resolved["source_state"])
    record_source_states = {
        _canonical_json(record["provenance"]["source_state"])
        for record in records
    }
    if record_source_states != {source_state_json}:
        raise PasskeyAnalysisError("completed cases differ from resolved source_state")

    _validate_coverage(records)
    tables = aggregate_passkey_results(records, resolved)
    _validate_quality_summary(quality, tables, resolved)
    return resolved, records, quality


def aggregate_passkey_results(
    records: Sequence[dict[str, Any]],
    resolved_manifest: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build auditable method, stratum, paired, and miss tables."""

    method_catalog = {
        method["id"]: {
            "method_id": method["id"],
            "method": method["method"],
            "method_order": index,
            "resolved_config_json": _canonical_json(method["method_config"]),
        }
        for index, method in enumerate(resolved_manifest["methods"])
    }
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        method_id = record["method"]["id"]
        if method_id not in method_catalog:
            raise PasskeyAnalysisError(f"record uses undeclared method {method_id!r}")
        by_method[method_id].append(record)

    method_rows = []
    length_rows = []
    position_rows = []
    cell_rows = []
    for method_id in STAGE_B_METHOD_IDS:
        metadata = method_catalog[method_id]
        method_records = by_method[method_id]
        method_rows.append(_accuracy_row(metadata, method_records))
        for prompt_length in PASSKEY_PROMPT_LENGTHS:
            subset = [
                record
                for record in method_records
                if record["input"]["prompt_length"] == prompt_length
            ]
            length_rows.append(
                _accuracy_row(metadata, subset, prompt_length=prompt_length)
            )
        for position_percent in PASSKEY_POSITIONS_PERCENT:
            subset = [
                record
                for record in method_records
                if record["input"]["position_percent"] == position_percent
            ]
            position_rows.append(
                _accuracy_row(metadata, subset, position_percent=position_percent)
            )
        for prompt_length in PASSKEY_PROMPT_LENGTHS:
            for position_percent in PASSKEY_POSITIONS_PERCENT:
                subset = [
                    record
                    for record in method_records
                    if record["input"]["prompt_length"] == prompt_length
                    and record["input"]["position_percent"] == position_percent
                ]
                cell_rows.append(
                    _accuracy_row(
                        metadata,
                        subset,
                        prompt_length=prompt_length,
                        position_percent=position_percent,
                    )
                )

    paired_rows = _paired_comparisons(records, method_catalog)
    miss_rows = _miss_rows(records, method_catalog)
    return {
        "method_summary": method_rows,
        "length_summary": length_rows,
        "position_summary": position_rows,
        "cell_summary": cell_rows,
        "paired_comparisons": paired_rows,
        "exact_misses": miss_rows,
    }


def write_passkey_analysis_outputs(
    analysis_dir: str | Path,
    tables: dict[str, list[dict[str, Any]]],
    *,
    resolved_manifest: dict[str, Any],
    quality_summary: dict[str, Any],
    make_plots: bool = True,
) -> list[Path]:
    """Write deterministic tables, protocol metadata, summary, and figures."""

    destination = Path(analysis_dir)
    if destination.exists() and any(destination.iterdir()):
        raise PasskeyAnalysisError(f"analysis directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for name in (
        "method_summary",
        "length_summary",
        "position_summary",
        "cell_summary",
        "paired_comparisons",
        "exact_misses",
    ):
        rows = tables[name]
        output_paths.append(_write_jsonl(destination / f"{name}.jsonl", rows))
        output_paths.append(_write_csv(destination / f"{name}.csv", rows))

    protocol = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "input_passkey_schema_version": PASSKEY_SCHEMA_VERSION,
        "protocol_stage": "stage_b",
        "case_count": 300,
        "input_group_count": 60,
        "method_ids": list(STAGE_B_METHOD_IDS),
        "prompt_lengths": list(PASSKEY_PROMPT_LENGTHS),
        "positions_percent": list(PASSKEY_POSITIONS_PERCENT),
        "generated_keys": resolved_manifest["generated_keys"],
        "primary_metric": (
            "first standalone five-digit number in the generated continuation "
            "equals the target"
        ),
        "secondary_metric": "target string is contained in the generated continuation",
        "confidence_interval": (
            "two-sided descriptive 95% Wilson score interval for a binomial proportion"
        ),
        "confidence_interval_z": WILSON_Z_95,
        "paired_comparison": (
            "exact-outcome agreement and discordance on identical prompt/key inputs; "
            "no significance test"
        ),
        "sampling_boundary": (
            "five deterministic keys over a designed native-context grid; intervals "
            "do not establish population-level generalization"
        ),
        "runtime_role": (
            "diagnostic only; CAGE is fake quantization and has no fused variable-group kernel"
        ),
        "source_state": resolved_manifest["source_state"],
        "validated_quality_summary": {
            "completed_cases": quality_summary["completed_cases"],
            "failure_records": quality_summary["failure_records"],
            "exact_matches": quality_summary["exact_matches"],
            "contains_matches": quality_summary["contains_matches"],
            "completion_gate": quality_summary["completion_gate"],
            "quality_gate": quality_summary["quality_gate"],
        },
    }
    protocol_path = destination / "analysis_protocol.json"
    protocol_path.write_text(
        json.dumps(
            protocol,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_paths.append(protocol_path)

    summary_path = destination / "passkey_summary.md"
    summary_path.write_text(_render_markdown(tables), encoding="utf-8")
    output_paths.append(summary_path)

    if make_plots:
        output_paths.extend(_plot_passkey(destination, tables))
    return output_paths


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if type(successes) is not int or type(trials) is not int:
        raise PasskeyAnalysisError("Wilson inputs must be integers")
    if trials <= 0 or successes < 0 or successes > trials:
        raise PasskeyAnalysisError("Wilson inputs require 0 <= successes <= trials and trials > 0")
    proportion = successes / trials
    z2 = WILSON_Z_95**2
    denominator = 1 + z2 / trials
    center = (proportion + z2 / (2 * trials)) / denominator
    half_width = (
        WILSON_Z_95
        * math.sqrt(
            proportion * (1 - proportion) / trials + z2 / (4 * trials**2)
        )
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == trials else min(1.0, center + half_width)
    return lower, upper


def _validate_resolved_manifest(resolved: dict[str, Any]) -> None:
    expected_fields = {
        "model",
        "methods",
        "prompt_template_id",
        "filler_id",
        "prompt_lengths",
        "passkey_positions_percent",
        "key_generation",
        "generation",
        "output_dir",
        "protocol_stage",
        "generated_keys",
        "source_state",
        "expanded_cases",
    }
    if set(resolved) != expected_fields:
        raise PasskeyAnalysisError(
            f"resolved manifest fields differ: expected={sorted(expected_fields)}, "
            f"actual={sorted(resolved)}"
        )
    expected_methods = [
        resolve_method(method, index)
        for index, method in enumerate(STAGE_B_RAW_METHODS)
    ]
    if resolved["protocol_stage"] != "stage_b":
        raise PasskeyAnalysisError("resolved protocol_stage must equal 'stage_b'")
    if resolved["methods"] != expected_methods:
        raise PasskeyAnalysisError("resolved methods differ from the declared Stage-B matrix")
    model = resolved["model"]
    if (
        not isinstance(model, dict)
        or model.get("dtype") != "float16"
        or model.get("device") != "cuda"
        or model.get("max_position_embeddings") != 4096
        or not isinstance(model.get("reference"), str)
        or not model["reference"]
    ):
        raise PasskeyAnalysisError("resolved model differs from the Stage-B protocol")
    if tuple(resolved["prompt_lengths"]) != PASSKEY_PROMPT_LENGTHS:
        raise PasskeyAnalysisError("resolved prompt lengths differ from the protocol")
    if tuple(resolved["passkey_positions_percent"]) != PASSKEY_POSITIONS_PERCENT:
        raise PasskeyAnalysisError("resolved positions differ from the protocol")
    if resolved["key_generation"] != {"seed": PASSKEY_KEY_SEED, "count": 5}:
        raise PasskeyAnalysisError("resolved key generation differs from the full protocol")
    if resolved["generated_keys"] != generate_passkeys(PASSKEY_KEY_SEED, 5):
        raise PasskeyAnalysisError("resolved generated keys are inconsistent")
    if resolved["generation"] != {
        "max_new_tokens": PASSKEY_MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "seed": PASSKEY_GENERATION_SEED,
    }:
        raise PasskeyAnalysisError("resolved generation differs from the protocol")
    source_state = resolved["source_state"]
    if not isinstance(source_state, dict) or source_state.get("dirty") is not False:
        raise PasskeyAnalysisError("resolved source state must be clean")
    if not isinstance(resolved["expanded_cases"], list):
        raise PasskeyAnalysisError("resolved expanded_cases must be a list")


def _validate_coverage(records: Sequence[dict[str, Any]]) -> None:
    method_counts = Counter(record["method"]["id"] for record in records)
    if method_counts != Counter({method_id: 60 for method_id in STAGE_B_METHOD_IDS}):
        raise PasskeyAnalysisError(f"method coverage differs: {dict(method_counts)}")
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        input_record = record["input"]
        groups[
            (
                input_record["prompt_length"],
                input_record["position_percent"],
                input_record["key_index"],
            )
        ].append(record)
    if len(groups) != 60:
        raise PasskeyAnalysisError(f"input group count must equal 60, got {len(groups)}")
    for identity, group in groups.items():
        if len(group) != 5 or {record["method"]["id"] for record in group} != set(
            STAGE_B_METHOD_IDS
        ):
            raise PasskeyAnalysisError(f"input group {identity} lacks five-method coverage")
        for field in (
            "target",
            "prompt_ids_sha256",
            "statement_token_start",
            "statement_token_end",
        ):
            if len({record["input"][field] for record in group}) != 1:
                raise PasskeyAnalysisError(f"input group {identity} differs in {field}")


def _validate_quality_summary(
    quality: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    resolved: dict[str, Any],
) -> None:
    if quality.get("schema_version") != PASSKEY_SCHEMA_VERSION:
        raise PasskeyAnalysisError("quality summary schema_version is invalid")
    if quality.get("protocol_stage") != "stage_b":
        raise PasskeyAnalysisError("quality summary protocol_stage is invalid")
    if quality.get("expanded_cases") != 300 or quality.get("completed_cases") != 300:
        raise PasskeyAnalysisError("quality summary case counts are invalid")
    if quality.get("failure_records") != 0:
        raise PasskeyAnalysisError("quality summary contains failures")
    if quality.get("completion_gate") != "PASS":
        raise PasskeyAnalysisError("quality summary completion gate did not pass")
    if quality.get("quality_gate") != "NOT_APPLICABLE":
        raise PasskeyAnalysisError("Stage-B quality gate must be NOT_APPLICABLE")

    methods = tables["method_summary"]
    exact = sum(row["exact_matches"] for row in methods)
    contains = sum(row["contains_matches"] for row in methods)
    if quality.get("exact_matches") != exact or quality.get("contains_matches") != contains:
        raise PasskeyAnalysisError("quality summary aggregate metrics are inconsistent")
    expected_accuracy = exact / 300
    if not math.isclose(
        quality.get("exact_match_accuracy", -1), expected_accuracy, rel_tol=0, abs_tol=1e-15
    ):
        raise PasskeyAnalysisError("quality summary exact_match_accuracy is inconsistent")

    stored_methods = quality.get("methods")
    if not isinstance(stored_methods, list) or len(stored_methods) != 5:
        raise PasskeyAnalysisError("quality summary methods must contain five rows")
    stored_by_id = {row.get("method_id"): row for row in stored_methods}
    for row in methods:
        stored = stored_by_id.get(row["method_id"])
        if (
            not isinstance(stored, dict)
            or stored.get("completed_cases") != row["trials"]
            or stored.get("exact_matches") != row["exact_matches"]
            or stored.get("contains_matches") != row["contains_matches"]
        ):
            raise PasskeyAnalysisError(
                f"quality summary method {row['method_id']} is inconsistent"
            )
    if resolved["source_state"].get("dirty") is not False:
        raise PasskeyAnalysisError("resolved source state is dirty")


def _accuracy_row(
    metadata: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    prompt_length: int | None = None,
    position_percent: int | None = None,
) -> dict[str, Any]:
    trials = len(records)
    if trials <= 0:
        raise PasskeyAnalysisError("accuracy stratum is empty")
    exact = sum(bool(record["generation"]["exact_match"]) for record in records)
    contains = sum(bool(record["generation"]["contains_target"]) for record in records)
    exact_low, exact_high = wilson_interval(exact, trials)
    contains_low, contains_high = wilson_interval(contains, trials)
    row = {
        **metadata,
        "trials": trials,
        "exact_matches": exact,
        "exact_errors": trials - exact,
        "exact_accuracy": exact / trials,
        "exact_ci95_low": exact_low,
        "exact_ci95_high": exact_high,
        "contains_matches": contains,
        "contains_errors": trials - contains,
        "contains_accuracy": contains / trials,
        "contains_ci95_low": contains_low,
        "contains_ci95_high": contains_high,
        "exact_miss_input_groups": len({
            _input_identity(record)
            for record in records
            if not record["generation"]["exact_match"]
        }),
    }
    if prompt_length is not None:
        row["prompt_length"] = prompt_length
    if position_percent is not None:
        row["position_percent"] = position_percent
    return row


def _paired_comparisons(
    records: Sequence[dict[str, Any]],
    method_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        groups[_input_identity(record)][record["method"]["id"]] = record
    rows = []
    for method_a, method_b in itertools.combinations(STAGE_B_METHOD_IDS, 2):
        both_correct = a_only = b_only = both_wrong = 0
        discordant = []
        for identity in sorted(groups):
            exact_a = bool(groups[identity][method_a]["generation"]["exact_match"])
            exact_b = bool(groups[identity][method_b]["generation"]["exact_match"])
            if exact_a and exact_b:
                both_correct += 1
            elif exact_a:
                a_only += 1
                discordant.append(identity)
            elif exact_b:
                b_only += 1
                discordant.append(identity)
            else:
                both_wrong += 1
        trials = len(groups)
        rows.append({
            "method_a_id": method_a,
            "method_a_order": method_catalog[method_a]["method_order"],
            "method_b_id": method_b,
            "method_b_order": method_catalog[method_b]["method_order"],
            "paired_trials": trials,
            "both_exact": both_correct,
            "method_a_only_exact": a_only,
            "method_b_only_exact": b_only,
            "neither_exact": both_wrong,
            "exact_outcome_agreement": (both_correct + both_wrong) / trials,
            "exact_accuracy_difference_b_minus_a": (b_only - a_only) / trials,
            "discordant_input_count": a_only + b_only,
            "discordant_inputs_json": _canonical_json([
                {
                    "prompt_length": identity[0],
                    "position_percent": identity[1],
                    "key_index": identity[2],
                }
                for identity in discordant
            ]),
        })
    return rows


def _miss_rows(
    records: Sequence[dict[str, Any]],
    method_catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        generation = record["generation"]
        if generation["exact_match"]:
            continue
        input_record = record["input"]
        rows.append({
            "method_id": record["method"]["id"],
            "method_order": method_catalog[record["method"]["id"]]["method_order"],
            "case_id": record["case_id"],
            "prompt_length": input_record["prompt_length"],
            "position_percent": input_record["position_percent"],
            "key_index": input_record["key_index"],
            "target": input_record["target"],
            "first_five_digit": generation["first_five_digit"],
            "contains_target": generation["contains_target"],
            "generated_token_count": generation["generated_token_count"],
            "response_text": generation["response_text"],
            "prompt_ids_sha256": input_record["prompt_ids_sha256"],
        })
    return sorted(
        rows,
        key=lambda row: (
            row["prompt_length"],
            row["position_percent"],
            row["key_index"],
            row["method_order"],
        ),
    )


def _render_markdown(tables: dict[str, list[dict[str, Any]]]) -> str:
    methods = tables["method_summary"]
    misses = tables["exact_misses"]
    paired = tables["paired_comparisons"]
    lines = [
        "# CAGE-KV Stage-B passkey pilot",
        "",
        "Primary metric: the first standalone five-digit number in the continuation equals the target. "
        "Containment is secondary.",
        "",
        "## Method summary",
        "",
        "| Method | Exact | Exact accuracy | Descriptive Wilson 95% CI | Containment |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in methods:
        lines.append(
            f"| {row['method_id']} | {row['exact_matches']}/{row['trials']} | "
            f"{_percent(row['exact_accuracy'])} | "
            f"[{_percent(row['exact_ci95_low'])}, {_percent(row['exact_ci95_high'])}] | "
            f"{row['contains_matches']}/{row['trials']} |"
        )

    lines.extend([
        "",
        "## Exact misses",
        "",
    ])
    if misses:
        lines.extend([
            "| Method | Length | Position | Key index | Target | Parsed | Response |",
            "|---|---:|---:|---:|---:|---|---|",
        ])
        for row in misses:
            parsed = row["first_five_digit"] if row["first_five_digit"] is not None else "None"
            response = json.dumps(row["response_text"], ensure_ascii=False).replace("|", "\\|")
            lines.append(
                f"| {row['method_id']} | {row['prompt_length']} | "
                f"{row['position_percent']}% | {row['key_index']} | {row['target']} | "
                f"{parsed} | `{response}` |"
            )
    else:
        lines.append("No exact misses.")

    fp16_pairs = [row for row in paired if row["method_a_id"] == "fp16"]
    lines.extend([
        "",
        "## Paired comparison with FP16",
        "",
        "| Method | Both exact | FP16 only exact | Method only exact | Neither exact | Difference vs FP16 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in fp16_pairs:
        lines.append(
            f"| {row['method_b_id']} | {row['both_exact']} | "
            f"{row['method_a_only_exact']} | {row['method_b_only_exact']} | "
            f"{row['neither_exact']} | "
            f"{_signed_percentage_points(row['exact_accuracy_difference_b_minus_a'])} |"
        )

    miss_groups = {
        (row["prompt_length"], row["position_percent"], row["key_index"], row["target"])
        for row in misses
    }
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        f"The {len(misses)} method-level exact misses occupy {len(miss_groups)} unique input group(s). "
        "Cell and paired tables retain that dependence rather than treating coincident misses as "
        "independent prompt draws.",
        "",
        "Wilson intervals are descriptive binomial intervals over this designed 60-input grid. "
        "The five deterministic keys are not a random population sample, so these intervals and "
        "paired differences do not establish statistical significance or generalization.",
        "",
        "CAGE uses the fake-quant prototype. Runtime and CUDA peaks are diagnostics and are not "
        "compressed-kernel performance claims. This pilot does not establish QA, LongBench, or "
        "perplexity quality.",
    ])
    return "\n".join(lines) + "\n"


def _plot_passkey(
    destination: Path,
    tables: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise PasskeyAnalysisError(
            "matplotlib is required for plots; install it or pass --no-plots"
        ) from error

    methods = tables["method_summary"]
    cells = tables["cell_summary"]
    figure, (accuracy_axis, heatmap_axis) = plt.subplots(
        1,
        2,
        figsize=(14.2, 6.2),
        gridspec_kw={"width_ratios": [0.9, 1.65]},
    )
    labels = [row["method_id"] for row in methods]
    y_positions = list(range(len(methods)))
    exact_values = [row["exact_accuracy"] for row in methods]
    lower_errors = [
        row["exact_accuracy"] - row["exact_ci95_low"] for row in methods
    ]
    upper_errors = [
        row["exact_ci95_high"] - row["exact_accuracy"] for row in methods
    ]
    accuracy_axis.errorbar(
        exact_values,
        y_positions,
        xerr=[lower_errors, upper_errors],
        fmt="o",
        color="#2f6f9f",
        ecolor="#6f8fa8",
        capsize=4,
        markersize=7,
        linewidth=1.2,
        label="Primary exact",
    )
    accuracy_axis.scatter(
        [row["contains_accuracy"] for row in methods],
        y_positions,
        marker="D",
        s=35,
        facecolors="none",
        edgecolors="#b05a2a",
        linewidths=1.2,
        label="Target containment",
        zorder=3,
    )
    for y_position, row in zip(y_positions, methods):
        accuracy_axis.annotate(
            f"{row['exact_matches']}/{row['trials']}",
            (row["exact_accuracy"], y_position),
            xytext=(-7, 8),
            textcoords="offset points",
            ha="right",
            fontsize=8,
        )
    minimum = min(row["exact_ci95_low"] for row in methods)
    accuracy_axis.set_xlim(max(0.0, minimum - 0.025), 1.012)
    accuracy_axis.set_yticks(y_positions, labels)
    accuracy_axis.invert_yaxis()
    accuracy_axis.set_xlabel("Accuracy")
    accuracy_axis.set_title("Overall accuracy (60 cases per method)")
    accuracy_axis.xaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
    )
    accuracy_axis.grid(True, axis="x", alpha=0.25)
    handles, legend_labels = accuracy_axis.get_legend_handles_labels()

    cell_lookup = {
        (row["method_id"], row["prompt_length"], row["position_percent"]): row
        for row in cells
    }
    columns = [
        (length, position)
        for length in PASSKEY_PROMPT_LENGTHS
        for position in PASSKEY_POSITIONS_PERCENT
    ]
    matrix = [
        [
            cell_lookup[(method_id, length, position)]["exact_matches"]
            for length, position in columns
        ]
        for method_id in STAGE_B_METHOD_IDS
    ]
    image = heatmap_axis.imshow(matrix, cmap="YlGn", vmin=0, vmax=5, aspect="auto")
    heatmap_axis.set_xticks(
        range(len(columns)),
        [f"{length}\n{position}%" for length, position in columns],
        fontsize=7.5,
    )
    heatmap_axis.set_yticks(range(len(STAGE_B_METHOD_IDS)), STAGE_B_METHOD_IDS)
    heatmap_axis.set_xlabel("Prompt length and statement position")
    heatmap_axis.set_title("Primary exact matches per cell (out of 5)")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            heatmap_axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="#ffffff" if value >= 3 else "#111111",
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=heatmap_axis, fraction=0.035, pad=0.025)
    colorbar.set_label("Exact matches")
    colorbar.set_ticks(range(0, 6))

    figure.suptitle(
        "CAGE-KV Stage-B native-context passkey pilot\n"
        "Wilson 95% intervals are descriptive over five deterministic keys",
        fontsize=14,
    )
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.27, 0.015),
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    png_path = destination / "passkey_accuracy.png"
    pdf_path = destination / "passkey_accuracy.pdf"
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path)
    plt.close(figure)
    return [png_path, pdf_path]


def _input_identity(record: dict[str, Any]) -> tuple[int, int, int]:
    input_record = record["input"]
    return (
        input_record["prompt_length"],
        input_record["position_percent"],
        input_record["key_index"],
    )


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PasskeyAnalysisError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PasskeyAnalysisError(f"{name} {path} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise PasskeyAnalysisError(f"blank line {line_number} in {path}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PasskeyAnalysisError(
                        f"row {line_number} in {path} is not an object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise PasskeyAnalysisError(f"cannot read JSONL {path}: {error}") from error
    return rows


def _read_csv_ids(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise PasskeyAnalysisError(f"cannot read CSV {path}: {error}") from error
    if not rows or "case_id" not in rows[0]:
        raise PasskeyAnalysisError(f"CSV {path} is empty or lacks case_id")
    return [row["case_id"] for row in rows]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    fields = _ordered_fields(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _ordered_fields(rows: Sequence[dict[str, Any]]) -> list[str]:
    fields = sorted({field for row in rows for field in row})
    preferred = [
        "method_id",
        "method_order",
        "method",
        "prompt_length",
        "position_percent",
        "trials",
        "exact_matches",
        "exact_accuracy",
        "exact_ci95_low",
        "exact_ci95_high",
        "contains_matches",
        "contains_accuracy",
    ]
    ordered = [field for field in preferred if field in fields]
    return ordered + [field for field in fields if field not in ordered]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _id_set_error(name: str, expected: set[str], actual: set[str]) -> str:
    return (
        f"{name} differ from expected IDs: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}"
    )


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _signed_percentage_points(value: float) -> str:
    return f"{100 * value:+.1f} pp"


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "PasskeyAnalysisError",
    "aggregate_passkey_results",
    "load_completed_passkey_matrix",
    "wilson_interval",
    "write_passkey_analysis_outputs",
]
