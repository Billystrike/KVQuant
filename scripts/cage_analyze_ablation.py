"""Analyze the frozen Llama-2-7B CAGE mechanism ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cage_ablation_analysis import (
    AblationAnalysisError,
    aggregate_ablation_results,
    load_completed_ablation_matrix,
    write_ablation_analysis_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Frozen 144-run ablation output.")
    parser.add_argument("--analysis-dir", required=True, help="New or empty analysis destination.")
    parser.add_argument("--no-plots", action="store_true", help="Write tables without matplotlib.")
    return parser.parse_args(argv)


def run_analysis(
    results_dir: str | Path,
    analysis_dir: str | Path,
    *,
    make_plots: bool = True,
) -> dict[str, Any]:
    resolved, runs = load_completed_ablation_matrix(results_dir)
    tables = aggregate_ablation_results(runs, resolved)
    outputs = write_ablation_analysis_outputs(
        analysis_dir,
        tables,
        resolved_manifest=resolved,
        run_count=len(runs),
        make_plots=make_plots,
    )
    decisions = tables["advancement_decisions"]
    return {
        "validated_runs": len(runs),
        "aggregate_points": len(tables["aggregate_points"]),
        "sample_contrasts": len(tables["sample_contrasts"]),
        "paired_contrasts": len(tables["paired_contrasts"]),
        "factorial_rows": len(tables["factorial_decomposition"]),
        "advancement_rows": len(decisions),
        "external_baseline_rows": len(tables["external_baselines"]),
        "advanced_side_only_variants": sum(row["advance_to_ppl_ablation"] for row in decisions),
        "output_files": len(outputs),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_analysis(
            args.results_dir,
            args.analysis_dir,
            make_plots=not args.no_plots,
        )
    except (OSError, AblationAnalysisError, ValueError) as error:
        print(f"ablation analysis error: {error}", file=sys.stderr)
        return 2
    for key, value in result.items():
        print(f"{key}={value}")
    print("CAGE_ABLATION_ANALYSIS_RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
