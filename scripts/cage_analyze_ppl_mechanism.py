"""Analyze the frozen 2,000-case paired PPL mechanism ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cage_ppl_mechanism_analysis import (
    MechanismPPLAnalysisError,
    aggregate_mechanism_results,
    load_completed_mechanism_matrix,
    write_mechanism_analysis_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Frozen 2,000-case output.")
    parser.add_argument("--analysis-dir", required=True, help="New or empty analysis destination.")
    parser.add_argument("--no-plots", action="store_true", help="Write tables without matplotlib.")
    return parser.parse_args(argv)


def run_analysis(
    results_dir: str | Path,
    analysis_dir: str | Path,
    *,
    make_plots: bool = True,
) -> dict[str, Any]:
    resolved, records, quality = load_completed_mechanism_matrix(results_dir)
    tables = aggregate_mechanism_results(records, resolved)
    outputs = write_mechanism_analysis_outputs(
        analysis_dir,
        tables,
        resolved_manifest=resolved,
        quality_summary=quality,
        make_plots=make_plots,
    )
    return {
        "validated_cases": len(records),
        "method_rows": len(tables["method_summary"]),
        "length_rows": len(tables["length_summary"]),
        "paired_rows": len(tables["paired_comparisons"]),
        "anchor_delta_rows": len(tables["anchor_deltas"]),
        "factorial_rows": len(tables["factorial_summary"]),
        "factorial_anchor_rows": len(tables["factorial_anchor_effects"]),
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
    except (OSError, MechanismPPLAnalysisError, ValueError) as error:
        print(f"PPL mechanism analysis error: {error}", file=sys.stderr)
        return 2
    for key, value in result.items():
        print(f"{key}={value}")
    print("PPL_MECHANISM_ANALYSIS_RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
