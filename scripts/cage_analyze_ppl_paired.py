"""Analyze the pre-registered CAGE 50-anchor paired PPL study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cage_ppl_paired_analysis import (
    PairedPPLAnalysisError,
    aggregate_paired_results,
    load_completed_paired_matrix,
    write_paired_analysis_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Frozen 1,000-case paired PPL output.",
    )
    parser.add_argument(
        "--analysis-dir",
        required=True,
        help="New or empty destination for paired tables and figures.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write validated tables without importing matplotlib.",
    )
    return parser.parse_args(argv)


def run_analysis(
    results_dir: str | Path,
    analysis_dir: str | Path,
    *,
    make_plots: bool = True,
) -> dict[str, int]:
    resolved, records, quality = load_completed_paired_matrix(results_dir)
    tables = aggregate_paired_results(records, resolved)
    outputs = write_paired_analysis_outputs(
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
    except (OSError, PairedPPLAnalysisError, ValueError) as error:
        print(f"paired PPL analysis error: {error}", file=sys.stderr)
        return 2

    for key, value in result.items():
        print(f"{key}={value}")
    print("PAIRED_PPL_ANALYSIS_RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
