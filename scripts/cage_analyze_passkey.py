"""Validate and summarize the CAGE Stage-B passkey pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cage_passkey_analysis import (
    PasskeyAnalysisError,
    aggregate_passkey_results,
    load_completed_passkey_matrix,
    write_passkey_analysis_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Frozen Stage-B output containing resolved manifest, cases, and summaries.",
    )
    parser.add_argument(
        "--analysis-dir",
        required=True,
        help="New or empty destination for passkey tables, summary, and figures.",
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
    resolved, records, quality = load_completed_passkey_matrix(results_dir)
    tables = aggregate_passkey_results(records, resolved)
    outputs = write_passkey_analysis_outputs(
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
        "position_rows": len(tables["position_summary"]),
        "cell_rows": len(tables["cell_summary"]),
        "paired_rows": len(tables["paired_comparisons"]),
        "exact_miss_rows": len(tables["exact_misses"]),
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
    except (OSError, PasskeyAnalysisError, ValueError) as error:
        print(f"passkey analysis error: {error}", file=sys.stderr)
        return 2

    for key, value in result.items():
        print(f"{key}={value}")
    print("PASSKEY_ANALYSIS_RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
