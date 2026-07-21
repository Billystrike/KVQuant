"""Generate paper-facing CAGE memory–perturbation tables and Pareto figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cage_pareto import (
    ParetoAnalysisError,
    aggregate_points,
    build_trends,
    load_completed_matrix,
    write_analysis_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Frozen matrix output containing manifest.resolved.json, runs, layers, and summary.",
    )
    parser.add_argument(
        "--analysis-dir",
        required=True,
        help="New or empty destination for aggregate tables, Pareto tables, and figures.",
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
    resolved, runs = load_completed_matrix(results_dir)
    points = aggregate_points(runs, resolved)
    trends = build_trends(points)
    outputs = write_analysis_outputs(
        analysis_dir,
        points,
        trends,
        resolved_manifest=resolved,
        run_count=len(runs),
        make_plots=make_plots,
    )
    pareto_count = sum(bool(point["is_pareto_global"]) for point in points)
    return {
        "validated_runs": len(runs),
        "aggregate_points": len(points),
        "trend_rows": len(trends),
        "pareto_points": pareto_count,
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
    except (OSError, ParetoAnalysisError, ValueError) as error:
        print(f"analysis error: {error}", file=sys.stderr)
        return 2

    for key, value in result.items():
        print(f"{key}={value}")
    print("ANALYSIS_RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
