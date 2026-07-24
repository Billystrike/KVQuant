"""Validate and analyze the CAGE cache-conditioned continuation PPL pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cage_ppl_analysis import (
    PPLAnalysisError,
    aggregate_ppl_results,
    load_completed_ppl_matrix,
    load_pareto_analysis,
    write_ppl_analysis_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Frozen 200-case PPL output containing resolved manifest, cases, and summaries.",
    )
    parser.add_argument(
        "--analysis-dir",
        required=True,
        help="New or empty destination for PPL tables, summary, and figures.",
    )
    parser.add_argument(
        "--pareto-analysis-dir",
        help=(
            "Optional frozen core Pareto analysis containing aggregate_points.jsonl; "
            "enables an exact-prompt-length joint selection table."
        ),
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
    pareto_analysis_dir: str | Path | None = None,
    make_plots: bool = True,
) -> dict[str, int]:
    resolved, records, quality = load_completed_ppl_matrix(results_dir)
    tables = aggregate_ppl_results(records, resolved)
    pareto_protocol = None
    pareto_rows = None
    if pareto_analysis_dir is not None:
        pareto_protocol, pareto_rows = load_pareto_analysis(pareto_analysis_dir)
    outputs = write_ppl_analysis_outputs(
        analysis_dir,
        tables,
        resolved_manifest=resolved,
        quality_summary=quality,
        pareto_protocol=pareto_protocol,
        pareto_rows=pareto_rows,
        make_plots=make_plots,
    )
    joint_rows = len(tables["length_summary"]) if pareto_rows is not None else 0
    return {
        "validated_cases": len(records),
        "method_rows": len(tables["method_summary"]),
        "length_rows": len(tables["length_summary"]),
        "anchor_rows": len(tables["anchor_summary"]),
        "paired_rows": len(tables["paired_vs_fp16"]),
        "joint_rows": joint_rows,
        "joint_exact_length_rows": 30 if pareto_rows is not None else 0,
        "output_files": len(outputs),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_analysis(
            args.results_dir,
            args.analysis_dir,
            pareto_analysis_dir=args.pareto_analysis_dir,
            make_plots=not args.no_plots,
        )
    except (OSError, PPLAnalysisError, ValueError) as error:
        print(f"PPL analysis error: {error}", file=sys.stderr)
        return 2

    for key, value in result.items():
        print(f"{key}={value}")
    print("PPL_ANALYSIS_RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
