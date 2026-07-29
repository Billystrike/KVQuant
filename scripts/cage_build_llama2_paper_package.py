"""Build the final frozen Llama-2-7B CAGE-KV paper evidence package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cage_llama2_paper_package import (
    Llama2PaperPackageError,
    build_final_paper_tables,
    load_final_paper_inputs,
    write_final_paper_package,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-evidence-dir", required=True)
    parser.add_argument("--local-ablation-analysis-dir", required=True)
    parser.add_argument("--ppl-mechanism-analysis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Write validated paper tables and prose without matplotlib.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_final_paper_inputs(
        args.paper_evidence_dir,
        args.local_ablation_analysis_dir,
        args.ppl_mechanism_analysis_dir,
    )
    tables = build_final_paper_tables(inputs)
    outputs = write_final_paper_package(
        args.output_dir,
        inputs,
        tables,
        make_plots=not args.no_plots,
    )
    exact = [
        row for row in tables["mechanism_length_evidence"]
        if row["cross_metric_join_status"] == "exact_prompt_length"
    ]
    return {
        "operating_point_rows": len(tables["operating_point_evidence"]),
        "comparison_summary_rows": len(tables["comparison_summary"]),
        "mechanism_length_rows": len(tables["mechanism_length_evidence"]),
        "mechanism_summary_rows": len(tables["mechanism_summary"]),
        "factorial_evidence_rows": len(tables["factorial_evidence"]),
        "claim_register_rows": len(tables["claim_register"]),
        "exact_cross_metric_join_rows": len(exact),
        "output_files": len(outputs),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, Llama2PaperPackageError) as error:
        print(f"Llama-2 paper package error: {error}", file=sys.stderr)
        return 2
    for key, value in result.items():
        print(f"{key}={value}")
    print("LLAMA2_FINAL_PAPER_PACKAGE_RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
