"""Build the Llama-2-7B cross-experiment CAGE-KV paper evidence ledger."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cage_paper_evidence import (
    PaperEvidenceError,
    build_paper_evidence,
    load_evidence_inputs,
    write_paper_evidence_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto-analysis-dir", required=True)
    parser.add_argument("--paired-ppl-analysis-dir", required=True)
    parser.add_argument("--passkey-analysis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Write validated tables and claim ledger without matplotlib.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_evidence_inputs(
        args.pareto_analysis_dir,
        args.paired_ppl_analysis_dir,
        args.passkey_analysis_dir,
    )
    tables = build_paper_evidence(inputs)
    outputs = write_paper_evidence_outputs(
        args.output_dir,
        inputs,
        tables,
        make_plots=not args.no_plots,
    )
    exact = [
        row for row in tables["operating_point_evidence"]
        if row["ppl_join_status"] == "exact_prompt_length"
    ]
    return {
        "operating_point_rows": len(tables["operating_point_evidence"]),
        "comparison_summary_rows": len(tables["comparison_summary"]),
        "exact_memory_ppl_join_rows": len(exact),
        "unmatched_native_context_rows": (
            len(tables["operating_point_evidence"]) - len(exact)
        ),
        "output_files": len(outputs),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, PaperEvidenceError) as error:
        print(f"paper evidence error: {error}", file=sys.stderr)
        return 2
    for key, value in result.items():
        print(f"{key}={value}")
    print("LLAMA2_PAPER_EVIDENCE_RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
