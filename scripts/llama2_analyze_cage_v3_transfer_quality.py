#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llama2_cage_v3_transfer_quality_analysis import (
    build_analysis,
    load_frozen_records,
    validate_results_receipt,
    write_analysis_outputs,
)
from utils.qwen3_cage_v4_data import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the frozen 600-case Llama-2 CAGE-v3 transfer-quality study")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_results_receipt(receipt, receipt_path=receipt_path)
    records, protocol = load_frozen_records(receipt, repo_root=REPO_ROOT)
    analysis = build_analysis(receipt_sha256=file_sha256(receipt_path), protocol=protocol, records=records)
    outputs = write_analysis_outputs(analysis, args.output_dir.resolve())
    manifest = {
        "schema_version": 1,
        "analysis_id": analysis["analysis_id"],
        "status": "pass",
        "claim_eligible": False,
        "receipt_sha256": analysis["receipt_sha256"],
        "case_count": analysis["case_count"],
        "target_token_count": analysis["target_token_count"],
        "promotion_pass": analysis["promotion_pass"],
        "candidate_outcome": analysis["candidate_outcome"],
        "outputs": outputs,
    }
    destination = args.output_dir.resolve() / "llama2_cage_v3_transfer_quality_analysis_manifest_v1.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
