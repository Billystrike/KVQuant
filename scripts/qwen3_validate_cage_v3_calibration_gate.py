#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qwen3_run_cage_v3_calibration import _validate_full_acceptance_gate
from utils.qwen3_cage_v3_execution import load_calibration_execution
from utils.qwen3_cage_v3_protocol import file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the frozen CAGE-v3 calibration acceptance gate and server artifacts"
    )
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--acceptance-gate", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    execution, execution_sha256, _, _ = load_calibration_execution(
        args.execution.resolve(),
        repo_root=REPO_ROOT,
        verify_artifacts=True,
    )
    gate_path = args.acceptance_gate.resolve()
    gate = _validate_full_acceptance_gate(
        gate_path,
        execution=execution,
        execution_sha256=execution_sha256,
    )
    report = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "gate_id": gate["gate_id"],
        "gate_sha256": file_sha256(gate_path),
        "execution_sha256": execution_sha256,
        "protocol_sha256": gate["protocol_sha256"],
        "manifest_sha256": gate["manifest_sha256"],
        "accepted_commit": gate["acceptance_source"]["git_commit"],
        "repeat_case_count": 6,
        "repeat_scientific_payload_sha256": gate["comparison"]["scientific_payload_sha256"],
        "calibration_full_authorization": gate["calibration_full_authorization"],
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
