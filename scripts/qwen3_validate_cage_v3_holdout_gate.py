#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qwen3_run_cage_v3_holdout import _validate_full_gate
from utils.qwen3_cage_v3_holdout import load_holdout_execution
from utils.qwen3_cage_v3_protocol import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen CAGE-v3 joint holdout acceptance gate")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--acceptance-gate", type=Path, required=True)
    args = parser.parse_args()
    execution, execution_sha256, _, _, _, decision = load_holdout_execution(
        args.execution.resolve(),
        repo_root=REPO_ROOT,
        verify_artifacts=True,
    )
    gate_path = args.acceptance_gate.resolve()
    gate = _validate_full_gate(
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
        "screen_decision_sha256": execution["screen_decision"]["sha256"],
        "selected_family_id": decision["decision"]["selected_family_id"],
        "accepted_commit": gate["acceptance_source"]["git_commit"],
        "partitions": {
            partition: {
                "case_count_per_repeat": record["case_count"],
                "case_ids_sha256": record["case_ids_sha256"],
                "scientific_payload_sha256": record["scientific_payload_sha256"],
                "repeat_a_log_sha256": record["repeat_a"]["execution_log_sha256"],
                "repeat_b_log_sha256": record["repeat_b"]["execution_log_sha256"],
                "comparison_sha256": record["comparison"]["sha256"],
            }
            for partition, record in gate["partitions"].items()
        },
        "holdout_full_authorization": gate["holdout_full_authorization"],
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
