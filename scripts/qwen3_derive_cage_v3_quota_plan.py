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

from utils.qwen3_cage_v3_calibration import (
    validate_and_derive_quota_plan,
    validate_receipt,
)
from utils.qwen3_cage_v3_execution import (
    expand_calibration_cases,
    load_calibration_execution,
    load_json,
)
from utils.qwen3_cage_v3_protocol import file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen CAGE-v3 calibration artifacts and derive six layer quota plans"
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = _parse_args()
    receipt_path = args.receipt.resolve()
    receipt = load_json(receipt_path)
    receipt_sha256 = file_sha256(receipt_path)
    execution_path = Path(receipt["execution"]["path"])
    if not execution_path.is_absolute():
        execution_path = REPO_ROOT / execution_path
    execution, execution_sha256, protocol, manifest = load_calibration_execution(
        execution_path,
        repo_root=REPO_ROOT,
        verify_artifacts=True,
    )
    validate_receipt(
        receipt,
        repo_root=REPO_ROOT,
        protocol_sha256=execution["protocol"]["sha256"],
        execution_sha256=execution_sha256,
    )
    gate_path = Path(receipt["acceptance_gate"]["path"])
    if not gate_path.is_absolute():
        gate_path = REPO_ROOT / gate_path
    expected_cases = expand_calibration_cases(
        execution=execution,
        execution_sha256=execution_sha256,
        protocol=protocol,
        manifest=manifest,
        stage="calibration_full",
    )
    plan = validate_and_derive_quota_plan(
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        expected_cases=expected_cases,
        protocol_sha256=execution["protocol"]["sha256"],
        execution_sha256=execution_sha256,
        gate_sha256=file_sha256(gate_path),
    )
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing quota plan: {output_path}")
    _write_atomic(output_path, plan)
    report = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "receipt_sha256": receipt_sha256,
        "calibration_scientific_payload_sha256": plan["calibration_scientific_payload_sha256"],
        "plan_count": len(plan["plans"]),
        "plans": [
            {
                "family_id": row["family_id"],
                "prompt_length": row["prompt_length"],
                "ranked_layer_indices": row["ranked_layer_indices"],
                "layer_two_bit_channel_quotas": row["layer_two_bit_channel_quotas"],
                "quota_total": row["quota_total"],
            }
            for row in plan["plans"]
        ],
        "screen_metrics_consumed": False,
        "holdout_metrics_consumed": False,
        "reserved_unseen_metrics_consumed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
