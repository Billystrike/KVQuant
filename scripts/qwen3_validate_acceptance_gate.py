#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_formal import (
    Qwen3FormalError,
    expand_partition_cases,
    file_sha256,
    load_acceptance_gate,
    load_execution_config,
    load_formal_protocol,
    validate_input_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the frozen Qwen3 full-execution acceptance gate"
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-gate", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    protocol, protocol_sha256 = load_formal_protocol(args.protocol.resolve())
    execution, execution_sha256 = load_execution_config(
        args.execution_config.resolve(),
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    input_manifest_sha256 = file_sha256(args.input_manifest.resolve())
    if input_manifest_sha256 != execution["input_manifest"]["sha256"]:
        raise Qwen3FormalError("input manifest SHA-256 differs from execution freeze")
    try:
        input_manifest = json.loads(args.input_manifest.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3FormalError(f"cannot load input manifest: {error}") from error
    validate_input_manifest(
        input_manifest,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    gate, gate_sha256 = load_acceptance_gate(
        args.acceptance_gate.resolve(),
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        execution=execution,
        execution_sha256=execution_sha256,
        input_manifest_sha256=input_manifest_sha256,
    )
    full_cases = {
        partition: expand_partition_cases(
            protocol=protocol,
            execution=execution,
            execution_sha256=execution_sha256,
            input_manifest=input_manifest,
            partition=partition,
            stage="full",
        )
        for partition in ("cage_qwen3", "kitty_qwen3")
    }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "gate_id": gate["gate_id"],
        "gate_sha256": gate_sha256,
        "protocol_sha256": protocol_sha256,
        "execution_sha256": execution_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "partitions": {
            name: {
                "case_count": record["expected_cases"],
                "scientific_payload_sha256": record["comparison"][
                    "scientific_payload_sha256"
                ],
                "comparison_sha256": record["comparison"]["sha256"],
                "full_case_count": len(full_cases[name]),
                "full_case_ids_sha256": hashlib.sha256(
                    json.dumps(
                        [case["case_id"] for case in full_cases[name]],
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            for name, record in gate["partitions"].items()
        },
        "failed_pre_case_attempt_count": len(gate["failed_pre_case_attempts"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
