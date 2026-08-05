#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_perturbation_protocol import (
    Qwen3PerturbationError,
    file_sha256,
    load_perturbation_acceptance_gate,
    load_perturbation_protocol,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the frozen Qwen3 perturbation full-run gate"
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--acceptance-gate", type=Path, required=True)
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = _parse_args()
    protocol, protocol_sha256 = load_perturbation_protocol(args.protocol.resolve())
    gate, gate_sha256 = load_perturbation_acceptance_gate(
        args.acceptance_gate.resolve(),
        perturbation_protocol_sha256=protocol_sha256,
        verify_artifacts=True,
    )
    current_commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=all"))
    if dirty:
        raise Qwen3PerturbationError("full-run gate validation requires a clean source tree")
    acceptance_source = gate["acceptance_source"]
    recorder_sha256 = file_sha256(
        REPO_ROOT / "utils" / "qwen3_perturbation_runtime.py"
    )
    adapter_sha256 = file_sha256(REPO_ROOT / "models" / "qwen3_cage.py")
    if recorder_sha256 != acceptance_source["recorder_sha256"]:
        raise Qwen3PerturbationError("recorder source changed after acceptance")
    if adapter_sha256 != acceptance_source["qwen3_cage_sha256"]:
        raise Qwen3PerturbationError("attention adapter changed after acceptance")
    omp_value = os.environ.get("OMP_NUM_THREADS")
    if omp_value != acceptance_source["omp_num_threads_observed"]:
        raise Qwen3PerturbationError("OMP_NUM_THREADS differs from acceptance")

    report = {
        "schema_version": 1,
        "status": "pass",
        "gate_id": gate["gate_id"],
        "gate_sha256": gate_sha256,
        "perturbation_protocol_id": protocol["protocol_id"],
        "perturbation_protocol_sha256": protocol_sha256,
        "current_source_commit": current_commit,
        "acceptance_source_commit": acceptance_source["cage_commit"],
        "recorder_sha256": recorder_sha256,
        "qwen3_cage_sha256": adapter_sha256,
        "omp_num_threads": omp_value,
        "partitions": {
            name: {
                "case_count_per_repeat": record["case_count_per_repeat"],
                "case_ids_sha256": record["case_ids_sha256"],
                "scientific_payload_sha256": record["scientific_payload_sha256"],
                "repeat_artifacts_verified": True,
                "bitwise_equal": True,
            }
            for name, record in gate["partitions"].items()
        },
        "authorized_full_cases": gate["full_run_authorization"]["total_cases"],
        "authorized_layer_records": gate["full_run_authorization"]["layer_records"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
