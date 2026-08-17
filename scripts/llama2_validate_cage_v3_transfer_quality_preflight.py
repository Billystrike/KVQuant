#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llama2_cage_v3_transfer_quality_protocol import (
    load_transfer_quality_protocol,
)
from utils.qwen3_cage_v4_data import file_sha256


def _source_state() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("repository must be clean for the frozen static preflight")
    return {"git_commit": commit, "dirty": False}


def _write_fresh_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen preflight output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the frozen Llama-2 CAGE-v3 transfer-quality protocol without corpus, model, or GPU access"
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    output = args.output.resolve()
    protocol, protocol_sha256 = load_transfer_quality_protocol(
        protocol_path, repo_root=REPO_ROOT
    )
    source_state = _source_state()
    rows = protocol["method_length_matrix"]
    report = {
        "schema_version": 1,
        "preflight_id": "llama2-7b-cage-v3-transfer-quality-static-preflight-v1",
        "status": "pass",
        "claim_eligible": False,
        "source_state": source_state,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "parent_transfer_protocol_sha256": protocol["parent_transfer_protocol"]["sha256"],
        "gpu_acceptance_postrun_receipt_sha256": protocol["gpu_acceptance_postrun_receipt"]["sha256"],
        "quota_plan_sha256": protocol["quota_plan"]["sha256"],
        "checks": {
            "protocol_frozen": True,
            "qwen3_negative_outcome_preserved": True,
            "production_gpu_acceptance_pass": True,
            "quota_plan_frozen_without_llama2_metrics": True,
            "model_identity_frozen": True,
            "input_identity_frozen": True,
            "all_64_targets_primary": True,
            "method_length_matrix_frozen": True,
            "packed_memory_points_frozen": True,
            "primary_thresholds_unchanged": True,
            "kitty_excluded_pending_separate_fidelity_protocol": True,
            "current_execution_boundary_closed": True,
        },
        "case_matrix": protocol["case_matrix"],
        "method_length_points": [
            {
                "prompt_length": row["prompt_length"],
                "method_ids": [method["id"] for method in row["methods"]],
                "packed_memory": row["packed_memory"],
            }
            for row in rows
        ],
        "primary_transfer_gate": protocol["primary_transfer_gate"],
        "boundary": {
            "gpu_used": False,
            "model_weights_accessed": False,
            "corpus_accessed": False,
            "quality_metrics_computed_or_read": False,
            "candidate_tuning_authorized": False,
            "quality_acceptance_execution_authorized": False,
            "full_600_case_execution_authorized": False,
            "kitty_llama_port_authorized": False,
            "paper_main_method_change_authorized": False,
            "runtime_claims_authorized": False,
            "exact_input_manifest_build_authorized": True,
            "manifest_build_corpus_access_limited_to_frozen_identity_and_anchor_materialization": True,
        },
        "next_step": "build and freeze the exact 50-anchor input manifest from the frozen WikiText-2 token stream; do not load model weights or compute quality metrics",
    }
    _write_fresh_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"output_sha256: {file_sha256(output)}")


if __name__ == "__main__":
    main()
