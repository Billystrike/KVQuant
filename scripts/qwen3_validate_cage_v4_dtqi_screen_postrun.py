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

from utils.qwen3_cage_v4_data import file_sha256, load_data_protocol, validate_input_manifest
from utils.qwen3_cage_v4_dtqi_acceptance import load_json
from utils.qwen3_cage_v4_dtqi_screen import expand_screen_cases, load_screen_execution
from utils.qwen3_cage_v4_dtqi_screen_postrun import AUDIT_ID, validate_artifact_manifest, validate_screen


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen 120-case CAGE-v4-DTQI screen")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite DTQI screen postrun audit: {output}")
    execution, execution_sha, protocol, _, quota = load_screen_execution(args.execution.resolve(), repo_root=REPO_ROOT, verify_artifacts=True)
    artifacts = load_json(args.artifacts.resolve())
    validate_artifact_manifest(artifacts)
    if artifacts["execution_sha256"] != execution_sha:
        raise RuntimeError("DTQI screen artifact/execution linkage mismatch")
    manifest = load_json(execution["input_manifest"]["path"])
    data_protocol, data_sha = load_data_protocol(REPO_ROOT / execution["data_protocol"]["path"])
    validate_input_manifest(manifest, protocol=data_protocol, protocol_sha256=data_sha)
    cases = expand_screen_cases(execution=execution, execution_sha256=execution_sha, protocol=protocol, quota_plan=quota, input_manifest=manifest)
    report = validate_screen(artifacts, cases)
    audit = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "status": "pass",
        "claim_eligible": False,
        "interpretation_performed": False,
        "holdout_accessed": False,
        "pg19_test_accessed": False,
        "execution_source_commit": artifacts["execution_source_commit"],
        "execution_sha256": execution_sha,
        "dtqi_protocol_sha256": execution["dtqi_protocol"]["sha256"],
        "gpu_acceptance_receipt_sha256": execution["gpu_acceptance_receipt"]["sha256"],
        "input_manifest_sha256": execution["input_manifest"]["sha256"],
        "artifact_manifest_path": str(args.artifacts.resolve()),
        "artifact_manifest_sha256": file_sha256(args.artifacts.resolve()),
        "screen": report,
        "execution_boundary": artifacts["execution_boundary"],
    }
    _write_atomic(output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
