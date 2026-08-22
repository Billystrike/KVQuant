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

from utils.llama2_cage_v3_transfer_quality_acceptance import lf_normalized_file_sha256
from utils.llama2_cage_v3_transfer_quality_full_design import (
    DESIGN_SHA256,
    validate_design,
    validate_server_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen Llama-2 CAGE-v3 600-case execution design")
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design_path = args.design.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite full-design preflight: {output}")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if lf_normalized_file_sha256(design_path) != DESIGN_SHA256:
        raise RuntimeError("full-design file hash changed")
    validate_design(design, repo_root=REPO_ROOT)
    server_artifacts = validate_server_artifacts(design, repo_root=REPO_ROOT)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout)
    if dirty:
        raise RuntimeError("full-design preflight requires a clean repository")
    report = {
        "schema_version": 1,
        "preflight_id": "llama2-7b-cage-v3-transfer-quality-full-design-preflight-v1",
        "status": "pass",
        "claim_eligible": False,
        "design_sha256": DESIGN_SHA256,
        "source_state": {"git_commit": commit, "dirty": False},
        "server_artifacts": server_artifacts,
        "case_matrix": {
            "method_length_point_count": 12,
            "anchors_per_point": 50,
            "full_case_count": 600,
            "target_count_per_case": 64,
            "total_target_count": 38400,
        },
        "environment_policy": {
            "known_pip_check_issue_count": 4,
            "environment_mutation_performed": False,
            "required_runtime_imports_checked_at_this_stage": False,
            "required_runtime_imports_must_pass_full_gate": True,
        },
        "execution_boundary": {
            "model_weights_accessed": False,
            "gpu_used": False,
            "quality_metrics_read_or_computed": False,
            "partial_results_interpreted": False,
            "full_runner_and_gate_implementation_authorized": True,
            "full_execution_gate_run_authorized": False,
            "full_600_case_execution_authorized": False,
            "paper_claims_authorized": False,
            "runtime_claims_authorized": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
