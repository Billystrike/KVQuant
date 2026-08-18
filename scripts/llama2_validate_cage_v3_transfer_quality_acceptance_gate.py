#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llama2_cage_v3_gpu_execution import full_file_sha256
from utils.llama2_cage_v3_transfer_quality_acceptance import (
    expand_acceptance_cases,
    lf_normalized_file_sha256,
    load_execution,
    load_json,
    load_server_input_manifest,
    require,
    validate_input_artifact_manifest,
)
from utils.llama2_cage_v3_transfer_quality_protocol import load_transfer_quality_protocol
from utils.qwen3_cage_v4_data import file_sha256


def _write_fresh(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite acceptance gate receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _gpu_name() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(len(names) == 1, "acceptance requires exactly one visible GPU")
    return names[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the read-only Llama-2 transfer-quality acceptance execution gate")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execution, execution_sha = load_execution(args.execution.resolve(), repo_root=REPO_ROOT)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    require(not status, "repository must be clean for acceptance gate")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    protocol, protocol_sha = load_transfer_quality_protocol(REPO_ROOT / execution["protocol"]["path"], repo_root=REPO_ROOT)
    input_artifacts = load_json(REPO_ROOT / execution["input_artifacts"]["path"])
    validate_input_artifact_manifest(input_artifacts)
    manifest = load_server_input_manifest(execution, protocol=protocol)
    cases = expand_acceptance_cases(execution=execution, execution_sha256=execution_sha, protocol=protocol, input_manifest=manifest)
    source_manifest = load_json(REPO_ROOT / execution["source_manifest"]["path"])
    require(source_manifest.get("source_id") == "llama2-7b-cage-v3-transfer-quality-acceptance-sources-v1", "source manifest identity changed")
    require(source_manifest.get("source_identity_mode") == "sha256_after_deterministic_crlf_to_lf_normalization", "source identity mode changed")
    frozen_sources = source_manifest.get("files", {})
    require(bool(frozen_sources), "source manifest is empty")
    for spec in frozen_sources.values():
        require(lf_normalized_file_sha256(REPO_ROOT / spec["path"]) == spec["sha256"], f"frozen source changed: {spec['path']}")
    model_root = Path(execution["model"]["reference"])
    require(file_sha256(model_root / "config.json") == execution["model"]["config_sha256"], "model config changed")
    weight_receipts = []
    for spec in execution["model"]["weight_files"]:
        path = model_root / spec["name"]
        require(path.stat().st_size == spec["size_bytes"], f"weight size changed: {spec['name']}")
        digest = full_file_sha256(path)
        require(digest == spec["sha256"], f"weight digest changed: {spec['name']}")
        weight_receipts.append({"name": spec["name"], "sha256": digest, "size_bytes": path.stat().st_size})
    import torch
    import transformers
    gpu = _gpu_name()
    checks = {
        "execution_frozen": True,
        "repository_clean": True,
        "protocol_frozen": protocol_sha == execution["protocol"]["sha256"],
        "input_artifact_frozen_and_passed": True,
        "input_manifest_frozen_and_valid": True,
        "exact_12_acceptance_cases": len(cases) == 12,
        "all_64_target_scoring_frozen": execution["scoring"]["primary_target_count_per_case"] == 64,
        "source_manifest_frozen": True,
        "all_execution_sources_frozen": True,
        "model_config_frozen": True,
        "full_weight_hashes_frozen": len(weight_receipts) == 2,
        "python_version": platform.python_version().startswith("3.10."),
        "torch_version": str(torch.__version__) == execution["environment"]["torch"],
        "transformers_version": transformers.__version__ == execution["environment"]["transformers"],
        "single_expected_gpu": gpu == execution["environment"]["gpu"],
        "no_model_weights_loaded": True,
        "no_quality_metric_computed_or_read": True,
        "no_cuda_computation": True,
    }
    require(all(checks.values()), "acceptance gate contains failed checks")
    gate = {
        "schema_version": 1,
        "gate_id": "llama2-7b-cage-v3-transfer-quality-acceptance-execution-gate-v1",
        "status": "pass",
        "claim_eligible": False,
        "source_state": {"git_commit": commit, "dirty": False},
        "execution_sha256": execution_sha,
        "protocol_sha256": protocol_sha,
        "input_manifest_sha256": execution["input_manifest"]["sha256"],
        "input_manifest_size_bytes": execution["input_manifest"]["size_bytes"],
        "case_count_per_repeat": len(cases),
        "repeat_count": 2,
        "weight_receipts": weight_receipts,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "transformers": transformers.__version__, "gpu": gpu},
        "frozen_sources": frozen_sources,
        "checks": checks,
        "decision": {
            "quality_acceptance_execution_authorized": True,
            "full_600_case_execution_authorized": False,
            "candidate_tuning_authorized": False,
            "paper_main_method_change_authorized": False,
            "runtime_claims_authorized": False,
        },
        "next_step": "check in this exact gate receipt before running acceptance repeat A or B",
    }
    _write_fresh(args.output.resolve(), gate)
    print(json.dumps(gate, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
