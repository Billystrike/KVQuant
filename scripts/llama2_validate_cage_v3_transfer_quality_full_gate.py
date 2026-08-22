#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, REQUIRED_CUBLAS_WORKSPACE_CONFIG):
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = REQUIRED_CUBLAS_WORKSPACE_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers

from utils.llama2_cage_v3_transfer_quality_acceptance import lf_normalized_file_sha256
from utils.llama2_cage_v3_transfer_quality_full_design import validate_design, validate_server_artifacts
from utils.qwen3_cage_v4_data import file_sha256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Llama-2 CAGE-v3 600-case execution gate")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite full execution gate output: {output}")
    plan_path = args.plan.resolve()
    plan = load_json(plan_path)
    require(plan.get("schema_version") == 1, "full gate plan schema changed")
    require(plan.get("plan_id") == "llama2-7b-cage-v3-transfer-quality-full-gate-plan-v1", "full gate plan identity changed")
    require(plan.get("status") == "frozen_before_full_gate_execution" and plan.get("claim_eligible") is False, "full gate plan status changed")
    require(
        plan.get("authorization_before_gate")
        == {
            "full_gate_execution": True,
            "full_600_case_execution": False,
            "quality_interpretation": False,
            "candidate_tuning": False,
            "paper_claims": False,
            "runtime_claims": False,
        },
        "full gate pre-authorization changed",
    )
    require(
        plan.get("resources")
        == {
            "minimum_free_disk_bytes": 1073741824,
            "model_weight_hashing_required": True,
            "model_weight_loading_forbidden": True,
        },
        "full gate resource policy changed",
    )
    design_spec = plan["design"]
    design_path = REPO_ROOT / design_spec["path"]
    require(lf_normalized_file_sha256(design_path) == design_spec["sha256"], "full gate design changed")
    design = load_json(design_path)
    validate_design(design, repo_root=REPO_ROOT)
    server_artifacts = validate_server_artifacts(design, repo_root=REPO_ROOT)
    preflight_spec = plan["design_preflight_receipt"]
    preflight_path = REPO_ROOT / preflight_spec["path"]
    require(lf_normalized_file_sha256(preflight_path) == preflight_spec["sha256"], "design preflight receipt changed")
    preflight = load_json(preflight_path)
    require(preflight.get("status") == "frozen_after_successful_static_full_design_preflight", "design preflight did not pass")
    require(preflight.get("decision", {}).get("full_gate_execution_authorized_after_source_freeze") is True, "full gate run is not authorized")
    require(preflight.get("decision", {}).get("full_600_case_execution_authorized") is False, "preflight prematurely authorized full execution")
    source_checks = {}
    for name, spec in plan["frozen_sources"].items():
        source_checks[name] = lf_normalized_file_sha256(REPO_ROOT / spec["path"]) == spec["sha256"]
    require(all(source_checks.values()), "one or more full execution sources changed")
    required_imports = {}
    for module_name in design["environment"]["required_imports"]:
        module = importlib.import_module(module_name)
        required_imports[module_name] = bool(module)
    full_runner_module = "scripts.llama2_run_cage_v3_transfer_quality_full"
    required_imports[full_runner_module] = bool(importlib.import_module(full_runner_module))
    require(all(required_imports.values()), "one or more required runtime imports failed")
    require(platform.python_version().startswith(design["environment"]["python_major_minor"] + "."), "Python version changed")
    require(str(torch.__version__) == design["environment"]["torch"], "torch version changed")
    require(transformers.__version__ == design["environment"]["transformers"], "Transformers version changed")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.get_device_name(0) == design["environment"]["gpu"], "GPU identity changed")
    model_root = Path(design["model"]["reference"])
    model_checks = {
        "config": file_sha256(model_root / "config.json") == design["model"]["config_sha256"],
    }
    for spec in design["model"]["weight_files"]:
        path = model_root / spec["name"]
        model_checks[spec["name"]] = path.is_file() and path.stat().st_size == spec["size_bytes"] and file_sha256(path) == spec["sha256"]
    require(all(model_checks.values()), "model artifact identity changed")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout)
    require(not dirty, "full gate requires a clean repository")
    free_bytes = shutil.disk_usage("/root/autodl-tmp").free
    require(free_bytes >= plan["resources"]["minimum_free_disk_bytes"], "insufficient free disk for full execution artifacts")
    checks = {
        "design": True,
        "acceptance_postrun": True,
        "design_preflight": True,
        "frozen_sources": all(source_checks.values()),
        "required_runtime_imports": all(required_imports.values()),
        "python": True,
        "torch": True,
        "transformers": True,
        "cuda": True,
        "gpu": True,
        "model_artifacts": all(model_checks.values()),
        "input_manifest": server_artifacts["input_record_count"] == 150,
        "full_output_fresh": server_artifacts["full_output_exists"] is False,
        "repository_clean": True,
        "disk": True,
        "no_model_weights_loaded": True,
        "no_quality_metrics_read_or_computed": True,
    }
    report = {
        "schema_version": 1,
        "gate_id": "llama2-7b-cage-v3-transfer-quality-full-execution-gate-v1",
        "status": "pass",
        "claim_eligible": False,
        "design_sha256": design_spec["sha256"],
        "plan_sha256": lf_normalized_file_sha256(plan_path),
        "source_state": {"git_commit": commit, "dirty": False},
        "frozen_source_sha256": plan["frozen_sources"],
        "checks": checks,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "required_imports": required_imports,
            "known_pip_check_issue_count": len(design["environment"]["known_pip_check_issues"]),
            "environment_mutation_performed": False,
        },
        "resources": {"free_disk_bytes": free_bytes},
        "case_count": 600,
        "target_count": 38400,
        "output_dir": design["output_and_resume"]["output_dir"],
        "execution_boundary": {
            "model_weight_files_hashed_but_not_loaded": True,
            "quality_metrics_read_or_computed": False,
            "partial_results_interpreted": False,
        },
        "decision": {
            "full_600_case_execution_authorized": True,
            "strict_resume_authorized": True,
            "quality_interpretation_authorized": False,
            "candidate_tuning_authorized": False,
            "paper_claims_authorized": False,
            "runtime_claims_authorized": False,
        },
    }
    require(all(checks.values()), "full gate checks did not all pass")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
