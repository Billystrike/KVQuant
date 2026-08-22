#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path


REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, REQUIRED_CUBLAS_WORKSPACE_CONFIG):
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = REQUIRED_CUBLAS_WORKSPACE_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoConfig

from scripts.llama2_run_cage_v3_transfer_quality_acceptance import (
    _atomic_json,
    _configure_model,
    _score_case,
)
from utils.llama2_cage_v3_transfer_quality_full import (
    FULL_RUN_ID,
    expand_full_cases,
    load_design,
    load_full_gate_receipt,
    load_json,
    require,
    scientific_payload,
    validate_full_case,
)
from utils.llama2_cage_v3_transfer_quality_manifest import validate_input_manifest
from utils.llama2_cage_v3_transfer_quality_protocol import load_transfer_quality_protocol
from utils.qwen3_cage_v4_data import canonical_sha256


def _source_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout)
    require(not dirty, "full execution requires a clean repository")
    return {"git_commit": commit, "dirty": False}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the frozen 600-case Llama-2 CAGE-v3 transfer-quality matrix")
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--gate-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    design, design_sha = load_design(args.design.resolve(), repo_root=REPO_ROOT)
    gate, gate_sha = load_full_gate_receipt(args.gate_receipt.resolve(), repo_root=REPO_ROOT)
    protocol, protocol_sha = load_transfer_quality_protocol(REPO_ROOT / design["protocol"]["path"], repo_root=REPO_ROOT)
    input_path = Path(design["input_manifest"]["path"])
    input_manifest = load_json(input_path)
    validate_input_manifest(input_manifest, protocol=protocol)
    cases = expand_full_cases(design=design, protocol=protocol, input_manifest=input_manifest, gate_receipt_sha256=gate_sha)
    source_state = _source_state()
    output_dir = args.output_dir.resolve()
    require(str(output_dir) == design["output_and_resume"]["output_dir"], "full output path changed")
    cases_dir = output_dir / "cases"
    failures_dir = output_dir / "failures"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(exist_ok=True)
    failures_dir.mkdir(exist_ok=True)

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.get_device_name(0) == design["environment"]["gpu"], "GPU identity changed")
    require(str(torch.__version__) == design["environment"]["torch"], "torch version changed")
    require(__import__("transformers").__version__ == design["environment"]["transformers"], "Transformers version changed")
    native_config = AutoConfig.from_pretrained(design["model"]["reference"], local_files_only=True)
    quota = load_json(REPO_ROOT / design["quota_plan"]["path"])
    expected_ids = [case["case_id"] for case in cases]
    resumed = 0
    new = 0
    for index, case in enumerate(cases, start=1):
        path = cases_dir / f"{case['case_id']}.json"
        if path.is_file():
            existing = load_json(path)
            validate_full_case(existing, case)
            require(existing["provenance"]["gate_receipt_sha256"] == gate_sha, "resumed gate receipt changed")
            resumed += 1
            print(f"[{index}/600] resumed {case['case_id']} {case['method']['id']} a={case['input']['identity']['anchor_index']}")
            continue
        seed = int(design["determinism"]["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"[{index}/600] running {case['method']['id']} a={case['input']['identity']['anchor_index']}")
        config, model_class = _configure_model(native_config, case, quota)
        model = model_class.from_pretrained(
            design["model"]["reference"],
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to("cuda:0").eval()
        require(sorted({str(parameter.device) for parameter in model.parameters()}) == ["cuda:0"], "model parameters are not exclusively on cuda:0")
        require(sorted({str(parameter.dtype) for parameter in model.parameters()}) == ["torch.float16"], "model parameters are not exclusively float16")
        try:
            scoring, cache, telemetry = _score_case(model, case)
            record = {
                "schema_version": 1,
                "status": "completed",
                "case_id": case["case_id"],
                "method": case["method"],
                "input": case["input"],
                "memory": case["memory"],
                "scoring": scoring,
                "cache": cache,
                "runtime_diagnostics": telemetry,
                "provenance": {
                    "full_run_id": FULL_RUN_ID,
                    "design_sha256": design_sha,
                    "gate_receipt_sha256": gate_sha,
                    "protocol_sha256": protocol_sha,
                    "input_manifest_sha256": design["input_manifest"]["sha256"],
                    "source_state": source_state,
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "transformers": __import__("transformers").__version__,
                    "gpu": torch.cuda.get_device_name(0),
                    "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                },
            }
            validate_full_case(record, case)
            _atomic_json(path, record)
            (failures_dir / f"{case['case_id']}.json").unlink(missing_ok=True)
            new += 1
            print(f"[{index}/600] completed {case['case_id']} mean_nll={scoring['mean_nll']:.9g}")
        except Exception as error:
            _atomic_json(failures_dir / f"{case['case_id']}.json", {"case_id": case["case_id"], "status": "failed", "error": str(error)})
            raise
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    records = [load_json(cases_dir / f"{case_id}.json") for case_id in expected_ids]
    for record, case in zip(records, cases):
        validate_full_case(record, case)
    payload = [scientific_payload(record) for record in records]
    method_counts = Counter(record["method"]["method"] for record in records)
    length_counts = Counter(str(record["input"]["identity"]["prompt_length"]) for record in records)
    identity = {
        "full_run_id": FULL_RUN_ID,
        "design_sha256": design_sha,
        "gate_receipt_sha256": gate_sha,
        "protocol_sha256": protocol_sha,
        "input_manifest_sha256": design["input_manifest"]["sha256"],
        "source_state": source_state,
        "expected_case_ids": expected_ids,
    }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "expected_cases": 600,
        "completed_cases": len(records),
        "failure_records": len(list(failures_dir.glob("*.json"))),
        "new_cases": new,
        "resumed_cases": resumed,
        "scientific_payload_sha256": canonical_sha256(payload),
        "method_counts": dict(sorted(method_counts.items())),
        "length_counts": dict(sorted(length_counts.items())),
        "target_count_per_case": 64,
        "total_target_count": 38400,
        "execution_boundary": {
            "full_execution_complete": True,
            "quality_interpretation_authorized": False,
            "candidate_tuning_authorized": False,
            "paper_claims_authorized": False,
            "runtime_claims_authorized": False,
        },
    }
    require(summary["completed_cases"] == 600 and summary["failure_records"] == 0, "full execution completion failed")
    _atomic_json(output_dir / "run_identity.json", identity)
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
