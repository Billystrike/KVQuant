#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llama2_cage_v3_gpu_acceptance_protocol import load_protocol, query_gpu_inventory
from utils.qwen3_cage_v4_data import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the read-only Llama-2 CAGE-v3 GPU acceptance static preflight")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite GPU preflight output: {output}")
    protocol, protocol_sha256 = load_protocol(args.protocol.resolve(), repo_root=REPO_ROOT)
    model = protocol["production_model"]
    model_path = Path(model["reference"])
    if file_sha256(model_path / "config.json") != model["config_sha256"]:
        raise ValueError("Llama-2 config hash mismatch")
    weight_files = []
    for spec in model["weight_files"]:
        path = model_path / spec["name"]
        if path.stat().st_size != spec["size_bytes"]:
            raise ValueError(f"weight size mismatch: {path}")
        weight_files.append({"name": spec["name"], "size_bytes": path.stat().st_size, "frozen_sha256": spec["sha256"]})
    gpu = query_gpu_inventory()
    expected = protocol["expected_environment"]
    checks = {
        "protocol_and_cpu_receipt": True,
        "model_config_identity": True,
        "weight_files_present_with_frozen_sizes": len(weight_files) == 2,
        "single_visible_gpu": gpu["visible_gpu_count"] == 1,
        "gpu_name": gpu["name"] == expected["expected_gpu_name"],
        "gpu_memory_capacity": gpu["memory_total_mib"] >= expected["minimum_gpu_memory_mib"],
        "synthetic_input_frozen": True,
        "no_model_weights_loaded": True,
        "no_cuda_computation": True,
        "no_corpus_or_quality_metric": True,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "preflight_id": "llama2-7b-cage-v3-production-gpu-acceptance-static-preflight-v1",
        "status": "pass" if not failures else "fail",
        "claim_eligible": False,
        "failures": failures,
        "checks": checks,
        "protocol_sha256": protocol_sha256,
        "cpu_acceptance_receipt_sha256": protocol["cpu_acceptance_receipt"]["sha256"],
        "gpu_inventory": gpu,
        "model_reference": str(model_path),
        "model_config_sha256": model["config_sha256"],
        "weight_files": weight_files,
        "execution_boundary": {
            "model_weights_loaded": False,
            "cuda_computation_performed": False,
            "corpus_accessed": False,
            "quality_metric_computed": False,
            "gpu_acceptance_execution_authorized": False,
            "formal_transfer_authorized": False,
            "runtime_claims_authorized": False,
        },
        "next_step": "freeze this passing preflight receipt before implementing and authorizing GPU acceptance execution",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    handle = tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if failures:
        raise RuntimeError(f"GPU acceptance static preflight failed: {failures}")


if __name__ == "__main__":
    main()
