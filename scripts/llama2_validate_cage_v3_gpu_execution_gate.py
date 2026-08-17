#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llama2_cage_v3_gpu_acceptance_protocol import load_protocol, query_gpu_inventory
from utils.llama2_cage_v3_gpu_execution import (
    full_file_sha256,
    load_object,
    require,
    validate_static_preflight_receipt_file,
)
from utils.qwen3_cage_v4_data import file_sha256


EXPECTED_SOURCES_SHA256 = "5a67eb0aeef944cc2ccda7516e6b88d9ddd803cd0bdabfa1385799aa6d604f4d"


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite GPU execution gate receipt: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Authorize the frozen Llama-2 CAGE-v3 production GPU acceptance")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_sha256 = load_protocol(args.protocol.resolve(), repo_root=REPO_ROOT)
    preflight_path = args.preflight_receipt.resolve()
    preflight = validate_static_preflight_receipt_file(preflight_path)
    sources_path = args.sources.resolve()
    require(file_sha256(sources_path) == EXPECTED_SOURCES_SHA256, "GPU execution source manifest hash mismatch")
    sources_manifest = load_object(sources_path)
    require(sources_manifest.get("schema_version") == 1, "GPU execution source manifest schema mismatch")
    require(
        sources_manifest.get("manifest_id") == "llama2-7b-cage-v3-production-gpu-acceptance-execution-sources-v1",
        "GPU execution source manifest identity mismatch",
    )
    require(
        sources_manifest.get("status") == "frozen_after_static_preflight_before_any_production_weight_gpu_execution",
        "GPU execution source manifest status mismatch",
    )
    require(sources_manifest.get("claim_eligible") is False, "GPU source manifest claim boundary changed")
    require(sources_manifest.get("protocol_sha256") == protocol_sha256, "GPU source manifest protocol mismatch")
    require(
        sources_manifest.get("preflight_receipt_sha256") == file_sha256(preflight_path),
        "GPU source manifest preflight mismatch",
    )
    require(not any(sources_manifest.get("execution_boundary", {}).values()), "GPU source manifest expanded authorization")

    frozen_sources = sources_manifest.get("sources", {})
    require(len(frozen_sources) == 10, "GPU frozen source count mismatch")
    source_checks = {}
    for name, spec in frozen_sources.items():
        source_checks[name] = file_sha256(REPO_ROOT / spec["path"]) == spec["sha256"]
    require(all(source_checks.values()), "one or more frozen GPU execution sources changed")

    source_commit = _git("rev-parse", "HEAD")
    repository_clean = _git("status", "--porcelain", "--untracked-files=all") == ""
    require(repository_clean, "repository must be clean before authorizing GPU acceptance")
    inventory = query_gpu_inventory()
    expected_environment = protocol["expected_environment"]
    gpu_identity = (
        inventory["name"] == expected_environment["expected_gpu_name"]
        and inventory["memory_total_mib"] >= expected_environment["minimum_gpu_memory_mib"]
        and inventory["visible_gpu_count"] == 1
    )
    require(gpu_identity, "GPU inventory changed after static preflight")

    model_spec = protocol["production_model"]
    model_path = Path(model_spec["reference"])
    model_config_match = file_sha256(model_path / "config.json") == model_spec["config_sha256"]
    require(model_config_match, "production model config hash mismatch")
    weight_receipts = []
    weight_hash_started = time.perf_counter()
    for spec in model_spec["weight_files"]:
        path = model_path / spec["name"]
        size_bytes = path.stat().st_size
        digest = full_file_sha256(path)
        match = size_bytes == spec["size_bytes"] and digest == spec["sha256"]
        require(match, f"production weight identity mismatch: {path}")
        weight_receipts.append(
            {
                "name": spec["name"],
                "size_bytes": size_bytes,
                "sha256": digest,
                "identity_match": match,
            }
        )
    weight_hash_seconds = time.perf_counter() - weight_hash_started
    preflight_weight_identity = [
        {"name": row["name"], "size_bytes": row["size_bytes"], "frozen_sha256": row["sha256"]}
        for row in weight_receipts
    ] == preflight["weight_files"]
    require(preflight_weight_identity, "full weight hashes differ from the static preflight receipt")

    checks = {
        "protocol_frozen": True,
        "static_preflight_receipt_frozen_and_passed": True,
        "source_manifest_frozen": True,
        "all_execution_sources_frozen": all(source_checks.values()),
        "repository_clean": repository_clean,
        "single_expected_gpu_with_sufficient_memory": gpu_identity,
        "production_model_config_identity": model_config_match,
        "production_weight_full_sha256_identity": all(row["identity_match"] for row in weight_receipts),
        "weight_identity_matches_static_preflight": preflight_weight_identity,
        "no_model_weights_loaded": True,
        "no_cuda_computation": True,
        "no_corpus_or_quality_metric": True,
    }
    require(all(checks.values()), "GPU execution gate check failed")
    receipt = {
        "schema_version": 1,
        "gate_id": "llama2-7b-cage-v3-production-gpu-acceptance-execution-gate-v1",
        "status": "pass",
        "claim_eligible": False,
        "protocol_sha256": protocol_sha256,
        "preflight_receipt_sha256": file_sha256(preflight_path),
        "sources_manifest_sha256": file_sha256(sources_path),
        "source_commit": source_commit,
        "frozen_sources": frozen_sources,
        "source_checks": source_checks,
        "gpu_inventory": inventory,
        "model_reference": str(model_path),
        "model_config_sha256": model_spec["config_sha256"],
        "weight_receipts": weight_receipts,
        "checks": checks,
        "diagnostic_telemetry": {"weight_full_sha256_seconds": weight_hash_seconds},
        "decision": {
            "gpu_acceptance_execution_authorized": True,
            "formal_transfer_authorized": False,
            "corpus_access_authorized": False,
            "quality_metric_authorized": False,
            "runtime_claims_authorized": False,
        },
        "next_step": "freeze this passing gate receipt before running repeat A and repeat B in fresh processes",
    }
    _write_atomic(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
