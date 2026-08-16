#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v3_promotion_full import expand_full_holdout_cases
from utils.qwen3_cage_v3_promotion_gate import load_promotion_gate
from utils.qwen3_cage_v3_promotion_protocol import load_promotion_protocol
from utils.qwen3_cage_v4_data import (
    canonical_sha256,
    file_sha256,
    load_data_protocol,
    validate_input_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CAGE-v3 promotion full-holdout authorization gate")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-gate", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_sha256 = load_promotion_protocol(args.protocol.resolve())
    manifest_path = args.input_manifest.resolve()
    manifest_sha256 = file_sha256(manifest_path)
    if manifest_sha256 != protocol["input_receipt"]["input_manifest_sha256"]:
        raise RuntimeError("promotion input manifest differs from protocol receipt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = REPO_ROOT / protocol["input_receipt"]["data_protocol_path"]
    data_protocol, data_sha256 = load_data_protocol(data_path)
    if data_sha256 != protocol["input_receipt"]["data_protocol_sha256"]:
        raise RuntimeError("promotion data protocol receipt mismatch")
    validate_input_manifest(manifest, protocol=data_protocol, protocol_sha256=data_sha256)
    gate, gate_sha256 = load_promotion_gate(
        args.acceptance_gate.resolve(),
        repo_root=REPO_ROOT,
        protocol_sha256=protocol_sha256,
        input_manifest_sha256=manifest_sha256,
        verify_server_artifacts=True,
    )
    expanded = {
        partition: expand_full_holdout_cases(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            gate_sha256=gate_sha256,
            manifest=manifest,
            partition=partition,
            repo_root=REPO_ROOT,
        )
        for partition in ("cage_qwen3", "kitty_qwen3")
    }
    report = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "gate_id": gate["gate_id"],
        "gate_sha256": gate_sha256,
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": manifest_sha256,
        "accepted_commit": gate["acceptance_source_commit"],
        "postrun_commit": gate["postrun_source_commit"],
        "partitions": {
            partition: {
                "acceptance_case_count_per_repeat": gate["partitions"][partition]["acceptance_case_count_per_repeat"],
                "acceptance_scientific_payload_sha256": gate["partitions"][partition]["scientific_payload_sha256"],
                "full_holdout_case_count": len(cases),
                "full_holdout_case_ids_sha256": canonical_sha256([case["case_id"] for case in cases]),
                "method_counts": dict(Counter(case["method"]["metric_method_id"] for case in cases)),
            }
            for partition, cases in expanded.items()
        },
        "full_holdout_authorization": gate["full_holdout_authorization"],
        "authorization_transition": "GPU full-holdout execution is authorized only by this checked-in post-acceptance gate",
        "holdout_method_metrics_read": False,
        "pg19_test_accessed": False,
        "interpretation_performed": False,
        "llama2_execution_authorized": False,
        "kitty_llama_port_authorized": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
