#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v4_data import (
    canonical_sha256,
    file_sha256,
    load_data_protocol,
    validate_input_manifest,
)
from utils.qwen3_cage_v4_gate import load_acceptance_gate
from utils.qwen3_cage_v4_metric_protocol import load_metric_protocol
from utils.qwen3_cage_v4_screen import expand_full_screen_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the joint PG-19 metric acceptance gate")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-gate", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_sha256 = load_metric_protocol(args.protocol.resolve())
    manifest_path = args.input_manifest.resolve()
    manifest_sha256 = file_sha256(manifest_path)
    if manifest_sha256 != protocol["input_receipt"]["input_manifest_sha256"]:
        raise RuntimeError("PG-19 manifest differs from metric protocol receipt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_protocol_path = REPO_ROOT / "configs" / "qwen3_8b_cage_v4_pg19_data_protocol_v1.json"
    data_protocol, data_protocol_sha256 = load_data_protocol(data_protocol_path)
    if data_protocol_sha256 != protocol["input_receipt"]["data_protocol_sha256"]:
        raise RuntimeError("PG-19 data protocol differs from metric protocol receipt")
    validate_input_manifest(
        manifest,
        protocol=data_protocol,
        protocol_sha256=data_protocol_sha256,
    )
    gate, gate_sha256 = load_acceptance_gate(
        args.acceptance_gate.resolve(),
        repo_root=REPO_ROOT,
        protocol_sha256=protocol_sha256,
        input_manifest_sha256=manifest_sha256,
        verify_artifacts=True,
    )
    expanded = {
        partition: expand_full_screen_cases(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
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
        "accepted_commit": gate["acceptance_source"]["git_commit"],
        "failed_pre_case_attempt_count": len(gate["failed_pre_case_attempts"]),
        "partitions": {
            partition: {
                "acceptance_case_count": gate["partitions"][partition]["case_count"],
                "acceptance_scientific_payload_sha256": gate["partitions"][partition][
                    "scientific_payload_sha256"
                ],
                "full_screen_case_count": len(cases),
                "full_screen_case_ids_sha256": canonical_sha256(
                    [case["case_id"] for case in cases]
                ),
                "compressed_perturbation_case_count": sum(
                    case["method"]["metric_method_id"] != "fp16" for case in cases
                ),
            }
            for partition, cases in expanded.items()
        },
        "authorization": gate["full_screen_authorization"],
        "authorization_transition": "full screen becomes authorized only through this post-acceptance gate",
        "still_prohibited": [
            "PG-19 holdout method metrics",
            "PG-19 test access",
            "CAGE-v4 candidate execution",
            "claim-eligible interpretation before full-screen analysis",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
