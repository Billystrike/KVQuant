#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v2_gate import load_cage_v2_acceptance_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the joint CAGE-v2 round1 acceptance gate")
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    gate, gate_sha256 = load_cage_v2_acceptance_gate(
        args.gate.resolve(),
        protocol_path=args.protocol.resolve(),
        manifest_path=args.manifest.resolve(),
        verify_artifacts=True,
    )
    report = {
        "schema_version": 1,
        "status": "pass",
        "gate_id": gate["gate_id"],
        "gate_sha256": gate_sha256,
        "claim_eligible": False,
        "protocol_sha256": gate["protocol"]["sha256"],
        "manifest_sha256": gate["development_manifest"]["sha256"],
        "execution_commit": gate["round1_screen_authorization"]["execution_commit"],
        "partitions": {
            partition: {
                "acceptance_cases_per_repeat": gate["partitions"][partition]["case_count_per_repeat"],
                "scientific_payload_sha256": gate["partitions"][partition]["scientific_payload_sha256"],
            }
            for partition in ("cage_qwen3", "kitty_qwen3")
        },
        "screen_authorization": gate["round1_screen_authorization"],
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
