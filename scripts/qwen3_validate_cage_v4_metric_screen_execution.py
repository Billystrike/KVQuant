#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256
from utils.qwen3_cage_v4_execution import load_screen_execution


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen PG-19 metric screen execution")
    parser.add_argument("--execution", type=Path, required=True)
    args = parser.parse_args()
    execution_path = args.execution.resolve()
    execution, execution_sha256, protocol, protocol_sha256, manifest, expanded = (
        load_screen_execution(
            execution_path,
            repo_root=REPO_ROOT,
            verify_artifacts=True,
        )
    )
    if manifest is None or expanded is None:
        raise RuntimeError("screen execution validation did not load frozen inputs")
    report = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "execution_id": execution["execution_id"],
        "execution_sha256": execution_sha256,
        "protocol_sha256": protocol_sha256,
        "acceptance_gate_sha256": execution["acceptance_gate"]["sha256"],
        "gate_preflight_sha256": execution["gate_preflight_receipt"]["sha256"],
        "input_manifest_sha256": execution["input_manifest"]["sha256"],
        "screen_document_count": sum(
            document["partition"] == "screen" for document in manifest["documents"]
        ),
        "screen_anchor_count": sum(anchor["partition"] == "screen" for anchor in manifest["anchors"]),
        "partitions": {
            partition: {
                "quality_case_count": len(cases),
                "compressed_perturbation_case_count": sum(
                    case["method"]["metric_method_id"] != "fp16" for case in cases
                ),
                "case_ids_sha256": canonical_sha256([case["case_id"] for case in cases]),
            }
            for partition, cases in expanded.items()
        },
        "execution_boundary": execution["execution_boundary"],
        "runner": {
            "path": "scripts/qwen3_run_cage_v4_metric_screen.py",
            "sha256": file_sha256(REPO_ROOT / "scripts" / "qwen3_run_cage_v4_metric_screen.py"),
            "stage_switch_exposed": False,
        },
        "protocol_pre_gate_full_authorization": protocol["execution_boundary"][
            "full_gpu_execution_authorized"
        ],
        "authorization_basis": "the separate checked-in post-acceptance execution receipt",
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
