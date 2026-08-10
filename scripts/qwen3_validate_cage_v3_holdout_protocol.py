#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v3_execution import canonical_sha256
from utils.qwen3_cage_v3_holdout import (
    HOLDOUT_ANCHORS,
    PARTITIONS,
    STAGES,
    expand_holdout_cases,
    load_holdout_execution,
)
from utils.qwen3_cage_v3_protocol import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen CAGE-v3 holdout authorization without reading metrics")
    parser.add_argument("--execution", type=Path, required=True)
    args = parser.parse_args()
    execution_path = args.execution.resolve()
    execution, execution_sha256, protocol, manifest, plan, decision = load_holdout_execution(
        execution_path,
        repo_root=REPO_ROOT,
        verify_artifacts=True,
    )
    partitions = {}
    all_anchors = set()
    for partition in PARTITIONS:
        partitions[partition] = {}
        for stage in STAGES:
            cases = expand_holdout_cases(
                execution=execution,
                execution_sha256=execution_sha256,
                protocol=protocol,
                manifest=manifest,
                plan=plan,
                partition=partition,
                stage=stage,
            )
            anchors = sorted({case["input"]["anchor_index"] for case in cases})
            all_anchors.update(anchors)
            partitions[partition][stage] = {
                "case_count": len(cases),
                "anchor_indices": anchors,
                "case_ids_sha256": canonical_sha256([case["case_id"] for case in cases]),
                "method_ids": sorted({case["method"]["id"] for case in cases}),
            }
    if not all_anchors.issubset(set(HOLDOUT_ANCHORS)) or set(all_anchors) & set(range(20, 50)):
        raise RuntimeError("holdout preflight expanded protected reserved anchors")
    report = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "metrics_read": False,
        "execution_sha256": execution_sha256,
        "screen_decision_sha256": file_sha256(REPO_ROOT / execution["screen_decision"]["path"]),
        "selected_family_id": decision["decision"]["selected_family_id"],
        "partitions": partitions,
        "holdout_anchor_indices": HOLDOUT_ANCHORS,
        "reserved_unseen_anchor_indices": list(range(20, 50)),
        "reserved_unseen_metrics_authorized": False,
        "holdout_full_requires_joint_acceptance_gate": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
