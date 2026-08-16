#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v3_promotion_protocol import load_promotion_protocol
from utils.qwen3_cage_v4_data import (
    canonical_sha256,
    file_sha256,
    load_data_protocol,
    validate_input_manifest,
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen Qwen3 CAGE-v3 promotion protocol without reading holdout method metrics")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--closeout-json", type=Path, required=True)
    parser.add_argument("--closeout-log", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    manifest_path = args.input_manifest.resolve()
    closeout_path = args.closeout_json.resolve()
    closeout_log_path = args.closeout_log.resolve()
    protocol, protocol_sha256 = load_promotion_protocol(protocol_path)

    for source in protocol["frozen_sources"].values():
        if isinstance(source, dict) and "path" in source:
            source_path = (REPO_ROOT / source["path"]).resolve()
            if file_sha256(source_path) != source["sha256"]:
                raise ValueError(f"frozen source SHA mismatch: {source_path}")

    input_receipt = protocol["input_receipt"]
    if file_sha256(manifest_path) != input_receipt["input_manifest_sha256"]:
        raise ValueError("promotion input manifest SHA mismatch")
    if manifest_path.stat().st_size != input_receipt["input_manifest_size_bytes"]:
        raise ValueError("promotion input manifest size mismatch")
    manifest = _load_json(manifest_path)
    data_path = (REPO_ROOT / input_receipt["data_protocol_path"]).resolve()
    data_protocol, data_sha256 = load_data_protocol(data_path)
    if data_sha256 != input_receipt["data_protocol_sha256"]:
        raise ValueError("PG-19 data protocol SHA mismatch")
    validate_input_manifest(manifest, protocol=data_protocol, protocol_sha256=data_sha256)
    case_ids_sha256 = canonical_sha256([case["case_id"] for case in manifest["cases"]])
    if case_ids_sha256 != input_receipt["case_ids_sha256"]:
        raise ValueError("PG-19 input case-ID hash mismatch")
    holdout_documents = [row for row in manifest["documents"] if row["partition"] == "holdout"]
    holdout_anchors = [row for row in manifest["anchors"] if row["partition"] == "holdout"]
    if len(holdout_documents) != 20 or len(holdout_anchors) != 40:
        raise ValueError("promotion holdout count mismatch")
    acceptance = protocol["execution_stages"]["gpu_acceptance"]
    selected = [
        row
        for row in holdout_anchors
        if row["document_id"] == acceptance["document_id"] and row["anchor_index"] == acceptance["anchor_index"]
    ]
    if len(selected) != 1:
        raise ValueError("promotion acceptance input is not uniquely frozen")

    prior = protocol["prior_boundary"]
    for path, receipt in (
        (closeout_path, prior["closeout_json"]),
        (closeout_log_path, prior["closeout_log"]),
    ):
        if file_sha256(path) != receipt["sha256"] or path.stat().st_size != receipt["size_bytes"]:
            raise ValueError(f"negative closeout artifact changed: {path}")
    closeout = _load_json(closeout_path)
    if closeout.get("status") != "pass" or closeout.get("decision", {}).get("candidate_outcome") != "close_cage_v4_as_negative":
        raise ValueError("negative closeout decision mismatch")
    if any(closeout.get("failures", {}).values()):
        raise ValueError("negative closeout contains failures")

    methods = protocol["method_grid"]
    report = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": file_sha256(manifest_path),
        "input_case_ids_sha256": case_ids_sha256,
        "negative_closeout_sha256": file_sha256(closeout_path),
        "negative_closeout_log_sha256": file_sha256(closeout_log_path),
        "holdout_document_count": len(holdout_documents),
        "holdout_anchor_count": len(holdout_anchors),
        "acceptance_input": {
            "document_id": acceptance["document_id"],
            "anchor_index": acceptance["anchor_index"],
            "anchor_id": selected[0]["anchor_id"],
            "input_case_ids": [row["case_id"] for row in selected[0]["cases"]],
        },
        "method_ids": [row["method_id"] for row in methods],
        "method_length_point_count": sum(len(row["points"]) for row in methods),
        "full_case_counts": {
            "cage_qwen3": 480,
            "kitty_qwen3": 120,
            "total": 600,
        },
        "promotion_gate": protocol["promotion_gate"],
        "memory_gate": protocol["memory_gate"],
        "boundary": {
            "reads_holdout_method_metrics": False,
            "gpu_acceptance_authorized_by_this_preflight": False,
            "full_holdout_authorized": False,
            "pg19_test_access_authorized": False,
            "llama2_execution_authorized": False,
            "kitty_llama_port_authorized": False,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
