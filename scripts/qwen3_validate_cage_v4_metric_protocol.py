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
from utils.qwen3_cage_v4_metric_protocol import load_metric_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen PG-19 metric-validity protocol")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--manifest-build-log", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    manifest_path = args.input_manifest.resolve()
    log_path = args.manifest_build_log.resolve()
    protocol, protocol_sha256 = load_metric_protocol(protocol_path)
    for source in protocol["frozen_method_sources"].values():
        if isinstance(source, dict) and "path" in source:
            source_path = (REPO_ROOT / source["path"]).resolve()
            if file_sha256(source_path) != source["sha256"]:
                raise ValueError(f"frozen method source SHA mismatch: {source_path}")
    receipt = protocol["input_receipt"]
    if file_sha256(manifest_path) != receipt["input_manifest_sha256"]:
        raise ValueError("metric-validity manifest SHA mismatch")
    if manifest_path.stat().st_size != receipt["input_manifest_size_bytes"]:
        raise ValueError("metric-validity manifest size mismatch")
    if file_sha256(log_path) != receipt["manifest_build_log_sha256"]:
        raise ValueError("metric-validity manifest build log SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_ids_sha256 = canonical_sha256([case["case_id"] for case in manifest["cases"]])
    if case_ids_sha256 != receipt["case_ids_sha256"]:
        raise ValueError("metric-validity case-ID hash mismatch")
    screen_documents = set(manifest["partitions"]["screen"])
    screen_anchors = [row for row in manifest["anchors"] if row["partition"] == "screen"]
    if len(screen_documents) != 20 or len(screen_anchors) != 40:
        raise ValueError("metric-validity screen input count mismatch")
    if protocol["execution_boundary"]["holdout_method_metrics_authorized"] is not False:
        raise ValueError("metric-validity protocol unexpectedly authorizes holdout")
    report = {
        "schema_version": 1,
        "status": "pass",
        "claim_eligible": False,
        "protocol_sha256": protocol_sha256,
        "input_manifest_sha256": file_sha256(manifest_path),
        "manifest_build_log_sha256": file_sha256(log_path),
        "case_ids_sha256": case_ids_sha256,
        "screen_document_count": len(screen_documents),
        "screen_anchor_count": len(screen_anchors),
        "method_count": len(protocol["method_grid"]),
        "frozen_method_source_checks": "pass",
        "method_length_point_count": sum(len(method["points"]) for method in protocol["method_grid"]),
        "full_quality_case_count": protocol["execution_boundary"]["full_quality_cases"],
        "compressed_perturbation_case_count": protocol["execution_boundary"]["compressed_perturbation_cases"],
        "practical_effect_policy": protocol["practical_effect_policy"],
        "holdout_method_metrics_authorized": False,
        "cage_v4_candidate_execution_authorized": False,
        "full_gpu_execution_authorized": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
