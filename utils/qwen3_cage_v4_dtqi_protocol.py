from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes
from utils.qwen3_cage_v3_screen import validate_quota_plan


PROTOCOL_ID = "qwen3-8b-cage-v4-dtqi-single-candidate-v1"
PROMPT_LENGTHS = (1024, 2048, 4032)
RESIDUAL_LENGTHS = (176, 288, 112)
PACKED_BYTES = (45_849_600, 70_050_816, 110_066_688)
KITTY_PRO_BYTES = (47_890_080, 71_782_560, 110_130_336)
EXPECTED_ANALYSIS_RECEIPT_SHA256 = "fdd363b822d3a34e3c6e078410290d792a2ef608c7d48042c42e4d73737c4268"
EXPECTED_ANALYSIS_MANIFEST_SHA256 = "2a99b30a750686a13330b194864e2e9fa451d920ad028b10d8dc5294af7afb4e"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dtqi_protocol(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    source = Path(path).resolve()
    protocol = json.loads(source.read_text(encoding="utf-8"))
    validate_dtqi_protocol(
        protocol,
        repo_root=Path(repo_root).resolve() if repo_root is not None else source.parents[1],
    )
    return protocol, file_sha256(source)


def validate_dtqi_protocol(protocol: dict[str, Any], *, repo_root: str | Path) -> None:
    root = Path(repo_root)
    if protocol.get("schema_version") != 1 or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("CAGE-v4-DTQI protocol identity mismatch")
    if protocol.get("status") != "frozen_after_metric_screen_before_any_candidate_metric":
        raise ValueError("CAGE-v4-DTQI protocol status mismatch")
    if protocol.get("claim_eligible") is not False:
        raise ValueError("CAGE-v4-DTQI development protocol must be claim-ineligible")

    evidence = protocol.get("metric_screen_evidence", {})
    if evidence.get("results_receipt_sha256") != EXPECTED_ANALYSIS_RECEIPT_SHA256:
        raise ValueError("CAGE-v4-DTQI results receipt changed")
    if evidence.get("analysis_manifest_sha256") != EXPECTED_ANALYSIS_MANIFEST_SHA256:
        raise ValueError("CAGE-v4-DTQI analysis manifest changed")
    if evidence.get("local_proxy_gate_pass") is not False:
        raise ValueError("CAGE-v4-DTQI must retain the failed local-proxy gate")
    if evidence.get("candidate_selection_performed") is not False:
        raise ValueError("CAGE-v4 candidate was selected before protocol freeze")
    if evidence.get("holdout_accessed") is not False:
        raise ValueError("CAGE-v4 holdout boundary changed")

    method = protocol.get("candidate", {})
    if method.get("candidate_id") != "cage-v4-dtqi":
        raise ValueError("CAGE-v4-DTQI candidate identity changed")
    if method.get("only_candidate") is not True:
        raise ValueError("CAGE-v4-DTQI must remain the only candidate")
    if method.get("key_importance") != {
        "global_query_weight": 0.5,
        "recent_query_weight": 0.5,
        "recent_query_window": "equal_to_frozen_residual_length_at_each_point",
        "key_variance_window": "complete_prefill_prompt",
        "ranking_scope": "prompt_persistent_per_layer_per_kv_head",
    }:
        raise ValueError("CAGE-v4-DTQI importance definition changed")
    fixed = method.get("inherited_storage", {})
    if fixed != {
        "source": "cage-v3-sr2-sink32-calibrated",
        "key_base_bits": 2,
        "key_refinement_bits": 2,
        "value_bits": 2,
        "key_base_group_size": 128,
        "key_refinement_group_size": 128,
        "value_group_size": 128,
        "sink_length": 32,
        "one_bit_channels": 0,
        "value_adaptive": False,
        "packed_representation_changed": False,
    }:
        raise ValueError("CAGE-v4-DTQI inherited storage changed")

    quota_spec = protocol.get("quota_plan", {})
    quota_path = root / quota_spec.get("path", "")
    if not quota_path.is_file() or file_sha256(quota_path) != quota_spec.get("sha256"):
        raise ValueError("CAGE-v4-DTQI quota-plan source changed")
    quota_plan = json.loads(quota_path.read_text(encoding="utf-8"))
    validate_quota_plan(quota_plan)

    points = method.get("points", [])
    expected_points = list(zip(PROMPT_LENGTHS, RESIDUAL_LENGTHS, PACKED_BYTES, KITTY_PRO_BYTES))
    if len(points) != len(expected_points):
        raise ValueError("CAGE-v4-DTQI point count changed")
    for point, (length, residual, packed, target) in zip(points, expected_points):
        if point != {
            "prompt_length": length,
            "residual_length": residual,
            "recent_query_window": residual,
            "packed_bytes": packed,
            "kitty_pro_target_bytes": target,
        }:
            raise ValueError("CAGE-v4-DTQI point definition changed")
        quotas = _quotas_for_length(quota_plan, length)
        report = estimate_qwen3_cage_v2_bytes(
            seq_len=length,
            residual_length=residual,
            one_bit_channels=0,
            two_bit_channels=quotas,
            sink_length=32,
        )
        if report["model_total_bytes"] != packed or packed > target:
            raise ValueError("CAGE-v4-DTQI packed-memory identity changed")

    success = protocol.get("success_policy", {})
    if success != {
        "primary_external_baseline": "kitty-pro-25pct",
        "predecessor_baseline": "cage-v3-sr2-sink32-calibrated",
        "memory": "candidate packed bytes must not exceed Kitty-Pro at all three prompt lengths",
        "overall_material_superiority_relative_ppl_percent": -1.0,
        "uncertainty": "paired document-cluster bootstrap 95 percent upper bound for mean NLL delta must be below zero versus both baselines",
        "per_length_noninferiority_margin_relative_ppl_percent": 0.5,
        "if_candidate_fails": "close CAGE-v4 as negative; do not access holdout metrics or reinterpret local MSE",
    }:
        raise ValueError("CAGE-v4-DTQI success policy changed")

    boundary = protocol.get("execution_boundary", {})
    expected_false = (
        "gpu_acceptance_authorized",
        "gpu_full_screen_authorized",
        "holdout_method_metrics_authorized",
        "pg19_test_access_authorized",
        "runtime_claims_authorized",
        "paper_claims_authorized",
    )
    if boundary.get("cpu_acceptance_authorized") is not True:
        raise ValueError("CAGE-v4-DTQI CPU acceptance is not authorized")
    if any(boundary.get(name) is not False for name in expected_false):
        raise ValueError("CAGE-v4-DTQI execution boundary changed")


def _quotas_for_length(plan: dict[str, Any], prompt_length: int) -> list[int]:
    matches = [
        row
        for row in plan["plans"]
        if row["family_id"] == "pure-sr2-sink32-uniform32"
        and row["prompt_length"] == prompt_length
    ]
    if len(matches) != 1:
        raise ValueError("matching CAGE-v3 sink32 quota plan missing")
    return list(matches[0]["layer_two_bit_channel_quotas"])


__all__ = [
    "EXPECTED_ANALYSIS_MANIFEST_SHA256",
    "EXPECTED_ANALYSIS_RECEIPT_SHA256",
    "KITTY_PRO_BYTES",
    "PACKED_BYTES",
    "PROMPT_LENGTHS",
    "PROTOCOL_ID",
    "RESIDUAL_LENGTHS",
    "file_sha256",
    "load_dtqi_protocol",
    "validate_dtqi_protocol",
]
