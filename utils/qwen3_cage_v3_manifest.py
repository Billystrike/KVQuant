from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from utils.qwen3_cage_v3_audit import candidate_selection_audit, token_ids_sha256
from utils.qwen3_cage_v3_protocol import ANCHOR_COUNT, PROMPT_LENGTHS, validate_cage_v3_protocol


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cage_v3_manifest(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    token_ids: Sequence[int],
    snapshot_sha256: str,
    token_audit: dict[str, Any],
    tokenizer_identity: dict[str, Any],
    source_state: dict[str, Any],
) -> dict[str, Any]:
    validate_cage_v3_protocol(protocol)
    data = protocol["development_data"]
    if len(token_ids) != data["token_count"] or token_ids_sha256(token_ids) != data["token_ids_sha256"]:
        raise ValueError("CAGE-v3 token identity differs from the frozen protocol")
    if snapshot_sha256 != data["snapshot_sha256"]:
        raise ValueError("CAGE-v3 snapshot hash differs from the frozen protocol")
    if token_audit.get("status") != "pass" or token_audit.get("claim_eligible") is not False:
        raise ValueError("CAGE-v3 token audit status/boundary mismatch")
    if token_audit.get("corpus", {}).get("token_ids_sha256") != data["token_ids_sha256"]:
        raise ValueError("CAGE-v3 token audit token hash differs")
    audits = token_audit.get("candidate_selection_audits", [])
    matching = [row for row in audits if row.get("anchor_count") == ANCHOR_COUNT]
    if len(matching) != 1:
        raise ValueError("CAGE-v3 token audit has no unique 50-anchor record")
    selection_audit = candidate_selection_audit(token_ids, ANCHOR_COUNT)
    if matching[0] != selection_audit:
        raise ValueError("CAGE-v3 50-anchor selection differs from the audited record")

    starts = selection_audit["continuation_starts"]
    cases = []
    anchor_records = []
    for anchor_index, start in enumerate(starts):
        continuation = list(token_ids[start : start + data["continuation_tokens"]])
        prompts = []
        for prompt_length in PROMPT_LENGTHS:
            prompt = list(token_ids[start - prompt_length : start])
            identity = {
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha256,
                "corpus_split": data["corpus_split"],
                "anchor_index": anchor_index,
                "continuation_start": start,
                "prompt_length": prompt_length,
                "prompt_ids_sha256": token_ids_sha256(prompt),
                "continuation_ids_sha256": token_ids_sha256(continuation),
            }
            case = {
                "case_id": canonical_sha256(identity)[:24],
                "identity": identity,
                "prompt_ids": prompt,
                "continuation_ids": continuation,
            }
            cases.append(case)
            prompts.append({
                "prompt_length": prompt_length,
                "case_id": case["case_id"],
                "prompt_ids_sha256": identity["prompt_ids_sha256"],
            })
        anchor_records.append({
            "anchor_index": anchor_index,
            "continuation_start": start,
            "full_window_ids_sha256": selection_audit["full_window_ids_sha256"][anchor_index],
            "continuation_ids_sha256": token_ids_sha256(continuation),
            "prompts": prompts,
        })
    manifest = {
        "schema_version": 1,
        "manifest_id": "qwen3-8b-cage-v3-train-development-inputs-v1",
        "status": "development_only_with_holdout_and_reserved_boundaries_frozen",
        "claim_eligible": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "snapshot_sha256": snapshot_sha256,
        "token_audit_sha256": data["token_audit_sha256"],
        "source_state": source_state,
        "tokenizer_identity": tokenizer_identity,
        "corpus_identity": {
            "id": data["corpus_id"],
            "config": data["corpus_config"],
            "split": data["corpus_split"],
            "revision": data["corpus_revision"],
            "dataset_fingerprint": data["dataset_fingerprint"],
            "join_separator": data["join_separator"],
        },
        "token_count": len(token_ids),
        "token_ids_sha256": token_ids_sha256(token_ids),
        "selection": {
            "selection_id": data["selection_id"],
            "anchor_count": ANCHOR_COUNT,
            "minimum_anchor_gap": selection_audit["minimum_anchor_gap"],
            "maximum_window_tokens": data["maximum_window_tokens"],
            "partitions": data["partitions"],
            "anchors": anchor_records,
        },
        "cases": cases,
    }
    validate_cage_v3_manifest(manifest, protocol=protocol, protocol_sha256=protocol_sha256)
    return manifest


def validate_cage_v3_manifest(
    manifest: dict[str, Any], *, protocol: dict[str, Any], protocol_sha256: str
) -> None:
    validate_cage_v3_protocol(protocol)
    data = protocol["development_data"]
    if manifest.get("schema_version") != 1 or manifest.get("claim_eligible") is not False:
        raise ValueError("CAGE-v3 manifest schema/boundary mismatch")
    if manifest.get("status") != "development_only_with_holdout_and_reserved_boundaries_frozen":
        raise ValueError("CAGE-v3 manifest status mismatch")
    if manifest.get("protocol_id") != protocol["protocol_id"] or manifest.get("protocol_sha256") != protocol_sha256:
        raise ValueError("CAGE-v3 manifest protocol identity mismatch")
    if manifest.get("snapshot_sha256") != data["snapshot_sha256"]:
        raise ValueError("CAGE-v3 manifest snapshot mismatch")
    if manifest.get("token_audit_sha256") != data["token_audit_sha256"]:
        raise ValueError("CAGE-v3 manifest token audit mismatch")
    if manifest.get("token_count") != data["token_count"] or manifest.get("token_ids_sha256") != data["token_ids_sha256"]:
        raise ValueError("CAGE-v3 manifest token identity mismatch")
    selection = manifest.get("selection", {})
    if selection.get("partitions") != data["partitions"] or selection.get("anchor_count") != ANCHOR_COUNT:
        raise ValueError("CAGE-v3 manifest anchor partition mismatch")
    if selection.get("minimum_anchor_gap") != data["minimum_anchor_gap"]:
        raise ValueError("CAGE-v3 manifest anchor gap mismatch")
    anchors = selection.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != ANCHOR_COUNT:
        raise ValueError("CAGE-v3 manifest anchor records mismatch")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != ANCHOR_COUNT * len(PROMPT_LENGTHS):
        raise ValueError("CAGE-v3 manifest case count mismatch")
    if len({case.get("case_id") for case in cases}) != len(cases):
        raise ValueError("CAGE-v3 manifest case IDs are not unique")
    for case in cases:
        identity = case.get("identity", {})
        prompt = case.get("prompt_ids")
        continuation = case.get("continuation_ids")
        if not isinstance(prompt, list) or len(prompt) != identity.get("prompt_length"):
            raise ValueError("CAGE-v3 manifest prompt length mismatch")
        if not isinstance(continuation, list) or len(continuation) != data["continuation_tokens"]:
            raise ValueError("CAGE-v3 manifest continuation length mismatch")
        if token_ids_sha256(prompt) != identity.get("prompt_ids_sha256"):
            raise ValueError("CAGE-v3 manifest prompt hash mismatch")
        if token_ids_sha256(continuation) != identity.get("continuation_ids_sha256"):
            raise ValueError("CAGE-v3 manifest continuation hash mismatch")
        if canonical_sha256(identity)[:24] != case.get("case_id"):
            raise ValueError("CAGE-v3 manifest case ID mismatch")


__all__ = [
    "build_cage_v3_manifest",
    "canonical_sha256",
    "validate_cage_v3_manifest",
]
