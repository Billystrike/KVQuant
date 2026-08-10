from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from utils.qwen3_cases import token_ids_sha256


PROMPT_LENGTHS = (1024, 2048, 4032)
CONTINUATION_TOKENS = 64
MAXIMUM_WINDOW_TOKENS = 4096
ANCHORS_PER_DOCUMENT = 2
PARTITION_COUNTS = {"calibration": 10, "screen": 20, "holdout": 20}
PARTITION_ORDER = ("calibration", "screen", "holdout")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data_protocol(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_data_protocol(payload)
    return payload, file_sha256(path)


def derive_document_partitions(documents: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    if len(documents) != 50:
        raise ValueError("PG-19 data protocol requires exactly 50 documents")
    required = {"document_id", "document_identity_sha256", "token_count"}
    if any(not isinstance(row, Mapping) or not required.issubset(row) for row in documents):
        raise ValueError("PG-19 document identity records are incomplete")
    if len({row["document_id"] for row in documents}) != 50:
        raise ValueError("PG-19 document IDs must be unique")
    ordered = sorted(
        documents,
        key=lambda row: (row["token_count"], row["document_identity_sha256"]),
    )
    partitions = {name: [] for name in PARTITION_ORDER}
    for stratum_index in range(10):
        stratum = ordered[stratum_index * 5 : (stratum_index + 1) * 5]
        within = sorted(stratum, key=lambda row: row["document_identity_sha256"])
        partitions["calibration"].append(within[0]["document_id"])
        partitions["screen"].extend(row["document_id"] for row in within[1:3])
        partitions["holdout"].extend(row["document_id"] for row in within[3:5])
    if any(len(partitions[name]) != count for name, count in PARTITION_COUNTS.items()):
        raise AssertionError("derived PG-19 partition counts are invalid")
    return partitions


def nonoverlapping_prompt_starts(token_count: int) -> tuple[int, int]:
    if type(token_count) is not int or token_count < ANCHORS_PER_DOCUMENT * MAXIMUM_WINDOW_TOKENS:
        raise ValueError("document is too short for two nonoverlapping 4096-token windows")
    slack = token_count - ANCHORS_PER_DOCUMENT * MAXIMUM_WINDOW_TOKENS
    starts = (slack // 3, MAXIMUM_WINDOW_TOKENS + (2 * slack) // 3)
    if starts[0] + MAXIMUM_WINDOW_TOKENS > starts[1]:
        raise AssertionError("PG-19 deterministic windows overlap")
    if starts[1] + MAXIMUM_WINDOW_TOKENS > token_count:
        raise AssertionError("PG-19 deterministic window exceeds the document")
    return starts


def validate_data_protocol(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("PG-19 data protocol must be an object")
    if payload.get("schema_version") != 1 or payload.get("claim_eligible") is not False:
        raise ValueError("PG-19 data protocol schema/boundary mismatch")
    if payload.get("protocol_id") != "qwen3-8b-cage-v4-pg19-document-inputs-v1":
        raise ValueError("PG-19 data protocol identity mismatch")
    source = payload.get("source_receipt")
    if not isinstance(source, Mapping):
        raise ValueError("PG-19 source receipt is missing")
    expected_source = {
        "source_candidate_sha256": "a48dcb48f732a16e61afd2934739094ed0ee9d575ab71254d543d9251ff59d91",
        "token_audit_sha256": "3b295e239da2f1d53c6d55241f69ee2a207f8f7c34987e6ba679ff0b78eaa71c",
        "token_audit_log_sha256": "8260b9c5f0facb50df41047f861d8b41353d84d99088e702857a032472442e76",
        "parquet_sha256": "81680529564d4ead1c0e3859509a62d86c7126c32afc95dce6bd98e729e491ef",
        "ordered_document_identities_sha256": "2ee1222163f954af64f06603d4c5737d7e28231387ebc4b4af145202242d98ba",
        "ordered_document_token_hashes_sha256": "58e917d8bb76a77a884fb8af54b0d68566a71a375bf8abe0d549fad7084d892e",
        "document_count": 50,
        "total_qwen3_tokens": 4272631,
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise ValueError("PG-19 source receipt mismatch")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("PG-19 selection record is missing")
    if selection.get("prompt_lengths") != list(PROMPT_LENGTHS):
        raise ValueError("PG-19 prompt lengths mismatch")
    if selection.get("continuation_tokens") != CONTINUATION_TOKENS:
        raise ValueError("PG-19 continuation length mismatch")
    if selection.get("maximum_window_tokens") != MAXIMUM_WINDOW_TOKENS:
        raise ValueError("PG-19 maximum window mismatch")
    if selection.get("anchors_per_document") != ANCHORS_PER_DOCUMENT:
        raise ValueError("PG-19 anchors-per-document mismatch")
    partitions = payload.get("frozen_partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != set(PARTITION_ORDER):
        raise ValueError("PG-19 frozen partitions mismatch")
    if any(len(partitions[name]) != PARTITION_COUNTS[name] for name in PARTITION_ORDER):
        raise ValueError("PG-19 frozen partition counts mismatch")
    flattened = [document_id for name in PARTITION_ORDER for document_id in partitions[name]]
    if len(set(flattened)) != 50:
        raise ValueError("PG-19 frozen partitions must cover 50 unique documents")
    boundary = payload.get("access_boundary")
    expected_boundary = {
        "calibration_inputs_available": True,
        "screen_inputs_available": True,
        "holdout_inputs_may_be_materialized_without_method_metrics": True,
        "holdout_method_metrics_authorized": False,
        "metric_protocol_frozen": False,
        "cage_v4_method_frozen": False,
        "gpu_execution_authorized": False,
        "paper_claims_authorized": False,
    }
    if boundary != expected_boundary:
        raise ValueError("PG-19 access boundary mismatch")


def build_input_manifest(
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    audit: dict[str, Any],
    audit_sha256: str,
    token_ids_by_document: Mapping[str, Sequence[int]],
    tokenizer_identity: dict[str, Any],
    source_state: dict[str, Any],
) -> dict[str, Any]:
    validate_data_protocol(protocol)
    if audit_sha256 != protocol["source_receipt"]["token_audit_sha256"]:
        raise ValueError("PG-19 token audit file hash mismatch")
    if audit.get("status") != "pass" or audit.get("claim_eligible") is not False:
        raise ValueError("PG-19 token audit status/boundary mismatch")
    documents = audit.get("documents")
    if not isinstance(documents, list) or len(documents) != 50:
        raise ValueError("PG-19 token audit documents mismatch")
    derived = derive_document_partitions(documents)
    if derived != protocol["frozen_partitions"]:
        raise ValueError("PG-19 frozen partitions differ from the deterministic derivation")
    partition_by_document = {
        document_id: name
        for name in PARTITION_ORDER
        for document_id in derived[name]
    }
    document_records = []
    anchor_records = []
    case_records = []
    for document in sorted(documents, key=lambda row: row["row_index"]):
        document_id = document["document_id"]
        raw_tokens = token_ids_by_document.get(document_id)
        if raw_tokens is None:
            raise ValueError(f"missing tokens for {document_id}")
        token_ids = list(raw_tokens)
        if len(token_ids) != document["token_count"]:
            raise ValueError(f"token count mismatch for {document_id}")
        if token_ids_sha256(token_ids) != document["token_ids_sha256"]:
            raise ValueError(f"token hash mismatch for {document_id}")
        partition = partition_by_document[document_id]
        document_records.append(
            {
                "row_index": document["row_index"],
                "document_id": document_id,
                "partition": partition,
                "token_count": len(token_ids),
                "token_ids_sha256": document["token_ids_sha256"],
                "text_sha256": document["text_sha256"],
                "metadata": document["metadata"],
            }
        )
        for anchor_index, prompt_start in enumerate(nonoverlapping_prompt_starts(len(token_ids))):
            full_window = token_ids[prompt_start : prompt_start + MAXIMUM_WINDOW_TOKENS]
            continuation = full_window[-CONTINUATION_TOKENS:]
            cases = []
            for prompt_length in PROMPT_LENGTHS:
                prompt_offset = max(PROMPT_LENGTHS) - prompt_length
                prompt = full_window[prompt_offset : max(PROMPT_LENGTHS)]
                full = [*prompt, *continuation]
                identity = {
                    "protocol_id": protocol["protocol_id"],
                    "protocol_sha256": protocol_sha256,
                    "partition": partition,
                    "document_id": document_id,
                    "anchor_index": anchor_index,
                    "prompt_start": prompt_start,
                    "prompt_length": prompt_length,
                    "continuation_tokens": CONTINUATION_TOKENS,
                    "prompt_ids_sha256": token_ids_sha256(prompt),
                    "continuation_ids_sha256": token_ids_sha256(continuation),
                    "full_ids_sha256": token_ids_sha256(full),
                }
                case_id = canonical_sha256(identity)[:24]
                cases.append({"case_id": case_id, "prompt_length": prompt_length})
                case_records.append({"case_id": case_id, "identity": identity})
            anchor_records.append(
                {
                    "anchor_id": canonical_sha256(
                        {
                            "protocol_sha256": protocol_sha256,
                            "document_id": document_id,
                            "anchor_index": anchor_index,
                            "prompt_start": prompt_start,
                        }
                    )[:24],
                    "partition": partition,
                    "document_id": document_id,
                    "anchor_index": anchor_index,
                    "prompt_start": prompt_start,
                    "continuation_start": prompt_start + max(PROMPT_LENGTHS),
                    "full_window_ids_sha256": token_ids_sha256(full_window),
                    "full_window_ids": full_window,
                    "cases": cases,
                }
            )
    manifest = {
        "schema_version": 1,
        "manifest_id": "qwen3-8b-cage-v4-pg19-document-inputs-v1",
        "status": "frozen_inputs_before_metric_or_method_protocol",
        "claim_eligible": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "token_audit_sha256": audit_sha256,
        "source_state": source_state,
        "tokenizer_identity": tokenizer_identity,
        "partitions": derived,
        "documents": document_records,
        "anchors": anchor_records,
        "cases": case_records,
        "access_boundary": protocol["access_boundary"],
    }
    validate_input_manifest(manifest, protocol=protocol, protocol_sha256=protocol_sha256)
    return manifest


def validate_input_manifest(
    manifest: Mapping[str, Any], *, protocol: dict[str, Any], protocol_sha256: str
) -> None:
    validate_data_protocol(protocol)
    if manifest.get("schema_version") != 1 or manifest.get("claim_eligible") is not False:
        raise ValueError("PG-19 manifest schema/boundary mismatch")
    if manifest.get("protocol_sha256") != protocol_sha256:
        raise ValueError("PG-19 manifest protocol hash mismatch")
    if manifest.get("partitions") != protocol["frozen_partitions"]:
        raise ValueError("PG-19 manifest partitions mismatch")
    documents = manifest.get("documents")
    anchors = manifest.get("anchors")
    cases = manifest.get("cases")
    if not isinstance(documents, list) or len(documents) != 50:
        raise ValueError("PG-19 manifest document count mismatch")
    if not isinstance(anchors, list) or len(anchors) != 100:
        raise ValueError("PG-19 manifest anchor count mismatch")
    if not isinstance(cases, list) or len(cases) != 300:
        raise ValueError("PG-19 manifest case count mismatch")
    if len({case.get("case_id") for case in cases}) != 300:
        raise ValueError("PG-19 manifest case IDs must be unique")
    case_by_id = {case["case_id"]: case for case in cases}
    per_partition = {name: 0 for name in PARTITION_ORDER}
    per_document: dict[str, list[Mapping[str, Any]]] = {}
    for anchor in anchors:
        partition = anchor.get("partition")
        if partition not in per_partition:
            raise ValueError("PG-19 manifest anchor partition mismatch")
        per_partition[partition] += 1
        per_document.setdefault(anchor.get("document_id"), []).append(anchor)
        full_window = anchor.get("full_window_ids")
        if not isinstance(full_window, list) or len(full_window) != MAXIMUM_WINDOW_TOKENS:
            raise ValueError("PG-19 manifest full window length mismatch")
        if token_ids_sha256(full_window) != anchor.get("full_window_ids_sha256"):
            raise ValueError("PG-19 manifest full window hash mismatch")
        prompt_start = anchor.get("prompt_start")
        expected_start = nonoverlapping_prompt_starts(
            next(row["token_count"] for row in documents if row["document_id"] == anchor["document_id"])
        )[anchor.get("anchor_index")]
        if prompt_start != expected_start:
            raise ValueError("PG-19 manifest prompt start mismatch")
        for compact_case in anchor.get("cases", []):
            case = case_by_id.get(compact_case.get("case_id"))
            if case is None:
                raise ValueError("PG-19 manifest anchor references an unknown case")
            identity = case["identity"]
            prompt_length = compact_case.get("prompt_length")
            prompt_offset = max(PROMPT_LENGTHS) - prompt_length
            prompt = full_window[prompt_offset : max(PROMPT_LENGTHS)]
            continuation = full_window[-CONTINUATION_TOKENS:]
            full = [*prompt, *continuation]
            if identity.get("prompt_ids_sha256") != token_ids_sha256(prompt):
                raise ValueError("PG-19 manifest prompt hash mismatch")
            if identity.get("continuation_ids_sha256") != token_ids_sha256(continuation):
                raise ValueError("PG-19 manifest continuation hash mismatch")
            if identity.get("full_ids_sha256") != token_ids_sha256(full):
                raise ValueError("PG-19 manifest full case hash mismatch")
            if canonical_sha256(identity)[:24] != case["case_id"]:
                raise ValueError("PG-19 manifest case ID mismatch")
    expected_anchor_counts = {name: PARTITION_COUNTS[name] * 2 for name in PARTITION_ORDER}
    if per_partition != expected_anchor_counts:
        raise ValueError("PG-19 manifest per-partition anchor counts mismatch")
    if any(len(rows) != 2 for rows in per_document.values()) or len(per_document) != 50:
        raise ValueError("PG-19 manifest per-document anchor counts mismatch")


__all__ = [
    "ANCHORS_PER_DOCUMENT",
    "CONTINUATION_TOKENS",
    "MAXIMUM_WINDOW_TOKENS",
    "PARTITION_COUNTS",
    "PARTITION_ORDER",
    "PROMPT_LENGTHS",
    "build_input_manifest",
    "canonical_sha256",
    "derive_document_partitions",
    "file_sha256",
    "load_data_protocol",
    "nonoverlapping_prompt_starts",
    "validate_data_protocol",
    "validate_input_manifest",
]
