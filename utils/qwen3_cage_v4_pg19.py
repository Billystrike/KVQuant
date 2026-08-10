from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EXPECTED_SOURCE = {
    "repo_id": "emozilla/pg19",
    "repo_type": "dataset",
    "revision": "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8",
    "filename": "data/validation-00000-of-00001-0f92e2337f79aeac.parquet",
    "size_bytes": 10_803_864,
    "sha256": "81680529564d4ead1c0e3859509a62d86c7126c32afc95dce6bd98e729e491ef",
    "split": "validation",
    "row_count": 50,
    "text_column": "text",
}
EXPECTED_MODEL = {
    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer_json_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "vocab_sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges_sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}
AUDIT_FULL_WINDOW_THRESHOLDS = (1088, 2112, 4096)
EXPECTED_AUDIT_ENVIRONMENT = {
    "python": "3.10.20",
    "huggingface_hub": "0.36.2",
    "pyarrow": "24.0.0",
    "transformers": "4.53.2",
}
EXPECTED_PRE_DOWNLOAD_AUDIT = {
    "source_commit": "6081d829b2a3628e3dc7b9c91491eedcd5a086e7",
    "log_path": "/root/autodl-tmp/kitty_setup_audit/pg19_pre_download_audit_6081d82.log",
    "log_sha256": "c7d7f1b9a6ac30b2b024e3f3287455e6c5759460f4009e4f6a8a0613f5102751",
    "log_size_bytes": 4976,
    "status": "pass",
}


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


def load_source_candidate(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_source_candidate(payload)
    return payload, file_sha256(path)


def validate_source_candidate(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("PG-19 source candidate must be an object")
    if payload.get("schema_version") != 1 or payload.get("claim_eligible") is not False:
        raise ValueError("PG-19 source candidate boundary mismatch")
    if payload.get("candidate_id") != "qwen3-cage-v4-pg19-validation-source-candidate-v1":
        raise ValueError("PG-19 candidate identity mismatch")
    mirror = payload.get("mirror_snapshot")
    if not isinstance(mirror, Mapping):
        raise ValueError("PG-19 mirror snapshot is missing")
    for key, expected in EXPECTED_SOURCE.items():
        if mirror.get(key) != expected:
            raise ValueError(f"PG-19 mirror snapshot {key} mismatch")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Qwen3 model identity is missing")
    for key, expected in EXPECTED_MODEL.items():
        if model.get(key) != expected:
            raise ValueError(f"Qwen3 model {key} mismatch")
    official = payload.get("official_source")
    if not isinstance(official, Mapping):
        raise ValueError("PG-19 official source record is missing")
    if official.get("license_spdx") != "Apache-2.0":
        raise ValueError("PG-19 official license mismatch")
    if mirror.get("license_metadata_status") != "absent_in_hugging_face_repo_tags":
        raise ValueError("PG-19 mirror license metadata status mismatch")
    if payload.get("pre_download_audit") != EXPECTED_PRE_DOWNLOAD_AUDIT:
        raise ValueError("PG-19 pre-download audit receipt mismatch")
    thresholds = payload.get("audit", {}).get("full_window_token_thresholds")
    if thresholds != list(AUDIT_FULL_WINDOW_THRESHOLDS):
        raise ValueError("PG-19 audit thresholds mismatch")
    boundary = payload.get("freeze_boundary")
    required_false = {
        "documents_selected",
        "development_partitions_frozen",
        "metric_protocol_frozen",
        "cage_v4_method_frozen",
        "gpu_execution_authorized",
        "paper_claims_authorized",
    }
    if not isinstance(boundary, Mapping) or any(boundary.get(key) is not False for key in required_false):
        raise ValueError("PG-19 source candidate must not freeze downstream decisions")
    environment = payload.get("environment_boundary")
    if not isinstance(environment, Mapping):
        raise ValueError("PG-19 environment boundary is missing")
    if environment.get("expected_versions") != EXPECTED_AUDIT_ENVIRONMENT:
        raise ValueError("PG-19 audit environment versions mismatch")


def audit_documents(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    text_column: str = "text",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != EXPECTED_SOURCE["row_count"]:
        raise ValueError(f"PG-19 validation must contain {EXPECTED_SOURCE['row_count']} rows")
    documents: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("PG-19 rows must be objects")
        text = row.get(text_column)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"PG-19 row {row_index} has invalid text")
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
        token_ids = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        if any(type(value) is not int or value < 0 for value in token_ids):
            raise ValueError(f"PG-19 row {row_index} produced invalid token ids")
        text_bytes = text.encode("utf-8")
        text_sha256 = hashlib.sha256(text_bytes).hexdigest()
        token_ids_sha256 = canonical_sha256(token_ids)
        metadata = {
            key: _json_scalar(value)
            for key, value in sorted(row.items())
            if key != text_column
        }
        identity_payload = {
            "source_revision": EXPECTED_SOURCE["revision"],
            "split": EXPECTED_SOURCE["split"],
            "row_index": row_index,
            "metadata": metadata,
            "text_sha256": text_sha256,
        }
        documents.append(
            {
                "row_index": row_index,
                "document_id": f"pg19-validation-{row_index:02d}-{canonical_sha256(identity_payload)[:16]}",
                "document_identity_sha256": canonical_sha256(identity_payload),
                "metadata": metadata,
                "utf8_bytes": len(text_bytes),
                "unicode_characters": len(text),
                "text_sha256": text_sha256,
                "token_count": len(token_ids),
                "token_ids_sha256": token_ids_sha256,
                "eligible_full_window_tokens": {
                    str(threshold): len(token_ids) >= threshold
                    for threshold in AUDIT_FULL_WINDOW_THRESHOLDS
                },
            }
        )
    token_counts = [record["token_count"] for record in documents]
    text_hashes = [record["text_sha256"] for record in documents]
    token_hashes = [record["token_ids_sha256"] for record in documents]
    summary = {
        "document_count": len(documents),
        "total_utf8_bytes": sum(record["utf8_bytes"] for record in documents),
        "total_unicode_characters": sum(record["unicode_characters"] for record in documents),
        "total_qwen3_tokens": sum(token_counts),
        "token_count_distribution": _integer_distribution(token_counts),
        "eligible_document_counts": {
            str(threshold): sum(count >= threshold for count in token_counts)
            for threshold in AUDIT_FULL_WINDOW_THRESHOLDS
        },
        "duplicate_text_sha256_groups": _duplicate_groups(text_hashes),
        "duplicate_token_ids_sha256_groups": _duplicate_groups(token_hashes),
        "ordered_document_identities_sha256": canonical_sha256(
            [record["document_identity_sha256"] for record in documents]
        ),
        "ordered_document_token_hashes_sha256": canonical_sha256(token_hashes),
    }
    return documents, summary


def _integer_distribution(values: Sequence[int]) -> dict[str, int | float]:
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("distribution values must be nonnegative integers")
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    median = (
        ordered[middle]
        if count % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "minimum": ordered[0],
        "median": median,
        "maximum": ordered[-1],
        "mean": sum(ordered) / count,
    }


def _duplicate_groups(values: Sequence[str]) -> list[list[int]]:
    positions: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        positions.setdefault(value, []).append(index)
    return [indices for _, indices in sorted(positions.items()) if len(indices) > 1]


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "as_py"):
        return _json_scalar(value.as_py())
    return str(value)


__all__ = [
    "AUDIT_FULL_WINDOW_THRESHOLDS",
    "EXPECTED_AUDIT_ENVIRONMENT",
    "EXPECTED_MODEL",
    "EXPECTED_PRE_DOWNLOAD_AUDIT",
    "EXPECTED_SOURCE",
    "audit_documents",
    "canonical_sha256",
    "file_sha256",
    "load_source_candidate",
    "validate_source_candidate",
]
