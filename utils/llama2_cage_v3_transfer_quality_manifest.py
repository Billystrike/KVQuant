from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from utils.qwen3_cases import continuation_anchor, token_ids_sha256
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


ARTIFACT_ID = "llama2-7b-cage-v3-transfer-quality-static-preflight-artifacts-v1"
EXPECTED_ARTIFACT_MANIFEST_SHA256 = "dfd7acb6b69a5d6f6da1659df3e4626216eac04a0709f36b77b20ab56cdc9a48"
EXPECTED_PREFLIGHT_SHA256 = "238508a3a1dbaaf3d81bc1d353c848e46d653c093c3972c4048c19e27c7a22e2"
EXPECTED_PROTOCOL_SHA256 = "355d73629dece9015288d10acdd7c95e08da0cff941a38395d54e1868380b654"
MANIFEST_ID = "llama2-7b-cage-v3-transfer-quality-input-manifest-v1"


class Llama2CageV3TransferQualityManifestError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3TransferQualityManifestError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3TransferQualityManifestError(
            f"cannot load JSON {path}: {error}"
        ) from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_corpus_snapshot(
    *,
    input_spec: Mapping[str, Any],
    texts: Sequence[str],
    tokenizer: Any,
    dataset_fingerprint: str | None,
    datasets_version: str,
) -> tuple[dict[str, Any], list[int]]:
    _require(not isinstance(texts, (str, bytes)), "WikiText rows must be a sequence")
    _require(all(isinstance(text, str) for text in texts), "WikiText rows must be strings")
    joined = input_spec["join_separator"].join(texts)
    joined_bytes = joined.encode("utf-8")
    encoded = tokenizer(joined, add_special_tokens=input_spec["add_special_tokens"])
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    _require(
        isinstance(token_ids, list)
        and all(type(value) is int and value >= 0 for value in token_ids),
        "tokenizer output must be a nonnegative integer list",
    )
    actual = {
        "rows": len(texts),
        "nonempty_rows": sum(bool(text.strip()) for text in texts),
        "joined_utf8_bytes": len(joined_bytes),
        "joined_text_sha256": hashlib.sha256(joined_bytes).hexdigest(),
        "token_count": len(token_ids),
        "token_ids_sha256": token_ids_sha256(token_ids),
    }
    expected = {
        "rows": input_spec["expected_rows"],
        "nonempty_rows": input_spec["expected_nonempty_rows"],
        "joined_utf8_bytes": input_spec["expected_joined_utf8_bytes"],
        "joined_text_sha256": input_spec["expected_joined_text_sha256"],
        "token_count": input_spec["expected_token_count"],
        "token_ids_sha256": input_spec["expected_token_ids_sha256"],
    }
    _require(actual == expected, f"WikiText corpus/token identity mismatch: {actual!r}")
    return (
        {
            **actual,
            "dataset_fingerprint": dataset_fingerprint,
            "datasets_version": datasets_version,
            "tokenizer_type": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        },
        token_ids,
    )


def validate_static_preflight_artifacts(
    manifest: Mapping[str, Any], *, artifact_path: Path, verify_server_files: bool
) -> dict[str, Any] | None:
    _require(file_sha256(artifact_path) == EXPECTED_ARTIFACT_MANIFEST_SHA256, "static artifact manifest file changed")
    _require(manifest.get("schema_version") == 1 and manifest.get("artifact_id") == ARTIFACT_ID, "static artifact identity changed")
    _require(manifest.get("status") == "pass" and manifest.get("claim_eligible") is False, "static artifact status changed")
    _require(manifest.get("execution_source_commit") == "ea966b368c63b36ec830b67f3de34f3dea643304", "static source commit changed")
    _require(manifest.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "static protocol link changed")
    preflight_spec = manifest.get("preflight", {})
    _require(
        preflight_spec
        == {
            "path": "/root/autodl-tmp/kitty_setup_audit/llama2_cage_v3_transfer_quality_static_preflight_ea966b3.json",
            "sha256": EXPECTED_PREFLIGHT_SHA256,
            "size_bytes": 4575,
        },
        "static preflight artifact changed",
    )
    log_spec = manifest.get("execution_log", {})
    _require(log_spec.get("sha256") == "37ca6bf0d398fa400fef5b41db52335f381d5ce00c1c8114402e49fda1b345b7", "static log digest changed")
    _require(log_spec.get("size_bytes") == 11752, "static log size changed")
    decision = manifest.get("decision", {})
    for key in (
        "static_preflight_pass",
        "exact_input_manifest_build_authorized",
        "corpus_access_limited_to_frozen_identity_and_anchor_materialization",
    ):
        _require(decision.get(key) is True, f"static manifest authorization changed: {key}")
    for key in (
        "model_weights_access_authorized",
        "quality_metrics_authorized",
        "quality_acceptance_execution_authorized",
        "full_600_case_execution_authorized",
    ):
        _require(decision.get(key) is False, f"static manifest boundary changed: {key}")
    if not verify_server_files:
        return None
    preflight_path = Path(preflight_spec["path"])
    log_path = Path(log_spec["path"])
    _require(preflight_path.is_file(), "static preflight JSON is missing")
    _require(log_path.is_file(), "static preflight log is missing")
    _require(preflight_path.stat().st_size == preflight_spec["size_bytes"], "static preflight size mismatch")
    _require(file_sha256(preflight_path) == preflight_spec["sha256"], "static preflight digest mismatch")
    _require(log_path.stat().st_size == log_spec["size_bytes"], "static log size mismatch")
    _require(file_sha256(log_path) == log_spec["sha256"], "static log digest mismatch")
    preflight = _load(preflight_path)
    _require(preflight.get("status") == "pass", "static preflight did not pass")
    _require(preflight.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "static preflight protocol mismatch")
    _require(preflight.get("source_state") == {"dirty": False, "git_commit": "ea966b368c63b36ec830b67f3de34f3dea643304"}, "static preflight source mismatch")
    _require(all(preflight.get("checks", {}).values()), "static preflight checks are incomplete")
    boundary = preflight.get("boundary", {})
    _require(boundary.get("exact_input_manifest_build_authorized") is True, "manifest build not authorized")
    for key in (
        "gpu_used",
        "model_weights_accessed",
        "corpus_accessed",
        "quality_metrics_computed_or_read",
        "quality_acceptance_execution_authorized",
        "full_600_case_execution_authorized",
    ):
        _require(boundary.get(key) is False, f"static preflight boundary changed: {key}")
    return preflight


def load_static_preflight_artifacts(
    path: Path, *, verify_server_files: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    value = _load(path)
    preflight = validate_static_preflight_artifacts(
        value, artifact_path=path, verify_server_files=verify_server_files
    )
    return value, preflight


def build_input_manifest(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    token_ids: Sequence[int],
    corpus_snapshot: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    input_spec = protocol["input"]
    _require(protocol_sha256 == EXPECTED_PROTOCOL_SHA256, "input manifest protocol changed")
    _require(len(token_ids) == input_spec["expected_token_count"], "token count mismatch")
    _require(token_ids_sha256(token_ids) == input_spec["expected_token_ids_sha256"], "token stream mismatch")
    _require(source_state.get("dirty") is False, "input manifest requires clean source")
    prompt_lengths = input_spec["prompt_lengths"]
    continuation_tokens = input_spec["continuation_tokens"]
    records = []
    for anchor_index in input_spec["anchor_indices"]:
        _require(input_spec["selection_id"] == "interior-51sts-v1", "selection rule changed")
        start = continuation_anchor(len(token_ids), anchor_index)
        continuation = list(token_ids[start : start + continuation_tokens])
        for prompt_length in prompt_lengths:
            prompt = list(token_ids[start - prompt_length : start])
            full = [*prompt, *continuation]
            _require(len(prompt) == prompt_length and len(continuation) == continuation_tokens, "input window shape mismatch")
            identity = {
                "protocol_sha256": protocol_sha256,
                "corpus_token_ids_sha256": input_spec["expected_token_ids_sha256"],
                "selection_id": input_spec["selection_id"],
                "anchor_index": anchor_index,
                "continuation_start": start,
                "prompt_length": prompt_length,
                "continuation_tokens": continuation_tokens,
                "prompt_ids_sha256": token_ids_sha256(prompt),
                "continuation_ids_sha256": token_ids_sha256(continuation),
                "full_ids_sha256": token_ids_sha256(full),
            }
            records.append(
                {
                    "input_id": canonical_sha256(identity)[:24],
                    "identity": identity,
                    "prompt_ids": prompt,
                    "continuation_ids": continuation,
                }
            )
    manifest = {
        "schema_version": 1,
        "manifest_id": MANIFEST_ID,
        "status": "frozen_inputs_only",
        "claim_eligible": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "static_preflight_sha256": EXPECTED_PREFLIGHT_SHA256,
        "source_state": dict(source_state),
        "model_reference": protocol["model"]["reference"],
        "tokenizer_identity": dict(tokenizer_identity),
        "corpus_snapshot": dict(corpus_snapshot),
        "selection": {
            "selection_id": input_spec["selection_id"],
            "anchor_count": len(input_spec["anchor_indices"]),
            "anchor_indices": input_spec["anchor_indices"],
            "prompt_lengths": prompt_lengths,
            "continuation_tokens": continuation_tokens,
            "input_record_count": len(records),
            "expanded_method_case_count": protocol["case_matrix"]["full_case_count"],
        },
        "inputs": records,
        "boundary": {
            "model_weights_accessed": False,
            "gpu_used": False,
            "quality_metrics_computed_or_read": False,
            "candidate_tuning_performed": False,
            "quality_acceptance_execution_authorized": False,
            "full_600_case_execution_authorized": False,
            "paper_claims_authorized": False,
        },
    }
    validate_input_manifest(manifest, protocol=protocol, token_ids=token_ids)
    return manifest


def validate_input_manifest(
    manifest: Mapping[str, Any], *, protocol: Mapping[str, Any], token_ids: Sequence[int] | None = None
) -> None:
    _require(manifest.get("schema_version") == 1 and manifest.get("manifest_id") == MANIFEST_ID, "input manifest identity mismatch")
    _require(manifest.get("status") == "frozen_inputs_only" and manifest.get("claim_eligible") is False, "input manifest status mismatch")
    _require(manifest.get("protocol_id") == protocol["protocol_id"], "input manifest protocol ID mismatch")
    _require(manifest.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "input manifest protocol hash mismatch")
    _require(manifest.get("static_preflight_sha256") == EXPECTED_PREFLIGHT_SHA256, "input manifest preflight mismatch")
    input_spec = protocol["input"]
    snapshot = manifest.get("corpus_snapshot", {})
    expected_snapshot = {
        "rows": input_spec["expected_rows"],
        "nonempty_rows": input_spec["expected_nonempty_rows"],
        "joined_utf8_bytes": input_spec["expected_joined_utf8_bytes"],
        "joined_text_sha256": input_spec["expected_joined_text_sha256"],
        "token_count": input_spec["expected_token_count"],
        "token_ids_sha256": input_spec["expected_token_ids_sha256"],
    }
    for key, expected in expected_snapshot.items():
        _require(snapshot.get(key) == expected, f"corpus snapshot changed: {key}")
    tokenizer = manifest.get("tokenizer_identity", {})
    _require(tokenizer.get("use_fast") is False, "tokenizer fast/slow path changed")
    expected_tokenizer_hashes = {
        "tokenizer_config.json": "f514e7c3008881b6ba7e6a0cdb44c71ce47dc335920dac143ae7bc788197e53a",
        "tokenizer.json": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
        "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
        "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
    }
    tokenizer_files = tokenizer.get("files", {})
    for name, expected_sha in expected_tokenizer_hashes.items():
        _require(
            tokenizer_files.get(name, {}).get("sha256") == expected_sha,
            f"tokenizer identity changed: {name}",
        )
    selection = manifest.get("selection", {})
    _require(selection.get("anchor_indices") == list(range(50)), "manifest anchors changed")
    _require(selection.get("prompt_lengths") == [1024, 2048, 4032], "manifest lengths changed")
    _require(selection.get("continuation_tokens") == 64, "manifest continuation changed")
    _require(selection.get("input_record_count") == 150, "manifest input count changed")
    _require(selection.get("expanded_method_case_count") == 600, "expanded case count changed")
    records = manifest.get("inputs", [])
    _require(isinstance(records, list) and len(records) == 150, "input records are incomplete")
    _require(len({record.get("input_id") for record in records}) == 150, "input IDs are not unique")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        identity = record.get("identity", {})
        prompt = record.get("prompt_ids")
        continuation = record.get("continuation_ids")
        _require(isinstance(prompt, list) and all(type(value) is int and value >= 0 for value in prompt), "prompt IDs are invalid")
        _require(isinstance(continuation, list) and len(continuation) == 64 and all(type(value) is int and value >= 0 for value in continuation), "continuation IDs are invalid")
        _require(len(prompt) == identity.get("prompt_length"), "prompt length mismatch")
        _require(token_ids_sha256(prompt) == identity.get("prompt_ids_sha256"), "prompt hash mismatch")
        _require(token_ids_sha256(continuation) == identity.get("continuation_ids_sha256"), "continuation hash mismatch")
        _require(token_ids_sha256([*prompt, *continuation]) == identity.get("full_ids_sha256"), "full input hash mismatch")
        _require(record.get("input_id") == canonical_sha256(identity)[:24], "input ID mismatch")
        grouped[identity.get("anchor_index")].append(record)
        if token_ids is not None:
            start = identity["continuation_start"]
            length = identity["prompt_length"]
            _require(prompt == list(token_ids[start - length : start]), "prompt does not match corpus stream")
            _require(continuation == list(token_ids[start : start + 64]), "continuation does not match corpus stream")
    _require(sorted(grouped) == list(range(50)), "grouped anchor indices changed")
    starts = []
    for anchor_index in range(50):
        group = grouped[anchor_index]
        _require(sorted(record["identity"]["prompt_length"] for record in group) == [1024, 2048, 4032], "per-anchor lengths changed")
        group_starts = {record["identity"]["continuation_start"] for record in group}
        _require(len(group_starts) == 1, "per-anchor continuation starts differ")
        _require(len({tuple(record["continuation_ids"]) for record in group}) == 1, "per-anchor continuations differ")
        start = next(iter(group_starts))
        _require(start == continuation_anchor(input_spec["expected_token_count"], anchor_index), "anchor formula changed")
        starts.append(start)
    _require(min(right - left for left, right in zip(starts, starts[1:])) >= 4096, "maximum-context anchor windows overlap")
    boundary = manifest.get("boundary", {})
    for key in (
        "model_weights_accessed",
        "gpu_used",
        "quality_metrics_computed_or_read",
        "candidate_tuning_performed",
        "quality_acceptance_execution_authorized",
        "full_600_case_execution_authorized",
        "paper_claims_authorized",
    ):
        _require(boundary.get(key) is False, f"input manifest boundary changed: {key}")


__all__ = [
    "EXPECTED_ARTIFACT_MANIFEST_SHA256",
    "EXPECTED_PREFLIGHT_SHA256",
    "Llama2CageV3TransferQualityManifestError",
    "build_input_manifest",
    "load_static_preflight_artifacts",
    "validate_input_manifest",
    "validate_corpus_snapshot",
    "validate_static_preflight_artifacts",
]
