from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


COMPARE_FIELDS = (
    "case_id",
    "method",
    "input",
    "memory",
    "layer_metrics",
    "aggregates",
    "cache",
)
PARTITIONS = ("cage_qwen3", "kitty_qwen3")


class CageV2AcceptanceGateError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV2AcceptanceGateError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CageV2AcceptanceGateError(f"JSON root must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV2AcceptanceGateError(message)


def _validate_static_gate(
    gate: dict[str, Any], *, protocol_sha256: str, manifest_sha256: str
) -> None:
    _require(gate.get("schema_version") == 1, "acceptance gate schema mismatch")
    _require(gate.get("status") == "pass", "acceptance gate is not passing")
    _require(gate.get("claim_eligible") is False, "acceptance gate must be claim-ineligible")
    _require(
        gate.get("protocol", {}).get("sha256") == protocol_sha256,
        "acceptance gate protocol hash mismatch",
    )
    _require(
        gate.get("development_manifest", {}).get("sha256") == manifest_sha256,
        "acceptance gate manifest hash mismatch",
    )
    _require(tuple(gate.get("partitions", {})) == PARTITIONS, "gate partition order mismatch")
    comparison = gate.get("comparison", {})
    _require(tuple(comparison.get("fields", ())) == COMPARE_FIELDS, "gate comparison fields mismatch")
    _require(
        comparison.get("required") == "bitwise_equal_json_numeric_payload",
        "gate comparison requirement mismatch",
    )
    authorization = gate.get("round1_screen_authorization", {})
    _require(authorization.get("claim_eligible") is False, "screen must be claim-ineligible")
    _require(
        authorization.get("requires_exact_execution_commit") is True,
        "screen must require the accepted execution commit",
    )
    _require(
        authorization.get("execution_commit") == gate["acceptance_source"]["cage_commit"],
        "screen and acceptance commits differ",
    )
    _require(
        authorization.get("anchor_indices") == [5, 6, 7, 8, 9],
        "screen anchor indices differ from the frozen round1 protocol",
    )
    anchor_count = len(authorization["anchor_indices"])
    for partition, count_key in (
        ("cage_qwen3", "cage_qwen3_cases"),
        ("kitty_qwen3", "kitty_qwen3_cases"),
    ):
        expected = gate["partitions"][partition]["case_count_per_repeat"] * anchor_count
        _require(authorization.get(count_key) == expected, f"{partition} screen count mismatch")
    expected_total = sum(
        gate["partitions"][partition]["case_count_per_repeat"]
        * anchor_count
        for partition in PARTITIONS
    )
    _require(authorization.get("total_cases") == expected_total, "screen case total mismatch")


def _validate_repeat(
    repeat: dict[str, Any],
    *,
    partition: str,
    partition_gate: dict[str, Any],
    protocol_sha256: str,
    manifest_sha256: str,
    acceptance_source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(repeat["output_dir"])
    identity_path = root / "run_identity.json"
    summary_path = root / "summary.json"
    _require(file_sha256(identity_path) == repeat["run_identity_sha256"], f"{partition} identity hash mismatch")
    _require(file_sha256(summary_path) == repeat["summary_sha256"], f"{partition} summary hash mismatch")
    _require(
        file_sha256(Path(repeat["execution_log"])) == repeat["execution_log_sha256"],
        f"{partition} execution log hash mismatch",
    )
    identity = load_json(identity_path)
    summary = load_json(summary_path)
    expected_count = partition_gate["case_count_per_repeat"]
    _require(identity.get("claim_eligible") is False, f"{partition} identity became claim-eligible")
    _require(identity.get("partition") == partition, f"{partition} identity partition mismatch")
    _require(identity.get("stage") == "acceptance", f"{partition} identity stage mismatch")
    _require(identity.get("protocol_sha256") == protocol_sha256, f"{partition} protocol mismatch")
    _require(identity.get("manifest_sha256") == manifest_sha256, f"{partition} manifest mismatch")
    source = identity.get("source_state", {})
    _require(source.get("git_commit") == acceptance_source["cage_commit"], f"{partition} CAGE commit mismatch")
    _require(source.get("dirty") is False, f"{partition} CAGE source is dirty")
    if partition == "kitty_qwen3":
        _require(source.get("kitty_commit") == acceptance_source["kitty_commit"], "Kitty commit mismatch")
        _require(source.get("transformers_commit") == acceptance_source["transformers_commit"], "Transformers commit mismatch")
        _require(source.get("kitty_dirty") is False, "Kitty source is dirty")
    for key, expected in (
        ("status", "pass"),
        ("claim_eligible", False),
        ("partition", partition),
        ("stage", "acceptance"),
        ("expected_cases", expected_count),
        ("completed_cases", expected_count),
        ("new_cases", expected_count),
        ("resumed_cases", 0),
        ("failure_records", 0),
        ("case_ids_sha256", partition_gate["case_ids_sha256"]),
    ):
        _require(summary.get(key) == expected, f"{partition} summary {key} mismatch")
    _require(summary.get("identity") == identity, f"{partition} summary identity mismatch")
    model = summary.get("model", {})
    _require(model.get("partition") == partition, f"{partition} model partition mismatch")
    _require(all(model.get("checks", {}).values()), f"{partition} model identity checks failed")
    model_sources = model.get("source_sha256", {})
    for model_key, gate_key in (
        ("runner", "runner_sha256"),
        ("qwen3_cage_v1", "qwen3_cage_v1_sha256"),
        ("qwen3_cage_v2", "qwen3_cage_v2_sha256"),
        ("cage_v2_quant", "cage_v2_quant_sha256"),
        ("cage_v2_memory", "cage_v2_memory_sha256"),
        ("recorder", "recorder_sha256"),
    ):
        _require(
            model_sources.get(model_key) == acceptance_source[gate_key],
            f"{partition} accepted source hash mismatch: {model_key}",
        )
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "cases").glob("*.json")):
        record = load_json(path)
        _require(record.get("status") == "completed", f"incomplete acceptance case: {path}")
        records[record["case_id"]] = record
    _require(len(records) == expected_count, f"{partition} case file count mismatch")
    return summary, records


def _validate_partition_artifacts(
    partition: str,
    partition_gate: dict[str, Any],
    *,
    protocol_sha256: str,
    manifest_sha256: str,
    acceptance_source: dict[str, Any],
) -> None:
    _, first = _validate_repeat(
        partition_gate["repeat_a"],
        partition=partition,
        partition_gate=partition_gate,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
        acceptance_source=acceptance_source,
    )
    _, second = _validate_repeat(
        partition_gate["repeat_b"],
        partition=partition,
        partition_gate=partition_gate,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
        acceptance_source=acceptance_source,
    )
    _require(set(first) == set(second), f"{partition} repeat case IDs differ")
    scientific_payload = []
    for case_id in sorted(first):
        left = {field: first[case_id][field] for field in COMPARE_FIELDS}
        right = {field: second[case_id][field] for field in COMPARE_FIELDS}
        _require(left == right, f"{partition} scientific repeat mismatch: {case_id}")
        scientific_payload.append(left)
    _require(
        canonical_sha256(scientific_payload) == partition_gate["scientific_payload_sha256"],
        f"{partition} scientific payload hash mismatch",
    )
    comparison_spec = partition_gate["comparison"]
    comparison_path = Path(comparison_spec["path"])
    _require(file_sha256(comparison_path) == comparison_spec["sha256"], f"{partition} comparison hash mismatch")
    comparison = load_json(comparison_path)
    _require(comparison.get("status") == "pass", f"{partition} comparison did not pass")
    _require(comparison.get("partition") == partition, f"{partition} comparison partition mismatch")
    _require(comparison.get("case_count") == len(scientific_payload), f"{partition} comparison count mismatch")
    _require(comparison.get("mismatch_case_ids") == [], f"{partition} comparison has mismatches")
    _require(tuple(comparison.get("compare_fields", ())) == COMPARE_FIELDS, f"{partition} compared fields mismatch")
    _require(
        comparison.get("scientific_payload_sha256") == partition_gate["scientific_payload_sha256"],
        f"{partition} comparison payload hash mismatch",
    )
    _require(
        comparison.get("comparator_sha256") == acceptance_source["comparator_sha256"],
        f"{partition} comparator source mismatch",
    )


def load_cage_v2_acceptance_gate(
    gate_path: Path,
    *,
    protocol_path: Path,
    manifest_path: Path,
    verify_artifacts: bool = True,
) -> tuple[dict[str, Any], str]:
    gate = load_json(gate_path)
    gate_sha256 = file_sha256(gate_path)
    protocol_sha256 = file_sha256(protocol_path)
    manifest_sha256 = file_sha256(manifest_path)
    _validate_static_gate(
        gate,
        protocol_sha256=protocol_sha256,
        manifest_sha256=manifest_sha256,
    )
    if verify_artifacts:
        source = gate["acceptance_source"]
        for partition in PARTITIONS:
            _validate_partition_artifacts(
                partition,
                gate["partitions"][partition],
                protocol_sha256=protocol_sha256,
                manifest_sha256=manifest_sha256,
                acceptance_source=source,
            )
    return gate, gate_sha256


__all__ = [
    "CageV2AcceptanceGateError",
    "COMPARE_FIELDS",
    "canonical_sha256",
    "file_sha256",
    "load_cage_v2_acceptance_gate",
]
