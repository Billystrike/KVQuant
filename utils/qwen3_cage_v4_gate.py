from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


GATE_ID = "qwen3-8b-cage-v4-pg19-metric-acceptance-gate-v1"
EXPECTED_KITTY_COMMIT = "dfd2c07b407d6b407179359207c612ab631f3ed1"
EXPECTED_TRANSFORMERS_COMMIT = "37f8b0b53512e6aae0cfd15746c133c101783178"
EXPECTED_SOURCE_HASHES = {
    "runner": "4991266a2c7c41e75823f1423c6de60e7ab02ec7e0a0c059afa8763940450209",
    "comparator": "2cab81f32e7d3b07082887587920ee8d370e5354c8157f7093ad99ae71a46303",
    "acceptance_utils": "f261a1671b413b5ddd674a1b47937add7d6512177f6f8650ac2717737d3bc88f",
    "recorder": "f699776de29f779379c4cfc2f6b65c9644503cfa8231606c717b8ff4d7e6ac82",
}
SOURCE_PATHS = {
    "runner": "scripts/qwen3_run_cage_v4_metric_acceptance.py",
    "comparator": "scripts/qwen3_compare_cage_v4_metric_acceptance.py",
    "acceptance_utils": "utils/qwen3_cage_v4_acceptance.py",
    "recorder": "utils/qwen3_perturbation_runtime.py",
}
EXPECTED_AUTHORIZATION = {
    "authorization_basis": "both frozen acceptance partitions passed two bitwise scientific-payload repeats",
    "input_partition": "screen",
    "screen_documents": 20,
    "screen_anchors": 40,
    "cage_quality_case_count": 600,
    "kitty_quality_case_count": 120,
    "cage_compressed_perturbation_case_count": 480,
    "kitty_compressed_perturbation_case_count": 120,
    "pg19_holdout_method_metrics": False,
    "pg19_test_access": False,
    "cage_v4_candidate_execution": False,
    "claim_eligible": False,
}
EXPECTED_PARTITION_RECEIPTS = {
    "cage_qwen3": {
        "case_count": 15,
        "case_ids_sha256": "d5e743270f61809dd7d9599e490a1e6c2db5b4311fc732f2c422f88a460680bb",
        "scientific_payload_sha256": "f1e636838b775748fbbc1f949133234e83f9964906aab9397a47754fce077f54",
        "comparison_sha256": "787cf83cc910bb52c39a581774af2f04a3bac3db440f47eb3ce123fa96a02bc0",
        "run_identity_sha256": "f4746051bf7d0068f9971789cfa3765907c63bfc1797197ab60cab2cc2cf5ed0",
        "summary_sha256": "b315f4d08d6370c56010dff9267c35f49901e7b07e8c9e360604de29f46687f5",
    },
    "kitty_qwen3": {
        "case_count": 3,
        "case_ids_sha256": "842a654274d139b9f853fbf517483f29182df9b90d96bfa8be15f8bf781f967a",
        "scientific_payload_sha256": "b23ab410c8000dbcd95ffed012913db0bc200fd114d9f4222df27ffa0e73c36f",
        "comparison_sha256": "cf31ba9baa28d593c835a7d857968504f2b7840a87f106ab20f32361cba6fe97",
        "run_identity_sha256": "e5ec6b6e41df1d14178c1cd38fc790f22f2368aa8a17c19c727fa9ea283e4879",
        "summary_sha256": "54ed086fd6e78b1c779571baf435e88fe5f949c6dbd25b778257c8e79ae1ce0f",
    },
}


class CageV4GateError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CageV4GateError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CageV4GateError(f"cannot load JSON artifact {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def shell_case_manifest_sha256(root: Path) -> str:
    lines = []
    for path in sorted((root / "cases").glob("*.json")):
        relative = path.relative_to(root).as_posix()
        lines.append(f"{file_sha256(path)}  {relative}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _validate_repeat(
    artifact: dict[str, Any],
    *,
    partition: str,
    case_count: int,
    accepted_commit: str,
) -> None:
    root = Path(artifact["output_dir"])
    identity_path = root / "run_identity.json"
    summary_path = root / "summary.json"
    log_path = Path(artifact["execution_log"])
    checks = (
        file_sha256(identity_path) == artifact["run_identity_sha256"],
        file_sha256(summary_path) == artifact["summary_sha256"],
        file_sha256(log_path) == artifact["execution_log_sha256"],
        log_path.stat().st_size == artifact["execution_log_size_bytes"],
        shell_case_manifest_sha256(root) == artifact["case_file_manifest_sha256"],
    )
    _require(all(checks), f"acceptance repeat artifact mismatch: {partition}")
    identity = _load(identity_path)
    summary = _load(summary_path)
    source = identity.get("source_state", {})
    _require(source.get("git_commit") == accepted_commit, "acceptance Git commit mismatch")
    _require(source.get("dirty") is False, "acceptance source was dirty")
    if partition == "kitty_qwen3":
        _require(source.get("kitty_commit") == EXPECTED_KITTY_COMMIT, "Kitty commit mismatch")
        _require(
            source.get("transformers_commit") == EXPECTED_TRANSFORMERS_COMMIT,
            "Transformers commit mismatch",
        )
        _require(source.get("kitty_dirty") is False, "Kitty acceptance source was dirty")
    expected_summary = (
        summary.get("status") == "pass",
        summary.get("claim_eligible") is False,
        summary.get("partition") == partition,
        summary.get("stage") == "acceptance",
        summary.get("expected_cases") == case_count,
        summary.get("completed_cases") == case_count,
        summary.get("new_cases") == 0,
        summary.get("resumed_cases") == case_count,
        summary.get("failure_records") == 0,
        summary.get("run_identity") == identity,
    )
    _require(all(expected_summary), f"acceptance repeat summary mismatch: {partition}")


def load_acceptance_gate(
    path: Path,
    *,
    repo_root: Path,
    protocol_sha256: str,
    input_manifest_sha256: str,
    verify_artifacts: bool,
) -> tuple[dict[str, Any], str]:
    gate = _load(path)
    static = (
        gate.get("schema_version") == 1,
        gate.get("gate_id") == GATE_ID,
        gate.get("status") == "pass",
        gate.get("claim_eligible") is False,
        gate.get("protocol_sha256") == protocol_sha256,
        gate.get("input_manifest_sha256") == input_manifest_sha256,
        gate.get("acceptance_source", {}).get("kitty_commit") == EXPECTED_KITTY_COMMIT,
        gate.get("acceptance_source", {}).get("transformers_commit")
        == EXPECTED_TRANSFORMERS_COMMIT,
        gate.get("acceptance_source", {}).get("source_sha256") == EXPECTED_SOURCE_HASHES,
        gate.get("full_screen_authorization") == EXPECTED_AUTHORIZATION,
        set(gate.get("partitions", {})) == {"cage_qwen3", "kitty_qwen3"},
    )
    _require(all(static), "CAGE-v4 acceptance gate static identity mismatch")
    for name, relative in SOURCE_PATHS.items():
        _require(
            file_sha256(repo_root / relative) == EXPECTED_SOURCE_HASHES[name],
            f"accepted source hash changed: {name}",
        )
    accepted_commit = gate["acceptance_source"].get("git_commit")
    _require(isinstance(accepted_commit, str) and len(accepted_commit) == 40, "accepted commit invalid")
    for partition, receipt in EXPECTED_PARTITION_RECEIPTS.items():
        case_count = receipt["case_count"]
        record = gate["partitions"][partition]
        _require(record.get("case_count") == case_count, "acceptance case count changed")
        _require(record.get("case_ids_sha256") == receipt["case_ids_sha256"], "acceptance case-ID receipt changed")
        _require(
            record.get("scientific_payload_sha256") == receipt["scientific_payload_sha256"],
            "acceptance scientific receipt changed",
        )
        _require(record.get("comparison", {}).get("sha256") == receipt["comparison_sha256"], "comparison receipt changed")
        _require(
            record["repeat_a"]["run_identity_sha256"]
            == record["repeat_b"]["run_identity_sha256"]
            == receipt["run_identity_sha256"],
            "repeat run-identity receipt changed",
        )
        _require(
            record["repeat_a"]["summary_sha256"]
            == record["repeat_b"]["summary_sha256"]
            == receipt["summary_sha256"],
            "repeat summary receipt changed",
        )
        if verify_artifacts:
            for label in ("repeat_a", "repeat_b"):
                _validate_repeat(
                    record[label],
                    partition=partition,
                    case_count=case_count,
                    accepted_commit=accepted_commit,
                )
            comparison = record["comparison"]
            comparison_path = Path(comparison["path"])
            comparison_log = Path(comparison["log_path"])
            _require(file_sha256(comparison_path) == comparison["sha256"], "comparison hash mismatch")
            _require(file_sha256(comparison_log) == comparison["log_sha256"], "comparison log mismatch")
            _require(comparison_log.stat().st_size == comparison["log_size_bytes"], "comparison log size mismatch")
            value = _load(comparison_path)
            comparison_checks = (
                value.get("status") == "pass",
                value.get("partition") == partition,
                value.get("case_count") == case_count,
                value.get("mismatch_case_ids") == [],
                value.get("scientific_payload_sha256") == record["scientific_payload_sha256"],
                value.get("comparator_sha256") == EXPECTED_SOURCE_HASHES["comparator"],
            )
            _require(all(comparison_checks), f"comparison payload mismatch: {partition}")
            identity = _load(Path(record["repeat_a"]["output_dir"]) / "run_identity.json")
            _require(
                canonical_sha256(identity["expected_case_ids"]) == record["case_ids_sha256"],
                f"acceptance case-ID hash mismatch: {partition}",
            )
    failures = gate.get("failed_pre_case_attempts")
    _require(isinstance(failures, list) and len(failures) == 1, "failed-attempt audit changed")
    _require(failures[0].get("cases_executed") == 0, "failed attempt consumed a case")
    if verify_artifacts:
        failed_log = Path(failures[0]["execution_log"])
        _require(file_sha256(failed_log) == failures[0]["execution_log_sha256"], "failed log mismatch")
        _require(failed_log.stat().st_size == failures[0]["execution_log_size_bytes"], "failed log size mismatch")
        failed_root = Path(failures[0]["output_dir"])
        completed = list((failed_root / "cases").glob("*.json")) if (failed_root / "cases").exists() else []
        _require(not completed, "failed pre-case attempt contains completed cases")
    return gate, file_sha256(path)


__all__ = [
    "CageV4GateError",
    "EXPECTED_AUTHORIZATION",
    "GATE_ID",
    "load_acceptance_gate",
    "shell_case_manifest_sha256",
]
