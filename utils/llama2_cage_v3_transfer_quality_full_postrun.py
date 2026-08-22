from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from utils.llama2_cage_v3_transfer_quality_acceptance import (
    SCIENTIFIC_FIELDS,
    lf_normalized_file_sha256,
    load_server_input_manifest,
)
from utils.llama2_cage_v3_transfer_quality_full import (
    FULL_RUN_ID,
    expand_full_cases,
    load_design,
    load_full_gate_receipt,
    validate_full_case,
)
from utils.llama2_cage_v3_transfer_quality_protocol import load_transfer_quality_protocol
from utils.qwen3_cage_v4_data import canonical_sha256, file_sha256


ARTIFACT_MANIFEST_ID = "llama2-7b-cage-v3-transfer-quality-full-artifacts-v1"
AUDIT_ID = "llama2-7b-cage-v3-transfer-quality-full-postrun-v1"
ARTIFACT_MANIFEST_SHA256 = "4ea3930a9466ad32ea6b868db403f185dec8bdd4dfa7039b30333ec14a37eddf"
EXPECTED_SCIENTIFIC_SHA256 = "e5058f62b3a70f539e540576a9b5eff37a360ad96fb0a0d63685dd5c2f98003b"


class Llama2CageV3TransferQualityFullPostrunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Llama2CageV3TransferQualityFullPostrunError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Llama2CageV3TransferQualityFullPostrunError(f"cannot load JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _verify_file(path: Path, *, sha256: str, size_bytes: int | None = None, label: str) -> None:
    _require(path.is_file(), f"{label} is missing")
    _require(file_sha256(path) == sha256, f"{label} hash mismatch")
    if size_bytes is not None:
        _require(path.stat().st_size == size_bytes, f"{label} size mismatch")


def shell_case_manifest_sha256(root: Path) -> str:
    lines = [
        f"{file_sha256(path)}  {path}\n"
        for path in sorted((root / "cases").glob("*.json"), key=lambda item: item.name)
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def validate_artifact_manifest(manifest: Mapping[str, Any], *, repo_root: Path) -> None:
    _require(manifest.get("schema_version") == 1, "artifact manifest schema changed")
    _require(manifest.get("artifact_manifest_id") == ARTIFACT_MANIFEST_ID, "artifact manifest identity changed")
    _require(
        manifest.get("status") == "frozen_after_complete_600_case_execution_before_read_only_postrun_audit",
        "artifact manifest status changed",
    )
    _require(manifest.get("claim_eligible") is False, "artifact claim boundary changed")
    design = manifest.get("design", {})
    _require(
        design == {
            "path": "configs/llama2_7b_cage_v3_transfer_quality_full_design_v1.json",
            "sha256": "bda724619ad986703ca359c6a7fa66a44d0d762151689895ce18cd6574483e7a",
        },
        "artifact design changed",
    )
    _require(lf_normalized_file_sha256(repo_root / design["path"]) == design["sha256"], "checked-in design changed")
    gate = manifest.get("gate_receipt", {})
    _require(
        gate == {
            "path": "configs/llama2_7b_cage_v3_transfer_quality_full_execution_gate_receipt_v1.json",
            "sha256": "47afaf6971fd84e99366380a774bb85d5b59db569b212ffebfe2ddc422a9547f",
        },
        "artifact gate receipt changed",
    )
    _require(file_sha256(repo_root / gate["path"]) == gate["sha256"], "checked-in gate receipt changed")
    run = manifest.get("full_run", {})
    _require(
        run.get("execution_source_commit") == "f22385c7288b8cbaa0fb04b018f62473a2136995"
        and run.get("run_identity_sha256") == "1cf47f577d5d3d7110f561f3ac658cb8e7304e5fcce544c7e64c8114f6bb51dd"
        and run.get("summary_sha256") == "02a5c4ff634063676a47d429bc6620923f46489e3461675402817ea15443a567"
        and run.get("case_file_hash_manifest_sha256") == "0e3191a3415f37911b0b1d6219cc3853f38470ca238ab62d00a15586be06859e"
        and run.get("scientific_payload_sha256") == EXPECTED_SCIENTIFIC_SHA256
        and (run.get("case_count"), run.get("target_count")) == (600, 38400)
        and (run.get("new_cases"), run.get("resumed_cases"), run.get("failure_count")) == (600, 0, 0),
        "full-run artifact identity changed",
    )
    _require(
        manifest.get("expected_counts")
        == {
            "method_counts": {"cage_v1": 150, "cage_v3": 150, "fp16": 150, "kivi": 150},
            "length_counts": {"1024": 200, "2048": 200, "4032": 200},
            "target_count_per_case": 64,
            "total_target_count": 38400,
        },
        "expected full-run counts changed",
    )
    _require(
        manifest.get("authorization_before_postrun")
        == {
            "read_only_postrun_audit": True,
            "frozen_analysis_implementation": False,
            "quality_interpretation": False,
            "candidate_tuning": False,
            "paper_claims": False,
            "runtime_claims": False,
        },
        "pre-postrun authorization changed",
    )


def _validate_log(spec: Mapping[str, Any]) -> dict[str, bool]:
    path = Path(spec["execution_log"])
    _verify_file(
        path,
        sha256=spec["execution_log_sha256"],
        size_bytes=spec["execution_log_size_bytes"],
        label="full execution log",
    )
    text = path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "nonempty": bool(text.strip()),
        "no_traceback": "Traceback (most recent call last)" not in text,
        "no_explicit_error": "ERROR:" not in text,
        "start_marker": spec["start_marker"] in text,
        "end_marker": spec["end_marker"] in text,
        "execution_pass": "EXEC_STATUS=0" in text,
        "completion_audit_pass": "AUDIT_STATUS=0" in text,
    }
    _require(all(checks.values()), "full execution log checks failed")
    return checks


def _validate_full_run(
    spec: Mapping[str, Any],
    *,
    expected_cases: list[dict[str, Any]],
    design_sha256: str,
    gate_receipt_sha256: str,
    protocol_sha256: str,
    input_manifest_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(spec["output_dir"])
    identity_path = root / "run_identity.json"
    summary_path = root / "summary.json"
    _verify_file(identity_path, sha256=spec["run_identity_sha256"], label="full run identity")
    _verify_file(summary_path, sha256=spec["summary_sha256"], label="full summary")
    identity = _load(identity_path)
    summary = _load(summary_path)
    expected_source_state = {"git_commit": spec["execution_source_commit"], "dirty": False}
    _require(
        identity.get("full_run_id") == FULL_RUN_ID
        and identity.get("design_sha256") == design_sha256
        and identity.get("gate_receipt_sha256") == gate_receipt_sha256
        and identity.get("protocol_sha256") == protocol_sha256
        and identity.get("input_manifest_sha256") == input_manifest_sha256
        and identity.get("source_state") == expected_source_state
        and identity.get("expected_case_ids") == [case["case_id"] for case in expected_cases],
        "full run identity changed",
    )
    expected_counts = {
        "method_counts": {"cage_v1": 150, "cage_v3": 150, "fp16": 150, "kivi": 150},
        "length_counts": {"1024": 200, "2048": 200, "4032": 200},
    }
    _require(
        summary.get("schema_version") == 1
        and summary.get("status") == "pass"
        and summary.get("claim_eligible") is False
        and summary.get("expected_cases") == 600
        and summary.get("completed_cases") == 600
        and summary.get("failure_records") == 0
        and summary.get("new_cases") == 600
        and summary.get("resumed_cases") == 0
        and summary.get("scientific_payload_sha256") == EXPECTED_SCIENTIFIC_SHA256
        and summary.get("method_counts") == expected_counts["method_counts"]
        and summary.get("length_counts") == expected_counts["length_counts"]
        and summary.get("target_count_per_case") == 64
        and summary.get("total_target_count") == 38400,
        "full summary changed",
    )
    boundary = summary.get("execution_boundary", {})
    _require(boundary.get("full_execution_complete") is True, "full completion boundary changed")
    for key in ("quality_interpretation_authorized", "candidate_tuning_authorized", "paper_claims_authorized", "runtime_claims_authorized"):
        _require(boundary.get(key) is False, f"full summary expanded authorization: {key}")
    failures = sorted((root / "failures").glob("*.json")) if (root / "failures").exists() else []
    _require(not failures, "full output contains failure files")
    records = []
    method_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    for case in expected_cases:
        record = _load(root / "cases" / f"{case['case_id']}.json")
        validate_full_case(record, case)
        provenance = record.get("provenance", {})
        _require(
            provenance.get("source_state") == expected_source_state
            and provenance.get("gate_receipt_sha256") == gate_receipt_sha256
            and provenance.get("protocol_sha256") == protocol_sha256
            and provenance.get("input_manifest_sha256") == input_manifest_sha256
            and provenance.get("python") == "3.10.20"
            and provenance.get("torch") == "2.4.1+cu121"
            and provenance.get("transformers") == "4.43.1"
            and provenance.get("gpu") == "NVIDIA GeForce RTX 4090 D",
            f"full case provenance changed: {case['case_id']}",
        )
        method_counts[record["method"]["method"]] += 1
        length_counts[str(record["input"]["identity"]["prompt_length"])] += 1
        records.append(record)
    _require(dict(sorted(method_counts.items())) == expected_counts["method_counts"], "case method counts changed")
    _require(dict(sorted(length_counts.items())) == expected_counts["length_counts"], "case length counts changed")
    payload = [{field: record[field] for field in SCIENTIFIC_FIELDS} for record in records]
    payload_sha = canonical_sha256(payload)
    _require(payload_sha == EXPECTED_SCIENTIFIC_SHA256, "full scientific payload changed")
    _require(shell_case_manifest_sha256(root) == spec["case_file_hash_manifest_sha256"], "full case manifest changed")
    return {
        "output_dir": str(root),
        "case_count": len(records),
        "failure_count": 0,
        "target_count": len(records) * 64,
        "method_counts": dict(sorted(method_counts.items())),
        "length_counts": dict(sorted(length_counts.items())),
        "run_identity_sha256": spec["run_identity_sha256"],
        "summary_sha256": spec["summary_sha256"],
        "case_file_hash_manifest_sha256": spec["case_file_hash_manifest_sha256"],
        "scientific_payload_sha256": payload_sha,
        "execution_source_commit": spec["execution_source_commit"],
        "execution_log": spec["execution_log"],
        "execution_log_sha256": spec["execution_log_sha256"],
        "execution_log_checks": _validate_log(spec),
    }, payload


def build_postrun_audit(manifest_path: str | Path, *, repo_root: Path) -> dict[str, Any]:
    source = Path(manifest_path).resolve()
    manifest = _load(source)
    _require(lf_normalized_file_sha256(source) == ARTIFACT_MANIFEST_SHA256, "artifact manifest file hash changed")
    validate_artifact_manifest(manifest, repo_root=repo_root)
    design, design_sha = load_design(repo_root / manifest["design"]["path"], repo_root=repo_root)
    gate, gate_sha = load_full_gate_receipt(repo_root / manifest["gate_receipt"]["path"], repo_root=repo_root)
    gate_output = manifest["gate_output"]
    _verify_file(
        Path(gate_output["path"]),
        sha256=gate_output["sha256"],
        size_bytes=gate_output["size_bytes"],
        label="full gate output",
    )
    protocol, protocol_sha = load_transfer_quality_protocol(repo_root / design["protocol"]["path"], repo_root=repo_root)
    input_manifest = load_server_input_manifest({"input_manifest": design["input_manifest"]}, protocol=protocol)
    expected_cases = expand_full_cases(
        design=design,
        protocol=protocol,
        input_manifest=input_manifest,
        gate_receipt_sha256=gate_sha,
    )
    full_run, _ = _validate_full_run(
        manifest["full_run"],
        expected_cases=expected_cases,
        design_sha256=design_sha,
        gate_receipt_sha256=gate_sha,
        protocol_sha256=protocol_sha,
        input_manifest_sha256=design["input_manifest"]["sha256"],
    )
    return {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "status": "pass",
        "claim_eligible": False,
        "interpretation_performed": False,
        "artifact_manifest_path": str(source),
        "artifact_manifest_sha256": ARTIFACT_MANIFEST_SHA256,
        "design_sha256": design_sha,
        "gate_receipt_sha256": gate_sha,
        "gate_output_sha256": gate_output["sha256"],
        "case_count": 600,
        "failure_count": 0,
        "target_count": 38400,
        "scientific_payload_sha256": EXPECTED_SCIENTIFIC_SHA256,
        "full_run": full_run,
        "environment_package_check": manifest["environment_package_check"],
        "next_step": {
            "frozen_analysis_protocol_implementation_authorized": True,
            "quality_interpretation_authorized_by_this_audit": False,
            "candidate_tuning_authorized": False,
            "paper_claims_authorized": False,
            "runtime_claims_authorized": False,
        },
    }


__all__ = [
    "ARTIFACT_MANIFEST_ID",
    "ARTIFACT_MANIFEST_SHA256",
    "AUDIT_ID",
    "EXPECTED_SCIENTIFIC_SHA256",
    "Llama2CageV3TransferQualityFullPostrunError",
    "build_postrun_audit",
    "shell_case_manifest_sha256",
    "validate_artifact_manifest",
]
