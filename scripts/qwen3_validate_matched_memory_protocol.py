#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_memory import (
    closest_memory_candidate,
    estimate_qwen3_cage_bytes,
    estimate_qwen3_kivi_bytes,
    estimate_qwen3_kitty_bytes,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recalculate every frozen Qwen3 matched-memory selection"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "configs" / "qwen3_8b_matched_memory_protocol_v1.json",
    )
    parser.add_argument("--kitty-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _same_float(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)


def main() -> None:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    kitty_audit_path = args.kitty_audit.resolve()
    output_path = args.output.resolve()
    protocol = _read_json(protocol_path)
    kitty_audit = _read_json(kitty_audit_path)
    matrix = protocol["matched_memory_matrix"]
    model = protocol["model"]
    memory = protocol["memory_accounting"]
    tolerance = float(matrix["relative_budget_tolerance"])
    audit_sha256 = _sha256(kitty_audit_path)

    checks: dict[str, bool] = {
        "schema_version": protocol.get("schema_version") == 1,
        "protocol_status": protocol.get("status")
        == "frozen_before_qwen3_cage_quality_results",
        "model_dimensions": (
            model.get("num_hidden_layers"),
            model.get("num_key_value_heads"),
            model.get("head_dim"),
        )
        == (36, 8, 128),
        "kitty_audit_is_json_object": isinstance(kitty_audit, dict),
        "kitty_audit_sha256": audit_sha256 == memory["kitty_memory_audit_sha256"],
        "cage_index_dtype": memory.get("cage_bucket_index_dtype") == "int64",
    }
    probe = estimate_qwen3_cage_bytes(seq_len=1024, residual_length=224)
    calculated_index_bytes = (
        probe["per_layer"]["bucket_index_bytes"] * model["num_hidden_layers"]
    )
    checks["cage_bucket_index_bytes"] = (
        calculated_index_bytes == memory["cage_bucket_index_bytes_model_wide"]
    )

    recalculations: list[dict[str, Any]] = []
    boosted_by_target = {
        "kitty-12.5pct": 16,
        "kitty-pro-25pct": 32,
    }
    for frozen in matrix["selections"]:
        seq_len = int(frozen["prompt_length"])
        target_method = frozen["target_method"]
        if target_method not in boosted_by_target:
            raise ValueError(f"unsupported target method: {target_method}")
        boosted_channels = boosted_by_target[target_method]
        target_report = estimate_qwen3_kitty_bytes(
            seq_len=seq_len,
            boosted_channels=boosted_channels,
        )
        target_bytes = int(target_report["model_total_bytes"])
        cage_candidates = [
            estimate_qwen3_cage_bytes(seq_len=seq_len, residual_length=residual)
            for residual in range(32, 513, 32)
        ]
        kivi_candidates = [
            estimate_qwen3_kivi_bytes(
                seq_len=seq_len,
                group_size=group,
                residual_length=residual,
            )
            for group in (32, 64, 128)
            for residual in range(group, 513, group)
        ]
        cage, cage_delta = closest_memory_candidate(cage_candidates, target_bytes)
        kivi, kivi_delta = closest_memory_candidate(kivi_candidates, target_bytes)
        row_id = f"{seq_len}:{target_method}"
        checks[f"{row_id}:target"] = target_bytes == frozen["target_bytes"]
        checks[f"{row_id}:cage_method"] = cage["method"] == frozen["cage_method"]
        checks[f"{row_id}:cage_bytes"] = (
            cage["model_total_bytes"] == frozen["cage_bytes"]
        )
        checks[f"{row_id}:cage_delta"] = _same_float(
            cage_delta, frozen["cage_relative_delta"]
        )
        if frozen["kivi_method"] is None:
            checks[f"{row_id}:kivi_excluded"] = abs(kivi_delta) > tolerance
            checks[f"{row_id}:kivi_null_fields"] = (
                frozen["kivi_bytes"] is None and frozen["kivi_relative_delta"] is None
            )
        else:
            checks[f"{row_id}:kivi_method"] = kivi["method"] == frozen["kivi_method"]
            checks[f"{row_id}:kivi_bytes"] = (
                kivi["model_total_bytes"] == frozen["kivi_bytes"]
            )
            checks[f"{row_id}:kivi_delta"] = _same_float(
                kivi_delta, frozen["kivi_relative_delta"]
            )
            checks[f"{row_id}:kivi_admitted"] = abs(kivi_delta) <= tolerance
        recalculations.append(
            {
                "prompt_length": seq_len,
                "target_method": target_method,
                "target": target_report,
                "closest_cage": cage,
                "cage_relative_delta": cage_delta,
                "closest_kivi": kivi,
                "kivi_relative_delta": kivi_delta,
                "kivi_admitted": abs(kivi_delta) <= tolerance,
            }
        )

    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "protocol_id": protocol["protocol_id"],
        "kitty_audit_path": str(kitty_audit_path),
        "kitty_audit_sha256": audit_sha256,
        "kitty_audit_top_level_keys": sorted(kitty_audit),
        "qwen3_memory_source_sha256": _sha256(
            REPO_ROOT / "utils" / "qwen3_memory.py"
        ),
        "cage_bucket_index_bytes_model_wide": calculated_index_bytes,
        "relative_budget_tolerance": tolerance,
        "checks": checks,
        "recalculations": recalculations,
    }
    _write_json_atomic(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"validation_json: {output_path}")
    print(f"validation_json_sha256: {_sha256(output_path)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
