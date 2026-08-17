#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from utils.llama2_cage_v3_gpu_execution import load_object, require


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two Llama-2 CAGE-v3 GPU acceptance repeats")
    parser.add_argument("--repeat-a", type=Path, required=True)
    parser.add_argument("--repeat-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left = load_object(args.repeat_a.resolve())
    right = load_object(args.repeat_b.resolve())
    require(left.get("repeat") == "a" and right.get("repeat") == "b", "repeat labels mismatch")
    require(left.get("status") == "pass" and right.get("status") == "pass", "a GPU repeat did not pass")
    require(left.get("protocol_sha256") == right.get("protocol_sha256"), "repeat protocol mismatch")
    require(left.get("gate_receipt_sha256") == right.get("gate_receipt_sha256"), "repeat gate mismatch")
    equal = left.get("scientific_payload") == right.get("scientific_payload")
    report = {
        "schema_version": 1,
        "comparison_id": "llama2-7b-cage-v3-production-gpu-acceptance-repeat-comparison-v1",
        "status": "pass" if equal else "fail",
        "claim_eligible": False,
        "required_consistency": "bitwise_equal_json_scientific_payload",
        "scientific_payload_equal": equal,
        "repeat_a": str(args.repeat_a.resolve()),
        "repeat_b": str(args.repeat_b.resolve()),
        "telemetry_excluded": True,
        "formal_transfer_authorized": False,
    }
    _write_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not equal:
        raise RuntimeError("GPU acceptance repeat scientific payloads differ")


if __name__ == "__main__":
    main()
