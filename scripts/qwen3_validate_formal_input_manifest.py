#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_formal import (
    Qwen3FormalError,
    file_sha256,
    load_formal_protocol,
    validate_input_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the neutral Qwen3 formal input manifest")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    protocol, protocol_sha256 = load_formal_protocol(args.protocol.resolve())
    manifest_path = args.input_manifest.resolve()
    manifest_sha256 = file_sha256(manifest_path)
    if args.expected_sha256 is not None and manifest_sha256 != args.expected_sha256:
        raise Qwen3FormalError("input manifest SHA-256 differs from --expected-sha256")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Qwen3FormalError(f"cannot load input manifest {manifest_path}: {error}") from error
    validate_input_manifest(
        manifest,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha256,
                "input_manifest": str(manifest_path),
                "input_manifest_sha256": manifest_sha256,
                "input_case_count": len(manifest["cases"]),
                "source_state": manifest["source_state"],
                "first_input_case_id": manifest["cases"][0]["input_case_id"],
                "last_input_case_id": manifest["cases"][-1]["input_case_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
