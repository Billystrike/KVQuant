#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_formal import file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot the offline CAGE-v2 development corpus")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "development_protocol_not_frozen_for_final_claims":
        raise RuntimeError("corpus snapshot requires the development-only CAGE-v2 protocol")
    data = protocol["development_data"]

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if status:
        raise RuntimeError("corpus snapshot requires a clean KVQuant source tree")

    import datasets

    dataset = datasets.load_dataset(
        data["corpus_id"],
        data["corpus_config"],
        split=data["corpus_split"],
    )
    texts = list(dataset["text"])
    if not texts or any(not isinstance(text, str) for text in texts):
        raise RuntimeError("development corpus text column is invalid")
    snapshot = {
        "schema_version": 1,
        "corpus": {
            "id": data["corpus_id"],
            "config": data["corpus_config"],
            "split": data["corpus_split"],
            "revision": data["corpus_revision"],
            "join_separator": data["join_separator"],
            "dataset_fingerprint": dataset._fingerprint,
            "datasets_version": datasets.__version__,
            "rows": len(texts),
            "nonempty_rows": sum(bool(text.strip()) for text in texts),
        },
        "texts": texts,
    }
    _write_atomic(output_path, snapshot)
    print(
        json.dumps(
            {
                "status": "pass",
                "claim_eligibility": "development_only",
                "protocol_sha256": file_sha256(protocol_path),
                "datasets": datasets.__version__,
                "dataset_fingerprint": dataset._fingerprint,
                "rows": len(texts),
                "output": str(output_path),
                "output_sha256": file_sha256(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
