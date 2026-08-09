#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = {
    "id": "Salesforce/wikitext",
    "config": "wikitext-2-raw-v1",
    "split": "train",
    "revision": "00aa25585682d4957f9e86edc73f59be7419af99",
    "join_separator": "\n\n",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot the offline WikiText-2 train split for CAGE-v3 identity audit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if status:
        raise RuntimeError("CAGE-v3 corpus snapshot requires a clean KVQuant source tree")

    import datasets

    dataset = datasets.load_dataset(
        CORPUS["id"],
        CORPUS["config"],
        split=CORPUS["split"],
        revision=CORPUS["revision"],
    )
    texts = list(dataset["text"])
    if not texts or any(not isinstance(text, str) for text in texts):
        raise RuntimeError("WikiText-2 train text column is invalid")
    joined = CORPUS["join_separator"].join(texts)
    joined_bytes = joined.encode("utf-8")
    raw_identity = {
        "rows": len(texts),
        "nonempty_rows": sum(bool(text.strip()) for text in texts),
        "joined_utf8_bytes": len(joined_bytes),
        "joined_text_sha256": hashlib.sha256(joined_bytes).hexdigest(),
    }
    snapshot = {
        "schema_version": 1,
        "purpose": "cage_v3_train_identity_audit_before_protocol_freeze",
        "claim_eligible": False,
        "corpus": {
            **CORPUS,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "datasets_version": datasets.__version__,
            **raw_identity,
        },
        "texts": texts,
    }
    _write_json_atomic(output_path, snapshot)
    print(json.dumps({
        "status": "pass",
        "claim_eligible": False,
        "corpus": snapshot["corpus"],
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
