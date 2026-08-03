#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED = {
    "id": "Salesforce/wikitext",
    "config": "wikitext-2-raw-v1",
    "split": "test",
    "revision": "00aa25585682d4957f9e86edc73f59be7419af99",
    "join_separator": "\n\n",
    "rows": 4358,
    "nonempty_rows": 2891,
    "joined_utf8_bytes": 1296370,
    "joined_text_sha256": "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the frozen WikiText raw test rows without tokenization"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = _parse_args()
    output_path = args.output.resolve()
    try:
        import datasets
        from datasets import load_dataset
    except Exception as error:
        raise RuntimeError(f"cannot import dataset export dependencies: {error}") from error

    dataset = load_dataset(
        EXPECTED["id"],
        EXPECTED["config"],
        split=EXPECTED["split"],
        revision=EXPECTED["revision"],
    )
    texts = list(dataset["text"])
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("WikiText text column must contain only strings")
    joined = EXPECTED["join_separator"].join(texts)
    joined_bytes = joined.encode("utf-8")
    actual = {
        "rows": len(texts),
        "nonempty_rows": sum(bool(text.strip()) for text in texts),
        "joined_utf8_bytes": len(joined_bytes),
        "joined_text_sha256": hashlib.sha256(joined_bytes).hexdigest(),
    }
    expected_identity = {
        name: EXPECTED[name]
        for name in (
            "rows",
            "nonempty_rows",
            "joined_utf8_bytes",
            "joined_text_sha256",
        )
    }
    if actual != expected_identity:
        raise ValueError(f"WikiText raw identity mismatch: {actual!r}")
    snapshot = {
        "schema_version": 1,
        "corpus": {
            "id": EXPECTED["id"],
            "config": EXPECTED["config"],
            "split": EXPECTED["split"],
            "revision": EXPECTED["revision"],
            "join_separator": EXPECTED["join_separator"],
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "datasets_version": datasets.__version__,
            **actual,
        },
        "texts": texts,
    }
    _write_json_atomic(output_path, snapshot)
    print(json.dumps(snapshot["corpus"], indent=2, sort_keys=True))
    print(f"snapshot_json: {output_path}")
    print(f"snapshot_json_sha256: {_sha256(output_path)}")


if __name__ == "__main__":
    main()
