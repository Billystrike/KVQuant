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

from utils.qwen3_cage_v4_data import build_input_manifest, file_sha256, load_data_protocol


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    )
    return {"git_commit": commit, "dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen document-level PG-19 inputs before CAGE-v4")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    audit_path = args.token_audit.resolve()
    parquet_path = args.parquet.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite PG-19 input manifest: {output_path}")

    protocol, protocol_sha256 = load_data_protocol(protocol_path)
    if file_sha256(audit_path) != protocol["source_receipt"]["token_audit_sha256"]:
        raise ValueError("PG-19 audit file identity differs from protocol")
    if file_sha256(parquet_path) != protocol["source_receipt"]["parquet_sha256"]:
        raise ValueError("PG-19 parquet identity differs from protocol")
    source_state = _git_state()
    if source_state["dirty"]:
        raise RuntimeError("PG-19 manifest build requires a clean KVQuant tree")

    import pyarrow.parquet as parquet
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    table = parquet.read_table(parquet_path)
    rows = table.to_pylist()
    if len(rows) != 50:
        raise ValueError("PG-19 validation parquet must contain 50 documents")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    tokenizer_identity = {
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "add_special_tokens": False,
    }
    audit_by_row = {row["row_index"]: row for row in audit["documents"]}
    token_ids_by_document = {}
    for row_index, row in enumerate(rows):
        audited = audit_by_row.get(row_index)
        if audited is None:
            raise ValueError(f"PG-19 audit lacks row {row_index}")
        encoded = tokenizer(
            row["text"],
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
        token_ids_by_document[audited["document_id"]] = (
            encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        )
    manifest = build_input_manifest(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        audit=audit,
        audit_sha256=file_sha256(audit_path),
        token_ids_by_document=token_ids_by_document,
        tokenizer_identity=tokenizer_identity,
        source_state=source_state,
    )
    _write_atomic(output_path, manifest)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "claim_eligible": False,
                "protocol_sha256": protocol_sha256,
                "token_audit_sha256": file_sha256(audit_path),
                "parquet_sha256": file_sha256(parquet_path),
                "document_count": len(manifest["documents"]),
                "anchor_count": len(manifest["anchors"]),
                "case_count": len(manifest["cases"]),
                "partitions": {
                    name: len(document_ids) for name, document_ids in manifest["partitions"].items()
                },
                "output": str(output_path),
                "output_sha256": file_sha256(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
