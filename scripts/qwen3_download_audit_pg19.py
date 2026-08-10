#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v4_pg19 import (
    EXPECTED_AUDIT_ENVIRONMENT,
    EXPECTED_MODEL,
    EXPECTED_SOURCE,
    audit_documents,
    file_sha256,
    load_source_candidate,
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return {"git_commit": commit, "dirty": bool(status)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and audit the exact PG-19 validation parquet for Qwen3 CAGE-v4 Stage 0"
    )
    parser.add_argument("--source-candidate", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_candidate.resolve()
    model_path = args.model.resolve()
    download_dir = args.download_dir.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output_path}")

    source_candidate, source_candidate_sha256 = load_source_candidate(source_path)
    source_state = _git_state()

    import huggingface_hub
    import pyarrow
    import pyarrow.parquet as parquet
    import tokenizers
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    downloaded = Path(
        hf_hub_download(
            repo_id=EXPECTED_SOURCE["repo_id"],
            repo_type=EXPECTED_SOURCE["repo_type"],
            revision=EXPECTED_SOURCE["revision"],
            filename=EXPECTED_SOURCE["filename"],
            local_dir=download_dir,
        )
    ).resolve()
    file_identity = {
        "path": str(downloaded),
        "size_bytes": downloaded.stat().st_size,
        "sha256": file_sha256(downloaded),
    }

    table = parquet.read_table(downloaded)
    columns = table.column_names
    text_column = EXPECTED_SOURCE["text_column"]
    if text_column not in columns:
        raise ValueError(f"PG-19 parquet lacks required text column: {columns}")
    rows = table.to_pylist()

    metadata_hashes = {
        "config_sha256": file_sha256(model_path / "config.json"),
        "tokenizer_config_sha256": file_sha256(model_path / "tokenizer_config.json"),
        "tokenizer_json_sha256": file_sha256(model_path / "tokenizer.json"),
        "vocab_sha256": file_sha256(model_path / "vocab.json"),
        "merges_sha256": file_sha256(model_path / "merges.txt"),
    }
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    documents, document_summary = audit_documents(rows, tokenizer=tokenizer, text_column=text_column)

    checks = {
        "source_clean": source_state["dirty"] is False,
        "download_size": file_identity["size_bytes"] == EXPECTED_SOURCE["size_bytes"],
        "download_sha256": file_identity["sha256"] == EXPECTED_SOURCE["sha256"],
        "row_count": table.num_rows == EXPECTED_SOURCE["row_count"],
        "text_column_present": text_column in columns,
        "model_metadata": all(metadata_hashes[key] == EXPECTED_MODEL[key] for key in metadata_hashes),
        "model_type": getattr(config, "model_type", None) == "qwen3",
        "python_version": platform.python_version() == EXPECTED_AUDIT_ENVIRONMENT["python"],
        "huggingface_hub_version": huggingface_hub.__version__
        == EXPECTED_AUDIT_ENVIRONMENT["huggingface_hub"],
        "pyarrow_version": pyarrow.__version__ == EXPECTED_AUDIT_ENVIRONMENT["pyarrow"],
        "transformers_version": transformers.__version__
        == EXPECTED_AUDIT_ENVIRONMENT["transformers"],
        "all_texts_unique": document_summary["duplicate_text_sha256_groups"] == [],
        "all_token_streams_unique": document_summary["duplicate_token_ids_sha256_groups"] == [],
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "claim_eligible": False,
        "purpose": "pg19_validation_identity_schema_and_qwen3_token_feasibility_audit_before_any_split",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
        "checks": checks,
        "source_state": source_state,
        "environment": {
            "python": platform.python_version(),
            "huggingface_hub": huggingface_hub.__version__,
            "pyarrow": pyarrow.__version__,
            "tokenizers": tokenizers.__version__,
            "transformers": transformers.__version__,
        },
        "source_candidate": {
            "path": str(source_path),
            "sha256": source_candidate_sha256,
            "candidate_id": source_candidate["candidate_id"],
        },
        "download": {
            **file_identity,
            "repo_id": EXPECTED_SOURCE["repo_id"],
            "revision": EXPECTED_SOURCE["revision"],
            "filename": EXPECTED_SOURCE["filename"],
            "split": EXPECTED_SOURCE["split"],
        },
        "parquet": {
            "row_count": table.num_rows,
            "column_count": table.num_columns,
            "columns": columns,
            "schema": str(table.schema),
        },
        "model": {
            "path": str(model_path),
            "revision": EXPECTED_MODEL["revision"],
            "model_type": getattr(config, "model_type", None),
            "metadata_hashes": metadata_hashes,
        },
        "tokenizer": {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "add_special_tokens": False,
        },
        "document_summary": document_summary,
        "documents": documents,
        "freeze_boundary": {
            "documents_selected": False,
            "development_partitions_frozen": False,
            "metric_protocol_frozen": False,
            "cage_v4_method_frozen": False,
            "gpu_execution_authorized": False,
            "paper_claims_authorized": False,
        },
        "source_hashes": {
            "audit_script_sha256": file_sha256(Path(__file__).resolve()),
            "audit_utils_sha256": file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v4_pg19.py"),
        },
    }
    _write_json_atomic(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"audit_json: {output_path}")
    print(f"audit_json_sha256: {file_sha256(output_path)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
