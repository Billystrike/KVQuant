#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_formal import (
    Qwen3FormalError,
    build_input_manifest,
    file_sha256,
    load_formal_protocol,
    read_corpus_snapshot,
)
from utils.qwen3_cases import token_ids_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen neutral Qwen3 formal input manifest")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _source_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    return {"git_commit": commit, "dirty": bool(status)}


def _write_json_atomic(path: Path, payload: dict) -> None:
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
    model_path = args.model.resolve()
    snapshot_path = args.corpus_snapshot.resolve()
    output_path = args.output.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    protocol, protocol_sha256 = load_formal_protocol(protocol_path)
    if str(model_path) != protocol["model"]["reference"]:
        raise Qwen3FormalError("model path differs from the frozen formal protocol")
    snapshot_corpus, texts = read_corpus_snapshot(snapshot_path)
    snapshot_sha256 = file_sha256(snapshot_path)

    try:
        import transformers
        from transformers import AutoConfig, AutoTokenizer
    except Exception as error:
        raise Qwen3FormalError(f"cannot import Qwen3 tokenizer dependencies: {error}") from error

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    if getattr(config, "model_type", None) != "qwen3":
        raise Qwen3FormalError("formal input model_type must be qwen3")
    if getattr(config, "max_position_embeddings", None) != protocol["model"]["native_context"]:
        raise Qwen3FormalError("formal input native context differs from protocol")

    joined = protocol["input"]["join_separator"].join(texts)
    joined_sha256 = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    if joined_sha256 != protocol["input"]["joined_text_sha256"]:
        raise Qwen3FormalError("joined corpus text differs from protocol")
    encoded = tokenizer(joined, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids_sha256(token_ids) != protocol["input"]["token_ids_sha256"]:
        raise Qwen3FormalError("Qwen token stream differs from protocol")

    expected_snapshot = {
        "id": protocol["input"]["corpus_id"],
        "config": protocol["input"]["corpus_config"],
        "split": protocol["input"]["corpus_split"],
        "revision": protocol["input"]["corpus_revision"],
        "join_separator": protocol["input"]["join_separator"],
    }
    if any(snapshot_corpus.get(name) != value for name, value in expected_snapshot.items()):
        raise Qwen3FormalError("corpus snapshot declared identity differs from protocol")

    source_state = _source_state()
    manifest = build_input_manifest(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        token_ids=token_ids,
        corpus_snapshot_sha256=snapshot_sha256,
        tokenizer_identity={
            "transformers": transformers.__version__,
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "add_special_tokens": False,
        },
        source_state=source_state,
    )
    _write_json_atomic(output_path, manifest)
    summary = {
        "status": "pass",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "input_manifest": str(output_path),
        "input_manifest_sha256": file_sha256(output_path),
        "source_state": source_state,
        "python": platform.python_version(),
        "transformers": transformers.__version__,
        "corpus_snapshot_sha256": snapshot_sha256,
        "token_count": len(token_ids),
        "token_ids_sha256": token_ids_sha256(token_ids),
        "input_case_count": len(manifest["cases"]),
        "first_input_case_id": manifest["cases"][0]["input_case_id"],
        "last_input_case_id": manifest["cases"][-1]["input_case_id"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
