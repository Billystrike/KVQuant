#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llama2_cage_v3_transfer_quality_manifest import (
    build_input_manifest,
    load_static_preflight_artifacts,
    validate_corpus_snapshot,
)
from utils.llama2_cage_v3_transfer_quality_protocol import (
    load_transfer_quality_protocol,
)
from utils.qwen3_cage_v4_data import file_sha256


TOKENIZER_FILES = {
    "tokenizer_config.json": "f514e7c3008881b6ba7e6a0cdb44c71ce47dc335920dac143ae7bc788197e53a",
    "tokenizer.json": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
    "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
}


def _source_state() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("repository must be clean for frozen input-manifest construction")
    return {"git_commit": commit, "dirty": False}


def _write_fresh_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen input manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen Llama-2 CAGE-v3 50-anchor input manifest without loading model weights"
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-artifacts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol, protocol_sha256 = load_transfer_quality_protocol(
        args.protocol.resolve(), repo_root=REPO_ROOT
    )
    load_static_preflight_artifacts(
        args.preflight_artifacts.resolve(), verify_server_files=True
    )
    source_state = _source_state()
    model = args.model.resolve()
    if str(model) != protocol["model"]["reference"]:
        raise RuntimeError("model path differs from the frozen protocol")
    config_path = model / "config.json"
    if file_sha256(config_path) != protocol["model"]["config_sha256"]:
        raise RuntimeError("model config digest mismatch")
    tokenizer_files = {}
    for name, expected_sha in TOKENIZER_FILES.items():
        path = model / name
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise RuntimeError(f"tokenizer file identity mismatch: {name}")
        tokenizer_files[name] = {"sha256": expected_sha, "size_bytes": path.stat().st_size}

    try:
        import datasets
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except Exception as error:
        raise RuntimeError(f"cannot import manifest dependencies: {error}") from error

    tokenizer = AutoTokenizer.from_pretrained(
        str(model), use_fast=False, local_files_only=True
    )
    input_spec = protocol["input"]
    dataset = load_dataset(
        input_spec["corpus_id"],
        input_spec["corpus_config"],
        split=input_spec["split"],
        revision=input_spec["revision"],
    )
    snapshot, token_ids = validate_corpus_snapshot(
        input_spec=input_spec,
        texts=list(dataset["text"]),
        tokenizer=tokenizer,
        dataset_fingerprint=getattr(dataset, "_fingerprint", None),
        datasets_version=datasets.__version__,
    )
    tokenizer_identity = {
        "implementation": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "use_fast": False,
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "files": tokenizer_files,
    }
    manifest = build_input_manifest(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        token_ids=token_ids,
        corpus_snapshot=snapshot,
        tokenizer_identity=tokenizer_identity,
        source_state=source_state,
    )
    _write_fresh_json(args.output.resolve(), manifest)
    summary = {
        "status": "pass",
        "manifest_id": manifest["manifest_id"],
        "source_state": source_state,
        "corpus_snapshot": snapshot,
        "tokenizer_identity": tokenizer_identity,
        "anchor_count": manifest["selection"]["anchor_count"],
        "input_record_count": manifest["selection"]["input_record_count"],
        "expanded_method_case_count": manifest["selection"]["expanded_method_case_count"],
        "boundary": manifest["boundary"],
        "output_path": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output.resolve()),
        "output_size_bytes": args.output.resolve().stat().st_size,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
