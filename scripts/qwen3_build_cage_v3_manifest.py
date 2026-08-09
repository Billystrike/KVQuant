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

from utils.qwen3_cage_v3_manifest import build_cage_v3_manifest
from utils.qwen3_cage_v3_protocol import file_sha256, load_cage_v3_protocol


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
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout)
    return {"git_commit": commit, "dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen CAGE-v3 train development manifest")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus-snapshot", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_sha256 = load_cage_v3_protocol(args.protocol.resolve())
    model_path = args.model.resolve()
    snapshot_path = args.corpus_snapshot.resolve()
    audit_path = args.token_audit.resolve()
    output_path = args.output.resolve()
    source_state = _git_state()
    if source_state["dirty"]:
        raise RuntimeError("CAGE-v3 manifest build requires a clean KVQuant tree")
    if str(model_path) != protocol["model"]["reference"]:
        raise ValueError("CAGE-v3 model path differs from protocol")
    if file_sha256(snapshot_path) != protocol["development_data"]["snapshot_sha256"]:
        raise ValueError("CAGE-v3 snapshot file hash differs from protocol")
    if file_sha256(audit_path) != protocol["development_data"]["token_audit_sha256"]:
        raise ValueError("CAGE-v3 token audit file hash differs from protocol")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    token_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    texts = snapshot.get("texts")
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise ValueError("CAGE-v3 snapshot texts are invalid")

    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    tokenizer_identity = {
        "transformers": transformers.__version__,
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "add_special_tokens": False,
    }
    tokenizer.model_max_length = 10**30
    joined = protocol["development_data"]["join_separator"].join(texts)
    encoded = tokenizer(joined, add_special_tokens=False)["input_ids"]
    token_ids = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
    manifest = build_cage_v3_manifest(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        token_ids=token_ids,
        snapshot_sha256=file_sha256(snapshot_path),
        token_audit=token_audit,
        tokenizer_identity=tokenizer_identity,
        source_state=source_state,
    )
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite CAGE-v3 manifest: {output_path}")
    _write_atomic(output_path, manifest)
    print(json.dumps({
        "status": "pass",
        "claim_eligible": False,
        "protocol_sha256": protocol_sha256,
        "snapshot_sha256": file_sha256(snapshot_path),
        "token_audit_sha256": file_sha256(audit_path),
        "token_count": manifest["token_count"],
        "token_ids_sha256": manifest["token_ids_sha256"],
        "anchor_count": manifest["selection"]["anchor_count"],
        "case_count": len(manifest["cases"]),
        "partitions": protocol["development_data"]["partitions"],
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
