#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qwen3_cage_v2_protocol import build_cage_v2_dev_manifest
from utils.qwen3_formal import file_sha256, read_corpus_snapshot
from utils.qwen3_cases import token_ids_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build claim-ineligible CAGE-v2 development inputs")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_protocol(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("CAGE-v2 feasibility protocol schema mismatch")
    if payload.get("status") != "development_protocol_not_frozen_for_final_claims":
        raise ValueError("CAGE-v2 feasibility protocol must remain development-only")
    return payload, file_sha256(path)


def _git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {"git_commit": commit, "dirty": dirty}


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
    model_path = args.model.resolve()
    snapshot_path = args.corpus_snapshot.resolve()
    output_path = args.output.resolve()
    protocol, protocol_sha256 = _load_protocol(protocol_path)
    if str(model_path) != protocol["model"]["reference"]:
        raise ValueError("model path differs from the CAGE-v2 development protocol")
    snapshot_identity, texts = read_corpus_snapshot(snapshot_path)
    data = protocol["development_data"]
    expected_corpus = {
        "id": data["corpus_id"],
        "config": data["corpus_config"],
        "split": data["corpus_split"],
        "revision": data["corpus_revision"],
        "join_separator": data["join_separator"],
    }
    if any(snapshot_identity.get(key) != value for key, value in expected_corpus.items()):
        raise ValueError("corpus snapshot identity differs from the development protocol")

    import transformers
    from transformers import AutoConfig, AutoTokenizer

    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if model_config.model_type != "qwen3":
        raise ValueError("development model must be Qwen3")
    if model_config.max_position_embeddings != protocol["model"]["native_context"]:
        raise ValueError("development model native context mismatch")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    joined = data["join_separator"].join(texts)
    encoded = tokenizer(joined, add_special_tokens=False)["input_ids"]
    token_ids = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
    manifest = build_cage_v2_dev_manifest(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        token_ids=token_ids,
        corpus_snapshot_sha256=file_sha256(snapshot_path),
        corpus_identity=expected_corpus,
        tokenizer_identity={
            "transformers": transformers.__version__,
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "add_special_tokens": False,
        },
        source_state=_git_state(),
    )
    manifest["joined_text_sha256"] = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    _write_atomic(output_path, manifest)
    print(
        json.dumps(
            {
                "status": "pass",
                "claim_eligibility": "development_only",
                "protocol_sha256": protocol_sha256,
                "corpus_snapshot_sha256": file_sha256(snapshot_path),
                "token_count": len(token_ids),
                "token_ids_sha256": token_ids_sha256(token_ids),
                "case_count": len(manifest["cases"]),
                "output": str(output_path),
                "output_sha256": file_sha256(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
