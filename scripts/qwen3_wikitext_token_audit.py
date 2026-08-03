#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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

from utils.qwen3_cases import (
    QWEN3_ANCHOR_COUNT,
    QWEN3_CONTINUATION_TOKENS,
    QWEN3_PROMPT_LENGTHS,
    QWEN3_SELECTION_ID,
    build_anchor_records,
    minimum_anchor_gap,
    token_ids_sha256,
)


EXPECTED = {
    "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer_json_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "vocab_sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges_sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model_max_position_embeddings": 40960,
    "tokenizer_model_max_length": 131072,
    "corpus_id": "Salesforce/wikitext",
    "corpus_config": "wikitext-2-raw-v1",
    "corpus_split": "test",
    "corpus_revision": "00aa25585682d4957f9e86edc73f59be7419af99",
    "join_separator": "\n\n",
    "rows": 4358,
    "nonempty_rows": 2891,
    "joined_utf8_bytes": 1296370,
    "joined_text_sha256": "696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen Qwen3 tokenizer identity and WikiText anchor inputs"
    )
    parser.add_argument("--model", type=Path, required=True)
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _source_state_identity() -> dict[str, Any]:
    def run(*arguments: str) -> bytes:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout

    commit = run("rev-parse", "HEAD").decode("ascii").strip()
    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "git_commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest() if status else None,
    }


def main() -> None:
    args = _parse_args()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    source_state = _source_state_identity()

    try:
        import datasets
        import transformers
        from datasets import load_dataset
        from transformers import AutoConfig, AutoTokenizer
    except Exception as error:
        raise RuntimeError(f"cannot import tokenizer audit dependencies: {error}") from error

    metadata_hashes = {
        "config_sha256": _sha256(model_path / "config.json"),
        "tokenizer_config_sha256": _sha256(model_path / "tokenizer_config.json"),
        "tokenizer_json_sha256": _sha256(model_path / "tokenizer.json"),
        "vocab_sha256": _sha256(model_path / "vocab.json"),
        "merges_sha256": _sha256(model_path / "merges.txt"),
    }
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    dataset = load_dataset(
        EXPECTED["corpus_id"],
        EXPECTED["corpus_config"],
        split=EXPECTED["corpus_split"],
        revision=EXPECTED["corpus_revision"],
    )
    texts = list(dataset["text"])
    joined = EXPECTED["join_separator"].join(texts)
    joined_bytes = joined.encode("utf-8")
    encoded = tokenizer(joined, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if not isinstance(token_ids, list):
        token_ids = token_ids.tolist()
    anchors = build_anchor_records(token_ids)
    raw_corpus = {
        "rows": len(texts),
        "nonempty_rows": sum(bool(text.strip()) for text in texts),
        "joined_utf8_bytes": len(joined_bytes),
        "joined_text_sha256": hashlib.sha256(joined_bytes).hexdigest(),
    }
    expected_raw = {
        name: EXPECTED[name]
        for name in (
            "rows",
            "nonempty_rows",
            "joined_utf8_bytes",
            "joined_text_sha256",
        )
    }
    checks = {
        "metadata_hashes": all(
            metadata_hashes[name] == EXPECTED[name] for name in metadata_hashes
        ),
        "model_type": getattr(config, "model_type", None) == "qwen3",
        "model_max_position_embeddings": getattr(
            config, "max_position_embeddings", None
        )
        == EXPECTED["model_max_position_embeddings"],
        "tokenizer_model_max_length": tokenizer.model_max_length
        == EXPECTED["tokenizer_model_max_length"],
        "raw_corpus_identity": raw_corpus == expected_raw,
        "token_ids_nonempty": len(token_ids) > max(QWEN3_PROMPT_LENGTHS),
        "anchor_count": len(anchors) == QWEN3_ANCHOR_COUNT,
        "anchor_windows_nonoverlapping": minimum_anchor_gap(anchors)
        >= max(QWEN3_PROMPT_LENGTHS) + QWEN3_CONTINUATION_TOKENS,
        "source_clean": source_state["dirty"] is False,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_state": source_state,
        "source_hashes": {
            "qwen3_cases_sha256": _sha256(REPO_ROOT / "utils" / "qwen3_cases.py"),
            "token_audit_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "python": platform.python_version(),
        "datasets": datasets.__version__,
        "transformers": transformers.__version__,
        "model_path": str(model_path),
        "model_revision": EXPECTED["model_revision"],
        "model_type": getattr(config, "model_type", None),
        "model_max_position_embeddings": getattr(
            config, "max_position_embeddings", None
        ),
        "metadata_hashes": metadata_hashes,
        "tokenizer": {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "add_special_tokens": False,
        },
        "corpus": {
            "id": EXPECTED["corpus_id"],
            "config": EXPECTED["corpus_config"],
            "split": EXPECTED["corpus_split"],
            "revision": EXPECTED["corpus_revision"],
            "join_separator": EXPECTED["join_separator"],
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            **raw_corpus,
            "token_count": len(token_ids),
            "token_ids_sha256": token_ids_sha256(token_ids),
        },
        "selection": {
            "selection_id": QWEN3_SELECTION_ID,
            "prompt_lengths": list(QWEN3_PROMPT_LENGTHS),
            "continuation_tokens": QWEN3_CONTINUATION_TOKENS,
            "anchor_count": QWEN3_ANCHOR_COUNT,
            "minimum_anchor_gap": minimum_anchor_gap(anchors),
            "anchors": anchors,
        },
        "checks": checks,
    }
    _write_json_atomic(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"token_audit_json: {output_path}")
    print(f"token_audit_json_sha256: {_sha256(output_path)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
