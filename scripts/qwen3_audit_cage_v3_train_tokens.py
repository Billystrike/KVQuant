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

from utils.qwen3_cage_v3_audit import CANDIDATE_ANCHOR_COUNTS, candidate_selection_audit, token_ids_sha256


EXPECTED_MODEL = {
    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer_json_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "vocab_sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges_sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "native_context": 40960,
    "tokenizer_model_max_length": 131072,
}
EXPECTED_CORPUS = {
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
    parser = argparse.ArgumentParser(description="Audit Qwen3-tokenized WikiText-2 train inputs before freezing CAGE-v3")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_path = args.model.resolve()
    snapshot_path = args.corpus_snapshot.resolve()
    output_path = args.output.resolve()
    source_state = _git_state()

    import transformers
    from transformers import AutoConfig, AutoTokenizer

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or set(snapshot) != {
        "schema_version", "purpose", "claim_eligible", "corpus", "texts"
    }:
        raise ValueError("CAGE-v3 train snapshot schema mismatch")
    if snapshot["schema_version"] != 1 or snapshot["claim_eligible"] is not False:
        raise ValueError("CAGE-v3 train snapshot boundary mismatch")
    corpus = snapshot["corpus"]
    texts = snapshot["texts"]
    if not isinstance(corpus, dict) or not isinstance(texts, list) or any(not isinstance(x, str) for x in texts):
        raise ValueError("CAGE-v3 train snapshot contents are invalid")

    metadata_hashes = {
        "config_sha256": _sha256(model_path / "config.json"),
        "tokenizer_config_sha256": _sha256(model_path / "tokenizer_config.json"),
        "tokenizer_json_sha256": _sha256(model_path / "tokenizer.json"),
        "vocab_sha256": _sha256(model_path / "vocab.json"),
        "merges_sha256": _sha256(model_path / "merges.txt"),
    }
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    joined = EXPECTED_CORPUS["join_separator"].join(texts)
    joined_bytes = joined.encode("utf-8")
    encoded = tokenizer(joined, add_special_tokens=False)["input_ids"]
    token_ids = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
    raw_identity = {
        "rows": len(texts),
        "nonempty_rows": sum(bool(text.strip()) for text in texts),
        "joined_utf8_bytes": len(joined_bytes),
        "joined_text_sha256": hashlib.sha256(joined_bytes).hexdigest(),
    }
    selections = [candidate_selection_audit(token_ids, count) for count in CANDIDATE_ANCHOR_COUNTS]
    checks = {
        "source_clean": source_state["dirty"] is False,
        "model_metadata": all(metadata_hashes[key] == EXPECTED_MODEL[key] for key in metadata_hashes),
        "model_type": getattr(config, "model_type", None) == "qwen3",
        "native_context": getattr(config, "max_position_embeddings", None) == EXPECTED_MODEL["native_context"],
        "tokenizer_model_max_length": tokenizer.model_max_length == EXPECTED_MODEL["tokenizer_model_max_length"],
        "corpus_identity": all(corpus.get(key) == value for key, value in EXPECTED_CORPUS.items()),
        "snapshot_raw_identity": all(corpus.get(key) == value for key, value in raw_identity.items()),
        "train_split_not_previously_used_validation_or_test": corpus.get("split") not in {"validation", "test"},
        "token_ids_nonempty": len(token_ids) > 4096,
        "candidate_windows_nonoverlapping": all(row["windows_nonoverlapping"] for row in selections),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "claim_eligible": False,
        "purpose": "identity_and_selection_feasibility_audit_before_cage_v3_protocol_freeze",
        "failures": failures,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_state": source_state,
        "python": platform.python_version(),
        "transformers": transformers.__version__,
        "model": {
            "path": str(model_path),
            "revision": EXPECTED_MODEL["revision"],
            "metadata_hashes": metadata_hashes,
            "model_type": getattr(config, "model_type", None),
            "native_context": getattr(config, "max_position_embeddings", None),
        },
        "tokenizer": {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "add_special_tokens": False,
        },
        "corpus": {
            **EXPECTED_CORPUS,
            "dataset_fingerprint": corpus.get("dataset_fingerprint"),
            "datasets_version": corpus.get("datasets_version"),
            **raw_identity,
            "snapshot_sha256": _sha256(snapshot_path),
            "token_count": len(token_ids),
            "token_ids_sha256": token_ids_sha256(token_ids),
        },
        "candidate_selection_audits": selections,
        "checks": checks,
        "freeze_boundary": {
            "exact_anchor_count_frozen": False,
            "exact_anchor_partition_frozen": False,
            "candidate_hyperparameters_frozen": False,
            "gpu_execution_authorized_by_this_artifact": False,
        },
        "source_hashes": {
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
            "audit_utils_sha256": _sha256(REPO_ROOT / "utils" / "qwen3_cage_v3_audit.py"),
        },
    }
    _write_json_atomic(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"audit_json: {output_path}")
    print(f"audit_json_sha256: {_sha256(output_path)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
