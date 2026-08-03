#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers import cache_utils as transformers_cache_utils
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3 import modeling_qwen3

from models.qwen3_kivi import Qwen3KiviCache, Qwen3KiviCacheConfig


EXPECTED = {
    "python": "3.10.20",
    "torch": "2.4.1+cu121",
    "transformers": "4.53.2",
    "cache_utils_sha256": "529e858b9bb0ba59a830713bbd5ab594e29f262bbad1bba860ff7de3248e5a5a",
    "qwen3_modeling_sha256": "c12b4c4a34a06e887f3df399b1b09e9966c12d2a19769c258b7379066e585ea4",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    torch.manual_seed(20260803)
    config = Qwen3Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])

    with torch.no_grad():
        reference_cache = DynamicCache()
        reference = model(
            input_ids=input_ids,
            past_key_values=reference_cache,
            use_cache=True,
        )
        kivi_cache = Qwen3KiviCache(
            Qwen3KiviCacheConfig(group_size=4, residual_length=4)
        )
        candidate = model(
            input_ids=input_ids,
            past_key_values=kivi_cache,
            use_cache=True,
        )
        key_storage_changed = any(
            not torch.equal(kivi_cache.key_cache[i], reference_cache.key_cache[i])
            for i in range(2)
        )
        value_storage_changed = any(
            not torch.equal(kivi_cache.value_cache[i], reference_cache.value_cache[i])
            for i in range(2)
        )
        continuation = model(
            input_ids=torch.tensor([[8]]),
            past_key_values=candidate.past_key_values,
            use_cache=True,
        )

    cache_utils_path = Path(transformers_cache_utils.__file__).resolve()
    qwen3_modeling_path = Path(modeling_qwen3.__file__).resolve()
    cache_utils_sha256 = _sha256(cache_utils_path)
    qwen3_modeling_sha256 = _sha256(qwen3_modeling_path)
    checks = {
        "python_version": platform.python_version() == EXPECTED["python"],
        "torch_version": str(torch.__version__) == EXPECTED["torch"],
        "transformers_version": transformers.__version__ == EXPECTED["transformers"],
        "cache_utils_sha256": cache_utils_sha256 == EXPECTED["cache_utils_sha256"],
        "qwen3_modeling_sha256": qwen3_modeling_sha256 == EXPECTED["qwen3_modeling_sha256"],
        "prefill_logits_exact": torch.equal(candidate.logits, reference.logits),
        "key_storage_changed": key_storage_changed,
        "value_storage_changed": value_storage_changed,
        "reported_seq_length": kivi_cache.get_seq_length() == 8,
        "key_quantized_lengths": kivi_cache.key_quantized_lengths == [8, 8],
        "value_quantized_lengths": kivi_cache.value_quantized_lengths == [4, 4],
        "continuation_logits_finite": bool(torch.isfinite(continuation.logits).all().item()),
        "cache_identity_preserved": continuation.past_key_values is kivi_cache,
        "no_persistent_bucket_indices": not hasattr(kivi_cache, "key_bucket_indices")
        and not hasattr(kivi_cache, "value_bucket_indices"),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "device": "cpu",
        "transformers_cache_utils_path": str(cache_utils_path),
        "transformers_cache_utils_sha256": cache_utils_sha256,
        "transformers_qwen3_modeling_path": str(qwen3_modeling_path),
        "transformers_qwen3_modeling_sha256": qwen3_modeling_sha256,
        "qwen3_kivi_source_sha256": _sha256(REPO_ROOT / "models" / "qwen3_kivi.py"),
        "group_size": 4,
        "residual_length": 4,
        "prefill_logits_exact": checks["prefill_logits_exact"],
        "key_storage_changed": key_storage_changed,
        "value_storage_changed": value_storage_changed,
        "reported_seq_length_after_continuation": kivi_cache.get_seq_length(),
        "key_quantized_lengths": kivi_cache.key_quantized_lengths,
        "value_quantized_lengths": kivi_cache.value_quantized_lengths,
        "continuation_logits_finite": checks["continuation_logits_finite"],
        "cache_identity_preserved": checks["cache_identity_preserved"],
        "no_persistent_bucket_indices": checks["no_persistent_bucket_indices"],
        "checks": checks,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
