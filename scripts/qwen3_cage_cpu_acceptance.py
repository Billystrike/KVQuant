#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers import cache_utils as transformers_cache_utils
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3 import modeling_qwen3

from models.cage_config import CageConfig
from models.qwen3_cage import Qwen3CageCache, install_qwen3_cage_attention


EXPECTED_TRANSFORMERS_VERSION = "4.53.2"
EXPECTED_PYTHON_VERSION = "3.10.20"
EXPECTED_TORCH_VERSION = "2.4.1+cu121"
EXPECTED_CACHE_UTILS_SHA256 = "529e858b9bb0ba59a830713bbd5ab594e29f262bbad1bba860ff7de3248e5a5a"
EXPECTED_QWEN3_MODELING_SHA256 = "c12b4c4a34a06e887f3df399b1b09e9966c12d2a19769c258b7379066e585ea4"


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
    installed = install_qwen3_cage_attention(model)
    cage_config = CageConfig(
        cage_enable=True,
        cage_k_group_sizes=[2, 4, 8],
        cage_k_clip_percentiles=[1.0, 1.0, 1.0],
        cage_v_group_sizes=[2, 4, 8],
        cage_v_clip_percentiles=[1.0, 1.0, 1.0],
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])

    with torch.no_grad():
        reference = model(
            input_ids=input_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        cage_cache = Qwen3CageCache(cage_config, residual_length=4)
        candidate = model(
            input_ids=input_ids,
            past_key_values=cage_cache,
            use_cache=True,
        )
        continuation = model(
            input_ids=torch.tensor([[8]]),
            past_key_values=candidate.past_key_values,
            use_cache=True,
        )

    prefill_max_abs_delta = float((candidate.logits - reference.logits).abs().max().item())
    cache_utils_path = Path(transformers_cache_utils.__file__).resolve()
    qwen3_modeling_path = Path(modeling_qwen3.__file__).resolve()
    cache_utils_sha256 = _sha256(cache_utils_path)
    qwen3_modeling_sha256 = _sha256(qwen3_modeling_path)
    report = {
        "schema_version": 1,
        "status": "pass",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "transformers_cache_utils_path": str(cache_utils_path),
        "transformers_cache_utils_sha256": cache_utils_sha256,
        "transformers_qwen3_modeling_path": str(qwen3_modeling_path),
        "transformers_qwen3_modeling_sha256": qwen3_modeling_sha256,
        "qwen3_cage_source_sha256": _sha256(Path(__file__).resolve().parents[1] / "models" / "qwen3_cage.py"),
        "device": "cpu",
        "installed_attention_modules": installed,
        "prefill_logits_exact": bool(torch.equal(candidate.logits, reference.logits)),
        "prefill_logits_max_abs_delta": prefill_max_abs_delta,
        "reported_seq_length_after_continuation": cage_cache.get_seq_length(),
        "key_quantized_lengths": cage_cache.key_quantized_lengths,
        "value_quantized_lengths": cage_cache.value_quantized_lengths,
        "key_bucket_shapes": [
            [list(index.shape) for index in policy.key_bucket_indices]
            for policy in cage_cache.layer_policies
        ],
        "value_bucket_shapes": [
            [list(index.shape) for index in policy.value_bucket_indices]
            for policy in cage_cache.layer_policies
        ],
        "continuation_logits_finite": bool(torch.isfinite(continuation.logits).all().item()),
        "cache_identity_preserved": continuation.past_key_values is cage_cache,
    }
    failures = [
        name
        for name, passed in {
            "python_version": platform.python_version() == EXPECTED_PYTHON_VERSION,
            "torch_version": str(torch.__version__) == EXPECTED_TORCH_VERSION,
            "transformers_version": transformers.__version__ == EXPECTED_TRANSFORMERS_VERSION,
            "transformers_cache_utils_sha256": cache_utils_sha256 == EXPECTED_CACHE_UTILS_SHA256,
            "transformers_qwen3_modeling_sha256": qwen3_modeling_sha256 == EXPECTED_QWEN3_MODELING_SHA256,
            "installed_attention_modules": installed == 2,
            "prefill_logits_exact": report["prefill_logits_exact"],
            "reported_seq_length_after_continuation": cage_cache.get_seq_length() == 8,
            "key_quantized_lengths": cage_cache.key_quantized_lengths == [8, 8],
            "value_quantized_lengths": cage_cache.value_quantized_lengths == [4, 4],
            "continuation_logits_finite": report["continuation_logits_finite"],
            "cache_identity_preserved": report["cache_identity_preserved"],
        }.items()
        if not passed
    ]
    if failures:
        report["status"] = "fail"
        report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
