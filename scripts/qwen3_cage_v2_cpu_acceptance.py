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
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from models.cage_v2_quant import fake_quant_k_sparse_refinement, fake_quant_k_uniform
from models.qwen3_cage import install_qwen3_cage_attention
from models.qwen3_cage_v2 import CageV2Config, Qwen3CageV2Cache
from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes
from utils.qwen3_memory import estimate_qwen3_kitty_bytes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
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
    return Qwen3ForCausalLM(config).eval()


def _closest_uniform_candidate(*, seq_len: int, boosted_channels: int) -> dict:
    target = estimate_qwen3_kitty_bytes(
        seq_len=seq_len,
        boosted_channels=boosted_channels,
    )["model_total_bytes"]
    rows = []
    for sink_length in (0, 32):
        for residual_length in range(32, 513, 32):
            for one_bit_channels in (0, 8, 16, 24, 32):
                for two_bit_channels in (0, 8, 16, 24, 32):
                    if one_bit_channels + two_bit_channels > 64:
                        continue
                    report = estimate_qwen3_cage_v2_bytes(
                        seq_len=seq_len,
                        residual_length=residual_length,
                        one_bit_channels=one_bit_channels,
                        two_bit_channels=two_bit_channels,
                        sink_length=sink_length,
                    )
                    total = report["model_total_bytes"]
                    rows.append(
                        (
                            abs(total - target),
                            total,
                            sink_length,
                            residual_length,
                            one_bit_channels,
                            two_bit_channels,
                        )
                    )
    _, total, sink, residual, one_bit, two_bit = min(rows)
    return {
        "seq_len": seq_len,
        "target": "kitty-12.5pct" if boosted_channels == 16 else "kitty-pro-25pct",
        "target_bytes": target,
        "cage_v2_bytes": total,
        "relative_delta": (total - target) / target,
        "sink_length": sink,
        "residual_length": residual,
        "one_bit_channels_per_head": one_bit,
        "two_bit_channels_per_head": two_bit,
        "selection_note": "byte-only draft enumeration; not a frozen quality selection",
    }


def main() -> None:
    torch.manual_seed(20260809)
    model = _tiny_model()
    installed = install_qwen3_cage_attention(model)
    cache = Qwen3CageV2Cache(
        CageV2Config(
            one_bit_channels=1,
            two_bit_channels=2,
            key_base_group_size=4,
            key_refinement_group_size=4,
            value_group_size=4,
            sink_length=1,
        ),
        residual_length=4,
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with torch.inference_mode():
        reference = model(
            input_ids=input_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        candidate = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )
        continuation = model(
            input_ids=torch.tensor([[8]]),
            past_key_values=cache,
            use_cache=True,
        )

    key = torch.randn(1, 2, 16, 8)
    base = fake_quant_k_uniform(key, group_size=8, bits=2)
    refined = fake_quant_k_sparse_refinement(
        key,
        one_bit_indices=torch.tensor([[2, 3], [2, 3]]),
        two_bit_indices=torch.tensor([[0, 1], [0, 1]]),
        base_group_size=8,
        refinement_group_size=8,
    )
    selected = torch.tensor([0, 1, 2, 3])
    base_selected_mse = (key[..., selected] - base[..., selected]).float().square().mean().item()
    refined_selected_mse = (
        (key[..., selected] - refined[..., selected]).float().square().mean().item()
    )

    checks = {
        "installed_attention_modules": installed == 2,
        "prefill_logits_exact": bool(torch.equal(candidate.logits, reference.logits)),
        "continuation_logits_finite": bool(torch.isfinite(continuation.logits).all()),
        "cache_identity_preserved": continuation.past_key_values is cache,
        "key_quantized_lengths": cache.key_quantized_lengths == [5, 5],
        "value_quantized_lengths": cache.value_quantized_lengths == [4, 4],
        "value_policy_is_uniform": all(
            not hasattr(policy, "value_bucket_indices") for policy in cache.layer_policies
        ),
        "refinement_improves_selected_key_mse": refined_selected_mse < base_selected_mse,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": checks,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "installed_attention_modules": installed,
        "base_selected_key_mse": base_selected_mse,
        "refined_selected_key_mse": refined_selected_mse,
        "selected_key_mse_relative_change": refined_selected_mse / base_selected_mse - 1.0,
        "key_quantized_lengths": cache.key_quantized_lengths,
        "value_quantized_lengths": cache.value_quantized_lengths,
        "source_sha256": {
            "cage_v2_quant": _sha256(REPO_ROOT / "models" / "cage_v2_quant.py"),
            "qwen3_cage_v1_adapter": _sha256(REPO_ROOT / "models" / "qwen3_cage.py"),
            "qwen3_cage_v2": _sha256(REPO_ROOT / "models" / "qwen3_cage_v2.py"),
            "qwen3_cage_v2_memory": _sha256(REPO_ROOT / "utils" / "qwen3_cage_v2.py"),
            "acceptance_runner": _sha256(Path(__file__).resolve()),
        },
        "draft_byte_matches": [
            _closest_uniform_candidate(seq_len=length, boosted_channels=boosted)
            for length in (1024, 2048, 4032)
            for boosted in (16, 32)
        ],
        "claim_boundary": {
            "representation": "fake_quant_accuracy_simulation",
            "memory": "packed_paper_estimate",
            "unsupported": [
                "realized_cuda_memory",
                "latency",
                "throughput",
                "final_quality_claim",
            ],
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if failures:
        raise RuntimeError(f"CAGE-v2 CPU acceptance failed: {failures}")


if __name__ == "__main__":
    main()
