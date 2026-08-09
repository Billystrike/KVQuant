#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from models.qwen3_cage import install_qwen3_cage_attention
from models.qwen3_cage_v2 import CageV2Config, Qwen3CageV2Cache
from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes
from utils.qwen3_cage_v3_protocol import (
    calibrated_tiered_two_bit_quotas,
    file_sha256,
    load_cage_v3_protocol,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU acceptance for the frozen CAGE-v3 method boundary")
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol, protocol_sha256 = load_cage_v3_protocol(protocol_path)
    torch.manual_seed(20260809)
    model = _tiny_model()
    installed = install_qwen3_cage_attention(model)
    cache = Qwen3CageV2Cache(
        CageV2Config(
            one_bit_channels=[0, 0],
            two_bit_channels=[4, 2],
            key_base_group_size=4,
            key_refinement_group_size=4,
            value_group_size=4,
            sink_length=1,
        ),
        residual_length=4,
    )
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with torch.inference_mode():
        reference = model(input_ids=input_ids, past_key_values=DynamicCache(), use_cache=True)
        candidate = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        continuation = model(
            input_ids=torch.tensor([[8]]),
            past_key_values=cache,
            use_cache=True,
        )

    quotas = calibrated_tiered_two_bit_quotas([float(index) for index in range(36)])
    memory_checks = []
    for family in protocol["candidate_families"]:
        counts = 32 if family["layer_allocation"] == "uniform32" else quotas
        for point in family["points"]:
            report = estimate_qwen3_cage_v2_bytes(
                seq_len=point["prompt_length"],
                residual_length=point["residual_length"],
                one_bit_channels=0,
                two_bit_channels=counts,
                sink_length=family["sink_length"],
            )
            memory_checks.append({
                "method_id": point["method_id"],
                "model_total_bytes": report["model_total_bytes"],
                "expected_bytes": point["packed_bytes"],
                "matches": report["model_total_bytes"] == point["packed_bytes"],
                "within_target": report["model_total_bytes"] <= point["target_bytes"],
            })

    policy_checks = []
    for layer_idx, policy in enumerate(cache.layer_policies):
        policy_checks.append({
            "layer_idx": layer_idx,
            "one_bit_channels": policy.one_bit_channels,
            "two_bit_channels": policy.two_bit_channels,
            "one_bit_index_shape": list(policy.one_bit_indices.shape),
            "two_bit_index_shape": list(policy.two_bit_indices.shape),
            "no_value_adaptive_state": not hasattr(policy, "value_bucket_indices"),
        })
    checks = {
        "installed_two_attention_modules": installed == 2,
        "prefill_logits_exact": bool(torch.equal(reference.logits, candidate.logits)),
        "continuation_logits_finite": bool(torch.isfinite(continuation.logits).all()),
        "cache_identity_preserved": continuation.past_key_values is cache,
        "one_bit_path_empty": all(row["one_bit_channels"] == 0 and row["one_bit_index_shape"] == [2, 0] for row in policy_checks),
        "per_layer_two_bit_quotas_applied": [row["two_bit_channels"] for row in policy_checks] == [4, 2],
        "value_path_uniform": all(row["no_value_adaptive_state"] for row in policy_checks),
        "all_protocol_memory_points_exact": all(row["matches"] and row["within_target"] for row in memory_checks),
        "calibrated_quota_total_preserved": sum(quotas) == 1152,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": checks,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "protocol_sha256": protocol_sha256,
        "policy_checks": policy_checks,
        "memory_checks": memory_checks,
        "calibrated_quota_example": quotas,
        "source_sha256": {
            "runner": file_sha256(Path(__file__).resolve()),
            "protocol": file_sha256(protocol_path),
            "qwen3_cage_v2_runtime": file_sha256(REPO_ROOT / "models" / "qwen3_cage_v2.py"),
            "cage_v2_quant": file_sha256(REPO_ROOT / "models" / "cage_v2_quant.py"),
            "cage_v3_protocol_utils": file_sha256(REPO_ROOT / "utils" / "qwen3_cage_v3_protocol.py"),
        },
        "claim_boundary": {
            "claim_eligible": False,
            "representation": "fake_quant_accuracy_simulation",
            "memory": "packed_paper_estimate_only",
            "gpu_screen_authorized": False,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if failures:
        raise RuntimeError(f"CAGE-v3 CPU acceptance failed: {failures}")


if __name__ == "__main__":
    main()
