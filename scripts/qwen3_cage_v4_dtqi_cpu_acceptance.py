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

from models.cage_importance import compute_key_importance
from models.qwen3_cage import install_qwen3_cage_attention
from models.qwen3_cage_v4 import (
    CageV4DTQIConfig,
    Qwen3CageV4DTQICache,
    compute_dtqi_key_importance,
)
from utils.qwen3_cage_v2 import estimate_qwen3_cage_v2_bytes
from utils.qwen3_cage_v4_dtqi_protocol import file_sha256, load_dtqi_protocol


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


def _quotas_for_length(plan: dict, prompt_length: int) -> list[int]:
    rows = [
        row
        for row in plan["plans"]
        if row["family_id"] == "pure-sr2-sink32-uniform32"
        and row["prompt_length"] == prompt_length
    ]
    if len(rows) != 1:
        raise RuntimeError("matching CAGE-v3 sink32 quota plan missing")
    return list(rows[0]["layer_two_bit_channel_quotas"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CPU acceptance for the single frozen CAGE-v4-DTQI candidate"
    )
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol, protocol_sha256 = load_dtqi_protocol(
        protocol_path,
        repo_root=REPO_ROOT,
    )

    quota_path = REPO_ROOT / protocol["quota_plan"]["path"]
    quota_plan = json.loads(quota_path.read_text(encoding="utf-8"))
    memory_checks = []
    for point in protocol["candidate"]["points"]:
        quotas = _quotas_for_length(quota_plan, point["prompt_length"])
        report = estimate_qwen3_cage_v2_bytes(
            seq_len=point["prompt_length"],
            residual_length=point["residual_length"],
            one_bit_channels=0,
            two_bit_channels=quotas,
            sink_length=32,
        )
        memory_checks.append(
            {
                "prompt_length": point["prompt_length"],
                "residual_length": point["residual_length"],
                "recent_query_window": point["recent_query_window"],
                "quota_total": sum(quotas),
                "quota_counts": {
                    "48": quotas.count(48),
                    "32": quotas.count(32),
                    "16": quotas.count(16),
                },
                "model_total_bytes": report["model_total_bytes"],
                "expected_bytes": point["packed_bytes"],
                "kitty_pro_target_bytes": point["kitty_pro_target_bytes"],
                "matches_cage_v3": report["model_total_bytes"] == point["packed_bytes"],
                "within_kitty_pro_target": report["model_total_bytes"]
                <= point["kitty_pro_target_bytes"],
                "window_equals_residual": point["recent_query_window"]
                == point["residual_length"],
            }
        )

    query = torch.ones(1, 1, 4, 2)
    query[:, :, :2, 0] = 10.0
    query[:, :, -2:, 1] = 6.0
    key = torch.tensor([[[[0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [2.0, 2.0]]]])
    global_importance = compute_key_importance(query, key, num_key_value_groups=1)
    dtqi_importance = compute_dtqi_key_importance(
        query,
        key,
        num_key_value_groups=1,
        recent_query_window=2,
    )

    torch.manual_seed(20260813)
    model = _tiny_model()
    installed = install_qwen3_cage_attention(model)
    cache = Qwen3CageV4DTQICache(
        CageV4DTQIConfig(
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
        reference = model(
            input_ids=input_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        candidate = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        continuation = model(
            input_ids=torch.tensor([[8]]),
            past_key_values=cache,
            use_cache=True,
        )

    policy_checks = []
    for layer_idx, policy in enumerate(cache.layer_policies):
        policy_checks.append(
            {
                "layer_idx": layer_idx,
                "importance_policy": policy.importance_policy,
                "global_query_weight": policy.global_query_weight,
                "recent_query_weight": policy.recent_query_weight,
                "recent_query_window": cache.residual_length,
                "one_bit_channels": policy.one_bit_channels,
                "two_bit_channels": policy.two_bit_channels,
                "one_bit_index_shape": list(policy.one_bit_indices.shape),
                "two_bit_index_shape": list(policy.two_bit_indices.shape),
                "no_value_adaptive_state": not hasattr(policy, "value_bucket_indices"),
            }
        )

    checks = {
        "installed_two_attention_modules": installed == 2,
        "prefill_logits_exact": bool(torch.equal(reference.logits, candidate.logits)),
        "continuation_logits_finite": bool(torch.isfinite(continuation.logits).all()),
        "cache_identity_preserved": continuation.past_key_values is cache,
        "dtqi_changes_synthetic_ranking": int(global_importance.argmax().item()) == 0
        and int(dtqi_importance.argmax().item()) == 1,
        "weights_fixed_half_half": all(
            row["global_query_weight"] == 0.5
            and row["recent_query_weight"] == 0.5
            for row in policy_checks
        ),
        "recent_window_bound_to_residual": all(
            row["recent_query_window"] == cache.residual_length
            for row in policy_checks
        ),
        "one_bit_path_empty": all(
            row["one_bit_channels"] == 0
            and row["one_bit_index_shape"] == [2, 0]
            for row in policy_checks
        ),
        "per_layer_two_bit_quotas_applied": [
            row["two_bit_channels"] for row in policy_checks
        ]
        == [4, 2],
        "value_path_uniform": all(
            row["no_value_adaptive_state"] for row in policy_checks
        ),
        "all_memory_points_exact": all(
            row["matches_cage_v3"]
            and row["within_kitty_pro_target"]
            and row["window_equals_residual"]
            and row["quota_total"] == 1152
            and row["quota_counts"] == {"48": 12, "32": 12, "16": 12}
            for row in memory_checks
        ),
        "gpu_execution_still_blocked": not protocol["execution_boundary"][
            "gpu_acceptance_authorized"
        ]
        and not protocol["execution_boundary"]["gpu_full_screen_authorized"],
        "holdout_and_test_still_blocked": not protocol["execution_boundary"][
            "holdout_method_metrics_authorized"
        ]
        and not protocol["execution_boundary"]["pg19_test_access_authorized"],
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "candidate_id": "cage-v4-dtqi",
        "claim_eligible": False,
        "failures": failures,
        "checks": checks,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "protocol_sha256": protocol_sha256,
        "synthetic_importance": {
            "global": global_importance.tolist(),
            "dtqi": dtqi_importance.tolist(),
            "global_top_channel": int(global_importance.argmax().item()),
            "dtqi_top_channel": int(dtqi_importance.argmax().item()),
        },
        "policy_checks": policy_checks,
        "memory_checks": memory_checks,
        "source_sha256": {
            "runner": file_sha256(Path(__file__).resolve()),
            "protocol": file_sha256(protocol_path),
            "dtqi_runtime": file_sha256(REPO_ROOT / "models" / "qwen3_cage_v4.py"),
            "cage_v2_runtime": file_sha256(REPO_ROOT / "models" / "qwen3_cage_v2.py"),
            "cage_v2_quant": file_sha256(REPO_ROOT / "models" / "cage_v2_quant.py"),
            "protocol_utils": file_sha256(
                REPO_ROOT / "utils" / "qwen3_cage_v4_dtqi_protocol.py"
            ),
            "quota_plan": file_sha256(quota_path),
        },
        "claim_boundary": {
            "representation": "fake_quant_accuracy_simulation",
            "memory": "packed_paper_estimate_only",
            "gpu_acceptance_authorized": False,
            "gpu_full_screen_authorized": False,
            "holdout_accessed": False,
            "pg19_test_accessed": False,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if failures:
        raise RuntimeError(f"CAGE-v4-DTQI CPU acceptance failed: {failures}")


if __name__ == "__main__":
    main()
