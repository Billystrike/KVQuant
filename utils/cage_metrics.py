from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from utils.cage_experiment_schema import ExperimentPointError


DEFAULT_TOP_K = 10
DEFAULT_METRICS_FILENAME = "cage_perturbation_metrics.jsonl"
_EPS = 1e-12


def compute_cage_perturbation_metrics(
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    key_states_hat: torch.Tensor,
    value_states: torch.Tensor,
    value_states_hat: torch.Tensor,
    o_proj_weight: torch.Tensor,
    key_importance: torch.Tensor,
    value_importance: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    num_key_value_groups: int | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, float]:
    """Compute CAGE perturbation metrics for one attention debug batch."""

    _require_4d("query_states", query_states)
    _require_4d("key_states", key_states)
    _require_4d("key_states_hat", key_states_hat)
    _require_4d("value_states", value_states)
    _require_4d("value_states_hat", value_states_hat)
    _require_2d("o_proj_weight", o_proj_weight)
    _require_same_shape("key_states", key_states, "key_states_hat", key_states_hat)
    _require_same_shape("value_states", value_states, "value_states_hat", value_states_hat)
    _require_matching_attention_shapes(query_states, key_states, value_states)
    _require_positive_int("top_k", top_k)
    for name, tensor in (
        ("query_states", query_states),
        ("key_states", key_states),
        ("key_states_hat", key_states_hat),
        ("value_states", value_states),
        ("value_states_hat", value_states_hat),
        ("o_proj_weight", o_proj_weight),
        ("key_importance", key_importance),
        ("value_importance", value_importance),
    ):
        _require_finite_tensor(name, tensor)

    num_key_value_groups = _resolve_num_key_value_groups(
        num_query_heads=query_states.shape[1],
        num_key_value_heads=key_states.shape[1],
        num_key_value_groups=num_key_value_groups,
    )
    repeated_keys = _repeat_kv(key_states, num_key_value_groups)
    repeated_keys_hat = _repeat_kv(key_states_hat, num_key_value_groups)
    repeated_values = _repeat_kv(value_states, num_key_value_groups)
    repeated_values_hat = _repeat_kv(value_states_hat, num_key_value_groups)

    reference_logits = _attention_logits(query_states, repeated_keys, attention_mask)
    perturbed_logits = _attention_logits(query_states, repeated_keys_hat, attention_mask)
    reference_scores = torch.softmax(reference_logits, dim=-1, dtype=torch.float32)
    perturbed_scores = torch.softmax(perturbed_logits, dim=-1, dtype=torch.float32)
    _require_finite_tensor("reference attention scores", reference_scores)
    _require_finite_tensor("perturbed attention scores", perturbed_scores)

    reference_value_output = torch.matmul(reference_scores.to(repeated_values.dtype), repeated_values)
    perturbed_value_output = torch.matmul(reference_scores.to(repeated_values_hat.dtype), repeated_values_hat)
    output_delta = reference_value_output - perturbed_value_output
    joint_output = torch.matmul(
        perturbed_scores.to(repeated_values_hat.dtype),
        repeated_values_hat,
    )
    joint_delta = reference_value_output - joint_output
    for name, tensor in (
        ("reference attention output", reference_value_output),
        ("perturbed value output", perturbed_value_output),
        ("joint attention output", joint_output),
    ):
        _require_finite_tensor(name, tensor)

    return {
        "relative_k_reconstruction_error": _as_float(_relative_l2_error(key_states, key_states_hat)),
        "attention_logit_mse": _as_float(F.mse_loss(reference_logits, perturbed_logits)),
        "attention_score_kl": _as_float(_attention_score_kl(reference_logits, perturbed_logits)),
        "topk_attention_overlap": _as_float(_topk_attention_overlap(reference_scores, perturbed_scores, top_k)),
        "weighted_key_error": _as_float(_weighted_channel_error(key_states, key_states_hat, key_importance)),
        "relative_v_reconstruction_error": _as_float(_relative_l2_error(value_states, value_states_hat)),
        "attention_output_mse": _as_float(F.mse_loss(reference_value_output, perturbed_value_output)),
        "post_o_proj_mse": _as_float(_post_o_proj_mse(output_delta, o_proj_weight)),
        "weighted_value_error": _as_float(_weighted_channel_error(value_states, value_states_hat, value_importance)),
        "joint_attention_output_mse": _as_float(F.mse_loss(reference_value_output, joint_output)),
        "joint_post_o_proj_mse": _as_float(_post_o_proj_mse(joint_delta, o_proj_weight)),
        "joint_attention_output_relative_error": _as_float(
            _relative_l2_error(reference_value_output, joint_output)
        ),
    }


def collect_cage_perturbation_metrics(
    config: Any,
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    key_states_hat: torch.Tensor,
    value_states: torch.Tensor,
    value_states_hat: torch.Tensor,
    o_proj_weight: torch.Tensor,
    key_importance: torch.Tensor,
    value_importance: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    num_key_value_groups: int | None = None,
    top_k: int = DEFAULT_TOP_K,
    metadata: dict[str, Any] | None = None,
) -> dict[str, float] | None:
    """Optionally compute and dump CAGE perturbation metrics from a config gate."""

    if not bool(getattr(config, "cage_collect_metrics", False)):
        return None

    metrics = compute_cage_perturbation_metrics(
        query_states=query_states,
        key_states=key_states,
        key_states_hat=key_states_hat,
        value_states=value_states,
        value_states_hat=value_states_hat,
        o_proj_weight=o_proj_weight,
        key_importance=key_importance,
        value_importance=value_importance,
        attention_mask=attention_mask,
        num_key_value_groups=num_key_value_groups,
        top_k=top_k,
    )
    dump_dir = getattr(config, "cage_dump_dir", None)
    if dump_dir:
        append_cage_metrics_jsonl(metrics, dump_dir=dump_dir, metadata=metadata)
    return metrics


def append_cage_metrics_jsonl(
    metrics: dict[str, float],
    *,
    dump_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    filename: str = DEFAULT_METRICS_FILENAME,
) -> Path:
    """Append one JSON-safe metric record and return the written path."""

    output_dir = Path(dump_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    record = dict(metadata or {})
    record.update(metrics)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    return output_path


def _attention_logits(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    logits = torch.matmul(
        query_states.float(),
        key_states.float().transpose(2, 3),
    ) / math.sqrt(query_states.shape[-1])
    if attention_mask is not None:
        _require_attention_mask_shape(attention_mask, logits)
        logits = logits + attention_mask.float()
    _require_finite_tensor("attention logits", logits)
    return logits


def _attention_score_kl(reference_logits: torch.Tensor, perturbed_logits: torch.Tensor) -> torch.Tensor:
    reference_log_probs = F.log_softmax(reference_logits, dim=-1, dtype=torch.float32)
    perturbed_log_probs = F.log_softmax(perturbed_logits, dim=-1, dtype=torch.float32)
    reference_probs = reference_log_probs.exp()
    kl = (reference_probs * (reference_log_probs - perturbed_log_probs)).sum(dim=-1).mean()
    _require_finite_tensor("attention score KL", kl)
    return kl.clamp_min(0)


def _topk_attention_overlap(
    reference_scores: torch.Tensor,
    perturbed_scores: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    k = min(top_k, reference_scores.shape[-1])
    _require_positive_int("effective_top_k", k)
    reference_topk = torch.topk(reference_scores, k=k, dim=-1).indices
    perturbed_topk = torch.topk(perturbed_scores, k=k, dim=-1).indices
    matches = reference_topk.unsqueeze(-1).eq(perturbed_topk.unsqueeze(-2)).any(dim=-1)
    return matches.float().mean()


def _post_o_proj_mse(output_delta: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, query_length, head_dim = output_delta.shape
    flattened = output_delta.transpose(1, 2).contiguous().reshape(batch_size, query_length, num_heads * head_dim)
    expected_columns = flattened.shape[-1]
    if o_proj_weight.shape[1] != expected_columns:
        raise ValueError(
            "o_proj_weight must have num_query_heads * head_dim columns, "
            f"got {o_proj_weight.shape[1]} and expected {expected_columns}"
        )
    projected = F.linear(flattened.float(), o_proj_weight.float())
    return projected.square().mean()


def _weighted_channel_error(
    reference: torch.Tensor,
    perturbed: torch.Tensor,
    importance: torch.Tensor,
) -> torch.Tensor:
    prepared_importance = _prepare_importance(importance, reference)
    channel_mse = (reference.double() - perturbed.double()).square().mean(dim=(0, 2))
    return (prepared_importance.double().clamp_min(0) * channel_mse).sum()


def _relative_l2_error(reference: torch.Tensor, perturbed: torch.Tensor) -> torch.Tensor:
    diff_norm = (reference.float() - perturbed.float()).square().sum().sqrt()
    reference_norm = reference.float().square().sum().sqrt().clamp_min(_EPS)
    return diff_norm / reference_norm


def _repeat_kv(states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    if num_key_value_groups == 1:
        return states
    return states[:, :, None, :, :].expand(
        states.shape[0],
        states.shape[1],
        num_key_value_groups,
        states.shape[2],
        states.shape[3],
    ).reshape(states.shape[0], states.shape[1] * num_key_value_groups, states.shape[2], states.shape[3])


def _prepare_importance(importance: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(importance, torch.Tensor):
        raise TypeError(f"importance must be a torch.Tensor, got {type(importance).__name__}")
    if importance.ndim == 3:
        importance = importance.mean(dim=0)
    elif importance.ndim != 2:
        raise ValueError(f"importance must have shape [H_kv, D] or [B, H_kv, D], got {tuple(importance.shape)}")
    expected_shape = (reference.shape[1], reference.shape[3])
    if tuple(importance.shape) != expected_shape:
        raise ValueError(f"importance must have shape {expected_shape}, got {tuple(importance.shape)}")
    return importance


def _resolve_num_key_value_groups(
    *,
    num_query_heads: int,
    num_key_value_heads: int,
    num_key_value_groups: int | None,
) -> int:
    if num_query_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} must be divisible by num_key_value_heads={num_key_value_heads}"
        )
    inferred = num_query_heads // num_key_value_heads
    if num_key_value_groups is None:
        return inferred
    _require_positive_int("num_key_value_groups", num_key_value_groups)
    if num_key_value_groups != inferred:
        raise ValueError(
            f"num_key_value_groups={num_key_value_groups} does not match "
            f"num_query_heads / num_key_value_heads = {inferred}"
        )
    return num_key_value_groups


def _as_float(value: torch.Tensor) -> float:
    value = value.detach().double()
    _require_finite_tensor("final metric", value)
    number = float(value.cpu().item())
    if not math.isfinite(number):
        raise ExperimentPointError("final metric contains non-finite values")
    return number


def _require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ExperimentPointError(f"{name} contains non-finite values")


def _require_4d(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, H, T, D], got {tuple(tensor.shape)}")


def _require_2d(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [hidden_size, num_heads * head_dim], got {tuple(tensor.shape)}")


def _require_same_shape(
    left_name: str,
    left: torch.Tensor,
    right_name: str,
    right: torch.Tensor,
) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape, got {left.shape} and {right.shape}")


def _require_matching_attention_shapes(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    if query_states.shape[0] != key_states.shape[0] or query_states.shape[0] != value_states.shape[0]:
        raise ValueError("query, key, and value batch dimensions must match")
    if key_states.shape[0:3] != value_states.shape[0:3] or key_states.shape[3] != value_states.shape[3]:
        raise ValueError("key_states and value_states must have matching [B, H_kv, T, D] shapes")
    if query_states.shape[3] != key_states.shape[3]:
        raise ValueError("query_states and key_states must have the same head_dim")


def _require_attention_mask_shape(attention_mask: torch.Tensor, logits: torch.Tensor) -> None:
    expected_shape = (logits.shape[0], 1, logits.shape[2], logits.shape[3])
    if tuple(attention_mask.shape) != expected_shape:
        raise ValueError(f"attention_mask must have shape {expected_shape}, got {tuple(attention_mask.shape)}")


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


__all__ = [
    "DEFAULT_METRICS_FILENAME",
    "DEFAULT_TOP_K",
    "append_cage_metrics_jsonl",
    "collect_cage_perturbation_metrics",
    "compute_cage_perturbation_metrics",
]
