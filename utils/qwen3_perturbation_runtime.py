from __future__ import annotations

from typing import Any

import torch

from models.cage_importance import compute_key_importance, compute_value_importance
from utils.cage_metrics import compute_cage_perturbation_metrics
from utils.qwen3_perturbation_protocol import LAYER_METRICS, Qwen3PerturbationError


class Qwen3PerturbationRecorder:
    """Capture the frozen local teacher-forced perturbation measurement.

    The lifecycle deliberately matches the Llama pilot: one no-cache FP16
    pass supplies the final-position query, candidate prefill supplies the
    exact prefix and channel importance, and the first candidate decode
    supplies both the current-token K/V and the cache history used by
    attention. Prefix tensors live on CPU between phases to bound GPU memory.
    """

    def __init__(self, *, expected_layers: int = 36, top_k: int = 10) -> None:
        self.expected_layers = expected_layers
        self.top_k = top_k
        self._modules: dict[int, Any] = {}
        self.reset_case()

    def install(self, model: torch.nn.Module) -> int:
        modules: dict[int, Any] = {}
        for module in model.modules():
            if not hasattr(module, "_qwen3_cage_original_forward"):
                continue
            layer_idx = getattr(module, "layer_idx", None)
            if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
                raise Qwen3PerturbationError("instrumented attention has no integer layer_idx")
            if layer_idx in modules:
                raise Qwen3PerturbationError("duplicate Qwen3 attention layer index")
            module._qwen3_perturbation_callback = self._callback
            modules[layer_idx] = module
        if sorted(modules) != list(range(self.expected_layers)):
            raise Qwen3PerturbationError("Qwen3 perturbation recorder did not find every layer")
        self._modules = modules
        return len(modules)

    def uninstall(self) -> int:
        removed = 0
        for module in self._modules.values():
            if hasattr(module, "_qwen3_perturbation_callback"):
                delattr(module, "_qwen3_perturbation_callback")
                removed += 1
        self._modules = {}
        return removed

    def reset_case(self) -> None:
        self.phase = "idle"
        self.reference_queries: dict[int, torch.Tensor] = {}
        self.prefill_keys: dict[int, torch.Tensor] = {}
        self.prefill_values: dict[int, torch.Tensor] = {}
        self.key_importance: dict[int, torch.Tensor] = {}
        self.value_importance: dict[int, torch.Tensor] = {}
        self.layer_records: dict[int, dict[str, Any]] = {}
        self.prompt_length: int | None = None

    def begin_reference(self, *, prompt_length: int) -> None:
        self.reset_case()
        self.prompt_length = _positive_int("prompt_length", prompt_length)
        self.phase = "reference"

    def begin_candidate_prefill(self) -> None:
        self._require_count("reference query", self.reference_queries)
        self.phase = "candidate_prefill"

    def begin_candidate_decode(self) -> None:
        for label, records in (
            ("prefill Key", self.prefill_keys),
            ("prefill Value", self.prefill_values),
            ("Key importance", self.key_importance),
            ("Value importance", self.value_importance),
        ):
            self._require_count(label, records)
        self.phase = "candidate_decode"

    def finish(self) -> list[dict[str, Any]]:
        self._require_count("layer metric", self.layer_records)
        records = [self.layer_records[index] for index in range(self.expected_layers)]
        self.phase = "complete"
        return records

    def _require_count(self, label: str, records: dict[int, Any]) -> None:
        if sorted(records) != list(range(self.expected_layers)):
            raise Qwen3PerturbationError(f"{label} capture is incomplete")

    def _callback(
        self,
        *,
        attention_module: Any,
        query_states: torch.Tensor,
        current_key_states: torch.Tensor,
        current_value_states: torch.Tensor,
        attention_key_states: torch.Tensor,
        attention_value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_value: Any,
    ) -> None:
        del past_key_value
        layer_idx = int(attention_module.layer_idx)
        if self.prompt_length is None:
            raise Qwen3PerturbationError("recorder callback fired before begin_reference")

        if self.phase == "reference":
            expected = self.prompt_length + 1
            self._require_lengths(
                query_states=query_states,
                current_key_states=current_key_states,
                attention_key_states=attention_key_states,
                current_length=expected,
                attention_length=expected,
            )
            self.reference_queries[layer_idx] = query_states[:, :, -1:, :].detach().cpu()
            return

        if self.phase == "candidate_prefill":
            expected = self.prompt_length
            self._require_lengths(
                query_states=query_states,
                current_key_states=current_key_states,
                attention_key_states=attention_key_states,
                current_length=expected,
                attention_length=expected,
            )
            groups = query_states.shape[1] // current_key_states.shape[1]
            key_importance = compute_key_importance(
                query_states,
                current_key_states,
                num_key_value_groups=groups,
            )
            value_importance = compute_value_importance(
                current_value_states,
                attention_module.o_proj.weight,
                num_heads=query_states.shape[1],
                num_key_value_heads=current_value_states.shape[1],
                head_dim=current_value_states.shape[-1],
            )
            self.prefill_keys[layer_idx] = current_key_states.detach().cpu()
            self.prefill_values[layer_idx] = current_value_states.detach().cpu()
            self.key_importance[layer_idx] = key_importance.detach().cpu()
            self.value_importance[layer_idx] = value_importance.detach().cpu()
            return

        if self.phase == "candidate_decode":
            self._require_lengths(
                query_states=query_states,
                current_key_states=current_key_states,
                attention_key_states=attention_key_states,
                current_length=1,
                attention_length=self.prompt_length + 1,
            )
            device = current_key_states.device
            exact_keys = torch.cat(
                (self.prefill_keys[layer_idx].to(device=device), current_key_states.detach()),
                dim=2,
            )
            exact_values = torch.cat(
                (self.prefill_values[layer_idx].to(device=device), current_value_states.detach()),
                dim=2,
            )
            normalized_mask = _single_query_attention_mask(
                attention_mask,
                history_length=self.prompt_length + 1,
            )
            metrics = compute_cage_perturbation_metrics(
                query_states=self.reference_queries[layer_idx].to(device=device),
                key_states=exact_keys,
                key_states_hat=attention_key_states,
                value_states=exact_values,
                value_states_hat=attention_value_states,
                o_proj_weight=attention_module.o_proj.weight,
                key_importance=self.key_importance[layer_idx].to(device=device),
                value_importance=self.value_importance[layer_idx].to(device=device),
                attention_mask=normalized_mask,
                num_key_value_groups=query_states.shape[1] // current_key_states.shape[1],
                top_k=self.top_k,
            )
            if tuple(metrics) != LAYER_METRICS:
                raise Qwen3PerturbationError("runtime metric order differs from the frozen protocol")
            self.layer_records[layer_idx] = {
                "layer_idx": layer_idx,
                "phase": "teacher_forced_decode",
                "query_source": "fp16_reference_final_position",
                "history_length": self.prompt_length + 1,
                "metrics": metrics,
            }
            return

        raise Qwen3PerturbationError(f"recorder callback fired in invalid phase {self.phase!r}")

    @staticmethod
    def _require_lengths(
        *,
        query_states: torch.Tensor,
        current_key_states: torch.Tensor,
        attention_key_states: torch.Tensor,
        current_length: int,
        attention_length: int,
    ) -> None:
        if query_states.shape[2] != current_length or current_key_states.shape[2] != current_length:
            raise Qwen3PerturbationError("current Q/K length differs from the measurement phase")
        if attention_key_states.shape[2] != attention_length:
            raise Qwen3PerturbationError("attention history length differs from the measurement phase")


def _single_query_attention_mask(
    attention_mask: torch.Tensor | None, *, history_length: int
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        if attention_mask.shape != (1, history_length):
            raise Qwen3PerturbationError("2D attention mask length mismatch")
        if bool((attention_mask != 1).any()):
            raise Qwen3PerturbationError("the frozen inputs must not contain padding")
        return None
    if attention_mask.ndim == 4:
        if attention_mask.shape[0] != 1 or attention_mask.shape[-1] != history_length:
            raise Qwen3PerturbationError("4D attention mask shape mismatch")
        return attention_mask[:, :, -1:, :]
    raise Qwen3PerturbationError("unsupported attention mask rank")


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Qwen3PerturbationError(f"{name} must be a positive integer")
    return value


__all__ = ["Qwen3PerturbationRecorder"]
