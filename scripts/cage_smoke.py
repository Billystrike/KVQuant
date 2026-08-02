"""Run a short KIVI vs CAGE-KV fake-mode generation smoke check.

This script is intended for a Linux GPU environment with the KIVI CUDA/Triton
extensions and model dependencies installed. The helper functions are kept
import-safe so CPU unit tests can validate argument and accounting behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cage_memory import summarize_cache_bytes


DEFAULT_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
DEFAULT_PROMPT = "The capital of France is"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test original KIVI and CAGE-KV fake generation on a short prompt.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face model id or local model directory. Use a local path on "
            "offline servers."
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text for the smoke run.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        choices=range(8, 17),
        metavar="[8-16]",
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device for generation. The original KIVI path usually requires cuda.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="Model dtype.",
    )
    parser.add_argument("--group-size", type=int, default=32, help="KIVI quantization group size.")
    parser.add_argument("--residual-length", type=int, default=32, help="Full-precision residual length.")
    parser.add_argument(
        "--collect-metrics",
        action="store_true",
        help="Collect optional CAGE perturbation metrics when the fake path supports them.",
    )
    parser.add_argument(
        "--dump-dir",
        default=None,
        help="Directory for optional CAGE metric JSONL output.",
    )
    return parser.parse_args()


def validate_model_reference(model_ref: str) -> str:
    """Validate obvious local-path typos while still allowing HF repo ids."""

    path = Path(model_ref)
    if path.exists():
        return model_ref

    normalized = model_ref.replace("\\", "/")
    looks_like_path = (
        path.is_absolute()
        or normalized.startswith(("./", "../", "~/"))
        or normalized.count("/") > 1
    )
    if looks_like_path:
        raise FileNotFoundError(
            f"Model path {model_ref!r} does not exist. Provide a valid --model "
            "directory or a reachable Hugging Face repo id."
        )
    return model_ref


def configure_generation_mode(
    config: Any,
    mode: str,
    *,
    collect_metrics: bool = False,
    dump_dir: str | None = None,
) -> Any:
    """Apply the minimum KIVI/CAGE config fields used by the smoke run."""

    config.use_flash = True
    config.k_bits = getattr(config, "k_bits", 2)
    config.v_bits = getattr(config, "v_bits", 2)
    config.group_size = getattr(config, "group_size", 32)
    config.residual_length = getattr(config, "residual_length", 32)

    if mode == "kivi":
        config.cage_enable = False
        config.cage_collect_metrics = False
        config.cage_dump_dir = None
    elif mode == "cage_fake":
        config.cage_enable = True
        config.cage_mode = "fake"
        config.cage_k_enable = True
        config.cage_v_enable = True
        config.cage_memory_summary = True
        config.cage_collect_metrics = bool(collect_metrics)
        config.cage_dump_dir = dump_dir
    else:
        raise ValueError(f"Unsupported generation mode {mode!r}; expected 'kivi' or 'cage_fake'")
    return config


def count_new_tokens(generated_ids: torch.Tensor, input_ids: torch.Tensor) -> int:
    if generated_ids.ndim != 2 or input_ids.ndim != 2:
        raise ValueError(
            "generated_ids and input_ids must both be rank-2 tensors shaped [batch, sequence]"
        )
    prompt_length = input_ids.shape[-1]
    output_length = generated_ids.shape[-1]
    if output_length < prompt_length:
        raise ValueError(
            f"Generated output length {output_length} is shorter than the prompt length {prompt_length}."
        )
    return int(output_length - prompt_length)


def load_tokenizer_and_model(
    model_ref: str,
    *,
    mode: str,
    device: str,
    dtype: torch.dtype,
    group_size: int,
    residual_length: int,
    collect_metrics: bool = False,
    dump_dir: str | None = None,
):
    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "Transformers is required for cage_smoke.py. Install project dependencies first."
        ) from exc

    try:
        config = AutoConfig.from_pretrained(model_ref)
        config.group_size = group_size
        config.residual_length = residual_length
        configure_generation_mode(
            config,
            mode=mode,
            collect_metrics=collect_metrics,
            dump_dir=dump_dir,
        )
        model_cls = _model_class_for_config(config)
        tokenizer = AutoTokenizer.from_pretrained(model_ref, use_fast=False)
        model = model_cls.from_pretrained(
            model_ref,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "Failed to load the model/tokenizer. Check that --model is reachable, "
            "the weights are downloaded, and KIVI CUDA/Triton extensions are installed."
        ) from exc

    model.to(device)
    model.eval()
    return tokenizer, model


def run_generation(
    *,
    label: str,
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )

    new_tokens = count_new_tokens(generated.sequences, input_ids)
    if new_tokens != max_new_tokens:
        raise RuntimeError(
            f"{label} generated {new_tokens} new tokens, expected {max_new_tokens}. "
            "Check generation stopping criteria and tokenizer EOS settings."
        )

    return {
        "label": label,
        "new_tokens": new_tokens,
        "text": tokenizer.decode(generated.sequences[0], skip_special_tokens=True),
        "cache_summary": summarize_generated_cache(generated),
    }


def summarize_generated_cache(generated: Any) -> list[dict[str, int | str]]:
    past_key_values = getattr(generated, "past_key_values", None)
    if past_key_values is None:
        return []
    return [summarize_cache_bytes(layer_cache) for layer_cache in past_key_values]


def main() -> int:
    args = parse_args()
    validate_model_reference(args.model)
    _require_device(args.device)
    dtype = _resolve_dtype(args.dtype)

    results = []
    for label, mode in (("kivi", "kivi"), ("cage_fake", "cage_fake")):
        tokenizer, model = load_tokenizer_and_model(
            args.model,
            mode=mode,
            device=args.device,
            dtype=dtype,
            group_size=args.group_size,
            residual_length=args.residual_length,
            collect_metrics=args.collect_metrics and mode == "cage_fake",
            dump_dir=args.dump_dir,
        )
        results.append(
            run_generation(
                label=label,
                tokenizer=tokenizer,
                model=model,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
        )
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(json.dumps({"model": args.model, "results": results}, indent=2, sort_keys=True))
    return 0


def _model_class_for_config(config: Any):
    model_type = getattr(config, "model_type", "")
    try:
        if model_type == "mistral":
            from models.mistral_kivi import MistralForCausalLM_KIVI

            return MistralForCausalLM_KIVI
        if model_type == "llama":
            from models.llama_kivi import LlamaForCausalLM_KIVI

            return LlamaForCausalLM_KIVI
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "Failed to import KIVI model classes. Build/install the CUDA extension "
            "under quant/ and ensure flash-attn/Triton are available."
        ) from exc

    raise ValueError(
        f"Unsupported model_type {model_type!r}. This smoke script currently supports llama and mistral."
    )


def _require_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this smoke script on a GPU server with the "
            "KIVI CUDA/Triton extension installed, or pass --device cpu only for "
            "paths that do not require the CUDA kernels."
        )


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


if __name__ == "__main__":
    raise SystemExit(main())
