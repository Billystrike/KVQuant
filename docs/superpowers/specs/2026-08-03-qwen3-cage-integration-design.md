# Qwen3 CAGE integration design

## Scope and source boundary

This design implements the Qwen3 CAGE quality-simulation path entirely in the
KVQuant repository. The frozen Kitty repository remains an unmodified external
baseline. No Kitty source is copied into KVQuant and no CAGE code is installed
into the accepted `kitty-qwen3` environment.

The adapter targets the exact Qwen3 implementation from the frozen
Summer-Summer Transformers submodule commit
`37f8b0b53512e6aae0cfd15746c133c101783178`. The audited source hashes are:

- `src/transformers/cache_utils.py`:
  `529e858b9bb0ba59a830713bbd5ab594e29f262bbad1bba860ff7de3248e5a5a`
- `src/transformers/models/qwen3/modeling_qwen3.py`:
  `c12b4c4a34a06e887f3df399b1b09e9966c12d2a19769c258b7379066e585ea4`

These are SHA-256 values of the raw Git blobs, not a Windows checkout with
line-ending conversion.

The accepted Kitty model copy differs from the standard Qwen3 attention path
primarily by replacing decode attention and cache updates with Kitty's packed
runtime. The accuracy-simulation path instead uses the standard Qwen3 model
with a `DynamicCache` subclass. CAGE follows the latter integration pattern.

## Architecture

`models/qwen3_cage.py` provides two components:

1. `Qwen3CageCache`, a `DynamicCache` subclass that owns fake-quantized K/V
   histories and persistent per-layer channel assignments.
2. `install_qwen3_cage_attention`, an instance-level adapter for standard
   `Qwen3Attention` modules. It preserves the loaded module and state-dict
   structure while adding post-RoPE query states and the output-projection
   weight to CAGE cache updates.

The adapter is required because the standard Cache API receives Key and Value
states but not the post-RoPE Query states or `o_proj` weight required by the
frozen `q2_var` Key and `wo_var` Value importance rules. Using a Key-only proxy
would silently change the CAGE method and is therefore rejected.

## Cache lifecycle

For the first update of each layer, CAGE computes assignments from the exact
post-RoPE Q/K and pre-attention V tensors. Prefill attention receives exact K/V,
matching both the existing Llama CAGE path and Kitty's `PostQuant=True`
accuracy-simulation behavior. The persistent cache is fake-quantized after the
prefill tensors have been captured.

During continuation, attention receives the previously fake-quantized history
plus the exact current chunk. Afterward, newly eligible Key blocks and Value
tokens are fake-quantized for the next update. Existing quantized prefixes are
never quantized a second time. Key blocks follow the frozen residual length;
Value retains the latest residual window.

The cache stores dequantized FP16 tensors because this is an accuracy
simulation. Packed cache bytes continue to be computed by the frozen paper
estimator and must not be inferred from CUDA allocation or tensor storage.

## Compatibility and fail-closed behavior

- Only standard Qwen3 attention modules from the frozen Transformers API are
  adapted.
- Qwen3 GQA is handled using 32 Query heads, 8 KV heads, and four Query heads
  per KV head for Qwen3-8B.
- The first acceptance path is batch size one, greedy/cache continuation. This
  matches paired-PPL and perturbation experiments and keeps persistent index
  accounting unambiguous.
- Assisted decoding/cache cropping and skipped decoder layers fail explicitly
  until separately accepted.
- The existing Llama classes, legacy tuple cache, and `cage-kv` dependency pins
  are unchanged.

## Environment and repository flow

The implementation is committed on `codex/qwen3-cage`. On the server, the
branch is executed from `/root/autodl-tmp/KVQuant` under a new
`cage-qwen3` Conda prefix cloned from the accepted `kitty-qwen3` prefix. The
KVQuant package must be exposed without dependency resolution so that its
Transformers 4.43.1 project pin cannot replace the frozen Qwen3 stack.

Official Kitty/Kitty-Pro runs remain in `kitty-qwen3`. Both environments read
the same immutable case manifest and write the same result schema to a neutral
run directory. Environment identity, model revision, tokenizer hashes, input
manifest hash, code commit, and per-case identifiers are checked before paired
statistics are produced.

## Acceptance gate

Before formal Qwen3 results, a tiny randomly initialized Qwen3 CPU test must
verify:

1. standard Qwen3 modules are found and adapted exactly once;
2. prefill logits match the exact DynamicCache path;
3. CAGE assignments have the Qwen3 GQA `[H_kv, D]` shape;
4. Key block flushing and Value residual rolling preserve sequence length;
5. a returned CAGE cache can continue with a later token;
6. all outputs and cache tensors are finite; and
7. unsupported cache operations fail rather than silently changing semantics.

Only after this CPU gate passes will the local Qwen3-8B GPU acceptance script
be enabled. KIVI and the unified formal runner are implemented after CAGE cache
semantics pass this gate, so a baseline cannot mask an integration failure.

## CPU acceptance result

The server CPU gate passed on commit
`6c8e0e793a599b24b5973565efec4ef22d8eac45` in the isolated
`cage-qwen3` environment. All five Qwen3 tests passed; prefill logits were
bit-identical to `DynamicCache`; GQA assignments, Key flushing, Value residual
rolling, and cache continuation matched their expected shapes and lengths.
The complete acceptance log SHA-256 is
`9de73e66d4ea9b8cf1df02dbf7337bf88a1cd4e822a8b64c46ef0e63c5d53bd2`.

## GPU acceptance result

The full Qwen3-8B GPU gate passed on commit
`475f6544528bb40b4090653cee20056dca8bef2d`. All 36 standard Qwen3 attention
modules were adapted; FP16 and CAGE prefill logits were bit-identical; the
three GQA bucket shapes were `[8, 42]`, `[8, 43]`, and `[8, 43]`; eight-token
generation and one-token cache continuation were finite and preserved the
expected lengths. The acceptance JSON SHA-256 is
`bbe995e3c2490dd96d12f8d17e703c7b83fd147e323fa0bf19304fa488a3b11d`
and the complete log SHA-256 is
`f27510a218111104c6f6854fd2dcb131bb841879329bca7010e11c3c220b4999`.
CUDA peaks remain acceptance diagnostics for the fake-quant implementation,
not realized packed-memory evidence.

## KIVI CPU acceptance result

The Qwen3 KIVI CPU gate passed on commit
`9d4184243c9feb6f6f0141f5abc137757dda9cac`. Nine quantization and Cache tests
passed. The persistent Key and Value histories changed under quantization while
prefill logits remained exact, continuation preserved the expected lengths,
and the implementation stored no persistent bucket indices. The complete log
SHA-256 is
`282bd9a8ab8ca51c47087c4d857d245f7e519c6c2f55253f3a45c183f7f32cc7`.
