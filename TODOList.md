# CAGE-KV TODOList

本文档用于把当前 KIVI 仓库改造成 CAGE-KV 原型。在所有历史 KV payload 都保持 INT2 的前提下，用通道重要性分配量化误差预算、分组粒度和 clipping 策略。

## 0. 方法边界

- [ ] 明确论文主张：Channel importance is used for INT2 rate-distortion allocation, not for mixed-precision channel boosting.
- [ ] 保持历史 Key cache 和 Value cache 的主 payload 全部为 INT2。
- [ ] 不引入少量 INT4 channel 作为主方法，避免和 Kitty 的核心贡献重合。
- [ ] 允许 metadata 增加，例如更多 scale/min 或 scale/zero-point，但需要在内存统计中单独计入。
- [ ] 先做 Python/fake-quant 正确性原型，再接入当前 CUDA pack/GEMV 路径。

## 1. 当前代码入口梳理

- [ ] `models/llama_kivi.py`
  - `LlamaAttention_KIVI.forward` 是主要改动入口。
  - 当前 Key cache 存储为：
    - `key_states_quant_trans`
    - `key_states_full`
    - `key_scale_trans`
    - `key_mn_trans`
  - 当前 Value cache 存储为：
    - `value_states_quant`
    - `value_states_full`
    - `value_scale`
    - `value_mn`
  - 当前 `past_key_value` tuple 固定为 9 项，最后一项是 `kv_seq_len`。

- [ ] `models/mistral_kivi.py`
  - 结构应与 Llama 基本一致。
  - Llama 原型跑通后，同步迁移相同的 CAGE-KV cache 逻辑。

- [ ] `quant/new_pack.py`
  - 当前 `triton_quantize_and_pack_along_last_dim(data, group_size, bit)` 只支持统一 group size。
  - Key 路径输入通常是 `[B, H_kv, D, T]`，沿最后一维 `T` 分组量化并 pack。
  - Value 路径输入通常是 `[B, H_kv, T, D]`，沿最后一维 `D` 分组量化并 pack。

- [ ] `quant/matmul.py`
  - `cuda_bmm_fA_qB_outer` 当前假设所有通道共享同一个 `group_size`。
  - 变量 group size 的高效版本需要新增 bucketed matmul，或先用多次调用当前 kernel 的方式拼起来。

- [ ] `models/utils_quant.py`
  - 这里适合先放 fake quant / dequant 工具，用于验证算法不依赖 CUDA kernel。

## 2. 新增配置项

在 `LlamaConfig` / `MistralConfig` 动态属性中加入以下字段。先不改 Transformers 原始 config 类，沿用当前 KIVI 的做法，直接在加载模型前给 `config` 赋值。

- [ ] 基础开关
  - `config.cage_enable = True`
  - `config.cage_mode = "fake"` 或 `"bucketed_cuda"`
  - `config.cage_k_enable = True`
  - `config.cage_v_enable = True`

- [ ] Key 重要性配置
  - `config.cage_k_importance = "q2_var"`
  - `config.cage_k_group_sizes = [32, 64, 128]`
  - `config.cage_k_clip_percentiles = [0.999, 0.995, 0.99]`
  - `config.cage_k_num_buckets = 3`
  - `config.cage_k_flush_length = 128`

- [ ] Value 重要性配置
  - `config.cage_v_importance = "wo_a2_var"`
  - `config.cage_v_group_sizes = [32, 64, 128]`
  - `config.cage_v_clip_percentiles = [0.999, 0.995, 0.99]`
  - `config.cage_v_num_buckets = 3`

- [ ] 预算和调试配置
  - `config.cage_policy = "quantile"` 先实现，之后再加 `"greedy_budget"`。
  - `config.cage_metadata_budget_ratio = 1.25` 表示 metadata 不超过 KIVI 同配置的 1.25 倍。
  - `config.cage_collect_metrics = False`
  - `config.cage_dump_dir = None`

- [ ] 完成标准
  - 所有字段缺省时，模型行为与原 KIVI 完全一致。
  - 打开 `cage_enable` 时才进入 CAGE-KV 路径。

## 3. Cache 数据结构重构

当前 tuple 可读性较差。先新增轻量 helper，不必立即大规模重构模型类。

- [ ] 新建 `models/cage_cache.py`。
- [ ] 定义 Key cache helper：
  - `key_quant_buckets`: list，每个元素保存一个 group size bucket 的 quantized key。
  - `key_full`: 未量化 residual Key。
  - `key_bucket_indices`: 每个 bucket 对应的 channel index。
  - `key_group_sizes`: 每个 bucket 的 group size。
  - `key_clip_percentiles`: 每个 bucket 的 clipping percentile。
  - `key_scales` / `key_mins`: 每个 bucket 的 scale 和 min。

- [ ] 定义 Value cache helper：
  - `value_quant_buckets`: list，每个元素保存一个 channel bucket 的 quantized value。
  - `value_full`: 最近保留的 full precision Value。
  - `value_bucket_indices`
  - `value_group_sizes`
  - `value_clip_percentiles`
  - `value_scales` / `value_mins`

- [ ] 提供 tuple 兼容函数：
  - `pack_cage_past_key_value(...)`
  - `unpack_cage_past_key_value(...)`
  - `is_cage_past_key_value(past_key_value)`

- [ ] 完成标准
  - 原始 KIVI tuple 不受影响。
  - CAGE tuple 能存取 `kv_seq_len`。
  - `generate(use_cache=True)` 不因为 cache 结构变化报错。

## 4. 通道重要性统计

### 4.1 Key importance

目标公式：

```text
I^K_c = E[q_c^2] * Var(k_c)
```

对于 GQA/MQA，多个 query head 共享一个 kv head，需要把对应 query heads 的 `E[q_c^2]` 汇总到同一个 kv head。

- [ ] 新建 `models/cage_importance.py`。
- [ ] 实现 `compute_key_importance(query_states, key_states, num_key_value_groups)`。
  - 输入 `query_states`: `[B, H_q, T, D]`
  - 输入 `key_states`: `[B, H_kv, T, D]`
  - 输出 `key_importance`: `[B, H_kv, D]` 或 `[H_kv, D]`
  - `q2 = mean(query_states ** 2, dim=token)`
  - 如果 `H_q > H_kv`，按 kv head 分组聚合 query heads。
  - `k_var = var(key_states, dim=token)`
  - `importance = q2_grouped * k_var`

- [ ] 增加数值保护。
  - `nan_to_num`
  - `clamp_min(0)`
  - 如果方差为 0，则 importance 置为 0。

- [ ] 支持 prefill 和 decode。
  - MVP：只在 prefill 时根据 prompt 计算一次 bucket。
  - 后续：decode 时用 EMA 更新统计，但不频繁重排历史 cache。

- [ ] 完成标准
  - shape 正确。
  - MHA 和 GQA 都能计算。
  - 不显著增加 prefill 显存峰值。

### 4.2 Value importance

目标近似：

```text
Delta o_t = A_t Delta V
I^V_c ~= ||W_O[:, c]||_2^2 * E[sum_tau A_{t,tau}^2] * Var(V_c)
```

注意：如果 `E[sum_tau A^2]` 对同一 head 内所有 channel 是常数，它不会改变该 head 内 channel 排序，但它可以影响不同 head 之间的预算分配。

- [ ] 实现 `compute_value_importance(value_states, o_proj_weight, attn_weights=None, num_heads, num_key_value_heads, head_dim)`。
  - 输入 `value_states`: `[B, H_kv, T, D]`
  - 输入 `o_proj_weight`: `[hidden_size, H_q * D]`
  - 输出 `value_importance`: `[B, H_kv, D]` 或 `[H_kv, D]`

- [ ] 计算 `W_O` 范数。
  - `wo = o_proj_weight.view(hidden_size, H_q, D)`
  - `wo_norm = sum(wo ** 2, dim=0)` 得到 `[H_q, D]`
  - 如果 GQA，把多个 query head 的 `wo_norm` 聚合到对应 kv head。

- [ ] 计算 Value 方差。
  - `v_var = var(value_states, dim=token)`

- [ ] 计算 attention 稀疏因子。
  - MVP：使用标量 1，不依赖 `attn_weights`。
  - 进阶：若 `attn_weights` 已经算出，则记录 `a2 = mean(sum(attn_weights ** 2, dim=-1))`。

- [ ] 完成标准
  - 不打开 `output_attentions` 也可以工作。
  - 若无 `attn_weights`，退化到 `wo_norm * Var(V)`。

## 5. Bucket 策略

### 5.1 Quantile policy, 先实现

- [ ] 新建 `assign_channel_buckets(importance, num_buckets, stable=True)`。
- [ ] 输入 importance `[H, D]` 或 `[B, H, D]`。
- [ ] 按每个 layer/head 内的分位数划分：
  - 高重要性 bucket：小 group size，例如 32，保守 clipping，例如 0.999。
  - 中重要性 bucket：中 group size，例如 64，clipping 0.995。
  - 低重要性 bucket：大 group size，例如 128，clipping 0.99。

- [ ] 返回：
  - `bucket_indices`: list of tensors。
  - `group_size_by_bucket`
  - `clip_percentile_by_bucket`
  - `bucket_rank_map`: `[H, D]`，便于 debug。

- [ ] 重要实现细节
  - 每个 bucket 的 channel 数最好补齐到 16 的倍数，因为 INT2 pack factor 是 16。
  - 如果不能整除，MVP 可以 padding channel；CUDA 路径需要保存 `valid_channel_count`。
  - bucket 划分要 deterministic，避免同一输入多次生成结果抖动。

- [ ] 完成标准
  - 所有 channel 都被分到且只分到一个 bucket。
  - high/mid/low bucket 非空；若某个 head_dim 太小，允许合并空 bucket。

### 5.2 Greedy budget policy, 第二阶段

目标：

```text
min sum_c I_c * ||X_c - Q_2bit(X_c; G_c, p_c)||^2
subject to metadata_bytes <= budget
```

- [ ] 为每个 channel 预估候选配置误差：
  - `(G=32, p=0.999)`
  - `(G=64, p=0.995)`
  - `(G=128, p=0.99)`
  - 可选 `(G=256, p=0.985)`，前提是 `cage_k_flush_length >= 256`。

- [ ] 从低成本配置开始，计算升级收益：
  - `gain = weighted_error_old - weighted_error_new`
  - `cost = metadata_bytes_new - metadata_bytes_old`
  - 选择 `gain / cost` 最大的升级，直到达到 metadata budget。

- [ ] 完成标准
  - policy 输出与 quantile policy 使用同一个 bucketed quant 接口。
  - 可以打印每层 metadata bytes 和 weighted error。

## 6. Clipping 设计

当前 KIVI 使用 min/max。CAGE-KV 需要按 bucket 使用不同 clipping percentile。

- [ ] 在 fake quant 中实现 asymmetric percentile clipping。
  - 对每个 group 计算 `q_low` 和 `q_high`。
  - 推荐：
    - `p=0.999`: lower `0.0005`, upper `0.9995`
    - `p=0.995`: lower `0.0025`, upper `0.9975`
    - `p=0.990`: lower `0.0050`, upper `0.9950`
  - 若后续发现 Key 分布主要是单侧 outlier，可以改成只裁 upper 或 lower 的非对称 clipping。

- [ ] scale/min 计算：
  - `mn = quantile(x, q_low)`
  - `mx = quantile(x, q_high)`
  - `scale = (mx - mn).clamp_min(eps) / (2 ** bits - 1)`
  - `code = round(clamp((x - mn) / scale, 0, 2 ** bits - 1))`

- [ ] 完成标准
  - 没有 NaN/Inf。
  - `mx == mn` 时不会除零。
  - clipping 开关关闭时等价于 min/max。

## 7. Fake Quant 原型

先不要改 CUDA。先在 PyTorch 中完成可验证的 CAGE fake-quant。

- [ ] 在 `models/utils_quant.py` 或新文件 `models/cage_quant.py` 中实现：
  - `fake_quant_k_by_channel_buckets(k, bucket_indices, group_sizes, clip_percentiles, bits=2)`
  - `fake_quant_v_by_channel_buckets(v, bucket_indices, group_sizes, clip_percentiles, bits=2)`

- [ ] Key fake quant 形状约定：
  - 输入 `k`: `[B, H_kv, T, D]`
  - 每个 bucket 选择一组 channel `idx`。
  - 对 `k[..., idx]` 转成 `[B, H_kv, D_bucket, T]`。
  - 沿 `T` 按 bucket 的 group size 做 INT2 fake quant。
  - 再转回 `[B, H_kv, T, D_bucket]` 并 scatter 回完整 K。

- [ ] Value fake quant 形状约定：
  - 输入 `v`: `[B, H_kv, T, D]`
  - 每个 bucket 选择一组 channel `idx`。
  - 对 `v[..., idx]` 沿最后一维 `D_bucket` 按 group size 做 INT2 fake quant。
  - scatter 回完整 V。

- [ ] 在 `LlamaAttention_KIVI.forward` 中增加临时 fake path。
  - prefill 阶段：计算 importance -> bucket -> fake quant K/V -> 正常 torch matmul。
  - decode 阶段：先可以只对新增 token 和历史 full fake cache 走慢路径，用于验证准确性。

- [ ] 完成标准
  - `config.cage_mode="fake"` 可以跑通一个短 prompt generation。
  - 关闭 CAGE 后输出和原 KIVI 路径一致。
  - fake CAGE 的 attention logits shape 与原 attention 完全一致。

## 8. Bucketed INT2 pack 原型

### 8.1 Key bucketed pack

- [ ] 在 `quant/new_pack.py` 新增：
  - `triton_quantize_and_pack_kcache_bucketed(k, bucket_indices, group_sizes, clip_percentiles, bit=2)`

- [ ] 输入：
  - `k`: `[B, H_kv, T, D]`

- [ ] 输出：
  - list of bucket records，每个 record 包含：
    - `qcode`: `[B, H_kv, D_bucket_padded, T / 16]`
    - `scale`: `[B, H_kv, D_bucket_padded, T / G_bucket]`
    - `mn`: `[B, H_kv, D_bucket_padded, T / G_bucket]`
    - `indices`: `[D_bucket]`
    - `valid_channels`
    - `group_size`
    - `clip_percentile`

- [ ] MVP 可以复用 `triton_quantize_and_pack_along_last_dim`。
  - 先用 `index_select` 拿出 bucket channel。
  - 转成 `[B, H_kv, D_bucket, T]`。
  - 调用现有 packer。
  - clipping 先在 PyTorch 里做，CUDA clipping 后续优化。

- [ ] 完成标准
  - 当只有一个 bucket 且 `group_size == config.group_size` 时，输出和原 KIVI packer 一致。
  - 每个 bucket 的 `T` 必须能被对应 group size 整除。

### 8.2 Value bucketed pack

- [ ] 新增：
  - `triton_quantize_and_pack_vcache_bucketed(v, bucket_indices, group_sizes, clip_percentiles, bit=2)`

- [ ] 输入：
  - `v`: `[B, H_kv, T, D]`

- [ ] 输出：
  - list of bucket records：
    - `qcode`: `[B, H_kv, T, D_bucket_padded / 16]`
    - `scale`: `[B, H_kv, T, D_bucket_padded / G_bucket]`
    - `mn`: `[B, H_kv, T, D_bucket_padded / G_bucket]`
    - `indices`
    - `valid_channels`
    - `group_size`

- [ ] 注意事项
  - Value 的 group size 是沿 channel 维度。
  - `D_bucket_padded` 需要同时满足 pack factor 16 和 group size 的整除要求。

- [ ] 完成标准
  - 单 bucket 时等价于原 KIVI value pack。
  - 多 bucket dequant 后能 scatter 回 `[B, H_kv, T, D]`。

## 9. Bucketed attention 计算

### 9.1 Key: QK bucketed matmul

MVP 先用多次当前 kernel，后续再融合。

- [ ] 新增 helper：
  - `bucketed_qk_matmul(query_states, key_bucket_records, bits=2)`

- [ ] 对每个 Key bucket：
  - `idx = bucket.indices`
  - `q_sub = query_states[..., idx]`
  - 调用当前 `cuda_bmm_fA_qB_outer(group_size, q_sub, qcode, scale, mn, bits)`
  - 得到该 bucket 对 attention logits 的贡献。
  - 对所有 bucket 输出求和。

- [ ] GQA 注意事项
  - 当前 local `models/llama_kivi.py` 中有 `assert self.num_key_value_groups == 1`。
  - 如果目标模型是 Llama3/Qwen3 这类 GQA 模型，需要先确认当前仓库分支是否支持 GQA。
  - 若保留当前断言，只能先在 MHA 模型上验证 CAGE-KV。

- [ ] 完成标准
  - 单 bucket 输出与原 `cuda_bmm_fA_qB_outer` 数值接近。
  - 多 bucket 输出与 dequant 后 torch matmul 误差可控。

### 9.2 Value: AV bucketed matmul

- [ ] 新增 helper：
  - `bucketed_av_matmul(attn_weights, value_bucket_records, bits=2, head_dim)`

- [ ] 对每个 Value bucket：
  - 调用当前 `cuda_bmm_fA_qB_outer(group_size, attn_weights, qv_bucket, scale, mn, bits)`。
  - 得到 `[B, H, M, D_bucket]`。
  - scatter 到完整 `[B, H, M, D]`。

- [ ] 完成标准
  - 单 bucket 等价于原 KIVI Value 路径。
  - 多 bucket 的输出与 fake dequant matmul 接近。

## 10. Decode / streaming 逻辑

### 10.1 Key cache

当前 KIVI 在 `key_states_full.shape[-2] == residual_length` 时把 residual quantize 后拼到历史 quant cache。

- [ ] CAGE-KV 新增 `cage_k_flush_length`。
  - 它必须能被所有 Key group sizes 整除。
  - 默认先用 128，对应 group sizes `[32, 64, 128]`。
  - 若要使用 group size 256，则 `cage_k_flush_length` 至少为 256。

- [ ] prefill 阶段：
  - 根据当前 prompt 的 Q/K/V 计算 importance。
  - 生成 bucket policy。
  - 将可整除的历史 Key block 量化成 bucketed INT2。
  - 尾部不足 `cage_k_flush_length` 的 Key 放入 full residual。

- [ ] decode 阶段：
  - 新 token 的 Key 追加到 `key_full`。
  - 当 `key_full` 达到 `cage_k_flush_length` 时，按已有 bucket policy 量化并追加到对应 bucket 的历史 qcode/scale/mn。
  - 不在 decode 中频繁重排旧 cache。

- [ ] 完成标准
  - `kv_seq_len` 始终等于 quantized history length + full residual length。
  - 不同 bucket 的历史时间长度一致。

### 10.2 Value cache

当前 KIVI 保留最近 `residual_length` 个 Value 为 full precision，超过后每次量化最旧 token。

- [ ] CAGE-KV 保持这个策略。
- [ ] 对被弹出的最旧 Value token，按 Value channel bucket 做 per-token INT2 quant。
- [ ] 如果 bucket policy 来自 prefill，则 decode 中复用该 policy。
- [ ] 完成标准
  - `value_full` 长度不超过 `residual_length`。
  - quantized value history 和 attention weights 的历史 token 数对齐。

## 11. 内存统计

CAGE-KV 的论文卖点之一是 payload 仍然 INT2，因此必须单独统计 metadata。

- [ ] 新增 `utils/cage_memory.py`。
- [ ] 实现：
  - `estimate_kivi_cache_bytes(...)`
  - `estimate_cage_cache_bytes(...)`
  - `summarize_cache_bytes(past_key_value)`

- [ ] 统计项：
  - INT2 payload bytes。
  - scale bytes。
  - min/zero-point bytes。
  - bucket indices bytes。
  - full precision residual bytes。
  - Python list/object overhead不计入论文主指标，但 debug 可以打印。

- [ ] Key metadata 估算：
  - 每个 bucket: `B * H_kv * D_bucket * ceil(T / G_bucket) * 2 tensors * 2 bytes`

- [ ] Value metadata 估算：
  - 每个 bucket: `B * H_kv * T * ceil(D_bucket / G_bucket) * 2 tensors * 2 bytes`

- [ ] 完成标准
  - 能输出每层每头的 metadata 增长。
  - 能与 KIVI 固定 group size 做公平对比。

## 12. 误差指标

除了任务准确率，需要加入面向论文的中间指标。

- [ ] 新增 `utils/cage_metrics.py`。
- [ ] Key 指标：
  - `relative_k_reconstruction_error`
  - `attention_logit_mse`
  - `attention_score_kl`
  - `topk_attention_overlap`

- [ ] Value 指标：
  - `relative_v_reconstruction_error`
  - `attention_output_mse = ||A V - A V_hat||^2`
  - `post_o_proj_mse = ||(A V - A V_hat) W_O||^2`

- [ ] Weighted error 指标：
  - `sum I^K_c * mse(K_c)`
  - `sum I^V_c * mse(V_c)`

- [ ] 完成标准
  - 能在一个 batch 上同时输出 KIVI 和 CAGE 的误差对比。
  - 这些指标可写入 `.jsonl`，方便后续画图。

## 13. 测试计划

### 13.1 单元测试

- [ ] 新建 `quant/test_cage_quant.py`。
- [ ] 测试 bucket assignment。
  - 所有 channel 覆盖一次。
  - bucket padding 后能恢复原 channel 顺序。

- [ ] 测试 Key fake quant。
  - 输入随机 `[B, H, T, D]`。
  - 单 bucket 时与原 fixed group fake quant 结果一致。
  - 多 bucket 时输出 shape 不变。

- [ ] 测试 Value fake quant。
  - 单 bucket 等价原 fixed group。
  - 多 bucket scatter 正确。

- [ ] 测试 pack/dequant。
  - bucketed pack -> dequant -> scatter。
  - 与 fake quant 结果误差接近。

### 13.2 模型 smoke test

- [ ] 新建 `scripts/cage_smoke.py`。
- [ ] 使用短 prompt，生成 8 到 16 个 token。
- [ ] 测试配置：
  - 原 KIVI: `cage_enable=False`
  - CAGE fake: `cage_enable=True, cage_mode="fake"`
  - CAGE bucketed CUDA: `cage_enable=True, cage_mode="bucketed_cuda"`

- [ ] 完成标准
  - 三种模式都不崩。
  - 输出 token 数正确。
  - cache 长度随 decode 正确增长。

### 13.3 数值一致性

- [ ] 对同一 batch 保存：
  - FP16 attention output。
  - KIVI attention output。
  - CAGE attention output。

- [ ] 对比：
  - CAGE fake vs CAGE bucketed CUDA。
  - 单 bucket CAGE vs 原 KIVI。

- [ ] 完成标准
  - 单 bucket CAGE 和原 KIVI 误差在可接受范围内。
  - CAGE fake 和 bucketed CUDA 的输出误差足够小。

## 14. 实验脚本

- [ ] 修改 `example.py` 或新增 `example_cage.py`。
  - 增加 CAGE 配置项。
  - 打印 cache memory summary。

- [ ] 修改 `pred_long_bench.py`。
  - 增加 CLI 参数：
    - `--cage`
    - `--cage-mode`
    - `--cage-k-group-sizes`
    - `--cage-v-group-sizes`
    - `--cage-policy`
    - `--cage-metadata-budget-ratio`

- [ ] 新增 `scripts/long_test_cage.sh`。
  - 保留原 KIVI 参数。
  - 增加 CAGE 参数。

- [ ] 新增 `scripts/profile_cage_memory.py`。
  - 对同一模型、同一 prompt length 输出每层 cache bytes。
  - 输出 CSV/JSONL。

## 15. 论文实验矩阵

### 15.1 Baselines

- [ ] FP16 KV cache。
- [ ] KIVI K2V2。
- [ ] KIVI K4V2 或 K2V4，用于说明 Key precision 的影响。
- [ ] CAGE-KV fixed group, 只换 clipping。
- [ ] CAGE-KV granularity only。
- [ ] CAGE-KV granularity + clipping。
- [ ] CAGE-KV granularity + clipping + Value importance。
- [ ] Kitty 如果没有官方代码，至少实现一个 approximate baseline：
  - Key top-K channels 使用 INT4。
  - 其他 Key channels INT2。
  - Value 保持 KIVI per-token INT2。
  - 这个 baseline 只用于内部判断，不一定作为正式论文主对比。

### 15.2 Ablations

- [ ] Key importance：
  - magnitude mean, Kitty-like。
  - `Var(K)` only。
  - `E[q^2] * Var(K)`。
  - oracle single-channel attention score sensitivity, 小样本离线分析用。

- [ ] Value importance：
  - none。
  - `Var(V)` only。
  - `||W_O||^2 * Var(V)`。
  - `||W_O||^2 * E[A^2] * Var(V)`。

- [ ] Group sizes：
  - `[64, 128]`
  - `[32, 64, 128]`
  - `[32, 64, 128, 256]`

- [ ] Clipping：
  - no clipping, min/max。
  - shared percentile。
  - importance-aware percentile。

- [ ] Budget：
  - same payload bits。
  - same total cache memory。
  - same metadata budget。

### 15.3 Metrics

- [ ] Task metrics：
  - GSM8K。
  - LongBench。
  - Passkey retrieval。
  - TruthfulQA/CoQA 如果脚本可用。

- [ ] Internal metrics：
  - attention logit MSE。
  - attention distribution KL。
  - output perturbation MSE。
  - cache bytes。
  - decode latency。
  - throughput。

## 16. CUDA 优化路线

先不要一上来改 `quant/csrc`。等 fake path 和 bucketed multi-kernel path 都正确后，再考虑融合 kernel。

- [ ] 阶段 1：bucketed multi-kernel。
  - 每个 bucket 调一次当前 `cuda_bmm_fA_qB_outer`。
  - 优点：实现快，便于验证。
  - 缺点：bucket 数多时 kernel launch overhead 较高。

- [ ] 阶段 2：channel reorder。
  - 将 channel 按 bucket 连续排列。
  - 减少 `index_select` 和 scatter 开销。
  - 需要在 Q/K/V/O projection 维度上谨慎处理，不能改变模型语义。
  - 更安全的做法是在 cache 内部重排，matmul 前重排 query，输出后 scatter。

- [ ] 阶段 3：融合 variable-group kernel。
  - kernel 内读取 `group_size_map[channel]` 或 `bucket_offset`。
  - scale index 从 `t // group_size` 变为按 bucket 计算。
  - 注意 divergent branch 可能降低性能。

- [ ] 阶段 4：metadata 压缩。
  - scale/min 可尝试 fp16 -> fp8/int8 二次量化。
  - bucket indices 可保存为 bitmask 或固定排序表。
  - 这属于后续优化，不影响第一版论文主线。

## 17. 常见坑

- [ ] `residual_length` 和 group size 的关系。
  - Key 的 flush length 必须能被所有 Key group sizes 整除。
  - Value 是 per-token quant，主要要求 channel bucket padded 后能被 group size 和 pack factor 整除。

- [ ] INT2 pack factor 是 16。
  - 沿哪个维度 pack，哪个维度就最好是 16 的倍数。

- [ ] scale 除零。
  - 所有 scale 都要 `clamp_min(1e-6)` 或类似保护。

- [ ] GQA。
  - 当前本地 `llama_kivi.py` 里有 `assert self.num_key_value_groups == 1`。
  - 如果要做 Llama3/Qwen3，必须先修 GQA 支持或使用支持 GQA 的分支。

- [ ] bucket policy 不要在 decode 中频繁改变。
  - 否则历史 cache 需要重排或重编码，代价很高。

- [ ] clipping 的 `torch.quantile` 可能慢。
  - fake path 可以先用。
  - CUDA path 可以先只支持 min/max，或用 topk/approx percentile 近似。

- [ ] metadata 可能吃掉 INT2 payload 的部分收益。
  - 每个实验都要同时报告 payload bytes 和 total cache bytes。

## 18. 推荐实现顺序

- [ ] Step 1：新增配置项和 `cage_enable` 分支，关闭时完全不影响原 KIVI。
- [ ] Step 2：实现 Key/Value importance 计算，并打印 bucket 分布。
- [ ] Step 3：实现 fake quant bucketed K/V，不改 CUDA。
- [ ] Step 4：在 prefill-only 场景验证 CAGE fake quant 的 attention/output 误差。
- [ ] Step 5：接入 decode fake path，跑短文本 generation。
- [ ] Step 6：实现 bucketed pack/dequant，单 bucket 对齐原 KIVI。
- [ ] Step 7：实现 bucketed QK/AV 多 kernel 计算，和 fake path 对齐。
- [ ] Step 8：加 memory estimator 和误差指标脚本。
- [ ] Step 9：跑 GSM8K/LongBench 小规模实验。
- [ ] Step 10：根据瓶颈决定是否写融合 CUDA kernel。

## 19. 第一版最小可交付目标

第一版不要追求最快，先追求论文方法成立。

- [ ] 支持 `config.cage_enable=True`。
- [ ] 支持 Key `E[q^2] * Var(K)` 重要性。
- [ ] 支持 Value `||W_O||^2 * Var(V)` 重要性。
- [ ] 支持 3 个 bucket：high/mid/low。
- [ ] 支持全 INT2 fake quant。
- [ ] 支持短 prompt generation。
- [ ] 输出 memory summary。
- [ ] 输出 attention logit MSE 和 output perturbation MSE。
- [ ] 能和 KIVI K2V2 做同输入对比。

## 20. 后续论文增强点

- [ ] 把 quantile policy 升级为 greedy metadata budget policy。
- [ ] 加入 attention sparsity factor `E[sum A^2]`。
- [ ] 支持 GQA/MQA 大模型。
- [ ] 实现 bucketed CUDA path。
- [ ] 实现 fused variable-group kernel。
- [ ] 做同等总 cache memory 下和 Kitty-style mixed precision 的对比。
- [ ] 画出 channel importance 与 oracle sensitivity 的相关性。
- [ ] 画出不同 bucket 的误差均衡效果。

