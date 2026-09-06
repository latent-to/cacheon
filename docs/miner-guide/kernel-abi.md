# Kernel ABI

The slot ABI is a write-into-output contract. The validator owns the call site,
input bindings, output allocation, reference behavior, and downstream engine.
Your implementation owns only the declared computation.

## Why the ABI is shaped this way

An inference runtime already owns long-lived tensor storage, graph-captured addresses,
streams, process groups, and downstream consumers. Allowing a candidate to return an
arbitrary replacement tensor would let it silently change allocation, aliasing, layout,
device, synchronization, and lifetime along with the math. That would make the measured
delta wider than the registered target and make graph replay unreliable.

Write-into-output keeps the ownership line observable:

```text
validator                         candidate                     validator
---------                         ---------                     ---------
allocate/fill inputs  ------->    read inputs
allocate + poison out ------->    write all logical cells  ---> validate binding
supply scalar/group    ------->    perform slot semantics   ---> compare reference
retain downstream path <------------------------------------    consume same storage
```

Poisoning is important. If the validator fills `out` with NaNs or sentinel data before a
call, a partial write cannot accidentally pass because an old buffer still contains
plausible values. Replaying with fresh logical inputs while keeping captured addresses
stable also detects kernels that bake capture-time data into the graph.

The ABI is therefore both a programming interface and the boundary of the causal claim.
You are free to choose algorithms, tiling, fusion *inside* the slot, and honest
specializations. You do not gain ownership of allocation or adjacent engine semantics.

## Core rules

Every entry implementation must:

- accept the arguments in the slot's exact order;
- write every element of every supplied output;
- honor the supplied output's shape, dtype, device, and stride;
- leave all inputs unchanged;
- remain inside its declared capability domain;
- return `None` (a return value is not used as the model output).

Do not allocate and return a replacement tensor. Do not alias outputs to inputs,
retain live tensors across calls, mutate weights, access the sampler, or infer
that outputs are always contiguous. Verification poisons outputs and checks
input mutation, so partial writes and illegal input reuse fail visibly.

Scalars such as `eps`, `sm_scale`, and `block_size` are inputs, not configuration
requests. Likewise, a supplied `group` is the exact distributed scope for the call; do
not construct a new global group or assume that ambient rank variables describe it.

The authoritative ABI objects are in
[slots.py](https://github.com/latent-to/cacheon/blob/main/cacheon/slots.py), with
output shape/stride checks in
[tensor_spec.py](https://github.com/latent-to/cacheon/blob/main/cacheon/tensor_spec.py).

## Op slots

### `activation.silu_and_mul`

```python
def silu_and_mul(x, out):
    # x: (..., 2*d); out: (..., d)
    d = x.shape[-1] // 2
    out.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])
```

The semantic result is `silu(gate) * up`.

### `norm.rmsnorm`

```python
def rmsnorm(x, weight, out, eps):
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
    out.copy_((y * weight.float()).to(out.dtype))
```

This is pure RMSNorm. The slot does not grant ownership of a residual add.

## Block slots

A candidate may call installed vendor libraries, compile CUDA with inline PTX,
or use Triton to implement its registered computation. Vendor-function adapters
recognize execution inside a candidate and call the underlying library directly;
they do not recursively select another candidate. Exceptions still propagate,
and the invocation scope is reset on both success and failure.

### Atomic sparse-attention family

GLM proposals target `attention.sparse_mla.v1` and implement both internal
members below. Indexer-query preparation, scores and top-k selection are one
member; attention-query preparation and sparse attention are the other. The separate score/top-k
proposal surfaces are retired. Key preparation and cache writes remain
engine-owned; unquantized projections and absorbed BMMs belong to `linear.dense`.

### `attention.sparse_mla`

```python
def sparse_mla(q, q_rope, positions, cos_sin_cache, is_neox,
               kv_cache, indices, seq_lens, out, value_dim, qk_scale, value_scale):
    ...
```

Raw queries are `q:(T,H,V)` and `q_rope:(T,H,R)`. Both arrive as strided views of
engine buffers, never contiguous, and verification supplies the same layouts.
Positions select rows from the cosine/sine table. Rotate split halves when `is_neox=True`, adjacent pairs
otherwise; concatenate the latent query and rotated query, clamp to
`[-448,448]` and convert to FP8-e4m3fn at unit scale. The existing cache
`kv_cache:(P,S,V+R)` is FP8-e4m3fn. Supplied `out:(T,H,V)` is BF16. Generic
inputs support float32, float16, BF16 and FP8; the GLM profile supplies raw BF16
queries and an independent FP8 cache.

For each query/head, consume only the first `min(seq_lens[t], K)` int32 indices.
The producer writes its selections first and pads after them; verification
never places `-1` inside that prefix, and the reference contributes nothing for
one. Physical index `i` addresses cache `[i // S, i % S]`; valid-looking
tail indices outside that prefix are ignored. Compute
`softmax(prepared_q @ selected_keys.T * qk_scale)`, combine the first `V` cache
components and apply `value_scale`. No selected keys means zero output.
`is_neox`, `value_dim` and the scales are capture-static; the seven tensors are
graph-dynamic. The reference rotates in FP64 and dequantizes only selected
cache rows for FP32 attention. BF16 output uses matched ratio ≥ 0.99 and
absolute/relative tolerance 0.02/0.02.

The pinned consumer is `DeepseekSparseAttnBackend._forward_trtllm` for prefill
and decode. Its boundary receives queries before RoPE/FP8 preparation and
runs after engine-owned key-cache updates. Positions and query values must
change correctly across replay. The [component example](https://github.com/latent-to/cacheon/tree/main/examples/miner_sparse_mla_torch)
is a development callable; it alone is not a complete GLM proposal.

### `attention.indexer_select`

```python
def indexer_select(q, key_pages, key_scales, weights, page_table,
                   row_to_batch, lengths, page_offsets, positions, cos_sin_cache,
                   q_scale_gate, num_init_tokens, num_local_tokens, top_k, out):
    ...
```

Queries `(T,H,D)` may be raw BF16/FP16/FP32 or prepared FP8-e4m3fn. Raw queries
and raw head weights use the same dtype: rotate the leading dimensions with
adjacent-pair RoPE, compute per-head scale `max(abs(q), 1e-4) / 448`, quantize
to FP8 and fold `q_scale_gate * scale` into FP32 head weights. For prepared FP8
queries, positions/table are `None`, weights are already FP32 and the gate is
one. Key pages `(P,S,D)` remain FP8 and per-key scales `(P,S)` are FP32. Both
are column views of one interleaved engine page buffer, so neither is
contiguous; verification hands over the same layout.
For each logical position in the row's
`lengths[t]` prefix, add `page_offsets[t]`, resolve its physical page through
`page_table[row_to_batch[t]]`, and sum over heads:
`weights[t,h] * max(dot(q[t,h], key) * key_scale, 0)` after query preparation.
Initial and trailing local token
counts mark priority scores as positive infinity. Select up to `top_k`,
translate directly to physical cache indices, and fill unused int32 output
positions with `-1`. Selection order does not matter; valid-row set overlap
must reach 0.99. Duplicate indices do not earn additional matches. Every emitted
index must lie in `[-1, P*S)`: verification rejects any other value, and the
live seam clamps before the engine dereferences it, so a clamped entry only
counts as a wrong selection. Verification histories span the declared context,
so most rows exceed `top_k` and selection is exercised on every registered shape.

The ten tensor inputs are graph-dynamic when present; the gate, priority counts
and `top_k` are capture-static. Paged decode and ragged prefill both use the producer's real
page table and query mapping. No serving object crosses the ABI. The pinned
consumers are `Indexer._get_topk_paged` and `Indexer._get_topk_ragged`, after
index-cache writes. `Indexer._fused_q_prepare_and_store` defers only query
preparation to this member while preserving engine-owned key writes.
A miner may fuse query preparation, scoring and selection without creating
an intermediate score matrix. The [component example](https://github.com/latent-to/cacheon/tree/main/examples/miner_indexer_select_torch)
is for development; the [complete atomic example](https://github.com/latent-to/cacheon/tree/main/examples/miner_sparse_attention_torch)
implements both required members. CPU checks do not establish CUDA capture,
rank coverage, full-model fidelity or arena availability.

### `linear.dense`

```python
def prepare(weight):
    # Canonical weight: (N, K) or (B, N, K).
    return build_layout(weight)

def dense(x, prepared, out):
    # x: (M, K) or (B, M, K); out: (..., M, N).
    ...
```

The unquantized GEMM family includes ordinary projections, the FP32 router
projection and absorbed attention BMMs. Respect actual input/output strides.
The supplied output uses the input dtype or FP32 as required by its consumer.
Weights remain static after preparation; activations change during graph
replay. The independent reference accumulates in FP64 before output rounding.
Bias, gate scaling and surrounding communication remain engine-owned.
Candidates may call ordinary Torch and installed vendor APIs inside the entry.

### `norm.fused_add_rmsnorm`

```python
def fused_add_rmsnorm(x, residual, weight, eps, out_norm, out_residual):
    # residual=None and out_residual=None select plain RMSNorm.
    ...
```

For a residual call, fill both outputs: round `x + residual` in the input dtype,
then compute the FP32 variance and normalized result. For a plain call, preserve
`x` and fill only `out_norm`; both residual arguments are `None`. GLM's plain
512/2048/6144-wide normalizations belong to this existing family, with no
separate GLM RMSNorm proposal lane.

## Prepare/forward MoE slots

`prepare` runs at load time and may build the representation consumed by the
serving entry. It must not mutate the raw inputs.

Routed-MoE and dense preparation and invocation run in inference mode, including
reuse after graph capture. Candidate-owned prepared workspaces remain writable
across those calls; the prohibition on mutating raw inputs still applies.

```python
def prepare(w13, w2):
    # w13: (E, 2*I, H), gate then up; w2: (E, H, I)
    return build_layout(w13, w2)

def fused_experts(x, topk_ids, topk_weights, prepared, out):
    # x: (M, H); routing arrays: (M, K); out: (M, H)
    ...
```

NVFP4 verification and live dispatch use the same tagged prepare form:

```python
prepare("nvfp4_layer", weights)
```

`weights` is a validator-owned view—not the SGLang layer—with packed `uint8`
E2M1 weights, swizzled E4M3 scales, `g1_alphas`/`g2_alphas`, inverse activation
scales, intermediate size and group size 16. `cacheon_w13_layout` is `gate_up`,
`up_gate_interleaved_64+sf_swizzled_128x4` (CuTe-DSL), or `trtllm_fp4_shuffled`
(TRTLLM permutations for both GEMMs/scales). The live runner supplies activation.
Only the first two layouts support `dequantize_prepare_args`.
Candidates may repack it; the validator dequantizes an independent fp32 oracle.

`weights.moe_runner_config.top_k` is the routing width, including fused shared
experts. Live preparation reads it from the serving layer; verification reads
it from `topk_ids.shape[-1]`. Neither path substitutes a zero placeholder.

`topk_weights` contains validator-supplied raw positive FP32 routing multipliers.
They are not promised to be probabilities: do not assume that a row sums to one,
or that its only value is `1.0` when `K == 1`. SGLang configurations that do not
renormalize routing, or that apply a routed scaling factor, make those distinctions
part of the result the kernel must preserve.

GLM-5.3 appends its static routing configuration:

```python
prepare("nvfp4_layer", weights, topk, routed_scaling)
```

`topk_weights` contains raw positive FP32 multipliers, not promised
probabilities. Preserve non-unit row sums, including when `K == 1`.

`moe.fused_experts` is local; the trusted path owns any later collective.

`moe.fused_routed_experts` owns the routing head as well as expert execution and
the weighted combine:

```python
def prepare(w13, w2, topk, routed_scaling):
    return build_layout(w13, w2, topk, routed_scaling)

def fused_routed_experts(
    x, router_logits, correction_bias, prepared, out
):
    ...
```

Selection uses `topk(sigmoid(router_logits) + correction_bias)`. Combine weights
come from the unbiased sigmoid scores, are renormalized, and are multiplied by
the registered routed scaling factor.

`moe.fused_experts_reduce` owns that trailing reduction and therefore receives a
process group:

```python
def fused_experts_reduce(
    x, topk_ids, topk_weights, prepared, out, group
):
    # Fill out with the sum of local expert results across group.
    ...
```

The validator does not replay a second stock reduce after this slot. That wider
authority is why it is a distributed contract.

The prepare/forward split exists because weight transformation and request-time work have
different lifetimes. Packing fixed expert weights once can be a legitimate optimization;
packing them on every token would distort the serving path. Conversely, `prepare` is not
an engine initializer: it receives only the registered weight inputs and returns the
representation used by this slot. It cannot patch SGLang, allocate unrelated persistent
state, or inspect future requests.

## Collective slots

Collective verification uses separate processes and the actual supplied group.
Do not create an unrelated global process group or assume rank/world size from
ambient environment variables.

### `collective.all_reduce`

```python
def all_reduce(x, out, group):
    tmp = x.clone()
    torch.distributed.all_reduce(tmp, group=group)
    out.copy_(tmp)
```

### `collective.all_gather_into_tensor`

```python
def all_gather_into_tensor(x, out, group):
    # x: (M, H); out: (world*M, H), in rank order
    ...
```

### `collective.reduce_scatter_tensor`

```python
def reduce_scatter_tensor(x, out, group):
    # x: (world*M, H); out: this rank's SUM-reduced (M, H) shard
    ...
```

GLM-5.3 rewards these two callables together through the atomic
`collective.dp_attention_exchange.v1` target. A bundle for that target must
implement both members.

### `collective.ar_residual_rmsnorm`

```python
def ar_residual_rmsnorm(
    x, residual, weight, eps, out_norm, out_residual, group
):
    # out_residual = sum_group(x) + residual
    # out_norm = rmsnorm(out_residual, weight, eps)
    ...
```

Both outputs must be filled. `x` differs by rank; `residual` and `weight` are
replicated inputs.

## Correctness is target-owned

The validator computes trusted references and applies the target contract. The
current catalog uses:

- elementwise tolerance for numerically equivalent op kernels;
- `matched_ratio` for dense, routed MoE, fused norm, and collectives whose
  legitimate reduction order can change rounding; and
- cosine similarity for low-bit expert boundaries.

Tolerance, ratio, overlap, reference, and model binding are not miner-selected
manifest values. Passing local `verify` demonstrates compatibility with its
diagnostic profiles; authoritative qualification also evaluates the candidate
inside the exact engine and against pristine quality evidence.

The comparators reflect the semantic output of each boundary:

- **all-close** asks whether every output cell implements essentially the same numeric
  operation;
- **matched ratio or cosine** permits the bounded rounding/reduction effects expected of
  a low-bit or reordered implementation without allowing the miner to choose its own
  tolerance.

Slot verification and end-to-end quality answer different questions. A per-call error can
fit a slot tolerance yet compound across layers, so qualification still uses candidate-
free pristine T evidence. Conversely, the candidate cannot redefine its local reference
by pointing at the current incumbent, which may itself contain prior proposals.

## Capability and fallback behavior

Before dispatch, the validator describes the live call and matches it against
the effective variant domain. Outside the declared domain, the trusted
incumbent path is used. That fallback is a safety property, but it cannot create
a win: a candidate that never runs, or runs only on immaterial calls, has no
positive marginal contribution.

Declare narrow domains honestly, then make sure diagnostic verification
actually exercises them. An exact model, phase, topology, dtype, or shape
predicate whose field is absent from the binding fails closed.

## Graph behavior

A bundle declares nothing about graph safety; the validator decides which slots
serve from the captured region. A crownable path must
have validator-produced graph observations for every applicable selected
variant and shape. CUDA host synchronization, data-dependent Python control
flow, allocations tied to replay values, pointer retention, or incomplete
replay writes will fail that stage. Continue with [Graph evidence](graph-safety.md).
