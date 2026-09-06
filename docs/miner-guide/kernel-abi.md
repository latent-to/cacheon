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

### `attention.sparse_mla`

```python
def sparse_mla(q, kv_cache, indices, seq_lens, out, value_dim, qk_scale, value_scale):
    ...
```

The sparse core consumes already prepared queries `q:(T,H,D)`, paged
latent-plus-positional keys `kv_cache:(P,S,D)`, physical token indices
`indices:(T,K)` and nonnegative per-query `seq_lens:(T)`. Indices and lengths
are int32. Queries/cache share float32, float16, bfloat16 or float8-e4m3fn
storage; `out:(T,H,V)` is always contiguous **bfloat16**, with `V=value_dim`.
A physical index `i` addresses cache row `[i // S, i % S]`.

For each query/head, use only the first `min(seq_lens[t], K)` indices,
discarding `-1` entries. The remaining indices must address the cache.
Trailing entries outside that prefix are ignored, even if they look valid.
Compute `softmax(q @ selected_keys.T * qk_scale)`, multiply by the first
`V` components of the selected cache rows, then apply `value_scale`.
No selected keys means zero output. The two scales are capture-static Python
scalars. Candidates preserve all inputs and fill the supplied output.

The independent FP32 reference dequantizes only selected rows, bounding its
temporary storage by top-k and head geometry. Component verification requires
matched ratio at least 0.99 with absolute/relative tolerance 0.02/0.02 for
every input dtype because output is BF16. This component policy does not
replace or relax the sealed full-model quality gate.

Both TRTLLM DSA prefill and decode in pinned SGLang reach the same
`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` symbol. The adapter
normalizes its one-query-per-row layout into this ABI. Its current domain is
explicit `trtllm-gen`, 32/64-token cache pages, scalar scales, and ordinary
output allocation. Dense MLA, multi-query native rows, LSE/sinks, DCP,
tensor scales and supplied-output variants remain outside that binding.
Compilation/tactic-profiling calls do not count as candidate execution.

This slot owns sparse QK, softmax and latent-value combination. It does not own
index scoring/selection, logical-to-physical translation, RoPE, quantization,
cache writes, projections or absorbed BMMs. The producer supplies per-query
causal selections, including earlier chunks and KV history. Positions affect
the already prepared query/cache tensors. Those tensors, indices and lengths
are all graph-dynamic; changing addresses is not the replay contract.

The [faithful example](https://github.com/latent-to/cacheon/tree/main/examples/miner_sparse_mla_torch)
uses bounded shape-based chunks and tensor-valued masks. It is a correctness
example, not performance evidence. CPU verification does not establish GPU
capture, rank coverage, exact-image fidelity or public arena availability.

### `attention.indexer_scores`

```python
def indexer_scores(q, key_pages, key_scales, weights, starts, ends,
                   page_table, row_to_batch, out):
    ...
```

Queries `q:(T,H,D)` and key pages `(P,S,D)` use FP8-e4m3fn storage. Per-key
scales `(P,S)`, head weights `(T,H)` and supplied output `(T,N)` use FP32.
The four window/mapping tensors are int32. For each query `t` and logical
key `j` in `[starts[t], ends[t])`, resolve physical page
`page_table[row_to_batch[t], j // S]`. Compute the FP32 score as the sum over
heads of `weights[t,h] * max(dot(q[t,h], key) * key_scale, 0)`. Query scales
are already folded into weights by the producer. Set every invalid cell to zero;
the producer masks its causal window before selecting top-k.

Ragged prefill uses a single page containing the concatenated keys and explicit
per-query windows. Paged decode supplies zero-copy key/scale views of the packed
cache, a compact page table and query ownership. Key pages and scales can be
strided. No engine or schedule object crosses the miner ABI. All eight inputs
are dynamic during replay. The independent reference evaluates the declared
math in FP64; verification uses absolute/relative tolerance 0.001/0.001.

The pinned binding covers the DeepGEMM score functions consumed by SGLang.
Top-k, RoPE and cache preparation are separate boundaries. The source
[example](https://github.com/latent-to/cacheon/tree/main/examples/miner_indexer_scores_torch)
uses small Torch tiles as a correctness starting point.

### `attention.indexer_topk`

```python
def indexer_topk(scores, lengths, row_starts, page_table, row_to_batch,
                 page_offsets, out, page_size, top_k):
    ...
```

Scores have shape `(T,N)`; all five metadata tensors are int32. For query `t`,
select up to `top_k` highest scores inside
`[row_starts[t], row_starts[t] + lengths[t])`, intersected with the score row.
Negative infinity is invalid; positive infinity represents priority tokens.
Translate a selected column `c` to logical token
`c - row_starts[t] + page_offsets[t]`, then through
`page_table[row_to_batch[t], logical // page_size] * page_size + logical % page_size`.
Write physical indices into contiguous int32 `out:(T,top_k)`, with unused
trailing entries set to `-1`. Selection order is immaterial. The correctness
contract requires mean valid-row set overlap of at least 0.99; duplicates do
not count as extra matches and padding must be rewritten.

Prefill and decode use this same ABI. Ragged prefill carries explicit query
ownership and chunk/history offsets; decode normally has one query per page-table
row and zero offsets. All six input tensors change across graph replay.
Scores may have padded row strides. The pinned SGLang adapter uses the producer's
compact page table and mapping helper at `DSATopKBackend.topk_transform` for
fused PAGED output. It owns selection and physical translation only; indexer
score GEMMs and sparse attention are separate operations.

The [Torch example](https://github.com/latent-to/cacheon/tree/main/examples/miner_indexer_topk_torch)
is a correctness starting point. A miner may implement this one operation with
CUDA, Triton or installed libraries without implementing the entire indexer.

### `linear.dense`

```python
def prepare(weight):
    # weight: (N, K)
    return build_layout(weight)

def dense(x, prepared, out):
    # x: (M, K); out: (M, N)
    ...
```

This boundary owns one unquantized local GEMM. Bias and any surrounding
row/column-parallel communication remain engine-owned.

### `norm.fused_add_rmsnorm`

```python
def fused_add_rmsnorm(x, residual, weight, eps, out_norm, out_residual):
    # out_residual = (x + residual) rounded to the input dtype
    # out_norm = rmsnorm(out_residual, weight, eps)
    ...
```

Both outputs are validator-allocated and must be filled. The registered
reference preserves the input-dtype residual-add rounding before the fp32
variance reduction.

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
