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

!!! warning "Not available on the current MiniMax-M3 arena"
    MiniMax-M3 uses `GemmaRMSNorm`, not the registered
    `RMSNorm.forward_cuda` callsite. This section defines the ABI, but miners
    must not pay for or submit `norm.rmsnorm` to the current mainnet arena.

## Block slots

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
- cosine similarity for the low-bit MiniMax-M3 expert boundaries.

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
