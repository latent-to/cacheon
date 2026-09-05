# Slot catalog

A slot is a validator-owned semantic boundary inside the pinned engine. A
contribution supplies an implementation for that boundary; the validator owns
the call site, inputs, output allocation, reference, and verification policy.

The registered API contains **11 slots**. The registry in
[`cacheon/slots.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/slots.py)
is authoritative; print it with `python -m cacheon.cli slots`.

## Registered slots

| Slot | Kind | Entry contract | Correctness |
|---|---|---|---|
| `activation.silu_and_mul` | op | `entry(x, out)` | allclose |
| `collective.all_gather_into_tensor` | collective | `entry(x, out, group)` | matched ratio ≥ 0.99 |
| `collective.all_reduce` | collective | `entry(x, out, group)` | matched ratio ≥ 0.99 |
| `collective.ar_residual_rmsnorm` | collective | `entry(x, residual, weight, eps, out_norm, out_residual, group)` | matched ratio ≥ 0.99 |
| `collective.reduce_scatter_tensor` | collective | `entry(x, out, group)` | matched ratio ≥ 0.99 |
| `linear.dense` | block | `prepare(weight)` + `entry(x, prepared, out)` | matched ratio ≥ 0.99 |
| `moe.fused_experts` | block | `prepare(w13, w2)` + `entry(x, topk_ids, topk_weights, prepared, out)` | cosine ≥ 0.985 |
| `moe.fused_experts_reduce` | collective | `prepare(w13, w2)` + `entry(x, topk_ids, topk_weights, prepared, out, group)` | cosine ≥ 0.985 |
| `moe.fused_routed_experts` | block | `prepare(w13, w2, topk, routed_scaling)` + `entry(x, router_logits, correction_bias, prepared, out)` | matched ratio ≥ 0.97 |
| `norm.fused_add_rmsnorm` | block | `entry(x, residual, weight, eps, out_norm, out_residual)` | matched ratio ≥ 0.99 |
| `norm.rmsnorm` | op | `entry(x, weight, out, eps)` | allclose |

The callable names in a bundle are selected by its manifest; the signatures
above describe their semantic argument order. Entries fill validator-allocated
outputs and do not return the tensor consumed by the model.

`moe.fused_routed_experts` is the fat MoE boundary: the implementation receives
router logits and owns the routing head (selection on `sigmoid(logits) +
correction_bias`, combine weights from the unbiased sigmoid scores, renormalized
and scaled by `routed_scaling`), the expert computation, and the weighted
combine. It binds the same engine seam as `moe.fused_experts` and exists for
runner backends that route inside the fused kernel, where materialized
`topk_ids`/`topk_weights` never appear at the seam.

## How to read a signature

Take `norm.rmsnorm` as the smallest example:

```text
entry(x, weight, out, eps)
```

The validator creates `x`, `weight`, and the scalar `eps`, allocates `out`, and
invokes the selected entry at the registered SGLang seam. The implementation
must preserve inputs and fill the declared output in place. Verification
computes a trusted high-precision/reference result for the same case and
applies the dtype tolerance. A returned tensor, a different allocation, or a
hidden change to `x` does not replace this contract.

Block slots follow the same ownership rule over a wider semantic region.
Collective slots add the validator-owned process group; candidate code may use
it but may not create a private group or let only some ranks fall back.

!!! warning "Arena availability is narrower than the catalog"
    Each arena seals its own subset of this table; GLM-5.3 registers five
    targets. See
    [Arena availability](../miner-guide/slots.md#arena-availability).

## Kinds

`op`
: A narrow single-device operation. It still executes inside an isolated
  candidate engine during production qualification.

`block`
: A wider compute region that can express algorithmic fusion while preserving
  a bounded tensor contract.

`collective`
: A distributed boundary that receives the validator-owned process group.
  Verification must cover the actual world size and all ranks. Once ranks
  select a collective candidate, a rank-local failure aborts that candidate
  engine; falling back on only one rank would diverge the collective.

## Correctness policies

| Mode | Meaning |
|---|---|
| `allclose` | Every element is inside dtype-specific absolute and relative tolerance |
| `matched_ratio` | A registered fraction of elements must be inside tolerance |
| `cosine` | Cosine similarity and, where configured, relative-norm error are bounded |
| `topk_overlap` | Selection overlap is measured rather than raw score equality |

The standard registered tolerances are `0.02/0.02` for bfloat16,
`0.01/0.01` for float16, and `1e-5/1e-5` for float32. These component checks
are necessary but not sufficient: production quality authority belongs to the
complete qualification profile and pristine reference engine.

Correctness is layered deliberately:

1. the slot verifier checks the registered tensor contract;
2. graph replay checks that dynamic inputs are refreshed and outputs are
   actually rewritten;
3. the current v7 resident B/C schedule (B′ only when needed), v8 two-process
   B/C/B′ schedule, or v9 mixed-cell B/C/B′ schedule measures the exact
   marginal substitution;
4. registered eager audit A checks sampled slot behavior outside charged reads;
5. after candidate teardown, pristine T grades sealed trajectories without
   candidate code present; and
6. fidelity/resource evidence checks that the candidate did not obtain speed
   by bypassing required work.

Passing layer one is necessary to debug a contribution, but only the complete
registered profile can issue a qualification PASS.

## CUDA graph contract

Production scoring keeps CUDA graphs enabled. Applicable graph-safe entries
must use static allocation, avoid host synchronization and data-dependent
Python control flow, preserve inputs, and fill only the declared outputs. The
validator refreshes registered dynamic inputs and poisons outputs between
replays before comparing against fresh trusted references.

An implementation that cannot establish graph safety is not credited for work the
baseline performs. Slots whose serving seam sits inside the captured region (the
MoE expert slots) are graph-verified; an eager warmup alone cannot produce the
qualifying execution receipt.

### Graph failure examples

- A candidate caches the first `seq_lens` value in Python: refreshed replay
  inputs expose the stale output.
- A candidate returns a correct tensor but leaves validator-owned `out`
  untouched: poisoned-output replay exposes that the serving call path would
  consume stale storage.
- A candidate allocates temporary tensors during capture: graph capture or
  replay policy fails rather than benchmarking an eager fallback.
- One collective rank selects the candidate while another selects baseline:
  the candidate engine aborts because mixed-rank continuation would not define
  one semantic result.

## Slot versus target

Slots define execution ABIs. Targets define reward identities. Most targets
map one-to-one to slots, but the target catalog also contains the atomic
`collective.dp_attention_exchange.v1` target and explicit exclusion between
overlapping wide and narrow targets. See [Target catalog](target-catalog.md).

For the invariant waist behind every slot, read
[Target and slot contract](../architecture/slot-contract.md).
