# Slots and contribution targets

The validator publishes two related registries:

- the **slot registry** defines callable seams and their ABIs;
- the **target catalog** defines crownable contribution identities.

Do not treat those words as synonyms. A slot is an execution boundary. A target
is a validator-owned reward unit over one or more slots.

You can print the slot registry from the checkout you are developing against:

```bash
python -m cacheon.cli slots
```

The authoritative sources are
[slots.py](https://github.com/latent-to/cacheon/blob/main/cacheon/slots.py) and
[target_catalog.py](https://github.com/latent-to/cacheon/blob/main/cacheon/target_catalog.py).

## Current slot catalog

There are 11 semantic slots. `entry` below means the callable named by your
manifest; it does not require the Python function itself to be named `entry`.

| Slot | Kind | Required call boundary | What the validator retains |
|---|---|---|---|
| `activation.silu_and_mul` | op | `entry(x, out)` | MLP activation output |
| `collective.all_gather_into_tensor` | collective | `entry(x, out, group)` | rank-ordered gathered tensor |
| `collective.all_reduce` | collective | `entry(x, out, group)` | sum across the supplied process group |
| `collective.ar_residual_rmsnorm` | collective | `entry(x, residual, weight, eps, out_norm, out_residual, group)` | reduced residual and normalized output |
| `collective.reduce_scatter_tensor` | collective | `entry(x, out, group)` | this rank's SUM-reduced shard |
| `linear.dense` | block | `prepare(weight)` plus `entry(x, prepared, out)` | local dense output; row/column-parallel communication stays outside |
| `moe.fused_experts` | block | `prepare(w13, w2)` plus `entry(x, topk_ids, topk_weights, prepared, out)` | local expert result; stock path owns the trailing reduction |
| `moe.fused_experts_reduce` | collective | `prepare(w13, w2)` plus `entry(x, topk_ids, topk_weights, prepared, out, group)` | already reduced expert result |
| `moe.fused_routed_experts` | block | `prepare(w13, w2, topk, routed_scaling)` plus `entry(x, router_logits, correction_bias, prepared, out)` | routed, combined expert result; the implementation owns the routing head |
| `norm.fused_add_rmsnorm` | block | `entry(x, residual, weight, eps, out_norm, out_residual)` | dtype-rounded residual and normalized output |
| `norm.rmsnorm` | op | `entry(x, weight, out, eps)` | pure RMSNorm output |

## Current GLM-5.3 availability

The GLM-5.3 arena seals `moe.fused_routed_experts`, `linear.dense`,
`norm.fused_add_rmsnorm`, `collective.all_reduce`, and atomic
`collective.dp_attention_exchange.v1` (all-gather plus reduce-scatter).

Pure SiLU/RMSNorm remain claimable inside wider fused targets, not separate GLM
lanes. KV-cache, radix, batching, and speculative policy remain engine-owned.

Sealed profiles replace generic shapes with local `6/32/4096`, TP-gathered
`24/128/16384`, prefill all-reduce `(16384, 6144)`, and decode DP-exchange
`(6, 6144)/(32, 6144)`. Dense also binds dimensions, role, and local TP.
Routed MoE uses bounded exact-geometry probes; mixed-cell qualification covers
its full prefill path.

## Current MiniMax-M3 availability

As of 2026-08-30, three registered slot contracts are **unavailable for paid
submission in the current MiniMax-M3 mainnet arena**:

- `norm.rmsnorm`: the deployed model uses `GemmaRMSNorm` at every relevant
  normalization callsite. The registered adapter patches the separate
  `RMSNorm.forward_cuda` boundary, so a candidate for this slot cannot execute.
- `activation.silu_and_mul`: the deployed model computes expert activation
  inside the MoE grouped-GEMM epilogue on 57 of 60 layers, and its three dense
  layers route a swigluoai function the registered adapter does not patch. A
  candidate for this slot loads but is never called.
- `moe.fused_experts_reduce`: sealed closed pending its full-engine
  outer-reduction proof.

Do not pay for those targets; intake parks them without consuming payment. The
remaining M3 targets are `moe.fused_experts`, `collective.all_reduce`, and
`collective.ar_residual_rmsnorm`.

Closure removes only the standalone lane. Activation remains claimable inside
`moe.fused_experts`, normalization inside `collective.ar_residual_rmsnorm`, and
a fused kernel is judged only by its named target contract. See the slot
contract's closure section.

Registration and installation remain different facts: the catalog can register
a slot before the pinned runtime binds a live adapter for it.

Collective slots are distributed contracts. `group` is the process group the
validator supplies, and every listed output is validator-allocated. Test with
the arena's world/TP size, not just one rank.

See [Kernel ABI](kernel-abi.md) for tensor semantics and
[Graph evidence](graph-safety.md) for capture requirements.

## Singleton targets

The current default target catalog registers one singleton target for each of
the 11 slots. Its target ID is the slot ID. A normal proposal therefore names
the slot target explicitly:

```toml
[competition]
target = "moe.fused_experts"
mode = "slot"
```

The catalog, not your manifest, binds that target to its member slot, ABI,
reference, verification profile, serving binding, correctness policy, and
allowed implementation features.

## The registered atomic target

The default catalog also registers:

| Target | Mode | Members | Displaces |
|---|---|---|---|
| `collective.dp_attention_exchange.v1` | `atomic` | `collective.all_gather_into_tensor`, `collective.reduce_scatter_tensor` | both corresponding singleton targets |

Use an atomic target only when the optimization's semantics genuinely require
the coupled boundary and your bundle implements all registered members:

```toml
[competition]
target = "collective.dp_attention_exchange.v1"
mode = "atomic"
```

Adding two unrelated `[[ops]]` rows does not create a target. Nor may a miner
declare membership, displacement, overlap, or a new target ID. An unregistered
combination is not a valid submission until the catalog registers it.

## Selecting a target

Start from the published arena, not from an isolated kernel idea:

1. Confirm that the arena activates the target, model, architecture, dtype,
   topology, and serving phase you intend to optimize.
2. Profile the exact incumbent stack and find a material wall-time boundary.
3. Choose the smallest registered target that contains the required delta.
4. Describe every specialization in an explicit capability domain.
5. If the change needs engine-wide setup, arbitrary SGLang edits, or semantics
   outside a registered target, stop: it is not submittable until the catalog
   registers that surface.

For a first offline implementation, `activation.silu_and_mul` and
`norm.rmsnorm` have the smallest single-process ABIs. They are useful for
learning the contract only: both are unavailable for paid submission in the
current MiniMax-M3 arena as stated above. Advanced collective and deep-MoE
targets require the matching multi-GPU and build environment to test honestly.

### A decision procedure

Walk the desired change from semantics outward:

1. **Name the changed outputs.** Which validator-visible tensor values differ from the
   incumbent implementation while remaining semantically equivalent?
2. **Find who owns those outputs.** Locate the narrowest slot whose ABI owns every changed
   value and no unrelated engine behavior.
3. **Check the live arena binding.** A registered target must also be active for the exact
   model, runtime, architecture, dtype, phase, topology, and graph regime you plan to
   optimize.
4. **Check required features.** If the implementation needs an override point, CUDA
   rebuild, dependency patch, or multiple members, the selected target must explicitly
   allow those observed features.
5. **Estimate materiality.** Profile the incumbent and calculate whether improving that
   boundary could move end-to-end critical-path time above calibrated noise.
6. **State the honest domain.** Use capability predicates for real specialization
   boundaries. Do not use them to hide failing shapes or claim a target that never routes.

The outcome should be one of exactly two shapes: one registered singleton or the exact
complete member set of a registered atomic target. “Closest available slot” is not a
third option, and unregistered work is not submittable.

### Worked choices

| Idea | Choice | Reasoning |
|---|---|---|
| fuse SiLU and multiply for a particular token range | `activation.silu_and_mul` singleton with a constrained variant | both operations and the output are already inside one slot |
| add a residual connection to pure RMSNorm | not `norm.rmsnorm`; use a matching registered collective boundary only if its full semantics apply, otherwise not submittable | the singleton RMSNorm contract explicitly does not own residual addition |
| replace local expert compute and its trailing reduction as one implementation | `moe.fused_experts_reduce` | this slot, unlike `moe.fused_experts`, owns the supplied-group reduction |
| jointly optimize GLM DP-attention exchange | `collective.dp_attention_exchange.v1`, implementing both members | the target owns the measured gather/scatter exchange as one reward unit |
| patch scheduler batching or invent a new attention seam | not submittable | engine control flow lies outside every component callable ABI |

This exercise prevents two common errors. Choosing a boundary that is too narrow makes
the desired optimization impossible without hidden side effects. Choosing one that is
too broad destroys the causal comparison: the validator can no longer attribute the
measured change to one registered reward unit.

## Target resolution fails closed

Production intake resolves the selected delta against trusted observations of
its features. A claim can be rejected when, for example:

- the target is unregistered or `mode` disagrees with the catalog;
- the bundle's implemented member set does not equal the target;
- an atomic claim omits a member;
- the bundle uses a feature the target does not allow;
- observed rebuild features are incomplete;
- multiple variants have overlapping capability domains;
- an `ops.setup` hook appears in a registered target (none currently allow it).

This is why a manifest that merely parses is not necessarily a crownable
proposal. Continue with [Bundle format](bundle-format.md).
