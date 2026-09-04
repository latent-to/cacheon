# Target catalog

The target catalog answers a different question from the slot registry:
**which smallest validator-registered semantic contribution is being proposed
and rewarded?** A complete isolated engine is the execution unit; one resolved
target is the reward unit.

Target identity is validator-owned. A bundle may request a target ID and mode,
but cannot define its members, overlap, displacement, features, correctness
contract, or conflicts.

## What a target specification freezes

A target is more than a name mapped to a slot. Its specification binds:

- singleton or atomic kind and exact member slots;
- displaced, required, and explicitly conflicting targets;
- admitted bundle features such as variants, prepare, override, CUDA source,
  or reviewed rebuild operations;
- the slot contract projection: ABI, graph inputs, reference, correctness,
  tolerances, KL threshold, and binding family.

The target-spec digest travels with proposals, stack entries, qualification,
integration, and releases; a target ID alone cannot reopen authority.

## Registered targets

The `target-catalog.v2` policy contains the 11 singleton slot targets plus one
atomic target:

| Target | Kind | Members / effect |
|---|---|---|
| `activation.silu_and_mul` | slot | Same-named slot |
| `collective.all_gather_into_tensor` | slot | Same-named slot |
| `collective.all_reduce` | slot | Same-named slot |
| `collective.ar_residual_rmsnorm` | slot | Same-named slot |
| `collective.reduce_scatter_tensor` | slot | Same-named slot |
| `linear.dense` | slot | Same-named slot |
| `moe.fused_experts` | slot | Experts without ownership of the trailing reduction |
| `moe.fused_experts_reduce` | slot | Experts plus their trailing reduction |
| `moe.fused_routed_experts` | slot | Routing head plus experts plus combine (the fat MoE boundary) |
| `norm.fused_add_rmsnorm` | slot | Residual add plus RMSNorm |
| `norm.rmsnorm` | slot | Same-named slot |
| `collective.dp_attention_exchange.v1` | atomic | Owns both DP-attention exchange members below |

The atomic target owns and displaces both
`collective.all_gather_into_tensor` and
`collective.reduce_scatter_tensor`. Those overlapping identities cannot be
active alongside the atomic target. This prevents one semantic change from
creating duplicate permanent reward titles.

Catalog registration defines identity and admission; it does not by itself
prove that a serving seam is installed or that an arena opens the target. See
[current arena availability](../miner-guide/slots.md#current-glm-53-availability).

## Resolution

A new contribution should declare exactly one competition target explicitly:

```toml
[competition]
target = "moe.fused_experts"
mode = "slot"
```

An atomic contribution declares its registered atomic identity:

```toml
[competition]
target = "collective.dp_attention_exchange.v1"
mode = "atomic"
```

For compatibility with older bundles, the resolver can infer an exact
singleton target or an unambiguous registered atomic member set when the table
is absent. Inference never creates a target, broadens allowed features, or
resolves legacy `system` ownership; explicit declarations remain the clearest
submission contract.

Intake independently observes every feature in the bundle, including CUDA
sources and rebuild operations, and requires the complete set to be admitted by
the registered target. An unknown field or extra capability cannot enlarge the
target. Legacy `mode = "system"` manifests remain parseable for migration but
do not resolve to a current reward title.

### Worked resolution examples

**Ordinary singleton.** A bundle requests `activation.silu_and_mul`, contains
one matching entry, and observes only features admitted by that target.
Resolution produces the singleton target and its frozen target-spec digest.
Adding an unrelated all-reduce row would not create a larger target; it would
make the bundle ineligible for the requested one.

**Atomic DP exchange.** A bundle requests
`collective.dp_attention_exchange.v1` and supplies the exact registered member
set. If it becomes active, catalog displacement removes the two overlapping
singleton exchange titles so one semantic change does not earn three
continuing rewards.

**Legacy inference.** A bundle without `[competition]` may resolve only when
its observed members identify an exact singleton or registered atomic target
unambiguously. A parseable `mode = "system"` row has no current registered
reward identity and therefore cannot resolve by nostalgia.

## Conflict and replacement

Runtime priority is not an ownership rule. Targets that can consume the same
live region are mutually exclusive: a wider target records directional
`displaces` ownership, while incomparable overlap records symmetric
`conflicts_with` exclusion. The MoE reduce-owning target therefore displaces
the plain expert target; they may not coexist under a reduce-first fallback.

Candidate planning applies exclusion in both directions. A wide candidate
removes its contained incumbents. A narrow candidate also removes an active
wide incumbent; the validator-owned base engine supplies computation outside
the narrow target. Unrelated contributions remain byte-identical. A future
megakernel can therefore be challenged without shadowing the challenger.

Catalog validation rejects conflicting active targets before materialization.
The execution gate still requires exact candidate member/rank completion; it is
the runtime backstop, not permission to discover an unreachable candidate after
a paid run.

## Versioning and identity

The complete catalog snapshot is embedded in stack manifests and bound by a
digest. Each target also binds a target-spec digest and a contract digest. A
validator does not reinterpret a historical contribution through whatever
catalog happens to be installed later: evaluation materializes a stack only
under the exact catalog it was sealed with, and a standing reward claim binds
the sealed target-spec digest. Reward projection reads each stack's own retained
catalog, including historical v1 composition and current v2 exclusions. It checks
active membership, displacement, requirements and applicable composition without
loading retired admission or provider registries. A changed installed catalog
does not require re-crowning; changed retained bytes or a substituted target-spec
digest still fail their existing evidence bindings.

Source: [`cacheon/target_catalog.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/target_catalog.py).
