# Target catalog

The target catalog answers one question: **which validator-registered
contribution is being proposed and rewarded?** A complete isolated engine is the
execution unit; one resolved target is the reward unit.

Target identity is validator-owned. A bundle may request a target ID and mode,
but cannot define its roots, features or correctness contract.

## What a target specification freezes

A target specification binds:

- its node roots, the modules of the served model it may replace;
- admitted bundle features such as variants, prepare, CUDA source, or reviewed
  rebuild operations;
- the contract projection: input and output ABI, reference, correctness,
  tolerances, and binding family.

The target-spec digest travels with proposals, stack entries, qualification,
integration, and releases; a target ID alone cannot reopen authority.

## Registered targets

| Target | Node roots | Replaces |
|---|---|---|
| `forward_pass` | `logits_processor`, `model` | The modules of the served model a bundle names |
| `prefix_cache` | `tree_cache` | The scheduler's prefix cache object |

`forward_pass` is the node target. Its roots are the two top-level modules SGLang
gives every causal LM, so it carries no model or arena identity. A bundle
resolves to it when every `slot` it declares is a
[node address](../architecture/slot-contract.md#node-addresses) at or under a
model root. Its resolved members are the addresses the bundle declared, from one
activation up to every root; one bundle may not name a node and another node
inside it. Two reservations overlap when any of their addresses do. Copy
detection treats every `forward_pass` bundle as one namespace, so a stolen body
relabelled at another width or padded with a second node is still a copy.

`prefix_cache` owns only `tree_cache`, the scheduler's
[prefix cache](../architecture/slot-contract.md#the-prefix-cache). It has one
runtime-object contract across models. It is a separate stack entry, so replacing
a cache contribution preserves the incumbent `forward_pass` contribution. A new
cache version replaces the previous cache version in full; the validator does not
merge their source or combine their policies automatically.

The two targets have disjoint roots, so they never compete for the same live
region: a transition replaces its own stack entry and leaves the other
byte-identical.

Catalog registration defines identity and admission; it does not by itself
prove that an arena opens the target. See
[current arena availability](../miner-guide/slots.md#current-glm-53-availability).

## Resolution

A new contribution should declare its competition target explicitly:

```toml
[competition]
target = "forward_pass"
mode = "slot"
```

Without the table, the resolver infers the target whose roots hold every
declared address. Inference never creates a target or broadens allowed
features. Legacy `mode = "atomic"` and `mode = "system"` manifests remain
parseable so retained bundles stay readable, but do not resolve to a current
reward title.

Intake independently observes every feature in the bundle, including CUDA
sources and rebuild operations, and requires the complete set to be admitted by
the registered target. An unknown field or extra capability cannot enlarge the
target; engine-wide `setup` is refused by every target.

## Versioning and identity

The complete catalog snapshot is embedded in stack manifests and bound by a
digest. Each target also binds a target-spec digest and a contract digest. A
validator does not reinterpret a historical contribution through whatever
catalog happens to be installed later: evaluation materializes a stack only
under the exact catalog it was sealed with, and a standing reward claim binds
the sealed target-spec digest. Reward projection reads each stack's own retained
catalog, including historical v1 composition and v2 displacement and conflict
rows from the retired operation-slot catalog, without loading retired admission
or provider registries. A changed installed catalog does not require
re-crowning; changed retained bytes or a substituted target-spec digest still
fail their existing evidence bindings.

The operation-slot targets (twelve hand-written slot contracts and two atomic
families) were retired once both arenas served node addresses. The two current
targets kept their target-spec and contract digests byte-for-byte.

Source: [`cacheon/target_catalog.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/target_catalog.py).
