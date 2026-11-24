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
| `scheduler` | `queue_order`, the listed scheduler decisions | The waiting queue's order and the scheduler decisions the validator lists |

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

`scheduler` owns the scheduler's decisions. `queue_order` asks
for the order of the waiting queue through a view without output lengths, which the
replay fixes. The other roots are the validator-listed objects in
[`ENGINE_OBJECTS`](https://github.com/latent-to/cacheon/blob/main/cacheon/seams.py),
each named by its [dotted path](../architecture/slot-contract.md#engine-objects)
where SGLang defines it: `PrefillAdder` (admission, chunking, preemption to
schedule), `SchedulePolicy.prefill_token_budget` and `shortest_prefill_chunk_limit`,
the `PrefillBudget`, `SWAPrefillBudget` and `ChainPrefillBudget` classes,
`NewTokenRatioTracker` (the decode reserve), `Scheduler.get_num_allocatable_reqs`,
`Scheduler._should_defer_prefill` (prefill/decode interleaving), and under KV pressure
`ScheduleBatch.check_decode_mem`, `retract_decode` and `_get_decode_retraction_order`.
Every other SGLang object is refused at
intake, among them configuration publication, `Scheduler.__init__`, `flush_cache`
and `init_req_max_new_tokens`, `SchedulePolicy.calc_priority`, speculative
verification, the sampler, cache classes and model code.

A scheduler entry has no check per call and is judged at the engine boundary: the
timed speed stage, quality against the teacher, and pristine T. It feeds every model
and cache slot new batches, so a `scheduler` submission is audited on every model and
cache slot of the stack it is assembled on. Until the arena has crowned a
`forward_pass` or `prefix_cache` contribution there is no such slot, and a
`scheduler` submission is parked at intake as `target_unavailable:scheduler`. The
stack holds one `scheduler` entry, so a later `scheduler` win replaces an earlier one.

The three targets have disjoint roots, so they never compete for the same live
region: a transition replaces its own stack entry and leaves the others
byte-identical. When a commission declares the crowned entries as the candidate's
[parent](../architecture/product-model.md#worked-example-a-later-3-delta-on-top-of-an-earlier-7-delta),
the validator assembles each candidate on them, so kernel, cache and scheduler wins
accumulate in one engine while each author keeps their own entry. The pieces are
never timed apart and multiplied: across 43 GLM-5.3-Flash rows a kernel bundle's
gain moved about 2% between the arena's three loads, so each candidate is timed as
the engine it is assembled into.

Catalog registration defines identity and admission; it does not by itself
prove that an arena opens the target. See
[current arena availability](../miner-guide/slots.md#current-availability).

## Resolution

A new contribution should declare its competition target explicitly:

```toml
[competition]
target = "forward_pass"
mode = "slot"
```

Without the table, the resolver infers the narrowest target whose roots hold
every declared address. Inference never creates a target or broadens allowed
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

Source: [`cacheon/target_catalog.py`](https://github.com/latent-to/cacheon/blob/main/cacheon/target_catalog.py).
