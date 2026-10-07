# Slots and contribution targets

An execution slot is either an address in the served model's module tree,
admitted by the `forward_pass` target, or `tree_cache`, the scheduler's prefix
cache, admitted by the `prefix_cache` target. A bundle changes one target, so
kernels and a cache are separate submissions. The arena still owns its model,
workload, evaluation policy and supported nodes.

## Current slot catalog

Use the published arena's supported-node information and exact runtime source
to select a module. These are examples of the address grammar, not a guarantee
that every model contains or admits each address:

| Address | Scope |
|---|---|
| `model.layers.*.mlp` | Every matching layer's MLP |
| `model.layers.3` | Decoder layer 3 |
| `model.layers.*.self_attn.q_b_proj` | Matching attention projection modules |
| `model` | Whole decoder stack, only where the arena can grade it |
| `tree_cache` | The scheduler's runtime cache object (see below) |

`*` matches exactly one segment. The binder refuses addresses that match no
module and parent/child claims that overlap. Binding succeeds only when the
requested method exists in the served model; actual execution and audit
coverage are checked separately.

## Current GLM-5.3 availability

The GLM node arena is `glm53-b300-node-v1`, using GLM-5.3 NVFP4 with
SGLang 0.5.20 on B300, TP4 and attention DP4. `forward_pass` and `prefix_cache`
are both open, and its incumbent is the composed champion implementation. Use the
[GLM development inputs](https://github.com/latent-to/cacheon/tree/main/examples/arena_inputs/glm53)
with that model and topology; Qwen checks do not establish GLM coverage.

## Arena availability

A node bundle selects its arena explicitly:

```toml
[competition]
target = "forward_pass"
mode = "slot"
arena = "<published-arena-id>"

[[ops]]
slot = "model.layers.*.mlp"
source = "kernels/forward.py"
entry = "forward"
```

The bundle uses that node's stock `forward` interface, with its module or
prepared state as the first argument. See [Kernel ABI](kernel-abi.md).

## The prefix cache

`tree_cache` names the scheduler's prefix cache instead of a module. The factory
`entry(cache)` receives the initialized runtime object and returns its type or a
concrete subclass, on every model. The validator installs those methods on the
existing object, preserving its components, host tier, allocator and request pool.
The factory may initialize its own fields on the cache. Declare no `prepare`; a row
declaring dtypes, architectures or metadata is never selected and the run fails.

```toml
[competition]
target = "prefix_cache"
mode = "slot"
arena = "<published-arena-id>"

[[ops]]
slot = "tree_cache"
source = "cache/policy.py"
entry = "build"
```

An identity implementation is:

```python
def build(cache):
    return type(cache)
```

An implementation can return a subclass with its own `evict`, `match_prefix` or
other cache methods. It inherits the runtime cache interface instead of naming a
particular SGLang cache class. The identity implementation establishes binding,
not a speed win.

A cache bundle is judged as the incumbent kernels with the candidate cache against
the incumbent kernels with the incumbent cache, which is stock SGLang's until a
cache is commissioned. The cache runs outside CUDA graphs, so it needs no captured
execution. The same content check runs on both arms: the KV behind every prefix
the cache serves is checked against what the engine computed for it, whichever
slot or host tier the bytes came through. Serving other bytes, keeping pages
across a flush, moving a request's own slots, claiming more tokens than the key, or
serving or protecting a length that is not whole pages stops the engine as the
candidate's failure, as does a `match_prefix` result that is not SGLang's `MatchResult`.
Validation covers full-attention, sliding-window and compressed state. See
[the prefix cache](../architecture/slot-contract.md#the-prefix-cache).

The prefix cache is a target only on arenas that serve with prefix caching. On an
arena whose engine configuration disables radix caching, a cache bundle stops with
`this hybrid runtime disables prefix caching`. The operator's announcement names
each arena's open targets.
At the GLM arena's sealed load, the stock cache already reaches the prefix hit
rate the workload allows, so a cache win comes from lower overhead or better
behavior under memory pressure rather than more hits.

Start each iteration from the current winning cache implementation, retaining its
useful behavior. A later cache version replaces the earlier version; two cache
classes are not automatically combined. Its reward is based on improvement over
the commissioned incumbent, including the existing kernels and cache.

## Selecting a target

1. Start with the exact arena model, image, engine configuration and topology.
2. Choose the smallest supported module enclosing the computation you change.
3. Preserve the rest of the method, its result structure and engine-state writes.
4. Run `verify` for import/interface errors, then `check` with the published
   development inputs. Inspect every claimed address and rank in the table.
5. Measure complete-engine performance against the current incumbent; the
   development check does not award a speed result.

A bundle may name several disjoint modules. That packaging capability must not
be confused with whether separate winning contributions compose in a particular
commission. Overlap and incumbent retention belong to the validator's stack
policy, never Python registration order.

## Target resolution fails closed

Malformed addresses, overlapping node claims,
unknown targets and unavailable arena targets do not become successful stock
runs. A successful interface smoke also does not establish a live node exists;
that needs the loaded model.

