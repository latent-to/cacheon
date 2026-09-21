# Slots and contribution targets

For a node arena, an execution slot is an address in the served model's module
tree. The registered `forward_pass` target admits a bundle's declared addresses.
The arena still owns its model, workload, evaluation policy and supported nodes.

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

`*` matches exactly one segment. The binder refuses addresses that match no
module and parent/child claims that overlap. Binding succeeds only when the
requested method exists in the served model; actual execution and audit
coverage are checked separately.

## Current GLM-5.3 availability

The GLM node arena is `glm53-b300-node-v1`, using GLM-5.3 NVFP4 with
SGLang 0.5.18 on B300, TP4 and attention DP4. Its incumbent is the composed
champion implementation. Use the [GLM development inputs](https://github.com/latent-to/cacheon/tree/main/examples/arena_inputs/glm53)
with that model and topology. Arena-specific inputs and supported boundaries
remain part of the published contract; Qwen checks do not establish GLM coverage.

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

Malformed addresses, mixed catalog-slot/node bundles, overlapping node claims,
unknown targets and unavailable arena targets do not become successful stock
runs. A successful interface smoke also does not establish a live node exists;
that needs the loaded model.

## Singleton targets

Old catalog fixtures remain addressable by their singleton target IDs for
reference verification. `cacheon slots` prints this retained catalog, not a
model's module tree. Do not use that list as node-arena availability.

## The registered atomic targets

Retained atomic targets describe old multi-slot contracts. Their fixtures and
historical identities remain tied to those contracts. A node bundle declares
module addresses under `forward_pass`; it does not request a new atomic target
for each fusion.
