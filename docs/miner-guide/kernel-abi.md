# Kernel ABI

A node bundle replaces the `forward` method of a named module in the arena's
served model. It accepts the arguments that stock receives and returns the same
result structure. Use the exact published image: module names, methods, weight
layouts and state belong to that pinned runtime.

## Core rules

```python
def prepare(module):
    # Optional; called once per bound module. Return your prepared state.
    return module


def forward(prepared, *args, **kwargs):
    return prepared.forward(*args, **kwargs)
```

The manifest names the entry callable; `forward` is only this example's name.
Without `prepare`, the first argument is the live module. A candidate may call
stock children: while it runs, nested dispatch uses stock. This does not imply
that other miners' child implementations are retained inside a parent claim.

The candidate must preserve the node's result structure, tensor semantics and
engine-state effects. It must leave the module's methods and unrelated serving
behavior alone. The scheduler, batching policy and engine configuration remain
outside the contribution. Ordinary CUDA compilation, library calls, workspaces
and prepared weight layouts are permitted within the declared computation.

There is no extra validator-provided `out` argument in the node ABI. Follow the
stock method's own arguments and return values, including tuples or records.
Do not copy the output-buffer signature from an old catalog example into a node
bundle.

## Choosing a boundary

`model.layers.*.mlp` names one module in each matching decoder layer; `*` matches
one path segment. A bundle can list several disjoint addresses. It cannot claim
a parent and its child together. A region that crosses module boundaries uses
an enclosing module, which must preserve the rest of that module's behavior.
See [Slots and targets](slots.md).

A collective inside a node still needs every participating rank to execute a
compatible communication sequence. A single-GPU smoke test cannot prove that.
Use the arena's actual tensor/data-parallel topology and workload, including
idle and uneven ranks.

Node metadata cannot restrict `num_tokens`, including the legacy
`min_num_tokens` and `max_num_tokens` fields. DP ranks may hold different local
batch sizes, so registry selection on that field could split a collective.
Implement shape and phase specialization inside the entry and preserve every
rank's required communication calls. Node dtype eligibility describes the
floating activation passed to `forward`, even when weights use packed storage;
integer-only calls use the module's parameter dtype.

## Correctness is target-owned

Node audit compares the candidate with stock on the same live call, including
recognized KV and recurrent-state writes. The engine restores the call between
reference passes. The numerical policy is described in the normative
[node contract](../architecture/slot-contract.md#node-addresses); a miner cannot
choose a looser comparator in bundle metadata.

`cacheon verify` checks scanning, imports, entry resolution and signatures. It
does not call `prepare` with a fabricated module or pretend to verify forward
math without the model. `cacheon check` runs the real binder and audit in the
published image. See [Your first bundle](your-first-kernel.md).

## Graph behavior

The graph pass uses the supplied arena engine configuration. Each claimed node
must appear in captured execution on every required rank. The eager audit pass
runs separately. Capture evidence does not establish fresh replay answers or
full-model fidelity: authoritative qualification also applies the pristine
end-to-end quality gate. See [Graph safety](graph-safety.md).

## Retained catalog fixtures

The source tree still contains catalog contracts and old example bundles used
by offline tests. Their named arguments are defined by `SlotSpec`; they do not
define the node ABI or imply current serving availability.

### Atomic sparse-attention family

The retained sparse-attention catalog target has two internal members. A node
bundle instead declares the served module that contains the computation.

### `attention.sparse_mla`

For the retained fixture's exact reference and shapes, consult the
[source catalog](https://github.com/latent-to/cacheon/blob/main/cacheon/slots.py)
and [reference table](../reference/slots-table.md). Arena node submissions use
the stock method interface described above.
