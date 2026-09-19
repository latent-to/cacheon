# Qwen H100 node slots

Dated engineering results from 2026-09-19 for the
[node address](../architecture/slot-contract.md#node-addresses) boundary. They
are not a crown, a commissioned arena, or an activation receipt.

## Setup

One H100 80 GB per engine, tensor parallelism one, the Qwen3.6 35B
mixture-of-experts checkpoint in BF16 on SGLang 0.5.19. Every run went through
the production bundle loader, scanner, seam table and receipts inside the arena
image with no network. Control bundles were generated, not hand-tuned:

- **do-nothing** returns `module.forward(*args, **kwargs)`;
- **wrong** returns the stock result scaled by 1.5;
- **honest** runs the stock node with every RMSNorm and SiLU at or inside it on
  SGLang's pure-torch `forward_native` path: the same math with different rounding.

Correctness runs were eager with every call audited. Speed runs had CUDA graphs on
and no audit. The served model has 1,098 named modules.

## Correctness at every width

| Bundle addresses | Nodes bound | Audited calls | do-nothing violations | wrong violations |
|---|---|---|---|---|
| `model.layers.*.mlp.shared_expert.act_fn` | 40 | 1,560 | 0 | 1,560 |
| `model.layers.*.mlp` | 40 | 1,560 | 0 | not run |
| `model.layers.*` (carries KV and recurrent state) | 40 | 1,560 | 0 | not run |
| `model` | 1 | 39 | 0 | 39 |
| `logits_processor` | 1 | 39 | 0 | 39 |
| four addresses: `mlp`, `linear_attn`, `attn`, `model.norm` | 81 | 3,159 | 0 | 3,159 |
| `model` and `logits_processor`: the whole forward pass | 2 | 78 | 0 | 78 |

Every do-nothing run produced output identical to the engine with no bundle.

Two wrong controls left the greedy output unchanged: the whole stack scaled by 1.5
(the final norm removes the scale) and the logits scaled by 1.5 (the argmax does
not move). An end-to-end token comparison cannot replace the per-node check.

## Speed with CUDA graphs

| Bundle | Output tokens per second |
|---|---|
| none | 1,089.0 |
| `model`, do-nothing | 1,087.8 |
| four addresses, do-nothing | 1,089.0 |

The bound `forward` is captured by the prefill and decode graph runners at every
width, and the dispatcher costs nothing measurable once captured.

## What the honest control changed

The first check also graded the arguments after the call. The honest norm failed
1,521 of 1,560 calls under it: the fused stock kernel overwrites its arguments and
returns them, while the native path returns fresh tensors. The 39 passing calls
were layer 0, whose norm takes no residual. Arguments are no longer graded; the
result and the engine-state rows are.

## The bar does not yet scale with width

With arguments ungraded, the honest control against the 0.995 per-call bar:

| Width | Audited calls | Violations | Worst per-call pass fraction |
|---|---|---|---|
| `model.layers.*.input_layernorm` | 1,560 | 0 | 1.0 |
| `model.layers.*.mlp` | 1,560 | 0 | 0.9995 |
| `model.layers.*` | 1,560 | 295 | 0.088 |
| `model` | 39 | 39 | 0.143 |

One elementwise tolerance separates honest from wrong at a single operation and
at a block. At a decoder layer and above, honest BF16 rounding compounds past it:
an honest whole-stack candidate fails every audited call. The node check is
therefore not yet usable for a verdict at those widths.

## Limits

- One model, one GPU per engine, no tensor or data parallelism. A node that
  contains a collective needs rank-identical audit sampling; that is not measured.
- The honest control perturbs norms and activations only. A candidate that also
  replaces matrix multiplies or attention sits further from stock.
- Recurrent-state rows are FP32 and cache rows were BF16 in these runs. FP8 cache
  rows are not measured.
