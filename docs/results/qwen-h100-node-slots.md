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

## Binding at every width

These runs used the first check (one elementwise tolerance, every call its own
verdict) and a synthetic token sequence. They show that one adapter binds, serves
and grades every width; the tolerance they used is replaced below.

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

## One fixed tolerance does not scale with width

With arguments ungraded, the honest control against one elementwise tolerance and
a 0.995 per-call bar:

| Width | Audited calls | Violations | Worst per-call pass fraction |
|---|---|---|---|
| `model.layers.*.input_layernorm` | 1,560 | 0 | 1.0 |
| `model.layers.*.mlp` | 1,560 | 0 | 0.9995 |
| `model.layers.*` | 1,560 | 295 | 0.088 |
| `model` | 39 | 39 | 0.143 |

One elementwise tolerance separates honest from wrong at a single operation and
at a block. At a decoder layer and above, honest BF16 rounding compounds past it:
an honest whole-stack candidate fails every audited call. Measured per row against
stock, the honest control sat 0.4% away at a block, 0.4% (90th percentile 1.1%) at
a decoder layer and 4.4% (90th percentile 8.3%, worst 10.7%) at the whole stack. A
mixture-of-experts routing flip moves one token by up to 30% and leaves the rest
of the call untouched.

## The tolerance measured on the same call

The check that replaced it runs three answers per audited call: stock, the honest
twin (stock with every fused op on its native path) and the candidate. A row of
the candidate passes within the larger of 2% and three times the twin's recent
90th-percentile row error on that node; rows pool into 256-row windows per node
and graded tensor; a window passes when 75% of its rows do. These runs used real
agent-session tokens (SWE-agent trajectories) with single requests and batches of
four and eight. **native** is the twin itself submitted as a candidate, so its
score is the noise the check has to absorb.

| Bundle | Nodes bound | Graded windows | Failed windows | Worst window |
|---|---|---|---|---|
| `model.layers.*.mlp`, honest | 40 | 160 | 0 | 1.0 |
| `model.layers.*.mlp`, native | 40 | 160 | 0 | 1.0 |
| `model.layers.*`, honest | 40 | 160 | 0 | 0.853 |
| `model.layers.*`, native | 40 | 160 | 0 | 0.885 |
| `model.layers.*`, wrong | 40 | 160 | 120 | 0.0 |
| `model`, honest | 1 | 4 | 0 | 0.914 |
| `model`, native | 1 | 4 | 0 | 0.944 |
| `model`, wrong | 1 | 4 | 3 | 0.0 |
| `logits_processor`, wrong | 1 | 4 | 1 | 0.0 |
| SiLU activation, wrong | 40 | 120 | 120 | 0.0 |
| four addresses, do-nothing | 81 | 323 | 0 | 1.0 |
| four addresses, wrong | 81 | 323 | 243 | 0.0 |
| whole forward pass, do-nothing | 2 | 8 | 0 | 1.0 |
| whole forward pass, wrong | 2 | 8 | 4 | 0.0 |

A longer run (40 more single-request prefills and a 160-token batch of eight)
gives the wide nodes enough windows to show their tail:

| Bundle | Graded windows | Failed windows | Worst window |
|---|---|---|---|
| `model.layers.*.input_layernorm`, honest | 1,960 | 0 | 1.0 |
| `model.layers.*`, honest | 2,120 | 0 | 0.844 |
| `model.layers.*`, native | 2,280 | 0 | 0.844 |
| `model`, honest | 57 | 0 | 0.914 |
| `model`, native | 57 | 0 | 0.896 |
| `model`, wrong | 57 | 47 | 0.0 |
| whole forward pass, do-nothing | 114 | 0 | 1.0 |

A wrong bundle's passing windows are the state rows and record fields it left to
stock: the ×1.5 control scales the result only. Any window below the bar fails the
bundle, and a wrong answer scores 0.0 where the noisiest honest window scored
0.853. The first version of this check kept one noise scale per address; layer 39
is several times noisier than layer 0, and the honest layer control failed 14 of
252 windows (worst 0.843 against a 0.90 bar). The scale and the row pool are now
kept per bound node.

## Limits

- One model, one GPU per engine, no tensor or data parallelism. A node that
  contains a collective needs rank-identical audit sampling; that is not measured.
- The twin perturbs SGLang's fused ops only. A node with none inside it (an
  attention call, a linear layer) is held to the 2% floor; no honest kernel has been
  measured against that floor beyond the norm, the activation and the MoE block.
- At the whole stack the tolerance is three times an honest 4–11%, so a candidate
  may sit that far from stock per token. That is the resolution honest BF16
  rounding leaves at that width, not a chosen leniency.
- A graded tensor needs 256 rows on audited calls before it produces a verdict.
  The logits processor yields one row per request, so short runs grade it rarely.
- Recurrent-state rows are FP32 and cache rows were BF16 in these runs. FP8 cache
  rows are not measured.
