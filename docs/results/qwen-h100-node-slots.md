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
| `model.layers.*`, honest | 2,280 | 0 | 0.844 |
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

## Stale under capture

The per-node check grades eager calls. A candidate could be honest there and, under
CUDA-graph capture, return the answer it computed during warm-up, so every replay
serves a stale tensor. Three such controls ran with graphs on, 16 real prompts of
1,024 tokens and 96 greedy tokens each; a stock engine then teacher-forced the
candidate's tokens.

| Bundle | tok/s | Distinct tokens per reply | Stock NLL of the output | Stock-argmax agreement | Top-20 KL |
|---|---|---|---|---|---|
| `model`, do-nothing | 1,744 | 59.0 | 0.095 | 0.958 | 0.008 |
| `model`, stale | 5,973 | 1.9 | 12.66 | 0.000 | 12.60 |
| `model.layers.20`, stale | 1,677 | 4.4 | 15.47 | 0.010 | 18.28 |
| `model.layers.20.mlp`, stale | 1,737 | 6.9 | 9.04 | 0.014 | 13.89 |

One stale block of forty moves every end-to-end quality metric by two orders of
magnitude. Every one of these bundles carried `captured: true` on its completed
receipt: the receipt proves the candidate was in the graph, and end-to-end quality
proves the graph reads fresh inputs.

The changes that followed (audit draw before selection, non-finite stock rows
ungraded) were re-run on the same node:
decoder layer honest 160 windows, 0 failed, worst 0.853; decoder layer wrong 120 of
160 failed; four addresses do-nothing 323, 0 failed; whole stack honest 4, 0 failed,
worst 0.914. The numbers are those of the first run.

## A candidate that rewires its node

`prepare` receives the live module. A hostile control used that to rebind the
rotary embedding's `forward_native` inside every decoder layer, which makes the twin
noisy and the tolerance wide. The adapter records every callable on the node's
modules at binding and compares before each audited reference.

| Bundle | Result |
|---|---|
| `model.layers.*`, hostile | refused: `a method inside node 'model.layers.*' was rebound after binding (MRotaryEmbedding: forward_native)` |
| `model.layers.*`, honest | 160 windows, 0 failed, worst 0.853 |
| four addresses, do-nothing | 323 windows, 0 failed |
| `model`, honest | 4 windows, 0 failed, worst 0.914 |

The first version of the record refused every honest bundle at every width with
`TopK: _forward_method`: SGLang's fused ops leave their dispatch target empty until
the first call, and the engine's own fill read as a rebind. An attribute that was
empty at binding may be filled with one of that module's own recorded methods; the
honest rows above are the re-run with that rule. The measured tolerance is also
capped at 40% per row, above the 33% the noisiest honest width earns and below a
wrong answer.

## FP8 KV cache

The arena serves an FP8 (`fp8_e4m3`) KV cache. The longer run above, repeated with
it:

| Bundle | Graded windows | Failed windows | Worst window |
|---|---|---|---|
| `model.layers.*`, honest | 2,280 | 0 | 0.856 |
| `model.layers.*`, native | 2,280 | 0 | 0.829 |
| `model.layers.*`, wrong | 2,280 | 1,960 | 0.0 |
| `model`, honest | 57 | 0 | 0.894 |

SGLang keeps an FP8 cache in `uint8` storage and holds the real type on the pool.
Graded as stored, a cache value near zero whose sign flips reads as a jump of 128
between two bytes that hold almost the same number, and nine in ten of the row
windows behind the first decoder-layer run (39,200 of 43,760) were bytes. The
adapter now grades those
rows as the FP8 numbers they hold. The decoder-layer rows above did not move; the
`model` row was 0.883 when graded as bytes.

The first FP8 runs put the native twin's worst window at exactly 0.75, the bar, and
the result repeated bit for bit. The cause was the control workload, not the cache:
every SWE-agent trajectory opens with the same 1,192 to 2,758-token system prompt,
and the controls sliced the first 300 tokens, so a batch of eight "different"
requests was one request eight times. Their rows were identical to four decimals,
one noisy single-token call counted eight times, and a 256-row window held 32
independent draws instead of 256. The rows above slice past the shared prefix.
Earlier tables on this page used the shared slices; their verdicts stand and their
worst-window figures are, if anything, pessimistic.

## At the arena's width

The runs above hold eight requests of a few hundred tokens. The arena's cells hold
48 requests of 8,192 tokens and 6 of 65,536, and the audit role runs every charged
batch at that width. The controls were repeated there, eager, FP8 cache, three
generated tokens per request.

The first audited 48-request decode step ran an 80 GiB H100 out of memory. The model
keeps 63 MB of FP32 recurrent state per request, stacked over its 30 linear-attention
layers in one tensor, so one copy of a batch's state rows is 2.81 GiB. The adapter
held four (before, stock, twin, candidate) with 5 GiB spare. It now holds one on the
device, stock's answer. The copy the call is put back from waits in pinned host
memory, and the twin's and the candidate's rows are compared 64M elements at a time.
Peak device memory with the audit running is 80,250 MiB.

How the host copy is laid out decides what an audited call costs. One host tensor
sliced along the request dimension makes every piece a strided copy that the CPU
gathers: 2.3 s to copy the state out and 0.7 s for each of three put-backs, 4.4 s a
call, with the scheduler at 98% CPU and the GPU idle. One pinned tensor per piece
costs 0.06 s out and 0.05 s back, 0.22 s a call.

| Bundle | Graded windows | Failed windows | Worst window |
|---|---|---|---|
| `model`, do-nothing | 104 | 0 | 1.0 |
| `model`, wrong | 104 | 100 | 0.0 |
| `model`, honest | 104 | 21 | 0.554 |
| `model.layers.*`, honest | 3,880 | 0 | 0.856 |
| `model.layers.*`, wrong | 3,880 | 3,840 | 0.0 |

With all forty layers audited on every call, the 48-request cell took 159 s and the
6-request 64k cell 110 s; the first audited 48-request decode step took 18 s.

The do-nothing row is the put-back check: stock, the twin and the candidate each ran
every call, and a candidate that is stock matched stock on every row of every window
at both context lengths.

The honest whole-model control fails, and the failure is the measurement, not the
candidate. All 21 failed windows are 8,192-token prefill chunks of the 64k requests,
at the final hidden state and the deepest layers' cache rows. There the twin itself,
which is stock with its fused norms and activations on their native paths, sits at a
median row error of 20 to 25% and a 90th percentile of 72 to 87% against stock; in a
request's first 8k chunk it sits at 13 to 15% and 41 to 63%. The candidate's errors
follow the same distribution. Forty layers of BF16 rounding compound over a 64k
context into differences as large as a wrong answer's, and the 40% cap cannot rise to
admit them because a result scaled by 1.5 sits at 50%. A per-layer claim compares one
layer's rounding and is graded cleanly at the same width.

## Stock throughput at the arena's width

Output tokens per second, stock, CUDA graphs, the H100-tuned fused-MoE table, static
memory fraction 0.93:

| KV cache | Linear-attention decode backend | 8k in, 1k out, 48 requests | 64k in, 4k out, 6 requests |
|---|---|---|---|
| FP8 | triton (default) | 1,903 | 516 |
| FP8 | flashinfer | 1,554 | 470 |
| BF16 | triton | 1,337 | 360 |

The `cutedsl` decode backend measured 2,150 and 524 and is excluded: what it generates
is not the model's output. Its first tokens are unrelated text and it answers none of
eight questions about a number planted earlier in the prompt; the default backend
answers eight of eight at 8k and six of six at 64k. The arena serves the first row.

## Limits

- A `model` claim cannot be graded by the row check at 64k context: an honest
  candidate fails it (see above). The forward pass is covered at that context by
  `model.layers.*` and the leaf nodes.
- SGLang stacks the recurrent state of every linear-attention layer in one tensor, so
  each audited node copies all thirty layers' rows aside, and a per-layer claim pays
  that forty times a step. The audit does not restrict the copy to the node's layer.
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
- Recurrent-state rows are FP32. Cache rows are graded in the cache's own type, so
  an FP8 cache is compared at FP8 resolution against stock's FP8 rows.
- Audited rows inside a prefix every request shares are correlated across requests,
  so a window drawn from them holds fewer independent draws than its row count.
