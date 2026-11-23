# Inkling 2 Mini on H200

These are measurements from 2026-11-23 UTC on one eight-GPU H200 host serving
`thinkingmachines/Inkling-2-Mini` in SGLang's Inkling 2 image. The host has 141 GB per
GPU, 192 vCPUs and 3 TB of host memory; its CPU frequency governor read `powersave`, and
the tenant could not change it. The arena's inputs and engine settings are in
[`examples/arena_inputs/inkling2mini`](https://github.com/latent-to/cacheon/blob/main/examples/arena_inputs/inkling2mini/README.md).
The runs chose the lane and its sealed options. They are not a stock reference, a crown or
a serving claim.

## Lane width

SPEED-Bench `throughput_1k` ran at temperature 1 with 1,024 output tokens. Each cell gives
per-user tokens/s (1,000 divided by the median time per output token), then output
tokens/s per GPU, at each client concurrency C:

| SGLang lane | C=1 | C=4 | C=16 | C=32 | C=64 |
|---|---|---|---|---|---|
| TP1, DSpark | 350 / 314 | 162 / 590 | 97 / 1,331 | 64 / 1,680 | 48 / 2,457 |
| TP2, DSpark | 398 / 197 | 296 / 502 | 171 / 1,128 | 113 / 1,537 | 78 / 1,958 |
| TP4, DSpark | 508 / 115 | 336 / 305 | 244 / 782 | 157 / 1,034 | 114 / 1,437 |
| TP1, no speculation | 178 / 174 | 127 / 495 | 57 / 878 | 48 / 1,373 | 37 / 2,023 |

DSpark doubles single-stream speed. At about 113 tokens/s per user, a TP2 lane serves 7%
more per GPU than TP4. TP1 has already dropped to 97 tokens/s per user at 16 streams. TP2
also leaves four lanes on the node.

## Agent replay

AIPerf replayed the AgentX 256K traces with their recorded pauses for 900 s on four TP2
lanes at a time. Two lanes ran SGLang's default pools. The other two ran the sealed options:
93% of device memory, the sliding-window pool at 5% of the full-attention pool, and a
host tier at twice the device pool. That gives 4.56M full-attention tokens per lane
against 3.83M, plus 149 GB of full-attention host cache per rank.

Per-user speed is output tokens over time to first token plus decode time.

| Lane | Sessions per GPU | Calls | Per-user p50 / p10 | TTFT p50 / p95 | Prompt mean | Output mean | Prefix hit | Output tokens/s per GPU |
|---|---:|---:|---|---|---:|---:|---:|---:|
| default | 20 | 572 | 34 / 19 | 0.34 / 4.9 s | 99.6K | 913 | 88.3% | 282 |
| sealed | 20 | 676 | 39 / 19 | 0.31 / 4.5 s | 91.8K | 787 | 90.9% | 289 |
| default | 30 | 636 | 19 / 10 | 0.46 / 9.2 s | 77.5K | 723 | 82.0% | 247 |
| sealed | 30 | 757 | 29 / 16 | 0.36 / 4.0 s | 86.8K | 684 | 90.3% | 279 |

On agent traffic a lane runs far below its SPEED-Bench curve.

## Slot checks

`cacheon check` ran each control in the qualification image on one two-GPU lane, with the
kit's engine options and its 16 development requests. The eager audit role ran first, then
the graphs-on role.

| Control | Target | Audit role | Graphs-on role | Wall time |
|---|---|---|---|---:|
| MoE identity (`miner_node_identity`) | `model.layers.*.mlp` | PASS: 1,752 calls graded, bitwise equal to stock | node called inside CUDA-graph capture on both ranks | 21 min |
| Cache identity, a subclass that calls stock | `tree_cache` | PASS: 32 requests graded by content | served every request on both ranks | 8 min |
| Wrong MoE (`miner_node_wrong`) | `model.layers.*.mlp` | FAIL at the first graded call on each rank, score 0 against the bar | not run | 3 min |
| Queue order (`miner_prefill_scheduler`) | `queue_order` | not audited | ordered the waiting queue on both ranks | 4 min |
| Pass-through of `_should_defer_prefill`, `check_decode_mem` and `retract_decode` | `engine` | not audited | all three bound; the first two called on both ranks | 4 min |

Retraction never ran, because these requests never filled the KV pool.

The audit role grades every call of an audited node. The identity control spent 17 of its
21 minutes there.
