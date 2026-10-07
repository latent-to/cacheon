# Prefix-cache slot

Functional GPU results from 2026-09-29 IST for the
[prefix-cache contract](../architecture/slot-contract.md#the-prefix-cache),
using source revisions `b09bec24` and `4ed4777`. These establish cache execution and validation;
they are not a speed qualification, crown or deployment.

## GLM runtime and controls

The original GLM checks used `b09bec24`.

Two lanes of four B300 GPUs each served GLM-5.3 NVFP4 on SGLang 0.5.20, with
TP4, DP attention, FP8 KV, 64-token pages, three MTP draft steps and four draft
tokens. Both used the same existing kernel contribution. The candidate lane
also supplied `entry(cache)` returning `type(cache)` at `tree_cache`.

The graph checks retained CUDA graphs and the native 120 GB host cache per rank.
Each DP rank had capacity for 1,274,688 device tokens and 2,637,184 host tokens.
The development engines capped running requests at 24 per lane. Request batches
had a 120-second deadline. The serving lifecycle and terminal
execution check were the production `cacheon.miner_check._engine` path.

## Reuse, eviction and restoration

The graph lifecycle completed on both lanes with no failed requests or cache
content refusals. The candidate cache produced completion receipts on all four
ranks, alongside the retained kernel contribution.

| Check | Requests per lane | Observed result |
|---|---:|---|
| Initial prefix and reuse | 24 requests, twice | 8,128 device-hit tokens per repeated request |
| First pressure bank and reuse | Four 262,016-token prompts, twice | 261,952 device-hit tokens per repeated request |
| Second pressure bank and reuse | Four new salted prompts, twice | First bank evicted from device; second bank reused |
| Revisit first bank | Four requests | 261,952 host-hit tokens and zero device-hit tokens per request |
| Reset, replay, reuse and two branch batches | 24 requests, four times | Completed after the guard drained and the cache reset |

The pressure banks were routed to DP rank zero in each TP4 lane. Their distinct
salts prevented cross-bank hits. Each lane restored 1,047,808 tokens from host
memory during the revisit, taking 7.43 and 7.48 seconds respectively, with no
request retractions. This is explicit host-hit metadata, not an inference from
aggregate cache hits. All four TP ranks participated in model execution; host
pressure was tested on one DP rank per lane.

The pressure inputs exercise cache lifecycle behavior rather than representing
the scored workload. Acceptance length was fixed at three with real draft tokens
for these graph checks. Individual case timings do not establish a speedup or
the validation overhead.

## Audit and corrupt-cache control

A separate eager audit used real MTP acceptance, with simulation disabled. The
honest candidate retained the kernel contribution and served three batches of
24 requests: a 512-token prefix, reuse, and a 576-token branch, with eight output
tokens. The batches took 26.71, 8.71 and 8.90 seconds after engine initialization.

The emitted cache receipts passed the actual typed receipt parser and audit gate:
72 audited calls, 18 on each of four ranks, zero violations, zero comparison
errors and zero refused references. The full-stack execution check completed.

The negative control supplied only a cache contribution and deliberately zeroed
a served KV page. Every rank emitted a candidate-failure receipt for serving
bytes that did not match the recorded prefix. It failed for the intended content
violation rather than passing with incorrect output or failing at admission.

## Coverage boundary

The GLM checks cover full-attention KV, draft and index state. Paged
sliding-window validation is implemented and tested against native SGLang cache
classes on CPU; no sliding-window model GPU result is claimed here. Validation of
per-request sliding-window rings, paged sliding-window index state and compressed
KV pages is implemented and tested on CPU only; no GPU result is claimed for them.
Recurrent checkpoints are refused rather than validated.

Target-level preservation through successive cache replacements is separately
tested through the production stack planner and materializer. The GPU controls
use development bundles and do not establish settlement or mainnet qualification
for a cache proposal.

## Final-source regression

Revision `4ed4777` completed a GLM TP4/DP4 regression with the retained
kernel contribution, native HiCache and graphs enabled: 24 short prefixes,
reuse, flush, replay and reuse, with cache completions on all four ranks and
the full-stack terminal completion gate satisfied.
