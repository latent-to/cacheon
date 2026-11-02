# DeepSeek-V4.1-Flash prefill/decode arena

Production-path results from 2026-10-27 UTC on one eight-GPU B300 host serving
DeepSeek-V4.1-Flash as one prefill/decode pair (prefill on GPUs 0-3, decode on
GPUs 4-7, each DEP4 with Mooncake moving KV between them), measured through the
[paired replay qualification](../validator-guide/qualification.md) at source
revision `b36a6af8`. The arena's inputs and engine settings are in
[`examples/arena_inputs/dsv41pd`](https://github.com/latent-to/cacheon/blob/main/examples/arena_inputs/dsv41pd/README.md).
They establish the stock reference the arena is sealed on and the production
path's cost against a harness that sends HTTP straight to the engines. They are
not a crown or a serving claim.

## Stock at the sealed load

Speed policy 20 replays 192 sessions in a closed loop from a 247-session pool of
agent traces (3,572 turns and 3,932,623 output tokens a window), each window
opening with one lockstep cold round. Three stock windows of pair `dsv41pd-n`:

| Window | Output tokens/s | Pooled decode median | Warm time to first token, median / p90 | Span |
|---|---:|---:|---|---:|
| 1 | 16,114 | 161.0 | 1.93 s / 4.42 s | 244 s |
| 2 | 16,368 | 163.3 | 2.23 s / 4.12 s | 240 s |
| 3 | 16,174 | 163.9 | 2.47 s / 4.34 s | 243 s |

The second stock boot of the retained reference read 16,031, 16,128 and 16,028
tokens/s at decode medians of 161.8, 162.8 and 164.7. The decode floor is 150
tokens/s per user, so 192 sessions holds it with about 8 percent margin. The cold round's first tokens take 26 s at the median: every
session's first prompt arrives at once with an empty cache. In the middle of a
window the node serves about 24,000 output tokens/s with about 130 turns
decoding and 55 waiting for their first token.

## Two production-path costs

The first stock reference at 192 sessions read a decode median of 130 and 14,700
tokens/s, and policy 20 refused it as mis-sealed. Two costs of the validator's own
path explained it.

- **Prefix-cache content check on stock.** The check also wrapped stock's trusted
  cache. On each finished request it hashed the request's new pages on the decode
  scheduler's thread, 16 percent of decode rank 0's samples, and the other DP ranks
  waited for that rank at the batch sync: 18 percent fewer tokens per
  decode-second than the harness at the same mean concurrency. The check now runs
  only around a candidate's cache (`38dcef8a`), and the decode median rose to 167.
- **Token IDs through one event loop.** The in-container session worker decodes each
  request frame and re-encodes the prompt for the prefill server. With prompts up to
  135k tokens, stdlib JSON was about 30 percent of a loop that ran at 70 to 94
  percent of a core, and turns queued before their first token while the prefill
  GPUs idled part of the time. Both now use orjson (`b36a6af8`): the loop runs at
  about 65 percent, and the warm first-token median fell from 3.0 s to 2.2 s.

## Remaining gap to the harness

The harness, SGLang's router with cache-aware routing on both roles, served the
same 192-session workload on the same image at 17,000 to 18,000 tokens/s, a
161-164 decode median and a 1.1-1.4 s first-token median. Production is about 9
percent below it, all of it before the first token: the session worker's loop
still carries every prompt, and the replay keeps each session on one DP rank of
each role, where the router moves a request to a less-loaded rank when ranks
drift apart. The four prefill ranks show that drift, often one near idle while
another is saturated.

## Controls on the commissioned image

The first identity run on `b36a6af8` read 15,679, 15,863 and 15,888 tokens/s
on its first candidate boot against the reference's 16,000 to 16,400, then
stopped without a verdict: the worker checkpoint that lets a replacement worker
resume the run keeps every turn's output IDs for each engine lifetime, and four
stock boots plus one candidate boot came to 516 MB against a 128 MiB bound. The
bound is now 1 GiB. The rerun on `aca86a85` published that checkpoint (72 MB
compressed) and stopped the identity after one candidate boot (15,704, 15,776
and 15,748 tokens/s), then failed to ship its evidence: the attempt record keeps
every window's read, about 17 MB over 15 windows, against a 16 MiB limit per
artifact. The limits are now 32 MiB an artifact and 44 MiB in all, under the
64 MiB response that carries them.

On `c0c31952` the identity completed. The wrong-MLP control's audit then
reported every layer wrong on all four ranks, but the run ended without a
decision: the audit's single colocated engine occupies the first four GPUs, and
the final-warmup device check required an active process on all eight. The
check now covers only the GPUs the audit engine occupies (`ebda4441`).

| Control | Expected | Result |
|---|---|---|
| Faithful identity on `model.layers.*.mlp` (`c0c31952`) | `FAIL`, every window within noise of 1.00 | `FAIL` `speed_threshold_not_met`: ratio 0.985 (standard error 0.012, required 1.027), stopped after one candidate boot by the futility margin; candidate windows 15,693 / 15,883 / 15,871 against a stock reference mean of 16,058 |
| Wrong MLP under the registered-control audit policy (`ebda4441`) | Speed non-`PASS`, then audit `FAIL` with violations | Speed non-`PASS` over two candidate boots (15,688 to 15,935 tokens/s), then `FAIL` `slot_audit_failed`: passing fraction 0.0 against 0.75 on all 40 layers and 4 ranks; attempt record 20.3 MB |

A faithful identity reads about 0.985 of stock on pairs `n`, `o` and `p`. Those
runs measured the reference in the identity's own evaluator process, so its
first candidate boot was the process's fifth engine lifetime, read against
lifetimes one to four: the reference position below accounts for it.

## Reference position

Replay rates fall with every engine lifetime one evaluator process runs:
against its first lifetime, PD stock costs 1.4 percent more in the second,
about 2 percent in the third and fourth and about 2.4 percent in the fifth and
sixth. The next qualification, a new process that first builds its bundle, starts fast again;
750 idle seconds inside one process, the GPUs back at 24 to 26 °C, do not. All of the added time is before the first token, while
decode time per token, the replay bridge's delay, turnaround and sessions in
flight stay flat. Earlier lifetimes' retained evidence lengthened every full
collection on the loop that times first tokens; with the evaluator's heap
collected and frozen at each lifetime's start (`d7665c3e`), the two lifetimes
of one process on the Verda host below agreed to 0.14 percent in their steady
windows.

Each qualification runs in a new process, so candidates read as its first two
lifetimes. Policy 20 measured all four reference boots in one process, handing
every candidate about 0.5 percent against a 1.206 percent bar: ten tickets on
2026-10-28 averaged +0.46 percent, and −0.08 percent against the reference's
first two boots; the one PASS (+1.42 percent, +0.88 there) was revoked. Policy
21 measures each stock pair as the first two lifetimes of its own process.

The first policy-21 reference, on an 8-GPU AWS B300 node (2026-10-30), read
mean window costs of 237.13 and 237.73 seconds in its first pair's process
and 236.88 and 237.84 in its second's: the second lifetime cost 0.25 and 0.41
percent more than the first, and stock against stock at matched positions,
the second pair read against the first, came to 1.0003 (99 percent band 0.973
to 1.028).

## A second host

On an 8-GPU B300 VM from Verda (2026-11-01), the same sealed workload's stock
windows took about 265 seconds against 237 on the AWS node, and stock lifetimes
spread 0.52 percent against 0.20. None of the difference was in the engines.

- **Tokenization on the clock.** The replay bridge re-templated and
  re-tokenized every prompt in each timed window and wrote its token IDs to
  `bridge.jsonl`. The client's gap between a turn's end and its next request
  was 0.47 to 0.49 s on Verda and 0.42 to 0.43 s on AWS. The evaluator now
  tokenizes the sealed slice once per process before the first timed window,
  and the gap is 0.011 s.
- **Pins from a topology the guest hides.** The VM lists 240 one-thread
  sockets, yet vCPU k and k+120 share a core and 16 of its 120 cores run at
  twice the speed of the rest. The sibling-free pin rule put each prefill
  scheduler on a decode scheduler's core, and the DP controllers busy-polled
  on slow cores. Engine-role processes now run on vCPUs the worker times at
  prepare, and it refuses a table with a slow or shared-core pin.

With both, stock windows took 214.6 to 222.9 s and the warm first-token
median fell from 4.0-4.4 s to 1.02 s. The first window of each lifetime still
ran 1.4 to 3.1 percent long: the stock Triton kernel `_committed_swa_locations`
takes the KV pool size as a `constexpr`, the runtime seed was built at another
pool size, and 43 variants compiled inside that window. Over four stock
lifetimes the third windows read 214.73 to 214.94 s, the second up to 1.3
percent long with nothing compiling. Stages from z4 on replay two sealed warm
windows untimed before the three scored ones and seal null and boot noise of
0.005 and 0.003, which puts the PASS boundary near 0.33 percent instead of
about 1.2. The first such lifetime read 214.83, 214.60 and 216.40 s; the second
215.38, 214.16 and 214.53 s.

## Time per qualification

On `f951c47e` (2026-10-31), a candidate whose speedup of 1.0252 cleared the
speed boundary but not its payment bar of 1.0393 ended `FAIL`
`below_payment_bar` at its speed stage, with no stock bracket, audit or
pristine T. The request took 37.3 minutes: two candidate lifetimes of three
windows each, then 90 seconds after the second lifetime's engines exited, in
which the worker graded the windows and wrote the 619 MB speed record (53
seconds), the 107 MB worker checkpoint (8 seconds), and the stage exit and
result (29 seconds). On the previous write path, the 727 MB speed record of a
PASS on 2026-10-30 was written 7.4 minutes after its stock bracket ended, and
that qualification ran 91 minutes with its audit and pristine T.
