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
sixth. A new process starts fast again; 750 idle seconds with the GPUs back at
24 to 26 °C do not. All of the added time is before the first token, while
decode time per token, the replay bridge's delay, turnaround and sessions in
flight stay flat. The in-process cause is not yet identified.

Each qualification runs in a new process, so candidates read as its first two
lifetimes. Policy 20 measured all four reference boots in one process, handing
every candidate about 0.5 percent against a 1.206 percent bar: ten tickets on
2026-10-28 averaged +0.46 percent, and −0.08 percent against the reference's
first two boots; the one PASS (+1.42 percent, +0.88 there) was revoked. Policy
21 measures each stock pair as the first two lifetimes of its own process.
