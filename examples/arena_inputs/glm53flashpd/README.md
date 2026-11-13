# GLM-5.3-Flash prefill/decode development check inputs

Use these inputs with the `nvidia/GLM-5.3-Flash-NVFP4` checkpoint (revision
`1d00738646cbb0c6d0a9eb403dac5f366f1473a7`, `modelopt_fp4`) on eight B300 GPUs.
The model is hybrid: KDA linear-attention layers between MLA layers with DSA
sparse attention, 288 routed experts with eight active, and a native MTP layer.
The engine image is SGLang's `dev-cu13` nightly at main `314bce7c`
(`lmsysorg/sglang@sha256:387e9021…`), used unmodified.

The arena serves the whole node as two TP2 prefill servers (GPUs 0-1 and 2-3,
`pd_prefill.servers`) feeding one TP4 decode engine (GPUs 4-7), with KV moving
between them over Mooncake's intra-node NVLink transport; no role runs DP attention.
Every role runs NVFP4 weights, an FP8 KV cache with TRT-LLM DSA kernels and CuTe DSL indexer
logits, FlashInfer TRT-LLM MoE and native MTP (EAGLE, five steps, top-k one, six draft tokens). The roles
differ only where `engine-config.json` says so under `pd_prefill` and `pd_decode`: TP2,
16K prefill chunks and a 350 GB write-through HiCache host tier per rank (page-first
layout) on prefill; 128 running requests, 128-row decode graphs, a decode-side radix cache
and ReplaySSM speculative state (`enable_linear_replayssm_spec`) on decode. The host tier
held the prefix hit rate at 0.81-0.94 over 300-466 sessions against 0.3-0.7 without it
(2026-11-11); its 1.4 TB is pinned inside the pair's lifetime, whose memory cap
`operator.json` seals at 1,792 GiB. The eager audit and pristine reference run the
prefill role's one-engine form (TP2, on GPUs 0-1 and 4-5) plus decode's ReplaySSM, which
SGLang refuses on a prefill server, so the audit grades the KDA rings decode writes. The replay routes each session to one
prefill server for all its turns, so its context stays in that server's radix cache.

The configuration is TEE-shaped: NVLS, symmetric-memory multicast and RDMA are off,
and the host tier is ordinary pinned host memory. The KV transfer uses legacy CUDA IPC handles; it has not been
measured under confidential computing.

## Speed policy

Speed policy 22 replays the sealed agent-trace pool in a closed loop at three session
concurrencies in each engine lifetime, low, mid and high (`replay_load` in
`operator.json`), and scores the geometric mean of the three served-throughput ratios
against the retained stock reference. One scored window per load (`replay_windows`)
runs highest first, with no warm window (`replay_warm_windows` 0); a lower load replays
the first sessions of the pool in proportion to its concurrency. On the commissioned pod
(2026-11-12) a stock lifetime's first 224-session window ran 691 s against 690 s for its
third, so the seeded engine needs no warm-up, while the packet's two warm windows had
cost 23 of the lifetime's 58 minutes. A window lasts as long as the slice's longest
session at this model's per-user decode speed: 690 s at 224 sessions, 535 s at 160 and
338 s at 96. A load is scored in seconds per warm turn only until its active sessions fall
below half the load after the last session start (`steady_span` 0.5), so the drain tail of
the few longest sessions does not set the cost: over four stock lifetimes of the
commissioned pair the per-boot spread of the geometric mean fell from 1.45% (whole span)
to 0.58%. Each load also holds the candidate's pooled
per-user decode median to its own floor (`decode_floors`). A candidate cannot win by
tuning one batch size: a gain at one load is diluted by the other two, and a loss in
per-user speed at any load fails it. The high load saturates prefill, so it rewards
prefill and time-to-first-token gains.

The sealed slice is the DeepSeek PD packet's 247-session pool (b318b740; its largest
recorded turn output, `tokens_per_prompt`, is 22,484), so the high load is at most 247 sessions. Stock's session ramps (2026-11-11) peaked at 256
sessions and saturated prefill near 200-220, so the loads are 96, 160 and 224 sessions
with floors 85, 82 and 80 tokens/s, about 10% under the commissioned pair's stock reference (pooled decode medians
95.2, 91.3 and 89.2 over four boots, 2026-11-12). The production replay's first-turn cold wave keeps decode near its
128-request cap at every load, so these sit well under the 11-11 rig medians (143.8, 113.0, 110.5). Stock must hold each
floor at its load. The `node` block uses the replacement B300 pod's measured pin table (2026-11-13).
Each GPU has a fast scheduler vCPU and a helper on a separate physical core. The client uses eight more fast
vCPUs. DP 1 requires no `engine_cpus`. A replacement measures one fresh stock pair on its physical
GPUs before scoring candidates. The operator seals this schedule with `speed_policy_version: 22`.
The speed scorer uses both stock boots in its uncertainty estimate.

The node audit's row floor is 15% (`audit_row_floor_ppm` 150000), not the 2% default: this
model's honest twin is bitwise, FP8 block-scaled projections sit at 6-13% row error at layer
0 and the wrong control at 50% (pod, 2026-11-11). It holds once `cacheon check
--audit-row-floor-ppm 150000` passes the validator-built FP8 control and fails the wrong one.

## Replay quality

The GLM quality policy selects 32 replay turns before seeing their teacher scores. Each arm
has the same input and output budget on a selected turn. Pristine T scores each arm's own
completion. The two metrics are mean negative log-likelihood and mean excess loss above
5 nats, calculated as `sum(max(0, token_nll - 5)) / token_count` on each turn.
The candidate limits on the mean paired increases are 0.08 and 0.01 nats per token,
respectively. Positive excesses retain their magnitude. The separate stock drift
limits are 0.15 and 0.015. Each bound uses the variation across the paired turns.

The candidate passes quality when both upper bounds clear their limits, fails on a
lower bound above its limit and receives `NO_DECISION` on overlap. Invalid stock controls
also yield `NO_DECISION`. The first stock boot supplies the baseline and the second supplies the control.
The commissioned prompt count is fixed for all entrants. An older qualification
reopens with its original thresholds and prompt selection.

## Development check

`cacheon check` runs the pair's one-engine form on two GPUs: the shared options with
the prefill role's settings, as the eager audit runs it. Mount this folder read-only at
`/arena`, the model at `/model` and a writable directory at `/work`. A bundle loads in
both engines of the pair. The open target is `forward_pass` (for example
`model.layers.*.mlp`); a `prefix_cache` bundle is refused at the decode engine, whose
hybrid pool runs without a radix cache.

Build the qualification image from the repository root:

```bash
build_dir=$(mktemp -d)
mkdir "$build_dir/src"
git archive HEAD | tar -x -C "$build_dir/src"
docker build -t cacheon-glm53flashpd -f examples/arena_inputs/glm53flashpd/Dockerfile "$build_dir"
```
