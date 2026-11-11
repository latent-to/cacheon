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
`operator.json` seals at 1,792 GiB. SGLang refuses
ReplaySSM on a prefill server, and the eager audit and pristine reference run the
prefill role's one-engine form (TP2, on GPUs 0-1 and 4-5), where the recurrent-state
audit grades the KDA state tensors without it. The replay routes each session to one
prefill server for all its turns, so its context stays in that server's radix cache.

The configuration is TEE-shaped: NVLS, symmetric-memory multicast and RDMA are off,
and the host tier is ordinary pinned host memory. The KV transfer uses legacy CUDA IPC handles; it has not been
measured under confidential computing.

## Speed policy

Speed policy 21 replays the sealed agent-trace pool in a closed loop at three session
concurrencies in each engine lifetime, low, mid and high (`replay_load` in
`operator.json`), and scores the geometric mean of the three served-throughput ratios
against the retained stock reference. One warm window at the high load precedes one
scored window per load (`replay_windows`), highest first; a lower load replays the
first sessions of the pool in proportion to its concurrency, so the three windows
together replay less work than the three whole-pool windows of a one-load lifetime and
the bundle's wall clock does not grow. Each load also holds the candidate's pooled
per-user decode median to its own floor (`decode_floors`). A candidate cannot win by
tuning one batch size: a gain at one load is diluted by the other two, and a loss in
per-user speed at any load fails it. The high load saturates prefill, so it rewards
prefill and time-to-first-token gains.

The values in `operator.json` are placeholders read off stock's measured load curve
(2026-11-11): about 130 concurrent sessions (~2 requests/s, ~160 tokens/s per user),
about 250 (~3.5 requests/s, decode median ~115, break-even) and about 300 (~4.5
requests/s, prefill-saturated); floors 145, 105 and 80 tokens/s. Recalibrate them from
the commissioned pair's stock reference: stock must hold each floor at its load, or the
reference campaign refuses the commission as mis-sealed before any candidate runs, and
the sealed slice must hold at least as many sessions as the high load. The `node` block
(scheduler pin starts, client CPU set and, with DP 1 everywhere, no `engine_cpus`),
`tokens_per_prompt` and `model_profile_key` are also placeholders for the commissioned
node and slice.

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
