# GLM-5.3-Flash prefill/decode development check inputs

Use these inputs with the `nvidia/GLM-5.3-Flash-NVFP4` checkpoint (revision
`1d00738646cbb0c6d0a9eb403dac5f366f1473a7`, `modelopt_fp4`) on eight B300 GPUs.
The model is hybrid: KDA linear-attention layers between MLA layers with DSA
sparse attention, 288 routed experts with eight active, and a native MTP layer.
The engine image is SGLang's `dev-cu13` nightly at main `314bce7c`
(`lmsysorg/sglang@sha256:387e9021…`), used unmodified.

The arena serves one prefill/decode pair on the whole node: prefill on GPUs 0-3
(TP4 with DP attention 2) and decode on GPUs 4-7 (TP4, DP 1), with KV moving between
them over Mooncake's intra-node NVLink transport. Both roles run NVFP4 weights, an
FP8 KV cache with TRT-LLM DSA kernels, FlashInfer TRT-LLM MoE and native MTP (EAGLE,
five steps, top-k one, six draft tokens). The roles differ only where
`engine-config.json` says so under `pd_prefill` and `pd_decode`: 16K prefill chunks on
prefill; DP 1, 128 running requests, 128-row decode graphs and ReplaySSM speculative
state (`enable_linear_replayssm_spec`) on decode. SGLang refuses ReplaySSM on a
prefill server, and the eager audit and pristine reference run the prefill role's
one-engine form, where the recurrent-state audit does not support it. The replay
routes each session to one prefill DP rank for all its turns; decode has one rank.

The configuration is TEE-shaped: NVLS, symmetric-memory multicast, the HiCache host
tier and RDMA are off. The KV transfer uses legacy CUDA IPC handles; it has not been
measured under confidential computing.

## Speed policy

Speed policy 21 replays the sealed agent-trace pool in a closed loop at three loads
in each engine lifetime, low, mid and high (`replay_load` in `operator.json`), each
after its own warm window, and scores the geometric mean of the three served-throughput
ratios against the retained stock reference. Each load also holds the candidate's pooled
per-user decode median to its own floor (`decode_floors`). A candidate cannot win by
tuning one batch size: a gain at one load is diluted by the other two, and a loss in
per-user speed at any load fails it.

The values in `operator.json` are placeholders until the pair's load curve is
measured: loads 16, 64 and 128 sessions; floors 240, 165 and 80 tokens/s, about 90
percent of stock's light-load (~270) and ~2 req/s (~185) per-user medians and the high-load
floor Shiv set (2026-11-11). Stock must hold each floor at its load, or the reference
campaign refuses the commission as mis-sealed before any candidate runs. The
`node` block (scheduler pin starts and client CPU set), `tokens_per_prompt` and
`model_profile_key` are also placeholders for the commissioned node and slice.

## Development check

`cacheon check` runs the pair's one-engine form on four GPUs: the shared options with
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
