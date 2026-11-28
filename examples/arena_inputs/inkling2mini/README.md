# Inkling 2 Mini development check inputs

Use these inputs with the `thinkingmachines/Inkling-2-Mini` checkpoint (revision
`7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6`) on H200 GPUs. The model is a looped
mixture of experts: of its 24 decoder layers, layers 6-17 run three times, so a token
makes 48 layer visits, 8 of which attend globally and the rest over a 1,024-token
sliding window. Each layer routes a token to 8 of 128 experts, stored in NVFP4 (layers
0-19) and MXFP8 (layers 20-23) and run through Marlin MoE kernels, since H200 has no FP4
tensor cores; the rest of the model is BF16. The DSpark draft is `mtp.safetensors` in
the same directory. The engine image is SGLang's Inkling 2 build
(`lmsysorg/sglang@sha256:efac853b…`, SGLang `0.0.0.dev1+inkling2`, torch
2.14.1+cu130), used unmodified.

The arena serves each pair of GPUs as one engine, so an eight-GPU node holds four lanes:
TP2 with expert parallelism 2 and DSpark speculative decoding. There is no
prefill/decode split, because SGLang's Inkling 2 DSpark worker refuses disaggregation.
`engine-config.json` seals FA4 attention, torch BF16 GEMMs, 128-token pages, a
300K-token context, 64 running requests, 32,768-token prefill chunks
(`chunked_prefill_size` and `max_prefill_tokens`), 93% of device memory for
weights and cache, and a sliding-window pool at 5% of the full-attention pool's
tokens (Inkling's default is 10%). It also sizes the hierarchical cache at 1×
the device pool (~205 GiB per TP2 lane, 410 GiB per NUMA socket so all four
lanes run concurrently on a 2 TiB host) with 6 parallel host-allocation threads.
An explicit `hicache_ratio` also turns off SGLang's host-memory auto-sizing. Inkling 2's
tensor-parallel MoE combine uses NVLink multicast, so unlike the prefill/decode arenas
this configuration leaves multicast on and is not TEE-shaped.

On the AgentX agent traces at 20 and 30 sessions per GPU, the larger pools and the host
tier did better than SGLang's defaults (2026-11-23):

- prefix hit rate rose from 88.3% and 82.0% to 90.9% and 90.3%;
- median per-user speed rose from 34 and 19 to 39 and 29 tokens/s.

A boot with a warm compile cache takes about three minutes. The
[results](../../../docs/results/inkling-2-mini-h200.md) record the lane-width choice
and these runs.

## Development check

`cacheon check` runs one lane on two GPUs with these options. Mount the model read-only
at `/model`, this folder at `/arena` and a writable directory at `/work`. Use
`development-requests.json` as the requests, each greedy for 48 tokens:

- four 8.3K-token prompts sharing a 7.96K-token prefix;
- the same four extended by about 200 tokens, so they hit the prefix cache;
- eight prompts of about 480 tokens.

Three targets bind on this model:

- `forward_pass` binds the MoE blocks, `model.layers.*.mlp`. Inkling 2 registers its
  decoder as `llm` behind a `model` property. On a DSpark verify window of up to 32
  tokens a block returns its routed rows (`RoutedOutput`) rather than one tensor, and
  the audit grades the rows. Attention, convolution and whole decoder-layer nodes do not
  bind yet.
- `prefix_cache` binds the scheduler's tree cache.
- `engine` binds the queue order and the listed scheduler decisions.

The audit role grades every call of an audited node. On these requests the MoE identity
control spent 17 of its 21 minutes there, so allow for that in the timeout. The
[results](../../../docs/results/inkling-2-mini-h200.md#slot-checks) list each control's
verdict.

Build the qualification image from the repository root, then check a bundle:

```bash
build_dir=$(mktemp -d)
mkdir "$build_dir/src"
git archive HEAD | tar -x -C "$build_dir/src"
docker build -t cacheon-inkling2mini -f examples/arena_inputs/inkling2mini/Dockerfile "$build_dir"
docker run --rm --gpus '"device=0,1"' --ipc=host --shm-size 32g \
  -v /path/to/Inkling-2-Mini:/model:ro -v "$PWD/examples/arena_inputs/inkling2mini:/arena:ro" \
  -v /path/to/bundles:/bundles:ro -v /path/to/work:/work cacheon-inkling2mini \
  /usr/bin/python3.12 -m cacheon.cli check /bundles/my_bundle --model /model \
  --engine-config /arena/engine-config.json --requests /arena/development-requests.json \
  --output /work/my_bundle --timeout-seconds 5400
```
