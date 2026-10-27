# DeepSeek-V4.1-Flash prefill/decode development check inputs

Use these inputs with the official `deepseek-ai/DeepSeek-V4.1-Flash-1017`
checkpoint (revision `1bae0e1088993b9b82de08f59de5be5a34e42170`: MXFP4 routed
experts, FP8 attention, dense layers and Engram tables) on eight B300 GPUs. The
engine image is SGLang's nightly `nightly-dev-20261027-5884e4fe`
(`lmsysorg/sglang@sha256:2200816a…`) with the Python and JIT sources of the
latent-to/sglang branch `cacheon/dsv41-pd-tee` @ `15338a8e64` laid over it: SGLang
main `d3a157a2`, the PD decode-radix guard and deterministic top-k ports, and the
upstream DeepSeek-V4.1 fixes #45347, #46676, #46618, #46093, #46593, #46088 and #46023.
Main beyond the nightly changes no compiled code, so the overlay equals a nightly
at that head.

The arena serves one prefill/decode pair on the whole node: prefill on GPUs 0-3
and decode on GPUs 4-7, each TP4 with expert parallelism 4, DP attention 4,
MegaMoE and DSpark, with KV moving between them over Mooncake's intra-node NVLink
transport. The roles differ only where `engine-config.json` says so under
`pd_prefill` and `pd_decode`: 128K prefill chunks and 128 running requests on
prefill; 192 running requests, 48-row decode graphs per DP rank, a 32-token stream
interval and a decode-side radix cache on decode; and MegaMoE's per-rank token cap
(32768 on prefill, 16384 on decode). The replay places each session on one DP
rank of each role for all its turns, which is what both radix caches need. The
configuration is TEE-shaped: the Engram tables stay in GPU memory, and NVLS,
symmetric-memory multicast and Engram peer shards are off. Two parts of the pair
are still unmeasured under confidential computing: the KV transfer (legacy CUDA IPC
handles, as SGLang's custom all-reduce uses) and MegaMoE's symmetric memory.

Speed policy 20 replays 192 sessions in a closed loop from a 247-session pool of
agent traces, each session capped at 16,000 output tokens, and scores served output
tokens per second against the retained stock reference, gated on a pooled
per-user decode median of at least 150 tokens per second. The session count is where
stock holds that floor: a development harness sending HTTP straight to the engines
measured 17,920 tokens/s at a 163.4 decode median at 192 sessions, and 160.8 and 164.4
on a re-run (8xB300, 2026-10-27). The production replay first read 130 there because
the prefix-cache content check also ran on stock's cache: on each finished request it
held a decode scheduler's loop, and the other DP ranks waited for it at the batch
sync (16% of decode rank 0's samples, 18% fewer tokens per decode-second at the same
mean concurrency). The check now runs only around a candidate's cache. The decode
radix cache is worth 11.7 percent of the harness's throughput (16,040 tokens/s
without it). A seeded boot of the pair takes
about four and a half minutes. Mooncake replaced NIXL: the same steady throughput,
a 165 s faster boot, and none of the whole-pool descriptor registration NIXL pays the
first time each decode rank meets a prefill rank (about 500 s of every boot's first window).

`cacheon check` runs the pair's one-engine form on four GPUs: the shared options
with the prefill role's settings, as the eager audit runs it. Mount this folder
read-only at `/arena`, the model at `/model` and a writable directory at `/work`,
and use the DeepSeek-V4.1 development requests in
`../dsv41flash/development-requests.json` (same tokenizer). A bundle loads in both
engines of the pair: its kernels see long prefill chunks on the prefill GPUs and
small decode and verify batches under CUDA graphs on the decode GPUs.

Build the qualification image as for the other arenas, on the PD engine image:

```bash
build_dir=$(mktemp -d)
mkdir "$build_dir/src"
git archive HEAD | tar -x -C "$build_dir/src"
docker build -t cacheon-dsv41pd -f examples/arena_inputs/dsv41pd/Dockerfile \
  --build-arg SGLANG_IMAGE=local/dsv41-pd-tee:15338a8e64 "$build_dir"
```
