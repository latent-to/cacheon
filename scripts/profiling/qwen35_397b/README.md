# Qwen3.5-397B-A17B-NVFP4 SGLang Profiling Runbook

This folder is separate from the DeepSeek V4 scripts. It is built for the Qwen
3.5 397B NVFP4 profiling pass where the main question is whether GDN linear
attention is worth optimizing relative to full attention, MoE, projection GEMMs,
router/glue, causal conv1d, and TP all-reduce.

## Verified Sources

- NVIDIA model card: `nvidia/Qwen3.5-397B-A17B-NVFP4` is the NVFP4 quantized
  Qwen3.5-397B-A17B checkpoint, 397B total / 17B active, tested on B200, and
  intended for SGLang. The model card sample uses `lmsysorg/sglang:v0.5.9` and
  `--quantization modelopt_fp4`.
  https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4
- SGLang Qwen3.5 docs: Qwen3.5 support requires current/main SGLang or Docker,
  lists the FP4 model ID as `nvidia/Qwen3.5-397B-A17B-NVFP4`, and recommends
  current Docker (`lmsysorg/sglang:latest`) for NVIDIA.
  https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5
- SGLang attention-backend docs: GDN is selected automatically by the model
  architecture; use `--linear-attn-prefill-backend` and
  `--linear-attn-decode-backend` to select GDN kernel backends. On Blackwell,
  the full-attention backend for hybrid GDN models can be `triton`,
  `trtllm_mha`, or `fa4`.
  https://lmsysorg.mintlify.app/docs/advanced_features/attention_backend
- SGLang source confirms the flags used here are accepted in current main:
  `modelopt_fp4`, `kv-cache-dtype`, `trtllm_mha`, `flashinfer_trtllm`,
  `flashinfer_cutlass`, `mamba-ssm-dtype`, and per-phase linear-attention
  backend flags.
  https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/server_args.py
- SGLang's Qwen3.5 optimization tracking issue explicitly calls out GDN prefill,
  GDN decode, Blackwell CuteDSL/FlashInfer work, NVFP4 MoE, TRTLLM_MHA for
  Blackwell, and all-reduce fusion as active optimization surfaces.
  https://github.com/sgl-project/sglang/issues/18590

## Container Choice

Default:

```bash
IMAGE=lmsysorg/sglang:v0.5.12.post1-cu130
```

Rationale: it is a current tagged CUDA 13 image, while NVIDIA's model card only
shows the older `v0.5.9` sample and SGLang docs say current/main is required for
Qwen3.5. If `v0.5.12.post1-cu130` regresses, fallback candidates are:

```bash
IMAGE=lmsysorg/sglang:latest
IMAGE=lmsysorg/sglang:v0.5.9
```

Always record the digest with `preflight_qwen35_pod.sh`.

## Hardware Requirement

The default experiment is now TP=2:

```bash
CUDA_VISIBLE_DEVICES=0,1
TP_SIZE=2
```

This changes the GDN per-rank kernel shape from the earlier TP=4 note. If the
full model has 64 value heads and 16 QK heads, TP=2 means each rank sees about
32 value heads / 8 QK heads / head_dim 128, not the TP=4 shape of 16 value heads
/ 4 QK heads. Use NCU LaunchStats from the GDN runs as the source of truth for
microbench calibration.

## Shared Model Flags

The scripts use:

```bash
--model-path /root/models/Qwen3.5-397B-A17B-NVFP4
--tp 2
--quantization modelopt_fp4
--kv-cache-dtype fp8_e4m3
--attention-backend trtllm_mha
--moe-runner-backend flashinfer_trtllm
--fp4-gemm-backend flashinfer_cutlass
--linear-attn-prefill-backend triton
--linear-attn-decode-backend triton
--mamba-ssm-dtype bfloat16
--trust-remote-code
```

`--kv-cache-dtype fp8_e4m3` is runtime KV cache quantization, not FP8 weights.
The checkpoint itself is NVFP4/modelopt FP4.

## Fast Start On A Pod

Copy only this folder if you want to keep the pod tidy:

```bash
rsync -az scripts/profiling/qwen35_397b/ shadeform@HOST:/home/shadeform/qwen35_profile/scripts/
ssh shadeform@HOST 'sudo mkdir -p /root/qwen35_profile && sudo rsync -a /home/shadeform/qwen35_profile/scripts/ /root/qwen35_profile/scripts/ && sudo chmod +x /root/qwen35_profile/scripts/*'
```

Then on the pod:

```bash
cd /root/qwen35_profile
scripts/preflight_qwen35_pod.sh
scripts/download_qwen35_nvfp4.sh
```

Abort before model download if NCU counters fail.

## NSYS Headline Runs

Requested full captures:

```bash
scripts/run_qwen35_nsys.sh eager
scripts/run_qwen35_nsys.sh cudagraph
scripts/run_qwen35_nsys.sh big
```

The `big` mode still defaults to `batch-size=128 input-len=16384 output-len=1024`
because that is the desired stress profile. On TP=2 it may OOM under
`bench_one_batch` because this tool bypasses server-side chunking. If it fails,
do not silently treat a reduced run as equivalent; rerun with explicit overrides
and label the result, for example:

```bash
BIG_BATCH_SIZE=64 BIG_INPUT_LEN=16384 BIG_OUTPUT_LEN=1024 scripts/run_qwen35_nsys.sh big
```

These export:

- `<name>.nsys-rep`
- `<name>.sqlite`
- `<name>_cuda_gpu_kern_sum.csv`
- `<name>_cuda_api_sum.csv`
- `<name>_cuda_gpu_trace.csv`
- `<name>_cuda_gpu_trace_head400.csv`

To include steady decode windows with CUDA profiler start/stop around the middle
of decode:

```bash
RUN_STEADY_DECODE=1 scripts/run_qwen35_nsys_suite.sh
```

or one run manually:

```bash
CAPTURE=decode PROFILE_START_STEP=192 PROFILE_STEPS=96 scripts/run_qwen35_nsys.sh eager
CAPTURE=decode PROFILE_START_STEP=192 PROFILE_STEPS=96 scripts/run_qwen35_nsys.sh cudagraph
```

Use the full captures for e2e shares and the decode-window captures for
launch-overhead/idle-gap tax.

## NCU GDN Roofline Runs

The first 2026-06-04 2xB300 run proved NCU counters were available, but live
TP=2 NCU over `bench_one_batch` did not produce usable `.ncu-rep` files:

- Kernel replay reached `chunk_gated_delta_rule_fwd_kkt_solve_kernel`, then
  failed while NCU tried to save/restore large live device memory from the full
  397B serving process.
- Application replay reached the GDN prefill/decode kernels and the benchmark
  completed, but the multiprocess TP=2/NCCL teardown hung before the report was
  written.

So the primary NCU path is now the standalone microbench. It runs one Python
process on one GPU, imports the same SGLang/FLA GDN functions, allocates TP=2
per-rank shapes directly, brackets measured launches with
`cudaProfilerStart/Stop`, and lets NCU kernel replay only the isolated GDN
kernels.

Run this first on the next pod:

```bash
cd /root/qwen35_profile
scripts/run_qwen35_ncu_gdn_microbench_suite.sh
```

This produces:

- `qwen35_gdn_micro_imports_imports.json`: exact modules/signatures found inside
  the SGLang image.
- `qwen35_gdn_prefill_micro_b1_t16384_full.ncu-rep`: prefill roofline/stalls for
  batch 1, 16k tokens, TP=2 per-rank GDN heads.
- `qwen35_gdn_decode_micro_b32_t1_full.ncu-rep`: packed decode roofline/stalls
  for batch 32.

Defaults:

```bash
GDN_QK_HEADS=8
GDN_V_HEADS=32
GDN_HEAD_DIM=128
GDN_VALUE_DIM=128
NCU_SET=full
NCU_REPLAY_MODE=kernel
NCU_PROFILE_FROM_START=off
NCU_CACHE_CONTROL=none
NCU_CLOCK_CONTROL=none
```

Optional heavier prefill occupancy point:

```bash
RUN_PREFILL_B8=1 scripts/run_qwen35_ncu_gdn_microbench_suite.sh
```

Use the old live scripts only as a secondary check after the microbench report
exists.

Live prefill, real 16k shape, batch 1:

```bash
scripts/run_qwen35_ncu_gdn_prefill.sh
```

The prefill NCU mode disables CUDA graph by default so Nsight Compute reports
individual kernel metrics instead of treating the graph as the profiled workload.

Default filter:

```text
chunk_gated_delta_rule|chunk_fwd_o|chunk_scaled_dot_kkt|solve_tril|chunk_local_cumsum|wy_fast
```

Decode:

```bash
scripts/run_qwen35_ncu_gdn_decode.sh
```

Default filter:

```text
fused_recurrent_gated_delta_rule_packed_decode
```

Both use `--set full`, so they include SpeedOfLight, Compute/Memory workload,
Occupancy, LaunchStats, SchedulerStats, and WarpStateStats where supported by
the installed Nsight Compute. This fixes the previous B300 MoE gap where we had
roofline but not exact stall reasons.

If decode kernel replay looks state-corrupted:

```bash
NCU_REPLAY_MODE=application NCU_LAUNCH_COUNT=1 scripts/run_qwen35_ncu_gdn_decode.sh
```

Do not spend a new pod trying this live path first. If the live report hangs or
fails again, stop immediately and keep the standalone microbench `.ncu-rep`.

## Postprocess

Classify an NSYS kernel table:

```bash
python3 scripts/classify_nsys_kernels.py results/qwen35_eager_full_*_cuda_gpu_kern_sum.csv
```

Extract NCU launch dimensions and key metrics:

```bash
python3 scripts/extract_ncu_launch_dims.py results/qwen35_gdn_prefill_full_*_raw.csv
python3 scripts/extract_ncu_launch_dims.py results/qwen35_gdn_decode_full_*_raw.csv
```

The GDN launch dimensions from the NCU raw CSV are the preferred "exact shape"
calibration for the microbench.

## What To Watch First

1. In `*_cuda_gpu_kern_sum.csv`, compute the decode share for MoE. If
   `trtllm_fp4_block_scale_moe`/MoE dominates decode, GDN is not the first lever.
2. Check whether any `multimem` / one-shot all-reduce kernel appears. The
   classifier separates fused/multimem all-reduce from plain NCCL/ring patterns.
3. In GDN prefill NCU, compare SM% vs DRAM%, then look at tensor/MMA pipe,
   achieved occupancy vs theoretical, waves/SM, and WarpStateStats stalls.
4. In GDN decode NCU, if it is memory-bound and near bandwidth roofline, custom
   language rewrites probably have little headroom.

## Possible Improvements Over The Initial Ask

- Run separate `CAPTURE=prefill` and `CAPTURE=decode` NSYS windows for steady
  phase attribution. A full-process NSYS capture is useful, but it includes model
  load and first-step behavior unless you filter carefully.
- Keep `--set full` for the two GDN NCU runs even though it is slower. The stall
  reasons are the difference between "probably memory latency" and an actionable
  kernel rewrite direction.
- If the big run OOMs under `bench_one_batch`, do not silently lower TP or
  context length. First record the OOM and then choose whether a server-side
  chunked-prefill profile is an acceptable different experiment.
