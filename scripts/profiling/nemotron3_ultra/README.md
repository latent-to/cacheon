# Nemotron 3 Ultra 550B-A55B-NVFP4 Profiling Runbook

This folder is for the day-0 Nemotron 3 Ultra profiling pass. Keep it separate
from Qwen3.5 and DeepSeek artifacts; the architecture and likely optimization
targets are different.

## Verified Sources

- NVIDIA Nemotron research page: Nemotron 3 Ultra was published June 4, 2026,
  has 550B total / 55B active parameters, uses a hybrid Mamba-Attention MoE
  architecture, LatentMoE, MTP, 1M context support, and is released in NVFP4 and
  BF16 checkpoints.
  https://research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/
- Hugging Face model card: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`,
  OpenMDW-1.1, 550B / 55B active, LatentMoE + Mamba-2 + MoE + Attention hybrid
  with MTP. The card says the minimum recommended single-node deployment for
  NVFP4 is 4x B200 and gives an SGLang recipe tested on 4x B200.
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
- LMSYS/SGLang blog: SGLang and Miles day-0 support exists. The blog lists NVFP4
  support on Blackwell GPUs, notes that pure TP is capped at 8 by `n_groups=8`,
  and shows a simpler `dev-nemotron3-ultra` / TP=8 launch command.
  https://www.lmsys.org/blog/2026-06-04-nvidia-run-nemotron-3-ultra/
- SGLang quantization docs: NVFP4 GEMM backend selection is configurable via
  `--fp4-gemm-backend`; FP8 GEMM via `--fp8-gemm-backend`. Auto-selection and
  backend availability differ by SM100/SM120 and installed FlashInfer/CUTLASS.
  https://sgl-project.github.io/advanced_features/quantization.html

## Corrections To The Pasted Research

- The most useful official SGLang baseline is currently the Hugging Face model
  card's 4x B200 recipe, not only the LMSYS blog's 8x B200 snippet.
- NVIDIA's official 4x B200 SGLang recipe does intentionally use fallback-like
  paths: `SGLANG_DISABLE_DEEP_GEMM=1`, `--fp8-gemm-backend triton`,
  `--moe-runner-backend triton`, `--mamba-scheduler-strategy no_buffer`, and
  `--disable-piecewise-cuda-graph`. That is a real optimization opening, but do
  not claim a win until NSYS proves the slice and NCU proves headroom.
- The model card says "minimum recommended" 4x B200 for single-node NVFP4. The
  LMSYS blog says NVFP4 is supported on 2x GB/B200/B300, but for profiling the
  official baseline, use 4x B200/B300 if at all possible. Treat 2x as a small
  context bring-up experiment, not the record-setting config.
- There is no concrete SGLang tok/s "peak" number in the public docs I found.
  NVIDIA reports relative throughput versus other models at 8k input / 64k
  output, but the actual local bar should be measured on our pod.

## Container And Hardware

Default image from the model card:

```bash
IMAGE=lmsysorg/sglang:v0.5.12.post1
```

Fallback image from the LMSYS blog:

```bash
IMAGE=lmsysorg/sglang:dev-nemotron3-ultra
```

Recommended profiling hardware:

```bash
4x B200 or 4x B300, non-confidential-computing, NCU counters enabled
```

Abort before model download if `ncu` returns `ERR_NVGPUCTRPERM`.

## Official Baseline Flags

The scripts default to the Hugging Face model-card SGLang recipe:

```bash
SGLANG_DISABLE_DEEP_GEMM=1
python3 -m sglang.launch_server \
  --model-path /model \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name nvidia/nemotron-3-ultra \
  --tp-size 4 \
  --ep-size 4 \
  --context-length 262144 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 32768 \
  --fp8-gemm-backend triton \
  --moe-runner-backend triton \
  --mamba-scheduler-strategy no_buffer \
  --disable-piecewise-cuda-graph \
  --reasoning-parser nemotron_3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --kv-cache-dtype fp8 \
  --trust-remote-code
```

Do not add `--quantization modelopt_fp4` by default. This is an offline NVFP4
checkpoint; SGLang should parse the quantization config from the model files.

## First Pod Sequence

1. Copy this folder to the VM.
2. Run `scripts/preflight_nemotron3_pod.sh`; stop immediately if NCU counters
   are restricted.
3. Put the HF token at `/root/token` or export `HF_TOKEN`.
4. Run `scripts/download_nemotron3_ultra_nvfp4.sh`.
5. Start the official baseline under NSYS:

```bash
scripts/run_nemotron3_server_nsys.sh official
```

6. When the server is ready, run prefill and decode captures:

```bash
CAPTURE=prefill NUM_PROMPTS=16 MAX_CONCURRENCY=16 RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=1 \
  scripts/bench_nemotron3_serving_profile.sh

CAPTURE=decode NUM_PROMPTS=16 MAX_CONCURRENCY=16 RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=512 \
  PROFILE_START_STEP=64 PROFILE_STEPS=128 scripts/bench_nemotron3_serving_profile.sh
```

7. Stop the container and export reports:

```bash
scripts/collect_nemotron3_nsys.sh
```

## Backend A/B Matrix

Only run these after the official baseline works and produces correct text.
Interleave runs and log clocks/power/temperature:

- `official`: Triton FP8 GEMM, Triton MoE, DeepGEMM disabled, no-buffer Mamba.
- `fp8_auto`: `SGLANG_DISABLE_DEEP_GEMM=0`, `FP8_GEMM_BACKEND=auto`, MoE still
  Triton. Tests whether DeepGEMM/FlashInfer/CUTLASS are usable for the FP8
  linears in this model.
- `moe_flashinfer`: `MOE_RUNNER_BACKEND=flashinfer_trtllm`,
  `FP4_GEMM_BACKEND=flashinfer_cutlass`. Tests the main MoE lever.
- `mamba_default`: drops `--mamba-scheduler-strategy no_buffer`. Tests whether
  the official no-buffer setting is still required or a throughput drag.
- `no_mtp`: disables speculative flags. This is not the production target, but
  it makes per-token kernel attribution easier and isolates launch overhead.

The only acceptable result is an e2e throughput comparison plus correctness
sanity. Backend crashes or gibberish are still useful because they explain why
the official command disabled them.

## What To Profile

NSYS first:

- Prefill pie: Mamba SSD/conv1d vs MoE vs attention vs FP8/NVFP4 GEMMs vs router
  and communication.
- Decode pie: same categories, with CUDA graph/eager launch overhead called out.
- MTP overhead: compare official vs `no_mtp` to understand how much MTP changes
  kernel mix and throughput.

NCU second, based on the NSYS top slices:

- If MoE dominates: NCU the live NVFP4 LatentMoE kernels and then build a
  standalone grouped-MoE microbench once `config.json` gives expert/top-k/latent
  shapes.
- If Mamba dominates prefill: NCU the Mamba-2 SSD/chunked-scan and causal-conv1d
  kernels; then write a standalone microbench matching `ssm_state_size`,
  `conv_kernel`, `n_groups`, `mamba_num_heads`, and chunk size from config.
- If FP8 linears dominate: compare `fp8-gemm-backend=triton` vs `auto`/CUTLASS/
  FlashInfer and NCU the winner and loser.

Do not spend hours NCU-profiling a live TP/EP serving process if replay starts
saving huge memory or hangs. As with Qwen3.5 GDN, fall back to a standalone
single-process microbench once the exact kernel and tensor shapes are known.

## Artifact Handling

Before deleting the VM:

```bash
cd /root
tar --zstd -cf /home/shadeform/nemotron3_ultra_profile_$(date +%Y%m%d_%H%M%S).tar.zst nemotron3_profile
sha256sum /home/shadeform/nemotron3_ultra_profile_*.tar.zst
```

Download to `/private/tmp/nemotron3_ultra_profile_download/` and verify SHA256
locally before shutting down.
