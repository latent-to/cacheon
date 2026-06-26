#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat <<'EOF'
Backend A/B matrix for Nemotron 3 Ultra.

Run these manually, one at a time, after the official baseline is correct.
Interleave if you are comparing performance: official / candidate / official /
candidate, while logging clocks and power.

official:
  scripts/run_nemotron3_server_nsys.sh official

fp8_auto:
  DISABLE_DEEP_GEMM=0 FP8_GEMM_BACKEND=auto scripts/run_nemotron3_server_nsys.sh fp8_auto

moe_flashinfer:
  MOE_RUNNER_BACKEND=flashinfer_trtllm FP4_GEMM_BACKEND=flashinfer_cutlass \
    scripts/run_nemotron3_server_nsys.sh moe_flashinfer

mamba_default:
  scripts/run_nemotron3_server_nsys.sh mamba_default

no_mtp:
  scripts/run_nemotron3_server_nsys.sh no_mtp

For each started server:
  CAPTURE=decode NUM_PROMPTS=16 MAX_CONCURRENCY=16 RANDOM_INPUT_LEN=8192 RANDOM_OUTPUT_LEN=512 \
    scripts/bench_nemotron3_serving_profile.sh
  scripts/collect_nemotron3_nsys.sh
EOF
