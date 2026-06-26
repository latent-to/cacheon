#!/usr/bin/env bash
set -euo pipefail

NEMO3_PROFILE_ROOT="${NEMO3_PROFILE_ROOT:-/root/nemotron3_profile}"
RESULTS_DIR="${RESULTS_DIR:-${NEMO3_PROFILE_ROOT}/results}"
LOGS_DIR="${LOGS_DIR:-${NEMO3_PROFILE_ROOT}/logs}"
MODEL_REPO="${MODEL_REPO:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4}"
MODEL_PATH="${MODEL_PATH:-/root/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4}"
IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.12.post1}"
TOKEN_FILE="${TOKEN_FILE:-/root/token}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
EP_SIZE="${EP_SIZE:-4}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/nemotron-3-ultra}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
FP8_GEMM_BACKEND="${FP8_GEMM_BACKEND:-triton}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-triton}"
MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-no_buffer}"
DISABLE_DEEP_GEMM="${DISABLE_DEEP_GEMM:-1}"
DISABLE_PIECEWISE_CUDA_GRAPH="${DISABLE_PIECEWISE_CUDA_GRAPH:-1}"
ENABLE_MTP="${ENABLE_MTP:-1}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

nemotron3_require_token() {
  if [[ -z "${HF_TOKEN:-}" && -r "${TOKEN_FILE}" ]]; then
    HF_TOKEN="$(tr -d '\n' < "${TOKEN_FILE}")"
    export HF_TOKEN
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Set HF_TOKEN or place the token in ${TOKEN_FILE}" >&2
    exit 2
  fi
}

nemotron3_quote_cmd() {
  printf "%q " "$@"
}

nemotron3_docker_env_args() {
  NEMO3_DOCKER_ENV_ARGS=(
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    -e HF_HOME=/root/.cache/huggingface
    -e HF_XET_HIGH_PERFORMANCE=1
    -e TOKENIZERS_PARALLELISM=false
    -e SAFETENSORS_FAST_GPU=1
    -e NVIDIA_TF32_OVERRIDE=1
    -e SGLANG_DISABLE_DEEP_GEMM="${DISABLE_DEEP_GEMM}"
    -e PYTHONUNBUFFERED=1
  )
  if [[ -n "${HF_TOKEN:-}" ]]; then
    NEMO3_DOCKER_ENV_ARGS+=(-e HF_TOKEN="${HF_TOKEN}")
  fi
}

nemotron3_server_args() {
  NEMO3_SERVER_ARGS=(
    python3 -m sglang.launch_server
    --model-path /model
    --host 0.0.0.0
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tp-size "${TP_SIZE}"
    --ep-size "${EP_SIZE}"
    --context-length "${CONTEXT_LENGTH}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --fp8-gemm-backend "${FP8_GEMM_BACKEND}"
    --moe-runner-backend "${MOE_RUNNER_BACKEND}"
    --mamba-scheduler-strategy "${MAMBA_SCHEDULER_STRATEGY}"
    --reasoning-parser nemotron_3
    --tool-call-parser qwen3_coder
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --trust-remote-code
    --log-level info
  )
  if [[ "${DISABLE_PIECEWISE_CUDA_GRAPH}" == "1" ]]; then
    NEMO3_SERVER_ARGS+=(--disable-piecewise-cuda-graph)
  fi
  if [[ "${ENABLE_MTP}" == "1" ]]; then
    NEMO3_SERVER_ARGS+=(
      --speculative-algorithm EAGLE
      --speculative-num-steps 5
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 5
    )
  fi
  if [[ -n "${FP4_GEMM_BACKEND}" ]]; then
    NEMO3_SERVER_ARGS+=(--fp4-gemm-backend "${FP4_GEMM_BACKEND}")
  fi
  if [[ -n "${EXTRA_SGLANG_ARGS}" ]]; then
    # Intentionally split by shell here so callers can pass multiple flags.
    # shellcheck disable=SC2206
    local extra=( ${EXTRA_SGLANG_ARGS} )
    NEMO3_SERVER_ARGS+=("${extra[@]}")
  fi
}
