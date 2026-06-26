#!/usr/bin/env bash
set -euo pipefail

# Shared defaults for Qwen3.5-397B-A17B-NVFP4 profiling.
# Override any of these via environment variables on the pod.

QWEN35_PROFILE_ROOT="${QWEN35_PROFILE_ROOT:-/root/qwen35_profile}"
RESULTS_DIR="${RESULTS_DIR:-${QWEN35_PROFILE_ROOT}/results}"
LOGS_DIR="${LOGS_DIR:-${QWEN35_PROFILE_ROOT}/logs}"
MODEL_REPO="${MODEL_REPO:-nvidia/Qwen3.5-397B-A17B-NVFP4}"
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen3.5-397B-A17B-NVFP4}"
IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.12.post1-cu130}"
TOKEN_FILE="${TOKEN_FILE:-${HOME}/token}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TP_SIZE="${TP_SIZE:-2}"

QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-trtllm_mha}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_trtllm}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-flashinfer_cutlass}"
LINEAR_ATTN_PREFILL_BACKEND="${LINEAR_ATTN_PREFILL_BACKEND:-triton}"
LINEAR_ATTN_DECODE_BACKEND="${LINEAR_ATTN_DECODE_BACKEND:-triton}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

qwen35_model_args() {
  QWEN35_MODEL_ARGS=(
    --model-path "${MODEL_PATH}"
    --tp "${TP_SIZE}"
    --quantization "${QUANTIZATION}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --attention-backend "${ATTENTION_BACKEND}"
    --moe-runner-backend "${MOE_RUNNER_BACKEND}"
    --fp4-gemm-backend "${FP4_GEMM_BACKEND}"
    --linear-attn-prefill-backend "${LINEAR_ATTN_PREFILL_BACKEND}"
    --linear-attn-decode-backend "${LINEAR_ATTN_DECODE_BACKEND}"
    --mamba-ssm-dtype "${MAMBA_SSM_DTYPE}"
    --trust-remote-code
  )
}

qwen35_mode_args() {
  local mode="${1:?mode required}"
  case "${mode}" in
    eager)
      QWEN35_MODE_ARGS=(
        --disable-cuda-graph
        --batch-size "${EAGER_BATCH_SIZE:-8}"
        --input-len "${EAGER_INPUT_LEN:-2048}"
        --output-len "${EAGER_OUTPUT_LEN:-512}"
      )
      ;;
    cudagraph)
      QWEN35_MODE_ARGS=(
        --batch-size "${CUDAGRAPH_BATCH_SIZE:-8}"
        --input-len "${CUDAGRAPH_INPUT_LEN:-2048}"
        --output-len "${CUDAGRAPH_OUTPUT_LEN:-512}"
      )
      ;;
    big)
      QWEN35_MODE_ARGS=(
        --batch-size "${BIG_BATCH_SIZE:-128}"
        --input-len "${BIG_INPUT_LEN:-16384}"
        --output-len "${BIG_OUTPUT_LEN:-1024}"
      )
      ;;
    gdn_prefill_roofline)
      QWEN35_MODE_ARGS=(
        --disable-cuda-graph
        --batch-size "${GDN_PREFILL_BATCH_SIZE:-1}"
        --input-len "${GDN_PREFILL_INPUT_LEN:-16384}"
        --output-len "${GDN_PREFILL_OUTPUT_LEN:-1}"
      )
      ;;
    gdn_decode_roofline)
      QWEN35_MODE_ARGS=(
        --disable-cuda-graph
        --batch-size "${GDN_DECODE_BATCH_SIZE:-32}"
        --input-len "${GDN_DECODE_INPUT_LEN:-2048}"
        --output-len "${GDN_DECODE_OUTPUT_LEN:-16}"
      )
      ;;
    *)
      echo "Unknown Qwen3.5 mode: ${mode}" >&2
      exit 2
      ;;
  esac
}

qwen35_quote_cmd() {
  printf "%q " "$@"
}

qwen35_require_token() {
  if [[ -z "${HF_TOKEN:-}" && -r "${TOKEN_FILE}" ]]; then
    HF_TOKEN="$(tr -d '\n' < "${TOKEN_FILE}")"
    export HF_TOKEN
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Set HF_TOKEN or place the token in ${TOKEN_FILE}" >&2
    exit 2
  fi
}

qwen35_docker_env_args() {
  QWEN35_DOCKER_ENV_ARGS=(
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    -e HF_HOME=/root/.cache/huggingface
    -e HF_XET_HIGH_PERFORMANCE=1
    -e TOKENIZERS_PARALLELISM=false
    -e SGLANG_USE_CUDA_IPC_TRANSPORT="${SGLANG_USE_CUDA_IPC_TRANSPORT:-1}"
    -e PYTHONUNBUFFERED=1
  )
  if [[ -n "${HF_TOKEN:-}" ]]; then
    QWEN35_DOCKER_ENV_ARGS+=(-e HF_TOKEN="${HF_TOKEN}")
  fi
}
