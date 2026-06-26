#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-cudagraph}"
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="${NAME:-optima_sglang_h200_fp8_mt_nsys_${MODE}_${STAMP}}"
PORT="${PORT:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_PATH="${MODEL_PATH:-/root/models/DeepSeek-V4-Flash-FP8}"
IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
RESULT_NAME="${RESULT_NAME:-nsys_serving_h200_fp8_mt_${MODE}_${STAMP}}"

case "${MODE}" in
  cudagraph)
    SGLANG_PROFILE_ARGS=()
    ;;
  nograph_nvtx)
    SGLANG_PROFILE_ARGS=(--disable-cuda-graph --enable-layerwise-nvtx-marker)
    ;;
  *)
    echo "MODE must be cudagraph or nograph_nvtx, got: ${MODE}" >&2
    exit 2
    ;;
esac

docker run -d \
  --name "${NAME}" \
  --gpus all \
  --privileged \
  --network=host \
  --ipc=host \
  --shm-size 64g \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -e CUDA_HOME=/usr/local/cuda \
  -e HF_HOME=/root/.cache/huggingface \
  -e SGLANG_DSV4_FP4_EXPERTS=0 \
  -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
  -e RESULT_NAME="${RESULT_NAME}" \
  -v /root/models:/root/models \
  -v /root/.cache:/root/.cache \
  -v /root/optima_profile:/opt/optima_profile \
  "${IMAGE}" \
  bash -lc 'set -euo pipefail
mkdir -p /opt/optima_profile/results
exec nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --cpuctxsw=none \
  -o "/opt/optima_profile/results/${RESULT_NAME}" \
  sglang serve \
    --trust-remote-code \
    --model-path '"${MODEL_PATH}"' \
    --tp 4 \
    --dp 4 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --cuda-graph-max-bs 128 \
    --max-running-requests 256 \
    --deepep-config '"'"'{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'"'"' \
    --host 0.0.0.0 \
    --port '"${PORT}"' \
    '"${SGLANG_PROFILE_ARGS[*]}"''
