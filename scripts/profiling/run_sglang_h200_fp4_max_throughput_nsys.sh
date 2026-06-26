#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-cudagraph}"
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="${NAME:-optima_sglang_h200_fp4_mt_nsys_${MODE}_${STAMP}}"
PORT="${PORT:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_PATH="${MODEL_PATH:-/root/models/DeepSeek-V4-Flash-FP4}"
IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
RESULT_NAME="${RESULT_NAME:-nsys_serving_h200_fp4_mt_${MODE}_${STAMP}}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-marlin}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-128}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"

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
    --moe-runner-backend '"${MOE_RUNNER_BACKEND}"' \
    --cuda-graph-max-bs '"${CUDA_GRAPH_MAX_BS}"' \
    --max-running-requests '"${MAX_RUNNING_REQUESTS}"' \
    --host 0.0.0.0 \
    --port '"${PORT}"' \
    '"${SGLANG_PROFILE_ARGS[*]}"''
