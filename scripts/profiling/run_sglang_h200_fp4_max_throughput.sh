#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-optima_sglang_h200_fp4_mt_$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_PATH="${MODEL_PATH:-/root/models/DeepSeek-V4-Flash-FP4}"
IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-marlin}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-128}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"

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
  -v /root/models:/root/models \
  -v /root/.cache:/root/.cache \
  -v /root/optima_profile:/opt/optima_profile \
  "${IMAGE}" \
  sglang serve \
    --trust-remote-code \
    --model-path "${MODEL_PATH}" \
    --tp 4 \
    --moe-runner-backend "${MOE_RUNNER_BACKEND}" \
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --host 0.0.0.0 \
    --port "${PORT}"
