#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:?set CONTAINER to the running SGLang container name}"
PORT="${PORT:-30000}"
MODEL_PATH="${MODEL_PATH:-/root/models/DeepSeek-V4-Flash-FP4}"
RESULT_NAME="${RESULT_NAME:-bench_profile_$(date +%Y%m%d_%H%M%S).json}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1024}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-512}"
PROFILE_START_STEP="${PROFILE_START_STEP:-20}"
PROFILE_STEPS="${PROFILE_STEPS:-40}"

docker exec "${CONTAINER}" bash -lc "
set -euo pipefail
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port ${PORT} \
  --dataset-name random \
  --model ${MODEL_PATH} \
  --num-prompts ${NUM_PROMPTS} \
  --random-input-len ${RANDOM_INPUT_LEN} \
  --random-output-len ${RANDOM_OUTPUT_LEN} \
  --random-range-ratio 0 \
  --request-rate inf \
  --max-concurrency ${MAX_CONCURRENCY} \
  --output-file /opt/optima_profile/results/${RESULT_NAME} \
  --disable-tqdm \
  --profile \
  --profile-activities CUDA_PROFILER \
  --profile-start-step ${PROFILE_START_STEP} \
  --profile-steps ${PROFILE_STEPS} \
  --warmup-requests 1
"
