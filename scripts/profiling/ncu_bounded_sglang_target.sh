#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-30000}"
MODEL_PATH="${MODEL_PATH:-/root/models/DeepSeek-V4-Flash-FP4}"
TP_SIZE="${TP_SIZE:-2}"
USE_MEGAMOE="${USE_MEGAMOE:-1}"
MEGAMOE_W4A4="${MEGAMOE_W4A4:-1}"
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-megamoe}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_mxfp4}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-auto}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-128}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.075}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MODE="${MODE:-nograph_nvtx}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"

RESULT_NAME="${RESULT_NAME:-ncu_bounded_$(date +%Y%m%d_%H%M%S)}"
SHORT_NUM_PROMPTS="${SHORT_NUM_PROMPTS:-16}"
SHORT_CONCURRENCY="${SHORT_CONCURRENCY:-16}"
SHORT_INPUT_LEN="${SHORT_INPUT_LEN:-512}"
SHORT_OUTPUT_LEN="${SHORT_OUTPUT_LEN:-64}"
PROFILE_START_STEP="${PROFILE_START_STEP:-2}"
PROFILE_STEPS="${PROFILE_STEPS:-8}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"

mkdir -p /opt/optima_profile/results /opt/optima_profile/logs
SERVER_LOG="/opt/optima_profile/logs/${RESULT_NAME}.server.log"
BENCH_JSON="/opt/optima_profile/results/bench_${RESULT_NAME}.json"

if [[ "${USE_MEGAMOE}" == "1" ]]; then
  MOE_ARGS=(--moe-a2a-backend "${MOE_A2A_BACKEND}")
else
  MOE_ARGS=(--moe-runner-backend "${MOE_RUNNER_BACKEND}")
fi

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

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

sglang serve \
  --trust-remote-code \
  --model-path "${MODEL_PATH}" \
  --tp "${TP_SIZE}" \
  "${MOE_ARGS[@]}" \
  --fp4-gemm-backend "${FP4_GEMM_BACKEND}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  "${SGLANG_PROFILE_ARGS[@]}" \
  ${EXTRA_SGLANG_ARGS} \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + 1800))
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    cat "${SERVER_LOG}" >&2 || true
    echo "SGLANG_SERVER_EXITED pid=${SERVER_PID}" >&2
    exit 1
  fi
  if (( SECONDS > deadline )); then
    tail -200 "${SERVER_LOG}" >&2 || true
    echo "SGLANG_SERVER_READY_TIMEOUT pid=${SERVER_PID}" >&2
    exit 1
  fi
  sleep 5
done

python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dataset-name random \
  --model "${MODEL_PATH}" \
  --num-prompts "${SHORT_NUM_PROMPTS}" \
  --random-input-len "${SHORT_INPUT_LEN}" \
  --random-output-len "${SHORT_OUTPUT_LEN}" \
  --random-range-ratio 0 \
  --request-rate inf \
  --max-concurrency "${SHORT_CONCURRENCY}" \
  --output-file "${BENCH_JSON}" \
  --disable-tqdm \
  --profile \
  --profile-activities CUDA_PROFILER \
  --profile-start-step "${PROFILE_START_STEP}" \
  --profile-steps "${PROFILE_STEPS}" \
  --warmup-requests "${WARMUP_REQUESTS}"

