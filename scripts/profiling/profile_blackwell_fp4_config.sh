#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-megamoe_w4a4}"
TRACE="${TRACE:-nsys}" # nsys or ncu
WORKLOAD="${WORKLOAD:-short}" # short or long
SGLANG_MODE="${SGLANG_MODE:-${MODE:-cudagraph}}" # cudagraph or nograph_nvtx
PROFILE_LABEL="${PROFILE_LABEL:-}"
PORT="${PORT:-30000}"
GPU_IDS="${GPU_IDS:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
MODEL_PATH="${MODEL_PATH:-/root/models/DeepSeek-V4-Flash-FP4}"
IMAGE="${IMAGE:-lmsysorg/sglang:dev-cu13}"
PROFILE_ROOT="${PROFILE_ROOT:-/root/optima_profile}"
STAMP="$(date +%Y%m%d_%H%M%S)"

SHORT_NUM_PROMPTS="${SHORT_NUM_PROMPTS:-128}"
SHORT_CONCURRENCY="${SHORT_CONCURRENCY:-128}"
SHORT_INPUT_LEN="${SHORT_INPUT_LEN:-1024}"
SHORT_OUTPUT_LEN="${SHORT_OUTPUT_LEN:-512}"
LONG_NUM_REQUESTS="${LONG_NUM_REQUESTS:-128}"
LONG_PROMPT_WORDS="${LONG_PROMPT_WORDS:-7000}"
LONG_OUTPUT_TOKENS="${LONG_OUTPUT_TOKENS:-4096}"
LONG_SETTLE_S="${LONG_SETTLE_S:-60}"
LONG_CAPTURE_S="${LONG_CAPTURE_S:-30}"

USE_MEGAMOE=0
MEGAMOE_W4A4=1
MOE_A2A_BACKEND=megamoe
MOE_RUNNER_BACKEND=flashinfer_mxfp4
FP4_GEMM_BACKEND=auto
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:-}"

case "${CONFIG}" in
  megamoe_w4a4)
    USE_MEGAMOE=1
    MOE_A2A_BACKEND=megamoe
    MEGAMOE_W4A4=1
    FP4_GEMM_BACKEND=auto
    ;;
  megamoe_w4a4_cutlass_gemm)
    USE_MEGAMOE=1
    MOE_A2A_BACKEND=megamoe
    MEGAMOE_W4A4=1
    FP4_GEMM_BACKEND=cutlass
    ;;
  flashinfer_mxfp4)
    USE_MEGAMOE=0
    MOE_RUNNER_BACKEND=flashinfer_mxfp4
    FP4_GEMM_BACKEND=auto
    ;;
  flashinfer_mxfp4_bf16)
    USE_MEGAMOE=0
    MOE_RUNNER_BACKEND=flashinfer_mxfp4
    FP4_GEMM_BACKEND=auto
    EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS} --flashinfer-mxfp4-moe-precision bf16"
    ;;
  flashinfer_cutlass)
    USE_MEGAMOE=0
    MOE_RUNNER_BACKEND=flashinfer_cutlass
    FP4_GEMM_BACKEND=flashinfer_cutlass
    ;;
  marlin)
    USE_MEGAMOE=0
    MOE_RUNNER_BACKEND=marlin
    FP4_GEMM_BACKEND=auto
    ;;
  *)
    echo "Unknown CONFIG=${CONFIG}" >&2
    exit 2
    ;;
esac

mkdir -p "${PROFILE_ROOT}/results" "${PROFILE_ROOT}/logs"

LABEL_PART=""
if [[ -n "${PROFILE_LABEL}" ]]; then
  LABEL_PART="_${PROFILE_LABEL}"
fi
NAME="optima_${TRACE}_${CONFIG}_${SGLANG_MODE}_${WORKLOAD}${LABEL_PART}_${STAMP}"
RESULT_NAME="${TRACE}_${CONFIG}_${SGLANG_MODE}_${WORKLOAD}${LABEL_PART}_${STAMP}"
docker rm -f "${NAME}" >/dev/null 2>&1 || true

if [[ "${TRACE}" == "nsys" ]]; then
  LAUNCH_SCRIPT="${PROFILE_ROOT}/scripts/profiling/run_sglang_blackwell_fp4_nsys.sh"
elif [[ "${TRACE}" == "ncu" ]]; then
  LAUNCH_SCRIPT="${PROFILE_ROOT}/scripts/profiling/run_sglang_blackwell_fp4_ncu.sh"
else
  echo "TRACE must be nsys or ncu, got ${TRACE}" >&2
  exit 2
fi

echo "PROFILE_CONFIG_BEGIN trace=${TRACE} config=${CONFIG} mode=${SGLANG_MODE} workload=${WORKLOAD} name=${NAME}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
TP_SIZE="${TP_SIZE}" \
PORT="${PORT}" \
MODE="${SGLANG_MODE}" \
MODEL_PATH="${MODEL_PATH}" \
IMAGE="${IMAGE}" \
NAME="${NAME}" \
RESULT_NAME="${RESULT_NAME}" \
USE_MEGAMOE="${USE_MEGAMOE}" \
MEGAMOE_W4A4="${MEGAMOE_W4A4}" \
MOE_A2A_BACKEND="${MOE_A2A_BACKEND}" \
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND}" \
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND}" \
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS}" \
"${LAUNCH_SCRIPT}"

deadline=$((SECONDS + 1800))
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do
  if (( SECONDS > deadline )); then
    docker logs "${NAME}" > "${PROFILE_ROOT}/logs/${NAME}.docker.log" 2>&1 || true
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
    echo "SERVER_READY_TIMEOUT name=${NAME}" >&2
    exit 1
  fi
  sleep 10
done
echo "SERVER_READY name=${NAME}"

if [[ "${WORKLOAD}" == "short" ]]; then
  CONTAINER="${NAME}" \
  PORT="${PORT}" \
  MODEL_PATH="${MODEL_PATH}" \
  RESULT_NAME="bench_${CONFIG}_${TRACE}_${STAMP}.json" \
  NUM_PROMPTS="${SHORT_NUM_PROMPTS}" \
  MAX_CONCURRENCY="${SHORT_CONCURRENCY}" \
  RANDOM_INPUT_LEN="${SHORT_INPUT_LEN}" \
  RANDOM_OUTPUT_LEN="${SHORT_OUTPUT_LEN}" \
  PROFILE_START_STEP="${PROFILE_START_STEP:-20}" \
  PROFILE_STEPS="${PROFILE_STEPS:-40}" \
  "${PROFILE_ROOT}/scripts/profiling/bench_sglang_profile.sh"
elif [[ "${WORKLOAD}" == "long" ]]; then
  docker exec "${NAME}" python3 /opt/optima_profile/scripts/profiling/steady_decode_profile.py \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "${MODEL_PATH}" \
    --num-requests "${LONG_NUM_REQUESTS}" \
    --max-workers "${LONG_NUM_REQUESTS}" \
    --prompt-words "${LONG_PROMPT_WORDS}" \
    --output-tokens "${LONG_OUTPUT_TOKENS}" \
    --settle-s "${LONG_SETTLE_S}" \
    --capture-s "${LONG_CAPTURE_S}" \
    --json-out "/opt/optima_profile/results/steady_decode_${CONFIG}_${STAMP}.json"
else
  echo "WORKLOAD must be short or long, got ${WORKLOAD}" >&2
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  exit 2
fi

docker stop -t 120 "${NAME}" >/dev/null || true
docker logs "${NAME}" > "${PROFILE_ROOT}/logs/${NAME}.docker.log" 2>&1 || true
docker rm "${NAME}" >/dev/null 2>&1 || true
echo "PROFILE_CONFIG_DONE trace=${TRACE} config=${CONFIG} mode=${SGLANG_MODE} workload=${WORKLOAD} result=${RESULT_NAME}"
