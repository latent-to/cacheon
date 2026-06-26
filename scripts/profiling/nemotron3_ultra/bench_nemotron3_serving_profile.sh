#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=nemotron3_common.sh
source "${SCRIPT_DIR}/nemotron3_common.sh"

CONTAINER="${CONTAINER:-$(cat "${NEMO3_PROFILE_ROOT}/last_container_name.txt")}"
CAPTURE="${CAPTURE:-decode}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BENCH_NAME="${BENCH_NAME:-nemotron3_${CAPTURE}_bench_${STAMP}}"

NUM_PROMPTS="${NUM_PROMPTS:-16}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-8192}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"

case "${CAPTURE}" in
  prefill)
    RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1}"
    PROFILE_START_STEP="${PROFILE_START_STEP:-0}"
    PROFILE_STEPS="${PROFILE_STEPS:-8}"
    ;;
  decode|full)
    RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-512}"
    PROFILE_START_STEP="${PROFILE_START_STEP:-64}"
    PROFILE_STEPS="${PROFILE_STEPS:-128}"
    ;;
  *)
    echo "CAPTURE must be prefill, decode, or full; got ${CAPTURE}" >&2
    exit 2
    ;;
esac

docker exec "${CONTAINER}" bash -lc "
set -euo pipefail
until curl -fsS http://127.0.0.1:${PORT}/v1/models >/dev/null; do
  sleep 5
done
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port ${PORT} \
  --dataset-name random \
  --model ${SERVED_MODEL_NAME} \
  --num-prompts ${NUM_PROMPTS} \
  --random-input-len ${RANDOM_INPUT_LEN} \
  --random-output-len ${RANDOM_OUTPUT_LEN} \
  --random-range-ratio 0 \
  --request-rate inf \
  --max-concurrency ${MAX_CONCURRENCY} \
  --output-file /opt/nemotron3_profile/results/${BENCH_NAME}.json \
  --disable-tqdm \
  --profile \
  --profile-activities CUDA_PROFILER \
  --profile-start-step ${PROFILE_START_STEP} \
  --profile-steps ${PROFILE_STEPS} \
  --warmup-requests ${WARMUP_REQUESTS}
"

echo "Bench output: ${RESULTS_DIR}/${BENCH_NAME}.json"
