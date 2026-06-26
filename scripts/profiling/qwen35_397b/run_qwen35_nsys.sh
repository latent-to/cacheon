#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=qwen35_common.sh
source "${SCRIPT_DIR}/qwen35_common.sh"

MODE="${1:-${MODE:-eager}}"
CAPTURE="${CAPTURE:-full}" # full, prefill, decode
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="${NAME:-qwen35_${MODE}_${CAPTURE}_${STAMP}}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
PROFILE_START_STEP="${PROFILE_START_STEP:-128}"
PROFILE_STEPS="${PROFILE_STEPS:-96}"

qwen35_model_args
qwen35_mode_args "${MODE}"
qwen35_docker_env_args

BENCH_CMD=(python3 -m sglang.bench_one_batch "${QWEN35_MODEL_ARGS[@]}" "${QWEN35_MODE_ARGS[@]}"
  --run-name "${NAME}"
  --result-filename "/opt/qwen35_profile/results/${NAME}.jsonl")

NSYS_ARGS=(nsys profile
  --force-overwrite=true
  --trace="${NSYS_TRACE}"
  --trace-fork-before-exec=true
  --cuda-graph-trace=node
  --sample=none
  --cpuctxsw=none
  -o "/opt/qwen35_profile/results/${NAME}")

case "${CAPTURE}" in
  full)
    ;;
  prefill)
    NSYS_ARGS+=(--capture-range=cudaProfilerApi --capture-range-end=stop)
    BENCH_CMD+=(--profile --profile-activities CUDA_PROFILER --profile-stage prefill
      --profile-filename-prefix "/opt/qwen35_profile/results/${NAME}_torch")
    ;;
  decode)
    NSYS_ARGS+=(--capture-range=cudaProfilerApi --capture-range-end=stop)
    BENCH_CMD+=(--profile --profile-activities CUDA_PROFILER --profile-stage decode
      --profile-start-step "${PROFILE_START_STEP}"
      --profile-steps "${PROFILE_STEPS}"
      --profile-filename-prefix "/opt/qwen35_profile/results/${NAME}_torch")
    ;;
  *)
    echo "CAPTURE must be full, prefill, or decode; got ${CAPTURE}" >&2
    exit 2
    ;;
esac

RUNNER=$(cat <<EOF
set -euo pipefail
mkdir -p /opt/qwen35_profile/results /opt/qwen35_profile/logs
echo "RUN_NAME=${NAME}"
echo "MODE=${MODE}"
echo "CAPTURE=${CAPTURE}"
echo "IMAGE=${IMAGE}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}"
$(qwen35_quote_cmd "${NSYS_ARGS[@]}" "${BENCH_CMD[@]}")
REP="/opt/qwen35_profile/results/${NAME}.nsys-rep"
nsys export --type sqlite --force-overwrite=true "\${REP}" -o "/opt/qwen35_profile/results/${NAME}.sqlite" || true
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "\${REP}" > "/opt/qwen35_profile/results/${NAME}_cuda_gpu_kern_sum.csv" || true
nsys stats --force-export=true --report cuda_api_sum --format csv "\${REP}" > "/opt/qwen35_profile/results/${NAME}_cuda_api_sum.csv" || true
nsys stats --force-export=true --report cuda_gpu_trace --format csv "\${REP}" > "/opt/qwen35_profile/results/${NAME}_cuda_gpu_trace.csv" || true
sed -n '1,400p' "/opt/qwen35_profile/results/${NAME}_cuda_gpu_trace.csv" > "/opt/qwen35_profile/results/${NAME}_cuda_gpu_trace_head400.csv" || true
EOF
)

docker run --rm \
  --gpus all \
  --privileged \
  --network=host \
  --ipc=host \
  --shm-size 128g \
  "${QWEN35_DOCKER_ENV_ARGS[@]}" \
  -v /root/models:/root/models \
  -v /root/.cache:/root/.cache \
  -v "${QWEN35_PROFILE_ROOT}:/opt/qwen35_profile" \
  "${IMAGE}" \
  bash -lc "${RUNNER}" 2>&1 | tee "${LOGS_DIR}/${NAME}.log"

echo "NSYS report: ${RESULTS_DIR}/${NAME}.nsys-rep"
echo "Kernel CSV: ${RESULTS_DIR}/${NAME}_cuda_gpu_kern_sum.csv"
echo "API CSV: ${RESULTS_DIR}/${NAME}_cuda_api_sum.csv"
