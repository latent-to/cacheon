#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=nemotron3_common.sh
source "${SCRIPT_DIR}/nemotron3_common.sh"

MODE="${1:-${MODE:-official}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="${NAME:-nemotron3_${MODE}_nsys_${STAMP}}"
RESULT_NAME="${RESULT_NAME:-${NAME}}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn}"

case "${MODE}" in
  official)
    ;;
  fp8_auto)
    DISABLE_DEEP_GEMM=0
    FP8_GEMM_BACKEND=auto
    ;;
  moe_flashinfer)
    MOE_RUNNER_BACKEND=flashinfer_trtllm
    FP4_GEMM_BACKEND=flashinfer_cutlass
    ;;
  mamba_default)
    MAMBA_SCHEDULER_STRATEGY=""
    ;;
  no_mtp)
    ENABLE_MTP=0
    ;;
  *)
    echo "MODE must be official, fp8_auto, moe_flashinfer, mamba_default, or no_mtp; got ${MODE}" >&2
    exit 2
    ;;
esac

nemotron3_require_token
nemotron3_docker_env_args
nemotron3_server_args

if [[ -z "${MAMBA_SCHEDULER_STRATEGY}" ]]; then
  # Remove the empty strategy pair if the default-scheduler A/B is requested.
  filtered=()
  skip_next=0
  for arg in "${NEMO3_SERVER_ARGS[@]}"; do
    if [[ "${skip_next}" == "1" ]]; then
      skip_next=0
      continue
    fi
    if [[ "${arg}" == "--mamba-scheduler-strategy" ]]; then
      skip_next=1
      continue
    fi
    filtered+=("${arg}")
  done
  NEMO3_SERVER_ARGS=("${filtered[@]}")
fi

echo "${NAME}" > "${NEMO3_PROFILE_ROOT}/last_container_name.txt"
echo "${RESULT_NAME}" > "${NEMO3_PROFILE_ROOT}/last_result_name.txt"

docker run -d \
  --name "${NAME}" \
  --gpus all \
  --privileged \
  --network=host \
  --ipc=host \
  --shm-size 128g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${NEMO3_DOCKER_ENV_ARGS[@]}" \
  -v "${MODEL_PATH}:/model:ro" \
  -v /root/.cache:/root/.cache \
  -v "${NEMO3_PROFILE_ROOT}:/opt/nemotron3_profile" \
  "${IMAGE}" \
  bash -lc "$(cat <<EOF
set -euo pipefail
mkdir -p /opt/nemotron3_profile/results /opt/nemotron3_profile/logs
echo "RUN_NAME=${NAME}"
echo "RESULT_NAME=${RESULT_NAME}"
echo "MODE=${MODE}"
echo "IMAGE=${IMAGE}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}"
exec nsys profile \
  --force-overwrite=true \
  --trace=${NSYS_TRACE} \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --cpuctxsw=none \
  -o /opt/nemotron3_profile/results/${RESULT_NAME} \
  $(nemotron3_quote_cmd "${NEMO3_SERVER_ARGS[@]}")
EOF
)"

echo "Started ${NAME}. Wait for http://127.0.0.1:${PORT}/v1/models, then run bench_nemotron3_serving_profile.sh."
