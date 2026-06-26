#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=qwen35_common.sh
source "${SCRIPT_DIR}/qwen35_common.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="${NAME:-qwen35_gdn_prefill_full_${STAMP}}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-chunk_gated_delta_rule|chunk_fwd_o|chunk_scaled_dot_kkt|solve_tril|chunk_local_cumsum|wy_fast}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-30}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-12}"
NCU_DEVICES="${NCU_DEVICES:-0}"
NCU_REPLAY_MODE="${NCU_REPLAY_MODE:-kernel}"

qwen35_model_args
qwen35_mode_args gdn_prefill_roofline
qwen35_docker_env_args

BENCH_CMD=(python3 -m sglang.bench_one_batch "${QWEN35_MODEL_ARGS[@]}" "${QWEN35_MODE_ARGS[@]}"
  --run-name "${NAME}"
  --result-filename "/opt/qwen35_profile/results/${NAME}.jsonl")

RUNNER=$(cat <<EOF
set -euo pipefail
mkdir -p /opt/qwen35_profile/results /opt/qwen35_profile/logs
ncu --target-processes all \
  --force-overwrite \
  --devices "${NCU_DEVICES}" \
  --kernel-name-base demangled \
  --kernel-name "regex:${NCU_KERNEL_REGEX}" \
  --launch-skip "${NCU_LAUNCH_SKIP}" \
  --launch-count "${NCU_LAUNCH_COUNT}" \
  --replay-mode "${NCU_REPLAY_MODE}" \
  --set full \
  --export "/opt/qwen35_profile/results/${NAME}" \
  $(qwen35_quote_cmd "${BENCH_CMD[@]}")
REPORT="/opt/qwen35_profile/results/${NAME}.ncu-rep"
ncu --import "\${REPORT}" --csv --page raw > "/opt/qwen35_profile/results/${NAME}_raw.csv"
ncu --import "\${REPORT}" --page details > "/opt/qwen35_profile/results/${NAME}_details.txt"
ncu --import "\${REPORT}" --page source > "/opt/qwen35_profile/results/${NAME}_source.txt" || true
ncu --import "\${REPORT}" --page session > "/opt/qwen35_profile/results/${NAME}_session.txt"
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

echo "NCU report: ${RESULTS_DIR}/${NAME}.ncu-rep"
echo "NCU raw CSV: ${RESULTS_DIR}/${NAME}_raw.csv"
echo "NCU details: ${RESULTS_DIR}/${NAME}_details.txt"
