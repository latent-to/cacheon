#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=qwen35_common.sh
source "${SCRIPT_DIR}/qwen35_common.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
PHASE="${PHASE:-prefill}" # prefill, decode, imports
NAME="${NAME:-qwen35_gdn_micro_${PHASE}_${STAMP}}"

GDN_QK_HEADS="${GDN_QK_HEADS:-8}"
GDN_V_HEADS="${GDN_V_HEADS:-32}"
GDN_HEAD_DIM="${GDN_HEAD_DIM:-128}"
GDN_VALUE_DIM="${GDN_VALUE_DIM:-128}"
GDN_DTYPE="${GDN_DTYPE:-bfloat16}"

PREFILL_BATCH="${PREFILL_BATCH:-1}"
PREFILL_SEQ_LEN="${PREFILL_SEQ_LEN:-16384}"
PREFILL_LAYOUT="${PREFILL_LAYOUT:-varlen}"
PREFILL_INITIAL_STATE="${PREFILL_INITIAL_STATE:-zero}"
PREFILL_ITERS="${PREFILL_ITERS:-1}"

DECODE_BATCH="${DECODE_BATCH:-32}"
DECODE_TOKENS="${DECODE_TOKENS:-1}"
DECODE_STATE_SLOTS="${DECODE_STATE_SLOTS:-${DECODE_BATCH}}"
DECODE_ITERS="${DECODE_ITERS:-2}"

MICRO_WARMUP="${MICRO_WARMUP:-3}"
MICRO_SEED="${MICRO_SEED:-0}"

NCU_DEVICES="${NCU_DEVICES:-0}"
NCU_REPLAY_MODE="${NCU_REPLAY_MODE:-kernel}"
NCU_CACHE_CONTROL="${NCU_CACHE_CONTROL:-none}"
NCU_CLOCK_CONTROL="${NCU_CLOCK_CONTROL:-none}"
NCU_PROFILE_FROM_START="${NCU_PROFILE_FROM_START:-off}"
NCU_SET="${NCU_SET:-full}"
NCU_SECTIONS="${NCU_SECTIONS:-}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"

case "${PHASE}" in
  prefill)
    NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-chunk_gated_delta_rule|chunk_fwd_o|chunk_scaled_dot_kkt|solve_tril|chunk_local_cumsum|wy_fast}"
    NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-12}"
    MICRO_ARGS=(
      --phase prefill
      --batch "${PREFILL_BATCH}"
      --seq-len "${PREFILL_SEQ_LEN}"
      --layout "${PREFILL_LAYOUT}"
      --initial-state "${PREFILL_INITIAL_STATE}"
      --iters "${PREFILL_ITERS}"
    )
    ;;
  decode)
    NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-fused_recurrent.*gated_delta|packed_decode.*gated_delta|gated_delta.*packed_decode}"
    NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
    MICRO_ARGS=(
      --phase decode
      --batch "${DECODE_BATCH}"
      --decode-tokens "${DECODE_TOKENS}"
      --state-slots "${DECODE_STATE_SLOTS}"
      --iters "${DECODE_ITERS}"
    )
    ;;
  imports)
    NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-.*}"
    NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
    MICRO_ARGS=(--phase imports)
    ;;
  *)
    echo "PHASE must be prefill, decode, or imports; got ${PHASE}" >&2
    exit 2
    ;;
esac

MICRO_ARGS+=(
  --qk-heads "${GDN_QK_HEADS}"
  --v-heads "${GDN_V_HEADS}"
  --head-dim "${GDN_HEAD_DIM}"
  --value-dim "${GDN_VALUE_DIM}"
  --dtype "${GDN_DTYPE}"
  --warmup "${MICRO_WARMUP}"
  --seed "${MICRO_SEED}"
)

qwen35_docker_env_args

NCU_COLLECT_ARGS=()
if [[ -n "${NCU_SECTIONS}" ]]; then
  IFS=',' read -r -a _sections <<< "${NCU_SECTIONS}"
  for section in "${_sections[@]}"; do
    NCU_COLLECT_ARGS+=(--section "${section}")
  done
elif [[ -n "${NCU_SET}" ]]; then
  NCU_COLLECT_ARGS+=(--set "${NCU_SET}")
fi

RUNNER=$(cat <<EOF
set -euo pipefail
mkdir -p /opt/qwen35_profile/results /opt/qwen35_profile/logs
if [[ "${PHASE}" == "imports" ]]; then
  python3 /opt/qwen35_profile/scripts/qwen35_gdn_microbench.py $(qwen35_quote_cmd "${MICRO_ARGS[@]}") \
    | tee "/opt/qwen35_profile/results/${NAME}_imports.json"
  exit 0
fi
ncu --target-processes application-only \
  --force-overwrite \
  --devices "${NCU_DEVICES}" \
  --profile-from-start "${NCU_PROFILE_FROM_START}" \
  --kernel-name-base demangled \
  --kernel-name "regex:${NCU_KERNEL_REGEX}" \
  --launch-skip "${NCU_LAUNCH_SKIP}" \
  --launch-count "${NCU_LAUNCH_COUNT}" \
  --replay-mode "${NCU_REPLAY_MODE}" \
  --cache-control "${NCU_CACHE_CONTROL}" \
  --clock-control "${NCU_CLOCK_CONTROL}" \
  $(qwen35_quote_cmd "${NCU_COLLECT_ARGS[@]}") \
  --export "/opt/qwen35_profile/results/${NAME}" \
  python3 /opt/qwen35_profile/scripts/qwen35_gdn_microbench.py $(qwen35_quote_cmd "${MICRO_ARGS[@]}")
REPORT="/opt/qwen35_profile/results/${NAME}.ncu-rep"
ncu --import "\${REPORT}" --csv --page raw > "/opt/qwen35_profile/results/${NAME}_raw.csv"
ncu --import "\${REPORT}" --page details --print-metric-name label-name > "/opt/qwen35_profile/results/${NAME}_details.txt"
ncu --import "\${REPORT}" --page source > "/opt/qwen35_profile/results/${NAME}_source.txt" || true
ncu --import "\${REPORT}" --page session > "/opt/qwen35_profile/results/${NAME}_session.txt"
EOF
)

docker run --rm \
  --gpus all \
  --privileged \
  --network=host \
  --ipc=host \
  --shm-size 32g \
  "${QWEN35_DOCKER_ENV_ARGS[@]}" \
  -v "${QWEN35_PROFILE_ROOT}:/opt/qwen35_profile" \
  "${IMAGE}" \
  bash -lc "${RUNNER}" 2>&1 | tee "${LOGS_DIR}/${NAME}.log"

if [[ "${PHASE}" != "imports" ]]; then
  echo "NCU report: ${RESULTS_DIR}/${NAME}.ncu-rep"
  echo "NCU raw CSV: ${RESULTS_DIR}/${NAME}_raw.csv"
  echo "NCU details: ${RESULTS_DIR}/${NAME}_details.txt"
fi
