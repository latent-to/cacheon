#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/opt/optima_profile}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results}"
LOGS_DIR="${LOGS_DIR:-${ROOT_DIR}/logs}"
TARGET="${TARGET:-${ROOT_DIR}/scripts/profiling/ncu_nvfp4_groupmm_target.py}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
NAME="${NAME:-ncu_nvfp4_groupmm_dsv4_w13_roofline_${STAMP}}"

TOTAL_TOKENS="${TOTAL_TOKENS:-896}"
N_DIM="${N_DIM:-4096}"
K_DIM="${K_DIM:-4096}"
NUM_EXPERTS="${NUM_EXPERTS:-257}"
WARMUP="${WARMUP:-3}"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS="${SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS:-1}"
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND="${SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND:-1}"

ncu \
  --force-overwrite \
  --target-processes all \
  --profile-from-start off \
  --set roofline \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section LaunchStats \
  --export "${RESULTS_DIR}/${NAME}" \
  python3 "${TARGET}" \
    --total-tokens "${TOTAL_TOKENS}" \
    --n "${N_DIM}" \
    --k "${K_DIM}" \
    --num-experts "${NUM_EXPERTS}" \
    --warmup "${WARMUP}" \
  2>&1 | tee "${LOGS_DIR}/${NAME}.log"

REPORT="${RESULTS_DIR}/${NAME}.ncu-rep"
ncu --import "${REPORT}" --page details > "${RESULTS_DIR}/${NAME}.details.txt"
ncu --import "${REPORT}" --page raw --csv > "${RESULTS_DIR}/${NAME}.raw.csv"
ncu --import "${REPORT}" --page source > "${RESULTS_DIR}/${NAME}.source.txt" || true
ncu --import "${REPORT}" --page session > "${RESULTS_DIR}/${NAME}.session.txt"

echo "NCU report: ${REPORT}"
echo "NCU details: ${RESULTS_DIR}/${NAME}.details.txt"
echo "NCU raw CSV: ${RESULTS_DIR}/${NAME}.raw.csv"
