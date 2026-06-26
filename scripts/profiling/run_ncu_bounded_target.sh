#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/opt/optima_profile}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results}"
LOGS_DIR="${LOGS_DIR:-${ROOT_DIR}/logs}"
TARGET="${TARGET:?TARGET must point to a Python profiling target}"
NAME="${NAME:-ncu_bounded_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ncu \
  --force-overwrite \
  --target-processes all \
  --profile-from-start off \
  --set roofline \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section LaunchStats \
  --export "${RESULTS_DIR}/${NAME}" \
  python3 "${TARGET}" "$@" \
  2>&1 | tee "${LOGS_DIR}/${NAME}.log"

REPORT="${RESULTS_DIR}/${NAME}.ncu-rep"
ncu --import "${REPORT}" --page details > "${RESULTS_DIR}/${NAME}.details.txt"
ncu --import "${REPORT}" --page raw --csv > "${RESULTS_DIR}/${NAME}.raw.csv"
ncu --import "${REPORT}" --page source > "${RESULTS_DIR}/${NAME}.source.txt" || true
ncu --import "${REPORT}" --page session > "${RESULTS_DIR}/${NAME}.session.txt"

echo "NCU report: ${REPORT}"
echo "NCU details: ${RESULTS_DIR}/${NAME}.details.txt"
echo "NCU raw CSV: ${RESULTS_DIR}/${NAME}.raw.csv"
