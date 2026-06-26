#!/usr/bin/env bash
set -euo pipefail

PROFILE_ROOT="${PROFILE_ROOT:-/root/optima_profile}"
PORT="${PORT:-30000}"
CONFIGS="${CONFIGS:-megamoe_w4a4 flashinfer_mxfp4 flashinfer_mxfp4_bf16 marlin flashinfer_cutlass}"
LONG_CONFIGS="${LONG_CONFIGS:-megamoe_w4a4 flashinfer_mxfp4}"

mkdir -p "${PROFILE_ROOT}/results" "${PROFILE_ROOT}/logs"
SUMMARY="${PROFILE_ROOT}/results/fp4_sweep_$(date +%Y%m%d_%H%M%S).txt"

run_one() {
  local trace="$1"
  local config="$2"
  local mode="$3"
  local workload="$4"
  echo "RUN trace=${trace} config=${config} mode=${mode} workload=${workload}" | tee -a "${SUMMARY}"
  set +e
  TRACE="${trace}" CONFIG="${config}" SGLANG_MODE="${mode}" WORKLOAD="${workload}" PORT="${PORT}" \
    "${PROFILE_ROOT}/scripts/profiling/profile_blackwell_fp4_config.sh" 2>&1 \
    | tee -a "${PROFILE_ROOT}/logs/${trace}_${config}_${mode}_${workload}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "DONE trace=${trace} config=${config} mode=${mode} workload=${workload} rc=${rc}" | tee -a "${SUMMARY}"
}

for config in ${CONFIGS}; do
  run_one nsys "${config}" cudagraph short
  run_one nsys "${config}" nograph_nvtx short
  run_one ncu "${config}" nograph_nvtx short
done

for config in ${LONG_CONFIGS}; do
  run_one nsys "${config}" cudagraph long
  run_one nsys "${config}" nograph_nvtx long
done

echo "SWEEP_SUMMARY ${SUMMARY}"
