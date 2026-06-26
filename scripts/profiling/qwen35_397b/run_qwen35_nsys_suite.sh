#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Headline e2e captures requested by the user.
CAPTURE=full "${SCRIPT_DIR}/run_qwen35_nsys.sh" eager
CAPTURE=full "${SCRIPT_DIR}/run_qwen35_nsys.sh" cudagraph
CAPTURE=full "${SCRIPT_DIR}/run_qwen35_nsys.sh" big

# Optional steady decode captures. Enable by setting RUN_STEADY_DECODE=1.
if [[ "${RUN_STEADY_DECODE:-0}" == "1" ]]; then
  CAPTURE=decode PROFILE_START_STEP="${PROFILE_START_STEP:-192}" PROFILE_STEPS="${PROFILE_STEPS:-96}" \
    "${SCRIPT_DIR}/run_qwen35_nsys.sh" eager
  CAPTURE=decode PROFILE_START_STEP="${PROFILE_START_STEP:-192}" PROFILE_STEPS="${PROFILE_STEPS:-96}" \
    "${SCRIPT_DIR}/run_qwen35_nsys.sh" cudagraph
fi
