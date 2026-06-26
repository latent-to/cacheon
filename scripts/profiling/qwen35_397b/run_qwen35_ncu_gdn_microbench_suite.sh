#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# First verify imports/signatures in the exact SGLang image.
PHASE=imports NAME=qwen35_gdn_micro_imports "${SCRIPT_DIR}/run_qwen35_ncu_gdn_microbench.sh"

# Core roofline: real 16k prefill token shape, TP=2 per-rank heads.
PHASE=prefill \
  NAME=qwen35_gdn_prefill_micro_b1_t16384_full \
  PREFILL_BATCH="${PREFILL_BATCH:-1}" \
  PREFILL_SEQ_LEN="${PREFILL_SEQ_LEN:-16384}" \
  PREFILL_LAYOUT="${PREFILL_LAYOUT:-varlen}" \
  NCU_LAUNCH_COUNT="${PREFILL_NCU_LAUNCH_COUNT:-12}" \
  "${SCRIPT_DIR}/run_qwen35_ncu_gdn_microbench.sh"

# Core roofline: packed decode kernel at batch 32.
PHASE=decode \
  NAME=qwen35_gdn_decode_micro_b32_t1_full \
  DECODE_BATCH="${DECODE_BATCH:-32}" \
  DECODE_TOKENS="${DECODE_TOKENS:-1}" \
  NCU_LAUNCH_COUNT="${DECODE_NCU_LAUNCH_COUNT:-1}" \
  "${SCRIPT_DIR}/run_qwen35_ncu_gdn_microbench.sh"

if [[ "${RUN_PREFILL_B8:-0}" == "1" ]]; then
  PHASE=prefill \
    NAME=qwen35_gdn_prefill_micro_b8_t16384_full \
    PREFILL_BATCH=8 \
    PREFILL_SEQ_LEN="${PREFILL_SEQ_LEN:-16384}" \
    PREFILL_LAYOUT=varlen \
    NCU_LAUNCH_COUNT="${PREFILL_B8_NCU_LAUNCH_COUNT:-12}" \
    "${SCRIPT_DIR}/run_qwen35_ncu_gdn_microbench.sh"
fi
