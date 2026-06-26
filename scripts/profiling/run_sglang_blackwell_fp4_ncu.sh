#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
PROFILE_MODE="${MODE:-cudagraph}"
export IMAGE="${IMAGE:-lmsysorg/sglang:dev-cu13}"
export NAME="${NAME:-optima_sglang_blackwell_fp4_ncu_${PROFILE_MODE}_${STAMP}}"
export RESULT_NAME="${RESULT_NAME:-ncu_serving_blackwell_fp4_${PROFILE_MODE}_${STAMP}}"
exec "${SCRIPT_DIR}/run_sglang_b200_fp4_ncu.sh" "$@"
