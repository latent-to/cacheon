#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=nemotron3_common.sh
source "${SCRIPT_DIR}/nemotron3_common.sh"

CONTAINER="${CONTAINER:-$(cat "${NEMO3_PROFILE_ROOT}/last_container_name.txt")}"
RESULT_NAME="${RESULT_NAME:-$(cat "${NEMO3_PROFILE_ROOT}/last_result_name.txt")}"

docker stop "${CONTAINER}" >/dev/null || true

REP="${RESULTS_DIR}/${RESULT_NAME}.nsys-rep"
if [[ ! -f "${REP}" ]]; then
  echo "Missing ${REP}; wait a few seconds for nsys to flush or check docker logs." >&2
  exit 1
fi

nsys export --type sqlite --force-overwrite=true "${REP}" -o "${RESULTS_DIR}/${RESULT_NAME}.sqlite" || true
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "${REP}" > "${RESULTS_DIR}/${RESULT_NAME}_cuda_gpu_kern_sum.csv" || true
nsys stats --force-export=true --report cuda_api_sum --format csv "${REP}" > "${RESULTS_DIR}/${RESULT_NAME}_cuda_api_sum.csv" || true
nsys stats --force-export=true --report cuda_gpu_trace --format csv "${REP}" > "${RESULTS_DIR}/${RESULT_NAME}_cuda_gpu_trace.csv" || true
sed -n '1,400p' "${RESULTS_DIR}/${RESULT_NAME}_cuda_gpu_trace.csv" > "${RESULTS_DIR}/${RESULT_NAME}_cuda_gpu_trace_head400.csv" || true

echo "Exported NSYS artifacts for ${RESULT_NAME}"
