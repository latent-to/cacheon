#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=qwen35_common.sh
source "${SCRIPT_DIR}/qwen35_common.sh"

qwen35_require_token

mkdir -p /root/models /root/.cache/huggingface "${RESULTS_DIR}" "${LOGS_DIR}"

docker pull "${IMAGE}"

docker run --rm \
  --network=host \
  -e HF_TOKEN \
  -e HF_XET_HIGH_PERFORMANCE=1 \
  -v /root/models:/root/models \
  -v /root/.cache:/root/.cache \
  "${IMAGE}" \
  bash -lc "set -euo pipefail
if ! command -v uv >/dev/null; then
  if command -v curl >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\"\${HOME}/.local/bin:\${PATH}\"
  else
    echo 'uv is required and curl is unavailable to install it' >&2
    exit 2
  fi
fi
VENV=\"\${HF_DOWNLOAD_VENV:-/tmp/optima_hf_download}\"
uv venv --clear \"\${VENV}\"
uv pip install --python \"\${VENV}/bin/python\" -q -U huggingface_hub hf_xet
\"\${VENV}/bin/hf\" download '${MODEL_REPO}' --local-dir '${MODEL_PATH}'
"

echo "Downloaded ${MODEL_REPO} to ${MODEL_PATH}"
