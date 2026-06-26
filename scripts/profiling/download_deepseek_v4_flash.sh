#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
MODEL_REPO="${MODEL_REPO:-deepseek-ai/DeepSeek-V4-Flash}"
MODEL_PATH="${MODEL_PATH:-/root/models/DeepSeek-V4-Flash-FP4}"
TOKEN_FILE="${TOKEN_FILE:-${HOME}/token}"

if [[ -z "${HF_TOKEN:-}" && -r "${TOKEN_FILE}" ]]; then
  HF_TOKEN="$(tr -d '\n' < "${TOKEN_FILE}")"
  export HF_TOKEN
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN or place the token in ${TOKEN_FILE}" >&2
  exit 2
fi

mkdir -p /root/models /root/.cache/huggingface

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
