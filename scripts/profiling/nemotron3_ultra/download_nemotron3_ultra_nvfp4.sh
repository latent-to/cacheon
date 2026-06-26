#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=nemotron3_common.sh
source "${SCRIPT_DIR}/nemotron3_common.sh"

nemotron3_require_token
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
python3 - <<PY
import json
from pathlib import Path
p = Path('${MODEL_PATH}') / 'config.json'
if p.exists():
    cfg = json.loads(p.read_text())
    keys = [
        'model_type','num_hidden_layers','hidden_size','num_attention_heads',
        'num_key_value_heads','n_routed_experts','num_experts_per_tok',
        'n_groups','ssm_state_size','conv_kernel','mamba_num_heads',
        'mamba_head_dim','chunk_size','max_position_embeddings',
    ]
    print(json.dumps({k: cfg.get(k) for k in keys}, indent=2, sort_keys=True))
else:
    print('config.json not found')
PY
"

echo "Downloaded ${MODEL_REPO} to ${MODEL_PATH}"
