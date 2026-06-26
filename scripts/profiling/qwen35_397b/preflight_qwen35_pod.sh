#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=qwen35_common.sh
source "${SCRIPT_DIR}/qwen35_common.sh"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

echo "== host gpu =="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv || true
if [[ -r /proc/driver/nvidia/params ]]; then
  grep -E "RmProfilingAdminOnly|RestrictProfiling" /proc/driver/nvidia/params || true
fi

echo "== docker =="
docker version
docker pull "${IMAGE}"

echo "== image digest and sglang help =="
docker image inspect "${IMAGE}" > "${RESULTS_DIR}/qwen35_image_inspect.json"
docker run --rm \
  --gpus "device=0" \
  --ipc=host \
  --network=host \
  "${IMAGE}" \
  bash -lc "set -euo pipefail
python3 - <<'PY'
import importlib.metadata as md
try:
    print('sglang', md.version('sglang'))
except Exception as exc:
    print('sglang_version_error', exc)
PY
python3 -m sglang.bench_one_batch --help
" | tee "${RESULTS_DIR}/bench_one_batch_help.txt"

echo "== ncu counter preflight, cap_sys_admin =="
set +e
docker run --rm \
  --gpus "device=0" \
  --ipc=host \
  --cap-add=SYS_ADMIN \
  "${IMAGE}" \
  bash -lc "set -euo pipefail
NCU=\$(command -v ncu || command -v /usr/local/cuda/bin/ncu)
\${NCU} --target-processes all --set roofline --launch-count 1 \
  --kernel-name-base demangled --force-overwrite \
  --export /tmp/qwen35_ncu_counter_preflight \
  python3 -c 'import torch; x=torch.randn((1024,1024),device=\"cuda\",dtype=torch.float16); y=x@x; torch.cuda.synchronize(); print(float(y[0,0]))'
"
rc=$?
set -e
if [[ "${rc}" -ne 0 ]]; then
  echo "cap_sys_admin failed; trying privileged"
  docker run --rm \
    --gpus "device=0" \
    --ipc=host \
    --privileged \
    "${IMAGE}" \
    bash -lc "set -euo pipefail
NCU=\$(command -v ncu || command -v /usr/local/cuda/bin/ncu)
\${NCU} --target-processes all --set roofline --launch-count 1 \
  --kernel-name-base demangled --force-overwrite \
  --export /tmp/qwen35_ncu_counter_preflight \
  python3 -c 'import torch; x=torch.randn((1024,1024),device=\"cuda\",dtype=torch.float16); y=x@x; torch.cuda.synchronize(); print(float(y[0,0]))'
"
fi

echo "PASS_QWEN35_PREFLIGHT image=${IMAGE}"
