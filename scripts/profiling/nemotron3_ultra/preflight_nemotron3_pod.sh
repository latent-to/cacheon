#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=nemotron3_common.sh
source "${SCRIPT_DIR}/nemotron3_common.sh"

mkdir -p /root/models /root/.cache/huggingface "${RESULTS_DIR}" "${LOGS_DIR}"

{
  date -Is
  hostname
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  grep -E "RmProfilingAdminOnly|RestrictProfiling" /proc/driver/nvidia/params || true
  docker version
} | tee "${LOGS_DIR}/preflight_host_$(date +%Y%m%d_%H%M%S).log"

docker pull "${IMAGE}"

docker run --rm \
  --gpus all \
  --privileged \
  --network=host \
  --ipc=host \
  --shm-size 16g \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  -v "${NEMO3_PROFILE_ROOT}:/opt/nemotron3_profile" \
  "${IMAGE}" \
  bash -lc 'set -euo pipefail
python3 - <<PY
import json, torch
print(json.dumps({
  "torch": torch.__version__,
  "cuda": torch.version.cuda,
  "device_count": torch.cuda.device_count(),
  "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}, indent=2))
PY
python3 -m sglang.launch_server --help | sed -n "1,260p" > /opt/nemotron3_profile/results/launch_server_help.txt
python3 -m sglang.bench_one_batch --help | sed -n "1,260p" > /opt/nemotron3_profile/results/bench_one_batch_help.txt || true
ncu --target-processes application-only --force-overwrite --set speedOfLight --launch-count 1 \
  --export /opt/nemotron3_profile/results/nemotron3_ncu_counter_preflight \
  python3 - <<PY
import torch
x=torch.randn((1024,1024),device="cuda")
y=x @ x
torch.cuda.synchronize()
print(float(y[0,0].detach().cpu()))
PY
ls -lh /opt/nemotron3_profile/results/nemotron3_ncu_counter_preflight.ncu-rep
'

echo "Preflight complete. If NCU did not produce ERR_NVGPUCTRPERM, counters are available."
