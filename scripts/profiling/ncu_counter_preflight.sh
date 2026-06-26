#!/usr/bin/env bash
set -euo pipefail

# Fail-fast test for Nsight Compute hardware counter access.
# Run this before downloading models or starting SGLang on a new pod.

IMAGE="${IMAGE:-lmsysorg/sglang:latest}"
GPU_DEVICE="${GPU_DEVICE:-0}"
MODE="${MODE:-cap_sys_admin}"
OUT="${OUT:-/tmp/ncu_counter_preflight}"

case "${MODE}" in
  cap_sys_admin)
    DOCKER_PRIVS=(--cap-add=SYS_ADMIN)
    ;;
  privileged)
    DOCKER_PRIVS=(--privileged)
    ;;
  none)
    DOCKER_PRIVS=()
    ;;
  *)
    echo "MODE must be cap_sys_admin, privileged, or none; got ${MODE}" >&2
    exit 2
    ;;
esac

echo "== host =="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits || true
if [[ -r /proc/driver/nvidia/params ]]; then
  grep -E "RmProfilingAdminOnly|RestrictProfiling" /proc/driver/nvidia/params || true
fi

echo "== ncu tiny matmul (${MODE}) =="
set +e
docker run --rm \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  "${DOCKER_PRIVS[@]}" \
  "${IMAGE}" \
  bash -lc "set -euo pipefail
    NCU=\$(command -v ncu || command -v /usr/local/cuda/bin/ncu)
    \${NCU} --target-processes all \
      --set roofline \
      --launch-count 1 \
      --kernel-name-base demangled \
      --export ${OUT} \
      --force-overwrite \
      python3 -c 'import torch; x=torch.randn((1024,1024),device=\"cuda\",dtype=torch.float16); y=torch.randn((1024,1024),device=\"cuda\",dtype=torch.float16); torch.cuda.synchronize(); z=x@y; torch.cuda.synchronize(); print(float(z[0,0]))'
  "
rc=$?
set -e

if [[ "${rc}" -eq 0 ]]; then
  echo "PASS_NCU_COUNTERS mode=${MODE} image=${IMAGE}"
  exit 0
fi

echo "FAIL_NCU_COUNTERS mode=${MODE} image=${IMAGE} rc=${rc}"
echo "If cap_sys_admin and privileged both fail, do not download models on this pod."
echo "Ask for host NVIDIA performance counters enabled: NVreg_RestrictProfilingToAdminUsers=0 / RmProfilingAdminOnly=0."
exit "${rc}"
