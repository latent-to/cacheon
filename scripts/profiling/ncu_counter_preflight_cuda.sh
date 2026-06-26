#!/usr/bin/env bash
set -euo pipefail

# Faster Nsight Compute counter gate for fresh pods.
# Uses a CUDA devel image, compiles a tiny kernel, and profiles one launch.

IMAGE="${IMAGE:-nvidia/cuda:13.0.1-devel-ubuntu24.04}"
GPU_DEVICE="${GPU_DEVICE:-0}"
MODE="${MODE:-cap_sys_admin}"
OUT="${OUT:-/tmp/ncu_counter_preflight_cuda}"

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

echo "== ncu tiny cuda kernel (${MODE}) =="
set +e
docker run --rm \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  "${DOCKER_PRIVS[@]}" \
  "${IMAGE}" \
  bash -lc 'set -euo pipefail
    NCU="$(command -v ncu || command -v /usr/local/cuda/bin/ncu)"
    cat > /tmp/ncu_tiny.cu << "CU"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void saxpy(float *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = a[i];
    float y = b[i];
    #pragma unroll 8
    for (int k = 0; k < 64; ++k) {
      x = x * 1.0001f + y;
    }
    out[i] = x;
  }
}

int main() {
  constexpr int n = 1 << 20;
  float *a = nullptr, *b = nullptr, *out = nullptr;
  cudaMalloc(&a, n * sizeof(float));
  cudaMalloc(&b, n * sizeof(float));
  cudaMalloc(&out, n * sizeof(float));
  cudaMemset(a, 1, n * sizeof(float));
  cudaMemset(b, 2, n * sizeof(float));
  saxpy<<<(n + 255) / 256, 256>>>(out, a, b, n);
  cudaDeviceSynchronize();
  float host = 0.0f;
  cudaMemcpy(&host, out, sizeof(float), cudaMemcpyDeviceToHost);
  std::printf("%f\n", host);
  cudaFree(a);
  cudaFree(b);
  cudaFree(out);
  return 0;
}
CU
    nvcc -O3 -arch=sm_100 /tmp/ncu_tiny.cu -o /tmp/ncu_tiny || nvcc -O3 /tmp/ncu_tiny.cu -o /tmp/ncu_tiny
    "${NCU}" --target-processes all \
      --set roofline \
      --launch-count 1 \
      --kernel-name-base demangled \
      --export "'"${OUT}"'" \
      --force-overwrite \
      /tmp/ncu_tiny
  '
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
