// Arm (poison) the rank's symmetric slot region with the Lamport sentinel.
//
// The data-as-flag protocol needs every slot word to start at the sentinel, otherwise a reader
// can mistake stale payload for an arrival. The Python path did this with a full-tensor fill,
// which walks the whole ring through the framework. This does it in one grid-stride kernel on
// the caller's stream, and returns the number of words armed so the caller can prove the whole
// ring was covered rather than trusting a silent success.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void arm_slots_kernel(int* __restrict__ words, long long n, int sentinel) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (; i < n; i += stride) words[i] = sentinel;
}

int64_t arm_slots(torch::Tensor slots, int64_t sentinel) {
  TORCH_CHECK(slots.is_cuda(), "arm_slots: slots must be a CUDA tensor");
  TORCH_CHECK(slots.scalar_type() == torch::kInt32, "arm_slots: slots must be int32");
  TORCH_CHECK(slots.is_contiguous(), "arm_slots: slots must be contiguous");
  const long long n = static_cast<long long>(slots.numel());
  if (n == 0) return 0;
  auto stream = c10::cuda::getCurrentCUDAStream(slots.device().index()).stream();
  const int threads = 256;
  long long blocks = (n + threads - 1) / threads;
  if (blocks > 4096) blocks = 4096;
  arm_slots_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      slots.data_ptr<int32_t>(), n, static_cast<int>(sentinel));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "arm_slots: kernel launch failed");
  return n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("arm_slots", &arm_slots, "Poison a rank's symmetric slot ring with the Lamport sentinel");
}
