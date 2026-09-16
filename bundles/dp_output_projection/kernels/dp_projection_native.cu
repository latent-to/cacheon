/*
 * Decode projection exchanges for four attention-DP ranks.
 *
 * Each transport has three slots with fixed logical-row addresses. A lane
 * pushes a vector, reads its incoming vectors locally, then resets what it read.
 * A writer cannot complete invocation n+1 before every rank enters n+1; entry
 * follows completion of n, including its resets. Thus reuse at n+3 follows all
 * resets of n. Nonempty, uniform padded calls and serialized streams are required.
 * Inactive rows advance epochs but leave the sentinel-filled payload untouched.
 *
 * The data word is also its arrival signal, using the protocol of the serving
 * GLM exchange. The BF16 pair (+0,-0), bits 0x80000000, travels as (+0,+0).
 * Other bit patterns are unchanged. System-relaxed loads/stores are atomic per
 * 32-bit word; every word in a vector must arrive before that vector is consumed.
 * X/R are copied to fixed private buffers for GEMM and normalization. No reader
 * retains a transport pointer after the exchange, so reuse needs no credit flag.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace sglang::dp_projection {

using Bf16 = __nv_bfloat16;
constexpr int World = 4;
constexpr int MaxRows = 32;
constexpr int InputParts = 4;
constexpr int InputBlocks = MaxRows * InputParts;
constexpr int InputThreads = 768;
constexpr int Slots = 3;
constexpr uint32_t Sentinel = 0x80000000u;
constexpr int HeaderBytes = InputBlocks * sizeof(uint64_t);

__device__ __forceinline__ uint64_t phase_clock() {
#ifdef DP_PHASE_TRACE
  uint64_t time;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time) : : "memory");
  return time;
#else
  return 0;
#endif
}

__device__ __forceinline__ void phase_mark(
    int64_t* trace, uint64_t epoch, int phase, uint64_t time = 0) {
#ifdef DP_PHASE_TRACE
  if (trace != nullptr && threadIdx.x == 0) {
    auto* row = trace + ((epoch & 255) * 128 + blockIdx.x) * 9;
    row[0] = epoch;
    row[phase] = time ? time : phase_clock();
  }
#endif
}

__device__ __forceinline__ uint4 load_vector(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint4 load_arrival(const void* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void store_arrival(void* p, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};"
               : : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ bool arrived(uint4 v) {
  return (v.x != Sentinel) & (v.y != Sentinel) & (v.z != Sentinel) & (v.w != Sentinel);
}

__device__ __forceinline__ uint4 canonicalize(uint4 v) {
  return make_uint4(v.x == Sentinel ? 0 : v.x, v.y == Sentinel ? 0 : v.y,
                   v.z == Sentinel ? 0 : v.z, v.w == Sentinel ? 0 : v.w);
}

__device__ __forceinline__ Bf16* payload(const int64_t* peers, int peer) {
  return reinterpret_cast<Bf16*>(peers[peer] + HeaderBytes);
}

__device__ uint64_t begin(const int64_t* peers, int rank) {
  __shared__ uint64_t current;
  if (threadIdx.x == 0) {
    current = ++reinterpret_cast<uint64_t*>(peers[rank])[blockIdx.x];
  }
  __syncthreads();
  return current;
}

__global__ __launch_bounds__(InputThreads, 1) void push_input_rows(
    const Bf16* x, const Bf16* residual, Bf16* gathered_x, Bf16* gathered_r,
    const int64_t* peers, int rank, int rows, int width, int hidden,
    int64_t x_stride, int64_t residual_stride, int64_t* trace) {
  const uint64_t started = threadIdx.x == 0 ? phase_clock() : 0;
  const uint64_t epoch = begin(peers, rank);
  phase_mark(trace, epoch, 1, started);
  const int row = blockIdx.x / InputParts;
  const int packed_width = width + hidden;
  const int slot_base = (epoch % Slots) * World * MaxRows * packed_width;
  const int chunk_width = ((packed_width + InputParts * 8 - 1) / (InputParts * 8)) * 8;
  const int first_col = (blockIdx.x % InputParts) * chunk_width;
  const int last_col = min(first_col + chunk_width, packed_width);
  const int p1 = (rank + 1) % World, p2 = (rank + 2) % World, p3 = (rank + 3) % World;
  const uint4 empty = make_uint4(Sentinel, Sentinel, Sentinel, Sentinel);
  phase_mark(trace, epoch, 2);
  if (row < rows) {
    for (int col = first_col + threadIdx.x * 8; col < last_col; col += InputThreads * 8) {
      const auto* source = col < width ? x + row * x_stride + col
                                      : residual + row * residual_stride + col - width;
      const uint4 own = canonicalize(load_vector(source));
      const int sent_offset = slot_base + (rank * MaxRows + row) * packed_width + col;
      store_arrival(payload(peers, p1) + sent_offset, own);
      store_arrival(payload(peers, p2) + sent_offset, own);
      store_arrival(payload(peers, p3) + sent_offset, own);
      phase_mark(trace, epoch, 3);

      auto* inbox = payload(peers, rank) + slot_base + row * packed_width + col;
      auto* a1 = inbox + p1 * MaxRows * packed_width;
      auto* a2 = inbox + p2 * MaxRows * packed_width;
      auto* a3 = inbox + p3 * MaxRows * packed_width;
      uint4 v1, v2, v3;
      do {
        // Issue all three loads before testing any peer; no serial peer wait.
        v1 = load_arrival(a1);
        v2 = load_arrival(a2);
        v3 = load_arrival(a3);
      } while (!(arrived(v1) & arrived(v2) & arrived(v3)));
      phase_mark(trace, epoch, 4);

      Bf16* output = col < width ? gathered_x : gathered_r;
      const int stride = col < width ? width : hidden;
      const int offset = row * stride + (col < width ? col : col - width);
      *reinterpret_cast<uint4*>(output + rank * rows * stride + offset) = own;
      *reinterpret_cast<uint4*>(output + p1 * rows * stride + offset) = v1;
      *reinterpret_cast<uint4*>(output + p2 * rows * stride + offset) = v2;
      *reinterpret_cast<uint4*>(output + p3 * rows * stride + offset) = v3;
      phase_mark(trace, epoch, 5);

      store_arrival(a1, empty);
      store_arrival(a2, empty);
      store_arrival(a3, empty);
      phase_mark(trace, epoch, 6);
    }
  }
  phase_mark(trace, epoch, 8);
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float reciprocal(float value) {
  float result;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ void quantize_eight(
    const float (&values)[8], float global_scale, uint32_t* output, uint8_t* scales) {
  float amax = 0.0f;
  #pragma unroll
  for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(values[j]));
  // Adjacent lanes own the two halves of one linear 16-value scale block.
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
  // Approximate reciprocal scaling moves exact E4M3 midpoints across the tie.
  // PyTorch's reference FP8 constructor rounds its double result through float.
  const float sf = __double2float_rn(
      __ddiv_rn(__dmul_rn(double(amax), double(global_scale)), 6.0));
  uint32_t code;
  float rounded_scale;
  asm("{ .reg .b16 fp8, lo, hi; .reg .b32 h2;\n"
      "cvt.rn.satfinite.e4m3x2.f32 fp8, 0f00000000, %2;\n"
      "cvt.u32.u16 %0, fp8;\n"
      "cvt.rn.f16x2.e4m3x2 h2, fp8;\n"
      "mov.b32 {lo, hi}, h2; cvt.f32.f16 %1, lo; }"
      : "=r"(code), "=f"(rounded_scale) : "f"(sf));
  // Preserve the pinned FlashInfer reciprocal order and its zero-scale rule.
  const float factor = rounded_scale == 0.0f ? 0.0f
      : reciprocal(__fmul_rn(rounded_scale, reciprocal(global_scale)));
  float scaled[8];
  #pragma unroll
  for (int j = 0; j < 8; ++j) scaled[j] = __fmul_rn(values[j], factor);
  uint32_t packed;
  asm("{ .reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3}; }"
      : "=r"(packed) : "f"(scaled[0]), "f"(scaled[1]), "f"(scaled[2]), "f"(scaled[3]),
        "f"(scaled[4]), "f"(scaled[5]), "f"(scaled[6]), "f"(scaled[7]));
  *output = packed;
  if ((threadIdx.x & 1) == 0) *scales = static_cast<uint8_t>(code);
}

template <int Hidden, typename Weight, bool Quantize>
__global__ __launch_bounds__(Hidden / 8, 1) void push_columns_add_norm(
    const Bf16* projected, const Bf16* gathered_r, const Weight* weight,
    Bf16* normalized, Bf16* local_residual, const int64_t* peers,
    int rank, int rows, float eps, int64_t* trace,
    uint8_t* fp4, uint8_t* scales, const float* global_scale) {
  const uint64_t started = threadIdx.x == 0 ? phase_clock() : 0;
  constexpr int Shard = Hidden / World;
  constexpr int Warps = Hidden / (8 * 32);
  const int row = blockIdx.x;
  const uint64_t epoch = begin(peers, rank);
  phase_mark(trace, epoch, 1, started);
  const int slot_base = (epoch % Slots) * World * MaxRows * Hidden;
  const int col = threadIdx.x * 8;
  phase_mark(trace, epoch, 2);
  if (row >= World * rows) {
    phase_mark(trace, epoch, 8);
    return;
  }
  if (col < Shard) {
    const uint4 value = canonicalize(load_vector(projected + row * Shard + col));
    const int offset = slot_base + row * Hidden + rank * Shard + col;
    #pragma unroll
    for (int peer = 0; peer < World; ++peer) {
      store_arrival(payload(peers, peer) + offset, value);
    }
  }
  phase_mark(trace, epoch, 3);
  auto* inbox = payload(peers, rank) + slot_base + row * Hidden + col;
  uint4 packed;
  do { packed = load_arrival(inbox); } while (!arrived(packed));
  phase_mark(trace, epoch, 4);
  const uint4 residual = load_vector(gathered_r + row * Hidden + col);
  phase_mark(trace, epoch, 5);
  store_arrival(inbox, make_uint4(Sentinel, Sentinel, Sentinel, Sentinel));
  phase_mark(trace, epoch, 6);

  const uint32_t p[4] = {packed.x, packed.y, packed.z, packed.w};
  const uint32_t r[4] = {residual.x, residual.y, residual.z, residual.w};
  uint32_t rounded[4] = {};
  float values[8], sum = 0.0f;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int shift = (j & 1) * 16;
    const float a = __bfloat162float(__ushort_as_bfloat16(p[j / 2] >> shift));
    const float b = __bfloat162float(__ushort_as_bfloat16(r[j / 2] >> shift));
    const Bf16 s = __float2bfloat16_rn(a + b);
    values[j] = __bfloat162float(s);
    sum += values[j] * values[j];
    rounded[j / 2] |= uint32_t(__bfloat16_as_ushort(s)) << shift;
  }
  if (row / rows == rank) {
    *reinterpret_cast<uint4*>(local_residual + (row % rows) * Hidden + col) =
        make_uint4(rounded[0], rounded[1], rounded[2], rounded[3]);
  }

  __shared__ float partial[Warps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  sum = warp_sum(sum);
  if (lane == 0) partial[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float total = lane < Warps ? partial[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) partial[0] = rsqrtf(total / Hidden + eps);
  }
  __syncthreads();
  phase_mark(trace, epoch, 7);
  const float inv = partial[0];
  uint32_t result[4] = {};
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int shift = (j & 1) * 16;
    const float gamma = static_cast<float>(weight[col + j]);
    const Bf16 y = __float2bfloat16_rn((values[j] * inv) * gamma);
    result[j / 2] |= uint32_t(__bfloat16_as_ushort(y)) << shift;
    if constexpr (Quantize) values[j] = __bfloat162float(y);
  }
  *reinterpret_cast<uint4*>(normalized + row * Hidden + col) =
      make_uint4(result[0], result[1], result[2], result[3]);
  if constexpr (Quantize) {
    quantize_eight(values, *global_scale,
                   reinterpret_cast<uint32_t*>(fp4 + (row * Hidden + col) / 2),
                   scales + (row * Hidden + col) / 16);
  }
  phase_mark(trace, epoch, 8);
}


void check_tensor(const torch::Tensor& x, at::ScalarType dtype) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == dtype,
              "projection tensor has the wrong device, layout or dtype");
}

void check_matrix(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0);
}

void check_peers(const torch::Tensor& peers, int64_t rank, const torch::Tensor& x) {
  check_tensor(peers, at::kLong);
  TORCH_CHECK(peers.device() == x.device() && peers.numel() == World);
  TORCH_CHECK(rank >= 0 && rank < World);
}

int64_t* trace_pointer(const torch::Tensor& trace, const torch::Tensor& x) {
  check_tensor(trace, at::kLong);
  TORCH_CHECK(trace.device() == x.device());
  TORCH_CHECK(trace.numel() == 0 || trace.numel() == 256 * 128 * 9);
  return trace.numel() ? trace.data_ptr<int64_t>() : nullptr;
}

int64_t input_arena_bytes(int64_t width, int64_t hidden) {
  return HeaderBytes + Slots * World * MaxRows * (width + hidden) * sizeof(Bf16);
}

int64_t payload_offset() { return HeaderBytes; }

int64_t output_arena_bytes(int64_t hidden) {
  return HeaderBytes + Slots * World * MaxRows * hidden * sizeof(Bf16);
}

void push_inputs(torch::Tensor x, torch::Tensor residual, torch::Tensor gathered_x,
                 torch::Tensor gathered_r, torch::Tensor peers, int64_t rank, torch::Tensor trace) {
  check_matrix(x); check_matrix(residual); check_matrix(gathered_x); check_matrix(gathered_r);
  check_peers(peers, rank, x);
  TORCH_CHECK(residual.device() == x.device() && gathered_x.device() == x.device()
              && gathered_r.device() == x.device());
  TORCH_CHECK(gathered_x.is_contiguous() && gathered_r.is_contiguous());
  const int rows = x.size(0), width = x.size(1), hidden = residual.size(1);
  TORCH_CHECK(rows > 0 && rows <= MaxRows && width % 8 == 0 && hidden % 8 == 0);
  TORCH_CHECK(residual.size(0) == rows);
  TORCH_CHECK(gathered_x.size(0) == World * rows && gathered_x.size(1) == width);
  TORCH_CHECK(gathered_r.size(0) == World * rows && gathered_r.size(1) == hidden);
  c10::cuda::CUDAGuard guard(x.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  push_input_rows<<<InputBlocks, InputThreads, 0, stream>>>(
      static_cast<const Bf16*>(x.data_ptr()), static_cast<const Bf16*>(residual.data_ptr()),
      static_cast<Bf16*>(gathered_x.data_ptr()), static_cast<Bf16*>(gathered_r.data_ptr()),
      peers.data_ptr<int64_t>(), rank, rows, width, hidden, x.stride(0), residual.stride(0),
      trace_pointer(trace, x));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "projection input exchange launch failed");
}

void column_gather_norm(torch::Tensor projected, torch::Tensor residual, torch::Tensor weight,
                        torch::Tensor normalized, torch::Tensor local_residual, torch::Tensor peers,
                        int64_t rank, double eps, torch::Tensor trace,
                        torch::Tensor fp4, torch::Tensor scales, torch::Tensor global_scale) {
  check_matrix(projected); check_matrix(residual); check_matrix(normalized); check_matrix(local_residual);
  TORCH_CHECK(projected.is_contiguous() && residual.is_contiguous()
              && normalized.is_contiguous() && local_residual.is_contiguous());
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous()
              && (weight.scalar_type() == at::kBFloat16 || weight.scalar_type() == at::kFloat));
  check_peers(peers, rank, projected);
  for (const auto& tensor : {residual, weight, normalized, local_residual})
    TORCH_CHECK(tensor.device() == projected.device());
  const int rows = local_residual.size(0), hidden = local_residual.size(1);
  TORCH_CHECK(rows > 0 && rows <= MaxRows && (hidden == 6144 || hidden == 4096));
  TORCH_CHECK(projected.size(0) == World * rows && projected.size(1) == hidden / World);
  TORCH_CHECK(residual.size(0) == World * rows && residual.size(1) == hidden);
  TORCH_CHECK(normalized.size(0) == World * rows && normalized.size(1) == hidden);
  TORCH_CHECK(weight.numel() == hidden);
  const bool quantize = global_scale.numel() != 0;
  if (quantize) {
    check_tensor(global_scale, at::kFloat); check_tensor(fp4, at::kByte); check_tensor(scales, at::kByte);
    TORCH_CHECK(global_scale.device() == projected.device() && fp4.device() == projected.device()
                && scales.device() == projected.device());
    TORCH_CHECK(global_scale.numel() == 1 && fp4.numel() == World * rows * hidden / 2
                && scales.numel() == World * rows * hidden / 16);
  }
  c10::cuda::CUDAGuard guard(projected.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(projected.get_device()).stream();
  #define LAUNCH(H, W, Q) push_columns_add_norm<H, W, Q><<<World * MaxRows, H / 8, 0, stream>>>( \
      static_cast<const Bf16*>(projected.data_ptr()), static_cast<const Bf16*>(residual.data_ptr()), \
      static_cast<const W*>(weight.data_ptr()), static_cast<Bf16*>(normalized.data_ptr()), \
      static_cast<Bf16*>(local_residual.data_ptr()), peers.data_ptr<int64_t>(), rank, rows, eps, \
      trace_pointer(trace, projected), fp4.data_ptr<uint8_t>(), \
      scales.data_ptr<uint8_t>(), global_scale.data_ptr<float>())
  #define DISPATCH(H, W) if (quantize) { LAUNCH(H, W, true); } else { LAUNCH(H, W, false); }
  if (weight.scalar_type() == at::kBFloat16) {
    if (hidden == 6144) { DISPATCH(6144, Bf16); } else { DISPATCH(4096, Bf16); }
  } else {
    if (hidden == 6144) { DISPATCH(6144, float); } else { DISPATCH(4096, float); }
  }
  #undef DISPATCH
  #undef LAUNCH
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "projection output exchange launch failed");
}

}  // namespace sglang::dp_projection

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  using namespace sglang::dp_projection;
  module.def("input_arena_bytes", &input_arena_bytes);
  module.def("output_arena_bytes", &output_arena_bytes);
  module.def("payload_offset", &payload_offset);
  module.def("push_inputs", &push_inputs);
  module.def("column_gather_norm", &column_gather_norm);
}
