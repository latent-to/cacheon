// Fused attention-DP output projection collective, four ranks, sm_103a.
//
// Two native launches bracket the split-K Triton GEMM (kernels/dpo.py), whose K splits reduce
// into one fp32 accumulator [128 rows][1536 shard columns] with relaxed L2 atomics:
//   scatter_rows   : every rank pushes its X row and, to each peer, only the residual columns of
//                    that peer's own output shard into the rank-major input ring, then lands the
//                    arrived rows in private gathered buffers.  A leading set of blocks in the same
//                    launch does nothing but pull the weight quarter into L2 with discarded TMA bulk
//                    copies, so the GEMM that follows starts warm.  Those blocks read only the
//                    prepared weight, so they take no ring epoch and do not wait on the predecessor:
//                    they warm while the previous call's merge still holds the machine.
//   merge_columns  : every rank reads the accumulator for its column shard, rounds to bf16, adds
//                    the residual columns it owns, pushes the shard for all rows, waits for the
//                    three foreign shards of its row, normalises and packs NVFP4, one block per
//                    gathered row.  Stores are unicast.  A rank's
//                    own quarter of a row never enters the ring - that word is already in a register
//                    - so it is neither pushed to itself nor polled back.
//                    The same threads write the accumulator back to zero on the lines they have
//                    just read, which re-arms it for the next call while those lines are still
//                    resident, so no separate clearing launch is needed and a captured graph
//                    replays without host involvement.  The blocks that own the shard cover it
//                    exactly once, so every live element is cleared exactly once.
// Arrival protocol: a bf16 pair word equal to 0x80000000 is the empty marker; real data with
// that bit pattern is +0/-0 and is transported as +0/+0 (value preserving).  Every ring has
// three slots addressed by a per-block epoch counter kept in the block's own header, so
// graph replays advance it without host involvement.  A rank re-fills a slot only two
// invocations after every peer finished reading it, which the lockstep structure guarantees.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace dpo {

using bf16 = __nv_bfloat16;
constexpr int W = 4;                       // ranks
constexpr int RMAX = 32;                   // padded local rows per rank
// Wide band: the push keeps 80 of the 148 blocks - its exchange is NVLink bound, not block bound -
// and the other 68 SMs run the L2 fill.  A fill block does not share an SM with a push block, so the
// only way to give one an SM is to stop pushing from it.
constexpr int PBLOCKS = 80;                // push blocks when a fill runs alongside
constexpr int FBLOCKS = 68;                // fill blocks, PBLOCKS + FBLOCKS = the 148 SMs
constexpr uint32_t FILL_CHUNK = 16384;     // bytes per TMA copy
constexpr int FILL_DEPTH = 8;              // copies in flight per fill block (131 KB of shared)
constexpr int FILL_PERCENT = 75;           // the whole quarter does not fit in the exchange window
// Narrow band: a shorter exchange, so fewer blocks come off the push and each carries a larger copy.
constexpr int NPBLOCKS = 116;
constexpr int NFBLOCKS = 32;
constexpr uint32_t NFILL_CHUNK = 65536;
constexpr int NFILL_DEPTH = 3;
constexpr int NFILL_PERCENT = 40;
constexpr int STHREADS = 768;
constexpr int SLOTS = 3;
constexpr uint32_t EMPTY = 0x80000000u;
constexpr int MCOL = RMAX * W;             // 128 gathered rows max
constexpr int HDR = 1024 * 8;              // epoch counters at the head of each ring: one per merge row and one
                                           // per scatter block, neither of which ever exceeds 1024

__device__ __forceinline__ uint4 ld_cg(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_sys(const void* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_sys(void* p, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ bool full(uint4 v) {
  return (v.x != EMPTY) & (v.y != EMPTY) & (v.z != EMPTY) & (v.w != EMPTY);
}
__device__ __forceinline__ uint4 scrub(uint4 v) {
  return make_uint4(v.x == EMPTY ? 0u : v.x, v.y == EMPTY ? 0u : v.y, v.z == EMPTY ? 0u : v.z, v.w == EMPTY ? 0u : v.w);
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// maps: [0..3] = peers' ring base pointers (index 4 reserved)
__device__ __forceinline__ char* ring_of(const int64_t* maps, int r) { return reinterpret_cast<char*>(maps[r]); }

__device__ __forceinline__ uint64_t next_epoch(const int64_t* maps, int rank, int idx) {
  __shared__ uint64_t e;
  if (threadIdx.x == 0) e = ++reinterpret_cast<uint64_t*>(maps[rank])[idx];
  __syncthreads();
  return e;
}

// ---------------------------------------------------------------------------------------------
// L2 fill: pull `bytes` of the weight quarter through L2 with TMA bulk copies into a small
// shared-memory ring and discard them.  cp.async.bulk.prefetch.L2 is only a hint and leaves most of
// the cold/warm gap behind; a real copy closes it.  The SM issues descriptors and waits on
// mbarriers, so the rate off a small block count is set by bytes in flight against the DRAM
// latency - `DEPTH` buffers of `CHUNK` - and not by the thread count.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect(uint32_t bar, uint32_t n) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t phase) {
  asm volatile("{ .reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W; }"
               :: "r"(bar), "r"(phase) : "memory");
}
__device__ __forceinline__ void bulk_copy(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}

// The wide band keeps the cooperative issue; the single-threaded issue below belongs to the
// narrow band, whose exchange is latency bound rather than bandwidth bound.
__device__ __noinline__ void tma_fill_wide(char* smem, const char* __restrict__ p, uint32_t bytes,
                                      uint32_t bidx, uint32_t nblk) {
  uint64_t* const barriers = reinterpret_cast<uint64_t*>(smem + size_t(FILL_DEPTH) * FILL_CHUNK);
  if (threadIdx.x == 0)
    for (int k = 0; k < FILL_DEPTH; ++k) mbar_init(smem_u32(&barriers[k]));
  __syncthreads();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  const uint64_t stride = uint64_t(nblk) * FILL_CHUNK;
  uint64_t cur = uint64_t(bidx) * FILL_CHUNK;
  uint64_t nxt = cur + uint64_t(FILL_DEPTH) * stride;
  if (threadIdx.x == 0)
    for (int k = 0; k < FILL_DEPTH; ++k) {
      const uint64_t b = cur + uint64_t(k) * stride;
      if (b < bytes) {
        const uint32_t n = min(uint32_t(FILL_CHUNK), uint32_t(bytes - b));
        mbar_arrive_expect(smem_u32(&barriers[k]), n);
        bulk_copy(smem_u32(smem + size_t(k) * FILL_CHUNK), p + b, n, smem_u32(&barriers[k]));
      }
    }
  __syncthreads();
  uint32_t phase = 0;
  int slot = 0;
  while (cur < bytes) {
    mbar_wait(smem_u32(&barriers[slot]), phase);
    __syncthreads();
    if (threadIdx.x == 0 && nxt < bytes) {
      const uint32_t n = min(uint32_t(FILL_CHUNK), uint32_t(bytes - nxt));
      mbar_arrive_expect(smem_u32(&barriers[slot]), n);
      bulk_copy(smem_u32(smem + size_t(slot) * FILL_CHUNK), p + nxt, n, smem_u32(&barriers[slot]));
    }
    __syncthreads();
    cur += stride; nxt += stride;
    if (++slot == FILL_DEPTH) { slot = 0; phase ^= 1u; }
  }
}

// __noinline__ keeps this out of the push path's register budget.
__device__ __noinline__ void tma_fill(char* smem, const char* __restrict__ p, uint32_t bytes,
                                      uint32_t bidx, uint32_t nblk, uint32_t chunk, int depth) {
  // One thread issues and retires the whole pipeline.  Nothing reads the landed bytes, so the rest
  // of the block has no work, and the pair of __syncthreads that a cooperative issue needs per chunk
  // is what caps the rate: below what the GEMM itself sustains, which makes every filled byte cost
  // more than it saves.
  if (threadIdx.x != 0) return;
  uint64_t* const barriers = reinterpret_cast<uint64_t*>(smem + size_t(depth) * chunk);
  for (int k = 0; k < depth; ++k) mbar_init(smem_u32(&barriers[k]));
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  const uint64_t stride = uint64_t(nblk) * chunk;
  uint64_t cur = uint64_t(bidx) * chunk;
  uint64_t nxt = cur + uint64_t(depth) * stride;
  for (int k = 0; k < depth; ++k) {
    const uint64_t b = cur + uint64_t(k) * stride;
    if (b < bytes) {
      const uint32_t n = min(chunk, uint32_t(bytes - b));
      mbar_arrive_expect(smem_u32(&barriers[k]), n);
      bulk_copy(smem_u32(smem + size_t(k) * chunk), p + b, n, smem_u32(&barriers[k]));
    }
  }
  uint32_t phase = 0;
  int slot = 0;
  while (cur < bytes) {
    mbar_wait(smem_u32(&barriers[slot]), phase);
    if (nxt < bytes) {
      const uint32_t n = min(chunk, uint32_t(bytes - nxt));
      mbar_arrive_expect(smem_u32(&barriers[slot]), n);
      bulk_copy(smem_u32(smem + size_t(slot) * chunk), p + nxt, n, smem_u32(&barriers[slot]));
    }
    cur += stride; nxt += stride;
    if (++slot == depth) { slot = 0; phase ^= 1u; }
  }
}

// ---------------------------------------------------------------------------------------------
// Kernel 1: scatter X|R rows to every rank, stream the weight shard toward L2, land arrivals.
// Ring A payload layout: [slot][src rank][row (RMAX)][width + hidden / W] bf16; the residual part
// of a region holds only the columns of the *receiving* rank's output shard, which is the only part
// of a peer's residual its merge ever adds.
// ---------------------------------------------------------------------------------------------
// The fill blocks lead the grid so that they are the first to be placed when the predecessor
// frees an SM, and they take the push blocks' indices out of the way: a push block keeps the ring
// header slot it has always had, which is what lets a captured graph replay the epoch unaided.
template <int NB, int NT, bool FILL>
__global__ void __launch_bounds__(NT, 1)
scatter_rows(const bf16* __restrict__ x, const bf16* __restrict__ res,
             bf16* __restrict__ gx, bf16* __restrict__ gr,
             const int64_t* __restrict__ maps, int rank, int rows, int width, int hidden,
             int64_t x_stride, int64_t r_stride,
             const char* __restrict__ shard, uint32_t fill_bytes, int pdl,
             uint32_t fill_chunk, int fill_depth, int release) {
  // The blocks ahead of the push set exist only to warm L2 with this rank's weight quarter; they
  // take no ring epoch, push nothing and land nothing.  They release dependents before filling so
  // the GEMM is gated on the push blocks alone.
  const int nfill = int(gridDim.x) - NB;
  if constexpr (FILL) {
    if (int(blockIdx.x) < nfill) {
      // No pdl_wait: this block reads only the prepared weight shard, which no kernel in the chain
      // writes, so it has no dependency on the predecessor and may begin the instant it is resident.
      if (pdl && release) pdl_trigger();
      extern __shared__ __align__(128) char smem[];
      if (fill_depth < 0)
        tma_fill_wide(smem, shard, fill_bytes, blockIdx.x, nfill);
      else
        tma_fill(smem, shard, fill_bytes, blockIdx.x, nfill, fill_chunk, fill_depth);
      return;
    }
  }
  const int pb = int(blockIdx.x) - nfill;
  // taken before the wait: the counter is in this rank's own ring header, so the round trip is
  // spent while the block is merely resident rather than at the head of the push
  const uint64_t epoch = next_epoch(maps, rank, pb);
  if (pdl) pdl_wait();
  // rows x parts blocks are active; parts grows as rows shrink so small batches still use every SM
  const int parts = NB / rows;
  const int row = pb / parts;
  const int sh = hidden / W;                 // residual columns per destination rank
  const int packed = width + sh;
  const size_t slot_base = size_t(epoch % SLOTS) * W * RMAX * packed;   // elements
  const int chunk = ((packed + parts * 8 - 1) / (parts * 8)) * 8;
  const int c0 = (pb % parts) * chunk;
  const int c1 = min(c0 + chunk, packed);
  const int p1 = (rank + 1) & 3, p2 = (rank + 2) & 3, p3 = (rank + 3) & 3;
  const uint4 empty = make_uint4(EMPTY, EMPTY, EMPTY, EMPTY);
  const bool active = row < rows && c0 < c1;

  // 1. push own chunk of this row to the three peers; own rows go straight to the private buffers
  if (active) {
    for (int c = c0 + threadIdx.x * 8; c < c1; c += NT * 8) {
      const size_t off = HDR + (slot_base + (size_t(rank) * RMAX + row) * packed + c) * sizeof(bf16);
      if (c < width) {
        const uint4 v = scrub(ld_cg(x + row * x_stride + c));
        st_sys(ring_of(maps, p1) + off, v);
        st_sys(ring_of(maps, p2) + off, v);
        st_sys(ring_of(maps, p3) + off, v);
        *reinterpret_cast<uint4*>(gx + (size_t(rank) * rows + row) * width + c) = v;
      } else {
        // each peer is sent a different quarter of the residual row: the columns it will merge.
        // The four reads are issued before the first store consumes one; ld_cg is a volatile asm
        // statement, so interleaving them with the stores would serialise four memory latencies.
        const bf16* r0 = res + row * r_stride + (c - width);
        const uint4 w1 = ld_cg(r0 + p1 * sh), w2 = ld_cg(r0 + p2 * sh), w3 = ld_cg(r0 + p3 * sh), w0 = ld_cg(r0 + rank * sh);
        st_sys(ring_of(maps, p1) + off, scrub(w1));
        st_sys(ring_of(maps, p2) + off, scrub(w2));
        st_sys(ring_of(maps, p3) + off, scrub(w3));
        *reinterpret_cast<uint4*>(gr + (size_t(rank) * rows + row) * sh + (c - width)) = scrub(w0);
      }
    }
  }
  if (pdl && release) pdl_trigger();
  // 2. land the three foreign chunks of this row, then re-arm what was read
  if (active) {
    char* own = ring_of(maps, rank) + HDR;
    for (int c = c0 + threadIdx.x * 8; c < c1; c += NT * 8) {
      const size_t base = slot_base + size_t(row) * packed + c;
      char* a1 = own + (base + size_t(p1) * RMAX * packed) * sizeof(bf16);
      char* a2 = own + (base + size_t(p2) * RMAX * packed) * sizeof(bf16);
      char* a3 = own + (base + size_t(p3) * RMAX * packed) * sizeof(bf16);
      uint4 v1, v2, v3;
      do { v1 = ld_sys(a1); v2 = ld_sys(a2); v3 = ld_sys(a3); } while (!(full(v1) & full(v2) & full(v3)));
      bf16* out = c < width ? gx : gr;
      const int stride = c < width ? width : sh;
      const size_t o = size_t(row) * stride + (c < width ? c : c - width);
      *reinterpret_cast<uint4*>(out + size_t(p1) * rows * stride + o) = v1;
      *reinterpret_cast<uint4*>(out + size_t(p2) * rows * stride + o) = v2;
      *reinterpret_cast<uint4*>(out + size_t(p3) * rows * stride + o) = v3;
      st_sys(a1, empty); st_sys(a2, empty); st_sys(a3, empty);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Kernel 2: merge projected column shards, residual add, RMSNorm, NVFP4.
// Ring B payload layout: [slot][row (MCOL)][hidden] bf16 — column c of row r arrives from rank c / shard.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ void pack_fp4(const float (&v)[8], float gscale, uint32_t* out, uint8_t* sc) {
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));      // 16-value block = two adjacent lanes
  // Block scale.  The reference forms amax * gscale / 6 in fp64 and rounds once to fp32; both
  // operands are bf16-derived floats, so the product is exact in fp32 as well and a correctly
  // rounded fp32 divide reproduces the reference byte for byte.  Blackwell runs fp64 at a small
  // fraction of the fp32 rate and `__ddiv_rn` expands to a long fp64 sequence.
  const float s = __fdiv_rn(__fmul_rn(amax, gscale), 6.f);
  uint32_t code; float sr;
  asm("{ .reg .b16 q, lo, hi; .reg .b32 h2;\n"
      "  cvt.rn.satfinite.e4m3x2.f32 q, 0f00000000, %2;\n"
      "  cvt.u32.u16 %0, q;\n"
      "  cvt.rn.f16x2.e4m3x2 h2, q;\n"
      "  mov.b32 {lo, hi}, h2; cvt.f32.f16 %1, lo; }" : "=r"(code), "=f"(sr) : "f"(s));
  float inv_g; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_g) : "f"(gscale));
  float den = __fmul_rn(sr, inv_g);
  float mul; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(mul) : "f"(den));
  mul = sr == 0.f ? 0.f : mul;
  float q[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) q[j] = __fmul_rn(v[j], mul);
  uint32_t packed;
  asm("{ .reg .b8 b0, b1, b2, b3;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "  mov.b32 %0, {b0, b1, b2, b3}; }"
      : "=r"(packed) : "f"(q[0]), "f"(q[1]), "f"(q[2]), "f"(q[3]), "f"(q[4]), "f"(q[5]), "f"(q[6]), "f"(q[7]));
  *out = packed;
  if ((threadIdx.x & 1) == 0) *sc = uint8_t(code);
}

template <int HID, bool QUANT>
__global__ void __launch_bounds__(HID / 8, 1)
merge_columns(float* __restrict__ acc_buf, const bf16* __restrict__ gr, const bf16* __restrict__ gamma,
              bf16* __restrict__ normed, bf16* __restrict__ local_res,
              const int64_t* __restrict__ maps, int rank, int rows, float eps,
              uint8_t* __restrict__ fp4, uint8_t* __restrict__ scales, const float* __restrict__ gscale,
              int pdl) {
  constexpr int SH = HID / W;             // columns per rank shard
  constexpr int THREADS = HID / 8;
  constexpr int WARPS = THREADS / 32;
  const int row = blockIdx.x;             // gathered row index, rank-major: src * rows + r
  const int col = threadIdx.x * 8;
  __shared__ float part[WARPS];
  __shared__ float total;
  __shared__ uint64_t epoch_s;
  // One epoch counter per gathered row, advanced by the single block that owns the row.
  if (threadIdx.x == 0) epoch_s = ++reinterpret_cast<uint64_t*>(maps[rank])[row];
  __syncthreads();
  const size_t slot_base = size_t(epoch_s % SLOTS) * MCOL * HID;
  const bool mine = col >= rank * SH && col < (rank + 1) * SH;
  // the gathered residual and the two norm parameters come from earlier kernels, so they are read
  // before the projection is waited on and their latency is spent inside the GEMM
  const uint4 rv = mine ? ld_cg(gr + size_t(row) * SH + (col - rank * SH)) : make_uint4(0u, 0u, 0u, 0u);
  const uint4 g4 = *reinterpret_cast<const uint4*>(gamma + col);
  float gsc = 0.f;
  if constexpr (QUANT) gsc = *gscale;
  // Released before the wait, not after the push.  The next call's push blocks wait on this
  // kernel regardless, so the only blocks that can use the head start are its fill blocks, and
  // those read nothing any kernel in the chain writes.
  if (pdl) pdl_trigger();
  if (pdl) pdl_wait();
  // 1. read the reduced fp32 accumulator for this rank's shard columns (one bf16 rounding point
  //    like the vendor GEMM), clear it for the next call, then push the bf16 row shard to every rank
  uint4 own = make_uint4(0u, 0u, 0u, 0u);
  if (mine) {
    float* src = acc_buf + size_t(row) * SH + (col - rank * SH);
    const float4 lo = *reinterpret_cast<const float4*>(src);
    const float4 hi = *reinterpret_cast<const float4*>(src + 4);
    const float acc[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
    *reinterpret_cast<float4*>(src) = zero;
    *reinterpret_cast<float4*>(src + 4) = zero;
    // the projection is rounded to bf16 first and the residual added to that, exactly the pair of
    // roundings the transported bf16 shard and the receiver's add used to make between them
    const uint32_t rr[4] = {rv.x, rv.y, rv.z, rv.w};
    uint32_t packed[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float a0 = __bfloat162float(__float2bfloat16_rn(acc[2 * j]));
      const float a1 = __bfloat162float(__float2bfloat16_rn(acc[2 * j + 1]));
      const float b0 = __bfloat162float(__ushort_as_bfloat16(uint16_t(rr[j])));
      const float b1 = __bfloat162float(__ushort_as_bfloat16(uint16_t(rr[j] >> 16)));
      packed[j] = uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(a0 + b0)))
                | (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(a1 + b1))) << 16);
    }
    const uint4 v = scrub(make_uint4(packed[0], packed[1], packed[2], packed[3]));
    own = v;
    const size_t off = HDR + (slot_base + size_t(row) * HID + col) * sizeof(bf16);
    // the scrubbed word this rank would send itself is the word it would poll back, so its own
    // quarter of the row stays in the register file and its ring slot is never armed
#pragma unroll
    for (int p = 0; p < W; ++p) if (p != rank) st_sys(ring_of(maps, p) + off, v);
  }
  // 2. wait for this block's columns of the row; the re-arm follows the outputs, because the slot
  //    is not refilled until two invocations later and the word is already in a register
  char* inbox = ring_of(maps, rank) + HDR + (slot_base + size_t(row) * HID + col) * sizeof(bf16);
  uint4 pv;
  if (mine) pv = own;
  else { do { pv = ld_sys(inbox); } while (!full(pv)); }
  // 3. norm statistics: the arriving word already carries projection + residual
  float v[8]; float ss = 0.f;
  {
    const uint32_t p[4] = {pv.x, pv.y, pv.z, pv.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      v[j] = __bfloat162float(__ushort_as_bfloat16(uint16_t(p[j >> 1] >> ((j & 1) * 16))));
      ss += v[j] * v[j];
    }
  }
  if (row / rows == rank)
    *reinterpret_cast<uint4*>(local_res + size_t(row % rows) * HID + col) = pv;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  ss = warp_sum(ss);
  if (lane == 0) part[warp] = ss;
  __syncthreads();
  if (warp == 0) {
    float t = lane < WARPS ? part[lane] : 0.f;
    t = warp_sum(t);
    if (lane == 0) total = t;
  }
  __syncthreads();
  if (threadIdx.x == 0) part[0] = rsqrtf(total / HID + eps);
  __syncthreads();
  const float inv = part[0];
  uint32_t o[4] = {0u, 0u, 0u, 0u};
  const uint32_t g[4] = {g4.x, g4.y, g4.z, g4.w};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int sh = (j & 1) * 16;
    const float gm = __bfloat162float(__ushort_as_bfloat16(uint16_t(g[j >> 1] >> sh)));
    const bf16 y = __float2bfloat16_rn((v[j] * inv) * gm);
    o[j >> 1] |= uint32_t(__bfloat16_as_ushort(y)) << sh;
    if constexpr (QUANT) v[j] = __bfloat162float(y);
  }
  *reinterpret_cast<uint4*>(normed + size_t(row) * HID + col) = make_uint4(o[0], o[1], o[2], o[3]);
  if constexpr (QUANT)
    pack_fp4(v, gsc, reinterpret_cast<uint32_t*>(fp4 + (size_t(row) * HID + col) / 2), scales + (size_t(row) * HID + col) / 16);
  if (!mine) st_sys(inbox, make_uint4(EMPTY, EMPTY, EMPTY, EMPTY));
}

// ---------------------------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------------------------
static void check_mat(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.dim() == 2 && t.scalar_type() == at::kBFloat16, "bf16 matrix expected");
  TORCH_CHECK(t.stride(1) == 1 && t.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "16-byte aligned rows expected");
}
static void check_maps(const torch::Tensor& maps, int64_t rank, const torch::Tensor& like) {
  TORCH_CHECK(maps.is_cuda() && maps.scalar_type() == at::kLong && maps.is_contiguous() && maps.numel() == W + 1 && maps.device() == like.device(), "maps");
  TORCH_CHECK(rank >= 0 && rank < W, "rank");
}

int64_t input_ring_bytes(int64_t width, int64_t hidden) { return HDR + int64_t(SLOTS) * W * RMAX * (width + hidden / W) * 2; }
int64_t column_ring_bytes(int64_t hidden) { return HDR + int64_t(SLOTS) * MCOL * hidden * 2; }
int64_t header_bytes() { return HDR; }

template <typename F>
static void launch_pdl(bool pdl, int cluster, cudaStream_t stream, dim3 grid, dim3 block, F kernel, void** args,
                       size_t smem = 0) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  int n = 0;
  if (pdl) {
    attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  if (cluster > 1) {
    attr[n].id = cudaLaunchAttributeClusterDimension;
    attr[n].val.clusterDim.x = cluster; attr[n].val.clusterDim.y = 1; attr[n].val.clusterDim.z = 1;
    ++n;
  }
  cfg.attrs = attr; cfg.numAttrs = n;
  const cudaError_t err = cudaLaunchKernelExC(&cfg, reinterpret_cast<const void*>(kernel), args);
  TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
}

void scatter(torch::Tensor x, torch::Tensor res, torch::Tensor gx, torch::Tensor gr, torch::Tensor maps, int64_t rank,
             torch::Tensor shard, bool fill, bool pdl, bool release) {
  check_mat(x); check_mat(res); check_mat(gx); check_mat(gr); check_maps(maps, rank, x);
  TORCH_CHECK(gx.is_contiguous() && gr.is_contiguous(), "gathered buffers must be contiguous");
  const int rows = x.size(0), width = x.size(1), hidden = res.size(1);
  TORCH_CHECK(rows > 0 && rows <= RMAX && width % 8 == 0 && hidden % 8 == 0 && res.size(0) == rows, "geometry");
  TORCH_CHECK(gx.size(0) == W * rows && gx.size(1) == width && gr.size(0) == W * rows && gr.size(1) == hidden / W, "gathered geometry");
  TORCH_CHECK(shard.is_cuda() && shard.is_contiguous() && shard.scalar_type() == at::kBFloat16
              && reinterpret_cast<uintptr_t>(shard.data_ptr()) % 128 == 0, "shard");
  c10::cuda::CUDAGuard guard(x.device());
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const bf16* xp = static_cast<const bf16*>(x.data_ptr()); const bf16* rp = static_cast<const bf16*>(res.data_ptr());
  bf16* gxp = static_cast<bf16*>(gx.data_ptr()); bf16* grp = static_cast<bf16*>(gr.data_ptr());
  const int64_t* mp = maps.data_ptr<int64_t>();
  int rk = int(rank), rw = rows, wd = width, hd = hidden;
  int64_t xs = x.stride(0), rs = res.stride(0);
  const char* sp = static_cast<const char*>(shard.data_ptr());
  const int64_t qbytes = shard.numel() * 2;
  uint32_t fb = uint32_t(uint64_t(qbytes) * (fill ? FILL_PERCENT : NFILL_PERCENT) / 100 & ~uint64_t(127));
  int pd = pdl ? 1 : 0, rel = release ? 1 : 0;
  uint32_t fc = fill ? FILL_CHUNK : NFILL_CHUNK;
  int fdp = fill ? -FILL_DEPTH : NFILL_DEPTH;   // negative selects the wide band's cooperative issue
  void* args[] = {&xp, &rp, &gxp, &grp, &mp, &rk, &rw, &wd, &hd, &xs, &rs, &sp, &fb, &pd, &fc, &fdp, &rel};
  // The push rate does not depend on how the vectors are spread over threads or blocks, so both
  // bands spend that slack on the fill instead.
  const size_t fsmem = size_t(fc) * abs(fdp) + size_t(abs(fdp)) * 8;
  if (!fill) {
    const cudaError_t nattr = cudaFuncSetAttribute(reinterpret_cast<const void*>(scatter_rows<NPBLOCKS, STHREADS, true>),
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize, int(fsmem));
    TORCH_CHECK(nattr == cudaSuccess, "narrow fill shared memory: ", cudaGetErrorString(nattr));
    launch_pdl(pdl, 1, stream, dim3(NPBLOCKS + NFBLOCKS), dim3(STHREADS),
               scatter_rows<NPBLOCKS, STHREADS, true>, args, fsmem);
    return;
  }
  const cudaError_t attr = cudaFuncSetAttribute(reinterpret_cast<const void*>(scatter_rows<PBLOCKS, STHREADS, true>),
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, int(fsmem));
  TORCH_CHECK(attr == cudaSuccess, "fill shared memory: ", cudaGetErrorString(attr));
  launch_pdl(pdl, 1, stream, dim3(PBLOCKS + FBLOCKS), dim3(STHREADS), scatter_rows<PBLOCKS, STHREADS, true>, args, fsmem);
}

void merge(torch::Tensor acc_buf, torch::Tensor gr, torch::Tensor gamma, torch::Tensor normed, torch::Tensor local_res,
           torch::Tensor maps, int64_t rank, double eps, torch::Tensor fp4, torch::Tensor scales, torch::Tensor gscale,
           bool pdl) {
  check_mat(gr); check_mat(normed); check_mat(local_res); check_maps(maps, rank, gr);
  TORCH_CHECK(gr.is_contiguous() && normed.is_contiguous() && local_res.is_contiguous(), "contiguous");
  TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous() && gamma.scalar_type() == at::kBFloat16, "gamma bf16");
  const int rows = local_res.size(0), hidden = local_res.size(1);
  TORCH_CHECK(rows > 0 && rows <= RMAX && hidden == 6144, "geometry");
  // reduced shard accumulator [MCOL][hidden / W] fp32; the GEMM's splits have already summed into it
  TORCH_CHECK(acc_buf.is_cuda() && acc_buf.is_contiguous() && acc_buf.scalar_type() == at::kFloat && acc_buf.dim() == 2
              && acc_buf.size(0) == MCOL && acc_buf.size(1) == hidden / W && acc_buf.device() == gr.device(), "accumulator");
  TORCH_CHECK(gr.size(0) == W * rows && gr.size(1) == hidden / W
              && normed.size(0) == W * rows && normed.size(1) == hidden && gamma.numel() == hidden, "shapes");
  const bool quant = gscale.numel() != 0;
  if (quant) {
    TORCH_CHECK(gscale.is_cuda() && gscale.scalar_type() == at::kFloat && gscale.numel() == 1, "gscale");
    TORCH_CHECK(fp4.is_cuda() && fp4.is_contiguous() && fp4.scalar_type() == at::kByte && fp4.numel() == int64_t(W) * rows * hidden / 2, "fp4");
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.scalar_type() == at::kByte && scales.numel() == int64_t(W) * rows * hidden / 16, "scales");
  }
  c10::cuda::CUDAGuard guard(gr.device());
  auto stream = c10::cuda::getCurrentCUDAStream(gr.get_device()).stream();
  float* pp = acc_buf.data_ptr<float>(); const bf16* grp = static_cast<const bf16*>(gr.data_ptr());
  const bf16* gp = static_cast<const bf16*>(gamma.data_ptr());
  bf16* np = static_cast<bf16*>(normed.data_ptr()); bf16* lp = static_cast<bf16*>(local_res.data_ptr());
  const int64_t* mp = maps.data_ptr<int64_t>();
  int rk = int(rank), rw = rows; float ep = float(eps);
  uint8_t* fp = quant ? fp4.data_ptr<uint8_t>() : nullptr; uint8_t* scp = quant ? scales.data_ptr<uint8_t>() : nullptr;
  const float* gsp = quant ? gscale.data_ptr<float>() : nullptr;
  int pd = pdl ? 1 : 0;
  void* args[] = {&pp, &grp, &gp, &np, &lp, &mp, &rk, &rw, &ep, &fp, &scp, &gsp, &pd};
  // One block per gathered row at every batch size.  The merge is one NVLink flight plus a norm, so
  // spreading a row over a cluster of column blocks buys occupancy it cannot use and pays two
  // cluster.sync() barriers for the row's sum of squares while the next call's scatter wants the SMs.
  const int gathered = W * rows;
#define LAUNCH(Q) launch_pdl(pdl, 1, stream, dim3(gathered), dim3(6144 / 8), merge_columns<6144, Q>, args)
  if (quant) LAUNCH(true); else LAUNCH(false);
#undef LAUNCH
}

}  // namespace dpo

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("input_ring_bytes", &dpo::input_ring_bytes);
  m.def("column_ring_bytes", &dpo::column_ring_bytes);
  m.def("header_bytes", &dpo::header_bytes);
  m.def("scatter", &dpo::scatter);
  m.def("merge", &dpo::merge);
}
