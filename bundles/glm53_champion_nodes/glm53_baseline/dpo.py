"""collective.dp_output_projection_norm: column-sharded projection for four attention-DP ranks.

Every rank pushes its attention rows to the three peers and sends each peer only the residual
columns of that peer's own output shard, streams its quarter of the projection weight, projects
every rank's rows onto that quarter with a split-K tcgen05 Triton GEMM whose splits reduce into
one fp32 accumulator through relaxed L2 atomics, then adds the residual to the rounded
accumulator, pushes the column shard back and finishes RMSNorm and NVFP4 packing on the full rows.

Replicating the whole residual row sends four times the bytes a peer's merge uses, because a rank
only ever adds the residual columns of the shard it owns; transposing it takes the scatter's row
payload from width + hidden to width + hidden / 4, a fifth off everything the exchange pushes,
polls, lands and re-arms.  The residual keeps its old rounding point - the projection is rounded
to bf16 first and the residual added to that - so the transported word is the one the receiver
used to compute and the NVFP4 match ratio is unchanged.

The accumulator is 393 KB rather than the twelve 9.4 MB partial slabs a workspace reduction
needs, so it stays resident in L2 and the merge reads a twenty-fourth of the bytes; the merge
re-zeroes the lines it has just read, which re-arms the accumulator for the next call at no extra
traffic and keeps graph replay self-contained.  Both exchanges shape themselves to the batch
(variable chunking in the row scatter, one block per row when the batch fills the machine and a
cluster of column blocks when it does not) so the long-context cell pays no serial tail.  Every
launch in the chain carries the programmatic dependent-launch attribute; only the wide band lets
its scatter release the GEMM early.

The scatter launch leads with a set of blocks that do nothing but pull this rank's weight quarter
through L2 with discarded TMA bulk copies, so the GEMM behind it starts warm.  A fill block does not
share an SM with a push block, so those SMs are taken from the push, which the exchange can spare: it
is NVLink bound and reads almost nothing from memory.  Those blocks read only the prepared weight, so
they neither take a ring epoch nor wait on the kernel that produced the batch, and they lead the grid
so that they are placed first: the warm therefore begins while the previous call's merge still owns
the machine, not when the exchange starts.

The GEMM releases the merge as soon as it has waited on the scatter instead of after its reduction.
The merge reads the accumulator only after its own dependent-launch wait, which waits for the whole
GEMM grid to retire, so the reduction is still safe and the merge spends its launch and its whole
pre-projection preamble inside the GEMM's runtime.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
from triton.tools.tensor_descriptor import TensorDescriptor

_WORLD = 4
_MCOL = _WORLD * 32
_workspaces = {}
_descriptors = {}


@lru_cache(maxsize=1)
def _native():
    import dpo_native

    return dpo_native


@triton.jit
def _shard_gemm(x_desc, w_desc, ws_ptr, M, N, k_per_split,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """acc[m, n] += X[m, K-chunk s] @ W[n, K-chunk s]^T over every split; rows past M are masked.

    The K splits reduce against each other with relaxed device-scope adds instead of each landing
    its own slab.  Ordering across splits is therefore unspecified and the fp32 sum is not
    bit-reproducible; the NVFP4 match ratio is what bounds that.  The accumulator must be zero on
    entry, which the merge kernel guarantees by re-zeroing exactly what it reads."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k0 = pid_k * k_per_split
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    # programmatic dependent launch (wide band only): the scatter kernel triggers before landing
    # the peers' rows; both controls are no-ops when the launch carries no PDL attribute
    gdc_wait()
    # The merge may be released here rather than after the reduction.  It reads this accumulator
    # only after its own griddepcontrol.wait, which waits for this whole grid to retire, so no split
    # can still be adding when it reads and re-zeroes; what it gains is its launch and its entire
    # pre-projection preamble inside this kernel's runtime.
    gdc_launch_dependents()
    for kk in range(0, k_per_split, BK):
        xt = x_desc.load([0, k0 + kk])
        w = w_desc.load([pid_n * BN, k0 + kk])
        acc = tl.dot(xt, w.T, acc)
    rm = tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    tl.atomic_add(ws_ptr + rm[:, None] * N + rn[None, :], acc,
                  mask=(rm[:, None] < M) & (rn[None, :] < N), sem="relaxed")


# (BN, BK, splits, warps, stages) per gathered-row band; the split count is derived from K below.
_CFG_WIDE = (128, 64, 12, 4, 6)     # M > 64  (BM = 128)
_CFG_NARROW = (64, 128, 12, 4, 4)   # M <= 64 (BM = 32 or 64)


def _gemm_plan(m, k):
    bn, bk, s, warps, stages = _CFG_WIDE if m > 64 else _CFG_NARROW
    bm = 128 if m > 64 else 64 if m > 32 else 32
    kps = triton.cdiv(triton.cdiv(k, s), bk) * bk
    return bm, bn, bk, kps, triton.cdiv(k, kps), warps, stages


def _descriptor(t, shape, strides, block):
    key = (t.data_ptr(), tuple(shape), tuple(strides), tuple(block))
    d = _descriptors.get(key)
    if d is None:
        d = _descriptors[key] = TensorDescriptor(t, list(shape), list(strides), list(block))
    return d


def _shard_gemm_launch(gx, shard, acc):
    """Split-K projection of the gathered rows onto this rank's weight quarter, reduced into acc."""
    m, k = gx.shape
    n = shard.shape[0]
    bm, bn, bk, kps, s, warps, stages = _gemm_plan(m, k)
    xd = _descriptor(gx, (m, k), (gx.stride(0), 1), (bm, bk))
    wd = _descriptor(shard, (n, k), (shard.stride(0), 1), (bn, bk))
    _shard_gemm[(n // bn, s)](xd, wd, acc, m, n, kps,
                              BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages, launch_pdl=True)


@dataclass
class Prepared:
    """The canonical parameters, bound once per layer; no copy of the weight is made."""

    weight: torch.Tensor
    gamma: torch.Tensor
    epsilon: float
    quant_scale: torch.Tensor


def prepare(weight, gamma, epsilon, quant_scale):
    if weight.dtype != torch.bfloat16 or weight.ndim != 2 or weight.stride(1) != 1:
        raise ValueError("projection requires a row-major BF16 weight matrix")
    if gamma.shape != weight.shape[:1] or quant_scale.numel() not in (0, 1):
        raise ValueError("normalization or quantization parameter shape differs")
    if weight.shape[0] % (_WORLD * 8) or weight.shape[1] % 8:
        raise ValueError("hidden width must split into 16-byte aligned column shards")
    _native()
    return Prepared(weight, gamma.contiguous(), float(epsilon), quant_scale)


class Workspace:
    """Symmetric rings for both exchanges plus the private gathered rows and the shard accumulator.

    Created eagerly on the first call of a process group, before any CUDA graph capture, and
    reused by every later call and replay; the rings are re-armed by the kernels themselves.
    """

    def __init__(self, group, device, width, hidden):
        import torch.distributed._symmetric_memory as symmetric

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("projection workspace must be created before capture")
        native = _native()
        self.rings, self.handles, self.maps = [], [], []
        for size in (native.input_ring_bytes(width, hidden), native.column_ring_bytes(hidden)):
            ring = symmetric.empty(size, dtype=torch.uint8, device=device)
            ring.view(torch.int32).fill_(-2147483648)
            ring[: native.header_bytes()].zero_()
            torch.cuda.current_stream(device).synchronize()
            handle = symmetric.rendezvous(ring, group)
            if handle is None or handle.world_size != _WORLD:
                raise RuntimeError("projection requires four mapped peers")
            pointers = [handle.get_buffer(p, (size,), torch.uint8).data_ptr() for p in range(_WORLD)]
            self.rings.append(ring)
            self.handles.append(handle)
            self.maps.append(torch.tensor(pointers + [0], dtype=torch.int64, device=device))
        self.gx = torch.empty((_MCOL, width), dtype=torch.bfloat16, device=device)
        # only this rank's own output shard of every gathered row ever crosses the wire
        self.gr = torch.empty((_MCOL, hidden // _WORLD), dtype=torch.bfloat16, device=device)
        # zeroed once here; every later call is re-armed by the merge kernel it feeds
        self.acc = torch.zeros((_MCOL, hidden // _WORLD), dtype=torch.float32, device=device)


def _workspace(group, weight):
    key = (group, weight.device, tuple(weight.shape))
    ws = _workspaces.get(key)
    if ws is None:
        ws = _workspaces[key] = Workspace(group, weight.device, weight.shape[1], weight.shape[0])
    return ws


def project_gather_norm(x, residual, prepared, out, local_residual, packed, scales, group):
    """Gather rows, project this rank's output columns, merge and normalise the replicated rows."""
    if dist.get_world_size(group) != _WORLD or not 0 < x.shape[0] <= 32:
        raise ValueError("projection requires TP4 and 1..32 padded local rows")
    rank = dist.get_rank(group)
    weight, gamma = prepared.weight, prepared.gamma
    hidden, width = weight.shape
    if x.shape[1] != width or residual.shape[1] != hidden:
        raise ValueError("projection geometry is outside the prepared envelope")
    ws = _workspace(group, weight)
    native = _native()
    rows = x.shape[0]
    shard_rows = hidden // _WORLD
    shard = weight[rank * shard_rows:(rank + 1) * shard_rows]
    gx, gr = ws.gx[:_WORLD * rows], ws.gr[:_WORLD * rows]
    # The weight quarter is warmed into L2 during the exchange only on the wide band.  The narrow
    # band keeps every block pushing: its GEMM already has more slack than its exchange does, and
    # giving the fill SMs there costs more than the warm start returns.
    # The scatter releases the GEMM early only on the wide band.  On the narrow band the exchange
    # keeps every SM until it finishes, so an early release buys the GEMM nothing but resident CTAs
    # spinning on the wait; its launch still carries the attribute, which is what takes the launch
    # itself off the critical path.
    native.scatter(x, residual, gx, gr, ws.maps[0], rank, shard, rows > 16, True, rows > 16)
    _shard_gemm_launch(gx, shard, ws.acc)
    native.merge(ws.acc, gr, gamma, out, local_residual, ws.maps[1], rank, prepared.epsilon,
                 packed, scales, prepared.quant_scale, True)
