"""Column-sharded output projection with fused distributed MLP preparation.

The entry fills validator-owned outputs and performs no framework patching.
The CUDA implementation is the preserved V1 algorithm, built by Cacheon's
existing native builder. PyTorch supplies the vendor dense projection.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F

_workspaces = {}


@lru_cache(maxsize=1)
def _native():
    import dp_projection_native

    return dp_projection_native


@dataclass
class Prepared:
    """Retain canonical weights without copying or mutating model storage."""

    weight: torch.Tensor
    gamma: torch.Tensor
    epsilon: float
    quant_scale: torch.Tensor


def prepare(weight, gamma, epsilon, quant_scale):
    """Bind the projection and normalization parameters once per model layer."""
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("projection requires a BF16 weight matrix")
    if gamma.shape != weight.shape[:1] or quant_scale.numel() not in (0, 1):
        raise ValueError("normalization or quantization parameter shape differs")
    _native()
    return Prepared(weight, gamma, epsilon, quant_scale)


class Workspace:
    """Keep peer mappings alive across serialized calls and graph replays."""

    def __init__(self, weight, group):
        import torch.distributed._symmetric_memory as symmetric

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("projection workspace must be warmed before capture")
        native = _native()
        hidden, width = weight.shape
        self.buffers, self.handles, self.peers = [], [], []
        for size in (native.input_arena_bytes(width, hidden), native.output_arena_bytes(hidden)):
            buffer = symmetric.empty(size, dtype=torch.uint8, device=weight.device)
            buffer.view(torch.int32).fill_(-2147483648)
            buffer[:native.payload_offset()].zero_()
            torch.cuda.current_stream(weight.device).synchronize()
            handle = symmetric.rendezvous(buffer, group)
            if handle is None or handle.world_size != 4:
                raise RuntimeError("projection requires four mapped peers")
            pointers = [handle.get_buffer(p, (size,), torch.uint8).data_ptr() for p in range(4)]
            self.buffers.append(buffer)
            self.handles.append(handle)
            self.peers.append(torch.tensor(pointers, dtype=torch.int64, device=weight.device))
        self.gathered_x = weight.new_empty((128, width))
        self.gathered_r = weight.new_empty((128, hidden))
        self.trace = torch.empty(0, dtype=torch.int64, device=weight.device)


def project_gather_norm(x, residual, prepared, out, local_residual, packed, scales, group):
    """Gather inputs, project output columns, and fill replicated normalized rows."""
    if dist.get_world_size(group) != 4 or not 0 < x.shape[0] <= 32:
        raise ValueError("projection requires TP4 and1..32 padded local rows")
    rank = dist.get_rank(group)
    weight, gamma = prepared.weight, prepared.gamma
    hidden, width = weight.shape
    if hidden not in (4096, 6144) or x.shape[1] != width or width % 8:
        raise ValueError("projection geometry is outside the measured envelope")
    key = (group, weight.device, tuple(weight.shape))
    if key not in _workspaces:
        _workspaces[key] = Workspace(weight, group)
    workspace = _workspaces[key]
    native = _native()
    rows = x.shape[0]
    gathered_x = workspace.gathered_x[:4 * rows]
    gathered_r = workspace.gathered_r[:4 * rows]
    native.push_inputs(x, residual, gathered_x, gathered_r,
                       workspace.peers[0], rank, workspace.trace)
    shard = hidden // 4
    projected = F.linear(gathered_x, weight[rank * shard:(rank + 1) * shard])
    native.column_gather_norm(
        projected, gathered_r, gamma, out, local_residual,
        workspace.peers[1], rank, prepared.epsilon, workspace.trace,
        packed, scales, prepared.quant_scale,
    )
