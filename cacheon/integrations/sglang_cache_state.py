"""Audit short-lived prefix state before the model can overwrite it.

Full-attention pages survive a forward pass; sliding-window pages need not. The
cache seam calls this adapter in the existing untimed audit role. It records
engine-produced state before the cache receives it, checks device hits
immediately, and checks host hits after the cache submits its loads. The
ordinary full-KV checker remains active in both serving roles.
"""

from __future__ import annotations

import secrets
import sys

from cacheon import audit
from cacheon.integrations.sglang_cache import _LAYERS, _MODULI, _STATE

_COMPONENTS = "sglang.srt.mem_cache.unified_cache.components"


def compressed_page_buffers(pool):
    """Expose compressed KV and index pages using the runtime's physical page numbering.

    An index page holds fewer slots than a full page; each index entry carries
    its buffers with their rows per full page.
    """
    kv, index = [], []
    unified = getattr(pool, "unified_kv_pool", None)
    for ratio, part in getattr(pool, "kv_pools", {}).items():
        if part is None:
            continue
        if unified is None:
            kv.append(part.kv_buffer)
        else:
            ratios = pool.compression_ratios[pool._stage_start:pool._stage_end]
            for name in ("kv_buffer", "kv_buffer_rope"):
                views = []
                for layer, buffer in enumerate(getattr(unified, name, ())):
                    if buffer is None or ratios[layer] != ratio:
                        continue
                    tail = buffer[unified.swa_pages:]
                    views.append(tail.reshape(-1, (pool.page_size // ratio) * buffer.shape[-1]))
                kv.append(views)
        indexer = pool.index_pools.get(ratio)
        if indexer is not None:
            per_page = (int(pool.page_size) // ratio) // int(indexer.page_size)
            index.append((indexer.contiguous_page_row_buffers(), per_page))
    return kv, index


def _evicted(req) -> int:
    """The window positions the runtime already let go of, by its own component name."""
    components = sys.modules.get(_COMPONENTS)
    if components is None:
        raise RuntimeError("tree_cache: the runtime's cache components are not loaded")
    return int(req.kv.get_evicted_seqlen(components.ComponentType.SWA))


class PrefixStateAudit:
    """State-family adaptation under the runtime-cache ABI, without model names."""

    def __init__(self, guard, ctx) -> None:
        self.guard, self.params = guard, ctx.params
        self.torch, self.requests = guard.torch, guard.requests
        self.pool = guard.allocator.get_kvcache()
        self.swa = getattr(self.pool, "swa_kv_pool", None)
        self.ring_size = int(getattr(self.pool, "swa_req_ring_size", None) or 0)
        self.ring = getattr(self.pool, "unified_kv_pool", None) if self.ring_size else None
        self.mamba = getattr(self.requests, "mamba_pool", None) if ctx.is_hybrid_ssm else None
        state = getattr(self.mamba, "mamba_cache", None)
        # Unknown layouts stay refused rather than serving unchecked state. Int8
        # checkpoints and ReplaySSM have different stored bytes and boundaries.
        if (ctx.is_hybrid_ssm and (state is None
                or getattr(state, "intermediate_ssm", None) is not None
                or getattr(self.requests, "mamba_ckpt_pool", None) is not None
                or getattr(self.mamba, "enable_linear_replayssm", False)
                or getattr(self.mamba, "enable_linear_replayssm_spec", False))
                or (ctx.is_hybrid_swa and self.swa is None and self.ring is None)):
            raise RuntimeError("tree_cache: state validation is unavailable for this hybrid pool")
        self.window = int(self.params.sliding_window_size or 0)
        if (self.swa is not None or self.ring is not None) and self.window <= 0:
            raise RuntimeError("tree_cache: sliding-window state has no positive window size")
        self.buffers = self._buffers()
        # A window pool that stores whole pages has fewer rows than slots.
        self.paged = {kind for kind, buffer in self.buffers.items()
                      if kind[0] == "swa" and buffer.shape[0] < self.swa.size}
        self.live = any(kind[0] in ("ring", "request", "mamba") for kind in self.buffers)
        self.weights = {}
        self.pending = {}
        self.owned = {}

    def _buffers(self):
        """Sample transferable window/checkpoint buffers and live request state."""
        from cacheon.integrations.sglang_dsa_state import KV_BUFFERS

        groups = {}
        if self.mamba is not None:
            state = self.mamba.mamba_cache
            groups["mamba", "temporal"] = list(state.temporal.unbind(0))
            for index, buffer in enumerate(state.conv):
                groups["mamba", f"conv{index}"] = list(buffer.unbind(0))
        if self.swa is not None:
            for name in (*KV_BUFFERS, "index_k_with_scale_buffer"):
                groups["swa", name] = list(getattr(self.swa, name, ()) or ())
        if self.ring is not None:
            for name in ("kv_buffer", "kv_buffer_rope"):
                groups["ring", name] = [b[:self.ring.num_slots * self.ring_size]
                                        for b in getattr(self.ring, name, ()) if b is not None]
        for name in ("compress_state_pools", "indexer_compress_state_pools"):
            groups["request", name] = [
                p.kv_score_buffer.kv_score[:self.pool.num_req_slots * p.ring_size].reshape(
                    self.pool.num_req_slots, -1)
                for p in getattr(self.pool, name, ())
                if p is not None and (self.ring is not None or getattr(p, "request_scoped", False))
            ]
        chosen = {}
        draw = secrets.SystemRandom()
        for kind, buffers in groups.items():
            buffers = [b for b in buffers if self.torch.is_tensor(b) and b.numel()]
            for index in sorted(draw.sample(range(len(buffers)), min(_LAYERS, len(buffers)))):
                chosen[(*kind, index)] = buffers[index]
        for family, present in (("swa", self.swa), ("ring", self.ring), ("mamba", self.mamba)):
            if present is not None and not any(key[0] == family for key in chosen):
                raise RuntimeError(f"tree_cache: no transferable {family} state recognized")
        return chosen

    def _state(self, kind, indices):
        """Read the rows a prefix names, in the engine's declared representation."""
        buffer = self.buffers[kind]
        if kind in self.paged:
            indices = indices[:, ::self.swa.page_size] // self.swa.page_size
        if bool(((indices < 0) | (indices >= buffer.shape[0])).any()):
            self.guard.refuse("a prefix names state slots outside the engine's pool")
        return buffer.index_select(0, indices.reshape(-1)).contiguous()

    def _signature(self, kind, indices):
        """Fingerprint the served bytes of one state kind at the named rows."""
        torch = self.torch
        raw = self._state(kind, indices)
        raw = raw.view(torch.uint8).reshape(len(indices), -1)
        words = raw.view(torch.int32) if raw.shape[1] % 4 == 0 else raw.to(torch.int32)
        if kind not in self.weights:
            self.weights[kind] = torch.randint(
                1, _MODULI[0][0], (words.shape[1],), device=self.guard.device,
                generator=self.guard.draws, dtype=torch.int64,
            )
        prime = _MODULI[0][0]
        return (words.to(torch.int64) * self.weights[kind] % prime).sum(1) % prime

    def _pairs(self, req, ids, lengths, indices, family, *, record):
        """Bind each state fingerprint to the namespaced token prefix it resumes."""
        for kind in self.buffers:
            if kind[0] != family:
                continue
            namespace = (req.extra_key, req.cache_salt or None, *kind)
            digests = self.guard._digests(ids, namespace, lengths)
            value = self._signature(kind, indices)
            cells = self.guard._cells(digests, value)
            if record:
                for cell in cells:
                    self.guard.table[cell] = True
            else:
                valid = self.guard.table[cells[0]] & self.guard.table[cells[1]]
                self.guard._flag(_STATE, ~valid)

    def _swa(self, req, ids, row, *, record, start=0, computed=False):
        if self.swa is None:
            return
        torch, page = self.torch, self.guard.page
        end = len(row) - (self.guard.bigram if record or computed else 0)
        # Record every computed page still resident: a later branch can reuse an
        # earlier window. On reads, require the whole live window regardless of
        # eviction metadata supplied by the cache.
        begin = max(start, _evicted(req)) if record else max(0, len(row) - self.window)
        # DSV4.1's window is smaller than a page: rounding its read start up
        # skips that whole live window. Include the first partially used page.
        first = (begin + page - 1) // page if record else begin // page
        last = end // page
        if first >= last:
            return
        pages = torch.arange(first, last, device=self.guard.device)
        slots = row[(pages * page)[:, None] + torch.arange(page, device=self.guard.device)]
        slots = self.pool.translate_loc_from_full_to_swa(slots.reshape(-1)).reshape_as(slots).long()
        lengths = (pages + 1) * page + self.guard.bigram
        self._pairs(req, ids, lengths, slots, "swa", record=record)

    def record(self, req, ids, row, held, *, finished):
        """Record computed windows and preserve an unfinished request's live ring state."""
        if audit._rate() <= 0:
            return
        if req.rid in self.pending:
            self.guard.refuse("a host prefix was consumed before its state audit completed")
        self._swa(req, ids, row, record=True, start=max(held.own, held.recorded))
        if self.mamba is not None:
            extra = self.params.enable_mamba_extra_buffer
            length = req.kv.mamba_last_track_seqlen if extra else len(ids)
            if length is not None and length > max(held.own, held.recorded):
                if length > len(ids):
                    self.guard.refuse("a recurrent checkpoint extends beyond computed tokens")
                slot = (req.kv.mamba_ping_pong_track_buffer[
                    self.requests.get_mamba_ping_pong_keep_idx(req)] if extra else req.kv.mamba_pool_idx)
                self._mamba_pair(req, ids, length, slot, record=True)
        if self.live and not finished:
            self.owned[req.rid] = self._ring_snapshot(req, len(row))

    def _check(self, req, ids, row, *, recurrent):
        self._swa(req, ids, row, record=False)
        if recurrent:
            slot = req.kv.mamba_cow_src_index
            # A native host load can restore directly into the request slot;
            # device matches defer their copy to the next forward pass.
            if slot is None:
                slot = req.kv.mamba_pool_idx
            if req.kv.mamba_needs_clear:
                self.guard.refuse("a recurrent prefix would clear its restored state")
            self._mamba_pair(req, ids, len(row), slot, record=False)
        self.guard.publish()

    def _mamba_slots(self, slot):
        """Translate checkpoint and request indices to the engine's physical rows."""
        if slot is None:
            self.guard.refuse("a recurrent prefix has no state slot")
        slots = self.torch.as_tensor(slot, device=self.guard.device, dtype=self.torch.int64).reshape(-1)
        if slots.numel() != 1 or bool((slots < 0).any()):
            self.guard.refuse("a recurrent prefix must name one valid state slot")
        return self.requests.translate_mamba_indices(slots).long().reshape(1, 1)

    def _mamba_pair(self, req, ids, length, slot, *, record):
        """Bind a recurrent checkpoint to the complete prefix that produced it."""
        lengths = self.torch.tensor([length], device=self.guard.device)
        self._pairs(req, ids, lengths, self._mamba_slots(slot), "mamba", record=record)

    def matched(self, params, result):
        """Check a served prefix before the first forward consumes its short-lived state."""
        req = params.req
        if audit._rate() <= 0 or req is None:
            return
        length = len(result.device_indices) + result.host_hit_length
        if not length:
            return
        if self.ring is not None:
            # The native ring is request-owned and never stored in tree nodes;
            # the model requires its trailing window to be freshly computed.
            if length > max(0, len(req.get_fill_ids()) - self.window):
                self.guard.refuse("a request-local ring requires replay of its trailing window")
            return
        ids = list(params.key.raw_token_ids())
        # Cache insertion also matches with req set but cow_mamba=False. That
        # lookup only rearranges KV: live recurrent state may be past the last
        # checkpoint and is preserved by settled(), not restored from this hit.
        recurrent = self.mamba is not None and params.cow_mamba
        if (result.host_hit_length or result.swa_host_hit_length
                or getattr(result, "mamba_host_hit_length", 0)):
            self.pending[req.rid] = (req, ids, length, recurrent)
        else:
            self._check(req, ids, result.device_indices, recurrent=recurrent)

    def ready(self, cache):
        """Drain submitted native loads only in the untimed audit, then check their destinations."""
        if not self.pending:
            return
        if self.guard.device.type == "cuda":
            controller = getattr(cache, "cache_controller", None)
            transfer = getattr(controller, "l2_transfer_engine", None)
            stream = getattr(transfer, "host_to_device_stream", None)
            if stream is None:
                raise RuntimeError("tree_cache: no native transfer stream for the hybrid host audit")
            # Wait only on the submitted copies, never on unrelated model/NCCL streams.
            self.torch.cuda.current_stream(self.guard.device).wait_stream(stream)
        for req, ids, length, recurrent in self.pending.values():
            if len(req.prefix_indices) < length:
                self.guard.refuse("a host load returned fewer prefix slots than it claimed")
            self._check(req, ids, req.prefix_indices[:length], recurrent=recurrent)
        self.pending.clear()
        self.guard.poll(block=True)

    def settled(self, req, row):
        """Check the live state a cache hands back to an unfinished request."""
        if audit._rate() <= 0 or row is None:
            return
        after = self.requests.req_to_token[req.kv.req_pool_idx, :len(row)].long()
        self._swa(req, req.get_fill_ids(), after, record=False, computed=True)
        before = self.owned.pop(req.rid, {})
        if self.live:
            after = self._ring_snapshot(req, len(row))
            for kind, value in before.items():
                self.guard._flag(_STATE, after[kind] != value)
        self.guard.publish()

    def _ring_snapshot(self, req, length):
        """Preserve live ring rows and request-scoped compressor state across cache calls."""
        slot = self.torch.tensor([req.kv.req_pool_idx], device=self.guard.device)
        rows = slot
        if self.ring is not None:
            positions = self.torch.arange(max(0, length - self.window), length, device=self.guard.device)
            rows = (slot * self.ring_size + positions % self.ring_size)[None]
        snapshot = {kind: self._signature(kind, rows if kind[0] == "ring" else slot)
                for kind in self.buffers if kind[0] in ("ring", "request")}
        if self.mamba is not None:
            rows = self._mamba_slots(req.kv.mamba_pool_idx)
            snapshot.update({kind: self._signature(kind, rows)
                             for kind in self.buffers if kind[0] == "mamba"})
        return snapshot

    def reset(self):
        """Forget pending handoffs after the owning guard drains its verdict."""
        self.pending.clear()
        self.owned.clear()
