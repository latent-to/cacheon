"""Audit short-lived prefix state before the model can overwrite it.

Full-attention pages survive a forward pass; sliding-window pages and recurrent
checkpoints need not. The cache seam calls this adapter in the existing untimed
audit role. It records engine-produced state before the cache receives it, checks
device hits immediately, and checks host hits after the cache submits its loads.
The ordinary full-KV checker remains active in both serving roles.
"""

from __future__ import annotations

import secrets

from cacheon import audit
from cacheon.integrations.sglang_cache import _LAYERS, _MODULI, _STATE


def compressed_page_buffers(pool):
    """Expose compressed KV and index pages using the runtime's physical page numbering."""
    kv, index = [], []
    unified = getattr(pool, "unified_kv_pool", None)
    for ratio, part in getattr(pool, "kv_pools", {}).items():
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
            index.append(indexer.contiguous_page_row_buffers())
    return kv, index


class PrefixStateAudit:
    """State-family adaptation under the runtime-cache ABI, without model names."""

    def __init__(self, guard, ctx) -> None:
        self.guard, self.params = guard, ctx.params
        self.torch, self.requests = guard.torch, guard.requests
        self.pool = guard.allocator.get_kvcache()
        self.swa = getattr(self.pool, "swa_kv_pool", None)
        self.mamba = getattr(self.requests, "mamba_pool", None)
        self.checkpoints = getattr(self.requests, "mamba_ckpt_pool", None)
        self.ring_size = int(getattr(self.pool, "swa_req_ring_size", None) or 0)
        self.ring = getattr(self.pool, "unified_kv_pool", None) if self.ring_size else None
        if (ctx.is_hybrid_swa and self.swa is None and self.ring is None) or (ctx.is_hybrid_ssm and self.mamba is None):
            raise RuntimeError("tree_cache: state validation is unavailable for this hybrid pool")
        self.window = int(self.params.sliding_window_size or 0)
        if (self.swa is not None or self.ring is not None) and self.window <= 0:
            raise RuntimeError("tree_cache: sliding-window state has no positive window size")
        self.buffers = self._buffers()
        self.weights = {}
        self.pending = {}
        self.owned = {}

    def _buffers(self):
        """Sample the engine's transferable state, excluding request-keyed replay scratch."""
        from cacheon.integrations.sglang_dsa_state import KV_BUFFERS

        groups = {}
        if self.swa is not None:
            for name in (*KV_BUFFERS, "index_k_with_scale_buffer"):
                groups["swa", name] = list(getattr(self.swa, name, ()) or ())
        if self.mamba is not None:
            for field, buffer, _, _ in self.mamba._iter_transfer_state_entries():
                groups.setdefault(("mamba", field), []).append(buffer)
        if self.ring is not None:
            for name in ("kv_buffer", "kv_buffer_rope"):
                groups["ring", name] = [b[:self.ring.num_slots * self.ring_size]
                                        for b in getattr(self.ring, name, ()) if b is not None]
            for name in ("compress_state_pools", "indexer_compress_state_pools"):
                groups["request", name] = [p.kv_score_buffer.kv_score.reshape(
                    -1, self.pool.get_ring_size(p.ratio) * p.kv_score_buffer.kv_score.shape[-1],
                ) for p in getattr(self.pool, name, ()) if p is not None]
        chosen = {}
        draw = secrets.SystemRandom()
        for kind, buffers in groups.items():
            buffers = [b for b in buffers if self.torch.is_tensor(b) and b.numel()]
            for index in sorted(draw.sample(range(len(buffers)), min(_LAYERS, len(buffers)))):
                chosen[(*kind, index)] = buffers[index]
        for family, present in (("swa", self.swa), ("mamba", self.mamba), ("ring", self.ring)):
            if present is not None and not any(key[0] == family for key in chosen):
                raise RuntimeError(f"tree_cache: no transferable {family} state recognized")
        return chosen

    def _state(self, kind, indices, *, checkpoint=False, encode=False):
        """Read the declared representation; encoded checkpoints have their own slot IDs."""
        buffer = self.buffers[kind]
        if kind[:2] == ("swa", "index_k_with_scale_buffer"):
            indices = indices[:, ::self.swa.page_size] // self.swa.page_size
        if checkpoint:
            field, index = kind[1:]
            if field == "temporal":
                buffer = self.checkpoints.temporal.qdata[index]
            elif field == "conv":
                layers = self.checkpoints.conv[0].shape[0]
                buffer = self.checkpoints.conv[index // layers][index % layers]
            else:
                raise RuntimeError(f"tree_cache: checkpoint codec does not carry {field}")
        if bool(((indices < 0) | (indices >= buffer.shape[0])).any()):
            self.guard.refuse("a prefix names state slots outside the engine's pool")
        raw = buffer.index_select(0, indices.reshape(-1))
        if self.checkpoints is not None and kind[:2] == ("mamba", "temporal"):
            if checkpoint:
                scale = self.checkpoints.temporal.scale[kind[2]].index_select(0, indices.reshape(-1))
                raw = (raw.float() * scale.float()).to(self.buffers[kind].dtype)
            elif encode:
                # Independent expression of the pinned codec: scale is rounded
                # before division, with nearest-even rounding and symmetric int8.
                scale = (raw.float().abs().amax(-2, keepdim=True).clamp(min=1e-8) / 127).to(raw.dtype)
                quantized = (raw.float() / scale.float()).round().clamp(-127, 127)
                raw = (quantized * scale.float()).to(raw.dtype)
        return raw.contiguous()

    def _signature(self, kind, indices, *, checkpoint=False, encode=False):
        """Fingerprint served bytes after any validator-declared checkpoint codec."""
        torch = self.torch
        raw = self._state(kind, indices, checkpoint=checkpoint, encode=encode)
        raw = raw.view(torch.uint8).reshape(len(indices), -1)
        words = raw.view(torch.int32) if raw.shape[1] % 4 == 0 else raw.to(torch.int32)
        if kind not in self.weights:
            self.weights[kind] = torch.randint(
                1, _MODULI[0][0], (words.shape[1],), device=self.guard.device,
                generator=self.guard.draws, dtype=torch.int64,
            )
        prime = _MODULI[0][0]
        return (words.to(torch.int64) * self.weights[kind] % prime).sum(1) % prime

    def _pairs(self, req, ids, lengths, indices, family, *, record, checkpoint=False):
        """Bind each state fingerprint to the namespaced token prefix it resumes."""
        for kind in self.buffers:
            if kind[0] != family:
                continue
            namespace = (req.extra_key, req.cache_salt or None, *kind)
            digests = self.guard._digests(ids, namespace, lengths)
            value = self._signature(kind, indices, checkpoint=checkpoint, encode=record)
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
        begin = (max(start, int(req.kv.swa_evicted_seqlen)) if record
                 else max(0, len(row) - self.window))
        first, last = (begin + page - 1) // page, end // page
        if first >= last:
            return
        pages = torch.arange(first, last, device=self.guard.device)
        slots = row[(pages * page)[:, None] + torch.arange(page, device=self.guard.device)]
        slots = self.pool.translate_loc_from_full_to_swa(slots.reshape(-1)).reshape_as(slots).long()
        lengths = (pages + 1) * page + self.guard.bigram
        self._pairs(req, ids, lengths, slots, "swa", record=record)

    def _physical(self, value):
        if value is None:
            self.guard.refuse("a served recurrent prefix has no checkpoint slot")
        index = self.torch.as_tensor(value, device=self.guard.device).reshape(-1)
        if len(index) != 1:
            self.guard.refuse("a recurrent prefix must name exactly one checkpoint")
        return self.requests.translate_mamba_indices(index).long()

    def record(self, req, ids, row, held, *, finished):
        """Record reusable checkpoints and preserve an unfinished request's live state."""
        if audit._rate() <= 0:
            return
        if req.rid in self.pending:
            self.guard.refuse("a host prefix was consumed before its state audit completed")
        self._swa(req, ids, row, record=True, start=max(held.own, held.recorded))
        if self.ring is not None and not finished:
            self.owned[req.rid] = self._ring_snapshot(req, len(row))
        if self.mamba is None:
            return
        extra = self.params.enable_mamba_extra_buffer
        length = req.kv.mamba_last_track_seqlen if extra else len(ids)
        if finished and not extra and self.mamba.replayssm_write_pos is not None:
            length -= int(self.mamba.replayssm_write_pos[req.kv.mamba_pool_idx].item())
        if length is not None and not 0 <= length <= len(ids):
            raise RuntimeError("tree_cache: engine checkpoint boundary is outside the committed prefix")
        if length:
            value = req.kv.mamba_pool_idx
            if extra:
                keep = self.requests.get_mamba_ping_pong_keep_idx(req)
                value = req.kv.mamba_ping_pong_track_buffer[keep]
            lengths = self.torch.tensor([length], device=self.guard.device)
            self._pairs(req, ids, lengths, self._physical(value), "mamba", record=True)
        if not finished:
            active = self._physical(req.kv.mamba_pool_idx)
            self.owned[req.rid] = {kind: self._signature(kind, active)
                                   for kind in self.buffers if kind[0] == "mamba"}

    def _check(self, req, ids, row, *, loaded_mamba=False):
        self._swa(req, ids, row, record=False)
        if self.mamba is not None:
            source = None if loaded_mamba else req.kv.mamba_cow_src_index
            if source is None:
                source = req.kv.mamba_pool_idx
            lengths = self.torch.tensor([len(row) + self.guard.bigram], device=self.guard.device)
            checkpoint = self.checkpoints is not None and not loaded_mamba
            indices = (self.torch.as_tensor(source, device=self.guard.device).reshape(-1).long()
                       if checkpoint else self._physical(source))
            self._pairs(req, ids, lengths, indices, "mamba", record=False, checkpoint=checkpoint)
        self.guard.publish()

    def matched(self, params, result):
        """Check a served prefix before the first forward consumes its short-lived state."""
        req = params.req
        if audit._rate() <= 0 or req is None or (self.mamba is not None and not params.cow_mamba):
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
        if result.host_hit_length or result.swa_host_hit_length or result.mamba_host_hit_length:
            self.pending[req.rid] = (req, ids, length, bool(result.mamba_host_hit_length))
        else:
            self._check(req, ids, result.device_indices)

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
        for req, ids, length, loaded_mamba in self.pending.values():
            if len(req.prefix_indices) < length:
                self.guard.refuse("a host load returned fewer prefix slots than it claimed")
            self._check(req, ids, req.prefix_indices[:length], loaded_mamba=loaded_mamba)
        self.pending.clear()
        self.guard.poll(block=True)

    def settled(self, req, row):
        """Check the live state a cache hands back to an unfinished request."""
        if audit._rate() <= 0 or row is None:
            return
        after = self.requests.req_to_token[req.kv.req_pool_idx, :len(row)].long()
        self._swa(req, req.get_fill_ids(), after, record=False, computed=True)
        before = self.owned.pop(req.rid, {})
        if self.ring is not None:
            after = self._ring_snapshot(req, len(row))
            for kind, value in before.items():
                self.guard._flag(_STATE, after[kind] != value)
        elif before:
            active = self._physical(req.kv.mamba_pool_idx)
            for kind, value in before.items():
                self.guard._flag(_STATE, self._signature(kind, active) != value)
        self.guard.publish()

    def _ring_snapshot(self, req, length):
        """Preserve live ring rows and request-scoped compressor state across cache calls."""
        slot = self.torch.tensor([req.kv.req_pool_idx], device=self.guard.device)
        positions = self.torch.arange(max(0, length - self.window), length, device=self.guard.device)
        rows = (slot * self.ring_size + positions % self.ring_size)[None]
        return {kind: self._signature(kind, rows if kind[0] == "ring" else slot)
                for kind in self.buffers if kind[0] in ("ring", "request")}

    def reset(self):
        """Forget pending handoffs after the owning guard drains its verdict."""
        self.pending.clear()
        self.owned.clear()
