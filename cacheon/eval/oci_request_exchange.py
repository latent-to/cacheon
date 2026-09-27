"""Host-clocked request exchange shared by batch and concurrent replay drivers.

Only the attached transport owns the file descriptors; the OCI manager still
owns process cleanup. This module replaces the blocking single-request exchange
without changing the token evidence format or trusting worker timestamps.
"""

from __future__ import annotations

import asyncio
import os
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from cacheon.eval.oci_outer_session import (
    BatchExecutionEvidence,
    AttachedSessionTransport,
    OuterSessionInfrastructureError,
    OuterSessionProtocolError,
    OuterSessionTimeoutError,
    SessionTransport,
    _now,
    _worker_error,
)
from cacheon.eval.oci_session_protocol import (
    CONTROL_MAGIC,
    EVIDENCE_MAGIC,
    FRAME_HEADER_BYTES,
    MAX_BATCH_REQUEST_BYTES,
    MAX_BATCH_RESPONSE_BYTES,
    MAX_CONTROL_BYTES,
    BatchRequest,
    BatchEvidence,
    SessionProtocolError,
    SlotAuditPolicy,
    _EVIDENCE_BINDING,
    decode_evidence_payload,
    decode_message,
    expected_evidence_payload_bytes,
    frame_message,
    parse_error_message,
    validate_audit_evidence,
)
from cacheon.eval.phase_latency import HostTokenClock


async def _ready(transport, *, writing: bool, deadline: float) -> None:
    loop = asyncio.get_running_loop()
    fd = transport._stdin_fd if writing else transport._stdout_fd
    add = loop.add_writer if writing else loop.add_reader
    remove = loop.remove_writer if writing else loop.remove_reader
    future = loop.create_future()

    def ready():
        remove(fd)
        if not future.done():
            future.set_result(None)

    try:
        add(fd, ready)
        await asyncio.wait_for(future, transport._remaining(deadline))
    except asyncio.TimeoutError:
        direction = "request write" if writing else "response read"
        raise OuterSessionTimeoutError(f"session {direction} timed out") from None
    except OSError as exc:
        raise transport._process_error(f"session pipe readiness failed: {exc}") from None
    finally:
        remove(fd)


async def write_frame(transport: AttachedSessionTransport, frame: bytes, *, deadline: float) -> None:
    """Write a complete request without blocking other in-flight responses."""
    transport._require_client()
    if not isinstance(frame, bytes) or not frame:
        raise OuterSessionInfrastructureError("session request frame is invalid")
    view = memoryview(frame)
    while view:
        await _ready(transport, writing=True, deadline=deadline)
        try:
            count = os.write(transport._stdin_fd, view)
        except (BlockingIOError, InterruptedError):
            continue
        except OSError as exc:
            raise transport._process_error(f"session request write failed: {exc}") from None
        if count <= 0:
            raise transport._process_error("session request write made no progress")
        view = view[count:]


async def read_exact(transport: AttachedSessionTransport, size: int, *, deadline: float) -> bytes:
    """Read bounded bytes; process reaping remains exclusively manager-owned."""
    transport._require_client()
    chunks = []
    while size:
        await _ready(transport, writing=False, deadline=deadline)
        try:
            chunk = os.read(transport._stdout_fd, min(size, 1 << 20))
        except (BlockingIOError, InterruptedError):
            continue
        except OSError as exc:
            raise transport._process_error(f"session response read failed: {exc}") from None
        if not chunk:
            raise transport._process_error("session ended before a complete response")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


async def _header(transport, deadline):
    header = await read_exact(transport, FRAME_HEADER_BYTES, deadline=deadline)
    return header[:4], struct.unpack(">I", header[4:])[0]


async def read_control(transport: AttachedSessionTransport, *, max_bytes: int, deadline: float) -> dict:
    """Decode a control message without consuming an evidence frame."""
    magic, size = await _header(transport, deadline)
    if magic != CONTROL_MAGIC:
        raise OuterSessionProtocolError("worker emitted wrong control-frame magic")
    if size > max_bytes:
        raise OuterSessionProtocolError("worker declared an oversized control frame")
    try:
        return decode_message(await read_exact(transport, size, deadline=deadline), max_bytes=max_bytes)
    except SessionProtocolError as exc:
        raise OuterSessionProtocolError(str(exc)) from None


async def read_response(
    transport: AttachedSessionTransport, requests: Mapping[str, BatchRequest], *,
    deadline: float, on_progress: Callable[[dict], None] | None = None,
) -> tuple[BatchRequest, BatchEvidence]:
    """Route one complete response by its existing request binding, before allocation."""
    try:
        while True:
            magic, size = await _header(transport, deadline)
            if magic == CONTROL_MAGIC:
                if size > MAX_CONTROL_BYTES:
                    raise OuterSessionProtocolError("worker declared an oversized control frame")
                message = decode_message(
                    await read_exact(transport, size, deadline=deadline), max_bytes=MAX_CONTROL_BYTES
                )
                request_id = message.get("request_id")
                request = requests.get(request_id) if type(request_id) is str else None
                if request is None:
                    raise OuterSessionProtocolError("worker control has no disclosed request binding")
                detail = parse_error_message(message, session_id=request.session_id,
                                             launch_digest=request.launch_digest, request=request)
                if detail is not None:
                    raise _worker_error(detail, diagnostic_provider=transport._diagnostic_provider())
                if on_progress is None or not request.measure_phase_latency:
                    raise OuterSessionProtocolError("worker emitted an early control frame")
                on_progress(message)
                continue
            if magic != EVIDENCE_MAGIC:
                raise OuterSessionProtocolError("worker emitted wrong evidence-frame magic")
            if not _EVIDENCE_BINDING.size <= size <= MAX_BATCH_RESPONSE_BYTES:
                raise OuterSessionProtocolError("worker evidence frame has the wrong exact size")
            prefix = await read_exact(transport, _EVIDENCE_BINDING.size, deadline=deadline)
            request_id = _EVIDENCE_BINDING.unpack(prefix)[2].hex()
            request = requests.get(request_id)
            if request is None:
                raise OuterSessionProtocolError("worker evidence has no disclosed request binding")
            if size != expected_evidence_payload_bytes(request):
                raise OuterSessionProtocolError("worker evidence frame has the wrong exact size")
            payload = prefix + await read_exact(transport, size - len(prefix), deadline=deadline)
            return request, decode_evidence_payload(payload, request=request)
    except SessionProtocolError as exc:
        raise OuterSessionProtocolError(str(exc)) from None


@dataclass
class _Pending:
    started: float
    tokens: HostTokenClock
    future: asyncio.Future[BatchExecutionEvidence]
    progress: Callable[[dict], None] | None


class RequestExchange:
    """One read owner and one frame writer for concurrent disclosed requests."""

    def __init__(
        self, transport: SessionTransport, *, clock: Callable[[], float],
        deadline: float, audit_policy: SlotAuditPolicy | None = None,
    ):
        self.transport, self.clock, self.deadline = transport, clock, deadline
        self.audit_policy = audit_policy
        self.requests: dict[str, BatchRequest] = {}
        self.pending: dict[str, _Pending] = {}
        self.writer = asyncio.Lock()
        self.reader = None
        self.failure = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        if self.reader is not None:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        for item in self.pending.values():
            item.future.cancel()
        self.pending.clear()
        self.requests.clear()

    def _progress(self, message):
        item = self.pending[message["request_id"]]
        item.tokens.observe(message)
        if item.progress is not None:
            item.progress(message)

    async def _read(self):
        try:
            while self.requests:
                request, evidence = await self.transport.aread_response(
                    self.requests, deadline=self.deadline, on_progress=self._progress
                )
                item = self.pending[request.request_id]
                receipts = ()
                if self.audit_policy is not None:
                    receipts = validate_audit_evidence(
                        await self.transport.aread_control(max_bytes=MAX_CONTROL_BYTES, deadline=self.deadline),
                        request=request, policy=self.audit_policy,
                    )
                if evidence.observed_tokens != request.prompt_count * request.max_new_tokens:
                    raise OuterSessionProtocolError("worker evidence token count is not exact")
                completed = _now(self.clock, previous=item.started)
                if completed <= item.started:
                    raise OuterSessionInfrastructureError("host batch clock did not advance")
                row = BatchExecutionEvidence(
                    request.batch_index, request.request_id, request.nonce, item.started, completed,
                    evidence.observed_tokens, evidence, receipts,
                    item.tokens.finish(evidence, completed) if request.measure_phase_latency else (),
                )
                del self.requests[request.request_id]
                del self.pending[request.request_id]
                item.future.set_result(row)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if isinstance(exc, SessionProtocolError):
                exc = OuterSessionProtocolError(str(exc))
            self.failure = exc
            for item in self.pending.values():
                if not item.future.done():
                    item.future.set_exception(exc)

    async def execute(
        self, request: BatchRequest, *, deadline: float,
        on_progress: Callable[[dict], None] | None = None,
    ) -> BatchExecutionEvidence:
        """Measure from host admission through exact final evidence, including pipe wait."""
        if self.failure is not None:
            raise self.failure
        deadline = min(deadline, self.deadline)
        if request.request_id in self.pending or (self.audit_policy is not None and self.pending):
            raise OuterSessionInfrastructureError("duplicate request or concurrent eager audit")
        started = _now(self.clock)
        future = asyncio.get_running_loop().create_future()
        self.requests[request.request_id] = request
        self.pending[request.request_id] = _Pending(
            started, HostTokenClock(request, self.clock, started), future, on_progress
        )
        try:
            async with self.writer:
                await self.transport.awrite_frame(
                    frame_message(request.to_dict(), max_bytes=MAX_BATCH_REQUEST_BYTES), deadline=deadline
                )
            if self.reader is None or self.reader.done():
                self.reader = asyncio.create_task(self._read())
            try:
                return await asyncio.wait_for(asyncio.shield(future), max(0, deadline - _now(self.clock)))
            except asyncio.TimeoutError:
                raise OuterSessionTimeoutError("session response read timed out") from None
        except BaseException as exc:
            self.failure = exc
            future.cancel()
            for item in self.pending.values():
                if not item.future.done():
                    item.future.set_exception(exc)
            if self.reader is not None:
                self.reader.cancel()
            raise
