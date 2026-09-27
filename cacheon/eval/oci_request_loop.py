"""Run disclosed requests concurrently over the existing isolated worker pipes.

The engine owns one event loop for its lifetime. Ordinary generation may overlap;
eager audits stay serial because their rank receipts describe one request boundary.
The host still owns admission, deadlines, clocks and container destruction.
"""

from __future__ import annotations

import asyncio
import os
import struct
import sys

from cacheon.eval.engine_worker import CandidateExecutionCoverageError
from cacheon.eval.oci_session_protocol import (
    CONTROL_MAGIC,
    FRAME_HEADER_BYTES,
    MAX_BATCH_REQUEST_BYTES,
    MAX_CONTROL_BYTES,
    AuditReceiptFacts,
    BatchRequest,
    SessionProtocolError,
    SlotAuditPolicy,
    audit_evidence_message,
    decode_message,
    evidence_frame,
    frame_message,
    validate_batch_request,
)
from cacheon.eval.phase_latency import engine_outputs, generate_outputs


class RequestFailure(Exception):
    """Carry the original failure and its request back to the worker error encoder."""

    def __init__(self, request: BatchRequest, cause: BaseException):
        self.request, self.cause = request, cause
        super().__init__()


async def serve_requests(
    handle: object,
    control_fd: int,
    protocol_fd: int,
    *,
    session_id: str,
    launch_digest: str,
    audit_policy: SlotAuditPolicy | None,
) -> None:
    """Accept ordered requests while prior generations await the shared engine."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=MAX_BATCH_REQUEST_BYTES)
    transport, _ = await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader),
        os.fdopen(os.dup(control_fd), "rb", buffering=0),
    )
    output_blocking = os.get_blocking(protocol_fd)
    os.set_blocking(protocol_fd, False)
    output_lock = asyncio.Lock()
    pending: set[asyncio.Task] = set()
    reading: asyncio.Task | None = None

    async def write_frame(frame: bytes) -> None:
        view = memoryview(frame)
        while view:
            try:
                count = os.write(protocol_fd, view)
                if count <= 0:
                    raise SessionProtocolError("worker response write made no progress")
                view = view[count:]
            except InterruptedError:
                continue
            except BlockingIOError:
                ready = loop.create_future()

                def writable():
                    loop.remove_writer(protocol_fd)
                    if not ready.done():
                        ready.set_result(None)

                loop.add_writer(protocol_fd, writable)
                try:
                    await ready
                finally:
                    loop.remove_writer(protocol_fd)

    async def emit(frame: bytes) -> None:
        # Once bytes cross the pipe, cancelling a partial frame would embed the
        # failing request's error inside another request's evidence (CPU repro:
        # 80,100 declared bytes / 65,528 written). Finish that frame first. The
        # host still owns the deadline and force-destroys an unresponsive worker.
        async with output_lock:
            writing = asyncio.create_task(write_frame(frame))
            try:
                await asyncio.shield(writing)
            except asyncio.CancelledError:
                await writing
                raise

    async def read_request() -> BatchRequest:
        header = await reader.readexactly(FRAME_HEADER_BYTES)
        size = struct.unpack(">I", header[4:])[0]
        if header[:4] != CONTROL_MAGIC or size > MAX_BATCH_REQUEST_BYTES:
            raise SessionProtocolError("controller sent an invalid request frame")
        payload = await reader.readexactly(size)
        return validate_batch_request(decode_message(payload, max_bytes=MAX_BATCH_REQUEST_BYTES))

    async def execute(request: BatchRequest) -> None:
        try:
            outputs = await generate_outputs(handle.engine, request, emit)
            evidence = engine_outputs(outputs, request=request)
            collector = getattr(handle, "collect_audit_receipts", None)
            if audit_policy is not None and not callable(collector):
                raise SessionProtocolError("audited engine lacks its raw audit receipt collector")
            try:
                handle.require_completion()
            except CandidateExecutionCoverageError as exc:
                if audit_policy is None:
                    raise
                # The audit gate grades an empty policy-bound receipt set
                # NO_DECISION. Retain the original cause in worker stderr.
                print(f"CACHEON-AUDIT-NOT-COVERED: {exc}", file=sys.stderr, flush=True)
                receipts = ()
            else:
                receipts = tuple(
                    AuditReceiptFacts.from_receipt_dict(row)
                    for row in (collector() if callable(collector) else ())
                )
            frame = evidence_frame(evidence, request=request)
            if audit_policy is not None:
                frame += frame_message(
                    audit_evidence_message(request=request, policy=audit_policy, receipts=receipts),
                    max_bytes=MAX_CONTROL_BYTES,
                )
            await emit(frame)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            raise RequestFailure(request, exc) from exc

    expected_index = 0
    seen_ids: set[str] = set()
    seen_nonces: set[str] = set()
    try:
        reading = asyncio.create_task(read_request())
        while True:
            done, _ = await asyncio.wait(pending | {reading}, return_when=asyncio.FIRST_COMPLETED)
            for task in done - {reading}:
                pending.remove(task)
                task.result()
            if reading not in done:
                continue
            try:
                request = reading.result()
            except asyncio.IncompleteReadError:
                # A closed input does not erase generations already disclosed.
                # Runtime/engine errors still interrupt the loop immediately.
                await asyncio.gather(*pending)
                raise SessionProtocolError("controller closed a partial session request") from None
            if (
                request.session_id != session_id
                or request.launch_digest != launch_digest
                or request.batch_index != expected_index
                or request.request_id in seen_ids
                or request.nonce in seen_nonces
            ):
                raise RequestFailure(request, SessionProtocolError(
                    "batch ordering, session, launch, or replay binding failed"
                ))
            seen_ids.add(request.request_id)
            seen_nonces.add(request.nonce)
            expected_index += 1
            if audit_policy is None:
                pending.add(asyncio.create_task(execute(request)))
            else:
                await execute(request)
            reading = asyncio.create_task(read_request())
    finally:
        tasks = pending | ({reading} if reading is not None else set())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        transport.close()
        os.set_blocking(protocol_fd, output_blocking)
