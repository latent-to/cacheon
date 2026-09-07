"""Trusted host transport and raw timing evidence for one isolated engine.

This module owns framed byte I/O and the host clock.  It never imports an
inference runtime, interprets quality, assigns an arm role, or accepts worker
timing.  OCI policy and resource construction belong to :mod:`oci_backend`;
process creation and cleanup remain exclusively manager-owned.
"""

from __future__ import annotations

import math
import os
import secrets
import select
import struct
import time
from dataclasses import dataclass
from typing import Callable, NoReturn, Protocol, Sequence

from cacheon.eval.oci_process import (
    OCIAttachedClient,
    OCIAttachedDiagnostic,
    OCILease,
    OCIProcessError,
    OCIProcessManager,
)
from cacheon.eval.oci_session_protocol import (
    CONTROL_MAGIC,
    EVIDENCE_MAGIC,
    FRAME_HEADER_BYTES,
    MAX_BATCH_REQUEST_BYTES,
    MAX_BATCH_RESPONSE_BYTES,
    MAX_CONTROL_BYTES,
    MAX_INIT_BYTES,
    AuditReceiptFacts,
    BatchEvidence,
    BatchRequest,
    EngineSessionConfig,
    RuntimePreflightFacts,
    SessionProtocolError,
    SlotAuditPolicy,
    batch_request,
    decode_evidence_payload,
    decode_message,
    expected_evidence_payload_bytes,
    frame_message,
    make_init,
    parse_error_message,
    preflight_accept_message,
    validate_batch_request,
    validate_audit_evidence,
    validate_preflight,
    validate_ready,
)
from cacheon.stack_identity import require_sha256_hex


class OuterSessionError(RuntimeError):
    """Base error for host transport and raw session execution."""

    def __init__(
        self,
        message: str,
        diagnostic_provider: Callable[[], OCIAttachedDiagnostic] | None = None,
    ) -> None:
        super().__init__(message)
        self._message = message
        self._diagnostic_provider = diagnostic_provider

    @property
    def message(self) -> str:
        return self._message

    def attach_diagnostic(
        self, provider: Callable[[], OCIAttachedDiagnostic] | None
    ) -> None:
        """Attach host-only failure evidence without copying candidate bytes."""

        if self._diagnostic_provider is None and callable(provider):
            self._diagnostic_provider = provider

    @property
    def diagnostic(self) -> OCIAttachedDiagnostic | None:
        if self._diagnostic_provider is None:
            return None
        try:
            value = self._diagnostic_provider()
        except BaseException:
            return None
        return value if type(value) is OCIAttachedDiagnostic else None

    def __str__(self) -> str:
        diagnostic = self.diagnostic
        if diagnostic is None:
            return self._message
        return f"{self._message}; {diagnostic.summary}"


class OuterSessionInfrastructureError(OuterSessionError):
    """Trusted host, OCI lifecycle, or pre-entry runtime failure."""


class OuterSessionTimeoutError(OuterSessionInfrastructureError):
    """The one absolute session deadline expired."""


class OuterSessionProcessError(OuterSessionInfrastructureError):
    """The attached client closed a protocol pipe before completion."""


class OuterSessionProtocolError(OuterSessionError):
    """The worker emitted malformed, stale, early, or extra protocol bytes."""


class OuterSessionWorkerError(OuterSessionError):
    """The worker emitted one valid, bounded error control frame."""


class OuterSessionCandidateError(OuterSessionWorkerError):
    """A rank receipt proved that candidate code killed the worker."""

    def __init__(
        self,
        message: str,
        diagnostic_provider: Callable[[], OCIAttachedDiagnostic] | None = None,
        *,
        candidate_failure: str,
        candidate_failure_type: str = "CandidateExecutionFailure",
    ) -> None:
        super().__init__(message, diagnostic_provider)
        self.candidate_failure = candidate_failure
        self.candidate_failure_type = candidate_failure_type


class SessionTransport(Protocol):
    def start(self) -> None: ...
    def has_pending_output(self) -> bool: ...
    def write_frame(self, frame: bytes, *, deadline: float) -> None: ...
    def read_control(self, *, max_bytes: int, deadline: float) -> dict: ...
    def read_evidence(self, request: BatchRequest, *, deadline: float) -> BatchEvidence: ...
    def finalize(self) -> None: ...
    def abort(self) -> None: ...


class AttachedSessionTransport:
    """Nonblocking pipes around one manager-owned attached OCI client."""

    def __init__(
        self,
        manager: OCIProcessManager,
        lease: OCILease,
        argv: Sequence[str],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(argv, (str, bytes)) or not argv:
            raise OuterSessionInfrastructureError("attached session argv is invalid")
        self.manager = manager
        self.lease = lease
        self.argv = tuple(argv)
        self.clock = clock
        self.client: OCIAttachedClient | None = None
        self._stdin_fd = -1
        self._stdout_fd = -1

    def start(self) -> None:
        if self.client is not None:
            raise OuterSessionInfrastructureError("attached session already started")
        try:
            client = self.manager.spawn_attached(self.lease, self.argv)
            self.client = client
            self._stdin_fd = client.stdin.fileno()
            self._stdout_fd = client.stdout.fileno()
            os.set_blocking(self._stdin_fd, False)
            os.set_blocking(self._stdout_fd, False)
        except (OSError, OCIProcessError) as exc:
            try:
                self.abort()
            except OuterSessionError as cleanup:
                raise OuterSessionInfrastructureError(
                    f"attached session start cleanup failed: {cleanup}"
                ) from exc
            raise OuterSessionInfrastructureError(
                f"could not start attached OCI session: {exc}"
            ) from None

    def _require_client(self) -> OCIAttachedClient:
        if self.client is None or self.client.closed:
            raise OuterSessionInfrastructureError("attached session is not live")
        return self.client

    def _process_error(self, message: str) -> OuterSessionProcessError:
        client = self.client
        provider = getattr(client, "stderr_diagnostic", None)
        return OuterSessionProcessError(
            message,
            provider if callable(provider) else None,
        )

    def stderr_diagnostic(self) -> OCIAttachedDiagnostic:
        client = self.client
        if client is None:
            raise OuterSessionInfrastructureError(
                "attached session has no stderr diagnostic"
            )
        return client.stderr_diagnostic()

    def _diagnostic_provider(
        self,
    ) -> Callable[[], OCIAttachedDiagnostic] | None:
        return self.stderr_diagnostic if self.client is not None else None

    def _diagnostic_error(
        self, error_type: type[OuterSessionError], message: str
    ) -> OuterSessionError:
        return error_type(message, self._diagnostic_provider())

    def _remaining(self, deadline: float) -> float:
        try:
            remaining = float(deadline) - float(self.clock())
        except Exception as exc:
            raise OuterSessionInfrastructureError(f"host clock failed: {exc}") from None
        if not math.isfinite(remaining) or remaining <= 0:
            raise OuterSessionTimeoutError("attached session deadline expired")
        return remaining

    def has_pending_output(self) -> bool:
        self._require_client()
        try:
            readable, _, _ = select.select([self._stdout_fd], [], [], 0)
        except OSError as exc:
            raise OuterSessionInfrastructureError(f"cannot inspect session output: {exc}") from None
        return bool(readable)

    def write_frame(self, frame: bytes, *, deadline: float) -> None:
        self._require_client()
        if not isinstance(frame, bytes) or not frame:
            raise OuterSessionInfrastructureError("session request frame is invalid")
        view = memoryview(frame)
        offset = 0
        while offset < len(view):
            try:
                _, writable, _ = select.select(
                    [], [self._stdin_fd], [], self._remaining(deadline)
                )
                if not writable:
                    raise OuterSessionTimeoutError("session request write timed out")
                count = os.write(self._stdin_fd, view[offset:])
            except (BlockingIOError, InterruptedError):
                continue
            except BrokenPipeError:
                raise self._process_error("session closed its request pipe") from None
            except OSError as exc:
                raise self._process_error(
                    f"session request write failed: {exc}"
                ) from None
            if count <= 0:
                raise self._process_error("session request write made no progress")
            offset += count

    def _read_exact(self, size: int, *, deadline: float) -> bytes:
        self._require_client()
        remaining = size
        chunks: list[bytes] = []
        while remaining:
            try:
                readable, _, _ = select.select(
                    [self._stdout_fd], [], [], self._remaining(deadline)
                )
                if not readable:
                    raise OuterSessionTimeoutError("session response read timed out")
                chunk = os.read(self._stdout_fd, min(remaining, 1 << 20))
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as exc:
                raise self._process_error(
                    f"session response read failed: {exc}"
                ) from None
            if not chunk:
                # Never poll/wait here: only the manager may reap the process group.
                raise self._process_error(
                    "session ended before a complete response"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _header(self, *, deadline: float) -> tuple[bytes, int]:
        header = self._read_exact(FRAME_HEADER_BYTES, deadline=deadline)
        return header[:4], struct.unpack(">I", header[4:])[0]

    def read_control(self, *, max_bytes: int, deadline: float) -> dict:
        magic, size = self._header(deadline=deadline)
        if magic != CONTROL_MAGIC:
            raise OuterSessionProtocolError("worker emitted wrong control-frame magic")
        if size > max_bytes:
            raise OuterSessionProtocolError("worker declared an oversized control frame")
        try:
            return decode_message(self._read_exact(size, deadline=deadline), max_bytes=max_bytes)
        except SessionProtocolError as exc:
            raise OuterSessionProtocolError(str(exc)) from None

    def read_evidence(self, request: BatchRequest, *, deadline: float) -> BatchEvidence:
        magic, size = self._header(deadline=deadline)
        if magic == CONTROL_MAGIC:
            if size > MAX_CONTROL_BYTES:
                raise OuterSessionProtocolError("worker declared an oversized error frame")
            try:
                message = decode_message(
                    self._read_exact(size, deadline=deadline), max_bytes=MAX_CONTROL_BYTES
                )
                detail = parse_error_message(
                    message,
                    session_id=request.session_id,
                    launch_digest=request.launch_digest,
                    request=request,
                )
            except SessionProtocolError as exc:
                raise OuterSessionProtocolError(str(exc)) from None
            if detail is not None:
                raise _worker_error(
                    detail, diagnostic_provider=self._diagnostic_provider()
                )
            raise OuterSessionProtocolError("worker emitted an early control frame")
        if magic != EVIDENCE_MAGIC:
            raise OuterSessionProtocolError("worker emitted wrong evidence-frame magic")
        exact = expected_evidence_payload_bytes(request)
        if size != exact or size > MAX_BATCH_RESPONSE_BYTES:
            raise OuterSessionProtocolError("worker evidence frame has the wrong exact size")
        try:
            payload = self._read_exact(size, deadline=deadline)
            return decode_evidence_payload(payload, request=request)
        except SessionProtocolError as exc:
            raise OuterSessionProtocolError(str(exc)) from None

    def finalize(self) -> None:
        if self.client is None:
            return
        try:
            self.client.finalize()
        except OCIProcessError as exc:
            raise self._diagnostic_error(
                OuterSessionInfrastructureError, f"session cleanup failed: {exc}"
            ) from None

    def abort(self) -> None:
        if self.client is None or self.client.closed:
            return
        try:
            self.client.abort()
        except OCIProcessError as exc:
            raise self._diagnostic_error(
                OuterSessionInfrastructureError, f"session cleanup failed: {exc}"
            ) from None


@dataclass(frozen=True)
class SessionExecutionPlan:
    """Host-only session inputs; only one prompt batch crosses at a time."""

    launch_digest: str
    expected_engine_config_digest: str
    engine_config: EngineSessionConfig
    expected_preflight: RuntimePreflightFacts
    prompt_batches: tuple[tuple[str, ...], ...]
    warmup_count: int
    conditioning_count: int
    max_new_tokens: int
    top_logprobs_num: int
    temperature: float
    expected_prompt_tokens: int | None = None
    expected_discovery_overlay_identity_digest: str | None = None
    audit_policy: SlotAuditPolicy | None = None
    batch_max_new_tokens: tuple[int, ...] = ()
    batch_expected_prompt_tokens: tuple[int | None, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.engine_config, EngineSessionConfig):
            raise OuterSessionInfrastructureError("engine_config is not typed")
        if self.audit_policy is not None and (
            type(self.audit_policy) is not SlotAuditPolicy
            or self.audit_policy.expected_member_count != self.engine_config.tp_size
        ):
            raise OuterSessionInfrastructureError(
                "audit policy is not typed or differs from engine TP"
            )
        if self.engine_config.digest != self.expected_engine_config_digest:
            raise OuterSessionInfrastructureError("engine config digest differs from plan")
        if self.expected_discovery_overlay_identity_digest is not None:
            try:
                identity = require_sha256_hex(
                    self.expected_discovery_overlay_identity_digest,
                    field="expected discovery overlay identity",
                )
            except ValueError as exc:
                raise OuterSessionInfrastructureError(str(exc)) from None
            if identity == "0" * 64:
                raise OuterSessionInfrastructureError(
                    "expected discovery overlay identity must not be all zero"
                )
        if (
            not isinstance(self.expected_preflight, RuntimePreflightFacts)
            or self.expected_preflight.launch_digest != self.launch_digest
            or self.expected_preflight.engine_config_digest
            != self.expected_engine_config_digest
        ):
            raise OuterSessionInfrastructureError("preflight facts differ from plan identity")
        if isinstance(self.prompt_batches, (str, bytes)) or not isinstance(
            self.prompt_batches, Sequence
        ):
            raise OuterSessionInfrastructureError("prompt_batches must be a sequence")
        if any(isinstance(batch, (str, bytes)) for batch in self.prompt_batches):
            raise OuterSessionInfrastructureError("each prompt batch must be a sequence")
        try:
            batches = tuple(tuple(batch) for batch in self.prompt_batches)
        except TypeError:
            raise OuterSessionInfrastructureError("each prompt batch must be a sequence") from None
        if not batches or type(self.warmup_count) is not int or not 1 <= self.warmup_count < len(batches):
            raise OuterSessionInfrastructureError("session requires warmup and timed batches")
        if type(self.conditioning_count) is not int or not 1 <= self.conditioning_count <= self.warmup_count:
            raise OuterSessionInfrastructureError("conditioning_count must be in 1..warmup_count")
        object.__setattr__(self, "prompt_batches", batches)
        if type(self.batch_max_new_tokens) is not tuple or type(
            self.batch_expected_prompt_tokens
        ) is not tuple:
            raise OuterSessionInfrastructureError(
                "per-batch request geometry must be exact tuples"
            )
        batch_tokens = self.batch_max_new_tokens
        batch_prompts = self.batch_expected_prompt_tokens
        if bool(batch_tokens) != bool(batch_prompts) or (
            batch_tokens
            and (len(batch_tokens) != len(batches) or len(batch_prompts) != len(batches))
        ):
            raise OuterSessionInfrastructureError(
                "per-batch request geometry must exactly cover prompt batches"
            )
        # Validate every controller-owned frame before any OCI/GPU resource starts.
        try:
            probe_session = "1" * 32
            init = make_init(
                self.engine_config,
                session_id=probe_session,
                launch_digest=self.launch_digest,
                expected_engine_config_digest=self.expected_engine_config_digest,
                audit_policy=self.audit_policy,
            )
            frame_message(init, max_bytes=MAX_INIT_BYTES)
            accept = preflight_accept_message(
                session_id=probe_session,
                launch_digest=self.launch_digest,
                facts=self.expected_preflight,
            )
            frame_message(accept, max_bytes=MAX_CONTROL_BYTES)
        except SessionProtocolError as exc:
            raise OuterSessionInfrastructureError(
                f"controller init violates protocol policy: {exc}"
            ) from None
        for index, prompts in enumerate(batches):
            max_new_tokens, expected_prompt_tokens = self.request_geometry(index)
            try:
                message = batch_request(
                    session_id="1" * 32,
                    launch_digest=self.launch_digest,
                    request_id="2" * 32,
                    nonce="3" * 32,
                    batch_index=index,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    top_logprobs_num=self.top_logprobs_num,
                    temperature=self.temperature,
                    expected_prompt_tokens=expected_prompt_tokens,
                )
                frame_message(message, max_bytes=MAX_BATCH_REQUEST_BYTES)
            except SessionProtocolError as exc:
                raise OuterSessionInfrastructureError(
                    f"controller batch {index} violates protocol policy: {exc}"
                ) from None

    def request_geometry(self, batch_index: int) -> tuple[int, int | None]:
        """Return the validator-sealed request shape for one prompt batch."""

        if type(batch_index) is not int or not 0 <= batch_index < len(self.prompt_batches):
            raise OuterSessionInfrastructureError("batch index is outside the session")
        if self.batch_max_new_tokens:
            return (
                self.batch_max_new_tokens[batch_index],
                self.batch_expected_prompt_tokens[batch_index],
            )
        return self.max_new_tokens, self.expected_prompt_tokens

    @property
    def quality_tokens_per_prompt(self) -> int:
        """Maximum teacher work per selected prompt, derived from sealed geometry."""
        return max(self.batch_max_new_tokens or (self.max_new_tokens,))


@dataclass(frozen=True)
class BatchExecutionEvidence:
    batch_index: int
    request_id: str
    nonce: str
    request_started_at: float
    response_completed_at: float
    token_numerator: int
    evidence: BatchEvidence
    audit_receipts: tuple[AuditReceiptFacts, ...] = ()

    @property
    def elapsed_seconds(self) -> float:
        return self.response_completed_at - self.request_started_at


@dataclass(frozen=True)
class SessionExecutionEvidence:
    session_id: str
    launch_digest: str
    preflight: RuntimePreflightFacts
    ready_completed_at: float
    batches: tuple[BatchExecutionEvidence, ...]
    warmup_count: int
    conditioning_count: int
    conditioning_started_at: float
    first_timed_completed_at: float
    conditioning_token_numerator: int
    session_completed_at: float
    audit_policy_digest: str | None = None

    @property
    def audit_receipts(self) -> tuple[AuditReceiptFacts, ...]:
        return self.batches[-1].audit_receipts if self.batches else ()


BoundaryCallback = Callable[[str, int, float], None]


def diagnostic_provider(
    transport: object,
) -> Callable[[], OCIAttachedDiagnostic] | None:
    """Return only the typed host diagnostic accessor exposed by a transport."""

    provider = getattr(transport, "stderr_diagnostic", None)
    return provider if callable(provider) else None


def _now(clock: Callable[[], float], *, previous: float | None = None) -> float:
    try:
        value = float(clock())
    except Exception as exc:
        raise OuterSessionInfrastructureError(f"host clock failed: {exc}") from None
    if not math.isfinite(value) or (previous is not None and value < previous):
        raise OuterSessionInfrastructureError("host clock moved backwards")
    return value


def _control_or_error(
    transport: SessionTransport,
    *,
    session_id: str,
    launch_digest: str,
    deadline: float,
) -> dict:
    message = transport.read_control(max_bytes=MAX_CONTROL_BYTES, deadline=deadline)
    try:
        detail = parse_error_message(
            message, session_id=session_id, launch_digest=launch_digest
        )
    except SessionProtocolError as exc:
        raise OuterSessionProtocolError(str(exc)) from None
    if detail is not None:
        raise _worker_error(
            detail, diagnostic_provider=diagnostic_provider(transport)
        )
    return message


def _worker_error(
    detail: tuple[str, str, str],
    *,
    diagnostic_provider: Callable[[], OCIAttachedDiagnostic] | None,
) -> OuterSessionWorkerError:
    """Preserve the worker's candidate attribution across the host boundary."""

    stage, error_type, message = detail
    rendered = ": ".join(detail)
    if error_type in {
        "CandidateEngineFailure",
        "CandidateExecutionFailure",
        "CandidateNeverExecutedError",
    }:
        return OuterSessionCandidateError(
            rendered,
            diagnostic_provider,
            candidate_failure=message,
            candidate_failure_type=error_type,
        )
    return OuterSessionWorkerError(rendered, diagnostic_provider)


def _fresh_id(seen: set[str]) -> str:
    value = secrets.token_hex(16)
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(char not in "0123456789abcdef" for char in value)
        or value == "0" * 32
        or value in seen
    ):
        raise OuterSessionInfrastructureError("system RNG repeated a session binding")
    seen.add(value)
    return value


def require_session_timeouts(
    started_at: float, named: Sequence[tuple[str, float]]
) -> None:
    """Refuse non-finite or non-positive phase timeouts before any I/O."""

    for name, value in named:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (name == "deadline" and value <= started_at)
            or (name != "deadline" and value <= 0)
        ):
            raise OuterSessionInfrastructureError(f"{name} is invalid")


def abort_failed_session(session, original: BaseException) -> NoReturn:
    """Attach diagnostics, abort the transport, and re-raise (or fail cleanup)."""

    if isinstance(original, OuterSessionError):
        original.attach_diagnostic(diagnostic_provider(session.transport))
    try:
        session.transport.abort()
    except BaseException as cleanup:
        error = OuterSessionInfrastructureError(
            f"session cleanup could not be proven: {cleanup}"
        )
        error.attach_diagnostic(diagnostic_provider(session.transport))
        session.closed = True
        raise error from original
    session.closed = True
    raise original


def perform_init_handshake(
    transport: SessionTransport,
    *,
    init: dict,
    session_id: str,
    launch_digest: str,
    expected_preflight: object,
    init_deadline: float,
    preflight_catch: tuple[type[BaseException], ...],
    worker_label: str = "worker",
) -> RuntimePreflightFacts:
    """Run the shared init→preflight→accept→ready handshake on a fresh worker.

    ``preflight_catch`` names the exception classes each controller converts to
    an infrastructure failure at the preflight step; the classes differ between
    controllers and participate in failure classification, so the caller owns
    them. The caller also owns ``transport.start()`` ordering by calling this
    first, the post-ready timestamps, and the trailing pending-output check.
    """

    transport.start()
    if transport.has_pending_output():
        raise OuterSessionProtocolError(f"{worker_label} emitted output before init")
    transport.write_frame(
        frame_message(init, max_bytes=MAX_INIT_BYTES), deadline=init_deadline
    )
    try:
        preflight = validate_preflight(
            _control_or_error(
                transport,
                session_id=session_id,
                launch_digest=launch_digest,
                deadline=init_deadline,
            ),
            session_id=session_id,
            launch_digest=launch_digest,
            expected_facts=expected_preflight,
        )
    except preflight_catch as exc:
        detail = exc.message if isinstance(exc, OuterSessionError) else str(exc)
        raise OuterSessionInfrastructureError(
            f"runtime preflight failed: {detail}",
            diagnostic_provider(transport),
        ) from None
    transport.write_frame(
        frame_message(
            preflight_accept_message(
                session_id=session_id,
                launch_digest=launch_digest,
                facts=preflight,
            ),
            max_bytes=MAX_CONTROL_BYTES,
        ),
        deadline=init_deadline,
    )
    ready = _control_or_error(
        transport,
        session_id=session_id,
        launch_digest=launch_digest,
        deadline=init_deadline,
    )
    try:
        validate_ready(
            ready,
            session_id=session_id,
            launch_digest=launch_digest,
        )
    except SessionProtocolError as exc:
        raise OuterSessionProtocolError(str(exc)) from None
    return preflight


class OpenedOuterSession:
    """Incremental trusted-host controller for one already-open engine lifetime.

    The plan is validated in full before ``start``. Callers may inspect retained
    batch evidence between ``execute_next`` calls, but only ``finish`` returns a
    session receipt. The compatibility runner below still requires every planned
    batch; crossover scheduling may close a validated prefix after its first read.
    """

    def __init__(
        self,
        plan: SessionExecutionPlan,
        *,
        transport: SessionTransport,
        deadline: float,
        init_timeout_s: float,
        batch_timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
        boundary_callback: BoundaryCallback | None = None,
    ) -> None:
        if type(plan) is not SessionExecutionPlan:
            raise OuterSessionInfrastructureError("session plan is not typed")
        started_at = _now(clock)
        require_session_timeouts(
            started_at,
            (
                ("deadline", deadline),
                ("init_timeout_s", init_timeout_s),
                ("batch_timeout_s", batch_timeout_s),
            ),
        )
        self.plan = plan
        self.transport = transport
        self.deadline = float(deadline)
        self.init_timeout_s = float(init_timeout_s)
        self.batch_timeout_s = float(batch_timeout_s)
        self.clock = clock
        self.boundary_callback = boundary_callback
        self.started_at = started_at
        self.seen: set[str] = set()
        self.session_id = _fresh_id(self.seen)
        self.batch_rows: list[BatchExecutionEvidence] = []
        self.conditioning_start_index = plan.warmup_count - plan.conditioning_count
        self.conditioning_started_at: float | None = None
        self.first_timed_completed_at: float | None = None
        self.preflight: RuntimePreflightFacts | None = None
        self.ready_completed_at = 0.0
        self.last_host_time = started_at
        self.started = False
        self.closed = False

    @property
    def next_batch_index(self) -> int:
        return len(self.batch_rows)

    def _phase_deadline(self, limit: float) -> float:
        return min(self.deadline, _now(self.clock) + limit)

    def _fail(self, original: BaseException) -> NoReturn:
        abort_failed_session(self, original)

    def start(self) -> None:
        if self.started or self.closed:
            raise OuterSessionInfrastructureError("session start order is invalid")
        try:
            self.preflight = perform_init_handshake(
                self.transport,
                init=make_init(
                    self.plan.engine_config,
                    session_id=self.session_id,
                    launch_digest=self.plan.launch_digest,
                    expected_engine_config_digest=(
                        self.plan.expected_engine_config_digest
                    ),
                    audit_policy=self.plan.audit_policy,
                ),
                session_id=self.session_id,
                launch_digest=self.plan.launch_digest,
                expected_preflight=self.plan.expected_preflight,
                init_deadline=self._phase_deadline(self.init_timeout_s),
                preflight_catch=(
                    SessionProtocolError,
                    OuterSessionProtocolError,
                    OuterSessionWorkerError,
                ),
            )
            self.ready_completed_at = _now(self.clock, previous=self.started_at)
            self.last_host_time = self.ready_completed_at
            if self.conditioning_start_index == 0:
                self.conditioning_started_at = self.ready_completed_at
            if self.transport.has_pending_output():
                raise OuterSessionProtocolError(
                    "worker emitted output before first request"
                )
            self.started = True
        except BaseException as exc:
            self._fail(exc)

    def execute_next(self) -> BatchExecutionEvidence:
        if not self.started or self.closed:
            raise OuterSessionInfrastructureError("session is not open")
        index = self.next_batch_index
        if index >= len(self.plan.prompt_batches):
            raise OuterSessionInfrastructureError("session has no remaining planned batch")
        prompts = self.plan.prompt_batches[index]
        max_new_tokens, expected_prompt_tokens = self.plan.request_geometry(index)
        try:
            request_id, nonce = _fresh_id(self.seen), _fresh_id(self.seen)
            request = validate_batch_request(
                batch_request(
                    session_id=self.session_id,
                    launch_digest=self.plan.launch_digest,
                    request_id=request_id,
                    nonce=nonce,
                    batch_index=index,
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    top_logprobs_num=self.plan.top_logprobs_num,
                    temperature=self.plan.temperature,
                    expected_prompt_tokens=expected_prompt_tokens,
                )
            )
            final_warmup = index == self.plan.warmup_count - 1
            first_timed = index == self.plan.warmup_count
            if final_warmup and self.boundary_callback is not None:
                self.boundary_callback("before_final_warmup", index, self.deadline)
            if first_timed and self.boundary_callback is not None:
                self.boundary_callback("before_first_timed", index, self.deadline)
            if self.transport.has_pending_output():
                raise OuterSessionProtocolError("worker emitted early or duplicate output")
            batch_deadline = self._phase_deadline(self.batch_timeout_s)
            request_started = _now(self.clock, previous=self.last_host_time)
            self.transport.write_frame(
                frame_message(request.to_dict(), max_bytes=MAX_BATCH_REQUEST_BYTES),
                deadline=batch_deadline,
            )
            evidence = self.transport.read_evidence(request, deadline=batch_deadline)
            audit_receipts: tuple[AuditReceiptFacts, ...] = ()
            if self.plan.audit_policy is not None:
                try:
                    audit_receipts = validate_audit_evidence(
                        _control_or_error(
                            self.transport,
                            session_id=self.session_id,
                            launch_digest=self.plan.launch_digest,
                            deadline=batch_deadline,
                        ),
                        request=request,
                        policy=self.plan.audit_policy,
                    )
                except SessionProtocolError as exc:
                    raise OuterSessionProtocolError(str(exc)) from None
            completed = _now(self.clock, previous=request_started)
            if completed <= request_started:
                raise OuterSessionInfrastructureError("host batch clock did not advance")
            token_numerator = len(prompts) * max_new_tokens
            if evidence.observed_tokens != token_numerator:
                raise OuterSessionProtocolError("worker evidence token count is not exact")
            row = BatchExecutionEvidence(
                index,
                request_id,
                nonce,
                request_started,
                completed,
                token_numerator,
                evidence,
                audit_receipts,
            )
            self.batch_rows.append(row)
            self.last_host_time = completed
            if index + 1 == self.conditioning_start_index:
                self.conditioning_started_at = completed
            if final_warmup and self.boundary_callback is not None:
                self.boundary_callback("after_final_warmup", index, self.deadline)
            if first_timed:
                self.first_timed_completed_at = completed
            if self.transport.has_pending_output():
                raise OuterSessionProtocolError(
                    "worker emitted trailing or duplicate output"
                )
            return row
        except BaseException as exc:
            self._fail(exc)

    def finish(self, *, require_all: bool = True) -> SessionExecutionEvidence:
        if type(require_all) is not bool or not self.started or self.closed:
            raise OuterSessionInfrastructureError("session finish order is invalid")
        minimum = self.plan.warmup_count + 1
        if len(self.batch_rows) < minimum or (
            require_all and len(self.batch_rows) != len(self.plan.prompt_batches)
        ):
            raise OuterSessionInfrastructureError(
                "session lacks the required planned batch coverage"
            )
        try:
            if self.transport.has_pending_output():
                raise OuterSessionProtocolError(
                    "worker emitted trailing or duplicate output before cleanup"
                )
            if _now(self.clock, previous=self.last_host_time) >= self.deadline:
                raise OuterSessionTimeoutError("session deadline expired before cleanup")
            self.transport.finalize()
            session_completed_at = _now(
                self.clock, previous=self.batch_rows[-1].response_completed_at
            )
            if session_completed_at > self.deadline:
                raise OuterSessionTimeoutError(
                    "session cleanup exceeded its absolute deadline"
                )
        except BaseException as exc:
            self._fail(exc)
        self.closed = True
        if (
            self.preflight is None
            or self.conditioning_started_at is None
            or self.first_timed_completed_at is None
        ):
            raise OuterSessionInfrastructureError(
                "session lacks required conditioning evidence"
            )
        conditioning_tokens = sum(
            row.token_numerator
            for row in self.batch_rows[
                self.conditioning_start_index : self.plan.warmup_count + 1
            ]
        )
        if self.first_timed_completed_at <= self.conditioning_started_at:
            raise OuterSessionInfrastructureError(
                "conditioning interval did not advance"
            )
        return SessionExecutionEvidence(
            session_id=self.session_id,
            launch_digest=self.plan.launch_digest,
            preflight=self.preflight,
            ready_completed_at=self.ready_completed_at,
            batches=tuple(self.batch_rows),
            warmup_count=self.plan.warmup_count,
            conditioning_count=self.plan.conditioning_count,
            conditioning_started_at=self.conditioning_started_at,
            first_timed_completed_at=self.first_timed_completed_at,
            conditioning_token_numerator=conditioning_tokens,
            session_completed_at=session_completed_at,
            audit_policy_digest=(
                None if self.plan.audit_policy is None else self.plan.audit_policy.digest
            ),
        )

    def abort(self) -> None:
        if self.closed:
            return
        try:
            self.transport.abort()
        finally:
            self.closed = True


def run_outer_session(
    plan: SessionExecutionPlan,
    *,
    transport: SessionTransport,
    deadline: float,
    init_timeout_s: float,
    batch_timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
    boundary_callback: BoundaryCallback | None = None,
) -> SessionExecutionEvidence:
    """Execute every planned batch and destroy the engine (legacy wrapper)."""

    session = OpenedOuterSession(
        plan,
        transport=transport,
        deadline=deadline,
        init_timeout_s=init_timeout_s,
        batch_timeout_s=batch_timeout_s,
        clock=clock,
        boundary_callback=boundary_callback,
    )
    session.start()
    while session.next_batch_index < len(plan.prompt_batches):
        session.execute_next()
    return session.finish()
