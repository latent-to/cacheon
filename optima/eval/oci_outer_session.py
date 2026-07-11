"""Trusted host timing loop for an entirely untrusted SGLang container tree."""

from __future__ import annotations

import dataclasses
import os
import select
import secrets
import signal
import statistics
import subprocess
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from optima.eval.oci_session_protocol import (
    EVIDENCE_MAGIC,
    TEACHER_EVIDENCE_MAGIC,
    FRAME_HEADER_BYTES,
    FRAME_MAGIC,
    MAX_BATCH_REQUEST_BYTES,
    MAX_BATCH_RESPONSE_BYTES,
    MAX_CONTROL_BYTES,
    MAX_INIT_BYTES,
    SESSION_SCHEMA,
    SessionProtocolError,
    batch_request,
    decode_evidence_payload,
    decode_teacher_evidence_payload,
    decode_message,
    frame_message,
    expected_evidence_payload_bytes,
    expected_teacher_payload_bytes,
    make_init,
    parse_error_message,
    teacher_request,
    validate_preflight,
    validate_ready,
)


class OuterSessionError(RuntimeError):
    retryable = False


class OuterSessionInfrastructureError(OuterSessionError):
    """Host/runtime failure which may succeed on a clean retry."""

    retryable = True


class OuterSessionTimeoutError(OuterSessionInfrastructureError):
    """A phase watchdog expired; attribution depends on B versus C."""


class OuterSessionProcessError(OuterSessionInfrastructureError):
    """The contained engine exited/broke its protocol pipe during an arm."""


class OuterSessionCandidateError(OuterSessionError):
    """Malformed/stale/early candidate evidence; deterministic terminal failure."""

    retryable = False


class OuterSessionPosthocCandidateError(OuterSessionCandidateError):
    """C's sealed output made the bounded stock-B' teacher phase fail."""


class SessionTransport(Protocol):
    def start(self) -> None: ...
    def has_pending_output(self) -> bool: ...
    def write_frame(self, frame: bytes, *, deadline: float) -> None: ...
    def read_frame(
        self, *, magic: bytes, max_bytes: int, deadline: float,
        exact_bytes: int | None = None,
    ) -> bytes: ...
    def expect_clean_exit(self, *, deadline: float) -> None: ...
    def abort(self) -> None: ...


class ContainerSessionTransport:
    """Nonblocking framed stdin/stdout transport around one foreground container."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        force_remove: Callable[[], None],
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.argv = tuple(argv)
        self.force_remove = force_remove
        self.popen_factory = popen_factory
        self.process: subprocess.Popen | None = None
        self._stdin_fd = -1
        self._stdout_fd = -1

    def start(self) -> None:
        try:
            process = self.popen_factory(
                list(self.argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
                shell=False,
            )
        except OSError as exc:
            raise OuterSessionInfrastructureError(
                f"could not start OCI session runtime: {exc}"
            ) from None
        if process.stdin is None or process.stdout is None:
            process.kill()
            raise OuterSessionInfrastructureError("OCI runtime did not expose session pipes")
        self.process = process
        self._stdin_fd = process.stdin.fileno()
        self._stdout_fd = process.stdout.fileno()
        os.set_blocking(self._stdin_fd, False)
        os.set_blocking(self._stdout_fd, False)

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OuterSessionTimeoutError("OCI session I/O watchdog expired")
        return remaining

    def _require_started(self) -> subprocess.Popen:
        if self.process is None:
            raise OuterSessionInfrastructureError("OCI session transport is not started")
        return self.process

    def has_pending_output(self) -> bool:
        self._require_started()
        readable, _, _ = select.select([self._stdout_fd], [], [], 0)
        return bool(readable)

    def write_frame(self, frame: bytes, *, deadline: float) -> None:
        process = self._require_started()
        view = memoryview(frame)
        offset = 0
        while offset < len(view):
            if process.poll() is not None:
                raise OuterSessionProcessError(
                    f"OCI session exited while receiving a request ({process.returncode})"
                )
            _, writable, _ = select.select(
                [], [self._stdin_fd], [], self._remaining(deadline)
            )
            if not writable:
                raise OuterSessionTimeoutError("OCI session request write timed out")
            try:
                written = os.write(self._stdin_fd, view[offset:])
            except BlockingIOError:
                continue
            except BrokenPipeError:
                raise OuterSessionProcessError(
                    "OCI session closed stdin while receiving a request"
                ) from None
            if written <= 0:
                raise OuterSessionInfrastructureError("OCI session request write made no progress")
            offset += written

    def _read_exact(self, size: int, *, deadline: float) -> bytes:
        process = self._require_started()
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            readable, _, _ = select.select(
                [self._stdout_fd], [], [], self._remaining(deadline)
            )
            if not readable:
                raise OuterSessionTimeoutError("OCI session response read timed out")
            try:
                chunk = os.read(self._stdout_fd, min(remaining, 1024 * 1024))
            except BlockingIOError:
                continue
            if not chunk:
                code = process.poll()
                raise OuterSessionProcessError(
                    f"OCI session ended before a complete response (exit={code})"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read_frame(
        self, *, magic: bytes, max_bytes: int, deadline: float,
        exact_bytes: int | None = None,
    ) -> bytes:
        header = self._read_exact(FRAME_HEADER_BYTES, deadline=deadline)
        if header[:4] != magic:
            size = int.from_bytes(header[4:8], "big")
            if (
                magic in {EVIDENCE_MAGIC, TEACHER_EVIDENCE_MAGIC}
                and header[:4] == FRAME_MAGIC
                and size <= MAX_CONTROL_BYTES
            ):
                payload = self._read_exact(size, deadline=deadline)
                try:
                    message = decode_message(payload, max_bytes=MAX_CONTROL_BYTES)
                except SessionProtocolError:
                    message = None
                try:
                    detail = parse_error_message(message)
                except SessionProtocolError:
                    detail = None
                if detail is not None:
                    stage, error_type, error_text = detail
                    if magic == TEACHER_EVIDENCE_MAGIC:
                        error_class = (
                            OuterSessionPosthocCandidateError
                            if error_type == "CandidateTeacherInputError"
                            else OuterSessionInfrastructureError
                        )
                    else:
                        error_class = OuterSessionCandidateError
                    raise error_class(
                        "OCI worker failed while producing evidence at "
                        f"{stage}: {error_type}: {error_text}"
                    )
            error_class = (
                OuterSessionInfrastructureError
                if magic == TEACHER_EVIDENCE_MAGIC
                else OuterSessionCandidateError
            )
            raise error_class(
                "OCI session emitted an early marker or wrong response magic "
                f"(expected={magic.hex()}, actual={header[:4].hex()}, "
                f"header={header.hex()})"
            )
        size = int.from_bytes(header[4:8], "big")
        if size > max_bytes:
            error_class = (
                OuterSessionInfrastructureError
                if magic == TEACHER_EVIDENCE_MAGIC
                else OuterSessionCandidateError
            )
            raise error_class(
                f"OCI session declared an oversized response ({size}>{max_bytes})"
            )
        if exact_bytes is not None and size != exact_bytes:
            error_class = (
                OuterSessionInfrastructureError
                if magic == TEACHER_EVIDENCE_MAGIC
                else OuterSessionCandidateError
            )
            raise error_class(
                f"OCI session evidence length {size} != required {exact_bytes}"
            )
        return self._read_exact(size, deadline=deadline)

    def expect_clean_exit(self, *, deadline: float) -> None:
        process = self._require_started()
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        # Drain only EOF. Any byte after the exact closed frame is a protocol attack.
        while True:
            readable, _, _ = select.select(
                [self._stdout_fd], [], [], self._remaining(deadline)
            )
            if not readable:
                raise OuterSessionTimeoutError("OCI session exit timed out")
            chunk = os.read(self._stdout_fd, 1)
            if chunk:
                raise OuterSessionCandidateError(
                    "OCI session emitted trailing bytes after its closed response"
                )
            break
        try:
            code = process.wait(timeout=self._remaining(deadline))
        except subprocess.TimeoutExpired:
            raise OuterSessionTimeoutError("OCI session process did not exit") from None
        if code != 0:
            raise OuterSessionProcessError(
                f"OCI session container exited with status {code}"
            )

    def abort(self) -> None:
        process = self.process
        cleanup_error: BaseException | None = None
        try:
            self.force_remove()
        except BaseException as exc:  # noqa: BLE001 - cleanup still reaps client
            cleanup_error = exc
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except OSError:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    process.kill()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if cleanup_error is None:
                        cleanup_error = OuterSessionInfrastructureError(
                            "OCI client process survived forced termination"
                        )
        if cleanup_error is not None:
            if isinstance(cleanup_error, OuterSessionError):
                raise cleanup_error
            raise OuterSessionInfrastructureError(
                f"could not prove OCI session cleanup: {cleanup_error}"
            ) from None


def _json_frame(message: dict[str, Any], *, max_bytes: int) -> bytes:
    return frame_message(message, max_bytes=max_bytes)


def _read_json(
    transport: SessionTransport, *, max_bytes: int, deadline: float,
    session_id: str,
) -> dict[str, Any]:
    payload = transport.read_frame(magic=FRAME_MAGIC, max_bytes=max_bytes, deadline=deadline)
    try:
        message = decode_message(payload, max_bytes=max_bytes)
        detail = parse_error_message(message, expected_session_id=session_id)
    except SessionProtocolError as exc:
        raise OuterSessionCandidateError(str(exc)) from None
    if detail is not None:
        stage, error_type, error_text = detail
        raise OuterSessionCandidateError(
            f"OCI worker failed at {stage}: {error_type}: {error_text}"
        )
    return message


def run_outer_timed_session(
    cfg,
    prompt_batches: Sequence[Sequence[str]],
    *,
    mode: str,
    transport: SessionTransport,
    init_timeout_s: float,
    batch_timeout_s: float,
    total_timeout_s: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
    before_first_timed: Callable[[], None] | None = None,
    warmup_timed_boundary: Callable[[str, str, int, float], None] | None = None,
    expected_runtime_attestation: Mapping[str, Any] | None = None,
    posthoc_plan=None,
) -> Any:
    """Return a host-constructed ``ModeResult`` (or audit tuple) from one session."""

    from optima.eval.throughput_kl import ModeResult

    if total_timeout_s is None:
        total_timeout_s = init_timeout_s
    if (
        isinstance(total_timeout_s, bool)
        or not isinstance(total_timeout_s, (int, float))
        or total_timeout_s <= 0
    ):
        raise OuterSessionInfrastructureError("outer session total timeout is invalid")
    session_deadline = time.monotonic() + float(total_timeout_s)

    def phase_deadline(limit: float) -> float:
        return min(session_deadline, time.monotonic() + float(limit))

    measure_cfg = (
        dataclasses.replace(
            cfg,
            timed_iters=1,
            warmup_iters=1,
            conditioning_iters=1,
            disable_cuda_graph=True,
        )
        if mode == "candidate_audit"
        else cfg
    )
    warmup_iters = int(measure_cfg.warmup_iters)
    conditioning_iters = int(measure_cfg.conditioning_iters)
    if not 1 <= conditioning_iters <= warmup_iters:
        raise OuterSessionInfrastructureError(
            "conditioning_iters must be in 1..warmup_iters"
        )
    conditioning_start_index = warmup_iters - conditioning_iters
    expected_batches = int(measure_cfg.warmup_iters) + int(measure_cfg.timed_iters)
    if len(prompt_batches) != expected_batches:
        raise OuterSessionInfrastructureError(
            f"prompt plan has {len(prompt_batches)} batches; expected {expected_batches}"
        )
    seen_prompts: set[str] = set()
    for batch in prompt_batches:
        if len(batch) != int(measure_cfg.num_prompts):
            raise OuterSessionInfrastructureError(
                "controller prompt batch count differs from EvalConfig"
            )
        for prompt in batch:
            if not isinstance(prompt, str) or prompt in seen_prompts:
                raise OuterSessionInfrastructureError(
                    "controller prompt plan requires globally disjoint strings"
                )
            seen_prompts.add(prompt)

    # Fully validate and size every controller-owned prompt frame before any
    # container is started.  A generator producing a 9 MiB prompt or an oversized
    # aggregate is a validator fault; it must never consume a GPU launch or become
    # a miner protocol disqualification later in the arm.
    for batch_index, prompts in enumerate(prompt_batches):
        try:
            probe = batch_request(
                session_id="0" * 32,
                request_id=f"{batch_index + 1:032x}",
                nonce=f"{batch_index + 1:032x}",
                batch_index=batch_index,
                warmup=batch_index < int(measure_cfg.warmup_iters),
                prompts=prompts,
                expected_count=int(measure_cfg.num_prompts),
            )
            _json_frame(probe, max_bytes=MAX_BATCH_REQUEST_BYTES)
        except SessionProtocolError as exc:
            raise OuterSessionInfrastructureError(
                f"controller prompt batch {batch_index} violates protocol policy: {exc}"
            ) from None

    session_id = secrets.token_hex(16)
    request_ids: set[str] = set()
    nonces: set[str] = set()
    samples: list[float] = []
    timed_per_prompt: list[tuple[list[int], list]] = []
    timed_texts: list[str] = []
    timed_batches: list[list[tuple[list[int], list]]] = []
    warmup_per_prompt: list[tuple[list[int], list]] = []
    warmup_texts: list[str] = []
    warmup_batches: list[list[tuple[list[int], list]]] = []
    conditioning_tok_per_s = 0.0
    conditioning_started_at: float | None = None
    conditioning_tokens = 0
    conditioning_batch_rates: list[float] = []
    total_tokens = 0
    teacher_traces: dict[tuple[str, int], tuple] = {}
    untrusted_engine_stage_started = False
    try:
        # Starting the OCI runtime is itself inside the cleanup boundary. A partial
        # start must still force-remove its container/process tree.
        transport.start()
        if transport.has_pending_output():
            raise OuterSessionInfrastructureError(
                "OCI runtime emitted output before candidate/session initialization"
            )
        init = make_init(cfg, mode=mode, session_id=session_id)
        deadline = phase_deadline(init_timeout_s)
        transport.write_frame(
            _json_frame(init, max_bytes=MAX_INIT_BYTES), deadline=deadline
        )
        try:
            preflight = _read_json(
                transport, max_bytes=MAX_CONTROL_BYTES, deadline=deadline,
                session_id=session_id,
            )
            validate_preflight(
                preflight,
                session_id=session_id,
                mode=mode,
                expected_runtime=(expected_runtime_attestation or {}),
            )
        except (SessionProtocolError, OuterSessionCandidateError) as exc:
            # This frame is emitted before any candidate or engine import.  A
            # malformed frame or mismatch is therefore a validator image/mount/
            # host fault, never a miner disqualification.
            raise OuterSessionInfrastructureError(str(exc)) from None
        # The worker emits preflight immediately before constructing EvalConfig and
        # entering the engine context. From this point onward a candidate launch may
        # import/execute miner code; timeout/process attribution can become terminal.
        untrusted_engine_stage_started = True
        ready = _read_json(
            transport, max_bytes=MAX_CONTROL_BYTES, deadline=deadline,
            session_id=session_id,
        )
        try:
            validate_ready(ready, session_id=session_id, mode=mode)
        except SessionProtocolError as exc:
            raise OuterSessionCandidateError(str(exc)) from None
        if transport.has_pending_output():
            raise OuterSessionCandidateError("OCI worker emitted an early batch marker")
        if conditioning_start_index == 0:
            # With no setup warmups, the engine's ready frame starts the charged
            # tail. Otherwise it starts at the exact completion of the final free
            # setup response below, before any candidate-controlled gap can open.
            conditioning_started_at = clock()

        for batch_index, prompts in enumerate(prompt_batches):
            request_id = secrets.token_hex(16)
            nonce = secrets.token_hex(16)
            if request_id in request_ids or nonce in nonces:  # pragma: no cover
                raise OuterSessionInfrastructureError("system RNG repeated a request binding")
            request_ids.add(request_id)
            nonces.add(nonce)
            warmup = batch_index < warmup_iters
            final_warmup = (
                warmup
                and batch_index == warmup_iters - 1
            )
            if final_warmup and warmup_timed_boundary is not None:
                warmup_timed_boundary(
                    "before_final_warmup",
                    mode,
                    batch_index,
                    session_deadline,
                )
            if not warmup and batch_index == warmup_iters:
                if warmup_timed_boundary is not None:
                    warmup_timed_boundary(
                        "before_first_timed",
                        mode,
                        batch_index,
                        session_deadline,
                    )
                if before_first_timed is not None:
                    before_first_timed()
            request = batch_request(
                session_id=session_id,
                request_id=request_id,
                nonce=nonce,
                batch_index=batch_index,
                warmup=warmup,
                prompts=prompts,
                expected_count=int(measure_cfg.num_prompts),
            )
            request_frame = _json_frame(
                request, max_bytes=MAX_BATCH_REQUEST_BYTES
            )
            if transport.has_pending_output():
                raise OuterSessionCandidateError(
                    "OCI worker emitted data before the next timed request"
                )
            deadline = phase_deadline(batch_timeout_s)
            started = clock()
            transport.write_frame(request_frame, deadline=deadline)
            evidence_payload = transport.read_frame(
                magic=EVIDENCE_MAGIC,
                max_bytes=MAX_BATCH_RESPONSE_BYTES,
                deadline=deadline,
                exact_bytes=expected_evidence_payload_bytes(
                    prompt_count=int(measure_cfg.num_prompts),
                    max_new_tokens=int(measure_cfg.max_new_tokens),
                    top_logprobs_num=int(measure_cfg.top_logprobs_num),
                    require_logprobs=True,
                    ignore_eos=bool(measure_cfg.ignore_eos),
                ),
            )
            completed = clock()
            elapsed = completed - started
            if elapsed <= 0:
                raise OuterSessionInfrastructureError("trusted host clock did not advance")
            if warmup and batch_index + 1 == conditioning_start_index:
                conditioning_started_at = completed
            try:
                evidence = decode_evidence_payload(
                    evidence_payload,
                    session_id=session_id,
                    request_id=request_id,
                    nonce=nonce,
                    batch_index=batch_index,
                    expected_prompts=int(measure_cfg.num_prompts),
                    max_new_tokens=int(measure_cfg.max_new_tokens),
                    top_logprobs_num=int(measure_cfg.top_logprobs_num),
                    ignore_eos=bool(measure_cfg.ignore_eos),
                    require_logprobs=True,
                    temperature=float(measure_cfg.temperature),
                )
            except SessionProtocolError as exc:
                raise OuterSessionCandidateError(str(exc)) from None
            if transport.has_pending_output():
                raise OuterSessionCandidateError(
                    "OCI worker emitted trailing/early bytes after batch evidence"
                )
            # Conditioning output is retained but never mixed into timed fidelity.
            # A candidate cannot skip warmup or use correct warmups to dilute corrupt
            # timed work.
            batch_evidence = list(evidence.per_prompt)
            if warmup:
                warmup_tokens = (
                    int(measure_cfg.num_prompts)
                    * int(measure_cfg.max_new_tokens)
                    if measure_cfg.ignore_eos else evidence.observed_tokens
                )
                if batch_index >= conditioning_start_index:
                    conditioning_tokens += warmup_tokens
                    conditioning_batch_rates.append(warmup_tokens / elapsed)
                warmup_batches.append(batch_evidence)
                warmup_per_prompt.extend(batch_evidence)
                warmup_texts.extend(evidence.texts)
                if final_warmup and warmup_timed_boundary is not None:
                    warmup_timed_boundary(
                        "after_final_warmup",
                        mode,
                        batch_index,
                        session_deadline,
                    )
            else:
                timed_batches.append(batch_evidence)
                timed_per_prompt.extend(batch_evidence)
                timed_texts.extend(evidence.texts)
                tokens = (
                    int(measure_cfg.num_prompts) * int(measure_cfg.max_new_tokens)
                    if measure_cfg.ignore_eos else evidence.observed_tokens
                )
                rate = tokens / elapsed
                samples.append(rate)
                if len(samples) == 1 and warmup_iters > 0:
                    assert conditioning_started_at is not None
                    if len(conditioning_batch_rates) != conditioning_iters:
                        raise OuterSessionInfrastructureError(
                            "conditioning tail lacks every declared charged-warmup rate"
                        )
                    conditioning_tokens += tokens
                    transition_elapsed = completed - conditioning_started_at
                    if transition_elapsed <= 0:
                        raise OuterSessionInfrastructureError(
                            "trusted conditioning-tail clock did not advance"
                        )
                    # No charged conditioning warmup may be sacrificed for cooldown,
                    # and no work/sleep inside the charged tail may be hidden. The
                    # complete tail rate charges gaps; the per-batch minimum charges
                    # a slow request; the first timed rate cannot be median-discarded.
                    conditioning_tok_per_s = min(
                        *conditioning_batch_rates,
                        rate,
                        conditioning_tokens / transition_elapsed,
                    )
                total_tokens += tokens

        # Only the surviving stock B' may receive the controller-only post-hoc
        # reference frame. B and C are already destroyed; C never learns which
        # prompt clusters are selected or sees any teacher output.
        if posthoc_plan is not None:
            from optima.eval.external_quality import PosthocReferencePlan

            if mode != "baseline" or type(posthoc_plan) is not PosthocReferencePlan:
                raise OuterSessionInfrastructureError(
                    "post-hoc reference scoring requires a typed stock B' plan"
                )
            planned = (*posthoc_plan.warmup_batches, *posthoc_plan.timed_batches)
            for batch_plan in planned:
                global_index = (
                    batch_plan.batch_index
                    if batch_plan.phase == "warmup"
                    else warmup_iters + batch_plan.batch_index
                )
                measured_prompts = prompt_batches[global_index]
                measured_runs = (
                    warmup_batches[batch_plan.batch_index]
                    if batch_plan.phase == "warmup"
                    else timed_batches[batch_plan.batch_index]
                )
                prompts: list[str] = []
                stock_ids: list[list[int]] = []
                for selected in batch_plan.prompts:
                    if (
                        not 0 <= selected.prompt_index < len(measured_prompts)
                        or measured_prompts[selected.prompt_index] != selected.prompt
                    ):
                        raise OuterSessionInfrastructureError(
                            "post-hoc selected prompt does not match B' prompt plan"
                        )
                    prompts.append(selected.prompt)
                    stock_ids.append(list(measured_runs[selected.prompt_index][0]))
                request_id = secrets.token_hex(16)
                nonce = secrets.token_hex(16)
                if request_id in request_ids or nonce in nonces:  # pragma: no cover
                    raise OuterSessionInfrastructureError(
                        "system RNG repeated a post-hoc request binding"
                    )
                request_ids.add(request_id)
                nonces.add(nonce)
                request = teacher_request(
                    session_id=session_id,
                    request_id=request_id,
                    nonce=nonce,
                    phase=batch_plan.phase,
                    batch_index=batch_plan.batch_index,
                    sealed_rollout_sha256=posthoc_plan.sealed_rollout_sha256,
                    prompts=prompts,
                    baseline_ids=[list(selected.baseline[0]) for selected in batch_plan.prompts],
                    candidate_ids=[list(selected.candidate[0]) for selected in batch_plan.prompts],
                    stock_control_ids=stock_ids,
                    expected_count=len(batch_plan.prompts),
                    expected_tokens=int(measure_cfg.max_new_tokens),
                )
                if transport.has_pending_output():
                    raise OuterSessionInfrastructureError(
                        "stock B' emitted data before a teacher request"
                    )
                deadline = phase_deadline(batch_timeout_s)
                try:
                    transport.write_frame(
                        _json_frame(request, max_bytes=MAX_BATCH_REQUEST_BYTES),
                        deadline=deadline,
                    )
                    payload = transport.read_frame(
                        magic=TEACHER_EVIDENCE_MAGIC,
                        max_bytes=MAX_BATCH_RESPONSE_BYTES,
                        deadline=deadline,
                        exact_bytes=expected_teacher_payload_bytes(
                            prompt_count=len(batch_plan.prompts),
                            token_count=int(measure_cfg.max_new_tokens),
                            top_logprobs_num=int(measure_cfg.top_logprobs_num),
                        ),
                    )
                    teacher_traces[(batch_plan.phase, batch_plan.batch_index)] = (
                        decode_teacher_evidence_payload(
                            payload,
                            session_id=session_id,
                            request_id=request_id,
                            nonce=nonce,
                            phase=batch_plan.phase,
                            batch_index=batch_plan.batch_index,
                            sealed_rollout_sha256=posthoc_plan.sealed_rollout_sha256,
                            expected_prompts=len(batch_plan.prompts),
                            token_count=int(measure_cfg.max_new_tokens),
                            top_logprobs_num=int(measure_cfg.top_logprobs_num),
                        )
                    )
                except OuterSessionPosthocCandidateError:
                    raise
                except (
                    OuterSessionTimeoutError,
                    OuterSessionProcessError,
                    OuterSessionCandidateError,
                    SessionProtocolError,
                ) as exc:
                    raise OuterSessionInfrastructureError(
                        f"stock B' teacher scoring failed without validated "
                        f"candidate causation: {exc}"
                    ) from None
                if transport.has_pending_output():
                    raise OuterSessionInfrastructureError(
                        "stock B' emitted trailing teacher evidence"
                    )

        # All authoritative evidence is now outside the container tree. Force-remove
        # immediately; the controller's device-state lane drains/attests next.
        transport.abort()
        audit_receipts, audit_members = [], []
    except OuterSessionPosthocCandidateError:
        transport.abort()
        raise
    except OuterSessionCandidateError as exc:
        transport.abort()
        if mode == "baseline":
            raise OuterSessionInfrastructureError(
                f"stock arm emitted invalid protocol evidence: {exc}"
            ) from None
        raise
    except (OuterSessionTimeoutError, OuterSessionProcessError) as exc:
        transport.abort()
        if mode != "baseline" and untrusted_engine_stage_started:
            raise OuterSessionCandidateError(
                f"candidate arm failed deterministically: {exc}"
            ) from None
        raise OuterSessionInfrastructureError(
            f"stock/pre-candidate OCI session failed: {exc}"
        ) from None
    except BaseException:
        transport.abort()
        raise

    timed_point = statistics.median(samples) if samples else 0.0
    effective_point = (
        min(timed_point, conditioning_tok_per_s)
        if warmup_iters > 0 else timed_point
    )
    result = ModeResult(
        tok_per_s=effective_point,
        tok_per_s_samples=samples,
        tokens=total_tokens,
        per_prompt=timed_per_prompt,
        conditioning_tok_per_s=conditioning_tok_per_s,
        texts=timed_texts,
        per_prompt_batches=timed_batches,
        warmup_per_prompt=warmup_per_prompt,
        warmup_texts=warmup_texts,
        warmup_per_prompt_batches=warmup_batches,
    )
    if posthoc_plan is not None:
        return result, teacher_traces
    return (result, audit_receipts, audit_members) if mode == "candidate_audit" else result
