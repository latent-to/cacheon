from __future__ import annotations

import json
import struct
from dataclasses import replace

import pytest

from optima.eval.oci_outer_session import (
    ContainerSessionTransport,
    OuterSessionCandidateError,
    OuterSessionInfrastructureError,
    OuterSessionPosthocCandidateError,
    OuterSessionTimeoutError,
    run_outer_timed_session,
)
from optima.eval.oci_session_protocol import (
    EVIDENCE_MAGIC,
    TEACHER_EVIDENCE_MAGIC,
    FRAME_MAGIC,
    MAX_BATCH_REQUEST_BYTES,
    MAX_BATCH_RESPONSE_BYTES,
    MAX_CONTROL_BYTES,
    MAX_INIT_BYTES,
    BatchEvidence,
    SessionProtocolError,
    batch_response,
    closed_message,
    decode_message,
    evidence_frame,
    error_message,
    expected_evidence_payload_bytes,
    frame_message,
    make_init,
    parse_evidence_frame_bytes,
    parse_frame_bytes,
    preflight_message,
    ready_message,
    teacher_evidence_frame,
    validate_batch_request,
    validate_batch_response,
    validate_close_request,
    validate_init,
    validate_teacher_request,
)
from optima.eval.external_quality import (
    TeacherForcedPromptTrace,
    TeacherForcedTrace,
    seal_posthoc_reference_plan,
)
from optima.eval.throughput_kl import EvalConfig
from optima.eval.throughput_kl import is_transient_launch_failure


def _cfg() -> EvalConfig:
    return EvalConfig(
        model_path="host-model",
        num_prompts=2,
        max_new_tokens=2,
        top_logprobs_num=2,
        warmup_iters=1,
        conditioning_iters=1,
        timed_iters=1,
        tp_size=4,
        isolate=True,
        allow_unsafe_no_isolation=False,
    )


def _batches():
    return [["warm-a", "warm-b"], ["timed-a", "timed-b"]]


def _runtime_attestation():
    return {
        "verified": True,
        "sglang_version": "0.0.test",
        "referee_source_digest": "sha256:" + "1" * 64,
        "model_revision": "2" * 40,
        "model_manifest_digest": "sha256:" + "3" * 64,
        "model_content_digest": "sha256:" + "4" * 64,
        "environment_sha256": "5" * 64,
        "gpu_count": 4,
        "gpu_architectures": ["sm103"] * 4,
        "topology_sha256": "6" * 64,
    }


def test_session_init_redacts_controller_prompt_plan_entropy():
    cfg = replace(_cfg(), prompt_seed=7_771_991_337, input_len=225_000)
    planned = _batches()
    init = make_init(cfg, mode="candidate", session_id="a" * 32)
    encoded = json.dumps(init, sort_keys=True)

    assert "eval_config" not in init
    assert "engine_config" in init
    assert "prompt_seed" not in init["engine_config"]
    assert "input_len" not in init["engine_config"]
    assert str(cfg.prompt_seed) not in encoded
    assert all(prompt not in encoded for batch in planned for prompt in batch)
    with pytest.raises(KeyError):
        _ = init["engine_config"]["prompt_seed"]


class ScriptedTransport:
    def __init__(
        self,
        *,
        early=False,
        trailing=False,
        wrong_nonce=False,
        duplicate_preflight=False,
        duplicate_ready=False,
        error_before_ready=False,
        empty_warmup=False,
        teacher_timeout=False,
        teacher_candidate_error=False,
    ):
        self.early = early
        self.trailing = trailing
        self.wrong_nonce = wrong_nonce
        self.duplicate_preflight = duplicate_preflight
        self.duplicate_ready = duplicate_ready
        self.error_before_ready = error_before_ready
        self.empty_warmup = empty_warmup
        self.teacher_timeout = teacher_timeout
        self.teacher_candidate_error = teacher_candidate_error
        self.started = False
        self.aborted = False
        self.exited = False
        self.pending = []
        self.pending_after_read = False
        self.events = []
        self.config = None
        self.teacher_requests = []

    def start(self):
        self.started = True
        self.events.append("start")

    def has_pending_output(self):
        if self.early:
            self.early = False
            return True
        if self.pending_after_read:
            return True
        return bool(self.pending)

    def write_frame(self, frame, *, deadline):
        self.events.append("write")
        message = parse_frame_bytes(frame, max_bytes=MAX_BATCH_REQUEST_BYTES)
        typ = message.get("type")
        if typ == "init":
            session_id, mode, self.config = validate_init(message)
            if self.duplicate_preflight:
                preflight = (
                    '{"schema":"optima-outer-session-v2","type":"preflight",'
                    '"type":"preflight","session_id":"%s","mode":"%s"}'
                    % (session_id, mode)
                ).encode()
            else:
                preflight = frame_message(
                    preflight_message(
                        session_id=session_id,
                        mode=mode,
                        runtime_attestation=_runtime_attestation(),
                    ),
                    max_bytes=MAX_CONTROL_BYTES,
                )[8:]
            if self.error_before_ready:
                ready = frame_message(
                    error_message(
                        session_id=session_id,
                        stage="engine",
                        error=RuntimeError("system overlay activation failed"),
                    ),
                    max_bytes=MAX_CONTROL_BYTES,
                )[8:]
            elif self.duplicate_ready:
                raw = (
                    '{"schema":"optima-outer-session-v2","type":"ready",'
                    '"type":"ready","session_id":"%s","mode":"%s"}'
                    % (session_id, mode)
                ).encode()
                ready = raw
            else:
                ready = frame_message(
                    ready_message(session_id=session_id, mode=mode),
                    max_bytes=MAX_CONTROL_BYTES,
                )[8:]
            self.pending = [(FRAME_MAGIC, preflight), (FRAME_MAGIC, ready)]
            return
        if typ == "batch":
            session_id, request_id, nonce, index, warmup, prompts = (
                validate_batch_request(
                    message, expected_count=int(self.config["num_prompts"])
                )
            )
            if self.wrong_nonce:
                nonce = "f" * 32
            if warmup and self.empty_warmup:
                evidence = BatchEvidence(
                    [([], []) for _ in prompts], ["" for _ in prompts], 0
                )
            else:
                per_prompt = []
                for prompt_index, _prompt in enumerate(prompts):
                    ids = [10 + prompt_index, 10 + prompt_index]
                    position = [(-0.1, ids[0], None), (-3.0, 99, None)]
                    per_prompt.append((ids, [list(position), list(position)]))
                evidence = BatchEvidence(per_prompt, ["", ""], 4)
            framed = evidence_frame(
                evidence,
                session_id=session_id,
                request_id=request_id,
                nonce=nonce,
                batch_index=index,
                require_logprobs=not (warmup and self.empty_warmup),
            )
            self.pending = [(EVIDENCE_MAGIC, framed[8:])]
            self.pending_after_read = self.trailing
            return
        if typ == "teacher":
            count = len(message.get("prompts", []))
            (
                session_id, request_id, nonce, phase, batch_index, seal,
                prompts, sources,
            ) = validate_teacher_request(
                message,
                expected_count=count,
                expected_tokens=int(self.config["max_new_tokens"]),
            )
            self.teacher_requests.append((phase, batch_index, prompts, sources))
            traces = []
            for prompt_index, _prompt in enumerate(prompts):
                def trace(source):
                    ids = sources[source][prompt_index]
                    positions = tuple(
                        ((-0.1, token, None), (-3.0, 99, None))
                        for token in ids
                    )
                    return TeacherForcedTrace(tuple(-0.1 for _ in ids), positions)

                traces.append(TeacherForcedPromptTrace(
                    prompt_token_count=3,
                    prompt_token_sha256=(f"{prompt_index + 1:x}" * 64)[:64],
                    baseline=trace("baseline"),
                    candidate=trace("candidate"),
                    stock_control=trace("stock_control"),
                ))
            framed = teacher_evidence_frame(
                traces,
                session_id=session_id,
                request_id=request_id,
                nonce=nonce,
                phase=phase,
                batch_index=batch_index,
                sealed_rollout_sha256=seal,
                token_count=int(self.config["max_new_tokens"]),
                top_logprobs_num=int(self.config["top_logprobs_num"]),
            )
            self.pending = [(TEACHER_EVIDENCE_MAGIC, framed[8:])]
            return
        if typ == "close":
            session_id, request_id, nonce = validate_close_request(message)
            framed = frame_message(
                closed_message(
                    session_id=session_id,
                    request_id=request_id,
                    nonce=nonce,
                    audit_receipts=[],
                    audit_members=[],
                ),
                max_bytes=MAX_CONTROL_BYTES,
            )
            self.pending = [(FRAME_MAGIC, framed[8:])]
            return
        raise AssertionError(typ)

    def read_frame(self, *, magic, max_bytes, deadline, exact_bytes=None):
        self.events.append("read")
        if magic == TEACHER_EVIDENCE_MAGIC and self.teacher_timeout:
            raise OuterSessionTimeoutError("forced teacher timeout")
        if magic == TEACHER_EVIDENCE_MAGIC and self.teacher_candidate_error:
            raise OuterSessionPosthocCandidateError(
                "CandidateTeacherInputError: out-of-vocabulary"
            )
        assert self.pending
        got_magic, payload = self.pending.pop(0)
        assert got_magic == magic
        if exact_bytes is not None and len(payload) != exact_bytes:
            raise OuterSessionCandidateError("wrong evidence size")
        return payload

    def expect_clean_exit(self, *, deadline):
        self.events.append("exit")
        self.exited = True

    def abort(self):
        self.aborted = True


class RecordingClock:
    def __init__(self):
        self.values = iter((0.0, 10.0, 11.0, 20.0, 22.0))
        self.events = []

    def __call__(self):
        self.events.append("clock")
        return next(self.values)


def test_outer_controller_owns_timing_and_constructs_mode_result():
    transport = ScriptedTransport()
    clock = RecordingClock()
    result = run_outer_timed_session(
        _cfg(),
        _batches(),
        mode="candidate",
        transport=transport,
        init_timeout_s=5,
        batch_timeout_s=5,
        clock=clock,
        expected_runtime_attestation=_runtime_attestation(),
    )
    assert result.tok_per_s_samples == [2.0]  # 4 fixed tokens / 2 trusted seconds
    # Eight warmup+first-timed tokens span the complete trusted ready -> first
    # timed completion tail (22s). That 4/11 tok/s aggregate—not the later 2
    # tok/s timed batch—is the mandatory floor.
    assert result.conditioning_tok_per_s == pytest.approx(4 / 11)
    assert result.tok_per_s == pytest.approx(4 / 11)
    assert result.tokens == 4
    assert len(result.per_prompt) == len(result.texts) == 2
    assert len(result.warmup_per_prompt) == len(result.warmup_texts) == 2
    assert len(result.per_prompt_batches) == len(result.warmup_per_prompt_batches) == 1
    assert transport.aborted and not transport.exited
    # One tail-start read plus exactly two reads around each request.
    assert clock.events == ["clock"] * 5
    assert transport.events.count("write") == 3  # init + exactly two batches
    assert transport.events.count("read") == 4  # preflight + ready + two batches


def test_stock_bookend_survives_only_for_sealed_posthoc_teacher_frames():
    position_a = ((-0.1, 10, None), (-3.0, 99, None))
    position_b = ((-0.1, 11, None), (-3.0, 99, None))
    batch = [
        ((10, 10), (position_a, position_a)),
        ((11, 11), (position_b, position_b)),
    ]
    plan = seal_posthoc_reference_plan(
        _batches(),
        baseline_batches=[batch, batch],
        candidate_batches=[batch, batch],
        warmup_iters=1,
        clusters_per_batch=2,
        expected_tokens=2,
        topk_num=2,
        selection_secret=b"x" * 32,
    )
    transport = ScriptedTransport()
    clock = RecordingClock()
    result, traces = run_outer_timed_session(
        _cfg(),
        _batches(),
        mode="baseline",
        transport=transport,
        init_timeout_s=5,
        batch_timeout_s=5,
        clock=clock,
        expected_runtime_attestation=_runtime_attestation(),
        posthoc_plan=plan,
    )
    assert result.tokens == 4
    assert set(traces) == {("warmup", 0), ("timed", 0)}
    assert [item[:2] for item in transport.teacher_requests] == [
        ("warmup", 0), ("timed", 0)
    ]
    assert transport.aborted


def test_teacher_timeout_is_retryable_infrastructure_without_proven_candidate_cause():
    position_a = ((-0.1, 10, None), (-3.0, 99, None))
    position_b = ((-0.1, 11, None), (-3.0, 99, None))
    batch = [
        ((10, 10), (position_a, position_a)),
        ((11, 11), (position_b, position_b)),
    ]
    plan = seal_posthoc_reference_plan(
        _batches(),
        baseline_batches=[batch, batch],
        candidate_batches=[batch, batch],
        warmup_iters=1,
        clusters_per_batch=2,
        expected_tokens=2,
        topk_num=2,
        selection_secret=b"y" * 32,
    )
    transport = ScriptedTransport(teacher_timeout=True)
    with pytest.raises(
        OuterSessionInfrastructureError, match="without validated candidate causation"
    ) as caught:
        run_outer_timed_session(
            _cfg(),
            _batches(),
            mode="baseline",
            transport=transport,
            init_timeout_s=5,
            batch_timeout_s=5,
            clock=RecordingClock(),
            expected_runtime_attestation=_runtime_attestation(),
            posthoc_plan=plan,
        )
    assert caught.value.retryable is True
    assert transport.aborted


def test_validated_candidate_teacher_input_error_is_terminal():
    position_a = ((-0.1, 10, None), (-3.0, 99, None))
    position_b = ((-0.1, 11, None), (-3.0, 99, None))
    batch = [
        ((10, 10), (position_a, position_a)),
        ((11, 11), (position_b, position_b)),
    ]
    plan = seal_posthoc_reference_plan(
        _batches(),
        baseline_batches=[batch, batch],
        candidate_batches=[batch, batch],
        warmup_iters=1,
        clusters_per_batch=2,
        expected_tokens=2,
        topk_num=2,
        selection_secret=b"z" * 32,
    )
    transport = ScriptedTransport(teacher_candidate_error=True)
    with pytest.raises(
        OuterSessionPosthocCandidateError, match="out-of-vocabulary"
    ) as caught:
        run_outer_timed_session(
            _cfg(),
            _batches(),
            mode="baseline",
            transport=transport,
            init_timeout_s=5,
            batch_timeout_s=5,
            clock=RecordingClock(),
            expected_runtime_attestation=_runtime_attestation(),
            posthoc_plan=plan,
        )
    assert caught.value.retryable is False
    assert transport.aborted


def test_slow_final_warmup_caps_otherwise_fast_timed_throughput():
    values = iter((0.0, 1.0, 21.0, 22.0, 23.0))
    result = run_outer_timed_session(
        _cfg(),
        _batches(),
        mode="candidate",
        transport=ScriptedTransport(),
        init_timeout_s=5,
        batch_timeout_s=5,
        clock=lambda: next(values),
        expected_runtime_attestation=_runtime_attestation(),
    )

    assert result.tok_per_s_samples == [4.0]
    assert result.conditioning_tok_per_s == 0.2
    assert result.tok_per_s == 0.2


def test_slow_earlier_warmup_cannot_buy_a_discarded_cooldown_window():
    cfg = replace(_cfg(), warmup_iters=2, conditioning_iters=2)
    batches = [
        ["warm-1a", "warm-1b"],
        ["warm-2a", "warm-2b"],
        ["timed-a", "timed-b"],
    ]
    # Ready at 0. Warmup 1 sacrifices 50 seconds; later batches each take one.
    # The per-batch minimum retains the sacrificed warmup instead of allowing the
    # candidate to use it as an uncharged cooldown before the final warmup.
    values = iter((0.0, 1.0, 51.0, 52.0, 53.0, 54.0, 55.0))
    result = run_outer_timed_session(
        cfg,
        batches,
        mode="candidate",
        transport=ScriptedTransport(),
        init_timeout_s=5,
        batch_timeout_s=60,
        clock=lambda: next(values),
        expected_runtime_attestation=_runtime_attestation(),
    )

    assert result.tok_per_s_samples == [4.0]
    assert result.conditioning_tok_per_s == pytest.approx(0.08)
    assert result.tok_per_s == pytest.approx(0.08)
    assert len(result.warmup_per_prompt_batches) == 2


def test_free_setup_warmup_absorbs_lazy_jit_before_continuous_tail():
    cfg = replace(_cfg(), warmup_iters=3, conditioning_iters=2)
    batches = [
        ["setup-a", "setup-b"],
        ["condition-1a", "condition-1b"],
        ["condition-2a", "condition-2b"],
        ["timed-a", "timed-b"],
    ]
    # The first request spends 50 seconds in legitimate request-lazy setup. The
    # declared two-warmup conditioning tail begins at its completed response and
    # then runs continuously at four tokens/second through first timed completion.
    values = iter((0.0, 50.0, 50.0, 51.0, 51.0, 52.0, 52.0, 53.0))
    result = run_outer_timed_session(
        cfg,
        batches,
        mode="candidate",
        transport=ScriptedTransport(),
        init_timeout_s=5,
        batch_timeout_s=60,
        clock=lambda: next(values),
        expected_runtime_attestation=_runtime_attestation(),
    )

    assert result.tok_per_s_samples == [4.0]
    assert result.conditioning_tok_per_s == pytest.approx(4.0)
    assert result.tok_per_s == pytest.approx(4.0)
    assert len(result.warmup_per_prompt_batches) == 3


def test_partial_transport_start_is_inside_abort_boundary():
    class PartialStart(ScriptedTransport):
        def start(self):
            self.started = True
            raise OuterSessionInfrastructureError("partial runtime start")

    transport = PartialStart()
    with pytest.raises(OuterSessionInfrastructureError, match="partial runtime"):
        run_outer_timed_session(
            _cfg(),
            _batches(),
            mode="candidate",
            transport=transport,
            init_timeout_s=5,
            batch_timeout_s=5,
            clock=RecordingClock(),
        )
    assert transport.aborted


@pytest.mark.parametrize(
    "prompt_batches,match",
    [
        ([['only-one-batch', 'still-one-batch']], "expected 2"),
        ([['warm-a'], ['timed-a']], "batch count"),
        ([['same', 'warm-b'], ['same', 'timed-b']], "globally disjoint"),
        ([['warm-a', 'warm-b'], ['timed-a', 7]], "globally disjoint"),
    ],
)
def test_invalid_controller_prompt_plan_is_infrastructure(
    prompt_batches, match,
):
    transport = ScriptedTransport()
    with pytest.raises(
        OuterSessionInfrastructureError,
        match=match,
    ) as failure:
        run_outer_timed_session(
            _cfg(),
            prompt_batches,
            mode="candidate",
            transport=transport,
            init_timeout_s=5,
            batch_timeout_s=5,
            clock=RecordingClock(),
        )
    assert failure.value.retryable is True
    assert not transport.started


def test_oversized_controller_prompt_is_rejected_before_transport_start():
    transport = ScriptedTransport()
    huge_prompt = "x" * (9 * 1024 * 1024)
    with pytest.raises(
        OuterSessionInfrastructureError,
        match="prompt batch 0 violates protocol policy|too large",
    ) as failure:
        run_outer_timed_session(
            _cfg(),
            [[huge_prompt, "warm-b"], ["timed-a", "timed-b"]],
            mode="candidate",
            transport=transport,
            init_timeout_s=5,
            batch_timeout_s=5,
            clock=RecordingClock(),
        )
    assert failure.value.retryable is True
    assert not transport.started and not transport.aborted


def test_candidate_preflight_timeout_is_validator_infrastructure():
    class PreflightTimeout(ScriptedTransport):
        def read_frame(self, *, magic, max_bytes, deadline, exact_bytes=None):
            raise OuterSessionTimeoutError("preflight read timed out")

    transport = PreflightTimeout()
    with pytest.raises(
        OuterSessionInfrastructureError,
        match="pre-candidate.*preflight read timed out",
    ) as failure:
        run_outer_timed_session(
            _cfg(),
            _batches(),
            mode="candidate",
            transport=transport,
            init_timeout_s=5,
            batch_timeout_s=5,
            clock=RecordingClock(),
            expected_runtime_attestation=_runtime_attestation(),
        )
    assert failure.value.retryable is True
    assert transport.aborted


def test_host_boundary_hook_spans_final_warmup_without_entering_timed_clock():
    events = []
    clock = RecordingClock()

    def boundary(event, mode, batch_index, deadline):
        events.append((event, mode, batch_index, deadline))

    result = run_outer_timed_session(
        _cfg(),
        _batches(),
        mode="candidate",
        transport=ScriptedTransport(),
        init_timeout_s=5,
        batch_timeout_s=5,
        clock=clock,
        warmup_timed_boundary=boundary,
    )

    assert [event[0] for event in events] == [
        "before_final_warmup",
        "after_final_warmup",
        "before_first_timed",
    ]
    assert all(event[1] == "candidate" for event in events)
    assert [event[2] for event in events] == [0, 0, 1]
    assert len(result.tok_per_s_samples) == 1


def test_typed_outer_failures_drive_retry_policy_without_message_matching():
    assert is_transient_launch_failure(OuterSessionInfrastructureError("runtime"))
    assert is_transient_launch_failure(OuterSessionTimeoutError("stock timeout"))
    assert not is_transient_launch_failure(OuterSessionCandidateError("bad evidence"))


def test_same_bad_protocol_is_infrastructure_for_stock_but_terminal_for_c():
    with pytest.raises(OuterSessionInfrastructureError) as stock:
        run_outer_timed_session(
            _cfg(), _batches(), mode="baseline",
            transport=ScriptedTransport(duplicate_ready=True),
            init_timeout_s=5, batch_timeout_s=5, clock=RecordingClock(),
        )
    assert stock.value.retryable is True

    with pytest.raises(OuterSessionCandidateError) as candidate:
        run_outer_timed_session(
            _cfg(), _batches(), mode="candidate",
            transport=ScriptedTransport(duplicate_ready=True),
            init_timeout_s=5, batch_timeout_s=5, clock=RecordingClock(),
        )
    assert candidate.value.retryable is False


@pytest.mark.parametrize("mode", ["baseline", "candidate"])
def test_pre_init_output_is_validator_infrastructure_not_miner_blame(mode):
    with pytest.raises(OuterSessionInfrastructureError):
        run_outer_timed_session(
            _cfg(), _batches(), mode=mode,
            transport=ScriptedTransport(early=True),
            init_timeout_s=5, batch_timeout_s=5, clock=RecordingClock(),
        )


def test_candidate_preflight_mismatch_is_validator_infrastructure():
    expected = _runtime_attestation()
    expected["environment_sha256"] = "9" * 64
    transport = ScriptedTransport()
    with pytest.raises(
        OuterSessionInfrastructureError,
        match="runtime attestation|environment",
    ) as failure:
        run_outer_timed_session(
            _cfg(),
            _batches(),
            mode="candidate",
            transport=transport,
            init_timeout_s=5,
            batch_timeout_s=5,
            clock=RecordingClock(),
            expected_runtime_attestation=expected,
        )
    assert failure.value.retryable is True
    assert transport.aborted


def test_malformed_candidate_preflight_is_validator_infrastructure():
    transport = ScriptedTransport(duplicate_preflight=True)
    with pytest.raises(
        OuterSessionInfrastructureError,
        match="duplicate JSON key",
    ) as failure:
        run_outer_timed_session(
            _cfg(),
            _batches(),
            mode="candidate",
            transport=transport,
            init_timeout_s=5,
            batch_timeout_s=5,
            clock=RecordingClock(),
            expected_runtime_attestation=_runtime_attestation(),
        )
    assert failure.value.retryable is True
    assert transport.aborted


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"trailing": True}, "trailing/early"),
        ({"wrong_nonce": True}, "binding mismatch"),
        ({"duplicate_ready": True}, "duplicate JSON key"),
        ({"empty_warmup": True}, "wrong evidence size"),
    ],
)
def test_outer_session_rejects_malicious_markers(kwargs, match):
    transport = ScriptedTransport(**kwargs)
    with pytest.raises(OuterSessionCandidateError, match=match):
        run_outer_timed_session(
            _cfg(),
            _batches(),
            mode="candidate",
            transport=transport,
            init_timeout_s=5,
            batch_timeout_s=5,
            clock=RecordingClock(),
        )
    assert transport.aborted


def test_control_protocol_rejects_duplicate_trailing_and_oversized_frames():
    duplicate = b'{"schema":"x","type":"a","type":"b"}'
    with pytest.raises(SessionProtocolError, match="duplicate JSON key"):
        decode_message(duplicate, max_bytes=1024)
    good = frame_message({"schema": "x"}, max_bytes=1024)
    with pytest.raises(SessionProtocolError, match="trailing"):
        parse_frame_bytes(good + b"x", max_bytes=1024)
    oversized = FRAME_MAGIC + struct.pack(">I", MAX_INIT_BYTES + 1)
    with pytest.raises(SessionProtocolError, match="declares more"):
        parse_frame_bytes(oversized, max_bytes=MAX_INIT_BYTES)


def test_transport_surfaces_bounded_worker_error_instead_of_losing_wrong_magic():
    frame = frame_message(
        error_message(
            session_id="1" * 32,
            stage="batch-2",
            error=RuntimeError("top-k evidence was incomplete"),
        ),
        max_bytes=MAX_CONTROL_BYTES,
    )
    chunks = iter((frame[:8], frame[8:]))
    transport = ContainerSessionTransport(("docker",), force_remove=lambda: None)
    transport._read_exact = lambda _size, *, deadline: next(chunks)  # type: ignore[method-assign]

    with pytest.raises(
        OuterSessionCandidateError,
        match="batch-2.*RuntimeError.*top-k evidence was incomplete",
    ):
        transport.read_frame(
            magic=EVIDENCE_MAGIC,
            max_bytes=MAX_BATCH_RESPONSE_BYTES,
            deadline=10.0,
        )


def test_outer_session_surfaces_worker_error_before_ready():
    transport = ScriptedTransport(error_before_ready=True)
    with pytest.raises(
        OuterSessionCandidateError,
        match="engine.*RuntimeError.*system overlay activation failed",
    ):
        run_outer_timed_session(
            _cfg(),
            _batches(),
            mode="candidate",
            transport=transport,
            init_timeout_s=5,
            batch_timeout_s=5,
            clock=RecordingClock(),
        )
    assert transport.aborted


def test_binary_evidence_is_exact_length_and_semantically_bound():
    cfg = _cfg()
    evidence = BatchEvidence(
        per_prompt=[
            ([7, 7], [[(-0.1, 7, None), (-3.0, 8, None)]] * 2),
            ([9, 9], [[(-0.1, 9, None), (-3.0, 10, None)]] * 2),
        ],
        texts=["", ""],
        observed_tokens=4,
    )
    kwargs = dict(
        session_id="1" * 32,
        request_id="2" * 32,
        nonce="3" * 32,
        batch_index=1,
        expected_prompts=2,
        max_new_tokens=2,
        top_logprobs_num=2,
        ignore_eos=True,
        require_logprobs=True,
        temperature=0.0,
    )
    frame = evidence_frame(
        evidence,
        session_id=kwargs["session_id"],
        request_id=kwargs["request_id"],
        nonce=kwargs["nonce"],
        batch_index=kwargs["batch_index"],
        require_logprobs=True,
    )
    expected = expected_evidence_payload_bytes(
        prompt_count=2,
        max_new_tokens=2,
        top_logprobs_num=2,
        require_logprobs=True,
        ignore_eos=True,
    )
    assert len(frame) == 8 + expected
    decoded = parse_evidence_frame_bytes(frame, **kwargs)
    assert decoded.observed_tokens == 4

    wrong = dict(kwargs, nonce="4" * 32)
    with pytest.raises(SessionProtocolError, match="binding mismatch"):
        parse_evidence_frame_bytes(frame, **wrong)
    with pytest.raises(SessionProtocolError, match="trailing"):
        parse_evidence_frame_bytes(frame + b"x", **kwargs)


def test_binary_evidence_preserves_sampled_token_and_raw_topk_independently():
    evidence = BatchEvidence(
        per_prompt=[([7], [[(-0.8, 8, None), (-1.0, 7, None)]])],
        texts=[""],
        observed_tokens=1,
    )
    frame = evidence_frame(
        evidence,
        session_id="1" * 32,
        request_id="2" * 32,
        nonce="3" * 32,
        batch_index=0,
        require_logprobs=True,
    )
    decoded = parse_evidence_frame_bytes(
        frame,
        session_id="1" * 32,
        request_id="2" * 32,
        nonce="3" * 32,
        batch_index=0,
        expected_prompts=1,
        max_new_tokens=1,
        top_logprobs_num=2,
        ignore_eos=True,
        require_logprobs=True,
        temperature=0.0,
    )

    assert decoded.per_prompt[0][0] == [7]
    assert decoded.per_prompt[0][1][0][0][1] == 8


def test_json_evidence_preserves_sampled_token_and_raw_topk_independently():
    message = batch_response(
        session_id="1" * 32,
        request_id="2" * 32,
        nonce="3" * 32,
        batch_index=0,
        items=[{
            "output_ids": [7],
            "top_logprobs": [[[-0.8, 8], [-1.0, 7]]],
            "text": "",
        }],
    )
    decoded = validate_batch_response(
        message,
        session_id="1" * 32,
        request_id="2" * 32,
        nonce="3" * 32,
        batch_index=0,
        expected_prompts=1,
        max_new_tokens=1,
        top_logprobs_num=2,
        ignore_eos=True,
        require_logprobs=True,
        temperature=0.0,
    )

    assert decoded.per_prompt[0][0] == [7]
    assert decoded.per_prompt[0][1][0][0][1] == 8


def test_host_rejects_topk_that_is_not_sorted_or_normalized():
    base = dict(
        session_id="1" * 32,
        request_id="2" * 32,
        nonce="3" * 32,
        batch_index=0,
        require_logprobs=True,
    )

    def decode(position, output=7):
        frame = evidence_frame(
            BatchEvidence([([output], [[*position]])], [""], 1), **base
        )
        return parse_evidence_frame_bytes(
            frame,
            expected_prompts=1,
            max_new_tokens=1,
            top_logprobs_num=2,
            ignore_eos=True,
            require_logprobs=True,
            temperature=0.0,
            **{key: base[key] for key in ("session_id", "request_id", "nonce", "batch_index")},
        )

    with pytest.raises(SessionProtocolError, match="descending"):
        decode([(-3.0, 7, None), (-0.1, 8, None)])
    with pytest.raises(SessionProtocolError, match="probability mass"):
        decode([(-0.1, 7, None), (-0.1, 8, None)])
