"""Phase delivery measurements survive interleaving and framing."""

import asyncio
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.support.pipes import engine_loop as engine_loop

pytestmark = pytest.mark.usefixtures("engine_loop")

from cacheon.eval.continuation_codec import ContinuationCodec
from cacheon.eval.oci_outer_session import (
    AttachedSessionTransport,
    BatchExecutionEvidence,
    run_outer_session,
)
from cacheon.eval.oci_session_protocol import (
    MAX_CONTROL_BYTES,
    BatchRequest,
    SessionProtocolError,
    evidence_frame,
    parse_frame_bytes,
    validate_batch_request,
)
from tests.support.pipes import generate as _generate
from cacheon.eval.phase_latency import HostTokenClock, engine_outputs, generate_outputs, token_boundary
from cacheon.eval.scoring import marginal_workload_digest
from tests.support.pipes import PipeClient, PipeManager
from tests.support.replay import replay_plan, replay_session
from tests.test_oci_outer_session import _Clock, _FakeTransport, _facts, _plan


def _request(**kwargs):
    return replace(
        BatchRequest(
            "a" * 32,
            "b" * 64,
            "c" * 32,
            "d" * 32,
            0,
            ("alpha", "beta"),
            4,
            0,
            0.0,
            5,
            True,
        ),
        **kwargs,
    )


def _chunk(index, count, *, complete=False, prompt_tokens=5):
    return {
        "index": index,
        "output_ids": list(range(10 * index, 10 * index + count)),
        "meta_info": {
            "completion_tokens": count,
            "prompt_tokens": prompt_tokens,
            "finish_reason": {"type": "length"} if complete else None,
        },
    }


def _engine(chunks, calls):
    async def generate(**kwargs):
        calls.append(kwargs)

        async def stream():
            for chunk in chunks:
                yield chunk

        return stream()

    return SimpleNamespace(async_generate=generate, loop=asyncio.get_event_loop())


def _message(request, index, boundary, token):
    return parse_frame_bytes(
        token_boundary(request, index, boundary, token), max_bytes=MAX_CONTROL_BYTES
    )


def test_interleaved_stream_preserves_exact_outputs_and_emits_only_two_boundaries_per_prompt():
    chunks = [
        _chunk(1, 1),
        _chunk(0, 2),
        _chunk(1, 3),
        _chunk(0, 4, complete=True),
        _chunk(1, 4, complete=True),
    ]
    frames, calls = [], []
    evidence = _generate(_engine(chunks, calls), _request(), frames.append)
    assert [prompt.output_ids for prompt in evidence.prompts] == [
        (0, 1, 2, 3),
        (10, 11, 12, 13),
    ]
    assert len(calls) == 1 and calls[0]["stream"] is True
    assert calls[0]["return_logprob"] is False
    assert calls[0]["sampling_params"]["max_new_tokens"] == 4
    assert [
        (m["prompt_index"], m["boundary"], m["token_id"])
        for m in (
            parse_frame_bytes(frame, max_bytes=MAX_CONTROL_BYTES) for frame in frames
        )
    ] == [(1, "first", 10), (0, "first", 0), (0, "last", 3), (1, "last", 13)]


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"index": 3}, "prompt index"),
        ({"output_ids": [0]}, "cumulative"),
        (
            {
                "meta_info": {
                    "completion_tokens": 2,
                    "finish_reason": {"type": "length"},
                }
            },
            "exact token budget",
        ),
    ],
)
def test_malformed_stream_cannot_be_reported_as_a_valid_measurement(changes, match):
    chunk = {**_chunk(0, 2), **changes}
    with pytest.raises(SessionProtocolError, match=match):
        _generate(_engine([chunk], []), _request(), lambda _: None)


def test_stream_exhaustion_and_repeated_completion_fail_loudly():
    for chunks, match in (
        ([_chunk(0, 1)], "every prompt"),
        ([_chunk(0, 4, complete=True)] * 2, "repeated"),
    ):
        with pytest.raises(SessionProtocolError, match=match):
            _generate(_engine(chunks, []), _request(), lambda _: None)


def test_live_intermediate_ids_can_advance_beyond_the_usage_snapshot():
    request = _request(prompts=("alpha",))
    intermediate = _chunk(0, 1)
    # The pinned engine shares this list, then awaits batch fan-in. Its
    # producer can append another token before the consumer sees the row.
    intermediate["output_ids"].append(1)
    evidence = _generate(
        _engine([intermediate, _chunk(0, 4, complete=True)], []), request, lambda _: None,
    )
    assert evidence.prompts[0].output_ids == (0, 1, 2, 3)
    final = _chunk(0, 4, complete=True)
    final["meta_info"]["completion_tokens"] = 3
    with pytest.raises(SessionProtocolError, match="cumulative"):
        _generate(_engine([final], []), request, lambda _: None)


def test_one_token_turn_retains_ttft_without_inventing_a_decode_interval():
    request = _request(prompts=("alpha",), max_new_tokens=1)
    clock = _Clock(2.0)
    observed = HostTokenClock(request, clock, 1.0)
    evidence = _generate(
        _engine([_chunk(0, 1, complete=True)], []), request,
        lambda frame: observed.observe(parse_frame_bytes(frame, max_bytes=MAX_CONTROL_BYTES)),
    )
    assert validate_batch_request(request.to_dict()) == request
    assert observed.finish(evidence, 3.0) == ((1.0, 1.0),)
    assert evidence.prompts[0].output_ids == (0,)


def test_tokenized_chat_inputs_and_sticky_rank_reach_the_same_generation_path():
    token_ids = [[7, 8, 9]]
    request = _request(prompts=(), input_ids=token_ids, routed_dp_rank=2, expected_prompt_tokens=3)
    token_ids[0].append(10)
    calls = []
    evidence = _generate(
        _engine([_chunk(0, 1, prompt_tokens=3), _chunk(0, 4, complete=True, prompt_tokens=3)], calls),
        request, lambda _frame: None,
    )
    assert validate_batch_request(request.to_dict()) == request
    assert calls[0]["input_ids"] == [[7, 8, 9]] and "prompt" not in calls[0]
    assert calls[0]["routed_dp_rank"] == 2 and request.prompt_count == 1
    assert evidence.prompts[0].prompt_tokens == 3


def test_real_frame_reader_uses_host_delivery_clock_and_checks_final_token_ids():
    request, client = _request(), PipeClient()
    clock = _Clock(10.0)
    collector = HostTokenClock(request, clock, 10.0)
    frames, calls = [], []
    raw = _generate(
        _engine(
            [
                _chunk(1, 1),
                _chunk(0, 1),
                _chunk(0, 4, complete=True),
                _chunk(1, 4, complete=True),
            ],
            calls,
        ),
        request,
        frames.append,
    )
    transport = AttachedSessionTransport(
        PipeManager(client), object(), ("worker",), clock=clock
    )
    transport.start()
    try:
        os.write(
            client.response_write,
            b"".join(frames) + evidence_frame(raw, request=request),
        )
        times = iter((11.0, 12.0, 15.0, 17.0))

        def received(message):
            clock.value = next(times)
            collector.observe(message)

        evidence = asyncio.run(transport.aread_response(
            {request.request_id: request}, deadline=100.0, on_progress=received
        ))[1]
        assert collector.finish(evidence, 18.0) == ((2.0, 5.0), (1.0, 7.0))
        assert evidence == raw
        assert not transport.has_pending_output()
    finally:
        client.close()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda m: m.update(nonce="e" * 32), "disclosed request"),
        (lambda m: m.update(worker_seconds=0.001), "disclosed request"),
        (lambda m: m.update(boundary="last"), "out of order"),
        (lambda m: m.update(prompt_index=-1), "malformed"),
    ],
)
def test_progress_cannot_supply_timing_or_escape_its_request(mutation, match):
    request, clock = _request(), _Clock(2.0)
    collector = HostTokenClock(request, clock, 1.0)
    message = _message(request, 0, "first", 0)
    mutation(message)
    with pytest.raises(SessionProtocolError, match=match):
        collector.observe(message)


def test_missing_duplicate_and_forged_token_boundaries_are_not_metrics():
    request, clock = _request(prompts=("alpha",)), _Clock(2.0)
    collector = HostTokenClock(request, clock, 1.0)
    raw = _generate(_engine([_chunk(0, 4, complete=True)], []), request, lambda _: None)
    first = _message(request, 0, "first", 0)
    collector.observe(first)
    with pytest.raises(SessionProtocolError, match="duplicate"):
        collector.observe(first)
    with pytest.raises(SessionProtocolError, match="lacks token boundaries"):
        collector.finish(raw, 5.0)
    clock.value = 4.0
    collector.observe(_message(request, 0, "last", 99))
    with pytest.raises(SessionProtocolError, match="disagree"):
        collector.finish(raw, 5.0)


def test_outer_session_retains_each_prompt_boundary_in_mixed_workload_windows():
    clock = _Clock()

    class StreamingTransport(_FakeTransport):
        async def aread_response(self, requests, *, deadline, on_progress=None):
            request = self.requests[-1]

            async def emit(frame):
                self.clock.advance(0.5)
                on_progress(parse_frame_bytes(frame, max_bytes=MAX_CONTROL_BYTES))

            outputs = await generate_outputs(
                _engine(
                    [
                        _chunk(1, 1),
                        _chunk(0, 1),
                        _chunk(0, request.max_new_tokens, complete=True),
                        _chunk(1, request.max_new_tokens, complete=True),
                    ],
                    [],
                ),
                request,
                emit,
            )
            return request, engine_outputs(outputs, request=request)

    plan = _plan(
        measure_phase_latency=True,
        expected_prompt_tokens=5,
        batch_max_new_tokens=(4, 4, 8),
        batch_expected_prompt_tokens=(5, 5, 5),
        top_logprobs_num=0,
    )
    transport = StreamingTransport(clock, _facts())
    result = run_outer_session(
        plan,
        transport=transport,
        deadline=1000.0,
        init_timeout_s=30.0,
        batch_timeout_s=30.0,
        clock=clock,
    )
    assert transport.finalized and len(result.batches) == 3
    assert all(
        row.prompt_latencies == ((1.5, 2.0), (1.0, 2.5)) for row in result.batches
    )
    codec = ContinuationCodec((BatchExecutionEvidence,))
    assert codec.decode(codec.encode(result.batches[-1])) == result.batches[-1]


def test_old_request_bytes_stay_unchanged_and_phase_identity_differs(tmp_path):
    legacy = replace(_request(), measure_phase_latency=False)
    assert "measure_phase_latency" not in legacy.to_dict()
    assert validate_batch_request(legacy.to_dict()) == legacy
    replay = replay_plan(tmp_path)
    assert marginal_workload_digest(replay_session(_plan(), replay)) != marginal_workload_digest(
        replay_session(_plan(measure_phase_latency=True), replay)
    )


def test_profile_enables_measurement_through_the_existing_commission_parser():
    from cacheon.eval.b300_sealed_qualification_commission import (
        sealed_qualification_commission,
    )
    from cacheon.eval.b300_registered_qualification_inputs import (
        B300RegisteredQualificationError,
    )
    from tests.test_b300_sealed_qualification_commission import _block

    block = _block()
    block["session"]["measure_phase_latency"] = True
    assert sealed_qualification_commission(block) is block
    block["session"]["measure_phase_latency"] = "true"
    with pytest.raises(B300RegisteredQualificationError, match="session block"):
        sealed_qualification_commission(block)
