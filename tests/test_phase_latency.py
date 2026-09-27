"""Phase delivery measurements survive interleaving, framing and retained regrade."""

import copy
import asyncio
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.support.pipes import engine_loop as engine_loop

pytestmark = pytest.mark.usefixtures("engine_loop")

from cacheon.eval.continuation_codec import ContinuationCodec, ContinuationCodecError
from cacheon.eval.crossover_runtime import TimedWindow
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
from cacheon.eval.resident_measurement import (
    CrossoverRuntimeError,
    _timed_windows,
    phase_cells,
)
from cacheon.eval.scoring import marginal_workload_digest
from tests.support.pipes import PipeClient, PipeManager
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
    windows = _timed_windows(result.batches)
    cells = phase_cells(windows)
    assert [(cell["output_tokens"], cell["timed_batches"]) for cell in cells] == [
        (4, 2),
        (8, 1),
    ]
    assert [float(cell["mean_ttft_seconds"]) for cell in cells] == [1.25, 1.25]
    assert float(cells[0]["mean_tpot_seconds"]) == pytest.approx(1 / 3)
    assert float(cells[1]["mean_tpot_seconds"]) == pytest.approx(1 / 7)
    assert float(cells[0]["end_to_end_output_tokens_per_second"]) == pytest.approx(3.2)
    assert TimedWindow.from_dict(windows[0].to_dict()) == windows[0]
    codec = ContinuationCodec((BatchExecutionEvidence,))
    assert codec.decode(codec.encode(result.batches[-1])) == result.batches[-1]


def test_old_request_and_continuation_bytes_stay_unchanged_and_phase_identity_differs():
    legacy = replace(_request(), measure_phase_latency=False)
    assert "measure_phase_latency" not in legacy.to_dict()
    assert validate_batch_request(legacy.to_dict()) == legacy
    codec = ContinuationCodec((TimedWindow,))
    old_payload = {
        "type": "cacheon.eval.crossover_runtime.TimedWindow",
        "value": {"batch_index": 2, "tokens": 8, "seconds": "2.5"},
    }
    assert codec.encode(codec.decode(old_payload)) == old_payload
    new = TimedWindow(2, 8, 2.5, 5, ((1.5, 2.0), (1.0, 2.5)))
    assert codec.decode(codec.encode(new)) == new
    bad = copy.deepcopy(old_payload)
    bad["value"]["prompt_latencies"] = []
    with pytest.raises(ContinuationCodecError, match="default must be omitted"):
        codec.decode(bad)
    assert marginal_workload_digest(_plan()) != marginal_workload_digest(
        _plan(measure_phase_latency=True)
    )


def test_report_does_not_blend_workloads_or_let_a_small_ttft_gain_invent_end_to_end_gain():
    baseline = (
        TimedWindow(0, 1000, 100.0, 8192, ((2.0, 99.0),)),
        TimedWindow(1, 4000, 400.0, 65536, ((80.0, 399.0),)),
    )
    candidate = (
        TimedWindow(0, 1000, 99.0 + 1 / 3, 8192, ((4 / 3, 98 + 1 / 3),)),
        baseline[1],
    )
    b, c = phase_cells(baseline), phase_cells(candidate)
    assert float(b[0]["mean_ttft_seconds"]) / float(
        c[0]["mean_ttft_seconds"]
    ) == pytest.approx(1.5)
    assert b[0]["mean_tpot_seconds"] == c[0]["mean_tpot_seconds"]
    assert (
        float(c[0]["end_to_end_output_tokens_per_second"])
        / float(b[0]["end_to_end_output_tokens_per_second"])
        < 1.01
    )
    assert b[1] == c[1]
    with pytest.raises(CrossoverRuntimeError, match="lacks a timed window"):
        phase_cells((baseline[0], TimedWindow(1, 4000, 400.0)))


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


def test_production_crossover_report_recomputes_cells_from_bound_raw_sessions(
    tmp_path, monkeypatch
):
    from cacheon.eval.oci_session_protocol import BatchEvidence, PromptEvidence
    from cacheon.eval.qualification_runner import ResidentSpeedWitness
    from tests import test_crossover_runtime as fixtures

    original = fixtures._Controller.execute_next

    def execute(controller):
        row = original(controller)
        count = controller.plan.max_new_tokens
        prompts = tuple(
            PromptEvidence(tuple(range(count)), ((),) * count, 5)
            for _ in controller.plan.prompt_batches[row.batch_index]
        )
        row = replace(
            row,
            evidence=BatchEvidence(prompts),
            prompt_latencies=((row.elapsed_seconds / 4, row.elapsed_seconds * 0.75),)
            * len(prompts),
        )
        controller.rows[-1] = row
        return row

    monkeypatch.setattr(fixtures._Controller, "execute_next", execute)
    plan, baseline, candidate, mount, _, _ = fixtures._rig(
        tmp_path,
        (0.9,),
        policy=fixtures._resident_policy(version=11),
    )
    plan = replace(
        plan,
        baseline=replace(
            plan.baseline,
            session_plan=replace(
                plan.baseline.session_plan, measure_phase_latency=True, max_new_tokens=4
            ),
        ),
        candidate=replace(
            plan.candidate,
            session_plan=replace(
                plan.candidate.session_plan,
                measure_phase_latency=True,
                max_new_tokens=4,
            ),
        ),
    )
    result = fixtures._speed(plan, baseline, candidate, mount)
    assert result.regrade(plan) == result.final_verdict
    witness = ResidentSpeedWitness.from_evidence(result, plan)
    raw = witness.to_dict()
    assert all(rate["windows"][0]["input_tokens"] == 5 for rate in raw["rates"])
    assert ResidentSpeedWitness.from_dict(raw) == witness
    forged = copy.deepcopy(raw)
    forged["rates"][0]["cells"] = [{"mean_ttft_seconds": "0.000001"}]
    with pytest.raises(CrossoverRuntimeError, match="fields differ"):
        ResidentSpeedWitness.from_dict(forged)
    rate = result.rates[0]
    altered = replace(rate.windows[0], prompt_latencies=((0.01, 0.5),))
    tampered = replace(
        result,
        rates=(replace(rate, windows=(altered, *rate.windows[1:])), *result.rates[1:]),
    )
    with pytest.raises(CrossoverRuntimeError, match="independently regrade"):
        tampered.regrade(plan)
