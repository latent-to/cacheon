"""The production worker overlaps requests without mixing their wire evidence."""

import asyncio
import os
import struct
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cacheon.eval.oci_request_loop import RequestFailure, serve_requests
from cacheon.eval.oci_session_protocol import (
    EVIDENCE_MAGIC,
    MAX_BATCH_REQUEST_BYTES,
    BatchRequest,
    SessionProtocolError,
    frame_message,
    parse_evidence_frame_bytes,
)


def _request(index, tokens):
    return BatchRequest(
        "a" * 32, "b" * 64, f"{index + 1:032x}", f"{index + 10:032x}",
        index, (str(index),), tokens, 0, 0.0,
    )


async def _read_frame(reader):
    header = await reader.readexactly(8)
    assert header[:4] == EVIDENCE_MAGIC
    return header + await reader.readexactly(struct.unpack(">I", header[4:])[0])


async def _exchange(engine, requests, check):
    control_read, control_write = os.pipe()
    output_read, output_write = os.pipe()
    reader = asyncio.StreamReader()
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader),
        os.fdopen(output_read, "rb", buffering=0),
    )
    task = asyncio.create_task(serve_requests(
        SimpleNamespace(engine=engine, require_completion=lambda: None),
        control_read, output_write,
        session_id=requests[0].session_id, launch_digest=requests[0].launch_digest,
        audit_policy=None,
    ))
    try:
        for request in requests:
            os.write(control_write, frame_message(request.to_dict(), max_bytes=MAX_BATCH_REQUEST_BYTES))
        await asyncio.wait_for(check(task, reader), 2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        transport.close()
        for fd in (control_read, control_write, output_write):
            os.close(fd)


@pytest.mark.parametrize("budgets", [(2, 3), (16384, 20000)])
def test_later_request_finishes_first_and_large_frames_stay_intact(budgets):
    async def run():
        first_started, second_finished = asyncio.Event(), asyncio.Event()
        completions = []

        async def generate(**kwargs):
            index = int(kwargs["prompt"][0])
            if index == 0:
                first_started.set()
                await second_finished.wait()
            else:
                await first_started.wait()
                second_finished.set()
            completions.append(index)
            return {"output_ids": [index + 11] * budgets[index], "meta_info": {"prompt_tokens": 5}}

        requests = [_request(index, tokens) for index, tokens in enumerate(budgets)]

        async def check(_task, reader):
            for index in (1, 0):
                evidence = parse_evidence_frame_bytes(await _read_frame(reader), request=requests[index])
                assert evidence.prompts[0].output_ids == (index + 11,) * budgets[index]
            assert completions == [1, 0]

        await _exchange(SimpleNamespace(async_generate=generate), requests, check)

    asyncio.run(run())


def test_engine_failure_interrupts_open_input_and_cancels_other_requests():
    async def run():
        waiting, cancelled = asyncio.Event(), asyncio.Event()
        original = RuntimeError("original engine failure")

        async def generate(**kwargs):
            if kwargs["prompt"] == ["0"]:
                waiting.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            await waiting.wait()
            raise original

        requests = [_request(0, 2), _request(1, 3)]

        async def check(task, _reader):
            with pytest.raises(RequestFailure) as caught:
                await task
            assert caught.value.cause is original
            assert caught.value.request == requests[1]
            assert cancelled.is_set()

        await _exchange(SimpleNamespace(async_generate=generate), requests, check)

    asyncio.run(run())


def test_duplicate_request_cannot_execute_while_the_original_is_in_flight():
    async def run():
        calls = []

        async def generate(**kwargs):
            calls.append(kwargs)
            await asyncio.Event().wait()

        first = _request(0, 2)

        async def check(task, _reader):
            with pytest.raises(RequestFailure) as caught:
                await task
            assert isinstance(caught.value.cause, SessionProtocolError)
            assert "replay binding" in str(caught.value.cause)
            assert len(calls) == 1

        await _exchange(SimpleNamespace(async_generate=generate), [first, replace(first, batch_index=1)], check)

    asyncio.run(run())


def test_failure_waits_for_started_frame_before_reporting_its_original_cause():
    async def run():
        control_read, control_write = os.pipe()
        output_read, output_write = os.pipe()
        fail, raised = asyncio.Event(), asyncio.Event()
        original = RuntimeError("second request failed during first response")
        requests = [_request(0, 20000), _request(1, 2)]

        async def generate(**kwargs):
            if kwargs["prompt"] == ["1"]:
                await fail.wait()
                raised.set()
                raise original
            return {"output_ids": [11] * 20000, "meta_info": {"prompt_tokens": 5}}

        task = asyncio.create_task(serve_requests(
            SimpleNamespace(engine=SimpleNamespace(async_generate=generate), require_completion=lambda: None),
            control_read, output_write,
            session_id=requests[0].session_id, launch_digest=requests[0].launch_digest,
            audit_policy=None,
        ))
        transport = None
        try:
            for request in requests:
                os.write(control_write, frame_message(request.to_dict(), max_bytes=MAX_BATCH_REQUEST_BYTES))
            # No reader drains the large response until the other engine call
            # has raised. Reading only the header leaves the writer backpressured.
            header = await asyncio.to_thread(os.read, output_read, 8)
            assert header[:4] == EVIDENCE_MAGIC
            fail.set()
            await raised.wait()
            reader = asyncio.StreamReader()
            transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader),
                os.fdopen(output_read, "rb", buffering=0),
            )
            with pytest.raises(RequestFailure) as caught:
                await task
            assert caught.value.cause is original
            assert caught.value.request == requests[1]
            payload = await reader.readexactly(struct.unpack(">I", header[4:])[0])
            evidence = parse_evidence_frame_bytes(header + payload, request=requests[0])
            assert evidence.prompts[0].output_ids == (11,) * 20000
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if transport is not None:
                transport.close()
            else:
                os.close(output_read)
            for fd in (control_read, control_write, output_write):
                os.close(fd)

    asyncio.run(asyncio.wait_for(run(), 2))


@pytest.mark.parametrize('accepted', [True, False])
def test_cache_boundary_requires_engine_acceptance(accepted):
    from cacheon.eval.oci_request_input import cache_flush_message
    from cacheon.eval.oci_session_protocol import MAX_CONTROL_BYTES, parse_frame_bytes

    async def run():
        control_read, control_write = os.pipe()
        output_read, output_write = os.pipe()
        reader = asyncio.StreamReader()
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(output_read, 'rb', buffering=0),
        )
        calls = []
        async def flush():
            calls.append('flush')
            return SimpleNamespace(success=accepted)
        engine = SimpleNamespace(tokenizer_manager=SimpleNamespace(flush_cache=flush))
        task = asyncio.create_task(serve_requests(
            SimpleNamespace(engine=engine), control_read, output_write,
            session_id='a' * 32, launch_digest='b' * 64, audit_policy=None,
        ))
        request = cache_flush_message(session_id='a' * 32, launch_digest='b' * 64,
                                      request_id='c' * 32, nonce='d' * 32, batch_index=0)
        try:
            os.write(control_write, frame_message(request, max_bytes=MAX_CONTROL_BYTES))
            if accepted:
                header = await asyncio.wait_for(reader.readexactly(8), 2)
                payload = await reader.readexactly(struct.unpack('>I', header[4:])[0])
                reply = parse_frame_bytes(header + payload, max_bytes=MAX_CONTROL_BYTES)
                assert reply == {**request, 'type': 'cache_flushed'}
            else:
                with pytest.raises(SessionProtocolError, match='refused cache flush'):
                    await asyncio.wait_for(task, 2)
            assert calls == ['flush']
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            transport.close()
            for fd in (control_read, control_write, output_write):
                os.close(fd)
    asyncio.run(run())
