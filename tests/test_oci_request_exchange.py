"""Exercise the shared host exchange against real pipes and the production worker."""

import asyncio
import os
import time
from types import SimpleNamespace

import pytest

from cacheon.eval.oci_outer_session import (
    AttachedSessionTransport,
    OuterSessionInfrastructureError,
    OuterSessionProcessError,
    OuterSessionTimeoutError,
)
from cacheon.eval.oci_request_exchange import RequestExchange
from cacheon.eval.oci_request_loop import serve_requests
from cacheon.eval.oci_session_protocol import BatchRequest
from tests.support.pipes import PipeClient, PipeManager


def _request(index, tokens):
    return BatchRequest(
        "a" * 32, "b" * 64, f"{index + 1:032x}", f"{index + 10:032x}",
        index, (str(index),), tokens, 0, 0.0, 5, True,
    )


@pytest.mark.parametrize("budgets", [(2, 3), (16384, 20000)])
def test_exchange_routes_reordered_outputs_and_progress_to_their_host_clocks(budgets):
    async def run():
        client = PipeClient()
        transport = AttachedSessionTransport(PipeManager(client), object(), ("worker",))
        transport.start()
        first_started, second_finished = asyncio.Event(), asyncio.Event()
        completed, progress = [], []

        async def generate(**kwargs):
            index = int(kwargs["prompt"][0])

            async def stream():
                if index == 0:
                    first_started.set()
                    await second_finished.wait()
                else:
                    await first_started.wait()
                for count in (1, budgets[index]):
                    yield {"index": 0, "output_ids": [index + 11] * count, "meta_info": {
                        "prompt_tokens": 5, "completion_tokens": count,
                        "finish_reason": {"type": "length"} if count > 1 else None,
                    }}
                completed.append(index)
                if index == 1:
                    second_finished.set()

            return stream()

        worker = asyncio.create_task(serve_requests(
            SimpleNamespace(engine=SimpleNamespace(async_generate=generate), require_completion=lambda: None),
            client.request_read, client.response_write,
            session_id="a" * 32, launch_digest="b" * 64, audit_policy=None,
        ))
        deadline = time.monotonic() + 3
        try:
            async with RequestExchange(transport, clock=time.monotonic, deadline=deadline) as exchange:
                rows = await asyncio.gather(*(
                    exchange.execute(_request(i, count), deadline=deadline,
                                     on_progress=lambda event: progress.append(event))
                    for i, count in enumerate(budgets)
                ))
            assert completed == [1, 0]
            assert rows[1].response_completed_at < rows[0].response_completed_at
            for index, row in enumerate(rows):
                assert row.request_id == _request(index, budgets[index]).request_id
                assert row.evidence.prompts[0].output_ids == (index + 11,) * budgets[index]
                first, last = row.prompt_latencies[0]
                assert 0 < first < last <= row.elapsed_seconds
                assert [(e["boundary"], e["token_id"]) for e in progress if e["request_id"] == row.request_id] == [
                    ("first", index + 11), ("last", index + 11),
                ]
            assert not transport.has_pending_output()
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            client.close()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["eof", "timeout"])
def test_exchange_propagates_transport_failure_to_every_pending_request(failure):
    async def run():
        client = PipeClient()
        transport = AttachedSessionTransport(PipeManager(client), object(), ("worker",))
        transport.start()
        if failure == "eof":
            os.close(client.response_write)
            client.response_write = -1
        deadline = time.monotonic() + 0.05
        try:
            async with RequestExchange(transport, clock=time.monotonic, deadline=deadline) as exchange:
                results = await asyncio.gather(*(
                    exchange.execute(_request(i, 2), deadline=deadline) for i in range(2)
                ), return_exceptions=True)
            expected = OuterSessionProcessError if failure == "eof" else OuterSessionTimeoutError
            assert all(isinstance(result, expected) for result in results)
            assert not exchange.requests and not exchange.pending
        finally:
            client.close()

    asyncio.run(run())


def test_exchange_writes_frames_in_batch_index_order_whatever_order_callers_arrive():
    """The production worker refuses an unexpected index, so it is the oracle here."""
    async def run():
        client = PipeClient()
        transport = AttachedSessionTransport(PipeManager(client), object(), ("worker",))
        transport.start()
        served = []

        async def generate(**kwargs):
            served.append(int(kwargs["prompt"][0]))

            async def stream():
                yield {"index": 0, "output_ids": [7], "meta_info": {
                    "prompt_tokens": 5, "completion_tokens": 1, "finish_reason": {"type": "length"},
                }}

            return stream()

        worker = asyncio.create_task(serve_requests(
            SimpleNamespace(engine=SimpleNamespace(async_generate=generate), require_completion=lambda: None),
            client.request_read, client.response_write,
            session_id="a" * 32, launch_digest="b" * 64, audit_policy=None,
        ))
        deadline = time.monotonic() + 3
        try:
            async with RequestExchange(transport, clock=time.monotonic, deadline=deadline) as exchange:
                rows = await asyncio.gather(*(
                    exchange.execute(_request(i, 1), deadline=deadline) for i in (2, 1, 0)
                ))
                assert served == [0, 1, 2]
                assert [row.batch_index for row in rows] == [2, 1, 0]
                assert exchange.next_index == 3 and not exchange.turns
                with pytest.raises(OuterSessionInfrastructureError, match="already written"):
                    await exchange.execute(_request(1, 1), deadline=deadline)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            client.close()

    asyncio.run(run())
