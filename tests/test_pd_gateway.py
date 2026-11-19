"""The routing endpoint receives generation work and the validator keeps its token boundaries."""

import json
from pathlib import Path

import pytest

from cacheon.eval.pd_engine import _GatewayEngine


def test_sealed_route_uses_the_commissioned_image_adapter(tmp_path):
    from cacheon.engine_tree import MaterializedEngineTree
    from cacheon.eval.oci_backend import CONTAINER_ENGINE_WORKER_POLICY, OCIBackendError
    from cacheon.eval.oci_engine_options import ENGINE_KWARG_KINDS, _validate_options
    from cacheon.eval.oci_worker_policy import worker_policy_files

    role = {"routing_policy": "power_of_two"}
    pair = {"pd_prefill": role, "pd_decode": {"engine_kwargs": {"tp_size": 2}},
            "disaggregation_transfer_backend": "mooncake"}
    config = _validate_options(pair, ENGINE_KWARG_KINDS, label="engine", error=ValueError)
    assert config["pd_prefill"] == role
    with pytest.raises(ValueError, match="prefill routing policy"):
        _validate_options({"pd_decode": role}, ENGINE_KWARG_KINDS, label="engine", error=ValueError)
    tree = MaterializedEngineTree(tmp_path, "", "", (), None)
    args = tree, tmp_path, CONTAINER_ENGINE_WORKER_POLICY, OCIBackendError
    plain, routed = worker_policy_files(*args), worker_policy_files(*args, native_router=True)
    modules = {Path(m.split(",dst=")[1].split(",")[0]).name for m in set(routed[0]) - set(plain[0])}
    assert modules == {"pd_engine.py", "oci_engine_options.py"}


def test_router_owns_placement_and_every_worker_flushes():
    web = pytest.importorskip("aiohttp.web")
    pytest.importorskip("orjson")
    client = _GatewayEngine("", [])
    received, flushed = [], []

    async def generate(request):
        body = await request.json()
        received.append(body)
        if not body["stream"]:
            return web.json_response({"text": body["text"]})
        response = web.StreamResponse()
        await response.prepare(request)
        for token in body["input_ids"]:
            event = json.dumps({"output_ids": [token]}).encode()
            await response.write(b"data: " + event[:6])
            await response.write(event[6:] + b"\n\n")
        if body["input_ids"] != [0]:
            await response.write(b"data: [DONE]\n\n")
        return response

    async def flush(request):
        flushed.append(request.match_info["worker"])
        return web.Response()

    async def exercise():
        app = web.Application()
        app.router.add_post("/generate", generate)
        app.router.add_post("/{worker}/flush_cache", flush)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            client.url = f"http://127.0.0.1:{runner.addresses[0][1]}"
            client.workers = [client.url + "/" + name for name in ("p0", "p1", "d0")]
            settings = {"sampling_params": {"max_new_tokens": 2}, "routed_dp_rank": 5}
            assert await client.async_generate(prompt=["x", "y"], **settings) == [{"text": "x"}, {"text": "y"}]
            rows = [r async for r in await client.async_generate(input_ids=[[1, 2], [3]],
                                                                  stream=True, **settings)]
            assert sorted((r["index"], r["output_ids"]) for r in rows) == [(0, [1]), (0, [2]), (1, [3])]
            with pytest.raises(RuntimeError, match="before .*DONE"):
                async for _ in await client.async_generate(input_ids=[[0]], stream=True, **settings):
                    pass
            assert (await client.flush_cache()).success and sorted(flushed) == ["d0", "p0", "p1"]
            assert all("routed_dp_rank" not in r and r["sampling_params"] == settings["sampling_params"]
                       for r in received)
        finally:
            await runner.cleanup()

    try:
        client.loop.run_until_complete(exercise())
    finally:
        client.shutdown()
