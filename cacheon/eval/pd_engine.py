"""Prefill/decode engine pair for speed policy 20, inside one OCI worker fence.

SGLang's disaggregation role is process-global, so the decode engine stays the in-process
``sgl.Engine`` the session loop already drives, and each prefill server is a companion child of the
same worker serving HTTP on loopback. The lane's GPUs run the prefill servers first, each on its own
``tp_size`` GPUs (``pd_prefill.servers`` of them, one by default), then the decode engine. One request
is one bootstrap room: the facade posts it to the session's prefill server and hands it to the decode
engine with the same room and DP rank, as SGLang's PD router does with ``--dp-aware``; a session keeps
one server, whose radix cache holds its context. Every role inherits the session environment, so a
bundle loads in each and each reports coverage. Untimed roles (the eager audit, the pristine
reference) run one colocated engine with the prefill role's settings, which size it for the long
prefills it serves, plus decode's ReplaySSM speculation: the audit on the first prefill GPUs and the reference on the decode role's, so
they run at once.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Any

ROLES = ("pd_prefill", "pd_decode")
HOST = "127.0.0.1"


def is_pair(kwargs: dict[str, Any]) -> bool:
    return ROLES[0] in kwargs


def _role(kwargs: dict[str, Any], env: dict[str, str], role: str) -> tuple[dict[str, Any], dict[str, str]]:
    shared = {key: value for key, value in kwargs.items()
              if key not in ROLES and key != "disaggregation_transfer_backend"}
    override = kwargs[role]
    return {**shared, **override.get("engine_kwargs", {})}, {**env, **override.get("engine_env", {})}


def colocated(kwargs: dict[str, Any], env: dict[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
    """The one-engine form of a pair for untimed roles; a non-pair config passes through.

    It takes the decode role's ReplaySSM speculation, which SGLang refuses on a prefill server, so the audit grades
    the recurrent writes the served decode makes (a TP2 GLM-5.3-Flash engine booted and served this form, 2026-11-11).
    """
    if not is_pair(kwargs):
        return kwargs, env
    form, form_env = _role(kwargs, env, ROLES[0])
    if _role(kwargs, env, ROLES[1])[0].get("enable_linear_replayssm_spec"):
        form["enable_linear_replayssm_spec"] = True
    return form, form_env


def layout(tp_size: int, kwargs: dict[str, Any]) -> tuple[int, int, int]:
    """(prefill servers, prefill TP, decode TP); one engine of ``tp_size`` ranks is (0, 0, tp_size)."""
    if not is_pair(kwargs):
        return 0, 0, tp_size
    tp = {"tp_size": tp_size}
    return (int(kwargs[ROLES[0]].get("servers", 1)), int(_role({**tp, **kwargs}, {}, ROLES[0])[0]["tp_size"]),
            int(_role({**tp, **kwargs}, {}, ROLES[1])[0]["tp_size"]))


def untimed_gpus(tp_size: int, kwargs: dict[str, Any]) -> tuple[slice, slice]:
    """Lane GPUs of a pair's one-engine form: the eager audit's (the first prefill server's), pristine T's (as many
    from the first decode GPU), so they run at once."""
    servers, prefill_tp, _ = layout(tp_size, kwargs)
    return slice(0, prefill_tp), slice(servers * prefill_tp, (servers + 1) * prefill_tp)


def lane_width(tp_size: int, kwargs: dict[str, Any]) -> int:
    servers, prefill_tp, decode_tp = layout(tp_size, kwargs)
    return servers * prefill_tp + decode_tp


def routes(kwargs: dict[str, Any]) -> int:
    """Sticky session routes: every prefill server's DP ranks, or one engine's DP ranks."""
    if not is_pair(kwargs):
        return int(kwargs.get("dp_size") or 1)
    return int(kwargs[ROLES[0]].get("servers", 1)) * int(_role(kwargs, {}, ROLES[0])[0].get("dp_size") or 1)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def split(kwargs: dict[str, Any], env: dict[str, str]):
    """Prefill server i on GPUs [i p, (i+1) p), decode on the rest; ports are picked on loopback, never sealed."""
    backend = kwargs["disaggregation_transfer_backend"]
    servers, prefill_tp, decode_tp = layout(int(kwargs["tp_size"]), kwargs)
    world, first = servers * prefill_tp + decode_tp, servers * prefill_tp
    decode, decode_env = _role(kwargs, env, ROLES[1])
    prefills = []
    for index in range(servers):
        prefill, prefill_env = _role(kwargs, env, ROLES[0])
        # With DP attention a server derives its ZMQ ports from its HTTP port (port + 234 onward): adjacent ports
        # collided and killed the prefill server at boot (pod, 2026-10-27).
        prefill.update(disaggregation_mode="prefill", disaggregation_transfer_backend=backend,
                       base_gpu_id=index * prefill_tp, disaggregation_bootstrap_port=_free_port(), host=HOST,
                       port=int(decode.get("port", 30000)) + 1000 * (index + 1))
        # One coverage roster over every process group, numbered in lane GPU order (cacheon.receipts). The fence has
        # only loopback, where SGLang's KV managers find no routable address and raise "Can not get local ip".
        prefill_env.update(CACHEON_RECEIPT_RANK_OFFSET=str(index * prefill_tp), CACHEON_RECEIPT_WORLD_SIZE=str(world),
                           SGLANG_HOST_IP=HOST)
        prefills.append((prefill, prefill_env))
    decode.update(disaggregation_mode="decode", disaggregation_transfer_backend=backend, base_gpu_id=first)
    decode_env.update(CACHEON_RECEIPT_RANK_OFFSET=str(first), CACHEON_RECEIPT_WORLD_SIZE=str(world), SGLANG_HOST_IP=HOST)
    return prefills, decode, decode_env


def start_prefill(kwargs: dict[str, Any], env: dict[str, str]) -> subprocess.Popen:
    """Boot the prefill server as a child of this worker; it boots while the decode engine does."""
    process = subprocess.Popen([sys.executable, "-m", "cacheon.eval.pd_engine"], stdin=subprocess.PIPE,
                               env={**os.environ, **env})
    process.stdin.write(json.dumps({k: v for k, v in kwargs.items() if k != "custom_sigquit_handler"}).encode())
    process.stdin.close()
    # Staggered so the two servers' self-chosen internal ports do not race (10-23 PD launch).
    time.sleep(10)
    return process


def wait_ready(process: subprocess.Popen, kwargs: dict[str, Any], deadline_s: float) -> str:
    url = f"http://{HOST}:{kwargs['port']}"
    while time.monotonic() < deadline_s:
        if process.poll() is not None:
            raise RuntimeError(f"prefill server exited {process.returncode} during boot")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return url
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError("prefill server did not become healthy before the boot deadline")


async def _unless_failed(prefill: asyncio.Future, step):
    """Await a decode step, but raise the prefill failure at once: without its KV, decode waits out
    SGLang's bootstrap timeout."""
    step = asyncio.ensure_future(step)
    await asyncio.wait({step, prefill}, return_when=asyncio.FIRST_COMPLETED)
    if prefill.done() and prefill.exception() is not None:
        step.cancel()
        raise prefill.exception()
    return await step


class _PairTokenizerManager:
    """The decode engine's tokenizer manager, whose cache flush also empties the prefill server's."""

    def __init__(self, pair: "PrefillDecodePair"):
        self._pair = pair

    def __getattr__(self, name):
        return getattr(self._pair.decode.tokenizer_manager, name)

    async def flush_cache(self):
        result = await self._pair.decode.tokenizer_manager.flush_cache()
        await asyncio.gather(*(self._pair.post(url, "/flush_cache?timeout=120", None) for url, _ in self._pair.servers))
        return result


class PrefillDecodePair:
    """The decode engine with every generation routed through its session's prefill server first."""

    def __init__(self, decode: Any, servers: list[tuple[str, int]], processes: list[subprocess.Popen],
                 prefill_ranks: int = 1, decode_ranks: int = 1):
        self.decode, self.servers, self.processes = decode, servers, processes
        self.prefill_ranks, self.decode_ranks = prefill_ranks, decode_ranks
        self.tokenizer_manager = _PairTokenizerManager(self)
        self._client = None

    def __getattr__(self, name):
        return getattr(self.decode, name)

    async def post(self, url: str, path: str, body: dict | None) -> None:
        import aiohttp
        import orjson

        if self._client is None:
            # SGLang's server drops a connection idle 5 s; aiohttp pooled it 15 s and sent the POST into the close
            # ("Server disconnected", first PD twin 2026-10-27). Its default 100-connection cap also queued prefills.
            self._client = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, force_close=True),
                                                 timeout=aiohttp.ClientTimeout(total=None))
        # aiohttp's stdlib encode of a 135k-token prompt held ~16% of the session worker's loop (2026-10-27).
        data = None if body is None else orjson.dumps(body)
        async with self._client.post(url + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else None) as response:
            if response.status != 200:
                raise RuntimeError(f"prefill server {path} returned {response.status}: {(await response.text())[:512]}")
            await response.read()

    async def async_generate(self, **kwargs):
        key = "input_ids" if "input_ids" in kwargs else "prompt"
        count = len(kwargs[key])
        rank = kwargs.pop("routed_dp_rank", None)
        rooms = [secrets.randbits(62) for _ in range(count)]
        url, port = self.servers[0 if rank is None else rank % len(self.servers)]
        if rank is not None:
            # The prefill server places a room on DP rank room % dp_size and refuses a routed rank that disagrees
            # ("follow_bootstrap_room conflict", 8xB300 2026-10-27), so each room carries the session's prefill rank.
            prefill_rank, decode_rank = rank // len(self.servers) % self.prefill_ranks, rank % self.decode_ranks
            rooms = [room - room % self.prefill_ranks + prefill_rank for room in rooms]
        route = {"bootstrap_host": [HOST] * count, "bootstrap_port": [port] * count, "bootstrap_room": rooms}
        prefill = asyncio.ensure_future(self.post(url, "/generate", {
            "input_ids" if key == "input_ids" else "text": kwargs[key],
            "sampling_params": kwargs["sampling_params"], **route,
            **({"routed_dp_rank": prefill_rank} if rank is not None else {}),
        }))
        result = await _unless_failed(prefill, self.decode.async_generate(
            **kwargs, **route, **({"routed_dp_rank": decode_rank, "disagg_prefill_dp_rank": prefill_rank}
                                  if rank is not None else {})))
        if not kwargs.get("stream"):
            await prefill
            return result
        return self._stream(result, prefill)

    @staticmethod
    async def _stream(stream, prefill):
        items = stream.__aiter__()
        while True:
            try:
                item = await _unless_failed(prefill, items.__anext__())
            except StopAsyncIteration:
                break
            yield item
        await prefill

    def shutdown(self) -> None:
        try:
            self.decode.shutdown()
        finally:
            if self._client is not None:
                self.decode.loop.run_until_complete(self._client.close())
            for process in self.processes:
                process.terminate()
            for process in self.processes:
                try:
                    process.wait(30)
                except subprocess.TimeoutExpired:
                    process.kill()


def _serve_prefill() -> None:
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import ServerArgs

    load_plugins()
    run_server(ServerArgs(**json.loads(sys.stdin.read())))


if __name__ == "__main__":
    _serve_prefill()
