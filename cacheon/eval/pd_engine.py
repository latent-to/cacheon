"""Prefill/decode engine pair for speed policy 20, inside one OCI worker fence.

SGLang's disaggregation role is process-global, so the decode engine stays the in-process
``sgl.Engine`` the session loop already drives, and the prefill server is a companion child of the
same worker serving HTTP on loopback. One request is one bootstrap room: the facade posts it to the
prefill server and hands it to the decode engine with the same room and DP rank, as SGLang's PD
router does with ``--dp-aware``. Both roles inherit the session environment, so a bundle loads in
both and both report coverage. Untimed roles (the eager audit, the pristine reference) run one
colocated engine with the prefill role's settings, which size it for the long prefills it serves.
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
    """The one-engine form of a pair for untimed roles; a non-pair config passes through."""
    return _role(kwargs, env, ROLES[0]) if is_pair(kwargs) else (kwargs, env)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def split(kwargs: dict[str, Any], env: dict[str, str]):
    """Prefill on GPUs [0, tp), decode on [tp, 2 tp); ports are picked on loopback, never sealed."""
    backend, tp = kwargs["disaggregation_transfer_backend"], int(kwargs["tp_size"])
    prefill, prefill_env = _role(kwargs, env, ROLES[0])
    decode, decode_env = _role(kwargs, env, ROLES[1])
    # With DP attention a server derives its ZMQ ports from its HTTP port (port + 234 onward): adjacent ports
    # collided and killed the prefill server at boot (pod, 2026-10-27).
    prefill.update(disaggregation_mode="prefill", disaggregation_transfer_backend=backend, base_gpu_id=0,
                   disaggregation_bootstrap_port=_free_port(), host=HOST, port=int(decode.get("port", 30000)) + 1000)
    decode.update(disaggregation_mode="decode", disaggregation_transfer_backend=backend, base_gpu_id=tp)
    # One coverage roster over both process groups: prefill ranks 0..tp-1, decode ranks tp..2tp-1 (cacheon.receipts).
    # The fence has only loopback, where SGLang's KV managers find no routable address and raise "Can not get local ip".
    prefill_env.update(CACHEON_RECEIPT_RANK_OFFSET="0", CACHEON_RECEIPT_WORLD_SIZE=str(2 * tp), SGLANG_HOST_IP=HOST)
    decode_env.update(CACHEON_RECEIPT_RANK_OFFSET=str(tp), CACHEON_RECEIPT_WORLD_SIZE=str(2 * tp), SGLANG_HOST_IP=HOST)
    return prefill, prefill_env, decode, decode_env


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
        await self._pair.post("/flush_cache?timeout=120", None)
        return result


class PrefillDecodePair:
    """The decode engine with every generation routed through the prefill server first."""

    def __init__(self, decode: Any, prefill_url: str, bootstrap_port: int, process: subprocess.Popen,
                 prefill_ranks: int = 1, decode_ranks: int = 1):
        self.decode, self.prefill_url, self.bootstrap_port, self.process = decode, prefill_url, bootstrap_port, process
        self.prefill_ranks, self.decode_ranks = prefill_ranks, decode_ranks
        self.tokenizer_manager = _PairTokenizerManager(self)
        self._client = None

    def __getattr__(self, name):
        return getattr(self.decode, name)

    async def post(self, path: str, body: dict | None) -> None:
        import aiohttp

        if self._client is None:
            self._client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        async with self._client.post(self.prefill_url + path, json=body) as response:
            if response.status != 200:
                raise RuntimeError(f"prefill server {path} returned {response.status}: {(await response.text())[:512]}")
            await response.read()

    async def async_generate(self, **kwargs):
        key = "input_ids" if "input_ids" in kwargs else "prompt"
        count = len(kwargs[key])
        rank = kwargs.pop("routed_dp_rank", None)
        rooms = [secrets.randbits(62) for _ in range(count)]
        if rank is not None:
            # The prefill server places a room on DP rank room % dp_size and refuses a routed rank that disagrees
            # ("follow_bootstrap_room conflict", 8xB300 2026-10-27), so each room carries the session's prefill rank.
            prefill_rank, decode_rank = rank % self.prefill_ranks, rank % self.decode_ranks
            rooms = [room - room % self.prefill_ranks + prefill_rank for room in rooms]
        route = {"bootstrap_host": [HOST] * count, "bootstrap_port": [self.bootstrap_port] * count, "bootstrap_room": rooms}
        prefill = asyncio.ensure_future(self.post("/generate", {
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
            self.process.terminate()
            try:
                self.process.wait(30)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _serve_prefill() -> None:
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import ServerArgs

    load_plugins()
    run_server(ServerArgs(**json.loads(sys.stdin.read())))


if __name__ == "__main__":
    _serve_prefill()
