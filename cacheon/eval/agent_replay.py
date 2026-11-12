"""Run a sealed AIPerf slice through the existing isolated-engine session.

The validator owns this loopback HTTP adapter, chat template and timestamps.
The candidate still sees only disclosed generation inputs over its OCI pipes.
AIPerf owns trajectory order; the bridge owns arrival timing: one cold round, then closed-loop arrival
at the sealed session count.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import secrets
import shutil
import signal
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import UUID

from cacheon.eval import aiperf_client, pd_engine
from cacheon.eval.agent_slice import SliceManifest, load_slice_manifest
from cacheon.eval.oci_session_protocol import BatchRequest, MAX_BATCH_REQUEST_BYTES
from cacheon.eval.oci_request_exchange import flush_cache
from cacheon.eval.service_capacity import (
    LoadRead, ServiceContract, ServiceEvidenceError, TurnRecord, attainment, fixed_work_rate,
)


@dataclass(frozen=True)
class AgentReplayPlan:
    """Validator-local inputs for one sealed load window, separate from candidate inputs."""

    manifest_path: Path
    loads: tuple[int, ...]
    aiperf_binary: Path
    tokenizer_path: Path
    output_directory: Path
    contract: ServiceContract
    arm: str
    window: int
    lane: str
    windows: int = 1
    elapsed_cost: bool = field(default=False, metadata={"wire_optional": True})
    max_work_seconds: int = field(default=0, metadata={"wire_optional": True})
    # Always true: lockstep rounds retired with speed policy 19. The field stays so sealed plans keep their wire shape.
    closed_loop: bool = field(default=False, metadata={"wire_optional": True})
    # Untimed windows before the sealed ones. Verda z3 lifetimes' third windows agreed to 0.04%; their first ran 1.4-3.1%
    # slow (stock Triton kernels keyed on batch and KV pool size compile on the clock), their second up to 1.3% (11-02).
    warm_windows: int = field(default=0, metadata={"wire_optional": True})
    slice: SliceManifest = field(init=False, repr=False)

    def __post_init__(self):
        manifest = load_slice_manifest(self.manifest_path)
        # A lifetime replays several loads highest first, in a closed loop: the top load's windows are the
        # one-load windows, so warm-up, quality controls and request indexing read them unchanged.
        if (type(self.loads) is not tuple or not self.loads or list(self.loads) != sorted(set(self.loads), reverse=True)
                or self.closed_loop is not True):
            raise ValueError("replay needs descending distinct sealed loads in a closed loop")
        for load in self.loads:
            manifest.expected_work(load)
        if manifest.loader != "weka_trace":
            raise ValueError("agent replay requires the sealed Weka loader")
        if self.arm not in ("incumbent", "candidate") or not self.lane or self.window < 1:
            raise ValueError("agent replay read identity is invalid")
        if type(self.windows) is not int or self.windows < 1:
            raise ValueError("agent replay needs at least one sealed window")
        if type(self.warm_windows) is not int or not 0 <= self.warm_windows <= 2:
            raise ValueError("agent replay warm windows are outside their bound")
        if type(self.elapsed_cost) is not bool:
            raise ValueError("replay cost accounting must be boolean")
        if type(self.max_work_seconds) is not int or not 0 <= self.max_work_seconds <= 7200:
            raise ValueError("replay work deadline is outside its bound")
        object.__setattr__(self, "slice", manifest)

    @property
    def sessions(self) -> int:
        """Sessions one window at the top load replays."""
        return self.sessions_at(self.loads[0])

    def sessions_at(self, load: int) -> int:
        """Sessions one window at ``load`` replays. Closed loop keeps the load in flight over the whole slice, each
        finished session handing its slot to the next: replaying only ``load`` sessions drained the window to its
        longest one (output per session spans 4.5x on the Weka slice, so mean concurrency fell to ~N/4.5). A lower
        load replays the slice's first sessions in proportion, so every load turns its slots over equally and a
        window lasts about as long at each load (Shiv 2026-11-11: the battery must not lengthen a bundle's wall)."""
        return max(load, self.slice.max_load * load // self.loads[0])

    def workload_identity(self) -> dict:
        """Bind the consumed bytes, root accounting and replay policy, not local paths."""
        return {
            **({"time_accounting": "elapsed-serving-span"} if self.elapsed_cost else {}),
            **({"max_work_seconds": self.max_work_seconds} if self.max_work_seconds else {}),
            "slice": self.slice.digest, "dataset": self.slice.dataset,
            "revision": self.slice.revision,
            "rules_json": json.dumps(self.slice.rules, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "expected_work": {str(load): self.slice.expected_work(self.sessions_at(load)) for load in self.loads},
            "loads": list(self.loads),
            "windows": self.windows,
            **({"warm_windows": self.warm_windows} if self.warm_windows else {}),
            "arrival": "closed-loop-after-cold-round",
            "contract": {key: format(value, ".17g") for key, value in asdict(self.contract).items()},
            "client": "aiperf-0.13.0", "scenario": None, "ignore_trace_delays": True,
            "cache_bust_identity": "sealed-slice-digest",
            "cache_boundary": "flush_device_and_host_before_each_load",
        }


def _session_key(messages: list[dict]) -> str:
    """The conversation's opening (system messages plus the first user turn), stable across its turns.

    AIPerf's per-session cache-bust marker sits at the head of the first user turn and is stripped
    so the key names the sealed session, not the marker a client happened to mint for it.
    """
    head = [("system", json.dumps(m.get("content"), sort_keys=True)[:4096])
            for m in messages if m.get("role") == "system"]
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
            if text.startswith("[rid:") and "\n\n" in text[:80]:
                text = text.split("\n\n", 1)[1]
            head.append(("user", text[:4096]))
            break
    return hashlib.sha256(json.dumps(head).encode()).hexdigest()


class _Placement:
    """Balanced session-to-rank placement for one read: each DP rank owns its own radix cache.

    A session's every turn goes to the rank chosen at its first turn. A new session takes
    the rank with the fewest turns in flight, then the fewest sessions placed, then the lowest
    index, so a read's roots land 0,1,2,3,... whatever the arrival timing. Hashing the opening
    instead re-randomised placement per run and cost 5-7% of the fixed-work rate at loads 12-16
    (calibration, 2026-09-27).
    """

    def __init__(self, ranks: int) -> None:
        self.ranks = ranks
        self.assigned = [0] * ranks
        self.inflight = [0] * ranks
        self.table: dict[str, int] = {}

    def acquire(self, messages: list[dict]) -> int:
        key = _session_key(messages)
        rank = self.table.get(key)
        if rank is None:
            rank = min(range(self.ranks), key=lambda r: (self.inflight[r], self.assigned[r], r))
            self.table[key] = rank
            self.assigned[rank] += 1
        self.inflight[rank] += 1
        return rank

    def release(self, rank: int) -> None:
        self.inflight[rank] -= 1


def _chat_input_ids(tokenizer, body):
    # Transformers 5 defaults to BatchEncoding. Iterating that mapping sent its
    # field names as IDs and stopped the first real GLM replay before generation.
    return tokenizer.apply_chat_template(
        body["messages"], tools=body.get("tools"), tokenize=True,
        add_generation_prompt=True, return_dict=False,
    )


# Templating a 135k-token agent turn costs ~0.21 s of CPU. On the bridge's event loop it capped the PD arena at
# 3.4 turns/s against the engines' ~16 and held every first-token frame behind it (TTFT median 47 s, 2026-10-27).
_TOKENIZE_WORKERS = 8
_pool_tokenizer = None


def _load_pool_tokenizer(path: str) -> None:
    global _pool_tokenizer
    from transformers import AutoTokenizer
    _pool_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)


def _pool_chat_input_ids(body: dict) -> tuple[list[int], str]:
    ids = _chat_input_ids(_pool_tokenizer, body)
    return ids, hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


# Every window replays the same sealed bodies: aiperf_client fixes the benchmark identity, so the markers repeat, and
# per-turn prompt counts matched across stock and candidate lifetimes (482,559,756 tokens a window, 2026-11-02). Yet
# each timed window re-templated every turn, holding 4.8-6.1 client vCPUs busy on Verda, and the bridge loop unpickled
# each turn's IDs and dumped them to bridge.jsonl: ~6 ms a request and ~480M int allocations a window on the one loop
# every first token passes through. Each distinct body is now tokenized once per evaluator process, by prime_replay
# before the first engine lifetime, into one tuple of shared ints that every later request reuses. A timed body the
# prime did not see fails the read: it would put the tokenizer back inside the clock.
_PROMPTS: dict[bytes, tuple[tuple[int, ...], str]] = {}
_PRIMED: set[str] = set()
_TOKEN_INTS: tuple[int, ...] = ()


def _shared(ids: list[int]) -> tuple[int, ...]:
    global _TOKEN_INTS
    top = max(ids, default=0)
    if top >= len(_TOKEN_INTS):
        _TOKEN_INTS = tuple(range(max(top + 1, 1 << 18)))
    return tuple(map(_TOKEN_INTS.__getitem__, ids))


def _prompt_ids(raw: bytes) -> tuple[tuple[int, ...], str]:
    """The primed canonical IDs and their digest for one exact request body."""
    cached = _PROMPTS.get(hashlib.sha256(raw).digest())
    if cached is None:
        raise RuntimeError("AIPerf sent a replay body the prime did not tokenize")
    return cached


_FIRST_ROUND_WAIT_S = 120.0


class _ColdRound:
    """Release the window's first ``openings`` conversations together, then let every later request leave on arrival.

    The client issues each conversation's next request the moment the previous one completes, so the sealed
    session count is the engine's offered concurrency, as a serving fleet sees it. Holding only the cold round
    keeps cold/warm membership a function of the slice: it waits for exactly ``openings`` conversations, so a
    slow client worker cannot shrink it, and fails every held request when the client has not opened them within
    ``_FIRST_ROUND_WAIT_S`` of its last arrival rather than idling to the session deadline. It is released in
    session-key order so rank placement is the same in every read, and each release is numbered in that order:
    the engine requires batch indices in dispatch order, which arrival order is not.
    """

    def __init__(self, openings: int, clock, first_wait_s: float = _FIRST_ROUND_WAIT_S):
        self.openings, self.clock, self.first_wait_s = openings, clock, first_wait_s
        self.pending: dict[str, tuple[str, asyncio.Future]] = {}
        self.released = 0
        self._timer = None

    async def hold(self, request_id: str, key: str) -> tuple[float, int]:
        """Wait for the cold round if it is still open; return the release time on the session clock and dispatch ordinal."""
        if self.released:
            self.released += 1
            return self.clock(), self.released - 1
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = (key, future)
        if self._timer is not None:
            self._timer.cancel()
        # Never inline: in the last opening's own hold() its future is already done, so it took
        # rank 0 before the others woke in session-key order, randomising placement (2026-09-30).
        full = len(self.pending) >= self.openings
        self._timer = asyncio.get_running_loop().call_later(
            0 if full else self.first_wait_s, self._release if full else self._starve)
        return await future

    def done(self, request_id: str) -> None:
        self.pending.pop(request_id, None)

    def _release(self) -> None:
        self._timer = None
        stamp = self.clock()
        for _key, future in sorted(self.pending.values(), key=lambda item: item[0]):
            if not future.done():
                future.set_result((stamp, self.released))
                self.released += 1
        self.pending.clear()

    def _starve(self) -> None:
        self._timer = None
        error = RuntimeError(f"cold round holds {len(self.pending)} of {self.openings} conversations "
                             f"after {self.first_wait_s:g} s without a new arrival")
        for _key, future in self.pending.values():
            if not future.done():
                future.set_exception(error)
        self.pending.clear()


class ReplayBridge:
    """Adapt trusted chat requests to canonical IDs and retain their actual token evidence."""

    def __init__(self, session, exchange, tokenizer, output: Path, load: int):
        self.session, self.exchange, self.tokenizer = session, exchange, tokenizer
        self.first_batch_index = session.next_batch_index
        self.rows, self.failure = {}, None
        self.stamps: dict[str, int] = {}
        self.rounds = _ColdRound(load, session.clock)
        self.placement = _Placement(pd_engine.routes(dict(session.plan.engine_config.engine_kwargs)))
        self.failed = asyncio.Event()
        self.offset_ns = time.time_ns() - round(session.clock() * 1e9)
        (output / "clock.json").write_text(json.dumps({
            "clock": "host-monotonic-to-unix-ns", "offset_ns": self.offset_ns,
        }) + "\n")
        self.raw = (output / "bridge.jsonl").open("x")

    def _ns(self, value):
        return self.offset_ns + round(value * 1e9)

    async def chat(self, http_request):
        """Deliver the real output, with server token usage and no synthetic successes."""
        from aiohttp import web

        rank = external_id = None
        try:
            raw = await http_request.read()
            body = json.loads(raw)
            external_id = http_request.headers["X-Request-ID"]
            request_id = UUID(external_id).hex
            if external_id in self.rows:
                raise ValueError("AIPerf repeated an X-Request-ID")
            if body.get("stream") is not True:
                raise ValueError("agent replay requires streaming requests")
            ids, digest = _prompt_ids(raw)
            count = body.get("max_tokens", body.get("max_completion_tokens"))
            self.rows[external_id] = None
            stamp, ordinal = await self.rounds.hold(external_id, _session_key(body["messages"]))
            self.stamps[external_id] = self._ns(stamp)
            index = self.first_batch_index + ordinal
            rank = self.placement.acquire(body["messages"])
            request = BatchRequest(
                self.session.session_id, self.session.plan.launch_digest,
                request_id, secrets.token_hex(16), index, (), count, 0,
                self.session.plan.temperature, len(ids), True, (ids,), rank,
            )
            if index == self.session.plan.warmup_count and self.session.boundary_callback:
                self.session.boundary_callback("before_first_timed", index, self.session.deadline)
            progress = asyncio.Queue()
            task = asyncio.create_task(self.exchange.execute(
                request, deadline=min(self.session.deadline, self.session.clock() + self.session.batch_timeout_s),
                on_progress=progress.put_nowait,
            ))
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(http_request)

            async def emit(text, *, usage=None):
                chunk = {
                    "id": external_id, "object": "chat.completion.chunk", "created": 0,
                    "model": body["model"], "choices": [{"index": 0, "delta": {"content": text},
                    "finish_reason": "length" if usage is not None else None}],
                }
                if usage is not None:
                    chunk["usage"] = usage
                await response.write(("data: " + json.dumps(chunk) + "\n\n").encode())

            prefix = ""
            event_task = asyncio.create_task(progress.get())
            try:
                done, _ = await asyncio.wait((task, event_task), return_when=asyncio.FIRST_COMPLETED)
                if event_task in done:
                    token = event_task.result()["token_id"]
                    prefix = self.tokenizer.decode([token], skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)
                    # Incomplete UTF-8 bytes cannot be rendered yet. Their token
                    # arrival is still timed by HostTokenClock, not HTTP text.
                    if "\ufffd" in prefix:
                        prefix = ""
                    if prefix:
                        await emit(prefix)
                row = await task
            finally:
                event_task.cancel()
                if not task.done():
                    task.cancel()
                await asyncio.gather(event_task, task, return_exceptions=True)
            prompt = row.evidence.prompts[0]
            text = self.tokenizer.decode(prompt.output_ids, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
            if not text.startswith(prefix):
                raise ValueError("tokenizer's final output changed its delivered prefix")
            usage = {"prompt_tokens": prompt.prompt_tokens, "completion_tokens": len(prompt.output_ids),
                     "total_tokens": prompt.prompt_tokens + len(prompt.output_ids)}
            self.rows[external_id] = row
            self.raw.write(json.dumps({
                "x_request_id": external_id, "batch_index": index,
                "input_sha256": digest, "output_ids": list(prompt.output_ids),
                "request_start_ns": self._ns(row.request_started_at),
                "first_token_ns": self._ns(row.request_started_at + row.prompt_latencies[0][0]),
                "request_end_ns": self._ns(row.response_completed_at),
                **usage,
            }) + "\n")
            self.raw.flush()
            await emit(text[len(prefix):], usage=usage)
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        except BaseException as exc:
            self.failure = exc
            self.failed.set()
            raise
        finally:
            if external_id is not None:
                self.rounds.done(external_id)
            if rank is not None:
                self.placement.release(rank)


def collect_read(plan: AgentReplayPlan, bridge: ReplayBridge, load: int) -> LoadRead:
    """Join client credits and source coordinates to host-observed pipe evidence.

    Ordinals follow the client's conversation coordinate, not its send time: two conversations of one
    root leave 1 ms apart in either order, which swapped ordinals on one lane and voided a complete read
    as different work (2026-10-01, off-chain request 959338a5).
    """
    records, seen, ordinals = [], set(), defaultdict(int)
    metadata = [json.loads(line)["metadata"] for line in
                (plan.output_directory / "aiperf" / "profile_export.jsonl").read_text().splitlines()]
    for meta in sorted(metadata, key=lambda r: (r["conversation_id"], r["turn_index"])):
        external_id = meta["x_request_id"]
        if external_id in seen or external_id not in bridge.rows:
            raise ServiceEvidenceError("AIPerf/bridge request join is not one-to-one")
        seen.add(external_id)
        row = bridge.rows[external_id]
        if row is None or meta.get("context_overflow_skip") or meta.get("was_cancelled"):
            raise ServiceEvidenceError("replay did not complete its fixed work")
        if meta["benchmark_phase"] != "profiling":
            continue
        root = meta["source_trace_id"]
        kind = "inner" if meta.get("source_inner_idx") is not None else "main"
        key = root, kind
        prompt = row.evidence.prompts[0]
        records.append(TurnRecord(
            root, kind, ordinals[key], bridge.stamps[external_id], bridge._ns(row.request_started_at),
            bridge._ns(row.request_started_at + row.prompt_latencies[0][0]),
            bridge._ns(row.response_completed_at), prompt.prompt_tokens, len(prompt.output_ids), "ok",
        ))
        ordinals[key] += 1
    if seen != set(bridge.rows):
        raise ServiceEvidenceError("bridge has requests absent from the retained AIPerf export")
    read = LoadRead(plan.arm, plan.window, plan.lane, load, tuple(records))
    rate = fixed_work_rate(read, plan.slice.expected_work(plan.sessions_at(load)), wall_time=plan.elapsed_cost)
    output = plan.output_directory
    with (output / "turns.jsonl").open("x") as f:
        for record in records:
            f.write(json.dumps({"arm": read.arm, "window": read.window, "lane": read.lane,
                                "load": read.load, **asdict(record)}) + "\n")
    (output / "read.json").write_text(json.dumps({
        **asdict(rate), "attainment": attainment(read, plan.contract),
        "workload": plan.workload_identity(),
    }, indent=2) + "\n")
    return read


def _current_slice(plan: AgentReplayPlan) -> SliceManifest:
    current = load_slice_manifest(plan.manifest_path)
    if current != plan.slice:
        raise ValueError("replay manifest changed after planning")
    return current


def _slice_pool(current: SliceManifest, output: Path, sessions: int) -> Path:
    pool = output / "slice"
    pool.mkdir()
    for source in sorted(current.directory.glob("*.json"))[:sessions]:
        (pool / source.name).symlink_to(source.resolve())
    return pool


def _aiperf_argv(plan: AgentReplayPlan, current: SliceManifest, output: Path, pool: Path, port: int, ipc: Path,
                 context: int | None, load: int) -> list[str]:
    argv = [str(plan.aiperf_binary.with_name("python")),
            str(Path(aiperf_client.__file__)), current.digest, str(output),
            "--model", str(plan.tokenizer_path),
            "--tokenizer", str(plan.tokenizer_path), "--tokenizer-trust-remote-code",
            "--url", f"http://127.0.0.1:{port}", "--endpoint-type", "chat", "--streaming",
            "--no-fixed-schedule", "--input-file", str(pool),
            "--custom-dataset-type", "weka_trace", "--num-conversations", str(plan.sessions_at(load)),
            "--concurrency", str(load), "--dataset-sampling-strategy", "sequential",
            "--cache-bust", "first_turn_prefix", "--ignore-trace-delays", "--use-server-token-count",
            "--random-seed", str(current.rules["seed"]), "--ui", "none",
            "--zmq-ipc-path", str(ipc), "--output-artifact-dir", str(output / "aiperf")]
    if context is not None:
        argv.extend(("--max-context-length", str(context)))
    return argv


_PRIME_CLIENT_S = 1800


def prime_replay(plan: AgentReplayPlan, *, context_length: int | None) -> None:
    """Tokenize every body of the sealed slice once in this evaluator process, before its first engine lifetime.

    AIPerf replays the slice exactly as a window does, against a responder that answers each request at once with no
    engine and no clock, so timed windows only look IDs up.
    """
    for load in plan.loads:
        identity = json.dumps([plan.slice.digest, str(plan.tokenizer_path), plan.sessions_at(load), [load],
                               context_length])
        if identity not in _PRIMED:
            asyncio.run(_prime(plan, context_length, load))
            _PRIMED.add(identity)


async def _prime(plan: AgentReplayPlan, context_length: int | None, load: int) -> None:
    from aiohttp import web

    current = _current_slice(plan)
    loop = asyncio.get_running_loop()
    pending: dict[bytes, asyncio.Future] = {}
    # A lower load replays the first sessions of the higher load's pool, so its bodies are all primed already and
    # `pending` stays empty; the guard counts requests received (GLM-5.3-Flash PD, three loads, 2026-11-12).
    received = 0
    # Short paths: AIPerf's Unix sockets live under both (see _run_load_read).
    work = Path(tempfile.mkdtemp(prefix="cacheon-prime-", dir="/tmp"))
    ipc = Path(tempfile.mkdtemp(prefix="cacheon-aiperf-", dir="/tmp"))
    with ProcessPoolExecutor(_TOKENIZE_WORKERS, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_load_pool_tokenizer, initargs=(str(plan.tokenizer_path),)) as pool:

        async def chat(http_request):
            nonlocal received
            received += 1
            raw = await http_request.read()
            body = json.loads(raw)
            key = hashlib.sha256(raw).digest()
            if key not in _PROMPTS and key not in pending:
                pending[key] = loop.run_in_executor(pool, _pool_chat_input_ids, body)
            count = body.get("max_tokens", body.get("max_completion_tokens")) or 1
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(http_request)
            await response.write(("data: " + json.dumps({
                "id": http_request.headers["X-Request-ID"], "object": "chat.completion.chunk", "created": 0,
                "model": body["model"], "choices": [{"index": 0, "delta": {"content": " "}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": count, "total_tokens": count + 1},
            }) + "\n\n").encode())
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        app = web.Application(client_max_size=MAX_BATCH_REQUEST_BYTES)
        app.router.add_post("/v1/chat/completions", chat)
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        client = None
        try:
            await site.start()
            argv = _aiperf_argv(plan, current, work, _slice_pool(current, work, plan.sessions_at(load)),
                                site._server.sockets[0].getsockname()[1], ipc, context_length, load)
            with (work / "aiperf.log").open("w") as log:
                client = await asyncio.create_subprocess_exec(
                    *argv, stdout=log, stderr=log, start_new_session=True, env={**os.environ, "TMPDIR": str(ipc)},
                )
                code = await asyncio.wait_for(client.wait(), _PRIME_CLIENT_S)
            if code:
                raise RuntimeError(f"AIPerf exited {code} priming the replay; see {work / 'aiperf.log'}")
            if not received:
                raise RuntimeError(f"AIPerf sent no request priming the replay; see {work / 'aiperf.log'}")
            for key, future in pending.items():
                ids, digest = await future
                _PROMPTS[key] = (_shared(ids), digest)
        finally:
            if client is not None and client.returncode is None:
                os.killpg(client.pid, signal.SIGKILL)
                await client.wait()
            await runner.cleanup()
            shutil.rmtree(ipc, ignore_errors=True)
    shutil.rmtree(work)


async def _run_load_read(session, plan: AgentReplayPlan, load: int, *, tokenizer) -> LoadRead:
    """Drain one sealed load on the opened engine and retain the scorer's input."""
    from aiohttp import web

    current = _current_slice(plan)
    output = plan.output_directory
    output.mkdir(parents=True, exist_ok=False)
    pool = _slice_pool(current, output, plan.sessions_at(load))
    async with session.exchange() as exchange:
        bridge = ReplayBridge(session, exchange, tokenizer, output, load)
        app = web.Application(client_max_size=MAX_BATCH_REQUEST_BYTES)
        app.router.add_post("/v1/chat/completions", bridge.chat)
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        client = None
        waiters = []
        # AIPerf binds Unix sockets under its IPC directory and, at larger loads, its dataset
        # loader's worker pool binds more under TMPDIR; sun_path holds 107 bytes and the spool's
        # TMPDIR can be deep (2026-09-27: a 121-byte path killed both arms' clients at load 12,
        # and the pool's socket did the same at load 24). Both live in this short directory.
        ipc = Path(tempfile.mkdtemp(prefix="cacheon-aiperf-", dir="/tmp"))
        try:
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            argv = _aiperf_argv(plan, current, output, pool, port, ipc,
                                session.plan.engine_config.engine_kwargs.get("context_length"), load)
            (output / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
            with (output / "aiperf.log").open("w") as log:
                client = await asyncio.create_subprocess_exec(
                    *argv, stdout=log, stderr=log, start_new_session=True,
                    env={**os.environ, "TMPDIR": str(ipc)},
                )
                waiters = [asyncio.create_task(client.wait()), asyncio.create_task(bridge.failed.wait())]
                done, _ = await asyncio.wait(waiters, timeout=max(0, session.deadline - session.clock()),
                                             return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    raise TimeoutError("AIPerf exceeded the engine session deadline")
            if bridge.failure is not None:
                raise bridge.failure
            code = waiters[0].result()
            if code:
                raise RuntimeError(f"AIPerf exited {code}; see {output / 'aiperf.log'}")
            read = collect_read(plan, bridge, load)
            rows = sorted(bridge.rows.values(), key=lambda row: row.batch_index)
            session.batch_rows.extend(rows)
            if session.first_timed_completed_at is None:
                session.first_timed_completed_at = min(row.response_completed_at for row in rows)
            session.last_host_time = max(row.response_completed_at for row in rows)
            return read
        finally:
            for waiter in waiters:
                waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
            if client is not None and client.returncode is None:
                os.killpg(client.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(client.wait(), 5)
                except asyncio.TimeoutError:
                    os.killpg(client.pid, signal.SIGKILL)
                    await client.wait()
            await runner.cleanup()
            bridge.raw.close()
            shutil.rmtree(ipc, ignore_errors=True)


async def run_replay(session, plan: AgentReplayPlan, *, tokenizer=None) -> tuple[LoadRead, ...]:
    """Execute the sealed windows of each load, highest first, flushing cache before each window's read.

    The warm windows replay once, at the top load, each into its own retained directory; their reads never reach
    the session's evidence or the grader. Every window is a fresh read of its load's fixed work, so the sealed
    window count is the budget that buys margin. The top load keeps the directory layout of a one-load plan.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plan.tokenizer_path, trust_remote_code=True, local_files_only=True)
    reads = []
    top = plan.loads[0]
    for warm in range(1, plan.warm_windows + 1):
        await flush_cache(session)
        await _run_load_read(session, replace(plan, window=warm, output_directory=plan.output_directory / f"warm{warm}"),
                             top, tokenizer=tokenizer)
    for load in plan.loads:
        directory = plan.output_directory / ("" if load == top else f"load{load}")
        for window in range(1, plan.windows + 1):
            await flush_cache(session)
            read_plan = replace(plan, window=window, output_directory=directory / f"window{window}")
            read = await _run_load_read(session, read_plan, load, tokenizer=tokenizer)
            session.replay_reads.append(read)
            reads.append(read)
    result = {"workload": plan.workload_identity(), "reads": [asdict(read) for read in reads]}
    (plan.output_directory / "window.json").write_text(json.dumps(result, indent=2) + "\n")
    return tuple(reads)
