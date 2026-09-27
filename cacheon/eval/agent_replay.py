"""Run a sealed AIPerf slice through the existing isolated-engine session.

The validator owns this loopback HTTP adapter, chat template and timestamps.
The candidate still sees only disclosed generation inputs over its OCI pipes.
AIPerf owns trajectory order; the bridge owns arrival timing as lockstep rounds.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shutil
import signal
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import UUID

from cacheon.eval import aiperf_client
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
    slice: SliceManifest = field(init=False, repr=False)

    def __post_init__(self):
        manifest = load_slice_manifest(self.manifest_path)
        if type(self.loads) is not tuple or len(self.loads) != 1:
            raise ValueError("replay window needs exactly one sealed load")
        for load in self.loads:
            manifest.expected_work(load)
        if manifest.loader != "weka_trace":
            raise ValueError("agent replay requires the sealed Weka loader")
        if self.arm not in ("incumbent", "candidate") or not self.lane or self.window < 1:
            raise ValueError("agent replay read identity is invalid")
        if type(self.windows) is not int or self.windows < 1:
            raise ValueError("agent replay needs at least one sealed window")
        object.__setattr__(self, "slice", manifest)

    def workload_identity(self) -> dict:
        """Bind the consumed bytes, root accounting and replay policy, not local paths."""
        return {
            "slice": self.slice.digest, "dataset": self.slice.dataset,
            "revision": self.slice.revision,
            "rules_json": json.dumps(self.slice.rules, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "expected_work": {str(load): self.slice.expected_work(load) for load in self.loads},
            "loads": list(self.loads),
            "windows": self.windows,
            "arrival": "lockstep-rounds",
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


_ROUND_SETTLE_S = 0.5
_FIRST_ROUND_WAIT_S = 120.0


class _Rounds:
    """Lockstep arrival barrier: a round releases every held request at once, after the previous round drained.

    The client issues each conversation's next request the moment the previous
    one completes, so its timing carries the engine's own jitter back into the
    arrival pattern and a read is never the same schedule twice (closed-loop
    calibration scattered 1% per read, 2026-09-27). Holding requests until
    nothing is in flight and no new request has arrived for ``_ROUND_SETTLE_S``
    makes the batch composition a function of the slice alone. The first round
    waits for exactly ``openings`` conversations instead, so a slow client
    worker cannot shrink it, and fails every held request when the client has
    not opened them within ``_FIRST_ROUND_WAIT_S`` of its last arrival rather
    than idling to the session deadline. A round is released in session-key
    order so rank placement is the same in every read, and each release is
    numbered in that order: the engine requires batch indices in dispatch
    order, which arrival order no longer is.
    """

    def __init__(self, openings: int, clock, settle_s: float = _ROUND_SETTLE_S,
                 first_wait_s: float = _FIRST_ROUND_WAIT_S):
        self.openings, self.clock, self.settle_s, self.first_wait_s = openings, clock, settle_s, first_wait_s
        self.pending: dict[str, tuple[str, asyncio.Future]] = {}
        self.inflight: set[str] = set()
        self.first_released = False
        self.released = 0
        self._timer = None

    async def hold(self, request_id: str, key: str) -> tuple[float, int]:
        """Wait for the round this request joins; return its release time on the session clock and dispatch ordinal."""
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = (key, future)
        self._arm()
        return await future

    def done(self, request_id: str) -> None:
        self.inflight.discard(request_id)
        self.pending.pop(request_id, None)
        self._arm()

    def _arm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.inflight or not self.pending:
            return
        loop = asyncio.get_running_loop()
        if self.first_released:
            self._timer = loop.call_later(self.settle_s, self._release)
        elif len(self.pending) >= self.openings:
            self._release()
        else:
            self._timer = loop.call_later(self.first_wait_s, self._starve)

    def _release(self) -> None:
        self._timer = None
        if self.inflight or not self.pending or (not self.first_released and len(self.pending) < self.openings):
            return
        self.first_released = True
        stamp = self.clock()
        for request_id, (_key, future) in sorted(self.pending.items(), key=lambda item: item[1][0]):
            if not future.done():
                self.inflight.add(request_id)
                future.set_result((stamp, self.released))
                self.released += 1
        self.pending.clear()

    def _starve(self) -> None:
        self._timer = None
        error = RuntimeError(f"first lockstep round holds {len(self.pending)} of {self.openings} conversations "
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
        self.rounds = _Rounds(load, session.clock)
        self.placement = _Placement(session.plan.engine_config.engine_kwargs.get("dp_size", 1))
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
            body = await http_request.json()
            external_id = http_request.headers["X-Request-ID"]
            request_id = UUID(external_id).hex
            if external_id in self.rows:
                raise ValueError("AIPerf repeated an X-Request-ID")
            if body.get("stream") is not True:
                raise ValueError("agent replay requires streaming requests")
            ids = _chat_input_ids(self.tokenizer, body)
            count = body.get("max_tokens", body.get("max_completion_tokens"))
            self.rows[external_id] = None
            stamp, ordinal = await self.rounds.hold(external_id, _session_key(body["messages"]))
            self.stamps[external_id] = self._ns(stamp)
            index = self.first_batch_index + ordinal
            rank = self.placement.acquire(body["messages"])
            request = BatchRequest(
                self.session.session_id, self.session.plan.launch_digest,
                request_id, secrets.token_hex(16), index, (), count, 0,
                self.session.plan.temperature, len(ids), True, (tuple(ids),), rank,
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
                "input_ids": ids, "output_ids": list(prompt.output_ids),
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


def collect_read(plan: AgentReplayPlan, bridge: ReplayBridge) -> LoadRead:
    """Join client credits and source coordinates to host-observed pipe evidence."""
    records, seen, ordinals = [], set(), defaultdict(int)
    metadata = [json.loads(line)["metadata"] for line in
                (plan.output_directory / "aiperf" / "profile_export.jsonl").read_text().splitlines()]
    for meta in sorted(metadata, key=lambda r: r["request_start_ns"]):
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
    (load,) = plan.loads
    read = LoadRead(plan.arm, plan.window, plan.lane, load, tuple(records))
    rate = fixed_work_rate(read, plan.slice.expected_work(load))
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


async def _run_load_read(session, plan: AgentReplayPlan, *, tokenizer) -> LoadRead:
    """Drain one sealed load on the opened engine and retain the scorer's input."""
    from aiohttp import web

    current = load_slice_manifest(plan.manifest_path)
    if current != plan.slice:
        raise ValueError("replay manifest changed after planning")
    output = plan.output_directory
    output.mkdir(parents=True, exist_ok=False)
    pool = output / "slice"
    pool.mkdir()
    (load,) = plan.loads
    for source in sorted(current.directory.glob("*.json"))[:load]:
        (pool / source.name).symlink_to(source.resolve())
    async with session.exchange() as exchange:
        bridge = ReplayBridge(session, exchange, tokenizer, output, load)
        app = web.Application(client_max_size=MAX_BATCH_REQUEST_BYTES)
        app.router.add_post("/v1/chat/completions", bridge.chat)
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        client = None
        waiters = []
        # AIPerf binds Unix sockets under its IPC directory and sun_path holds 107 bytes; the
        # spool's TMPDIR can be deep (2026-09-27: a 121-byte path killed both arms' clients).
        ipc = Path(tempfile.mkdtemp(prefix="cacheon-aiperf-", dir="/tmp"))
        try:
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            argv = [str(plan.aiperf_binary.with_name("python")),
                    str(Path(aiperf_client.__file__)), current.digest, str(output),
                    "--model", str(plan.tokenizer_path),
                    "--tokenizer", str(plan.tokenizer_path), "--tokenizer-trust-remote-code",
                    "--url", f"http://127.0.0.1:{port}", "--endpoint-type", "chat", "--streaming",
                    "--no-fixed-schedule", "--input-file", str(pool),
                    "--custom-dataset-type", "weka_trace", "--num-conversations", str(load),
                    "--concurrency", str(load), "--dataset-sampling-strategy", "sequential",
                    "--cache-bust", "first_turn_prefix", "--ignore-trace-delays", "--use-server-token-count",
                    "--random-seed", str(current.rules["seed"]), "--ui", "none",
                    "--zmq-ipc-path", str(ipc), "--output-artifact-dir", str(output / "aiperf")]
            context = session.plan.engine_config.engine_kwargs.get("context_length")
            if context is not None:
                argv.extend(("--max-context-length", str(context)))
            (output / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
            with (output / "aiperf.log").open("w") as log:
                client = await asyncio.create_subprocess_exec(*argv, stdout=log, stderr=log, start_new_session=True)
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
            read = collect_read(plan, bridge)
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


async def run_replay(session, plan: AgentReplayPlan, *, tokenizer=None, before_read=None) -> tuple[LoadRead, ...]:
    """Execute the sealed windows of one load, flushing cache before each window's read.

    ``before_read`` synchronizes paired lanes after their cache flushes and
    before releasing either client's first request of that window, and returns
    whether the sealed sequential rule still wants that window. Single-lane
    execution uses the identical read and retained evidence path without a
    peer barrier. Every window is a fresh read of the same fixed work; the
    scorer averages the paired ratios, so the sealed window count is the
    budget that buys margin.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plan.tokenizer_path, trust_remote_code=True, local_files_only=True)
    reads = []
    (load,) = plan.loads
    for window in range(1, plan.windows + 1):
        await flush_cache(session)
        if before_read is not None and not await before_read(load, window):
            break
        read_plan = replace(plan, window=window, output_directory=plan.output_directory / f"window{window}")
        read = await _run_load_read(session, read_plan, tokenizer=tokenizer)
        session.replay_reads.append(read)
        reads.append(read)
    result = {"workload": plan.workload_identity(), "reads": [asdict(read) for read in reads]}
    (plan.output_directory / "window.json").write_text(json.dumps(result, indent=2) + "\n")
    return tuple(reads)
