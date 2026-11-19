"""Run a sealed AIPerf slice through the existing isolated-engine session.

The validator owns this loopback HTTP adapter, chat template and timestamps.
The candidate still sees only disclosed generation inputs over its OCI pipes.
AIPerf owns trajectory order; the bridge owns arrival timing: one cold round, then closed-loop arrival
at the sealed session count, or under ``pause_order`` release in the order of the trace's recorded pauses.
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
# The validator's own replay client failing is infrastructure: untyped, one AIPerf exit closed the worker's
# adapter epoch and parked the queue behind a transport hold (2026-11-13 and 2026-11-14).
from cacheon.eval.oci_outer_session import OuterSessionInfrastructureError, OuterSessionTimeoutError
from cacheon.eval.oci_request_exchange import flush_cache
from cacheon.eval.phase_latency import _cache_report
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
    max_work_seconds: int = 0
    # Untimed windows before the sealed ones. Verda z3 lifetimes' third windows agreed to 0.04%; their first ran 1.4-3.1%
    # slow (stock Triton kernels keyed on batch and KV pool size compile on the clock), their second up to 1.3% (11-02).
    warm_windows: int = 0
    # Every session is live from the cold round and a freed slot goes to the waiting turn whose recorded pause ends
    # first. The Weka sessions are alive 25x longer than they are in flight (48-session slice, 2026-11-11); arrival
    # on completion never let a paused session's KV go cold, so no tier or retention choice could be scored.
    pause_order: bool = False
    slice: SliceManifest = field(init=False, repr=False)

    def __post_init__(self):
        manifest = load_slice_manifest(self.manifest_path)
        # A lifetime replays several loads highest first: the top load's windows are the one-load windows, so
        # warm-up, quality controls and request indexing read them unchanged.
        if type(self.loads) is not tuple or not self.loads or list(self.loads) != sorted(set(self.loads), reverse=True):
            raise ValueError("replay needs descending distinct sealed loads")
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
        if type(self.pause_order) is not bool:
            raise ValueError("replay release order must be boolean")
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
            "time_accounting": "elapsed-serving-span",
            **({"max_work_seconds": self.max_work_seconds} if self.max_work_seconds else {}),
            "slice": self.slice.digest, "dataset": self.slice.dataset,
            "revision": self.slice.revision,
            "rules_json": json.dumps(self.slice.rules, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "expected_work": {str(load): self.slice.expected_work(self.sessions_at(load)) for load in self.loads},
            "loads": list(self.loads),
            "windows": self.windows,
            **({"warm_windows": self.warm_windows} if self.warm_windows else {}),
            "arrival": "closed-loop-after-cold-round",
            **({"release_order": "recorded-pause"} if self.pause_order else {}),
            "contract": {key: format(value, ".17g") for key, value in asdict(self.contract).items()},
            "client": "aiperf-0.13.0", "scenario": None, "ignore_trace_delays": True,
            "cache_bust_identity": "sealed-slice-digest",
            "cache_boundary": ("flush_device_and_host_before_each_load" if len(self.loads) == 1
                               else "flush_device_and_host_before_top_load_windows"),
        }


class _Placement:
    """Balanced session-to-rank placement for one read: each DP rank owns its own radix cache.

    Requests sharing the replay's session key go to the rank chosen at the first turn. A new
    session takes the rank with the fewest turns in flight, then the fewest sessions placed, then the lowest
    index, so a read's roots land 0,1,2,3,... whatever the arrival timing. Hashing the opening
    instead re-randomised placement per run and cost 5-7% of the fixed-work rate at loads 12-16
    (calibration, 2026-09-27).
    """

    def __init__(self, ranks: int) -> None:
        self.ranks = ranks
        self.assigned = [0] * ranks
        self.inflight = [0] * ranks
        self.table: dict[int, int] = {}

    def acquire(self, key: int) -> int:
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
# prime did not see fails the read: it would put the tokenizer back inside the clock. The prime also records each
# body's place in the sealed slice: (session order, root turn or None, recorded pause in seconds before that session's
# next root turn or None). Subagent and inner requests carry no turn: the trace records no pause for them.
_Place = tuple[int, int | None, float | None]
_PROMPTS: dict[bytes, tuple[tuple[int, ...], str, _Place]] = {}
_PRIMED: set[str] = set()
_TOKEN_INTS: tuple[int, ...] = ()


def _request_place(messages, trace_place: _Place, *, pause_order: bool) -> _Place:
    """Use the public conversation opening for ordinary replays and trace coordinates for recorded pauses."""
    if pause_order:
        return trace_place
    # Sorting by the dataset ordinal instead kept all 96 opening positions on the same warm
    # replica across loads. Only 56 opening positions kept their parity on the retained mainnet node.
    head = [("system", json.dumps(m.get("content"), sort_keys=True)[:4096])
            for m in messages if m.get("role") == "system"]
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content")
            text = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
            if text.startswith("[rid:") and "\n\n" in text[:80]:
                text = text.split("\n\n", 1)[1]
            head.append(("user", text[:4096]))
            break
    return int.from_bytes(hashlib.sha256(json.dumps(head).encode()).digest(), "big"), None, None


def _shared(ids: list[int]) -> tuple[int, ...]:
    global _TOKEN_INTS
    top = max(ids, default=0)
    if top >= len(_TOKEN_INTS):
        _TOKEN_INTS = tuple(range(max(top + 1, 1 << 18)))
    return tuple(map(_TOKEN_INTS.__getitem__, ids))


def _prompt_ids(raw: bytes) -> tuple[tuple[int, ...], str, _Place]:
    """The primed canonical IDs, their digest and the slice place for one exact request body."""
    cached = _PROMPTS.get(hashlib.sha256(raw).digest())
    if cached is None:
        raise RuntimeError("AIPerf sent a replay body the prime did not tokenize")
    return cached


def _places(current: SliceManifest, export: Path, seen: dict[str, bytes]) -> dict[bytes, _Place]:
    """Join the prime's request bodies to the client's source coordinates and the trace's own pauses.

    A Weka request's ``think_time`` is the idle gap before it (t minus the previous request's end, checked on the
    48-session slice), and ``source_outer_idx`` is its index in the session file's requests.
    """
    order = {session.id: index for index, session in enumerate(current.sessions)}
    places, roots = {}, defaultdict(list)
    for line in export.read_text().splitlines():
        meta = json.loads(line)["metadata"]
        key, trace = seen[meta["x_request_id"]], meta["source_trace_id"]
        places[key] = (order[trace], None, None)
        if meta["conversation_id"] == trace and meta.get("source_inner_idx") is None:
            roots[trace].append((meta["turn_index"], meta["source_outer_idx"], key))
    for trace, turns in roots.items():
        requests = json.loads((current.directory / f"{order[trace]:03d}_{trace}.json").read_text())["requests"]
        turns.sort()
        for (turn, _outer, key), following in zip(turns, [*turns[1:], None]):
            pause = None if following is None else float(requests[following[1]].get("think_time") or 0.0)
            places[key] = (order[trace], turn, pause)
    return places


_FIRST_ROUND_WAIT_S = 120.0


class _Arrival:
    """Release the window's first ``openings`` conversations together, then every later request by the sealed rule.

    The client issues each conversation's next request the moment the previous one completes, so the sealed
    session count is the engine's offered concurrency, as a serving fleet sees it. Holding only the cold round
    keeps cold/warm membership a function of the slice: it waits for exactly ``openings`` conversations, so a
    slow client worker cannot shrink it, and fails every held request when the client has not opened them within
    ``_FIRST_ROUND_WAIT_S`` of its last arrival rather than idling to the session deadline. It is released in
    sealed session order so rank placement is the same in every read, and each release is numbered in that order:
    the engine requires batch indices in dispatch order, which arrival order is not.

    Without ``limit`` a later request leaves on arrival. With it (pause order) every session of the window is an
    opening, ``limit`` requests are in flight, and a freed slot goes to the waiting request that is due first: a
    cold start at once, a session's next root turn at the moment the session went idle plus the pause the trace
    recorded before that turn. Nothing waits for the wall clock, so a session that paused for an hour returns
    behind the turns that paused for seconds, with its KV wherever the cache left it. An idle session's next turn
    is expected before the client sends it and one slot is kept for it, not handed to a later-due request in the
    client's milliseconds; requests held behind a turn that never arrives fail after ``first_wait_s``.
    """

    def __init__(self, openings: int, clock, limit: int | None = None, first_wait_s: float = _FIRST_ROUND_WAIT_S):
        self.openings, self.clock, self.limit, self.first_wait_s = openings, clock, limit, first_wait_s
        self.released = 0
        self._timer = None
        self.places: dict[str, _Place] = {}
        self.inflight: set[str] = set()
        # Held requests as (due time, sealed session order, future); the cold round is due at zero.
        self.waiting: dict[str, tuple[float, int, asyncio.Future]] = {}
        # Per sealed session: requests held or in flight, the pause before its next root turn, that turn's due time.
        self.active: dict[int, int] = defaultdict(int)
        self.pauses: dict[int, float] = {}
        self.expected: dict[int, float] = {}

    async def hold(self, request_id: str, place: _Place) -> tuple[float, int]:
        """Wait for release; return the release time on the session clock and dispatch ordinal."""
        if self.limit is None and self.released:
            self.released += 1
            return self.clock(), self.released - 1
        order, turn, _pause = self.places[request_id] = place
        self.active[order] += 1
        future = asyncio.get_running_loop().create_future()
        if self.released:
            due = self.expected.pop(order, None)
            if turn is not None:
                self.pauses.pop(order, None)
            self.waiting[request_id] = (self.clock() if due is None or turn is None else due, order, future)
            self._grant()
            return await future
        self.waiting[request_id] = (0.0, order, future)
        # Never inline: in the last opening's own hold() its future is already done, so it took
        # rank 0 before the others woke in session order, randomising placement (2026-09-30).
        full = len(self.waiting) >= self.openings
        self._arm(0 if full else self.first_wait_s, self._grant if full else lambda: self._fail(
            f"cold round holds {len(self.waiting)} of {self.openings} conversations"))
        return await future

    def done(self, request_id: str) -> None:
        self.waiting.pop(request_id, None)
        place = self.places.pop(request_id, None)
        if place is None:
            return
        order, _turn, pause = place
        if request_id in self.inflight and pause is not None:
            self.pauses[order] = pause
        self.inflight.discard(request_id)
        self.active[order] -= 1
        if not self.active[order] and order in self.pauses:
            self.expected[order] = self.clock() + self.pauses[order]
        if self.limit is not None and self.released:
            self._grant()

    def _arm(self, delay: float | None, callback=None) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None if delay is None else asyncio.get_running_loop().call_later(delay, callback)

    def _grant(self) -> None:
        stamp = self.clock()
        free = float("inf") if self.limit is None else self.limit - len(self.inflight)
        for request_id, (due, _order, future) in sorted(self.waiting.items(), key=lambda item: item[1][:2]):
            if free <= sum(1 for expected in self.expected.values() if expected < due):
                break
            del self.waiting[request_id]
            if not future.done():
                self.inflight.add(request_id)
                future.set_result((stamp, self.released))
                self.released += 1
                free -= 1
        held = bool(self.waiting) and free > 0
        self._arm(self.first_wait_s if held else None, lambda: self._fail(
            f"{len(self.expected)} idle conversations sent no next turn; {len(self.waiting)} requests are held behind them"))

    def _fail(self, cause: str) -> None:
        self._timer = None
        error = RuntimeError(f"{cause} after {self.first_wait_s:g} s without a new arrival")
        for _due, _order, future in self.waiting.values():
            if not future.done():
                future.set_exception(error)
        self.waiting.clear()


class ReplayBridge:
    """Adapt trusted chat requests to canonical IDs and retain their actual token evidence."""

    def __init__(self, session, exchange, tokenizer, output: Path, load: int, live: int = 0):
        self.session, self.exchange, self.tokenizer = session, exchange, tokenizer
        self.first_batch_index = session.next_batch_index
        self.rows, self.failure = {}, None
        self.stamps: dict[str, int] = {}
        self.cached: dict[str, int | None] = {}
        # ``live`` sessions open at once under pause order; without it the client's concurrency is the limit.
        self.rounds = _Arrival(live or load, session.clock, load if live else None)
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
            ids, digest, place = _prompt_ids(raw)
            place = _request_place(body["messages"], place, pause_order=self.rounds.limit is not None)
            count = body.get("max_tokens", body.get("max_completion_tokens"))
            self.rows[external_id] = None
            stamp, ordinal = await self.rounds.hold(external_id, place)
            self.stamps[external_id] = self._ns(stamp)
            index = self.first_batch_index + ordinal
            rank = self.placement.acquire(place[0])
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

            prefix, cache = "", {}
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
                while not progress.empty():  # the last boundary carries the engine's cached-token report
                    cache.update(_cache_report(progress.get_nowait()))
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
            self.rows[external_id], self.cached[external_id] = row, cache.get("cached_tokens")
            self.raw.write(json.dumps({
                "x_request_id": external_id, "batch_index": index,
                "input_sha256": digest, "output_ids": list(prompt.output_ids), "requested_dp_rank": rank,
                "request_start_ns": self._ns(row.request_started_at),
                "first_token_ns": self._ns(row.request_started_at + row.prompt_latencies[0][0]),
                "request_end_ns": self._ns(row.response_completed_at),
                **usage, **cache,
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
            bridge.cached[external_id],
        ))
        ordinals[key] += 1
    if seen != set(bridge.rows):
        raise ServiceEvidenceError("bridge has requests absent from the retained AIPerf export")
    read = LoadRead(plan.arm, plan.window, plan.lane, load, tuple(records))
    rate = fixed_work_rate(read, plan.slice.expected_work(plan.sessions_at(load)))
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
            "--concurrency", str(plan.sessions_at(load) if plan.pause_order else load),
            "--dataset-sampling-strategy", "sequential",
            "--cache-bust", "first_turn_prefix", "--ignore-trace-delays", "--use-server-token-count",
            "--random-seed", str(current.rules["seed"]), "--ui", "none",
            # AIPerf 0.13.0 collectors are on by default and unread; one missing two heartbeats exits 1 after the work
            # completed and cost the whole qualification (GLM PD prime 2026-11-13 19:22Z, timed window 11-14 06:26Z).
            "--no-server-metrics", "--no-gpu-telemetry",
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
    # `pending` stays empty; the guard reads the requests received (GLM-5.3-Flash PD, three loads, 2026-11-12).
    seen: dict[str, bytes] = {}
    # Short paths: AIPerf's Unix sockets live under both (see _run_load_read).
    work = Path(tempfile.mkdtemp(prefix="cacheon-prime-", dir="/tmp"))
    ipc = Path(tempfile.mkdtemp(prefix="cacheon-aiperf-", dir="/tmp"))
    with ProcessPoolExecutor(_TOKENIZE_WORKERS, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_load_pool_tokenizer, initargs=(str(plan.tokenizer_path),)) as pool:

        async def chat(http_request):
            raw = await http_request.read()
            body = json.loads(raw)
            key = seen[http_request.headers["X-Request-ID"]] = hashlib.sha256(raw).digest()
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
                try:
                    code = await asyncio.wait_for(client.wait(), _PRIME_CLIENT_S)
                except asyncio.TimeoutError:
                    raise OuterSessionTimeoutError(f"AIPerf did not finish priming the replay; see {work / 'aiperf.log'}") from None
            if code:
                raise OuterSessionInfrastructureError(f"AIPerf exited {code} priming the replay; see {work / 'aiperf.log'}")
            if not seen:
                raise OuterSessionInfrastructureError(f"AIPerf sent no request priming the replay; see {work / 'aiperf.log'}")
            places = _places(current, work / "aiperf" / "profile_export.jsonl", seen)
            for key, future in pending.items():
                ids, digest = await future
                _PROMPTS[key] = (_shared(ids), digest, places[key])
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
        bridge = ReplayBridge(session, exchange, tokenizer, output, load,
                              plan.sessions_at(load) if plan.pause_order else 0)
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
                    raise OuterSessionTimeoutError("AIPerf exceeded the engine session deadline")
            if bridge.failure is not None:
                raise bridge.failure
            code = waiters[0].result()
            if code:
                raise OuterSessionInfrastructureError(f"AIPerf exited {code}; see {output / 'aiperf.log'}")
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
    """Execute the sealed windows of each load, highest first, flushing cache before each top-load window's read.

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
    # A lower load replays the first sessions of the top load's pool, so it keeps their prefixes in the device and
    # host tiers: flushing before it made each lower load cold-prefill the same sessions again, 375 of 533 s at 160
    # and 196 of 337 s at 96 on GLM-5.3-Flash PD, whose HiCache holds them (2026-11-12). Both arms run the same rule.
    for load in plan.loads:
        directory = plan.output_directory / ("" if load == top else f"load{load}")
        for window in range(1, plan.windows + 1):
            if load == top:
                await flush_cache(session)
            read_plan = replace(plan, window=window, output_directory=directory / f"window{window}")
            read = await _run_load_read(session, read_plan, load, tokenizer=tokenizer)
            session.replay_reads.append(read)
            reads.append(read)
    result = {"workload": plan.workload_identity(), "reads": [asdict(read) for read in reads]}
    (plan.output_directory / "window.json").write_text(json.dumps(result, indent=2) + "\n")
    return tuple(reads)
