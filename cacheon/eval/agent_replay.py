"""Run a sealed AIPerf slice through the existing isolated-engine session.

The validator owns this loopback HTTP adapter, chat template and timestamps.
The candidate still sees only disclosed generation inputs over its OCI pipes.
AIPerf owns trajectory scheduling; no second session scheduler lives here.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import secrets
import signal
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import UUID

from cacheon.eval.agent_slice import SliceManifest, load_slice_manifest
from cacheon.eval.oci_session_protocol import BatchRequest, MAX_BATCH_REQUEST_BYTES
from cacheon.eval.service_capacity import (
    LoadRead, ServiceContract, ServiceEvidenceError, TurnRecord, attainment, fixed_work_rate,
)


@dataclass(frozen=True)
class AgentReplayPlan:
    """Validator-local inputs for one finite load read, separate from candidate inputs."""

    manifest_path: Path
    load: int
    aiperf_binary: Path
    tokenizer_path: Path
    output_directory: Path
    contract: ServiceContract
    arm: str
    window: int
    lane: str
    ramp_duration_s: float = 0.0
    slice: SliceManifest = field(init=False, repr=False)

    def __post_init__(self):
        manifest = load_slice_manifest(self.manifest_path)
        manifest.expected_work(self.load)
        if manifest.loader != "weka_trace":
            raise ValueError("agent replay requires the sealed Weka loader")
        if self.arm not in ("incumbent", "candidate") or not self.lane or self.window < 1:
            raise ValueError("agent replay read identity is invalid")
        if not math.isfinite(self.ramp_duration_s) or self.ramp_duration_s < 0:
            raise ValueError("agent replay ramp must be finite and non-negative")
        object.__setattr__(self, "slice", manifest)

    def workload_identity(self) -> dict:
        """Bind the consumed bytes, root accounting and replay policy, not local paths."""
        return {
            "slice": self.slice.digest, "dataset": self.slice.dataset,
            "revision": self.slice.revision,
            "rules_json": json.dumps(self.slice.rules, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "expected_work": self.slice.expected_work(self.load), "load": self.load,
            "ramp_duration_s": format(self.ramp_duration_s, ".17g"),
            "contract": {key: format(value, ".17g") for key, value in asdict(self.contract).items()},
            "client": "aiperf-0.13.0", "idle_cap_s": 0,
        }


def _rank(messages: list[dict], ranks: int) -> int:
    # Preserve the working profitability proxy's affinity rule, including its
    # first-user prefix. AIPerf's per-session cache buster is inside this opening.
    head = [("system", json.dumps(m.get("content"), sort_keys=True)[:4096])
            for m in messages if m.get("role") == "system"]
    for message in messages:
        if message.get("role") == "user":
            head.append(("user", json.dumps(message.get("content"), sort_keys=True)[:4096]))
            break
    return int.from_bytes(hashlib.sha256(json.dumps(head).encode()).digest()[:8], "big") % ranks


class ReplayBridge:
    """Adapt trusted chat requests to canonical IDs and retain their actual token evidence."""

    def __init__(self, session, exchange, tokenizer, output: Path):
        self.session, self.exchange, self.tokenizer = session, exchange, tokenizer
        self.rows, self.failure = {}, None
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

        try:
            body = await http_request.json()
            external_id = http_request.headers["X-Request-ID"]
            request_id = UUID(external_id).hex
            if external_id in self.rows:
                raise ValueError("AIPerf repeated an X-Request-ID")
            if body.get("stream") is not True:
                raise ValueError("agent replay requires streaming requests")
            ids = self.tokenizer.apply_chat_template(
                body["messages"], tools=body.get("tools"), tokenize=True,
                add_generation_prompt=True,
            )
            count = body.get("max_tokens", body.get("max_completion_tokens"))
            index = self.session.plan.warmup_count + len(self.rows)
            request = BatchRequest(
                self.session.session_id, self.session.plan.launch_digest,
                request_id, secrets.token_hex(16), index, (), count, 0,
                self.session.plan.temperature, len(ids), True, (tuple(ids),),
                _rank(body["messages"], self.session.plan.engine_config.engine_kwargs.get("dp_size", 1)),
            )
            if index == self.session.plan.warmup_count and self.session.boundary_callback:
                self.session.boundary_callback("before_first_timed", index, self.session.deadline)
            self.rows[external_id] = None
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
            root, kind, ordinals[key], meta["credit_issued_ns"], bridge._ns(row.request_started_at),
            bridge._ns(row.request_started_at + row.prompt_latencies[0][0]),
            bridge._ns(row.response_completed_at), prompt.prompt_tokens, len(prompt.output_ids), "ok",
        ))
        ordinals[key] += 1
    if seen != set(bridge.rows):
        raise ServiceEvidenceError("bridge has requests absent from the retained AIPerf export")
    read = LoadRead(plan.arm, plan.window, plan.lane, plan.load, tuple(records))
    rate = fixed_work_rate(read, plan.slice.expected_work(plan.load))
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


async def run_replay(session, plan: AgentReplayPlan, *, tokenizer=None) -> None:
    """Drain one sealed load on the opened engine and retain the scorer's input."""
    from aiohttp import web

    current = load_slice_manifest(plan.manifest_path)
    if current != plan.slice:
        raise ValueError("replay manifest changed after planning")
    output = plan.output_directory
    output.mkdir(parents=True, exist_ok=False)
    pool = output / "slice"
    pool.mkdir()
    for source in sorted(current.directory.glob("*.json"))[:plan.load]:
        (pool / source.name).symlink_to(source.resolve())
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plan.tokenizer_path, trust_remote_code=True, local_files_only=True)
    async with session.exchange() as exchange:
        bridge = ReplayBridge(session, exchange, tokenizer, output)
        app = web.Application(client_max_size=MAX_BATCH_REQUEST_BYTES)
        app.router.add_post("/v1/chat/completions", bridge.chat)
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        client = None
        waiters = []
        try:
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            argv = [str(plan.aiperf_binary), "profile", "--model", str(plan.tokenizer_path),
                    "--tokenizer", str(plan.tokenizer_path), "--tokenizer-trust-remote-code",
                    "--url", f"http://127.0.0.1:{port}", "--endpoint-type", "chat", "--streaming",
                    "--scenario", "inferencex-agentx-mvp", "--input-file", str(pool),
                    "--custom-dataset-type", "weka_trace", "--num-conversations", str(plan.load),
                    "--concurrency", str(plan.load), "--dataset-sampling-strategy", "sequential",
                    "--trajectory-start-min-ratio", "0", "--trajectory-start-max-ratio", "0",
                    "--cache-bust", "first_turn_prefix", "--system-idle-gap-cap-seconds", "0",
                    "--unsafe-override", "--use-server-token-count",
                    "--random-seed", str(current.rules["seed"]), "--ui", "none",
                    "--output-artifact-dir", str(output / "aiperf")]
            context = session.plan.engine_config.engine_kwargs.get("context_length")
            if context is not None:
                argv.extend(("--max-context-length", str(context)))
            if plan.ramp_duration_s:
                argv.extend(("--concurrency-ramp-duration", str(plan.ramp_duration_s)))
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
            collect_read(plan, bridge)
            rows = sorted(bridge.rows.values(), key=lambda row: row.batch_index)
            session.batch_rows.extend(rows)
            session.first_timed_completed_at = min(row.response_completed_at for row in rows)
            session.last_host_time = max(row.response_completed_at for row in rows)
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
