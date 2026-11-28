"""Native multi-turn token replay over the isolated engine session exchange.

Replays sealed pre-tokenized real-text coding/agent trajectories directly over
``session.exchange()`` without an external HTTP/AIPerf client, supporting decoupled
``(active_concurrency L, resident_sessions S)``, recorded pause-order arrival, clean
per-load cache flushes, validator-sealed ideal prefix-cache accounting, host-clocked
streaming ITL bounds, and real generated-output continuation (``feedback_turns``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import struct
import time

from cacheon.eval import pd_engine
from cacheon.eval.agent_slice import SliceManifest
from cacheon.eval.oci_session_protocol import BatchRequest
from cacheon.eval.phase_latency import _cache_report
from cacheon.eval.service_capacity import LoadRead, ServiceEvidenceError, TurnRecord


def _lcp_len(left: list[int] | tuple[int, ...], right: list[int] | tuple[int, ...]) -> int:
    limit = min(len(left), len(right))
    i = 0
    while i < limit and left[i] == right[i]:
        i += 1
    return i


def _load_session_turns(current: SliceManifest, sessions: int) -> list[tuple[str, list[dict]]]:
    rows: list[tuple[str, list[dict]]] = []
    for index, session in enumerate(current.sessions[:sessions]):
        path = current.directory / f"{index:03d}_{session.id}.json"
        trace = json.loads(path.read_text(encoding="utf-8"))
        turns = trace.get("turns")
        if type(turns) is not list or len(turns) != session.main_turns or session.inner_requests != 0:
            raise ServiceEvidenceError(f"multi-turn session {session.id!r} differs from manifest turn counts")
        rows.append((session.id, turns))
    return rows


def _turn_prompt(
    turns: list[dict], ordinal: int, prev_prompt: tuple[int, ...] | None,
    prev_out: tuple[int, ...] | None, *, feedback: bool,
) -> tuple[tuple[int, ...], int]:
    """Derive exact token IDs and validator-sealed ideal cached prefix length for one turn."""
    turn = turns[ordinal]
    if ordinal == 0:
        ids = tuple(turn["prompt_ids"])
        return ids, 0
    if feedback:
        if prev_prompt is None or prev_out is None:
            raise ServiceEvidenceError("feedback turn lacks prior turn output")
        if "delta_ids" in turn:
            delta = tuple(turn["delta_ids"])
        else:
            prev_turn = turns[ordinal - 1]
            full = turn["prompt_ids"]
            asst_len = int(prev_turn.get("asst_len", prev_turn["out_len"]))
            start = min(len(full), len(prev_turn["prompt_ids"]) + asst_len)
            delta = tuple(full[start:])
        prefix = prev_prompt + prev_out
        return prefix + delta, len(prefix)
    ids = tuple(turn["prompt_ids"])
    prev_static = turns[ordinal - 1]["prompt_ids"]
    return ids, _lcp_len(prev_static, ids)


def reconstruct_turn_inputs(
    current: SliceManifest, sessions: int, outputs_by_turn: dict[tuple[str, int], tuple[int, ...]],
    *, feedback: bool,
) -> dict[tuple[str, int], tuple[int, ...]]:
    """Deterministically reconstruct every turn's canonical input IDs from the sealed slice and retained outputs."""
    result: dict[tuple[str, int], tuple[int, ...]] = {}
    for root_id, turns in _load_session_turns(current, sessions):
        prev_prompt: tuple[int, ...] | None = None
        prev_out: tuple[int, ...] | None = None
        for ordinal in range(len(turns)):
            ids, _ = _turn_prompt(turns, ordinal, prev_prompt, prev_out, feedback=feedback)
            result[(root_id, ordinal)] = ids
            prev_prompt = ids
            if feedback:
                prev_out = outputs_by_turn[(root_id, ordinal)]
    return result


async def run_multiturn_load_read(session, plan, current: SliceManifest, load: int) -> LoadRead:
    """Execute one decoupled ``(load, sessions_at(load))`` real-text multi-turn window."""
    from cacheon.eval.agent_replay import _Arrival, _Placement, _record_load_read

    output = plan.output_directory
    output.mkdir(parents=True, exist_ok=False)
    nsess = plan.sessions_at(load)
    traces = _load_session_turns(current, nsess)
    offset_ns = time.time_ns() - round(session.clock() * 1e9)
    (output / "clock.json").write_text(json.dumps({
        "clock": "host-monotonic-to-unix-ns", "offset_ns": offset_ns,
    }) + "\n")

    def to_ns(value: float) -> int:
        return offset_ns + round(value * 1e9)

    first_batch_index = session.next_batch_index
    total_out = sum(int(t["out_len"]) for _rid, turns in traces for t in turns)
    stagger_s = (
        0.7 * total_out / max(1.0, load * plan.contract.decode_floor_tps * nsess)
        if plan.pause_order and nsess > load
        else 0.0
    )
    rounds = _Arrival(
        nsess if plan.pause_order else load,
        session.clock,
        load if plan.pause_order else None,
        open_stagger_s=stagger_s,
    )
    placement = _Placement(pd_engine.routes(dict(session.plan.engine_config.engine_kwargs)))
    records: list[TurnRecord] = []
    batch_rows: list[object] = []
    bridge_lines: list[str] = []

    async with session.exchange() as exchange:
        async def run_session_turns(sess_order: int, root_id: str, turns: list[dict]) -> None:
            prev_prompt: tuple[int, ...] | None = None
            prev_out: tuple[int, ...] | None = None
            for ordinal, turn in enumerate(turns):
                out_len = int(turn["out_len"])
                pause_next = (
                    float(turns[ordinal + 1].get("think_time") or 0.0)
                    if ordinal + 1 < len(turns)
                    else None
                )
                ids, ideal_cached = _turn_prompt(
                    turns, ordinal, prev_prompt, prev_out, feedback=plan.feedback_turns,
                )
                req_id = secrets.token_hex(16)
                rank = None
                try:
                    stamp, dispatch_ord = await rounds.hold(req_id, (sess_order, ordinal, pause_next))
                    batch_index = first_batch_index + dispatch_ord
                    rank = placement.acquire(sess_order)
                    request = BatchRequest(
                        session.session_id, session.plan.launch_digest,
                        req_id, secrets.token_hex(16), batch_index, (), out_len, 0,
                        session.plan.temperature, len(ids), True, (ids,), rank,
                    )
                    if batch_index == session.plan.warmup_count and session.boundary_callback:
                        session.boundary_callback("before_first_timed", batch_index, session.deadline)
                    cache: dict[str, object] = {}

                    def on_progress(message: dict) -> None:
                        if message.get("boundary") == "last":
                            cache.update(_cache_report(message))

                    row = await exchange.execute(
                        request,
                        deadline=min(session.deadline, session.clock() + session.batch_timeout_s),
                        on_progress=on_progress,
                    )
                finally:
                    rounds.done(req_id)
                    if rank is not None:
                        placement.release(rank)
                prompt = row.evidence.prompts[0]
                prev_prompt = ids
                prev_out = tuple(prompt.output_ids)
                max_itl_ns = round(row.prompt_max_itl_s[0] * 1e9) if row.prompt_max_itl_s else None
                record = TurnRecord(
                    root_id, "main", ordinal, to_ns(stamp), to_ns(row.request_started_at),
                    to_ns(row.request_started_at + row.prompt_latencies[0][0]),
                    to_ns(row.response_completed_at), prompt.prompt_tokens, len(prompt.output_ids),
                    "ok", cache.get("cached_tokens"), max_itl_ns, ideal_cached,
                )
                records.append(record)
                batch_rows.append(row)
                input_sha = hashlib.sha256(struct.pack(f">{len(ids)}I", *ids)).hexdigest()
                bridge_lines.append(json.dumps({
                    "request_id": req_id, "batch_index": batch_index, "root_session_id": root_id,
                    "outer": ordinal, "input_sha256": input_sha, "output_ids": list(prompt.output_ids),
                    "requested_dp_rank": rank, "request_start_ns": record.request_start_ns,
                    "first_token_ns": record.first_token_ns, "request_end_ns": record.request_end_ns,
                    "max_itl_ns": max_itl_ns, "ideal_cached_tokens": ideal_cached,
                    "prompt_tokens": prompt.prompt_tokens, "completion_tokens": len(prompt.output_ids),
                    **cache,
                }))

        await asyncio.gather(*(
            run_session_turns(order, root_id, turns) for order, (root_id, turns) in enumerate(traces)
        ))

    records.sort(key=lambda r: (r.root_session_id, r.ordinal))
    (output / "bridge.jsonl").write_text("\n".join(bridge_lines) + "\n")
    return _record_load_read(session, plan, current, load, records, batch_rows)


__all__ = ["reconstruct_turn_inputs", "run_multiturn_load_read"]
