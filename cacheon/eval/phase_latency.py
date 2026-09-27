"""Streaming token boundaries and their host-observed delivery latencies.

The worker sends token facts, never timestamps. Only the controller clock may
measure TTFT and time after the first delivered token. Both observations still
include serving, scheduling and transport overhead; neither is a GPU timer.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable

from cacheon.eval.oci_session_protocol import (
    MAX_CONTROL_BYTES,
    SESSION_SCHEMA,
    BatchEvidence,
    BatchRequest,
    PromptEvidence,
    SessionProtocolError,
    frame_message,
)

_BINDING = ("session_id", "launch_digest", "request_id", "nonce", "batch_index")
_FIELDS = frozenset(
    (*_BINDING, "schema", "type", "prompt_index", "boundary", "token_id")
)


def token_boundary(
    request: BatchRequest, index: int, boundary: str, token: int
) -> bytes:
    """Frame one token boundary under the existing batch identity."""
    return frame_message(
        {
            **{key: getattr(request, key) for key in _BINDING},
            "schema": SESSION_SCHEMA,
            "type": "token_boundary",
            "prompt_index": index,
            "boundary": boundary,
            "token_id": token,
        },
        max_bytes=MAX_CONTROL_BYTES,
    )


async def generate_outputs(
    engine: object, request: BatchRequest, emit: Callable[[bytes], Awaitable[None]] | None
) -> object:
    """Run one generation, preserving final outputs in the original prompt order."""
    generate = getattr(engine, "async_generate", None)
    if not callable(generate):
        raise SessionProtocolError("engine does not expose async_generate()")
    kwargs = {
        **({"input_ids": [list(row) for row in request.input_ids]}
           if request.input_ids else {"prompt": list(request.prompts)}),
        **({"routed_dp_rank": request.routed_dp_rank} if request.routed_dp_rank is not None else {}),
        "sampling_params": {
            "temperature": request.temperature,
            "max_new_tokens": request.max_new_tokens,
            "ignore_eos": True,
        },
        # Width zero disables the engine's logprob gather entirely so no
        # eval-side CPU work shares the clock with a timed read.
        "return_logprob": request.top_logprobs_num > 0,
        "logprob_start_len": -1,
        "top_logprobs_num": request.top_logprobs_num,
    }
    if not request.measure_phase_latency:
        return await generate(**kwargs)
    if emit is None:
        raise SessionProtocolError("phase measurement lacks its output transport")
    first: dict[int, int] = {}
    final: dict[int, dict] = {}
    async for row in await generate(**kwargs, stream=True):
        if type(row) is not dict:
            raise SessionProtocolError("stream output is not an object")
        index, meta = row.get("index"), row.get("meta_info")
        if (
            type(index) is not int
            or not 0 <= index < request.prompt_count
            or type(meta) is not dict
        ):
            raise SessionProtocolError(
                "stream output lacks its prompt index or metadata"
            )
        ids = row.get("output_ids") or meta.get("output_ids")
        if (
            not isinstance(ids, (tuple, list))
            or not ids
            or type(ids[0]) is not int
            or type(ids[-1]) is not int
        ):
            raise SessionProtocolError("stream output lacks token IDs")
        reported = meta.get("completion_tokens")
        finished = meta.get("finish_reason") is not None
        # SGLang 0.5.20 shares its live ID list on intermediate chunks. The
        # batch fan-in can deliver that list after it has grown beyond the
        # chunk's usage snapshot. Final chunks carry their own copied IDs.
        if (type(reported) is not int or not 1 <= reported <= len(ids) <= request.max_new_tokens
                or (finished and reported != len(ids))):
            raise SessionProtocolError(
                f"stream output is not cumulative token evidence: ids={len(ids)} "
                f"reported={reported if type(reported) is int else None} "
                f"budget={request.max_new_tokens} finished={finished}"
            )
        if index in final:
            raise SessionProtocolError("stream repeated a completed prompt")
        if index not in first:
            first[index] = ids[0]
            await emit(token_boundary(request, index, "first", ids[0]))
        elif ids[0] != first[index]:
            raise SessionProtocolError("stream changed its first token")
        if finished:
            if len(ids) != request.max_new_tokens:
                raise SessionProtocolError(
                    "stream completed before its exact token budget"
                )
            final[index] = row
            await emit(token_boundary(request, index, "last", ids[-1]))
    if len(final) != request.prompt_count:
        raise SessionProtocolError("stream ended without every prompt's final output")
    return [final[index] for index in range(request.prompt_count)]


def engine_outputs(outputs: object, *, request: BatchRequest) -> BatchEvidence:
    """Project engine output onto the existing token-only evidence contract."""
    if isinstance(outputs, dict):
        rows = [outputs]
    elif isinstance(outputs, list):
        rows = outputs
    else:
        raise SessionProtocolError("engine output must be an object or array")
    if len(rows) != request.prompt_count:
        raise SessionProtocolError("engine output prompt count is invalid")
    prompts: list[PromptEvidence] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SessionProtocolError("engine output item is not an object")
        metadata = row.get("meta_info")
        if not isinstance(metadata, dict):
            raise SessionProtocolError("engine output metadata is missing")
        # The engine's own prompt token count is the only input-length
        # authority; a missing count is an infrastructure fault, never a
        # candidate verdict.
        prompt_tokens = metadata.get("prompt_tokens")
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise SessionProtocolError("engine output lacks its prompt token count")
        raw_ids = row.get("output_ids") or metadata.get("output_ids")
        raw_topk = metadata.get("output_top_logprobs")
        if (
            raw_topk is None
            and request.top_logprobs_num == 0
            and isinstance(raw_ids, (list, tuple))
        ):
            # A pure-generation read runs the engine with the logprob path
            # disabled; the evidence still carries exact empty positions.
            raw_topk = [()] * len(raw_ids)
        if not isinstance(raw_ids, (list, tuple)) or not isinstance(
            raw_topk, (list, tuple)
        ):
            raise SessionProtocolError("engine output lacks token/top-k evidence")
        # The binary evidence encoder already validates token IDs, exact lengths,
        # logprob values and probability mass. Strip detokenized text here; do not
        # maintain a second numerical validator ahead of that same boundary.
        if any(
            not isinstance(position, (list, tuple))
            or any(not isinstance(entry, (list, tuple)) or len(entry) < 2 for entry in position)
            for position in raw_topk
        ):
            raise SessionProtocolError("engine output top-k entry is malformed")
        prompts.append(PromptEvidence(
            tuple(raw_ids),
            tuple(tuple((entry[0], entry[1]) for entry in position) for position in raw_topk),
            prompt_tokens,
        ))
    return BatchEvidence(tuple(prompts))


class HostTokenClock:
    """Timestamp request-bound output events and check them against final tokens."""

    def __init__(
        self, request: BatchRequest, clock: Callable[[], float], started: float
    ):
        self.request, self.clock, self.started = request, clock, started
        self.previous = started
        self.events: dict[tuple[int, str], tuple[float, int]] = {}

    def observe(self, message: dict) -> None:
        """Accept one first/final token event in this batch's host clock domain."""
        now = float(self.clock())
        if not math.isfinite(now) or now < self.previous:
            raise SessionProtocolError("host token clock moved backwards")
        if (
            type(message) is not dict
            or set(message) != _FIELDS
            or message["schema"] != SESSION_SCHEMA
            or message["type"] != "token_boundary"
            or any(
                type(message[key]) is not type(getattr(self.request, key))
                or message[key] != getattr(self.request, key)
                for key in _BINDING
            )
        ):
            raise SessionProtocolError(
                "token boundary differs from the disclosed request"
            )
        index, boundary = message["prompt_index"], message["boundary"]
        if (
            type(index) is not int
            or not 0 <= index < self.request.prompt_count
            or boundary not in ("first", "last")
            or type(message["token_id"]) is not int
        ):
            raise SessionProtocolError("token boundary is malformed")
        key = index, boundary
        if key in self.events or (
            boundary == "last" and (index, "first") not in self.events
        ):
            raise SessionProtocolError("token boundary is duplicate or out of order")
        self.events[key] = (now - self.started, message["token_id"])
        self.previous = now

    def finish(
        self, evidence: BatchEvidence, completed: float
    ) -> tuple[tuple[float, float], ...]:
        """Return each prompt's relative delivery times after exact output validation."""
        if len(self.events) != 2 * len(evidence.prompts):
            raise SessionProtocolError("phase measurement lacks token boundaries")
        rows = []
        for index, prompt in enumerate(evidence.prompts):
            first, token = self.events[index, "first"]
            last, final_token = self.events[index, "last"]
            if token != prompt.output_ids[0] or final_token != prompt.output_ids[-1]:
                raise SessionProtocolError(
                    "token boundaries disagree with final evidence"
                )
            if not 0 < first <= last <= completed - self.started or (
                first == last and len(prompt.output_ids) > 1
            ):
                raise SessionProtocolError(
                    "phase measurement lacks positive host timing spans"
                )
            rows.append((first, last))
        return tuple(rows)


def parse_batch_request(message: object, *, fields: frozenset[str]) -> BatchRequest:
    """Extend the existing closed request shape only for explicit phase measurement."""
    from cacheon.eval.oci_session_protocol import _exact_object

    if isinstance(message, dict):
        fields = fields | (message.keys() & {"input_ids", "routed_dp_rank"})
        if "input_ids" in message and not message["input_ids"]:
            raise SessionProtocolError("empty input_ids must be omitted")
        if "routed_dp_rank" in message and message["routed_dp_rank"] is None:
            raise SessionProtocolError("null routed_dp_rank must be omitted")
    if isinstance(message, dict) and "measure_phase_latency" in message:
        fields = fields | {"measure_phase_latency"}
        if message["measure_phase_latency"] is not True:
            raise SessionProtocolError("phase measurement field must be canonical true")
    row = _exact_object(message, fields=fields, label="batch request")
    if row["schema"] != SESSION_SCHEMA or row["type"] != "batch_request":
        raise SessionProtocolError("batch request schema/type mismatch")
    prompts = row["prompts"]
    if not isinstance(prompts, list):
        raise SessionProtocolError("batch request prompts must be an array")
    return BatchRequest(**{
        key: value for key, value in row.items() if key not in {"schema", "type"}
    })



__all__ = [
    "HostTokenClock",
    "generate_outputs",
    "token_boundary",
    "parse_batch_request",
]
