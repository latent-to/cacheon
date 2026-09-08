"""Streaming token boundaries and their host-observed delivery latencies.

The worker sends token facts, never timestamps. Only the controller clock may
measure TTFT and time after the first delivered token. Both observations still
include serving, scheduling and transport overhead; neither is a GPU timer.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from cacheon.eval.oci_session_protocol import (
    MAX_CONTROL_BYTES,
    SESSION_SCHEMA,
    BatchEvidence,
    BatchRequest,
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


def generate_outputs(
    engine: object, request: BatchRequest, emit: Callable[[bytes], None] | None
) -> object:
    """Run one generation, preserving final outputs in the original prompt order."""
    generate = getattr(engine, "generate", None)
    if not callable(generate):
        raise SessionProtocolError("engine does not expose generate()")
    kwargs = {
        "prompt": list(request.prompts),
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
        return generate(**kwargs)
    if emit is None:
        raise SessionProtocolError("phase measurement lacks its output transport")
    first: dict[int, int] = {}
    final: dict[int, dict] = {}
    for row in generate(**kwargs, stream=True):
        if type(row) is not dict:
            raise SessionProtocolError("stream output is not an object")
        index, meta = row.get("index"), row.get("meta_info")
        if (
            type(index) is not int
            or not 0 <= index < len(request.prompts)
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
        if (
            len(ids) != meta.get("completion_tokens")
            or len(ids) > request.max_new_tokens
        ):
            raise SessionProtocolError("stream output is not cumulative token evidence")
        if index in final:
            raise SessionProtocolError("stream repeated a completed prompt")
        if index not in first:
            first[index] = ids[0]
            emit(token_boundary(request, index, "first", ids[0]))
        elif ids[0] != first[index]:
            raise SessionProtocolError("stream changed its first token")
        if meta.get("finish_reason") is not None:
            if len(ids) != request.max_new_tokens:
                raise SessionProtocolError(
                    "stream completed before its exact token budget"
                )
            final[index] = row
            emit(token_boundary(request, index, "last", ids[-1]))
    if len(final) != len(request.prompts):
        raise SessionProtocolError("stream ended without every prompt's final output")
    return [final[index] for index in range(len(request.prompts))]


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
            or not 0 <= index < len(self.request.prompts)
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
            if not 0 < first < last <= completed - self.started:
                raise SessionProtocolError(
                    "phase measurement lacks positive host timing spans"
                )
            rows.append((first, last))
        return tuple(rows)


def parse_batch_request(message: object, *, fields: frozenset[str]) -> BatchRequest:
    """Extend the existing closed request shape only for explicit phase measurement."""
    from cacheon.eval.oci_session_protocol import _exact_object

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
    return BatchRequest(
        row["session_id"],
        row["launch_digest"],
        row["request_id"],
        row["nonce"],
        row["batch_index"],
        tuple(prompts),
        row["max_new_tokens"],
        row["top_logprobs_num"],
        row["temperature"],
        row["expected_prompt_tokens"],
        row.get("measure_phase_latency", False),
    )  # type: ignore[arg-type]


__all__ = [
    "HostTokenClock",
    "generate_outputs",
    "token_boundary",
    "parse_batch_request",
]
