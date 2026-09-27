"""Disclosed generation inputs shared by text batches and tokenized chat turns.

Tokenized input avoids a text round trip that can add a second BOS or lose chat
control tokens. The existing protocol keeps its public request type and wire
identity; input and routing options are absent from legacy text-request bytes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def cache_flush_message(*, session_id: str, launch_digest: str, request_id: str,
                        nonce: str, batch_index: int) -> dict[str, object]:
    """Bind a cache boundary to one session and its next undisclosed request."""
    from cacheon.eval.oci_session_protocol import SESSION_SCHEMA, _binding_id, _bounded_int, _digest
    return {
        "schema": SESSION_SCHEMA, "type": "cache_flush",
        "session_id": _binding_id(session_id, field_name="session_id"),
        "launch_digest": _digest(launch_digest, field_name="launch_digest"),
        "request_id": _binding_id(request_id, field_name="request_id"),
        "nonce": _binding_id(nonce, field_name="nonce"),
        "batch_index": _bounded_int(batch_index, field_name="batch_index", minimum=0, maximum=2_147_483_647),
    }


@dataclass(frozen=True)
class BatchRequest:
    """One host-disclosed prompt batch and its exact evidence shape."""

    session_id: str
    launch_digest: str
    request_id: str
    nonce: str
    batch_index: int
    prompts: tuple[str, ...]
    max_new_tokens: int
    top_logprobs_num: int
    temperature: float
    expected_prompt_tokens: int | None = None
    measure_phase_latency: bool = False
    input_ids: tuple[tuple[int, ...], ...] = ()
    routed_dp_rank: int | None = None

    def __post_init__(self) -> None:
        from cacheon.eval.oci_session_protocol import (
            MAX_NEW_TOKENS, MAX_PROMPT_CHARS, MAX_PROMPTS_PER_BATCH,
            MAX_PROMPT_TOKENS, MAX_TOP_LOGPROBS, MAX_TOTAL_PROMPT_CHARS,
            SessionProtocolError, _binding_id, _bounded_float, _bounded_int,
            _digest, expected_evidence_payload_bytes,
        )
        for name in ("session_id", "request_id", "nonce"):
            object.__setattr__(self, name, _binding_id(getattr(self, name), field_name=name))
        if len({self.session_id, self.request_id, self.nonce}) != 3:
            raise SessionProtocolError("session_id, request_id, and nonce must be distinct")
        object.__setattr__(self, "launch_digest", _digest(
            self.launch_digest, field_name="launch_digest"
        ))
        object.__setattr__(self, "batch_index", _bounded_int(
            self.batch_index, field_name="batch_index", minimum=0,
            maximum=2_147_483_647,
        ))
        if (
            isinstance(self.prompts, (str, bytes))
            or not isinstance(self.prompts, Sequence)
            or len(self.prompts) > MAX_PROMPTS_PER_BATCH
        ):
            raise SessionProtocolError("batch prompts count is invalid")
        clean: list[str] = []
        total_chars = 0
        for prompt in self.prompts:
            if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT_CHARS:
                raise SessionProtocolError("batch contains an invalid/oversized prompt")
            total_chars += len(prompt)
            if total_chars > MAX_TOTAL_PROMPT_CHARS:
                raise SessionProtocolError("batch exceeds its total prompt-character bound")
            clean.append(prompt)
        object.__setattr__(self, "prompts", tuple(clean))
        if not clean and not self.input_ids:
            raise SessionProtocolError("batch prompts count is invalid")
        if not isinstance(self.input_ids, (list, tuple)) or bool(clean) == bool(self.input_ids):
            raise SessionProtocolError("request requires exactly one of text prompts or token IDs")
        if self.input_ids:
            if not 1 <= len(self.input_ids) <= MAX_PROMPTS_PER_BATCH or any(
                not isinstance(row, (list, tuple)) or not 1 <= len(row) <= MAX_PROMPT_TOKENS
                or any(type(token) is not int or not 0 <= token <= 2_147_483_647 for token in row)
                for row in self.input_ids
            ):
                raise SessionProtocolError("request input token IDs are malformed")
        object.__setattr__(self, "input_ids", tuple(tuple(row) for row in self.input_ids))
        if self.routed_dp_rank is not None:
            _bounded_int(self.routed_dp_rank, field_name="routed_dp_rank", minimum=0, maximum=2_147_483_647)
        object.__setattr__(self, "max_new_tokens", _bounded_int(
            self.max_new_tokens, field_name="max_new_tokens", minimum=1,
            maximum=MAX_NEW_TOKENS,
        ))
        # Width zero is the pure-generation read: no logprob collection rides
        # the clock and the evidence carries exact empty top-k positions.
        object.__setattr__(self, "top_logprobs_num", _bounded_int(
            self.top_logprobs_num, field_name="top_logprobs_num", minimum=0,
            maximum=MAX_TOP_LOGPROBS,
        ))
        object.__setattr__(self, "temperature", _bounded_float(
            self.temperature, field_name="temperature", minimum=0.0, maximum=100.0
        ))
        if self.expected_prompt_tokens is not None:
            object.__setattr__(self, "expected_prompt_tokens", _bounded_int(
                self.expected_prompt_tokens, field_name="expected_prompt_tokens",
                minimum=1, maximum=MAX_PROMPT_TOKENS,
            ))
        if type(self.measure_phase_latency) is not bool:
            raise SessionProtocolError("phase measurement flag must be boolean")
        expected_evidence_payload_bytes(self)

    @property
    def prompt_count(self) -> int:
        """Count requests independently of whether text or canonical IDs cross the pipe."""
        return len(self.input_ids) if self.input_ids else len(self.prompts)

    def to_dict(self) -> dict[str, object]:
        """Retain old text-request bytes and omit unused input/routing options."""
        from cacheon.eval.oci_session_protocol import SESSION_SCHEMA
        return {
            **({"input_ids": [list(row) for row in self.input_ids]} if self.input_ids else {}),
            **({"routed_dp_rank": self.routed_dp_rank} if self.routed_dp_rank is not None else {}),
            **({"measure_phase_latency": True} if self.measure_phase_latency else {}),
            "schema": SESSION_SCHEMA, "type": "batch_request",
            "session_id": self.session_id, "launch_digest": self.launch_digest,
            "request_id": self.request_id, "nonce": self.nonce,
            "batch_index": self.batch_index, "prompts": list(self.prompts),
            "max_new_tokens": self.max_new_tokens,
            "top_logprobs_num": self.top_logprobs_num,
            "temperature": self.temperature,
            "expected_prompt_tokens": self.expected_prompt_tokens,
        }
