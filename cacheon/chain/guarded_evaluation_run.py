"""Typed adapter result that exposes an evaluation's exact remote request ID."""

from __future__ import annotations

from dataclasses import dataclass

from cacheon.chain.evaluation_coordinator import EvaluationRun
from cacheon.chain.remote_qualification_evidence import (
    RemoteEvaluationDispatcherError,
)
from cacheon.stack_identity import require_sha256_hex


@dataclass(frozen=True)
class GuardedEvaluationRun:
    """One exact completed run paired with the request that produced it."""

    request_id: str
    run: EvaluationRun

    def __post_init__(self) -> None:
        try:
            request_id = require_sha256_hex(
                self.request_id,
                field="guarded request id",
            )
        except (TypeError, ValueError) as exc:
            raise RemoteEvaluationDispatcherError(str(exc)) from None
        if type(self.run) is not EvaluationRun:
            raise RemoteEvaluationDispatcherError(
                "guarded evaluation run is not exact"
            )
        object.__setattr__(self, "request_id", request_id)


__all__ = ["GuardedEvaluationRun"]
