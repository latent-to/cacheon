"""Typed execution disposition, separate from any miner decision.

``COMPLETE`` / ``REQUEUE`` / ``HOLD`` describe what CPU orchestration may do
with one qualification request.  ``PASS`` / ``FAIL`` / ``NO_DECISION``
describe what an evaluation said about miner work.  The two never share a
field, and no rule here maps an infrastructure failure onto a miner ``FAIL``.

``REQUEUE`` exists for exactly one shape of evidence: an authenticated,
closed, pre-resident refusal that the pod signs with the shared worker
credential only after proving the resident-entry marker is absent
(``remote_worker_pod_service.require_pre_resident_failure``).  Unauthenticated
failure results, unknown failure codes, ambiguous markers, and completed
evidence with retained ``NO_DECISION`` outcomes never requeue.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from cacheon.chain.evaluation_recovery import (
    WORKER_PRE_RESIDENT_REASON_PREFIX,
    WORKER_PRE_RESIDENT_RELEASE_REASONS,
)
from cacheon.chain.remote_evaluation_dispatcher import RemoteWorkerCredential
from cacheon.stack_identity import canonical_digest, require_sha256_hex


SCHEMA_PRE_RESIDENT_REFUSAL = "cacheon-pre-resident-refusal-v1"
_REFUSAL_DOMAIN = b"cacheon.chain.pre-resident-refusal.v1"
_REFUSAL_FIELDS = frozenset(
    {
        "auth_tag",
        "credential_id",
        "failure_code",
        "marker",
        "request_id",
        "schema",
        "state",
        "worker_epoch",
    }
)

# The closed adapter-protocol refusals a pod may emit before resident entry.
# Must stay derivable from the durable release reasons the recovery event
# contract accepts; ``worker_pre_resident_release_reason`` binds the two.
PRE_RESIDENT_REQUEUE_FAILURES = frozenset(
    {"adapter_request_failed", "adapter_start_failed"}
)

_ALLOWED_DECISIONS = frozenset({"", "PASS", "FAIL", "NO_DECISION"})


class ExecutionDispositionError(ValueError):
    """A typed disposition, outcome, or refusal proof is malformed."""


class ExecutionDisposition(str, Enum):
    COMPLETE = "complete"
    REQUEUE = "requeue"
    HOLD = "hold"


def worker_pre_resident_release_reason(failure_code: str) -> str:
    """Return the one durable release reason for a closed pre-resident code."""

    if failure_code not in PRE_RESIDENT_REQUEUE_FAILURES:
        raise ExecutionDispositionError(
            "pre-resident release reason requires a closed refusal code"
        )
    reason = WORKER_PRE_RESIDENT_REASON_PREFIX + failure_code
    if reason not in WORKER_PRE_RESIDENT_RELEASE_REASONS:
        raise ExecutionDispositionError(
            "pre-resident refusal codes differ from durable release reasons"
        )
    return reason


def _refusal_tag(credential: RemoteWorkerCredential, digest: str) -> str:
    return hmac.new(
        credential.secret,
        _REFUSAL_DOMAIN + b"\0" + digest.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()


@dataclass(frozen=True)
class AuthenticatedPreResidentRefusal:
    """Pod-signed proof one request failed closed before resident entry."""

    request_id: str
    failure_code: str
    worker_epoch: str
    credential_id: str
    auth_tag: str

    def __post_init__(self) -> None:
        try:
            require_sha256_hex(self.request_id, field="refusal request id")
        except (TypeError, ValueError) as exc:
            raise ExecutionDispositionError(str(exc)) from None
        if (
            self.failure_code not in PRE_RESIDENT_REQUEUE_FAILURES
            or type(self.worker_epoch) is not str
            or len(self.worker_epoch) != 32
            or any(char not in "0123456789abcdef" for char in self.worker_epoch)
            or type(self.credential_id) is not str
            or not 1 <= len(self.credential_id) <= 128
            or type(self.auth_tag) is not str
            or len(self.auth_tag) != 64
            or any(char not in "0123456789abcdef" for char in self.auth_tag)
        ):
            raise ExecutionDispositionError("pre-resident refusal is malformed")

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "credential_id": self.credential_id,
            "failure_code": self.failure_code,
            "marker": "absent",
            "request_id": self.request_id,
            "schema": SCHEMA_PRE_RESIDENT_REFUSAL,
            "state": "no_decision",
            "worker_epoch": self.worker_epoch,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(_REFUSAL_DOMAIN.decode("ascii"), self._unsigned_dict())

    @property
    def release_reason(self) -> str:
        return worker_pre_resident_release_reason(self.failure_code)

    def to_payload(self) -> dict[str, object]:
        return {**self._unsigned_dict(), "auth_tag": self.auth_tag}


def seal_pre_resident_refusal(
    request: Mapping[str, Any],
    failure_code: str,
    credential: RemoteWorkerCredential,
) -> AuthenticatedPreResidentRefusal:
    """Sign one closed pre-resident refusal for one verified spool request."""

    if type(credential) is not RemoteWorkerCredential:
        raise ExecutionDispositionError("refusal credential is not exactly typed")
    unsigned = AuthenticatedPreResidentRefusal(
        request_id=str(request.get("request_id", "")),
        failure_code=failure_code,
        worker_epoch=str(request.get("worker_epoch", "")),
        credential_id=credential.credential_id,
        auth_tag="0" * 64,
    )
    return AuthenticatedPreResidentRefusal(
        request_id=unsigned.request_id,
        failure_code=unsigned.failure_code,
        worker_epoch=unsigned.worker_epoch,
        credential_id=unsigned.credential_id,
        auth_tag=_refusal_tag(credential, unsigned.digest),
    )


def reopen_pre_resident_refusal(
    payload: object,
    *,
    request_id: str,
    worker_epoch: str,
    credential: RemoteWorkerCredential,
) -> AuthenticatedPreResidentRefusal:
    """Authenticate one refusal payload against its exact request identity."""

    if type(credential) is not RemoteWorkerCredential:
        raise ExecutionDispositionError("refusal credential is not exactly typed")
    if (
        type(payload) is not dict
        or set(payload) != _REFUSAL_FIELDS
        or payload.get("schema") != SCHEMA_PRE_RESIDENT_REFUSAL
        or payload.get("state") != "no_decision"
        or payload.get("marker") != "absent"
    ):
        raise ExecutionDispositionError("pre-resident refusal payload is not closed")
    refusal = AuthenticatedPreResidentRefusal(
        request_id=payload["request_id"],
        failure_code=payload["failure_code"],
        worker_epoch=payload["worker_epoch"],
        credential_id=payload["credential_id"],
        auth_tag=payload["auth_tag"],
    )
    if (
        refusal.request_id != request_id
        or refusal.worker_epoch != worker_epoch
        or refusal.credential_id != credential.credential_id
        or not hmac.compare_digest(
            refusal.auth_tag, _refusal_tag(credential, refusal.digest)
        )
    ):
        raise ExecutionDispositionError(
            "pre-resident refusal does not authenticate for this request"
        )
    return refusal


def infrastructure_result_payload(
    request: Mapping[str, Any],
    failure_code: str,
    credential: RemoteWorkerCredential | None,
) -> dict[str, object]:
    """Payload for one no-decision result; authenticated when pre-resident."""

    if credential is not None and failure_code in PRE_RESIDENT_REQUEUE_FAILURES:
        return seal_pre_resident_refusal(
            request, failure_code, credential
        ).to_payload()
    return {
        "failure_code": failure_code,
        "request_id": request["request_id"],
        "state": "no_decision",
    }


@dataclass(frozen=True)
class ExecutionOutcome:
    """One typed pairing of miner decision and CPU execution disposition."""

    disposition: ExecutionDisposition
    decision: str = ""
    failure_code: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.disposition) is not ExecutionDisposition
            or self.decision not in _ALLOWED_DECISIONS
            or type(self.failure_code) is not str
            or type(self.reason) is not str
            or len(self.reason) > 2_048
        ):
            raise ExecutionDispositionError("execution outcome is malformed")
        if self.failure_code and self.decision != "NO_DECISION":
            raise ExecutionDispositionError(
                "infrastructure failure never becomes a miner decision"
            )
        if self.disposition is ExecutionDisposition.REQUEUE and (
            (
                self.failure_code not in PRE_RESIDENT_REQUEUE_FAILURES
                and self.failure_code != WORKER_INFRASTRUCTURE_REQUEUE_FAILURE
            )
            or self.decision != "NO_DECISION"
        ):
            raise ExecutionDispositionError(
                "REQUEUE requires a closed NO_DECISION infrastructure failure"
            )
        if self.disposition is ExecutionDisposition.HOLD and not self.reason:
            raise ExecutionDispositionError("HOLD requires a durable reason")
        if self.disposition is ExecutionDisposition.COMPLETE and (
            self.failure_code or self.reason
        ):
            raise ExecutionDispositionError("COMPLETE carries no failure or reason")


# Closed umbrella failure code for a worker-terminated request with no
# authenticated refusal and no completed response: requeue-class (owner ruling
# 2026-08-10). Raw worker codes outside the closed vocabularies normalize to
# this marker in the typed outcome; the store release reason keeps the raw code.
WORKER_INFRASTRUCTURE_REQUEUE_FAILURE = "worker_infrastructure_result"

# Durable HELD reason written by the qualification dispatcher before
# infrastructure results became requeue-class; the store accepts exactly this
# reason when migrating a parked recovery back into the queue.
WORKER_INFRASTRUCTURE_HOLD_REASON = (
    f"transport_hold:{WORKER_INFRASTRUCTURE_REQUEUE_FAILURE}"
)


def resolve_infrastructure_result(
    failure_code: str | None,
    refusal: AuthenticatedPreResidentRefusal | None,
    *,
    request_id: str,
) -> ExecutionOutcome:
    """Disposition for one result-ready infrastructure failure observation."""

    code = failure_code or ""
    if (
        type(refusal) is AuthenticatedPreResidentRefusal
        and refusal.request_id == request_id
        and refusal.failure_code == code
    ):
        return ExecutionOutcome(
            ExecutionDisposition.REQUEUE,
            decision="NO_DECISION",
            failure_code=code,
        )
    return ExecutionOutcome(
        ExecutionDisposition.HOLD,
        decision="NO_DECISION",
        failure_code=code,
        reason="unproven_infrastructure_failure",
    )


def resolve_completed_result(has_no_decision: bool) -> ExecutionOutcome:
    """Disposition for one completed authenticated product; never REQUEUE."""

    if type(has_no_decision) is not bool:
        raise ExecutionDispositionError("completed resolution input is not boolean")
    if has_no_decision:
        return ExecutionOutcome(
            ExecutionDisposition.HOLD,
            decision="NO_DECISION",
            reason="post_publication_no_decision",
        )
    return ExecutionOutcome(ExecutionDisposition.COMPLETE)


__all__ = [
    "AuthenticatedPreResidentRefusal",
    "WORKER_INFRASTRUCTURE_HOLD_REASON",
    "WORKER_INFRASTRUCTURE_REQUEUE_FAILURE",
    "ExecutionDisposition",
    "ExecutionDispositionError",
    "ExecutionOutcome",
    "PRE_RESIDENT_REQUEUE_FAILURES",
    "SCHEMA_PRE_RESIDENT_REFUSAL",
    "infrastructure_result_payload",
    "reopen_pre_resident_refusal",
    "resolve_completed_result",
    "resolve_infrastructure_result",
    "seal_pre_resident_refusal",
    "worker_pre_resident_release_reason",
]
