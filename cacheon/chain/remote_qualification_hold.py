"""Path-free, authenticated qualification HOLD products.

A worker emits this product only after an exact qualification request is
resident or otherwise past the point where generic release is safe.  It is a
closed control-plane result, not a miner verdict, evidence artifact, retry
plan, or replacement request.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cacheon.arena_service import ArenaScreenReceipt
from cacheon.chain.remote_qualification_evidence import (
    RemoteEvaluationDispatcherError,
    RemoteQualificationProduct,
)
from cacheon.eval.qualification_intake import QualificationReservation
from cacheon.stack_identity import canonical_digest, require_sha256_hex


_SCHEMA_VERSION = 1
_MAX_COHORT_MEMBERS = 256


class RemoteQualificationHoldReason(str, Enum):
    """Closed worker reasons that preserve the exact qualification request."""

    GRAPH_EVIDENCE_INCOMPLETE = "graph_evidence_incomplete"
    GRAPH_EVIDENCE_UNAVAILABLE = "graph_evidence_unavailable"
    GRAPH_EXIT_PUBLICATION_AMBIGUOUS = "graph_exit_publication_ambiguous"


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RemoteEvaluationDispatcherError(str(exc)) from None


def _digest_tuple(value: object, field: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value or len(value) > _MAX_COHORT_MEMBERS:
        raise RemoteEvaluationDispatcherError(f"{field} are malformed")
    return tuple(_digest(item, field) for item in value)


def _service_identity(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise RemoteEvaluationDispatcherError("service identity is malformed")
    return value


@dataclass(frozen=True)
class RemoteQualificationHoldProduct:
    """One exact HOLD bound to a qualification request and its sealed cohort."""

    request_digest: str
    service_identity: str
    service_digest: str
    worker_readiness_digest: str
    ready_receipt_digest: str
    ready_epoch: int
    screen_lane: str
    reservation_digests: tuple[str, ...]
    selected_delta_digests: tuple[str, ...]
    candidate_digests: tuple[str, ...]
    reason: RemoteQualificationHoldReason
    diagnostic_digest: str = ""
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "request_digest",
            "service_digest",
            "worker_readiness_digest",
            "ready_receipt_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(
            self, "service_identity", _service_identity(self.service_identity)
        )
        reservations = _digest_tuple(
            self.reservation_digests, "reservation digests"
        )
        deltas = _digest_tuple(
            self.selected_delta_digests, "selected delta digests"
        )
        candidates = _digest_tuple(self.candidate_digests, "candidate digests")
        if (
            type(self.ready_epoch) is not int
            or self.ready_epoch < 0
            or self.screen_lane not in {"primary", "reproduction"}
            or type(self.reason) is not RemoteQualificationHoldReason
            or type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
            or not (
                len(reservations) == len(deltas) == len(candidates)
            )
            or len(set(reservations)) != len(reservations)
            or len(set(candidates)) != len(candidates)
            or (self.screen_lane == "reproduction" and len(reservations) != 1)
        ):
            raise RemoteEvaluationDispatcherError(
                "remote qualification HOLD authority is malformed"
            )
        diagnostic = self.diagnostic_digest
        if diagnostic:
            diagnostic = _digest(diagnostic, "diagnostic digest")
        elif type(diagnostic) is not str:
            raise RemoteEvaluationDispatcherError("diagnostic digest is malformed")
        object.__setattr__(self, "reservation_digests", reservations)
        object.__setattr__(self, "selected_delta_digests", deltas)
        object.__setattr__(self, "candidate_digests", candidates)
        object.__setattr__(self, "diagnostic_digest", diagnostic)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_digests": list(self.candidate_digests),
            "diagnostic_digest": self.diagnostic_digest,
            "kind": "remote_qualification_hold",
            "ready_epoch": self.ready_epoch,
            "ready_receipt_digest": self.ready_receipt_digest,
            "reason": self.reason.value,
            "request_digest": self.request_digest,
            "reservation_digests": list(self.reservation_digests),
            "schema_version": self.schema_version,
            "screen_lane": self.screen_lane,
            "selected_delta_digests": list(self.selected_delta_digests),
            "service_digest": self.service_digest,
            "service_identity": self.service_identity,
            "worker_readiness_digest": self.worker_readiness_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.chain.remote-qualification-hold.v1", self.to_dict()
        )


RemoteEvaluationResponsePayload = (
    ArenaScreenReceipt | RemoteQualificationProduct | RemoteQualificationHoldProduct
)


def remote_qualification_hold_to_dict(
    product: RemoteQualificationHoldProduct,
) -> dict[str, object]:
    if type(product) is not RemoteQualificationHoldProduct:
        raise RemoteEvaluationDispatcherError(
            "remote qualification HOLD is not exactly typed"
        )
    return product.to_dict()


def remote_qualification_hold_from_dict(
    value: object,
) -> RemoteQualificationHoldProduct:
    """Strictly reopen an untrusted, path-free qualification HOLD."""

    fields = {
        "candidate_digests",
        "diagnostic_digest",
        "kind",
        "ready_epoch",
        "ready_receipt_digest",
        "reason",
        "request_digest",
        "reservation_digests",
        "schema_version",
        "screen_lane",
        "selected_delta_digests",
        "service_digest",
        "service_identity",
        "worker_readiness_digest",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value["kind"] != "remote_qualification_hold"
        or type(value["reservation_digests"]) is not list
        or type(value["selected_delta_digests"]) is not list
        or type(value["candidate_digests"]) is not list
    ):
        raise RemoteEvaluationDispatcherError(
            "remote qualification HOLD fields are not closed"
        )
    try:
        return RemoteQualificationHoldProduct(
            request_digest=value["request_digest"],  # type: ignore[arg-type]
            service_identity=value["service_identity"],  # type: ignore[arg-type]
            service_digest=value["service_digest"],  # type: ignore[arg-type]
            worker_readiness_digest=value["worker_readiness_digest"],  # type: ignore[arg-type]
            ready_receipt_digest=value["ready_receipt_digest"],  # type: ignore[arg-type]
            ready_epoch=value["ready_epoch"],  # type: ignore[arg-type]
            screen_lane=value["screen_lane"],  # type: ignore[arg-type]
            reservation_digests=tuple(value["reservation_digests"]),
            selected_delta_digests=tuple(value["selected_delta_digests"]),
            candidate_digests=tuple(value["candidate_digests"]),
            reason=RemoteQualificationHoldReason(value["reason"]),
            diagnostic_digest=value["diagnostic_digest"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )
    except RemoteEvaluationDispatcherError:
        raise
    except (TypeError, ValueError) as exc:
        raise RemoteEvaluationDispatcherError(
            "remote qualification HOLD is invalid"
        ) from exc


def capture_remote_qualification_hold(
    request: object,
    *,
    reason: RemoteQualificationHoldReason,
    diagnostic_digest: str = "",
) -> RemoteQualificationHoldProduct:
    """Derive every authority binding from the consumed request bytes."""

    # Deferred to avoid a module cycle: the response codec imports this module.
    from cacheon.chain.remote_evaluation_dispatcher import RemoteEvaluationRequest

    if type(request) is not RemoteEvaluationRequest or request.stage != "qualification":
        raise RemoteEvaluationDispatcherError(
            "remote qualification HOLD requires an exact qualification request"
        )
    body = request.body
    try:
        candidates = body["candidates"]
        if type(candidates) is not list:
            raise TypeError("candidates are not a list")
        reservations = tuple(
            QualificationReservation.from_dict(row["reservation"])
            for row in candidates
        )
        candidate_digests = tuple(
            _digest(row["candidate_digest"], "candidate digest")
            for row in candidates
        )
        return RemoteQualificationHoldProduct(
            request_digest=request.digest,
            service_identity=request.service_identity,
            service_digest=body["service_digest"],  # type: ignore[arg-type]
            worker_readiness_digest=request.worker_readiness_digest,
            ready_receipt_digest=request.ready_receipt_digest,
            ready_epoch=request.ready_epoch,
            screen_lane=body["screen_lane"],  # type: ignore[arg-type]
            reservation_digests=tuple(
                row.reservation_digest for row in reservations
            ),
            selected_delta_digests=tuple(
                row.selected_delta_digest for row in reservations
            ),
            candidate_digests=candidate_digests,
            reason=reason,
            diagnostic_digest=diagnostic_digest,
        )
    except RemoteEvaluationDispatcherError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise RemoteEvaluationDispatcherError(
            "remote qualification HOLD request authority is invalid"
        ) from exc


def verify_remote_qualification_hold_request(
    product: RemoteQualificationHoldProduct,
    request: object,
) -> None:
    """Independently derive and compare every request-bound HOLD field."""

    if type(product) is not RemoteQualificationHoldProduct:
        raise RemoteEvaluationDispatcherError(
            "remote qualification HOLD is not exactly typed"
        )
    expected = capture_remote_qualification_hold(
        request,
        reason=product.reason,
        diagnostic_digest=product.diagnostic_digest,
    )
    if product != expected:
        raise RemoteEvaluationDispatcherError(
            "remote qualification HOLD differs from its request authority"
        )


def durable_remote_qualification_hold_reason(
    product: RemoteQualificationHoldProduct,
) -> str:
    """Project the closed worker enum into the durable recovery reason."""

    if type(product) is not RemoteQualificationHoldProduct:
        raise RemoteEvaluationDispatcherError(
            "remote qualification HOLD is not exactly typed"
        )
    return f"remote_qualification_hold:{product.reason.value}"


def is_exact_remote_stage_payload(payload: object, stage: object) -> bool:
    """Return whether an authenticated payload belongs to the exact stage."""

    if stage == "screen":
        return type(payload) is ArenaScreenReceipt
    if stage == "qualification":
        return type(payload) in {
            RemoteQualificationProduct,
            RemoteQualificationHoldProduct,
        }
    return False


__all__ = [
    "RemoteQualificationHoldProduct",
    "RemoteQualificationHoldReason",
    "RemoteEvaluationResponsePayload",
    "capture_remote_qualification_hold",
    "durable_remote_qualification_hold_reason",
    "is_exact_remote_stage_payload",
    "remote_qualification_hold_from_dict",
    "remote_qualification_hold_to_dict",
    "verify_remote_qualification_hold_request",
]
