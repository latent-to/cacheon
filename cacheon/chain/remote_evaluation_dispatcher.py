"""Authenticated remote execution for durable evaluation leases.

This module is the CPU-side boundary between :mod:`evaluation_coordinator` and
an out-of-process worker fleet.  It deliberately exposes only two typed worker
operations (screen and qualification).  There is no command, argv, environment,
module, or shell field in the protocol.

The CPU remains authoritative for FIFO claims, lease heartbeats, typed result
reopening, and CAS commits.  A transport invocation happens while the intake
controller is closed.  Transport or worker failures release the durable lease
without consuming an evaluation attempt, so another READY worker can reclaim it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Protocol

from cacheon.arena_service import (
    SCREEN_STAGES,
    ArenaScreenReceipt,
    PromotionDecision,
)
from cacheon.chain.evaluation_coordinator import (
    ClaimedQualificationEvaluation,
    ClaimedScreenEvaluation,
    EvaluationCoordinator,
    EvaluationResultEnvelope,
    EvaluationRun,
    WorkerReadiness,
    _LeaseHeartbeat,
)
from cacheon.chain.evaluation_leases import EvaluationLease, EvaluationLeaseMember
from cacheon.chain.remote_qualification_evidence import (
    _SCHEMA_VERSION,
    RemoteEvaluationDispatcherError,
    RemoteEvidenceArtifact,
    RemoteQualificationProduct,
    _digest,
    capture_remote_qualification_product,
    import_remote_qualification_evidence,
    remote_qualification_product_from_dict,
    _screen_receipt_from_dict,
    qualification_batch_from_dict,
    qualification_batch_to_dict,
    remote_qualification_product_to_dict,
)
from cacheon.chain.remote_qualification_hold import (
    RemoteEvaluationResponsePayload,
    RemoteQualificationHoldProduct,
    remote_qualification_hold_from_dict,
    remote_qualification_hold_to_dict,
    verify_remote_qualification_hold_request,
)
from cacheon.eval.qualification_intake import (
    QualificationReservation,
)
from cacheon.eval.native_artifact import NativeArtifactFile
from cacheon.stack_manifest import EvaluationStackManifest
from cacheon.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    sha256_hex,
)


_REQUEST_AUTH_DOMAIN = b"cacheon.remote-evaluation.request-auth.v1"
_RESPONSE_AUTH_DOMAIN = b"cacheon.remote-evaluation.response-auth.v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_AUTH_TAG = re.compile(r"[0-9a-f]{64}\Z")
_MAX_PROTOCOL_BODY_BYTES = 64 << 20

REMOTE_EVALUATION_PROTOCOL_DIGEST = canonical_digest(
    "cacheon.chain.remote-evaluation-protocol.v3",
    {
        "operations": ["screen", "qualification"],
        "qualification_payloads": [
            "remote_qualification_product",
            "remote_qualification_hold",
        ],
        "qualification_product": {
            "authority_manifest": True,
            "cpu_evidence_import": True,
            "incumbent_stack_tree": True,
            "screen_lane": True,
        },
        "request_auth": "hmac-sha256",
        "response_auth": "hmac-sha256",
        "result_encoding": "canonical-json",
        "shell_authority": False,
    },
)


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise RemoteEvaluationDispatcherError(f"{field_name} is malformed")
    return value

def _canonical_object(value: bytes, label: str) -> dict[str, object]:
    if type(value) is not bytes or not value or len(value) > _MAX_PROTOCOL_BODY_BYTES:
        raise RemoteEvaluationDispatcherError(f"{label} bytes are outside protocol bounds")
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteEvaluationDispatcherError(f"{label} is not canonical JSON") from exc
    if type(decoded) is not dict or canonical_json_bytes(decoded) != value:
        raise RemoteEvaluationDispatcherError(f"{label} is not a canonical object")
    return decoded

def _mac(secret: bytes, domain: bytes, digest: str) -> str:
    return hmac.new(secret, domain + b"\0" + digest.encode("ascii"), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class RemoteWorkerCredential:
    """CPU/worker shared authentication material, never placed on the wire."""

    credential_id: str
    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "credential_id", _identifier(self.credential_id, "credential_id"))
        if type(self.secret) is not bytes or not 32 <= len(self.secret) <= 4096:
            raise RemoteEvaluationDispatcherError(
                "remote worker credential must contain 32 to 4096 secret bytes"
            )

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.chain.remote-worker-credential.v1",
            {
                "credential_id": self.credential_id,
                "secret_sha256": sha256_hex(self.secret),
            },
        )

@dataclass(frozen=True)
class RemoteWorkerTransportIdentity:
    """Pinned identity of one closed worker transport and READY fleet."""

    transport_id: str
    endpoint_identity_digest: str
    protocol_digest: str
    credential_digest: str
    service_digest: str
    worker_readiness_digest: str
    max_response_bytes: int = 16 << 20
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "transport_id", _identifier(self.transport_id, "transport_id"))
        for name in (
            "endpoint_identity_digest",
            "protocol_digest",
            "credential_digest",
            "service_digest",
            "worker_readiness_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.protocol_digest != REMOTE_EVALUATION_PROTOCOL_DIGEST:
            raise RemoteEvaluationDispatcherError("remote transport protocol is unsupported")
        if (
            type(self.max_response_bytes) is not int
            or not 1 <= self.max_response_bytes <= _MAX_PROTOCOL_BODY_BYTES
            or type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
        ):
            raise RemoteEvaluationDispatcherError("remote transport bounds are malformed")

    def to_dict(self) -> dict[str, object]:
        return {
            "credential_digest": self.credential_digest,
            "endpoint_identity_digest": self.endpoint_identity_digest,
            "max_response_bytes": self.max_response_bytes,
            "protocol_digest": self.protocol_digest,
            "schema_version": self.schema_version,
            "service_digest": self.service_digest,
            "transport_id": self.transport_id,
            "worker_readiness_digest": self.worker_readiness_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RemoteWorkerTransportIdentity":
        fields = {
            "credential_digest",
            "endpoint_identity_digest",
            "max_response_bytes",
            "protocol_digest",
            "schema_version",
            "service_digest",
            "transport_id",
            "worker_readiness_digest",
        }
        if type(value) is not dict or set(value) != fields:
            raise RemoteEvaluationDispatcherError(
                "remote transport identity fields are not closed"
            )
        try:
            return cls(**value)  # type: ignore[arg-type]
        except RemoteEvaluationDispatcherError:
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RemoteEvaluationDispatcherError(
                "remote transport identity is invalid"
            ) from exc

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.chain.remote-worker-transport-identity.v1", self.to_dict()
        )

@dataclass(frozen=True)
class RemoteEvaluationRequest:
    """Canonical, authenticated request bound to one exact durable lease."""

    transport_identity_digest: str
    worker_readiness_digest: str
    ready_receipt_digest: str
    ready_epoch: int
    service_identity: str
    lease_id: str
    generation: int
    stage: str
    owner: str
    members: tuple[EvaluationLeaseMember, ...]
    claimed_block: int
    initial_expires_block: int
    body_kind: str
    body_bytes: bytes = field(repr=False)
    body_sha256: str
    credential_id: str
    auth_tag: str = field(repr=False)
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "transport_identity_digest",
            "worker_readiness_digest",
            "ready_receipt_digest",
            "lease_id",
            "body_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "credential_id", _identifier(self.credential_id, "credential_id"))
        members = tuple(self.members)
        if (
            type(self.ready_epoch) is not int
            or self.ready_epoch < 0
            or not isinstance(self.service_identity, str)
            or not self.service_identity
            or len(self.service_identity) > 512
            or any(
                ord(char) < 32 or ord(char) == 127
                for char in self.service_identity
            )
            or type(self.generation) is not int
            or self.generation <= 0
            or self.stage not in {"screen", "qualification"}
            or not isinstance(self.owner, str)
            or not self.owner
            or not members
            or any(type(row) is not EvaluationLeaseMember for row in members)
            or len({row.reservation_id for row in members}) != len(members)
            or type(self.claimed_block) is not int
            or self.claimed_block < 0
            or type(self.initial_expires_block) is not int
            or self.initial_expires_block <= self.claimed_block
            or self.body_kind != f"{self.stage}_work"
            or type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
            or not isinstance(self.auth_tag, str)
            or _AUTH_TAG.fullmatch(self.auth_tag) is None
        ):
            raise RemoteEvaluationDispatcherError("remote evaluation request is malformed")
        object.__setattr__(self, "members", members)
        try:
            EvaluationLease(
                self.lease_id,
                self.generation,
                self.stage,
                self.owner,
                members,
                self.claimed_block,
                self.initial_expires_block,
                self.initial_expires_block,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RemoteEvaluationDispatcherError(
                "remote request lease projection is invalid"
            ) from exc
        body = _canonical_object(self.body_bytes, "remote request body")
        if sha256_hex(self.body_bytes) != self.body_sha256:
            raise RemoteEvaluationDispatcherError("remote request body digest differs")
        _validate_request_body(self.stage, body)
        if _request_body_reservation_ids(self.stage, body) != tuple(
            row.reservation_id for row in members
        ):
            raise RemoteEvaluationDispatcherError(
                "remote request body differs from its lease members"
            )

    @property
    def body(self) -> dict[str, object]:
        return _canonical_object(self.body_bytes, "remote request body")

    @property
    def lease_reservation_ids(self) -> tuple[str, ...]:
        return tuple(row.reservation_id for row in self.members)

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "body_kind": self.body_kind,
            "body_sha256": self.body_sha256,
            "claimed_block": self.claimed_block,
            "credential_id": self.credential_id,
            "generation": self.generation,
            "initial_expires_block": self.initial_expires_block,
            "lease_id": self.lease_id,
            "members": [row.to_dict() for row in self.members],
            "owner": self.owner,
            "ready_epoch": self.ready_epoch,
            "ready_receipt_digest": self.ready_receipt_digest,
            "schema_version": self.schema_version,
            "service_identity": self.service_identity,
            "stage": self.stage,
            "transport_identity_digest": self.transport_identity_digest,
            "worker_readiness_digest": self.worker_readiness_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.chain.remote-evaluation-request.v1", self._unsigned_dict()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self._unsigned_dict(),
            "auth_tag": self.auth_tag,
            "body": self.body,
            "request_digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RemoteEvaluationRequest":
        fields = {
            "auth_tag",
            "body",
            "body_kind",
            "body_sha256",
            "claimed_block",
            "credential_id",
            "generation",
            "initial_expires_block",
            "lease_id",
            "members",
            "owner",
            "ready_epoch",
            "ready_receipt_digest",
            "request_digest",
            "schema_version",
            "service_identity",
            "stage",
            "transport_identity_digest",
            "worker_readiness_digest",
        }
        if type(value) is not dict or set(value) != fields:
            raise RemoteEvaluationDispatcherError("remote request fields are not closed")
        if type(value["body"]) is not dict or type(value["members"]) is not list:
            raise RemoteEvaluationDispatcherError("remote request arrays or body are malformed")
        members = []
        for row in value["members"]:
            if type(row) is not dict or set(row) != {"reservation_id", "prior_status"}:
                raise RemoteEvaluationDispatcherError("remote request member is malformed")
            try:
                members.append(EvaluationLeaseMember(**row))
            except (TypeError, ValueError, RuntimeError) as exc:
                raise RemoteEvaluationDispatcherError(
                    "remote request member is invalid"
                ) from exc
        request = cls(
            transport_identity_digest=value["transport_identity_digest"],  # type: ignore[arg-type]
            worker_readiness_digest=value["worker_readiness_digest"],  # type: ignore[arg-type]
            ready_receipt_digest=value["ready_receipt_digest"],  # type: ignore[arg-type]
            ready_epoch=value["ready_epoch"],  # type: ignore[arg-type]
            service_identity=value["service_identity"],  # type: ignore[arg-type]
            lease_id=value["lease_id"],  # type: ignore[arg-type]
            generation=value["generation"],  # type: ignore[arg-type]
            stage=value["stage"],  # type: ignore[arg-type]
            owner=value["owner"],  # type: ignore[arg-type]
            members=tuple(members),
            claimed_block=value["claimed_block"],  # type: ignore[arg-type]
            initial_expires_block=value["initial_expires_block"],  # type: ignore[arg-type]
            body_kind=value["body_kind"],  # type: ignore[arg-type]
            body_bytes=canonical_json_bytes(value["body"]),
            body_sha256=value["body_sha256"],  # type: ignore[arg-type]
            credential_id=value["credential_id"],  # type: ignore[arg-type]
            auth_tag=value["auth_tag"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )
        if value["request_digest"] != request.digest:
            raise RemoteEvaluationDispatcherError("remote request identity differs")
        return request

@dataclass(frozen=True)
class AuthenticatedRemoteEvaluationResponse:
    """Canonical worker response authenticated against its exact request."""

    request_digest: str
    transport_identity_digest: str
    worker_readiness_digest: str
    ready_receipt_digest: str
    ready_epoch: int
    stage: str
    payload_kind: str
    payload_bytes: bytes = field(repr=False)
    payload_sha256: str
    payload_digest: str
    credential_id: str
    auth_tag: str = field(repr=False)
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "request_digest",
            "transport_identity_digest",
            "worker_readiness_digest",
            "ready_receipt_digest",
            "payload_sha256",
            "payload_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "credential_id", _identifier(self.credential_id, "credential_id"))
        if (
            type(self.ready_epoch) is not int
            or self.ready_epoch < 0
            or self.stage not in {"screen", "qualification"}
            or (
                self.stage == "screen"
                and self.payload_kind != "arena_screen_receipt"
            )
            or (
                self.stage == "qualification"
                and self.payload_kind
                not in {
                    "remote_qualification_product",
                    "remote_qualification_hold",
                }
            )
            or type(self.schema_version) is not int
            or self.schema_version != _SCHEMA_VERSION
            or not isinstance(self.auth_tag, str)
            or _AUTH_TAG.fullmatch(self.auth_tag) is None
        ):
            raise RemoteEvaluationDispatcherError("remote evaluation response is malformed")
        _canonical_object(self.payload_bytes, "remote response payload")
        if sha256_hex(self.payload_bytes) != self.payload_sha256:
            raise RemoteEvaluationDispatcherError("remote response payload digest differs")

    @property
    def payload(self) -> dict[str, object]:
        return _canonical_object(self.payload_bytes, "remote response payload")

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "credential_id": self.credential_id,
            "payload_digest": self.payload_digest,
            "payload_kind": self.payload_kind,
            "payload_sha256": self.payload_sha256,
            "ready_epoch": self.ready_epoch,
            "ready_receipt_digest": self.ready_receipt_digest,
            "request_digest": self.request_digest,
            "schema_version": self.schema_version,
            "stage": self.stage,
            "transport_identity_digest": self.transport_identity_digest,
            "worker_readiness_digest": self.worker_readiness_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(
            "cacheon.chain.remote-evaluation-response.v1", self._unsigned_dict()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self._unsigned_dict(),
            "auth_tag": self.auth_tag,
            "payload": self.payload,
            "response_digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "AuthenticatedRemoteEvaluationResponse":
        fields = {
            "auth_tag",
            "credential_id",
            "payload",
            "payload_digest",
            "payload_kind",
            "payload_sha256",
            "ready_epoch",
            "ready_receipt_digest",
            "request_digest",
            "response_digest",
            "schema_version",
            "stage",
            "transport_identity_digest",
            "worker_readiness_digest",
        }
        if type(value) is not dict or set(value) != fields or type(value["payload"]) is not dict:
            raise RemoteEvaluationDispatcherError("remote response fields are not closed")
        response = cls(
            request_digest=value["request_digest"],  # type: ignore[arg-type]
            transport_identity_digest=value["transport_identity_digest"],  # type: ignore[arg-type]
            worker_readiness_digest=value["worker_readiness_digest"],  # type: ignore[arg-type]
            ready_receipt_digest=value["ready_receipt_digest"],  # type: ignore[arg-type]
            ready_epoch=value["ready_epoch"],  # type: ignore[arg-type]
            stage=value["stage"],  # type: ignore[arg-type]
            payload_kind=value["payload_kind"],  # type: ignore[arg-type]
            payload_bytes=canonical_json_bytes(value["payload"]),
            payload_sha256=value["payload_sha256"],  # type: ignore[arg-type]
            payload_digest=value["payload_digest"],  # type: ignore[arg-type]
            credential_id=value["credential_id"],  # type: ignore[arg-type]
            auth_tag=value["auth_tag"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )
        if value["response_digest"] != response.digest:
            raise RemoteEvaluationDispatcherError("remote response identity differs")
        return response


class AuthenticatedWorkerTransport(Protocol):
    """Closed worker transport; implementations must enforce endpoint pinning."""

    identity: RemoteWorkerTransportIdentity

    def run_screen(
        self,
        request: RemoteEvaluationRequest,
        *,
        job: ClaimedScreenEvaluation,
    ) -> AuthenticatedRemoteEvaluationResponse: ...

def _payload_encoding(payload: object) -> tuple[str, bytes, str]:
    if type(payload) is ArenaScreenReceipt:
        return "arena_screen_receipt", canonical_json_bytes(payload.to_dict()), payload.digest
    if type(payload) is RemoteQualificationProduct:
        return (
            "remote_qualification_product",
            canonical_json_bytes(remote_qualification_product_to_dict(payload)),
            payload.digest,
        )
    if type(payload) is RemoteQualificationHoldProduct:
        return (
            "remote_qualification_hold",
            canonical_json_bytes(remote_qualification_hold_to_dict(payload)),
            payload.digest,
        )
    raise RemoteEvaluationDispatcherError("remote response payload is not exactly typed")

def _validate_request_body(stage: str, value: dict[str, object]) -> None:
    if stage == "screen":
        fields = {
            "candidate_digest",
            "kind",
            "publication",
            "reservation",
            "schema_version",
            "screen_attempt",
            "screen_policy",
            "service_digest",
        }
        if (
            set(value) != fields
            or value["kind"] != "screen_work"
            or value["schema_version"] != _SCHEMA_VERSION
        ):
            raise RemoteEvaluationDispatcherError("screen request body is not closed")
        try:
            candidate_digest = _digest(value["candidate_digest"], "candidate_digest")
            _digest(value["service_digest"], "service_digest")
            if type(value["screen_attempt"]) is not int or value["screen_attempt"] <= 0:
                raise RemoteEvaluationDispatcherError("screen attempt is malformed")
            reservation = QualificationReservation.from_dict(value["reservation"])
            publication_digest = _publication_wire_digest(value["publication"])
            _validate_screen_policy(value["screen_policy"])
            if publication_digest != reservation.submission_digest:
                raise RemoteEvaluationDispatcherError("screen request publication differs")
            reservation_value = reservation.to_dict()
            reservation_value.pop("arrival_order")
            expected_candidate = canonical_digest(
                "cacheon.arena.candidate-binding",
                {
                    "publication_digest": publication_digest,
                    "reservation": reservation_value,
                    "screen_attempt": value["screen_attempt"],
                },
            )
            if candidate_digest != expected_candidate:
                raise RemoteEvaluationDispatcherError("screen request candidate differs")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RemoteEvaluationDispatcherError("screen request body is invalid") from exc
        return
    fields = {
        "candidates",
        "kind",
        "qualification_policy_digest",
        "schema_version",
        "screen_lane",
        "service_digest",
    }
    if (
        set(value) != fields
        or value["kind"] != "qualification_work"
        or value["schema_version"] != _SCHEMA_VERSION
        or type(value["candidates"]) is not list
    ):
        raise RemoteEvaluationDispatcherError("qualification request body is not closed")
    try:
        _digest(value["qualification_policy_digest"], "qualification_policy_digest")
        service_digest = _digest(value["service_digest"], "service_digest")
    except (TypeError, ValueError) as exc:
        raise RemoteEvaluationDispatcherError("qualification request body is invalid") from exc
    candidate_fields = {"candidate_digest", "publication", "reservation", "screen_receipt"}
    if (
        not value["candidates"]
        or value["screen_lane"] not in {"primary", "reproduction"}
        or (
            value["screen_lane"] == "reproduction"
            and len(value["candidates"]) != 1
        )
    ):
        raise RemoteEvaluationDispatcherError("qualification request cohort is empty")
    reservations = []
    for row in value["candidates"]:
        if type(row) is not dict or set(row) != candidate_fields:
            raise RemoteEvaluationDispatcherError("qualification request candidate is malformed")
        candidate_digest = _digest(row["candidate_digest"], "candidate_digest")
        reservation = QualificationReservation.from_dict(row["reservation"])
        publication_digest = _publication_wire_digest(row["publication"])
        receipt = _screen_receipt_from_dict(row["screen_receipt"])
        reservation_value = reservation.to_dict()
        reservation_value.pop("arrival_order")
        expected_candidate = canonical_digest(
            "cacheon.arena.candidate-binding",
            {
                "publication_digest": publication_digest,
                "reservation": reservation_value,
                "screen_attempt": receipt.screen_attempt,
            },
        )
        if (
            publication_digest != reservation.submission_digest
            or receipt.candidate_digest != candidate_digest
            or receipt.service_digest != service_digest
            or receipt.decision is not PromotionDecision.PROMOTE
            or candidate_digest != expected_candidate
        ):
            raise RemoteEvaluationDispatcherError("qualification request provenance differs")
        reservations.append(reservation)

def _request_body_reservation_ids(
    stage: str,
    value: dict[str, object],
) -> tuple[str, ...]:
    try:
        if stage == "screen":
            return (
                QualificationReservation.from_dict(
                    value["reservation"]
                ).reservation_digest,
            )
        return tuple(
            QualificationReservation.from_dict(row["reservation"]).reservation_digest
            for row in value["candidates"]  # type: ignore[union-attr]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RemoteEvaluationDispatcherError(
            "remote request reservation projection is invalid"
        ) from exc

def _publication_wire_digest(value: object) -> str:
    fields = {
        "address_digest",
        "content_hash",
        "directories",
        "files",
        "publication_digest",
        "schema",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema"] != "cacheon.worker-bundle-publication.v1"
        or type(value["directories"]) is not list
        or type(value["files"]) is not list
    ):
        raise RemoteEvaluationDispatcherError("worker publication wire fields are not closed")
    directories = value["directories"]
    if (
        any(not isinstance(row, str) for row in directories)
        or directories != sorted(set(directories))
        or not value["files"]
    ):
        raise RemoteEvaluationDispatcherError("worker publication inventory is malformed")
    files = []
    for row in value["files"]:
        if type(row) is not dict or set(row) != {"path", "sha256", "size"}:
            raise RemoteEvaluationDispatcherError("worker publication file is malformed")
        try:
            files.append(NativeArtifactFile(**row))
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RemoteEvaluationDispatcherError(
                "worker publication file is invalid"
            ) from exc
    if files != sorted(set(files), key=lambda row: row.path):
        raise RemoteEvaluationDispatcherError("worker publication files are not canonical")
    for name in ("address_digest", "content_hash", "publication_digest"):
        _digest(value[name], name)
    return canonical_digest(
        "cacheon.chain.worker-bundle-publication",
        {
            "address_digest": value["address_digest"],
            "content_hash": value["content_hash"],
            "directories": directories,
            "files": [row.to_dict() for row in files],
            "publication_digest": value["publication_digest"],
            "schema": value["schema"],
        },
    )

def _validate_screen_policy(value: object) -> None:
    if type(value) is not dict or set(value) != {"crownable", "stages"}:
        raise RemoteEvaluationDispatcherError("screen request policy is not closed")
    stages = value["stages"]
    if value["crownable"] is not False or type(stages) is not list:
        raise RemoteEvaluationDispatcherError("screen request policy is malformed")
    if len(stages) != len(SCREEN_STAGES):
        raise RemoteEvaluationDispatcherError("screen request stages are incomplete")
    for expected, row in zip(SCREEN_STAGES, stages, strict=True):
        if (
            type(row) is not dict
            or set(row) != {"stage", "timeout_ms"}
            or row["stage"] != expected
            or type(row["timeout_ms"]) is not int
            or row["timeout_ms"] <= 0
        ):
            raise RemoteEvaluationDispatcherError("screen request stage is malformed")

def _request_body_for_screen(
    coordinator: EvaluationCoordinator,
    claim: ClaimedScreenEvaluation,
) -> dict[str, object]:
    return {
        "candidate_digest": claim.candidate.digest,
        "kind": "screen_work",
        "publication": claim.publication.to_dict(),
        "reservation": claim.candidate.reservation.to_dict(),
        "schema_version": _SCHEMA_VERSION,
        "screen_attempt": claim.candidate.screen_attempt,
        "screen_policy": coordinator.service.manifest.screens.to_dict(),
        "service_digest": coordinator.service.identity,
    }

def _request_body_for_qualification(
    coordinator: EvaluationCoordinator,
    claim: ClaimedQualificationEvaluation,
) -> dict[str, object]:
    return {
        "candidates": [
            {
                "candidate_digest": candidate.digest,
                "publication": publication.to_dict(),
                "reservation": candidate.reservation.to_dict(),
                "screen_receipt": receipt.to_dict(),
            }
            for candidate, publication, receipt in zip(
                claim.candidates, claim.publications, claim.screen_receipts, strict=True
            )
        ],
        "kind": "qualification_work",
        "qualification_policy_digest": (
            coordinator.service.manifest.qualification_policy_digest
        ),
        "schema_version": _SCHEMA_VERSION,
        "screen_lane": claim.screen_lane,
        "service_digest": coordinator.service.identity,
    }

def seal_remote_request(
    lease: EvaluationLease,
    readiness: WorkerReadiness,
    service_identity: str,
    transport_identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
    body: dict[str, object],
) -> RemoteEvaluationRequest:
    """Seal one closed request; usable by durable spool transports."""

    if type(lease) is not EvaluationLease or type(readiness) is not WorkerReadiness:
        raise RemoteEvaluationDispatcherError("remote request authority is not exactly typed")
    if (
        type(transport_identity) is not RemoteWorkerTransportIdentity
        or type(credential) is not RemoteWorkerCredential
    ):
        raise RemoteEvaluationDispatcherError("remote request transport authority is not exact")
    if transport_identity.credential_digest != credential.digest:
        raise RemoteEvaluationDispatcherError("remote request credential identity differs")
    if (
        transport_identity.worker_readiness_digest != readiness.digest
        or transport_identity.service_digest != readiness.service_digest
        or service_identity != f"{readiness.arena_id}@{readiness.service_digest}"
    ):
        raise RemoteEvaluationDispatcherError("remote request READY authority differs")
    body_bytes = canonical_json_bytes(body)
    request = RemoteEvaluationRequest(
        transport_identity_digest=transport_identity.digest,
        worker_readiness_digest=readiness.digest,
        ready_receipt_digest=readiness.ready_receipt_digest,
        ready_epoch=readiness.ready_epoch,
        service_identity=service_identity,
        lease_id=lease.lease_id,
        generation=lease.generation,
        stage=lease.stage,
        owner=lease.owner,
        members=lease.members,
        claimed_block=lease.claimed_block,
        initial_expires_block=lease.initial_expires_block,
        body_kind=f"{lease.stage}_work",
        body_bytes=body_bytes,
        body_sha256=sha256_hex(body_bytes),
        credential_id=credential.credential_id,
        auth_tag="0" * 64,
    )
    return replace(
        request,
        auth_tag=_mac(credential.secret, _REQUEST_AUTH_DOMAIN, request.digest),
    )

def verify_remote_request(
    request: RemoteEvaluationRequest,
    transport_identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> None:
    """Authenticate a request before a worker acts on it."""

    if type(request) is not RemoteEvaluationRequest:
        raise RemoteEvaluationDispatcherError("remote request is not exactly typed")
    if (
        type(transport_identity) is not RemoteWorkerTransportIdentity
        or type(credential) is not RemoteWorkerCredential
    ):
        raise RemoteEvaluationDispatcherError("remote request verifier authority is not exact")
    if (
        request.transport_identity_digest != transport_identity.digest
        or request.worker_readiness_digest != transport_identity.worker_readiness_digest
        or request.body["service_digest"] != transport_identity.service_digest
        or request.credential_id != credential.credential_id
        or transport_identity.credential_digest != credential.digest
        or not hmac.compare_digest(
            request.auth_tag,
            _mac(credential.secret, _REQUEST_AUTH_DOMAIN, request.digest),
        )
    ):
        raise RemoteEvaluationDispatcherError("remote request authentication failed")

def seal_remote_response(
    request: RemoteEvaluationRequest,
    payload: RemoteEvaluationResponsePayload,
    transport_identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> AuthenticatedRemoteEvaluationResponse:
    """Seal one typed worker result against the authenticated request."""

    verify_remote_request(request, transport_identity, credential)
    payload_kind, payload_bytes, payload_digest = _payload_encoding(payload)
    response = AuthenticatedRemoteEvaluationResponse(
        request_digest=request.digest,
        transport_identity_digest=transport_identity.digest,
        worker_readiness_digest=request.worker_readiness_digest,
        ready_receipt_digest=request.ready_receipt_digest,
        ready_epoch=request.ready_epoch,
        stage=request.stage,
        payload_kind=payload_kind,
        payload_bytes=payload_bytes,
        payload_sha256=sha256_hex(payload_bytes),
        payload_digest=payload_digest,
        credential_id=credential.credential_id,
        auth_tag="0" * 64,
    )
    return replace(
        response,
        auth_tag=_mac(credential.secret, _RESPONSE_AUTH_DOMAIN, response.digest),
    )

def reopen_remote_response(
    request: RemoteEvaluationRequest,
    response: AuthenticatedRemoteEvaluationResponse,
    transport_identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> RemoteEvaluationResponsePayload:
    """Authenticate bytes, then independently reopen the exact typed result."""

    if (
        type(request) is not RemoteEvaluationRequest
        or type(response) is not AuthenticatedRemoteEvaluationResponse
    ):
        raise RemoteEvaluationDispatcherError("remote response authority is not exactly typed")
    verify_remote_request(request, transport_identity, credential)
    if len(canonical_json_bytes(response.to_dict())) > transport_identity.max_response_bytes:
        raise RemoteEvaluationDispatcherError("remote response exceeds its declared bound")
    if (
        response.request_digest != request.digest
        or response.transport_identity_digest != transport_identity.digest
        or response.worker_readiness_digest != request.worker_readiness_digest
        or response.ready_receipt_digest != request.ready_receipt_digest
        or response.ready_epoch != request.ready_epoch
        or response.stage != request.stage
        or response.credential_id != credential.credential_id
        or transport_identity.credential_digest != credential.digest
        or not hmac.compare_digest(
            response.auth_tag,
            _mac(credential.secret, _RESPONSE_AUTH_DOMAIN, response.digest),
        )
    ):
        raise RemoteEvaluationDispatcherError("remote response authentication failed")
    payload: RemoteEvaluationResponsePayload
    if response.stage == "screen":
        payload = _screen_receipt_from_dict(response.payload)
        observed_digest = payload.digest
    elif response.payload_kind == "remote_qualification_product":
        payload = remote_qualification_product_from_dict(response.payload)
        if (
            payload.service_digest != request.body["service_digest"]
            or payload.worker_readiness_digest != request.worker_readiness_digest
            or payload.ready_receipt_digest != request.ready_receipt_digest
            or payload.ready_epoch != request.ready_epoch
            or payload.screen_lane != request.body["screen_lane"]
            or tuple(
                row.reservation_digest
                for row in payload.authority_manifest.reservations
            )
            != request.lease_reservation_ids
        ):
            raise RemoteEvaluationDispatcherError(
                "remote qualification product differs from its request authority"
            )
        observed_digest = payload.digest
    else:
        payload = remote_qualification_hold_from_dict(response.payload)
        verify_remote_qualification_hold_request(payload, request)
        observed_digest = payload.digest
    if observed_digest != response.payload_digest:
        raise RemoteEvaluationDispatcherError("remote typed result digest differs")
    return payload


class RemoteEvaluationDispatcher:
    """Standing CPU service entry point for one durable remote evaluation."""

    def __init__(
        self,
        *,
        coordinator: EvaluationCoordinator,
        transport: AuthenticatedWorkerTransport,
        credential: RemoteWorkerCredential,
        qualification_evidence_root: str | Path | None = None,
        qualification_incumbent_stack: EvaluationStackManifest | None = None,
        qualification_incumbent_tree_digest: str | None = None,
    ):
        if (
            type(coordinator) is not EvaluationCoordinator
            or type(credential) is not RemoteWorkerCredential
        ):
            raise RemoteEvaluationDispatcherError("remote dispatcher authority is not exact")
        identity = getattr(transport, "identity", None)
        if type(identity) is not RemoteWorkerTransportIdentity:
            raise RemoteEvaluationDispatcherError("remote transport has no exact identity")
        if not callable(getattr(transport, "run_screen", None)):
            raise RemoteEvaluationDispatcherError("remote transport is not closed and typed")
        if (
            identity.service_digest != coordinator.service.identity
            or identity.worker_readiness_digest != coordinator.readiness.digest
            or identity.credential_digest != credential.digest
        ):
            raise RemoteEvaluationDispatcherError("remote transport differs from CPU authority")
        coordinator.readiness.validate(coordinator.service)
        evidence_root = (
            None
            if qualification_evidence_root is None
            else Path(qualification_evidence_root)
        )
        if evidence_root is not None and (
            not evidence_root.is_absolute()
            or evidence_root != Path(os.path.normpath(evidence_root))
        ):
            raise RemoteEvaluationDispatcherError(
                "CPU qualification evidence root is not canonical and absolute"
            )
        configured = (
            evidence_root is not None,
            qualification_incumbent_stack is not None,
            qualification_incumbent_tree_digest is not None,
        )
        if any(configured) and not all(configured):
            raise RemoteEvaluationDispatcherError(
                "CPU qualification authority must configure evidence and incumbent together"
            )
        if qualification_incumbent_stack is not None:
            if type(qualification_incumbent_stack) is not EvaluationStackManifest:
                raise RemoteEvaluationDispatcherError(
                    "CPU qualification incumbent stack is not exactly typed"
                )
            incumbent_tree_digest = _digest(
                qualification_incumbent_tree_digest,
                "qualification_incumbent_tree_digest",
            )
            runtime = coordinator.service.manifest.runtime
            if (
                qualification_incumbent_stack.runtime_digest
                != runtime.runtime_digest
                or qualification_incumbent_stack.base_engine_digest
                != runtime.base_engine_digest
                or qualification_incumbent_stack.arena_digest
                != coordinator.service.identity
            ):
                raise RemoteEvaluationDispatcherError(
                    "CPU qualification incumbent differs from the sealed service"
                )
        else:
            incumbent_tree_digest = None
        self.coordinator = coordinator
        self.transport = transport
        self.credential = credential
        self.transport_identity = identity
        self.qualification_evidence_root = evidence_root
        self.qualification_incumbent_stack = qualification_incumbent_stack
        self.qualification_incumbent_tree_digest = incumbent_tree_digest

    def _validate_live_transport(self) -> None:
        self.coordinator.readiness.validate(self.coordinator.service)
        if getattr(self.transport, "identity", None) != self.transport_identity:
            raise RemoteEvaluationDispatcherError(
                "remote transport identity drifted before claim"
            )

    def _release_after_remote_error(
        self,
        lease: EvaluationLease,
        *,
        reason: str,
        cause: BaseException,
        result_digest: str = "",
    ) -> None:
        try:
            self.coordinator._release(lease, reason=reason, result_digest=result_digest)
        except BaseException as release_error:
            raise RemoteEvaluationDispatcherError(
                f"{reason}; durable infrastructure release also failed: {release_error}"
            ) from cause
        raise RemoteEvaluationDispatcherError(reason) from cause

    def dispatch_screen_once(self) -> EvaluationRun | None:
        """Claim the exact FIFO screen row, invoke remotely, and CAS-commit."""
        self._validate_live_transport()
        claim = self.coordinator.claim_screen()
        if claim is None:
            return None
        heartbeat = _LeaseHeartbeat(self.coordinator, claim.lease)
        try:
            heartbeat.start()
        except BaseException as exc:
            self._release_after_remote_error(
                claim.lease,
                reason="remote_screen_heartbeat_start",
                cause=exc,
            )
        try:
            request = seal_remote_request(
                claim.lease,
                self.coordinator.readiness,
                self.coordinator.service.manifest.service_id,
                self.transport_identity,
                self.credential,
                _request_body_for_screen(self.coordinator, claim),
            )
            response = self.transport.run_screen(request, job=claim)
            receipt = reopen_remote_response(
                request, response, self.transport_identity, self.credential
            )
            if type(receipt) is not ArenaScreenReceipt:
                raise RemoteEvaluationDispatcherError(
                    "remote screen returned another payload type"
                )
        except BaseException as exc:
            lease, heartbeat_error = heartbeat.stop()
            self._release_after_remote_error(
                lease,
                reason="remote_screen_infrastructure",
                cause=heartbeat_error or exc,
            )
        lease, heartbeat_error = heartbeat.stop()
        claim = replace(claim, lease=lease)
        envelope = EvaluationResultEnvelope.seal(
            lease, self.coordinator.readiness, self.coordinator.service, receipt
        )
        if heartbeat_error is not None:
            self._release_after_remote_error(
                lease,
                reason="remote_screen_heartbeat",
                cause=heartbeat_error,
                result_digest=envelope.digest,
            )
        self.coordinator.commit_screen_result(claim, receipt, envelope)
        return EvaluationRun(lease, envelope, receipt, "completed")


__all__ = [
    "AuthenticatedRemoteEvaluationResponse",
    "AuthenticatedWorkerTransport",
    "REMOTE_EVALUATION_PROTOCOL_DIGEST",
    "RemoteEvidenceArtifact",
    "RemoteEvaluationDispatcher",
    "RemoteEvaluationDispatcherError",
    "RemoteEvaluationRequest",
    "RemoteEvaluationResponsePayload",
    "RemoteQualificationProduct",
    "RemoteQualificationHoldProduct",
    "RemoteWorkerCredential",
    "RemoteWorkerTransportIdentity",
    "capture_remote_qualification_product",
    "import_remote_qualification_evidence",
    "qualification_batch_from_dict",
    "qualification_batch_to_dict",
    "remote_qualification_product_from_dict",
    "remote_qualification_product_to_dict",
    "remote_qualification_hold_from_dict",
    "remote_qualification_hold_to_dict",
    "reopen_remote_response",
    "seal_remote_request",
    "seal_remote_response",
    "verify_remote_request",
]
