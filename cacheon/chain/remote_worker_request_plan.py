"""Crash-safe, single-carrier planning for remote qualification requests.

The durable evaluation store owns whether a planned request may be published.
This module only creates and reopens the exact spool carrier selected by that
store.  Planning samples time once and derives every identity.  Materializing
never writes ``REQUEST_READY``; publishing that marker is a separate,
idempotent point of no return.

Recovery is deliberately fail-closed.  Exact unpublished hidden state may be
repaired, but a changed authority, duplicate carrier, publication marker,
dispatch evidence, missing published carrier, or ambiguous local result raises
``QualificationRecoveryHold``.  No recovery path creates a replacement plan.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn

from cacheon.chain.evaluation_leases import (
    EvaluationLease,
    EvaluationLeaseMember,
)
from cacheon.chain.remote_evaluation_dispatcher import (
    AuthenticatedRemoteEvaluationResponse,
    RemoteEvaluationDispatcherError,
    RemoteEvaluationRequest,
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
    verify_remote_request,
)
from cacheon.chain.remote_worker_artifact_recovery import (
    PlannedQualificationArtifact,
    QUALIFICATION_ARTIFACT_ROLES,
    copy_stable_artifact,
    publication_archive,
    qualification_source_map,
    stable_artifact_identity,
)
from cacheon.chain.remote_worker_registration import verify_registration
from cacheon.chain.remote_worker_spool import (
    DOMAIN_REQUEST,
    EPOCH,
    MAX_JOB_SECONDS,
    MAX_WIRE_PAYLOAD_BYTES,
    SCHEMA_DISPATCH_STATE,
    SCHEMA_REQUEST,
    RemoteWorkerError,
    atomic_bytes,
    atomic_json,
    artifact_for_role,
    fail,
    load_json,
    require_closed,
    require_digest,
    require_int,
    spool_canonical_json,
    spool_digest,
    verify_adapter_result,
    verify_lease,
    verify_request,
)


SCHEMA_QUALIFICATION_REQUEST_PLAN = "cacheon-qualification-request-plan-v1"
DOMAIN_QUALIFICATION_REQUEST_PLAN = "cacheon.chain.qualification-request-plan.v1"
_PLAN_FIELDS = frozenset(
    {
        "artifacts",
        "created_at_unix",
        "credential_digest",
        "deadline_unix",
        "lease",
        "plan_digest",
        "queued_at_unix_ns",
        "registration_digest",
        "remote_request",
        "request_id",
        "schema",
        "transport_identity_digest",
        "worker_epoch",
    }
)
_LOCK_NAME = ".qualification-request-plans.lock"


class QualificationRecoveryHold(RemoteWorkerError):
    """Recovery cannot continue without an explicit operator/store decision."""

    def __init__(self, code: str, request_id: str, detail: str):
        self.code = code
        self.request_id = request_id
        super().__init__(f"qualification recovery HOLD [{code}] {detail}")


def _lease_dict(lease: EvaluationLease) -> dict[str, object]:
    return {
        "claimed_block": lease.claimed_block,
        "expires_block": lease.expires_block,
        "generation": lease.generation,
        "initial_expires_block": lease.initial_expires_block,
        "lease_id": lease.lease_id,
        "members": [member.to_dict() for member in lease.members],
        "owner": lease.owner,
        "stage": lease.stage,
    }


def _lease_from_dict(value: object) -> EvaluationLease:
    row = verify_lease(value)
    return EvaluationLease(
        row["lease_id"],
        row["generation"],
        row["stage"],
        row["owner"],
        tuple(
            EvaluationLeaseMember(member["reservation_id"], member["prior_status"])
            for member in row["members"]
        ),
        row["claimed_block"],
        row["initial_expires_block"],
        row["expires_block"],
    )


@dataclass(frozen=True)
class QualificationRequestPlan:
    """Immutable identities needed to reopen exactly one qualification carrier."""

    registration_digest: str
    worker_epoch: str
    transport_identity_digest: str
    credential_digest: str
    remote_request: RemoteEvaluationRequest
    lease: EvaluationLease
    artifacts: tuple[PlannedQualificationArtifact, ...]
    created_at_unix: int
    deadline_unix: int
    queued_at_unix_ns: int
    request_id: str
    plan_digest: str

    def __post_init__(self) -> None:
        for field, value in (
            ("registration digest", self.registration_digest),
            ("transport identity digest", self.transport_identity_digest),
            ("credential digest", self.credential_digest),
            ("request id", self.request_id),
            ("plan digest", self.plan_digest),
        ):
            require_digest(value, field)
        if not isinstance(self.worker_epoch, str) or EPOCH.fullmatch(self.worker_epoch) is None:
            fail("qualification plan worker epoch is malformed")
        if type(self.remote_request) is not RemoteEvaluationRequest:
            fail("qualification plan request is not exactly typed")
        if type(self.lease) is not EvaluationLease or self.lease.stage != "qualification":
            fail("qualification plan lease is not exactly typed")
        artifacts = tuple(self.artifacts)
        if (
            len(artifacts) != len(QUALIFICATION_ARTIFACT_ROLES)
            or tuple(row.role for row in artifacts) != QUALIFICATION_ARTIFACT_ROLES
        ):
            fail("qualification plan artifact roles are incomplete or reordered")
        object.__setattr__(self, "artifacts", artifacts)
        require_int(self.created_at_unix, "plan creation time", minimum=1)
        require_int(
            self.deadline_unix,
            "plan deadline",
            minimum=self.created_at_unix + 1,
        )
        require_int(self.queued_at_unix_ns, "plan queue time", minimum=1)
        if self.deadline_unix - self.created_at_unix > MAX_JOB_SECONDS:
            fail("qualification plan deadline exceeds deployment ceiling")
        if self.remote_request.stage != "qualification":
            fail("qualification plan contains another request stage")
        if not _request_matches_lease(self.remote_request, self.lease):
            fail("qualification plan request differs from its lease")
        if self.request_id != spool_digest(DOMAIN_REQUEST, self.outer_unsigned()):
            fail("qualification plan request id differs from its exact envelope")
        if self.plan_digest != spool_digest(
            DOMAIN_QUALIFICATION_REQUEST_PLAN, self._unsigned_plan()
        ):
            fail("qualification request plan digest mismatch")

    def outer_unsigned(self) -> dict[str, object]:
        return {
            "artifacts": [row.to_dict() for row in self.artifacts],
            "created_at_unix": self.created_at_unix,
            "deadline_unix": self.deadline_unix,
            "lease": _lease_dict(self.lease),
            "queued_at_unix_ns": self.queued_at_unix_ns,
            "ready_receipt_digest": self.remote_request.ready_receipt_digest,
            "schema": SCHEMA_REQUEST,
            "service_identity": self.remote_request.service_identity,
            "worker_epoch": self.worker_epoch,
            "worker_readiness_digest": self.remote_request.worker_readiness_digest,
        }

    def _unsigned_plan(self) -> dict[str, object]:
        return {
            "artifacts": [row.to_dict() for row in self.artifacts],
            "created_at_unix": self.created_at_unix,
            "credential_digest": self.credential_digest,
            "deadline_unix": self.deadline_unix,
            "lease": _lease_dict(self.lease),
            "queued_at_unix_ns": self.queued_at_unix_ns,
            "registration_digest": self.registration_digest,
            "remote_request": self.remote_request.to_dict(),
            "request_id": self.request_id,
            "schema": SCHEMA_QUALIFICATION_REQUEST_PLAN,
            "transport_identity_digest": self.transport_identity_digest,
            "worker_epoch": self.worker_epoch,
        }

    def request_dict(self) -> dict[str, object]:
        return {**self.outer_unsigned(), "request_id": self.request_id}

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned_plan(), "plan_digest": self.plan_digest}

    @classmethod
    def _build(
        cls,
        *,
        registration_digest: str,
        worker_epoch: str,
        transport_identity_digest: str,
        credential_digest: str,
        remote_request: RemoteEvaluationRequest,
        lease: EvaluationLease,
        artifacts: tuple[PlannedQualificationArtifact, ...],
        created_at_unix: int,
        deadline_unix: int,
        queued_at_unix_ns: int,
        request_id: str,
        plan_digest: str,
    ) -> "QualificationRequestPlan":
        instance = object.__new__(cls)
        for name, value in (
            ("registration_digest", registration_digest),
            ("worker_epoch", worker_epoch),
            ("transport_identity_digest", transport_identity_digest),
            ("credential_digest", credential_digest),
            ("remote_request", remote_request),
            ("lease", lease),
            ("artifacts", artifacts),
            ("created_at_unix", created_at_unix),
            ("deadline_unix", deadline_unix),
            ("queued_at_unix_ns", queued_at_unix_ns),
            ("request_id", request_id),
            ("plan_digest", plan_digest),
        ):
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    @classmethod
    def from_dict(cls, value: object) -> "QualificationRequestPlan":
        row = require_closed(value, _PLAN_FIELDS, "qualification request plan")
        try:
            request = RemoteEvaluationRequest.from_dict(row["remote_request"])
            artifacts_value = row["artifacts"]
            if type(artifacts_value) is not list:
                fail("qualification plan artifacts are not an array")
            artifacts = tuple(
                PlannedQualificationArtifact.from_dict(item)
                for item in artifacts_value
            )
            return cls._build(
                registration_digest=row["registration_digest"],
                worker_epoch=row["worker_epoch"],
                transport_identity_digest=row["transport_identity_digest"],
                credential_digest=row["credential_digest"],
                remote_request=request,
                lease=_lease_from_dict(row["lease"]),
                artifacts=artifacts,
                created_at_unix=row["created_at_unix"],
                deadline_unix=row["deadline_unix"],
                queued_at_unix_ns=row["queued_at_unix_ns"],
                request_id=row["request_id"],
                plan_digest=row["plan_digest"],
            )
        except RemoteWorkerError:
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            fail(f"qualification request plan is malformed: {exc}")


@dataclass(frozen=True)
class QualificationPrepublicationProof:
    """Spool-local proof that no point-of-no-return marker exists."""

    plan_digest: str
    request_id: str
    carrier_materialized: bool

    def __post_init__(self) -> None:
        require_digest(self.plan_digest, "prepublication plan digest")
        require_digest(self.request_id, "prepublication request id")
        if type(self.carrier_materialized) is not bool:
            fail("prepublication carrier state is not exactly boolean")


@dataclass(frozen=True)
class PlannedQualificationObservation:
    """One exact, non-mutating observation of a planned request."""

    plan_digest: str
    request_id: str
    state: Literal[
        "planned_unpublished",
        "carrier_materialized",
        "request_ready",
        "result_ready",
        "completed_response",
    ]
    carrier_path: Path | None
    dispatch_state: str | None = None
    failure_code: str | None = None
    response: AuthenticatedRemoteEvaluationResponse | None = None

    def __post_init__(self) -> None:
        require_digest(self.plan_digest, "planned observation digest")
        require_digest(self.request_id, "planned observation request id")
        states = {
            "planned_unpublished",
            "carrier_materialized",
            "request_ready",
            "result_ready",
            "completed_response",
        }
        if self.state not in states:
            fail("planned observation state is not closed")
        has_carrier = self.carrier_path is not None
        if has_carrier and (
            not isinstance(self.carrier_path, Path)
            or not self.carrier_path.is_absolute()
        ):
            fail("planned observation carrier path is malformed")
        if has_carrier != (self.state != "planned_unpublished"):
            fail("planned observation carrier presence conflicts with its state")
        if self.dispatch_state not in {None, "transferred", "result_received"}:
            fail("planned observation dispatch state is not closed")
        if self.failure_code is not None and (
            not isinstance(self.failure_code, str)
            or not self.failure_code
            or self.failure_code.strip() != self.failure_code
        ):
            fail("planned observation failure code is malformed")
        if self.state == "result_ready":
            if self.failure_code is None or self.response is not None:
                fail("planned result observation is inconsistent")
        elif self.failure_code is not None:
            fail("planned observation has a failure outside result-ready state")
        if self.state == "completed_response":
            if type(self.response) is not AuthenticatedRemoteEvaluationResponse:
                fail("planned completed response is not exactly typed")
        elif self.response is not None:
            fail("planned observation has a response before completion")


def _request_matches_lease(
    request: RemoteEvaluationRequest, lease: EvaluationLease
) -> bool:
    return (
        request.lease_id == lease.lease_id
        and request.generation == lease.generation
        and request.stage == lease.stage
        and request.owner == lease.owner
        and request.members == lease.members
        and request.claimed_block == lease.claimed_block
        and request.initial_expires_block == lease.initial_expires_block
    )


def _hold(
    plan: QualificationRequestPlan, code: str, detail: str
) -> NoReturn:
    raise QualificationRecoveryHold(code, plan.request_id, detail)


def _validate_request_authority(
    request: RemoteEvaluationRequest,
    lease: EvaluationLease,
    registration: Mapping[str, Any],
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> dict[str, Any]:
    verified = verify_registration(registration)
    if (
        type(request) is not RemoteEvaluationRequest
        or type(lease) is not EvaluationLease
        or type(identity) is not RemoteWorkerTransportIdentity
        or type(credential) is not RemoteWorkerCredential
        or request.stage != "qualification"
        or lease.stage != "qualification"
        or not _request_matches_lease(request, lease)
    ):
        fail("qualification plan authority is not exact")
    try:
        verify_remote_request(request, identity, credential)
    except RemoteEvaluationDispatcherError as exc:
        fail(f"qualification request authentication failed: {exc}")
    if (
        verified["transport_identity_digest"] != identity.digest
        or verified["credential_digest"] != credential.digest
        or verified["worker_readiness_digest"] != request.worker_readiness_digest
        or verified["ready_receipt_digest"] != request.ready_receipt_digest
        or verified["service_identity"] != request.service_identity
        or verified["worker_epoch"] == ""
    ):
        fail("qualification request differs from worker registration")
    return verified


def create_qualification_request_plan(
    registration: Mapping[str, Any],
    lease: EvaluationLease,
    request: RemoteEvaluationRequest,
    artifact_inputs: Sequence[tuple[str, Path]],
    *,
    deadline_seconds: int,
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> QualificationRequestPlan:
    """Sample time once and derive one immutable, caller-ID-free plan."""

    verified = _validate_request_authority(
        request, lease, registration, identity, credential
    )
    duration = require_int(
        deadline_seconds,
        "deadline seconds",
        minimum=1,
        maximum=MAX_JOB_SECONDS,
    )
    sources = qualification_source_map(artifact_inputs)
    payload = load_json(sources["qualification_payload"], maximum=MAX_WIRE_PAYLOAD_BYTES)
    if spool_canonical_json(payload) != spool_canonical_json(request.to_dict()):
        fail("qualification payload differs from authenticated request")
    artifacts = tuple(
        PlannedQualificationArtifact(role, digest, size)
        for role in QUALIFICATION_ARTIFACT_ROLES
        for size, digest in (stable_artifact_identity(sources[role]),)
    )
    queued_at = time.time_ns()
    created_at = queued_at // 1_000_000_000
    deadline = created_at + duration
    outer = {
        "artifacts": [row.to_dict() for row in artifacts],
        "created_at_unix": created_at,
        "deadline_unix": deadline,
        "lease": _lease_dict(lease),
        "queued_at_unix_ns": queued_at,
        "ready_receipt_digest": verified["ready_receipt_digest"],
        "schema": SCHEMA_REQUEST,
        "service_identity": verified["service_identity"],
        "worker_epoch": verified["worker_epoch"],
        "worker_readiness_digest": verified["worker_readiness_digest"],
    }
    request_id = spool_digest(DOMAIN_REQUEST, outer)
    unsigned_plan = {
        "artifacts": [row.to_dict() for row in artifacts],
        "created_at_unix": created_at,
        "credential_digest": credential.digest,
        "deadline_unix": deadline,
        "lease": _lease_dict(lease),
        "queued_at_unix_ns": queued_at,
        "registration_digest": verified["registration_digest"],
        "remote_request": request.to_dict(),
        "request_id": request_id,
        "schema": SCHEMA_QUALIFICATION_REQUEST_PLAN,
        "transport_identity_digest": identity.digest,
        "worker_epoch": verified["worker_epoch"],
    }
    return QualificationRequestPlan._build(
        registration_digest=verified["registration_digest"],
        worker_epoch=verified["worker_epoch"],
        transport_identity_digest=identity.digest,
        credential_digest=credential.digest,
        remote_request=request,
        lease=lease,
        artifacts=artifacts,
        created_at_unix=created_at,
        deadline_unix=deadline,
        queued_at_unix_ns=queued_at,
        request_id=request_id,
        plan_digest=spool_digest(DOMAIN_QUALIFICATION_REQUEST_PLAN, unsigned_plan),
    )


def _assert_plan_authority(
    plan: QualificationRequestPlan,
    registration: Mapping[str, Any],
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> None:
    try:
        verified = _validate_request_authority(
            plan.remote_request, plan.lease, registration, identity, credential
        )
    except (RemoteWorkerError, RemoteEvaluationDispatcherError) as exc:
        _hold(plan, "authority_changed", str(exc))
    if (
        verified["registration_digest"] != plan.registration_digest
        or verified["worker_epoch"] != plan.worker_epoch
        or identity.digest != plan.transport_identity_digest
        or credential.digest != plan.credential_digest
    ):
        _hold(plan, "authority_changed", "registration, worker, or credential changed")


def _carrier_paths(plan: QualificationRequestPlan, outbox: Path) -> tuple[Path, Path]:
    return (
        outbox / f"{plan.queued_at_unix_ns:020d}-{plan.request_id}",
        outbox / f".planned-{plan.request_id}",
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _plan_lock(plan: QualificationRequestPlan, outbox: Path) -> Iterator[None]:
    try:
        outbox.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        _hold(plan, "spool_root_changed", f"outbox cannot be opened: {exc}")
    if outbox.is_symlink() or not outbox.is_dir():
        _hold(plan, "spool_root_changed", "outbox is not a regular directory")
    lock_path = outbox / _LOCK_NAME
    if lock_path.is_symlink():
        _hold(plan, "spool_lock_changed", "plan lock is a symlink")
    try:
        lock = lock_path.open("a+b")
    except OSError as exc:
        _hold(plan, "spool_lock_changed", f"plan lock cannot be opened: {exc}")
    with lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _find_duplicate_carriers(
    plan: QualificationRequestPlan, outbox: Path, final: Path, hidden: Path
) -> list[Path]:
    duplicates: list[Path] = []
    suffix = f"-{plan.request_id}"
    for child in outbox.iterdir():
        if child in {final, hidden} or child.name == _LOCK_NAME:
            continue
        suspect = child.name.endswith(suffix)
        request_path = child / "request.json"
        if child.is_dir() and not child.is_symlink() and request_path.is_file():
            try:
                suspect = suspect or load_json(request_path).get("request_id") == plan.request_id
            except RemoteWorkerError:
                if child.name.endswith(suffix):
                    suspect = True
        if suspect:
            duplicates.append(child)
    return duplicates


def _verify_exact_carrier(
    plan: QualificationRequestPlan,
    path: Path,
    registration: Mapping[str, Any],
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        _hold(plan, "carrier_tampered", "carrier is not a regular directory")
    try:
        request = verify_request(
            load_json(path / "request.json"),
            path,
            registration,
            identity=identity,
            credential=credential,
        )
    except RemoteWorkerError as exc:
        _hold(plan, "carrier_tampered", str(exc))
    if spool_canonical_json(request) != spool_canonical_json(plan.request_dict()):
        _hold(plan, "carrier_mismatch", "carrier differs from stored plan")
    return request


def _ready_state(plan: QualificationRequestPlan, carrier: Path) -> bool:
    ready = carrier / "REQUEST_READY"
    if not ready.exists():
        return False
    if ready.is_symlink() or not ready.is_file():
        _hold(plan, "ready_tampered", "REQUEST_READY is not a regular file")
    try:
        value = ready.read_bytes()
    except OSError as exc:
        _hold(plan, "ready_tampered", f"REQUEST_READY is unreadable: {exc}")
    if value != (plan.request_id + "\n").encode():
        _hold(plan, "ready_tampered", "REQUEST_READY binds another request")
    return True


def _dispatch_state(
    plan: QualificationRequestPlan, carrier: Path, *, ready: bool
) -> str | None:
    path = carrier / "dispatch-state.json"
    if not path.exists():
        return None
    if not ready:
        _hold(plan, "dispatch_before_publish", "dispatch state exists without REQUEST_READY")
    try:
        value = load_json(path)
    except RemoteWorkerError as exc:
        _hold(plan, "dispatch_tampered", str(exc))
    fields = set(value)
    if (
        fields not in (
            {"request_id", "schema", "state", "updated_at_unix", "worker_epoch"},
            {
                "archive_sha256",
                "request_id",
                "schema",
                "state",
                "updated_at_unix",
                "worker_epoch",
            },
        )
        or value["schema"] != SCHEMA_DISPATCH_STATE
        or value["request_id"] != plan.request_id
        or value["worker_epoch"] != plan.worker_epoch
        or value["state"] not in {"transferred", "result_received"}
    ):
        _hold(plan, "dispatch_tampered", "dispatch state fields or identity changed")
    try:
        require_int(value["updated_at_unix"], "dispatch update time", minimum=1)
        if "archive_sha256" in value:
            require_digest(value["archive_sha256"], "dispatch archive digest")
    except RemoteWorkerError as exc:
        _hold(plan, "dispatch_tampered", str(exc))
    return value["state"]


def _hidden_state(
    plan: QualificationRequestPlan,
    hidden: Path,
    registration: Mapping[str, Any],
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> Literal["absent", "partial", "complete"]:
    if not hidden.exists():
        return "absent"
    if hidden.is_symlink() or not hidden.is_dir():
        _hold(plan, "hidden_tampered", "hidden carrier is not a regular directory")
    if (hidden / "REQUEST_READY").exists() or (hidden / "dispatch-state.json").exists():
        _hold(plan, "published_hidden", "hidden carrier contains publication evidence")
    request_path = hidden / "request.json"
    if request_path.exists():
        _verify_exact_carrier(plan, hidden, registration, identity, credential)
        return "complete"
    allowed = {"blobs"}
    for child in hidden.iterdir():
        if child.name in allowed or child.name.startswith(".request.json."):
            continue
        _hold(plan, "hidden_tampered", "hidden carrier contains unexpected state")
    blobs = hidden / "blobs"
    if blobs.exists():
        if blobs.is_symlink() or not blobs.is_dir():
            _hold(plan, "hidden_tampered", "hidden blobs path changed type")
        expected = {row.sha256 for row in plan.artifacts}
        if any(child.name not in expected for child in blobs.iterdir()):
            _hold(plan, "hidden_tampered", "hidden carrier contains an unknown blob")
    return "partial"


def _local_result(
    plan: QualificationRequestPlan,
    carrier: Path | None,
    results_root: Path,
    registration: Mapping[str, Any],
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> tuple[str | None, str | None, AuthenticatedRemoteEvaluationResponse | None]:
    root = results_root / plan.request_id
    if not root.exists():
        return None, None, None
    if carrier is None:
        _hold(plan, "published_carrier_missing", "local result exists without its carrier")
    if root.is_symlink() or not root.is_dir():
        _hold(plan, "result_tampered", "local result is not a regular directory")
    ready = root / "RESULT_READY"
    if ready.is_symlink() or not ready.is_file():
        _hold(plan, "result_partial", "local result lacks an exact RESULT_READY")
    try:
        ready_bytes = ready.read_bytes()
    except OSError as exc:
        _hold(plan, "result_tampered", f"RESULT_READY is unreadable: {exc}")
    if ready_bytes != (plan.request_id + "\n").encode():
        _hold(plan, "result_tampered", "RESULT_READY binds another request")
    try:
        result = verify_adapter_result(
            load_json(root / "result.json"),
            root,
            plan.request_dict(),
            registration,
            request_root=carrier,
            identity=identity,
            credential=credential,
        )
    except RemoteWorkerError as exc:
        _hold(plan, "result_tampered", str(exc))
    if result["state"] != "completed":
        return "result_ready", result["failure_code"], None
    try:
        response = AuthenticatedRemoteEvaluationResponse.from_dict(
            load_json(artifact_for_role(result, root, "adapter_result"), maximum=64 << 20)
        )
    except (RemoteWorkerError, RemoteEvaluationDispatcherError) as exc:
        _hold(plan, "result_tampered", str(exc))
    return "completed_response", None, response


def _inspect_locked(
    plan: QualificationRequestPlan,
    outbox: Path,
    results_root: Path,
    registration: Mapping[str, Any],
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> PlannedQualificationObservation:
    final, hidden = _carrier_paths(plan, outbox)
    duplicates = _find_duplicate_carriers(plan, outbox, final, hidden)
    if duplicates:
        _hold(plan, "duplicate_carrier", "multiple paths claim the planned request id")
    hidden_status = _hidden_state(plan, hidden, registration, identity, credential)
    if final.exists() and hidden_status != "absent":
        _hold(plan, "duplicate_carrier", "final and hidden carriers both exist")
    carrier: Path | None = None
    ready = False
    dispatch: str | None = None
    if final.exists():
        _verify_exact_carrier(plan, final, registration, identity, credential)
        carrier = final
        ready = _ready_state(plan, final)
        dispatch = _dispatch_state(plan, final, ready=ready)
    result_state, failure, response = _local_result(
        plan, carrier, results_root, registration, identity, credential
    )
    if result_state is not None and not ready:
        _hold(plan, "result_before_publish", "local result exists before REQUEST_READY")
    if dispatch == "result_received" and result_state is None:
        _hold(plan, "result_missing", "dispatch says result_received but result is absent")
    if response is not None:
        state = "completed_response"
    elif result_state is not None:
        state = "result_ready"
    elif ready:
        state = "request_ready"
    elif carrier is not None:
        state = "carrier_materialized"
    else:
        state = "planned_unpublished"
    return PlannedQualificationObservation(
        plan.plan_digest,
        plan.request_id,
        state,
        carrier,
        dispatch,
        failure,
        response,
    )


def inspect_planned_qualification(
    plan: QualificationRequestPlan,
    outbox: Path,
    results_root: Path,
    registration: Mapping[str, Any],
    *,
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> PlannedQualificationObservation:
    _assert_plan_authority(plan, registration, identity, credential)
    with _plan_lock(plan, outbox):
        return _inspect_locked(
            plan, outbox, results_root, registration, identity, credential
        )


def _prepublication_locked(
    plan: QualificationRequestPlan,
    outbox: Path,
    results_root: Path,
    registration: Mapping[str, Any],
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> QualificationPrepublicationProof:
    observed = _inspect_locked(
        plan, outbox, results_root, registration, identity, credential
    )
    if observed.state not in {"planned_unpublished", "carrier_materialized"}:
        _hold(plan, "already_published", "prepublication proof follows durable evidence")
    return QualificationPrepublicationProof(
        plan.plan_digest,
        plan.request_id,
        observed.state == "carrier_materialized",
    )


def prove_planned_qualification_prepublication(
    plan: QualificationRequestPlan,
    outbox: Path,
    results_root: Path,
    registration: Mapping[str, Any],
    *,
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> QualificationPrepublicationProof:
    _assert_plan_authority(plan, registration, identity, credential)
    with _plan_lock(plan, outbox):
        return _prepublication_locked(
            plan, outbox, results_root, registration, identity, credential
        )


def materialize_planned_qualification(
    plan: QualificationRequestPlan,
    artifact_inputs: Sequence[tuple[str, Path]],
    outbox: Path,
    results_root: Path,
    registration: Mapping[str, Any],
    *,
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> PlannedQualificationObservation:
    """Create or exactly reopen the plan's carrier, without publishing it."""

    _assert_plan_authority(plan, registration, identity, credential)
    try:
        sources = qualification_source_map(artifact_inputs)
        identities = tuple(
            stable_artifact_identity(sources[expected.role])
            for expected in plan.artifacts
        )
    except RemoteWorkerError as exc:
        _hold(plan, "artifact_changed", str(exc))
    if any(
        observed != (expected.size, expected.sha256)
        for observed, expected in zip(identities, plan.artifacts, strict=True)
    ):
        _hold(plan, "artifact_changed", "materialization input differs from plan")
    with _plan_lock(plan, outbox):
        observed = _inspect_locked(
            plan, outbox, results_root, registration, identity, credential
        )
        if observed.state != "planned_unpublished":
            return observed
        final, hidden = _carrier_paths(plan, outbox)
        hidden_status = _hidden_state(plan, hidden, registration, identity, credential)
        if hidden_status == "complete":
            os.replace(hidden, final)
            _fsync_directory(outbox)
            return _inspect_locked(
                plan, outbox, results_root, registration, identity, credential
            )
        if hidden_status == "partial":
            _prepublication_locked(
                plan, outbox, results_root, registration, identity, credential
            )
            shutil.rmtree(hidden)
            _fsync_directory(outbox)
        hidden.mkdir(mode=0o700)
        blobs = hidden / "blobs"
        blobs.mkdir(mode=0o700)
        for expected in plan.artifacts:
            try:
                copy_stable_artifact(
                    sources[expected.role],
                    blobs / expected.sha256,
                    expected_size=expected.size,
                    expected_sha256=expected.sha256,
                )
            except RemoteWorkerError as exc:
                _hold(plan, "artifact_changed", str(exc))
        atomic_json(hidden / "request.json", plan.request_dict(), mode=0o400)
        _verify_exact_carrier(plan, hidden, registration, identity, credential)
        _fsync_directory(blobs)
        _fsync_directory(hidden)
        os.replace(hidden, final)
        _fsync_directory(outbox)
        return _inspect_locked(
            plan, outbox, results_root, registration, identity, credential
        )


def publish_planned_qualification(
    plan: QualificationRequestPlan,
    outbox: Path,
    results_root: Path,
    registration: Mapping[str, Any],
    *,
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> PlannedQualificationObservation:
    """Idempotently publish the one exact materialized carrier."""

    _assert_plan_authority(plan, registration, identity, credential)
    with _plan_lock(plan, outbox):
        observed = _inspect_locked(
            plan, outbox, results_root, registration, identity, credential
        )
        if observed.state == "planned_unpublished":
            _hold(plan, "carrier_missing", "cannot publish before exact materialization")
        if observed.state == "carrier_materialized":
            assert observed.carrier_path is not None
            atomic_bytes(
                observed.carrier_path / "REQUEST_READY",
                (plan.request_id + "\n").encode(),
                mode=0o400,
            )
        return _inspect_locked(
            plan, outbox, results_root, registration, identity, credential
        )


__all__ = [
    "PlannedQualificationObservation",
    "QualificationPrepublicationProof",
    "QualificationRecoveryHold",
    "QualificationRequestPlan",
    "create_qualification_request_plan",
    "inspect_planned_qualification",
    "materialize_planned_qualification",
    "prove_planned_qualification_prepublication",
    "publication_archive",
    "publish_planned_qualification",
]
