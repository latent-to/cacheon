"""CPU-side SSH shuttle between the durable spool and one registered pod.

This module owns endpoint-pinned transport only: packing sealed request
carriers, transferring them to the registered pod over host-key-pinned SSH,
pulling verified result archives back, and the standing CPU transfer loop.
``DurableSpoolAuthenticatedWorkerTransport`` implements the tracked
``AuthenticatedWorkerTransport`` protocol for both screen and qualification
work; qualification additionally requires an injected publication resolver so
the CPU-owned immutable publication bytes accompany the authenticated wire
request.  Nothing here evaluates, settles, or manufactures a candidate
``FAIL``; transport failures surface as infrastructure errors that release
the durable lease without consuming an attempt.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cacheon.chain.evaluation_coordinator import ClaimedScreenEvaluation
from cacheon.chain.evaluation_leases import EvaluationLease
from cacheon.chain.remote_evaluation_dispatcher import (
    AuthenticatedRemoteEvaluationResponse,
    RemoteEvaluationDispatcherError,
    RemoteEvaluationRequest,
    verify_remote_request,
)
from cacheon.chain.remote_worker_artifact_recovery import publication_archive
from cacheon.chain.remote_worker_request_plan import (
    PlannedQualificationObservation,
    QualificationPrepublicationProof,
    QualificationRecoveryHold,
    QualificationRequestPlan,
    create_qualification_request_plan,
    inspect_planned_qualification as inspect_qualification_plan,
    materialize_planned_qualification as materialize_qualification_plan,
    prove_planned_qualification_prepublication as prove_qualification_plan,
    publish_planned_qualification as publish_qualification_plan,
)
from cacheon.chain.remote_worker_registration import (
    registration_credential,
    registration_is_current,
    registration_transport_identity,
    verify_registration,
)
from cacheon.chain.remote_worker_spool import (
    MAX_JOB_SECONDS,
    SAFE_ID,
    append_event,
    artifact_for_role,
    atomic_bytes,
    atomic_json,
    enqueue_request,
    fail,
    file_sha256,
    heartbeat_payload,
    iter_queue,
    load_json,
    pack_directory,
    require_digest,
    require_int,
    safe_extract,
    spool_canonical_json,
    strict_json_object,
    verify_adapter_result,
    verify_heartbeat,
    verify_result_ready,
    write_dispatch_state,
    write_local_no_decision,
    RemoteWorkerError,
)


DEFAULT_POLL_SECONDS = 5
DEFAULT_HEARTBEAT_SECONDS = 10
DEFAULT_MAX_HEARTBEAT_AGE = 45


@dataclass(frozen=True)
class RemotePodSite:
    """Closed pod-side installation coordinates used by the CPU shuttle."""

    pod_root: str
    service_path: str

    def __post_init__(self) -> None:
        for name in ("pod_root", "service_path"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.startswith("/")
                or SAFE_ID.fullmatch(value) is None
            ):
                fail(f"pod site {name} must be a closed absolute path")


def ssh_base(registration: Mapping[str, Any]) -> list[str]:
    known_hosts = Path(registration["known_hosts_path"])
    if known_hosts.is_symlink() or not known_hosts.is_file():
        fail("registered known_hosts is missing")
    if file_sha256(known_hosts) != registration["known_hosts_sha256"]:
        fail("registered known_hosts bytes changed")
    return [
        "ssh",
        "-p",
        str(registration["pod_port"]),
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=2",
        f"{registration['pod_user']}@{registration['pod_host']}",
    ]


def scp_base(registration: Mapping[str, Any]) -> list[str]:
    return [
        "scp",
        "-q",
        "-P",
        str(registration["pod_port"]),
        "-o",
        f"UserKnownHostsFile={registration['known_hosts_path']}",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
    ]


def run_checked(
    command: list[str], *, timeout: int = 60, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        fail(f"closed transport command failed: {type(exc).__name__}: {exc}")


def remote_python_command(
    registration: Mapping[str, Any],
    site: RemotePodSite,
    *arguments: str,
) -> list[str]:
    for value in arguments:
        if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
            fail("remote service argument is not closed")
    return [registration["python_executable"], site.service_path, *arguments]


def remote_json(
    registration: Mapping[str, Any],
    site: RemotePodSite,
    mode: str,
    request_id: str,
) -> dict[str, Any] | None:
    require_digest(request_id, "request id")
    completed = run_checked(
        ssh_base(registration)
        + remote_python_command(registration, site, mode, "--request-id", request_id),
        timeout=30,
        capture=True,
    )
    output = completed.stdout.strip()
    if not output:
        return None
    return strict_json_object(output)


def transfer_request(
    registration: Mapping[str, Any],
    site: RemotePodSite,
    job_dir: Path,
    request: Mapping[str, Any],
) -> None:
    request_id = request["request_id"]
    archive_dir = job_dir / "transport"
    archive_dir.mkdir(exist_ok=True, mode=0o700)
    archive = archive_dir / f"{request_id}.tar"
    if not archive.exists():
        archive_sha = pack_directory(
            job_dir, "request.json", request["artifacts"], archive
        )
    else:
        archive_sha = file_sha256(archive)
    remote_part = f"{site.pod_root}/incoming/.{request_id}.{archive_sha}.tar.part"
    run_checked(
        scp_base(registration)
        + ["--", str(archive), f"root@{registration['pod_host']}:{remote_part}"],
        timeout=1800,
    )
    run_checked(
        ssh_base(registration)
        + remote_python_command(
            registration,
            site,
            "accept-request",
            "--request-id",
            request_id,
            "--archive-sha256",
            archive_sha,
            "--archive-size",
            str(archive.stat().st_size),
        ),
        timeout=120,
    )
    write_dispatch_state(
        job_dir,
        request_id,
        "transferred",
        registration["worker_epoch"],
        archive_sha256=archive_sha,
    )


def pull_result(
    registration: Mapping[str, Any],
    site: RemotePodSite,
    job_dir: Path,
    request: Mapping[str, Any],
    results_root: Path,
    *,
    identity,
    credential,
) -> bool:
    request_id = request["request_id"]
    ready = remote_json(registration, site, "result-status", request_id)
    if ready is None or ready.get("state") != "ready":
        return False
    ready = verify_result_ready(ready, request, registration)
    archive_sha = ready["archive_sha256"]
    archive_size = ready["archive_size"]
    incoming = results_root / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive = incoming / f"{request_id}.{archive_sha}.tar"
    if not archive.exists():
        remote = (
            f"root@{registration['pod_host']}:"
            f"{site.pod_root}/outgoing/{request_id}.{archive_sha}.tar"
        )
        run_checked(scp_base(registration) + ["--", remote, str(archive)], timeout=1800)
    if archive.stat().st_size != archive_size or file_sha256(archive) != archive_sha:
        fail("result archive bytes differ from pod receipt")
    temporary = results_root / f".{request_id}.extracting"
    if temporary.exists():
        shutil.rmtree(temporary)
    safe_extract(archive, temporary)
    result = verify_adapter_result(
        load_json(temporary / "result.json"),
        temporary,
        request,
        registration,
        request_root=job_dir,
        identity=identity,
        credential=credential,
    )
    final = results_root / request_id
    if final.exists():
        existing = verify_adapter_result(
            load_json(final / "result.json"),
            final,
            request,
            registration,
            request_root=job_dir,
            identity=identity,
            credential=credential,
        )
        if spool_canonical_json(existing) != spool_canonical_json(result):
            fail("local result identity collision")
        shutil.rmtree(temporary)
    else:
        os.replace(temporary, final)
        atomic_bytes(final / "RESULT_READY", (request_id + "\n").encode(), mode=0o400)
    run_checked(
        ssh_base(registration)
        + remote_python_command(
            registration,
            site,
            "ack-result",
            "--request-id",
            request_id,
            "--archive-sha256",
            archive_sha,
        ),
        timeout=60,
    )
    write_dispatch_state(
        job_dir,
        request_id,
        "result_received",
        registration["worker_epoch"],
        archive_sha256=archive_sha,
    )
    return True


def remote_heartbeat(
    registration: Mapping[str, Any], site: RemotePodSite, max_age: int
) -> dict[str, Any]:
    completed = run_checked(
        ssh_base(registration)
        + remote_python_command(registration, site, "heartbeat-status"),
        timeout=30,
        capture=True,
    )
    output = completed.stdout.strip()
    if not output:
        fail("pod worker heartbeat is absent")
    return verify_heartbeat(strict_json_object(output), registration, max_age)


def _observe_worker_hold(
    previous_state: str | None,
    heartbeat: Mapping[str, Any],
    spool_root: Path,
    registration: Mapping[str, Any],
) -> str | None:
    """Track pod hold states without ever exiting the relay.

    ``epoch_failed`` (legacy latch) and ``adapter_cooldown`` are worker-side
    conditions; the relay's job is transport, and the pod's ``accept-request``
    and ``pull-result`` verbs keep working through both.  Exiting here is what
    turned one adapter fault into a dead dispatcher and a stalled spool on
    mainnet 2026-08-10.  One event per transition keeps the spool log quiet.
    """
    state = heartbeat["state"]
    if state not in ("epoch_failed", "adapter_cooldown"):
        return None
    if state != previous_state:
        append_event(
            spool_root,
            "dispatcher_worker_cooldown",
            adapter_start_count=heartbeat["adapter_start_count"],
            consecutive_adapter_failures=heartbeat[
                "consecutive_adapter_failures"
            ],
            worker_epoch=registration["worker_epoch"],
            worker_state=state,
        )
    return state


def cpu_serve(
    *,
    registration_path: Path,
    current_registration_path: Path,
    spool_root: Path,
    site: RemotePodSite,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    max_heartbeat_age: int = DEFAULT_MAX_HEARTBEAT_AGE,
) -> None:
    """Standing CPU transfer loop: shuttle sealed requests and results."""

    registration = verify_registration(load_json(registration_path))
    identity = registration_transport_identity(registration)
    credential = registration_credential(
        registration, Path(registration["credential_path"])
    )
    outbox = spool_root / "outbox"
    results = spool_root / "results"
    state_root = spool_root / "state"
    for path in (outbox, results, state_root):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = state_root / f"dispatch-{registration['worker_epoch']}.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("another dispatcher already owns this worker epoch")
        append_event(
            spool_root, "dispatcher_started", worker_epoch=registration["worker_epoch"]
        )
        worker_hold_state: str | None = None
        while True:
            if not registration_is_current(registration, current_registration_path):
                append_event(
                    spool_root,
                    "dispatcher_superseded",
                    worker_epoch=registration["worker_epoch"],
                )
                return
            try:
                heartbeat = remote_heartbeat(registration, site, max_heartbeat_age)
                worker_hold_state = _observe_worker_hold(
                    worker_hold_state, heartbeat, spool_root, registration
                )
                queue = iter_queue(
                    outbox, registration, identity=identity, credential=credential
                )
                active = heartbeat.get("active_request_id")
                for job_dir, request in queue:
                    request_id = request["request_id"]
                    state_path = job_dir / "dispatch-state.json"
                    state = load_json(state_path) if state_path.exists() else None
                    if state is not None and state.get("state") == "result_received":
                        continue
                    local_result = results / request_id
                    if (local_result / "RESULT_READY").is_file():
                        verify_adapter_result(
                            load_json(local_result / "result.json"),
                            local_result,
                            request,
                            registration,
                            request_root=job_dir,
                            identity=identity,
                            credential=credential,
                        )
                        write_dispatch_state(
                            job_dir,
                            request_id,
                            "result_received",
                            registration["worker_epoch"],
                        )
                        continue
                    if pull_result(
                        registration,
                        site,
                        job_dir,
                        request,
                        results,
                        identity=identity,
                        credential=credential,
                    ):
                        append_event(
                            spool_root, "result_received", request_id=request_id
                        )
                        active = None
                        continue
                    if active not in {None, request_id}:
                        break
                    if state is None or state.get("state") != "transferred":
                        if request["deadline_unix"] <= int(time.time()):
                            write_local_no_decision(
                                results, request, "request_deadline_elapsed"
                            )
                            write_dispatch_state(
                                job_dir,
                                request_id,
                                "result_received",
                                registration["worker_epoch"],
                            )
                            append_event(
                                spool_root,
                                "request_expired_before_transfer",
                                request_id=request_id,
                            )
                            continue
                        transfer_request(registration, site, job_dir, request)
                        append_event(
                            spool_root, "request_transferred", request_id=request_id
                        )
                    break
                atomic_json(
                    state_root / "heartbeat.json",
                    heartbeat_payload(registration, "running", active),
                )
            except RemoteWorkerError as exc:
                append_event(spool_root, "dispatcher_retry", error=str(exc)[:1024])
            time.sleep(poll_seconds)


class DurableSpoolAuthenticatedWorkerTransport:
    """AuthenticatedWorkerTransport backed by the standing VM/pod spool.

    ``RemoteEvaluationDispatcher`` owns the durable lease and its finalized-
    block heartbeat while a call waits; the separate standing CPU transfer
    loop moves the sealed carriers.  Screen and qualification are both
    supported through the same registered epoch: qualification additionally
    requires ``qualification_publication_resolver`` so the CPU-owned
    immutable publication accompanies the authenticated wire request, and it
    fails closed when the resolver is not configured.
    """

    def __init__(
        self,
        *,
        registration_path: str | os.PathLike[str],
        spool_root: str | os.PathLike[str],
        credential_path: str | os.PathLike[str],
        qualification_publication_resolver: Callable[[object], Sequence[object]]
        | None = None,
        response_timeout_seconds: int = MAX_JOB_SECONDS,
        poll_seconds: int = 2,
    ):
        self.registration_path = Path(registration_path)
        self.registration = verify_registration(load_json(self.registration_path))
        self.spool_root = Path(spool_root)
        self.credential_path = Path(credential_path)
        if self.credential_path != Path(self.registration["credential_path"]):
            fail("transport credential path differs from registration")
        self.credential = registration_credential(
            self.registration, self.credential_path
        )
        self.identity = registration_transport_identity(self.registration)
        if (
            qualification_publication_resolver is not None
            and not callable(qualification_publication_resolver)
        ):
            fail("qualification publication resolver is not callable")
        self.qualification_publication_resolver = qualification_publication_resolver
        self.response_timeout_seconds = require_int(
            response_timeout_seconds,
            "response timeout seconds",
            minimum=1,
            maximum=MAX_JOB_SECONDS,
        )
        self.poll_seconds = require_int(
            poll_seconds, "transport poll seconds", minimum=1, maximum=30
        )
        for path in (
            self.spool_root / "outbox",
            self.spool_root / "results",
            self.spool_root / "state",
        ):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)

    @staticmethod
    def _lease_dict(lease: object) -> dict[str, Any]:
        if type(lease) is not EvaluationLease:
            fail("spool transport lease is not exactly typed")
        return {
            "claimed_block": lease.claimed_block,
            "expires_block": lease.expires_block,
            "generation": lease.generation,
            "initial_expires_block": lease.initial_expires_block,
            "lease_id": lease.lease_id,
            "members": [row.to_dict() for row in lease.members],
            "owner": lease.owner,
            "stage": lease.stage,
        }

    def _require_dispatcher_liveness(self) -> None:
        heartbeat_path = self.spool_root / "state" / "heartbeat.json"
        if not heartbeat_path.exists():
            fail("standing CPU transfer dispatcher has no heartbeat")
        verify_heartbeat(
            load_json(heartbeat_path),
            self.registration,
            DEFAULT_MAX_HEARTBEAT_AGE,
        )

    def _await_completed_result(
        self, request_id: str, job_dir: Path, *, stage: str
    ) -> AuthenticatedRemoteEvaluationResponse:
        deadline = time.monotonic() + self.response_timeout_seconds
        result_root = self.spool_root / "results" / request_id
        while time.monotonic() < deadline:
            ready = result_root / "RESULT_READY"
            if ready.is_file() and not ready.is_symlink():
                result = verify_adapter_result(
                    load_json(result_root / "result.json"),
                    result_root,
                    load_json(job_dir / "request.json"),
                    self.registration,
                    request_root=job_dir,
                    identity=self.identity,
                    credential=self.credential,
                )
                if result["state"] != "completed":
                    raise RemoteEvaluationDispatcherError(
                        f"remote {stage} infrastructure: {result['failure_code']}"
                    )
                response_path = artifact_for_role(result, result_root, "adapter_result")
                return AuthenticatedRemoteEvaluationResponse.from_dict(
                    load_json(response_path, maximum=64 << 20)
                )
            self._require_dispatcher_liveness()
            time.sleep(self.poll_seconds)
        raise RemoteEvaluationDispatcherError(
            f"remote {stage} response exceeded the transport deadline"
        )

    def run_screen(self, request, *, job):
        if type(request) is not RemoteEvaluationRequest or type(job) is not ClaimedScreenEvaluation:
            raise RemoteEvaluationDispatcherError(
                "spool screen transport requires exact request/job types"
            )
        try:
            verify_remote_request(request, self.identity, self.credential)
            self._require_dispatcher_liveness()
            scratch = Path(
                tempfile.mkdtemp(prefix=".screen.", dir=self.spool_root / "state")
            )
            try:
                wire_path = scratch / "screen-request.json"
                atomic_json(wire_path, request.to_dict(), mode=0o400)
                publication_path = scratch / "candidate-publication.tar"
                publication_archive(job.publication, publication_path)
                request_id, job_dir = enqueue_request(
                    self.registration,
                    self._lease_dict(job.lease),
                    (
                        ("screen_payload", wire_path),
                        ("candidate_publication", publication_path),
                    ),
                    self.spool_root / "outbox",
                    deadline_seconds=self.response_timeout_seconds,
                    identity=self.identity,
                    credential=self.credential,
                )
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
            return self._await_completed_result(request_id, job_dir, stage="screen")
        except RemoteWorkerError as exc:
            raise RemoteEvaluationDispatcherError(str(exc)) from exc

    def _with_qualification_artifacts(self, request, operation):
        """Rebuild exact path-free inputs and invoke one non-network operation."""

        if type(request) is not RemoteEvaluationRequest or request.stage != "qualification":
            raise RemoteEvaluationDispatcherError(
                "spool qualification transport requires an exact qualification request"
            )
        resolver = self.qualification_publication_resolver
        if resolver is None:
            raise RemoteEvaluationDispatcherError(
                "spool qualification publication authority is not configured"
            )
        verify_remote_request(request, self.identity, self.credential)
        publications = tuple(resolver(request))
        candidates = request.body["candidates"]
        if (
            len(publications) != 1
            or type(candidates) is not list
            or len(candidates) != 1
            or type(candidates[0]) is not dict
            or publications[0].to_dict() != candidates[0]["publication"]
        ):
            raise RemoteEvaluationDispatcherError(
                "resolved qualification publication differs from authenticated work"
            )
        scratch = Path(
            tempfile.mkdtemp(prefix=".qualification.", dir=self.spool_root / "state")
        )
        try:
            wire_path = scratch / "qualification-request.json"
            atomic_json(wire_path, request.to_dict(), mode=0o400)
            publication_path = scratch / "candidate-publication.tar"
            publication_archive(publications[0], publication_path)
            lease = EvaluationLease(
                request.lease_id,
                request.generation,
                request.stage,
                request.owner,
                request.members,
                request.claimed_block,
                request.initial_expires_block,
                request.initial_expires_block,
            )
            return operation(
                lease,
                (
                    ("qualification_payload", wire_path),
                    ("candidate_publication", publication_path),
                ),
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def plan_qualification_request(self, request) -> QualificationRequestPlan:
        """Derive one plan; do not materialize or publish its carrier."""

        return self._with_qualification_artifacts(
            request,
            lambda lease, artifacts: create_qualification_request_plan(
                self.registration,
                lease,
                request,
                artifacts,
                deadline_seconds=self.response_timeout_seconds,
                identity=self.identity,
                credential=self.credential,
            ),
        )

    def materialize_planned_qualification(
        self, plan: QualificationRequestPlan, request: RemoteEvaluationRequest
    ) -> PlannedQualificationObservation:
        return self._with_qualification_artifacts(
            request,
            lambda _lease, artifacts: materialize_qualification_plan(
                plan,
                artifacts,
                self.spool_root / "outbox",
                self.spool_root / "results",
                self.registration,
                identity=self.identity,
                credential=self.credential,
            ),
        )

    def inspect_planned_qualification(
        self, plan: QualificationRequestPlan
    ) -> PlannedQualificationObservation:
        return inspect_qualification_plan(
            plan,
            self.spool_root / "outbox",
            self.spool_root / "results",
            self.registration,
            identity=self.identity,
            credential=self.credential,
        )

    def prove_planned_qualification_prepublication(
        self, plan: QualificationRequestPlan
    ) -> QualificationPrepublicationProof:
        return prove_qualification_plan(
            plan,
            self.spool_root / "outbox",
            self.spool_root / "results",
            self.registration,
            identity=self.identity,
            credential=self.credential,
        )

    def publish_planned_qualification(
        self, plan: QualificationRequestPlan
    ) -> PlannedQualificationObservation:
        return publish_qualification_plan(
            plan,
            self.spool_root / "outbox",
            self.spool_root / "results",
            self.registration,
            identity=self.identity,
            credential=self.credential,
        )

    def resume_planned_qualification(
        self, plan: QualificationRequestPlan
    ) -> AuthenticatedRemoteEvaluationResponse:
        observed = self.inspect_planned_qualification(plan)
        if observed.response is not None:
            return observed.response
        if observed.state == "result_ready":
            raise RemoteEvaluationDispatcherError(
                f"remote qualification infrastructure: {observed.failure_code}"
            )
        if observed.state != "request_ready" or observed.carrier_path is None:
            raise QualificationRecoveryHold(
                "request_not_published",
                plan.request_id,
                "resume requires the exact published carrier",
            )
        return self._await_completed_result(
            plan.request_id, observed.carrier_path, stage="qualification"
        )

    def run_qualification(self, request):
        if type(request) is not RemoteEvaluationRequest or request.stage != "qualification":
            raise RemoteEvaluationDispatcherError(
                "spool qualification transport requires an exact qualification request"
            )
        resolver = self.qualification_publication_resolver
        if resolver is None:
            raise RemoteEvaluationDispatcherError(
                "spool qualification publication authority is not configured"
            )
        try:
            verify_remote_request(request, self.identity, self.credential)
            self._require_dispatcher_liveness()
            publications = tuple(resolver(request))
            if len(publications) != 1:
                raise RemoteEvaluationDispatcherError(
                    "resident-v3 qualification transport requires one publication"
                )
            publication = publications[0]
            candidates = request.body["candidates"]
            if (
                type(candidates) is not list
                or len(candidates) != 1
                or type(candidates[0]) is not dict
                or publication.to_dict() != candidates[0]["publication"]
            ):
                raise RemoteEvaluationDispatcherError(
                    "resolved qualification publication differs from authenticated work"
                )
            scratch = Path(
                tempfile.mkdtemp(
                    prefix=".qualification.", dir=self.spool_root / "state"
                )
            )
            try:
                wire_path = scratch / "qualification-request.json"
                atomic_json(wire_path, request.to_dict(), mode=0o400)
                publication_path = scratch / "candidate-publication.tar"
                publication_archive(publication, publication_path)
                lease = EvaluationLease(
                    request.lease_id,
                    request.generation,
                    request.stage,
                    request.owner,
                    request.members,
                    request.claimed_block,
                    request.initial_expires_block,
                    request.initial_expires_block,
                )
                request_id, job_dir = enqueue_request(
                    self.registration,
                    self._lease_dict(lease),
                    (
                        ("qualification_payload", wire_path),
                        ("candidate_publication", publication_path),
                    ),
                    self.spool_root / "outbox",
                    deadline_seconds=self.response_timeout_seconds,
                    identity=self.identity,
                    credential=self.credential,
                )
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
            return self._await_completed_result(
                request_id, job_dir, stage="qualification"
            )
        except RemoteWorkerError as exc:
            raise RemoteEvaluationDispatcherError(str(exc)) from exc
