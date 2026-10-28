"""Move qualification progress through the existing CPU/pod spool.

The CPU keeps immutable snapshots and retires the old lease before dispatching
replacement work. Completed local results still import through their original
worker authority; a new worker is not permission to repeat completed work.
"""

from __future__ import annotations

import copy
import fcntl
import os

from cacheon.chain.remote_worker_spool import (
    append_event, artifact_for_role, atomic_bytes, atomic_json, authenticated_wire_request,
    file_sha256, load_json, require_digest, spool_canonical_json, fail, MAX_WIRE_PAYLOAD_BYTES,
)
from cacheon.chain.remote_worker_registration import (
    registration_credential, registration_transport_identity, verify_registration,
    verify_pod_registration,
)
from cacheon.stack_identity import canonical_digest


def _work_key(body):
    # A new commissioning changes service/policy and their derived stack IDs.
    # The worker compares the actual executable and scoring contract on restore.
    return canonical_digest("cacheon.qualification.checkpoint-work.v2", {
        "screen_lane": body["screen_lane"],
        "candidates": [row["candidate_digest"] for row in body["candidates"]],
    })


def retain_authority(transport, manifest):
    """Keep an epoch's worker and arena authority so its results import after replacement."""
    root = transport.spool_root / "checkpoints" / "authorities" / transport.registration["worker_epoch"]
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, data in (("registration.json", spool_canonical_json(transport.registration)),
                       ("credential.secret", transport.credential_path.read_bytes()),
                       ("service.json", spool_canonical_json(manifest.to_dict()))):
        path = root / name
        if path.exists():
            if path.read_bytes() != data:
                if name != "registration.json" or any(
                    load_json(path)[field] != transport.registration[field]
                    for field in ("transport_identity_digest", "worker_readiness_digest",
                                  "credential_digest", "ready_receipt_digest", "service_identity", "worker_epoch")):
                    fail("retained worker verification authority changed")
        else:
            atomic_bytes(path, data, mode=0o400)


def checkpoint_status(paths, request_id):
    """Expose one authenticated request's published checkpoint to its CPU relay."""
    from cacheon.chain.remote_evaluation_dispatcher import RemoteEvaluationRequest

    request_id = require_digest(request_id, "checkpoint request id")
    registration = verify_pod_registration(paths)
    jobs = [paths.root / state / request_id for state in ("processing", "completed")]
    jobs = [path for path in jobs if path.is_dir()]
    if not jobs:
        return
    if len(jobs) != 1:
        fail("checkpoint request has ambiguous live/completed carriers")
    job, = jobs
    request = load_json(job / "request.json")
    wire = RemoteEvaluationRequest.from_dict(authenticated_wire_request(request, job,
        identity=registration_transport_identity(registration),
        credential=registration_credential(registration, paths.credential)))
    pointer = paths.root / "checkpoints" / (wire.digest + ".json")
    if pointer.is_file():
        import sys

        row = load_json(pointer)
        row.update(request_id=request_id, worker_epoch=registration["worker_epoch"])
        sys.stdout.buffer.write(spool_canonical_json(row) + b"\n")


def pull_checkpoint(registration, site, job, request, spool_root, *, identity, credential):
    """Copy, verify and fsync progress before it becomes a recovery choice."""
    from cacheon.chain.remote_evaluation_dispatcher import RemoteEvaluationRequest
    from cacheon.chain.ssh_worker_transport import remote_json, copy_remote_file
    from cacheon.eval.qualification_worker_checkpoint import read_snapshot

    ready = remote_json(registration, site, "checkpoint-status", request["request_id"])
    if ready is None:
        return
    wire = RemoteEvaluationRequest.from_dict(authenticated_wire_request(
        request, job, identity=identity, credential=credential))
    digest = require_digest(ready.get("sha256"), "checkpoint digest")
    if (ready.get("worker_epoch") != registration["worker_epoch"]
            or ready.get("request_id") != request["request_id"]
            or ready.get("request_digest") != wire.digest):
        fail("checkpoint differs from the active authenticated request")
    root = spool_root / "checkpoints"
    (root / "index").mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = root / (digest + ".json.gz")
    if not destination.exists():
        temporary = root / ("." + digest + ".part")
        copy_remote_file(registration, temporary, f"{site.pod_root}/checkpoints/{digest}.json.gz",
                         upload=False, timeout=180)
        snapshot = read_snapshot(temporary, digest)
        if snapshot["origin"]["request"] != wire.digest or temporary.stat().st_size != ready["size"]:
            fail("checkpoint content differs from its published request")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.chmod(0o400)
        os.replace(temporary, destination)
    else:
        read_snapshot(destination, digest)
    atomic_json(root / "index" / (_work_key(wire.body) + ".json"), ready, mode=0o400)


def checkpoint_for_request(transport, body):
    """Find this cohort's progress across worker commissioning generations."""
    from cacheon.chain.remote_worker_request_plan import _LOCK_NAME
    from cacheon.chain.ssh_worker_transport import DurableSpoolAuthenticatedWorkerTransport

    if type(transport) is not DurableSpoolAuthenticatedWorkerTransport:
        return None
    path = transport.spool_root / "checkpoints" / "index" / (_work_key(body) + ".json")
    if not path.exists():
        # The relay archives carriers under this lock; a move mid-scan failed the claim.
        with (transport.spool_root / "outbox" / _LOCK_NAME).open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            retained = _previous_work_checkpoint(transport, body)
        if retained is None:
            return None
        atomic_json(path, retained, mode=0o400)
    ready = load_json(path)
    if ready["worker_epoch"] == transport.registration["worker_epoch"]:
        return None
    digest = require_digest(ready["sha256"], "retained checkpoint digest")
    if (transport.spool_root / "checkpoints" / "retired" / (digest + ".json")).exists():
        return None  # A replacement refused it as other work; the cohort restarts fresh.
    if file_sha256(transport.spool_root / "checkpoints" / (digest + ".json.gz")) != digest:
        fail("CPU-held checkpoint changed before replacement dispatch")
    return digest


def _previous_work_checkpoint(transport, body):
    """Reindex retained v1 progress using its original authenticated request.

    The October 14 replacement changed commissioning identities and missed a
    completed speed stage. Keep those expensive bytes and the old index intact.
    """
    from cacheon.eval.qualification_worker_checkpoint import read_snapshot
    from cacheon.chain.remote_evaluation_dispatcher import RemoteEvaluationRequest

    spool = transport.spool_root
    matches = []
    for pointer in (spool / "checkpoints" / "index").glob("*.json"):
        ready = load_json(pointer)
        if ready["worker_epoch"] == transport.registration["worker_epoch"]:
            continue
        jobs = [job for area in ("outbox", "retired", "outbox-archive")
                for job in (spool / area).glob("*-" + ready["request_id"])]
        if not jobs:
            continue
        if len(jobs) != 1:
            fail("retained checkpoint has ambiguous request carriers")
        job, = jobs
        request = load_json(job / "request.json")
        # Another cohort's entry never gates this claim; only a match is authenticated.
        if _work_key(load_json(artifact_for_role(request, job, "qualification_payload"),
                               maximum=MAX_WIRE_PAYLOAD_BYTES)["body"]) != _work_key(body):
            continue
        root = spool / "checkpoints" / "authorities" / ready["worker_epoch"]
        registration = verify_registration(load_json(root / "registration.json"))
        wire = RemoteEvaluationRequest.from_dict(authenticated_wire_request(request, job,
            identity=registration_transport_identity(registration),
            credential=registration_credential(registration, root / "credential.secret")))
        snapshot = read_snapshot(spool / "checkpoints" / (ready["sha256"] + ".json.gz"), ready["sha256"])
        if snapshot["origin"]["request"] != ready["request_digest"] or wire.digest != ready["request_digest"]:
            fail("retained checkpoint differs from its original request")
        stages = snapshot["stages"]
        progress = tuple(name in stages for name in ("quality", "audit", "speed"))
        matches.append((progress, request["queued_at_unix_ns"], ready))
    return max(matches, key=lambda row: row[:2])[2] if matches else None


def request_body(dispatcher, claim, retained_body):
    """Preserve an old request verbatim or bind one CPU-held progress snapshot."""
    from cacheon.chain.qualification_request import qualification_request_body

    body = qualification_request_body(dispatcher.coordinator, claim,
        incumbent_stack_digest=dispatcher.qualification_incumbent_stack.digest,
        incumbent_tree_digest=dispatcher.qualification_incumbent_tree_digest,
        retained_body=retained_body)
    if retained_body is None:
        checkpoint = checkpoint_for_request(dispatcher.transport, body)
        if checkpoint is not None:
            body["checkpoint_digest"] = checkpoint
    return body


def retire_refused_checkpoint(dispatcher, recovery, plan, product):
    """Retire a checkpoint its replacement refused as other work and requeue the cohort fresh.

    The worker refuses before execution. The saved bytes and index stay; this
    record keeps later requests from carrying them, so the cohort restarts under
    the live commission with no decision and no consumed attempt.
    """
    from cacheon.chain.execution_disposition import (
        WORKER_INFRASTRUCTURE_REQUEUE_FAILURE, ExecutionDisposition, ExecutionOutcome)
    from cacheon.chain.recoverable_qualification_dispatcher import RecoverableQualificationRequeue
    from cacheon.chain.remote_qualification_hold import RemoteQualificationHoldProduct, RemoteQualificationHoldReason

    code = RemoteQualificationHoldReason.CHECKPOINT_COMPUTATION_CHANGED
    digest = plan.remote_request.body.get("checkpoint_digest")
    if type(product) is not RemoteQualificationHoldProduct or product.reason is not code or digest is None:
        return None
    spool = dispatcher.transport.spool_root
    record = spool / "checkpoints" / "retired" / (digest + ".json")
    if not record.exists():  # Written before the release, so a restart repeats only the release.
        record.parent.mkdir(mode=0o700, exist_ok=True)
        atomic_json(record, {"checkpoint_sha256": digest, "request_id": plan.request_id,
                             "worker_epoch": plan.worker_epoch, "failure_type": product.failure_type,
                             "failure_message": product.failure_message}, mode=0o400)
        append_event(spool, "checkpoint_retired", request_id=plan.request_id, checkpoint_sha256=digest)
    recovery = dispatcher._renew_before_transition(recovery)
    store, point, current = dispatcher._current_recovery(recovery.recovery_id)
    try:
        store.release_worker_infrastructure_recovery(current, failure_code=code.value, current_block=point[0])
    finally:
        store.close()
    return RecoverableQualificationRequeue(recovery.recovery_id, plan.request_id, ExecutionOutcome(
        ExecutionDisposition.REQUEUE, decision="NO_DECISION",
        failure_code=WORKER_INFRASTRUCTURE_REQUEUE_FAILURE, reason=code.value))


def install_checkpoint(runtime, outer, request_dir, wire):
    """Install only the checkpoint explicitly sealed into the replacement request."""
    from cacheon.eval.qualification_worker_checkpoint import read_snapshot

    digest = wire.body.get("checkpoint_digest")
    carriers = [row for row in outer["artifacts"] if row["role"] == "qualification_checkpoint"]
    if digest is None:
        if carriers:
            fail("request carries an unrequested qualification checkpoint")
        return
    if len(carriers) != 1 or carriers[0]["sha256"] != digest:
        fail("requested checkpoint is absent or different")
    source = artifact_for_role(outer, request_dir, "qualification_checkpoint")
    read_snapshot(source, digest)
    root = runtime.qualification_continuation_store.root / "imports"
    root.mkdir(mode=0o700, exist_ok=True)
    target = root / (digest + ".json.gz")
    if not target.exists():
        atomic_bytes(target, source.read_bytes(), mode=0o400)
    elif file_sha256(target) != digest:
        fail("installed checkpoint changed")
    pointer = root / (wire.digest + ".json")
    row = {"sha256": digest}
    if pointer.exists() and load_json(pointer) != row:
        fail("replacement request checkpoint changed")
    atomic_json(pointer, row, mode=0o400)


def recover_replaced_worker(dispatcher):
    """Import an old completed product or retire its lease before replacement."""
    from cacheon.arena_service import ArenaService
    from cacheon.chain.evaluation_coordinator import WorkerReadiness
    from cacheon.chain.execution_disposition import AUTHORITY_CHANGED_HOLD_REASON
    from cacheon.chain.evaluation_recovery import RecoveryPhase
    from cacheon.chain.mainnet_screen_dispatcher import RemoteOnlyArenaProvider, _manifest_from_dict
    from cacheon.chain.ssh_worker_transport import DurableSpoolAuthenticatedWorkerTransport

    transport = dispatcher.transport
    if type(transport) is not DurableSpoolAuthenticatedWorkerTransport:
        return None
    # Every dispatched epoch keeps its arena service before it can publish paid work.
    retain_authority(transport, dispatcher.coordinator.service.manifest)
    store, point = dispatcher._open_store()
    try:
        recovery = store.pending_qualification_recovery(owner=dispatcher.coordinator.owner)
        if recovery is None or not recovery.request_id:
            return None
        plan = store.reopen_recovery_request_plan(recovery)
        if plan.worker_epoch == transport.registration["worker_epoch"]:
            return None
        result_root = transport.spool_root / "results" / plan.request_id
        marked = (result_root / "RESULT_READY").is_file()
        if result_root.exists() and not marked:
            # Unmarked result bytes may be finished paid work: hold under that cause, never rerun.
            if recovery.phase is not RecoveryPhase.HELD:
                store.hold_recovery(recovery, current_block=point[0], reason="transport_hold:result_partial")
            return None
        completed = marked and load_json(result_root / "result.json")["state"] == "completed"
        if not completed:
            # Park the old carrier before release. A crash here reopens the
            # same stored plan and repeats this move without creating work.
            old = transport.spool_root / "outbox" / f"{plan.queued_at_unix_ns:020d}-{plan.request_id}"
            retired = transport.spool_root / "retired"
            retired.mkdir(mode=0o700, exist_ok=True)
            if old.exists():
                if (retired / old.name).exists():
                    fail("retired checkpoint carrier collision")
                old.rename(retired / old.name)
            if recovery.phase is not RecoveryPhase.HELD:
                recovery = store.hold_recovery(recovery, current_block=point[0],
                                               reason=AUTHORITY_CHANGED_HOLD_REASON)
            store.release_worker_infrastructure_recovery(recovery,
                failure_code="worker_replaced", current_block=point[0],
                live_worker_epoch=transport.registration["worker_epoch"])
            return None
    finally:
        store.close()
    # No network or evaluator call is needed: the normal dispatcher consumes the
    # complete local result under the worker and arena service that produced it.
    # Recommissioning changes the service, which can neither re-seal nor commit it.
    root = transport.spool_root / "checkpoints" / "authorities" / plan.worker_epoch
    manifest = _manifest_from_dict(load_json(root / "service.json"))
    retained = copy.copy(transport)
    retained.registration = verify_registration(load_json(root / "registration.json"))
    retained.identity = registration_transport_identity(retained.registration)
    retained.credential_path = root / "credential.secret"
    retained.credential = registration_credential(retained.registration, retained.credential_path)
    original = copy.copy(dispatcher)
    original.transport = retained
    original.transport_identity = retained.identity
    original.credential = retained.credential
    original.coordinator = copy.copy(dispatcher.coordinator)
    original.coordinator.service = ArenaService(manifest, RemoteOnlyArenaProvider(manifest.provider_digest))
    original.coordinator.readiness = WorkerReadiness(**retained.registration["worker_readiness"])
    return original.dispatch_once()
