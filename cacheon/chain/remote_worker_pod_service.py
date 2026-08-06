"""Pod-side spool service for one commissioned remote worker epoch.

This module owns the pod half of the durable spool: accepting verified request
archives, selecting the next request in queue order, supervising exactly one
persistent fixed-adapter process, finalizing and publishing closed results,
and answering status verbs over the pinned SSH channel.  All filesystem
coordinates come from one closed :class:`PodPaths`; nothing here selects a
command, module, environment, or output path from request content.

Epoch failure policy: a command-level adapter failure after the resident
worker was entered is epoch-fatal on its first occurrence.  The service parks
in an ``epoch_failed`` heartbeat state instead of silently restarting the
adapter, so a transport or candidate fault can never cause an avoidable
resident-model unload or spend another miner request on an unproven engine.
"""

from __future__ import annotations

import contextlib
import fcntl
import io
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cacheon.chain.remote_worker_registration import (
    PodPaths,
    registration_credential,
    registration_transport_identity,
    verify_fixed_adapter,
    verify_pod_registration,
)
from cacheon.chain.remote_worker_spool import (
    ALLOWED_FAILURE_CODES,
    HEX64,
    MAX_ARCHIVE_BYTES,
    MAX_CONSECUTIVE_ADAPTER_FAILURES,
    MAX_JOB_SECONDS,
    SCHEMA_ADAPTER_COMMAND,
    SCHEMA_ADAPTER_CONTROL,
    SCHEMA_ADAPTER_RESULT,
    SCHEMA_RESULT_READY,
    DOMAIN_RESULT_READY,
    append_event,
    atomic_bytes,
    atomic_json,
    fail,
    file_sha256,
    finalize_adapter_response,
    heartbeat_payload,
    load_json,
    pack_directory,
    require_digest,
    require_int,
    safe_extract,
    spool_canonical_json,
    spool_digest,
    strict_json_object,
    verify_adapter_result,
    verify_request,
    RemoteWorkerError,
)
from cacheon.stack_identity import sha256_hex


def adapter_environment(
    registration: Mapping[str, Any],
    paths: PodPaths,
    request_id: str | None,
) -> dict[str, str]:
    environment = {
        "CACHEON_REMOTE_READY_RECEIPT_DIGEST": registration["ready_receipt_digest"],
        "CACHEON_REMOTE_CREDENTIAL_PATH": str(paths.credential),
        "CACHEON_REMOTE_REGISTRATION_PATH": str(paths.registration),
        "CACHEON_REMOTE_TRANSPORT_IDENTITY_DIGEST": registration[
            "transport_identity_digest"
        ],
        "CACHEON_REMOTE_WORKER_EPOCH": registration["worker_epoch"],
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(
            str(device) for device in registration["lane_devices"]
        ),
        "HOME": "/root",
        "LANG": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(paths.root / "tmp"),
    }
    if request_id is not None:
        environment["CACHEON_REMOTE_REQUEST_ID"] = request_id
    return environment


def infrastructure_result(
    request: Mapping[str, Any], result_root: Path, failure_code: str
) -> None:
    if failure_code not in ALLOWED_FAILURE_CODES:
        fail("pod failure code is not registered")
    result_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    blobs = result_root / "blobs"
    blobs.mkdir(mode=0o700)
    payload = (
        spool_canonical_json(
            {
                "failure_code": failure_code,
                "request_id": request["request_id"],
                "state": "no_decision",
            }
        )
        + b"\n"
    )
    digest = sha256_hex(payload)
    atomic_bytes(blobs / digest, payload, mode=0o400)
    atomic_json(
        result_root / "result.json",
        {
            "artifacts": [
                {"role": "adapter_result", "sha256": digest, "size": len(payload)}
            ],
            "failure_code": failure_code,
            "request_id": request["request_id"],
            "response_digest": None,
            "response_sha256": None,
            "schema": SCHEMA_ADAPTER_RESULT,
            "state": "no_decision",
        },
        mode=0o400,
    )


def publish_result(
    registration: Mapping[str, Any],
    request: Mapping[str, Any],
    result_root: Path,
    *,
    request_root: Path,
    outgoing_root: Path,
    events_root: Path,
    identity,
    credential,
) -> None:
    result = verify_adapter_result(
        load_json(result_root / "result.json"),
        result_root,
        request,
        registration,
        request_root=request_root,
        identity=identity,
        credential=credential,
    )
    outgoing_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    request_id = request["request_id"]
    temporary_archive = outgoing_root / f".{request_id}.tar"
    archive_sha = pack_directory(
        result_root, "result.json", result["artifacts"], temporary_archive
    )
    archive_size = temporary_archive.stat().st_size
    final_archive = outgoing_root / f"{request_id}.{archive_sha}.tar"
    if final_archive.exists():
        if file_sha256(final_archive) != archive_sha:
            fail("outgoing result archive collision")
        temporary_archive.unlink()
    else:
        os.replace(temporary_archive, final_archive)
    unsigned = {
        "archive_sha256": archive_sha,
        "archive_size": archive_size,
        "ready_receipt_digest": registration["ready_receipt_digest"],
        "request_id": request_id,
        "schema": SCHEMA_RESULT_READY,
        "state": "ready",
        "worker_epoch": registration["worker_epoch"],
        "worker_readiness_digest": registration["worker_readiness_digest"],
    }
    ready = {
        **unsigned,
        "ready_digest": spool_digest(DOMAIN_RESULT_READY, unsigned),
    }
    atomic_json(outgoing_root / f"{request_id}.ready.json", ready, mode=0o400)
    append_event(
        events_root, "result_published", request_id=request_id, state=result["state"]
    )


def recover_interrupted(
    registration: Mapping[str, Any],
    paths: PodPaths,
    *,
    identity,
    credential,
) -> None:
    processing = paths.root / "processing"
    results = paths.root / "results"
    if not processing.exists():
        return
    for job in sorted(processing.iterdir()):
        if not job.is_dir() or job.is_symlink() or HEX64.fullmatch(job.name) is None:
            continue
        request = verify_request(
            load_json(job / "request.json"),
            job,
            registration,
            identity=identity,
            credential=credential,
        )
        result_root = results / request["request_id"]
        if not result_root.exists():
            infrastructure_result(request, result_root, "pod_service_restart")
        publish_result(
            registration,
            request,
            result_root,
            request_root=job,
            outgoing_root=paths.root / "outgoing",
            events_root=paths.root,
            identity=identity,
            credential=credential,
        )
        completed = paths.root / "completed" / request["request_id"]
        completed.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not completed.exists():
            os.replace(job, completed)


def decode_adapter_control(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > 64 * 1024 or not raw.endswith(b"\n"):
        fail("persistent adapter emitted a malformed control frame")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeError as exc:
        fail(f"persistent adapter emitted invalid JSON: {exc}")
    value = strict_json_object(decoded)
    if (
        raw != spool_canonical_json(value) + b"\n"
        or value.get("schema") != SCHEMA_ADAPTER_CONTROL
    ):
        fail("persistent adapter control frame is not canonical")
    return value


class PersistentAdapterProcess:
    """One fixed adapter process and commissioned worker per pod epoch."""

    def __init__(
        self,
        registration: Mapping[str, Any],
        *,
        paths: PodPaths,
        heartbeat_seconds: int,
        adapter_arguments: tuple[str, ...] = (),
    ) -> None:
        self.registration = registration
        self.paths = paths
        self.heartbeat_seconds = heartbeat_seconds
        self.adapter_arguments = tuple(adapter_arguments)
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: io.BufferedWriter | None = None
        self.start_count = 0
        self.consecutive_failures = 0

    @property
    def alive(self) -> bool:
        process = self.process
        return process is not None and process.poll() is None

    def record_result(self, *, completed: bool) -> None:
        if type(completed) is not bool:
            fail("adapter completion state is not boolean")
        self.consecutive_failures = 0 if completed else self.consecutive_failures + 1

    def _heartbeat(self, request_id: str | None, state: str) -> None:
        atomic_json(
            self.paths.root / "heartbeat.json",
            heartbeat_payload(
                self.registration,
                state,
                request_id,
                adapter_start_count=self.start_count,
                adapter_alive=self.alive,
                consecutive_adapter_failures=self.consecutive_failures,
            ),
            mode=0o400,
        )

    def _read_control(
        self,
        *,
        deadline: int,
        request_id: str | None,
        state: str,
    ) -> dict[str, Any] | None:
        process = self.process
        if process is None or process.stdout is None:
            return None
        while int(time.time()) < deadline:
            if process.poll() is not None:
                return None
            remaining = max(0.0, deadline - time.time())
            timeout = min(float(self.heartbeat_seconds), remaining)
            readable, _, _ = select.select([process.stdout], [], [], timeout)
            self._heartbeat(request_id, state)
            if not readable:
                continue
            raw = process.stdout.readline()
            if not raw:
                return None
            try:
                return decode_adapter_control(raw)
            except RemoteWorkerError:
                return None
        return None

    def _start(self, *, deadline: int, request_id: str) -> bool:
        verify_fixed_adapter(self.paths.adapter, self.registration)
        log_root = self.paths.root / "logs"
        log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.log_handle = (log_root / "persistent-adapter.log").open("ab", buffering=0)
        self.process = subprocess.Popen(
            [
                self.registration["python_executable"],
                str(self.paths.adapter),
                "--serve",
                *self.adapter_arguments,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_handle,
            env=adapter_environment(self.registration, self.paths, None),
            start_new_session=True,
        )
        self.start_count += 1
        append_event(
            self.paths.root,
            "adapter_process_started",
            adapter_pid=self.process.pid,
            adapter_start_count=self.start_count,
            request_id=request_id,
            worker_epoch=self.registration["worker_epoch"],
        )
        ready = self._read_control(
            deadline=deadline,
            request_id=request_id,
            state="adapter_starting",
        )
        if ready != {
            "schema": SCHEMA_ADAPTER_CONTROL,
            "state": "ready",
        }:
            append_event(
                self.paths.root,
                "adapter_process_start_failed",
                adapter_start_count=self.start_count,
                request_id=request_id,
                worker_epoch=self.registration["worker_epoch"],
            )
            self.close()
            return False
        append_event(
            self.paths.root,
            "adapter_process_ready",
            adapter_pid=self.process.pid,
            adapter_start_count=self.start_count,
            request_id=request_id,
            worker_epoch=self.registration["worker_epoch"],
        )
        return True

    def evaluate(
        self,
        request: Mapping[str, Any],
        job_root: Path,
        result_root: Path,
        *,
        deadline: int,
    ) -> str | None:
        request_id = request["request_id"]
        process = self.process
        if process is None:
            if self.start_count:
                return "adapter_exit_nonzero"
            if not self._start(deadline=deadline, request_id=request_id):
                return "adapter_start_failed"
            process = self.process
        elif process.poll() is not None:
            return "adapter_exit_nonzero"
        assert process is not None
        if process.stdin is None:
            return "adapter_exit_nonzero"
        command = {
            "operation": "evaluate",
            "request_dir": str(job_root),
            "request_id": request_id,
            "result_dir": str(result_root),
            "schema": SCHEMA_ADAPTER_COMMAND,
        }
        try:
            process.stdin.write(spool_canonical_json(command) + b"\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            return "adapter_exit_nonzero"
        control = self._read_control(
            deadline=deadline,
            request_id=request_id,
            state="evaluating",
        )
        expected = {
            "request_id": request_id,
            "schema": SCHEMA_ADAPTER_CONTROL,
            "state": "completed",
        }
        if control == expected:
            return None
        request_failed = {
            "request_id": request_id,
            "schema": SCHEMA_ADAPTER_CONTROL,
            "state": "request_failed",
        }
        if control == request_failed:
            return "adapter_request_failed"
        epoch_failed = {
            "request_id": request_id,
            "schema": SCHEMA_ADAPTER_CONTROL,
            "state": "epoch_failed",
        }
        if control == epoch_failed:
            return "adapter_epoch_failed"
        timed_out = int(time.time()) >= deadline
        return "adapter_timeout" if timed_out else "adapter_exit_nonzero"

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is not None:
            if process.stdin is not None:
                with contextlib.suppress(BrokenPipeError, OSError, ValueError):
                    process.stdin.close()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=20)
            if process.stdout is not None:
                process.stdout.close()
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


def next_incoming(
    registration: Mapping[str, Any],
    paths: PodPaths,
    *,
    identity,
    credential,
) -> tuple[Path, dict[str, Any]] | None:
    incoming = paths.root / "incoming"
    candidates: list[tuple[int, str, Path, dict[str, Any]]] = []
    for marker in incoming.glob("*.ready.json"):
        request_id = marker.name[: -len(".ready.json")]
        if HEX64.fullmatch(request_id) is None:
            continue
        receipt = load_json(marker)
        if set(receipt) != {"archive_sha256", "archive_size", "request_id", "worker_epoch"}:
            fail("incoming request marker fields are not closed")
        if (
            receipt["request_id"] != request_id
            or receipt["worker_epoch"] != registration["worker_epoch"]
        ):
            fail("incoming request marker binding is invalid")
        archive_sha = require_digest(receipt["archive_sha256"], "incoming archive digest")
        archive = incoming / f"{request_id}.{archive_sha}.tar"
        scratch = paths.root / "inbox" / request_id
        if scratch.exists():
            request = verify_request(
                load_json(scratch / "request.json"),
                scratch,
                registration,
                identity=identity,
                credential=credential,
            )
        else:
            safe_extract(archive, scratch)
            request = verify_request(
                load_json(scratch / "request.json"),
                scratch,
                registration,
                identity=identity,
                credential=credential,
            )
        candidates.append((request["queued_at_unix_ns"], request_id, scratch, request))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, request_id, scratch, request = candidates[0]
    processing = paths.root / "processing" / request_id
    if not processing.exists():
        os.replace(scratch, processing)
    marker = incoming / f"{request_id}.ready.json"
    marker.unlink(missing_ok=True)
    return processing, request


def run_adapter(
    registration: Mapping[str, Any],
    paths: PodPaths,
    job_root: Path,
    request: Mapping[str, Any],
    heartbeat_seconds: int,
    adapter_process: PersistentAdapterProcess | None = None,
    *,
    identity,
    credential,
    adapter_arguments: tuple[str, ...] = (),
) -> Path:
    request_id = request["request_id"]
    results = paths.root / "results"
    results.mkdir(parents=True, exist_ok=True, mode=0o700)
    final = results / request_id
    if final.exists():
        verify_adapter_result(
            load_json(final / "result.json"),
            final,
            request,
            registration,
            request_root=job_root,
            identity=identity,
            credential=credential,
        )
        return final
    temporary = results / f".{request_id}.{os.getpid()}"
    temporary.mkdir(mode=0o700)
    deadline = min(request["deadline_unix"], int(time.time()) + MAX_JOB_SECONDS)
    owns_process = adapter_process is None
    process = adapter_process or PersistentAdapterProcess(
        registration,
        paths=paths,
        heartbeat_seconds=heartbeat_seconds,
        adapter_arguments=adapter_arguments,
    )
    try:
        failure = process.evaluate(request, job_root, temporary, deadline=deadline)
    finally:
        if owns_process:
            process.close()
    if failure is not None:
        shutil.rmtree(temporary, ignore_errors=True)
        infrastructure_result(request, temporary, failure)
    else:
        try:
            finalize_adapter_response(
                request,
                job_root,
                temporary,
                identity=identity,
                credential=credential,
            )
            verify_adapter_result(
                load_json(temporary / "result.json"),
                temporary,
                request,
                registration,
                request_root=job_root,
                identity=identity,
                credential=credential,
            )
        except RemoteWorkerError:
            shutil.rmtree(temporary, ignore_errors=True)
            infrastructure_result(request, temporary, "adapter_result_invalid")
    os.replace(temporary, final)
    return final


def pod_serve(
    paths: PodPaths,
    *,
    poll_seconds: int,
    heartbeat_seconds: int,
    adapter_arguments: tuple[str, ...] = (),
) -> None:
    registration = verify_pod_registration(paths)
    identity = registration_transport_identity(registration)
    credential = registration_credential(registration, paths.credential)
    for name in (
        "completed",
        "inbox",
        "incoming",
        "logs",
        "outgoing",
        "processing",
        "results",
        "tmp",
        "verify",
    ):
        (paths.root / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = paths.root / "service.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("another pod worker service already owns this root")
        recover_interrupted(
            registration, paths, identity=identity, credential=credential
        )
        append_event(
            paths.root, "pod_service_started", worker_epoch=registration["worker_epoch"]
        )
        adapter_process = PersistentAdapterProcess(
            registration,
            paths=paths,
            heartbeat_seconds=heartbeat_seconds,
            adapter_arguments=adapter_arguments,
        )
        try:
            while True:
                verify_pod_registration(paths)
                adapter_process._heartbeat(None, "idle")
                item = next_incoming(
                    registration, paths, identity=identity, credential=credential
                )
                if item is not None:
                    job_root, request = item
                    request_id = request["request_id"]
                    append_event(paths.root, "adapter_started", request_id=request_id)
                    result = run_adapter(
                        registration,
                        paths,
                        job_root,
                        request,
                        heartbeat_seconds,
                        adapter_process,
                        identity=identity,
                        credential=credential,
                    )
                    result_row = verify_adapter_result(
                        load_json(result / "result.json"),
                        result,
                        request,
                        registration,
                        request_root=job_root,
                        identity=identity,
                        credential=credential,
                    )
                    # A typed request-local refusal leaves the commissioned
                    # adapter and resident model healthy.  Every other adapter
                    # failure is epoch-fatal on its first occurrence.
                    adapter_process.record_result(
                        completed=(
                            result_row["state"] == "completed"
                            or result_row["failure_code"] == "adapter_request_failed"
                        )
                    )
                    publish_result(
                        registration,
                        request,
                        result,
                        request_root=job_root,
                        outgoing_root=paths.root / "outgoing",
                        events_root=paths.root,
                        identity=identity,
                        credential=credential,
                    )
                    completed = paths.root / "completed" / request_id
                    if not completed.exists():
                        os.replace(job_root, completed)
                    append_event(paths.root, "adapter_finished", request_id=request_id)
                    if (
                        adapter_process.consecutive_failures
                        >= MAX_CONSECUTIVE_ADAPTER_FAILURES
                    ):
                        append_event(
                            paths.root,
                            "resident_epoch_failed",
                            adapter_start_count=adapter_process.start_count,
                            consecutive_adapter_failures=(
                                adapter_process.consecutive_failures
                            ),
                            worker_epoch=registration["worker_epoch"],
                        )
                        # Fail the epoch closed without tearing down a still-
                        # live resident model.  A human may inspect or recover
                        # the epoch; this service never turns a candidate or
                        # transport failure into an avoidable model unload.
                        while True:
                            verify_pod_registration(paths)
                            adapter_process._heartbeat(None, "epoch_failed")
                            time.sleep(poll_seconds)
                time.sleep(poll_seconds)
        finally:
            adapter_process.close()


def accept_request(
    paths: PodPaths,
    *,
    request_id: str,
    archive_sha256: str,
    archive_size: int,
) -> None:
    registration = verify_pod_registration(paths)
    identity = registration_transport_identity(registration)
    credential = registration_credential(registration, paths.credential)
    exact_id = require_digest(request_id, "request id")
    archive_sha = require_digest(archive_sha256, "request archive digest")
    exact_size = require_int(
        archive_size, "request archive size", minimum=1, maximum=MAX_ARCHIVE_BYTES
    )
    incoming = paths.root / "incoming"
    part = incoming / f".{exact_id}.{archive_sha}.tar.part"
    final = incoming / f"{exact_id}.{archive_sha}.tar"
    ready = incoming / f"{exact_id}.ready.json"
    if final.exists() and ready.exists():
        if final.stat().st_size != exact_size or file_sha256(final) != archive_sha:
            fail("existing request archive identity collision")
        print("accepted")
        return
    if part.is_symlink() or not part.is_file():
        fail("incoming request archive part is absent")
    if part.stat().st_size != exact_size or file_sha256(part) != archive_sha:
        fail("incoming request archive checksum mismatch")
    scratch = paths.root / "verify" / f".{exact_id}.{os.getpid()}"
    if scratch.exists():
        shutil.rmtree(scratch)
    safe_extract(part, scratch)
    try:
        request = verify_request(
            load_json(scratch / "request.json"),
            scratch,
            registration,
            identity=identity,
            credential=credential,
        )
        if request["request_id"] != exact_id:
            fail("incoming request archive changed request id")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    os.chmod(part, 0o400)
    os.replace(part, final)
    atomic_json(
        ready,
        {
            "archive_sha256": archive_sha,
            "archive_size": exact_size,
            "request_id": exact_id,
            "worker_epoch": registration["worker_epoch"],
        },
        mode=0o400,
    )
    append_event(paths.root, "request_accepted", request_id=exact_id)
    print("accepted")


def heartbeat_status(paths: PodPaths) -> None:
    path = paths.root / "heartbeat.json"
    if not path.exists():
        return
    sys.stdout.buffer.write(spool_canonical_json(load_json(path)) + b"\n")


def result_status(paths: PodPaths, request_id: str) -> None:
    exact_id = require_digest(request_id, "request id")
    path = paths.root / "outgoing" / f"{exact_id}.ready.json"
    if not path.exists():
        return
    sys.stdout.buffer.write(spool_canonical_json(load_json(path)) + b"\n")


def ack_result(paths: PodPaths, request_id: str, archive_sha256: str) -> None:
    exact_id = require_digest(request_id, "request id")
    archive_sha = require_digest(archive_sha256, "archive digest")
    outgoing = paths.root / "outgoing"
    archive = outgoing / f"{exact_id}.{archive_sha}.tar"
    ready = outgoing / f"{exact_id}.ready.json"
    if not archive.exists() or not ready.exists():
        fail("cannot acknowledge absent result")
    receipt = load_json(ready)
    if receipt.get("archive_sha256") != archive_sha or receipt.get("request_id") != exact_id:
        fail("result acknowledgement identity differs")
    retained = paths.root / "acked" / exact_id
    retained.mkdir(parents=True, exist_ok=True, mode=0o700)
    for source in (archive, ready):
        destination = retained / source.name
        if destination.exists():
            if file_sha256(destination) != file_sha256(source):
                fail("acknowledged result collision")
            source.unlink()
        else:
            os.replace(source, destination)
    append_event(paths.root, "result_acknowledged", request_id=exact_id)
    print("acknowledged")
