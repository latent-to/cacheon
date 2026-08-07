"""Durable spool schemas for the CPU-validator <-> remote worker transport.

This module owns the content-addressed spool wire format: sealed request
carriers, artifact inventories, adapter results, result-ready receipts, worker
heartbeats, and the safe pack/extract primitives that move them.  It is
deployment plumbing, not evaluation authority: nothing here can screen,
qualify, settle, or manufacture a candidate ``FAIL``.  Authenticated payload
semantics (HMAC request/response types) remain owned by
:mod:`cacheon.chain.remote_evaluation_dispatcher`; durable lease semantics
remain owned by :mod:`cacheon.chain.evaluation_leases`.

Every digest uses :func:`cacheon.stack_identity.canonical_digest`.  The spool
formats are wire-compatible with the previously deployed private service; the
schema and domain strings below are consensus-adjacent identity inputs and
must not change without a protocol review.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

from cacheon.chain.evaluation_leases import (
    EvaluationLease,
    EvaluationLeaseError,
    EvaluationLeaseMember,
)
from cacheon.chain.remote_evaluation_dispatcher import (
    AuthenticatedRemoteEvaluationResponse,
    RemoteEvaluationDispatcherError,
    RemoteEvaluationRequest,
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
    reopen_remote_response,
    verify_remote_request,
)
from cacheon.chain import remote_qualification_hold as remote_hold
from cacheon.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
    sha256_hex,
)


SCHEMA_REGISTRATION = "cacheon-remote-worker-registration-v1"
SCHEMA_REQUEST = "cacheon-remote-evaluation-request-v1"
SCHEMA_ADAPTER_RESULT = "cacheon-b300-adapter-result-v1"
SCHEMA_RESULT_READY = "cacheon-remote-result-ready-v1"
SCHEMA_HEARTBEAT = "cacheon-remote-worker-heartbeat-v2"
SCHEMA_DISPATCH_STATE = "cacheon-remote-dispatch-state-v1"
SCHEMA_ADAPTER_COMMAND = "cacheon-b300-adapter-command-v1"
SCHEMA_ADAPTER_CONTROL = "cacheon-b300-adapter-control-v1"

DOMAIN_REGISTRATION = "cacheon.chain.remote-worker-registration.v1"
DOMAIN_REQUEST = "cacheon.chain.remote-evaluation-request.v1"
DOMAIN_RESULT_READY = "cacheon.chain.remote-result-ready.v1"
DOMAIN_HEARTBEAT = "cacheon.chain.remote-worker-heartbeat.v1"

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
EPOCH = re.compile(r"[0-9a-f]{32}\Z")
HOST = re.compile(r"[A-Za-z0-9.-]{1,253}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9._:@/+-]{1,512}\Z")
ROLE = re.compile(r"[a-z][a-z0-9._-]{0,63}\Z")
WIRE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
OWNER_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

REQUEST_FIELDS = frozenset(
    {
        "artifacts",
        "created_at_unix",
        "deadline_unix",
        "lease",
        "queued_at_unix_ns",
        "ready_receipt_digest",
        "request_id",
        "schema",
        "service_identity",
        "worker_epoch",
        "worker_readiness_digest",
    }
)
ARTIFACT_FIELDS = frozenset({"role", "sha256", "size"})
ADAPTER_RESULT_FIELDS = frozenset(
    {
        "artifacts",
        "failure_code",
        "request_id",
        "response_digest",
        "response_sha256",
        "schema",
        "state",
    }
)
RESULT_READY_FIELDS = frozenset(
    {
        "archive_sha256",
        "archive_size",
        "ready_digest",
        "ready_receipt_digest",
        "request_id",
        "schema",
        "state",
        "worker_epoch",
        "worker_readiness_digest",
    }
)
HEARTBEAT_FIELDS = frozenset(
    {
        "active_request_id",
        "adapter_alive",
        "adapter_start_count",
        "consecutive_adapter_failures",
        "heartbeat_digest",
        "pid",
        "ready_receipt_digest",
        "schema",
        "state",
        "time_unix",
        "worker_epoch",
        "worker_readiness_digest",
    }
)

ALLOWED_ARTIFACT_ROLES = frozenset(
    {
        "adapter_result",
        "arena_authority",
        "candidate_publication",
        "claim",
        "evaluation_evidence",
        "incumbent_authority",
        "qualification_authority",
        "qualification_payload",
        "screen_payload",
        "worker_log",
    }
)
OUTPUT_ARTIFACT_ROLES = frozenset(
    {"adapter_result", "evaluation_evidence", "worker_log"}
)
ALLOWED_FAILURE_CODES = frozenset(
    {
        "adapter_epoch_failed",
        "adapter_exit_nonzero",
        "adapter_request_failed",
        "adapter_result_invalid",
        "adapter_start_failed",
        "adapter_timeout",
        "pod_service_restart",
        "request_deadline_elapsed",
        "worker_epoch_superseded",
    }
)

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_WIRE_PAYLOAD_BYTES = 64 << 20
MAX_ARTIFACTS = 64
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
MAX_ARCHIVE_BYTES = 5 * 1024 * 1024 * 1024
MAX_JOB_SECONDS = 4 * 60 * 60
# A command-level adapter failure means the standing engine is no longer a
# proven service.  Fail the epoch on the first such event; never spend another
# miner request silently booting a replacement model.
MAX_CONSECUTIVE_ADAPTER_FAILURES = 1
NATIVE_ARTIFACT_MANIFEST = ".cacheon-native-artifact.json"


class RemoteWorkerError(RuntimeError):
    """The remote worker transport cannot proceed without changing authority."""


def fail(message: str) -> NoReturn:
    raise RemoteWorkerError(message)


def _reject_constant(value: str) -> NoReturn:
    fail(f"JSON contains a non-integer number: {value}")


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            fail(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def spool_canonical_json(value: object) -> bytes:
    """Canonical spool encoding: the one product identity encoding."""

    try:
        return canonical_json_bytes(value)
    except StackIdentityError as exc:
        fail(f"payload is not canonical JSON data: {exc}")


def spool_digest(domain: str, payload: object) -> str:
    """Semantic digest of one spool object under the shared identity envelope."""

    try:
        return canonical_digest(domain, payload)
    except StackIdentityError as exc:
        fail(f"payload cannot be digested: {exc}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        fail(f"{field} is not lowercase SHA-256")
    return value


def require_int(
    value: object,
    field: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{field} is outside its integer bounds")
    if maximum is not None and value > maximum:
        fail(f"{field} is outside its integer bounds")
    return value


def require_text(
    value: object,
    field: str,
    *,
    maximum: int = 512,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > maximum
        or OWNER_CONTROL.search(value) is not None
        or (pattern is not None and pattern.fullmatch(value) is None)
    ):
        fail(f"{field} is malformed")
    return value


def require_closed(row: object, fields: frozenset[str], name: str) -> dict[str, Any]:
    if type(row) is not dict or set(row) != fields:
        fail(f"{name} fields are not closed")
    return row


def strict_json_object(text: str) -> dict[str, Any]:
    """Decode one closed JSON object from trusted-channel text."""

    try:
        value = json.loads(
            text,
            object_pairs_hook=_no_duplicates,
            parse_float=_reject_constant,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        fail(f"strict JSON object is invalid: {exc}")
    if type(value) is not dict:
        fail("strict JSON root is not an object")
    return value


def load_json(path: Path, *, maximum: int = MAX_JSON_BYTES) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"JSON path is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        fail(f"JSON path has unsafe size: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_float=_reject_constant,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"could not read closed JSON {path}: {exc}")
    if type(value) is not dict:
        fail(f"JSON root is not an object: {path}")
    return value


def atomic_bytes(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def atomic_json(path: Path, value: object, *, mode: int = 0o600) -> None:
    atomic_bytes(path, spool_canonical_json(value) + b"\n", mode=mode)


def append_event(root: Path, event: str, **fields: object) -> None:
    row = {"event": event, "time_unix": int(time.time()), **fields}
    encoded = spool_canonical_json(row) + b"\n"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "events.jsonl").open("ab", buffering=0) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(encoded)
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)
    print(encoded.decode("utf-8").rstrip(), flush=True)


def verify_lease(row: object) -> dict[str, Any]:
    """Verify one lease projection through the exact durable lease type."""

    fields = frozenset(
        {
            "claimed_block",
            "expires_block",
            "generation",
            "initial_expires_block",
            "lease_id",
            "members",
            "owner",
            "stage",
        }
    )
    value = require_closed(row, fields, "evaluation lease")
    members_value = value["members"]
    if type(members_value) is not list or not members_value:
        fail("lease members are malformed")
    try:
        members = tuple(
            EvaluationLeaseMember(
                **require_closed(
                    item,
                    frozenset({"prior_status", "reservation_id"}),
                    "lease member",
                )
            )
            for item in members_value
        )
        EvaluationLease(
            value["lease_id"],
            value["generation"],
            value["stage"],
            value["owner"],
            members,
            value["claimed_block"],
            value["initial_expires_block"],
            value["expires_block"],
        )
    except EvaluationLeaseError as exc:
        fail(f"evaluation lease projection is invalid: {exc}")
    except (TypeError, ValueError) as exc:
        fail(f"evaluation lease projection is malformed: {exc}")
    return value


def verify_artifacts(
    items: object,
    root: Path,
    *,
    allow_output_roles: bool,
) -> list[dict[str, Any]]:
    if type(items) is not list or len(items) > MAX_ARTIFACTS:
        fail("artifact list is malformed")
    total = 0
    seen: set[tuple[str, str]] = set()
    verified: list[dict[str, Any]] = []
    for raw in items:
        item = require_closed(raw, ARTIFACT_FIELDS, "artifact")
        role = require_text(item["role"], "artifact role", pattern=ROLE, maximum=64)
        if role not in ALLOWED_ARTIFACT_ROLES:
            fail(f"artifact role is not registered: {role}")
        if not allow_output_roles and role in OUTPUT_ARTIFACT_ROLES:
            fail(f"output-only artifact role appeared in a request: {role}")
        digest = require_digest(item["sha256"], "artifact sha256")
        size = require_int(item["size"], "artifact size", maximum=MAX_ARTIFACT_BYTES)
        key = (role, digest)
        if key in seen:
            fail("artifact list contains a duplicate role/digest")
        seen.add(key)
        total += size
        if total > MAX_ARCHIVE_BYTES:
            fail("artifact list exceeds total transfer limit")
        path = root / "blobs" / digest
        if path.is_symlink() or not path.is_file():
            fail(f"artifact blob is missing: {digest}")
        if path.stat().st_size != size or file_sha256(path) != digest:
            fail(f"artifact blob does not match manifest: {digest}")
        verified.append(dict(item))
    return verified


def artifact_for_role(request: Mapping[str, Any], root: Path, role: str) -> Path:
    matches = [row for row in request["artifacts"] if row["role"] == role]
    if len(matches) != 1:
        fail(f"request must contain exactly one {role} artifact")
    return root / "blobs" / matches[0]["sha256"]


def contains_command_surface(value: object) -> bool:
    forbidden = {"argv", "command", "entrypoint", "env", "executable", "module", "shell"}
    if type(value) is dict:
        if forbidden & set(value):
            return True
        return any(contains_command_surface(item) for item in value.values())
    if type(value) is list:
        return any(contains_command_surface(item) for item in value)
    return False


def authenticated_wire_request(
    request: Mapping[str, Any],
    root: Path,
    *,
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> dict[str, Any]:
    """Reopen and authenticate the exact HMAC wire request inside a carrier."""

    lease = verify_lease(request["lease"])
    role = f"{lease['stage']}_payload"
    value = load_json(
        artifact_for_role(request, root, role), maximum=MAX_WIRE_PAYLOAD_BYTES
    )
    try:
        wire = RemoteEvaluationRequest.from_dict(value)
        verify_remote_request(wire, identity, credential)
    except RemoteEvaluationDispatcherError as exc:
        fail(f"authenticated remote request HMAC/type validation failed: {exc}")
    if wire.stage != lease["stage"] or wire.body_kind != f"{lease['stage']}_work":
        fail("authenticated remote request stage differs from its outer lease")
    if contains_command_surface(wire.body):
        fail("authenticated remote request contains a forbidden command surface")
    return wire.to_dict()


def verify_request(
    row: object,
    root: Path,
    registration: Mapping[str, Any],
    *,
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> dict[str, Any]:
    value = require_closed(row, REQUEST_FIELDS, "remote evaluation request")
    if value["schema"] != SCHEMA_REQUEST:
        fail("remote evaluation request schema is unsupported")
    unsigned = dict(value)
    supplied = require_digest(unsigned.pop("request_id"), "request_id")
    if spool_digest(DOMAIN_REQUEST, unsigned) != supplied:
        fail("request semantic digest mismatch")
    if value["worker_epoch"] != registration["worker_epoch"]:
        fail("request binds a different worker epoch")
    if value["ready_receipt_digest"] != registration["ready_receipt_digest"]:
        fail("request binds a different READY receipt")
    if value["worker_readiness_digest"] != registration["worker_readiness_digest"]:
        fail("request binds a different WorkerReadiness")
    if value["service_identity"] != registration["service_identity"]:
        fail("request binds a different arena service")
    created = require_int(value["created_at_unix"], "request creation time", minimum=1)
    queued = require_int(value["queued_at_unix_ns"], "request queue time", minimum=1)
    deadline = require_int(value["deadline_unix"], "request deadline", minimum=created + 1)
    if deadline - created > MAX_JOB_SECONDS:
        fail("request deadline exceeds deployment ceiling")
    if queued // 1_000_000_000 < created - 5:
        fail("request queue time predates request creation")
    lease = verify_lease(value["lease"])
    expected_kind = (
        "screen_payload" if lease["stage"] == "screen" else "qualification_payload"
    )
    artifacts = verify_artifacts(value["artifacts"], root, allow_output_roles=False)
    if sum(item["role"] == "candidate_publication" for item in artifacts) != 1:
        fail("request must retain exactly one candidate publication")
    if sum(item["role"] == expected_kind for item in artifacts) != 1:
        fail("request must retain exactly one stage payload")
    wire = authenticated_wire_request(
        value, root, identity=identity, credential=credential
    )
    if (
        wire["lease_id"] != lease["lease_id"]
        or wire["generation"] != lease["generation"]
        or wire["stage"] != lease["stage"]
        or wire["owner"] != lease["owner"]
        or wire["members"] != lease["members"]
        or wire["claimed_block"] != lease["claimed_block"]
        or wire["initial_expires_block"] != lease["initial_expires_block"]
        or wire["worker_readiness_digest"] != registration["worker_readiness_digest"]
        or wire["ready_receipt_digest"] != registration["ready_receipt_digest"]
        or wire["ready_epoch"] != registration["worker_readiness"]["ready_epoch"]
        or wire["service_identity"] != registration["service_identity"]
        or wire["transport_identity_digest"] != registration["transport_identity_digest"]
    ):
        fail("authenticated evaluation work differs from the transport lease")
    return value


def enqueue_request(
    registration: Mapping[str, Any],
    lease: Mapping[str, Any],
    artifact_inputs: Sequence[tuple[str, Path]],
    outbox: Path,
    *,
    deadline_seconds: int,
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> tuple[str, Path]:
    if not artifact_inputs:
        fail("at least one artifact is required")
    now = int(time.time())
    deadline = now + require_int(
        deadline_seconds,
        "deadline seconds",
        minimum=1,
        maximum=MAX_JOB_SECONDS,
    )
    queued_at = time.time_ns()
    outbox.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = Path(tempfile.mkdtemp(prefix=".request.", dir=outbox))
    try:
        blobs = temporary / "blobs"
        blobs.mkdir(mode=0o700)
        artifacts: list[dict[str, Any]] = []
        for role, source in artifact_inputs:
            digest = file_sha256(source)
            destination = blobs / digest
            if not destination.exists():
                with source.open("rb") as incoming, destination.open("xb") as output:
                    shutil.copyfileobj(incoming, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                os.chmod(destination, 0o400)
            artifacts.append(
                {"role": role, "sha256": digest, "size": source.stat().st_size}
            )
        unsigned: dict[str, Any] = {
            "artifacts": artifacts,
            "created_at_unix": now,
            "deadline_unix": deadline,
            "lease": lease,
            "queued_at_unix_ns": queued_at,
            "ready_receipt_digest": registration["ready_receipt_digest"],
            "schema": SCHEMA_REQUEST,
            "service_identity": registration["service_identity"],
            "worker_epoch": registration["worker_epoch"],
            "worker_readiness_digest": registration["worker_readiness_digest"],
        }
        request_id = spool_digest(DOMAIN_REQUEST, unsigned)
        request = {**unsigned, "request_id": request_id}
        atomic_json(temporary / "request.json", request, mode=0o400)
        verify_request(
            request,
            temporary,
            registration,
            identity=identity,
            credential=credential,
        )
        final = outbox / f"{queued_at:020d}-{request_id}"
        if final.exists():
            fail("request queue identity collision")
        os.replace(temporary, final)
        atomic_bytes(final / "REQUEST_READY", (request_id + "\n").encode(), mode=0o400)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return request_id, final


def tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o400
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def pack_directory(
    root: Path,
    manifest_name: str,
    artifacts: Iterable[Mapping[str, Any]],
    destination: Path,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with tarfile.open(temporary, "w") as archive:
            manifest = root / manifest_name
            data = manifest.read_bytes()
            archive.addfile(tar_info(manifest_name, len(data)), io.BytesIO(data))
            seen: set[str] = set()
            for item in sorted(
                artifacts, key=lambda row: (str(row["sha256"]), str(row["role"]))
            ):
                digest = str(item["sha256"])
                if digest in seen:
                    continue
                seen.add(digest)
                path = root / "blobs" / digest
                with path.open("rb") as handle:
                    archive.addfile(
                        tar_info(f"blobs/{digest}", path.stat().st_size), handle
                    )
        if Path(temporary).stat().st_size > MAX_ARCHIVE_BYTES:
            fail("packed archive exceeds transfer limit")
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
    return file_sha256(destination)


def safe_extract(archive_path: Path, destination: Path) -> None:
    if archive_path.is_symlink() or not archive_path.is_file():
        fail("transfer archive is not a regular file")
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        fail("transfer archive exceeds limit")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        with tarfile.open(archive_path, "r:") as archive:
            members = archive.getmembers()
            names: set[str] = set()
            for member in members:
                if (
                    not member.isfile()
                    or member.name in names
                    or member.name.startswith("/")
                    or ".." in Path(member.name).parts
                    or not (
                        member.name in {"request.json", "result.json"}
                        or re.fullmatch(r"blobs/[0-9a-f]{64}", member.name)
                    )
                ):
                    fail("transfer archive contains an unsafe member")
                names.add(member.name)
            for member in members:
                source = archive.extractfile(member)
                if source is None:
                    fail("transfer archive member is unreadable")
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                os.chmod(target, 0o400)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def verify_result_ready(
    row: object,
    request: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    value = require_closed(row, RESULT_READY_FIELDS, "result-ready receipt")
    if value["schema"] != SCHEMA_RESULT_READY or value["state"] != "ready":
        fail("result-ready receipt state/schema is invalid")
    unsigned = dict(value)
    supplied = require_digest(unsigned.pop("ready_digest"), "result ready digest")
    if spool_digest(DOMAIN_RESULT_READY, unsigned) != supplied:
        fail("result-ready receipt semantic digest mismatch")
    require_digest(value["archive_sha256"], "result archive digest")
    require_int(
        value["archive_size"], "result archive size", minimum=1, maximum=MAX_ARCHIVE_BYTES
    )
    for field in (
        "request_id",
        "ready_receipt_digest",
        "worker_epoch",
        "worker_readiness_digest",
    ):
        if value[field] != request[field]:
            fail(f"result-ready receipt changed request binding: {field}")
    if value["worker_epoch"] != registration["worker_epoch"]:
        fail("result-ready receipt came from a superseded worker")
    return value


def authenticated_wire_response(
    value: object,
    wire_request: Mapping[str, Any],
    *,
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> str:
    """Authenticate a wire response against its exact request; return its digest."""

    try:
        typed_request = RemoteEvaluationRequest.from_dict(wire_request)
        response = AuthenticatedRemoteEvaluationResponse.from_dict(value)
        reopen_remote_response(typed_request, response, identity, credential)
        return response.digest
    except RemoteEvaluationDispatcherError as exc:
        fail(f"authenticated remote response is invalid: {exc}")


def verify_adapter_result(
    row: object,
    root: Path,
    request: Mapping[str, Any],
    registration: Mapping[str, Any],
    *,
    request_root: Path,
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> dict[str, Any]:
    value = require_closed(row, ADAPTER_RESULT_FIELDS, "adapter result")
    if value["schema"] != SCHEMA_ADAPTER_RESULT or value["request_id"] != request["request_id"]:
        fail("adapter result schema/request binding is invalid")
    state = value["state"]
    if state not in {"completed", "no_decision"}:
        fail("adapter result state is invalid")
    artifacts = verify_artifacts(value["artifacts"], root, allow_output_roles=True)
    if not any(item["role"] == "adapter_result" for item in artifacts):
        fail("adapter result does not retain its typed payload bytes")
    if state == "no_decision":
        if (
            value["response_digest"] is not None
            or value["response_sha256"] is not None
            or value["failure_code"] not in ALLOWED_FAILURE_CODES
        ):
            fail("infrastructure no-decision result is malformed")
        return value
    if value["failure_code"] is not None:
        fail("completed adapter result contains an infrastructure failure")
    response_digest = require_digest(value["response_digest"], "response digest")
    response_sha = require_digest(value["response_sha256"], "response sha256")
    response_path = artifact_for_role(value, root, "adapter_result")
    if file_sha256(response_path) != response_sha:
        fail("authenticated response bytes differ from adapter result")
    wire_request = authenticated_wire_request(
        request, request_root, identity=identity, credential=credential
    )
    response_value = load_json(response_path, maximum=MAX_WIRE_PAYLOAD_BYTES)
    observed_response_digest = authenticated_wire_response(
        response_value, wire_request, identity=identity, credential=credential
    )
    if observed_response_digest != response_digest:
        fail("authenticated response digest differs from adapter result")
    try:
        typed_request = RemoteEvaluationRequest.from_dict(wire_request)
        response = AuthenticatedRemoteEvaluationResponse.from_dict(response_value)
        payload = reopen_remote_response(typed_request, response, identity, credential)
    except RemoteEvaluationDispatcherError as exc:
        fail(f"authenticated adapter response is invalid: {exc}")
    if not remote_hold.is_exact_remote_stage_payload(
        payload, request["lease"]["stage"]
    ):
        fail("completed response is not the exact stage payload")
    return value


def finalize_adapter_response(
    request: Mapping[str, Any],
    request_root: Path,
    result_root: Path,
    *,
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> None:
    """Turn the adapter's one canonical response.json into a closed result."""

    response_path = result_root / "response.json"
    if response_path.is_symlink() or not response_path.is_file():
        fail("fixed adapter did not write response.json")
    raw = response_path.read_bytes()
    if not raw or len(raw) > identity.max_response_bytes:
        fail("fixed adapter response is outside the registered bound")
    value = load_json(response_path, maximum=MAX_WIRE_PAYLOAD_BYTES)
    canonical = spool_canonical_json(value) + b"\n"
    if raw != canonical:
        fail("fixed adapter response is not canonical JSON plus newline")
    wire_request = authenticated_wire_request(
        request, request_root, identity=identity, credential=credential
    )
    response_digest = authenticated_wire_response(
        value, wire_request, identity=identity, credential=credential
    )
    digest = sha256_hex(raw)
    blobs = result_root / "blobs"
    blobs.mkdir(mode=0o700)
    blob = blobs / digest
    os.chmod(response_path, 0o400)
    os.replace(response_path, blob)
    atomic_json(
        result_root / "result.json",
        {
            "artifacts": [
                {"role": "adapter_result", "sha256": digest, "size": len(raw)}
            ],
            "failure_code": None,
            "request_id": request["request_id"],
            "response_digest": response_digest,
            "response_sha256": digest,
            "schema": SCHEMA_ADAPTER_RESULT,
            "state": "completed",
        },
        mode=0o400,
    )


def write_local_no_decision(
    results_root: Path, request: Mapping[str, Any], failure_code: str
) -> None:
    if failure_code not in ALLOWED_FAILURE_CODES:
        fail("local failure code is not registered")
    result_id = request["request_id"]
    final = results_root / result_id
    if final.exists():
        return
    temporary = Path(tempfile.mkdtemp(prefix=f".{result_id}.", dir=results_root))
    try:
        payload = (
            spool_canonical_json(
                {
                    "failure_code": failure_code,
                    "request_id": result_id,
                    "state": "no_decision",
                }
            )
            + b"\n"
        )
        digest = sha256_hex(payload)
        blobs = temporary / "blobs"
        blobs.mkdir(mode=0o700)
        atomic_bytes(blobs / digest, payload, mode=0o400)
        result = {
            "artifacts": [
                {"role": "adapter_result", "sha256": digest, "size": len(payload)}
            ],
            "failure_code": failure_code,
            "request_id": result_id,
            "response_digest": None,
            "response_sha256": None,
            "schema": SCHEMA_ADAPTER_RESULT,
            "state": "no_decision",
        }
        atomic_json(temporary / "result.json", result, mode=0o400)
        os.replace(temporary, final)
        atomic_bytes(final / "RESULT_READY", (result_id + "\n").encode(), mode=0o400)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def heartbeat_payload(
    registration: Mapping[str, Any],
    state: str,
    active_request_id: str | None,
    *,
    adapter_start_count: int = 0,
    adapter_alive: bool = False,
    consecutive_adapter_failures: int = 0,
) -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "active_request_id": active_request_id,
        "adapter_alive": adapter_alive,
        "adapter_start_count": adapter_start_count,
        "consecutive_adapter_failures": consecutive_adapter_failures,
        "pid": os.getpid(),
        "ready_receipt_digest": registration["ready_receipt_digest"],
        "schema": SCHEMA_HEARTBEAT,
        "state": state,
        "time_unix": int(time.time()),
        "worker_epoch": registration["worker_epoch"],
        "worker_readiness_digest": registration["worker_readiness_digest"],
    }
    return {
        **unsigned,
        "heartbeat_digest": spool_digest(DOMAIN_HEARTBEAT, unsigned),
    }


def verify_heartbeat(
    row: object, registration: Mapping[str, Any], max_age: int
) -> dict[str, Any]:
    value = require_closed(row, HEARTBEAT_FIELDS, "worker heartbeat")
    unsigned = dict(value)
    supplied = require_digest(unsigned.pop("heartbeat_digest"), "heartbeat digest")
    if value["schema"] != SCHEMA_HEARTBEAT or spool_digest(DOMAIN_HEARTBEAT, unsigned) != supplied:
        fail("worker heartbeat identity is invalid")
    for field in ("ready_receipt_digest", "worker_epoch", "worker_readiness_digest"):
        if value[field] != registration[field]:
            fail(f"worker heartbeat changed registration binding: {field}")
    observed = require_int(value["time_unix"], "heartbeat time", minimum=1)
    age = int(time.time()) - observed
    if age < -5 or age > max_age:
        fail(f"worker heartbeat is outside liveness bound: age={age}s")
    if value["active_request_id"] is not None:
        require_digest(value["active_request_id"], "active request id")
    if type(value["adapter_alive"]) is not bool:
        fail("worker heartbeat adapter liveness is not boolean")
    starts = require_int(
        value["adapter_start_count"],
        "worker heartbeat adapter start count",
        minimum=0,
    )
    failures = require_int(
        value["consecutive_adapter_failures"],
        "worker heartbeat consecutive adapter failures",
        minimum=0,
        maximum=MAX_CONSECUTIVE_ADAPTER_FAILURES,
    )
    if value["adapter_alive"] and starts == 0:
        fail("worker heartbeat reports an unstarted live adapter")
    if value["state"] == "epoch_failed" and failures < MAX_CONSECUTIVE_ADAPTER_FAILURES:
        fail("worker heartbeat failed epoch lacks its failure threshold")
    return value


def write_dispatch_state(
    job_dir: Path,
    request_id: str,
    state: str,
    worker_epoch: str,
    *,
    archive_sha256: str | None = None,
) -> None:
    value: dict[str, Any] = {
        "request_id": request_id,
        "schema": SCHEMA_DISPATCH_STATE,
        "state": state,
        "updated_at_unix": int(time.time()),
        "worker_epoch": worker_epoch,
    }
    if archive_sha256 is not None:
        value["archive_sha256"] = archive_sha256
    atomic_json(job_dir / "dispatch-state.json", value)


def iter_queue(
    outbox: Path,
    registration: Mapping[str, Any],
    *,
    identity: RemoteWorkerTransportIdentity,
    credential: RemoteWorkerCredential,
) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[int, str, Path, dict[str, Any]]] = []
    if not outbox.exists():
        return []
    for path in outbox.iterdir():
        if not path.is_dir() or path.is_symlink() or not (path / "REQUEST_READY").is_file():
            continue
        request = verify_request(
            load_json(path / "request.json"),
            path,
            registration,
            identity=identity,
            credential=credential,
        )
        rows.append((request["queued_at_unix_ns"], request["request_id"], path, request))
    rows.sort(key=lambda item: (item[0], item[1]))
    return [(path, request) for _, _, path, request in rows]
