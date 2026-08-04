#!/usr/bin/env python3
"""Closed, restartable CPU-validator <-> B300 worker transport.

This module is deployment plumbing, not evaluation authority.  The CPU side
relays immutable request archives from a trusted coordinator spool to one
explicitly registered worker epoch over host-key-pinned SSH.  The pod side
invokes exactly one root-owned, SHA-256-pinned adapter at a fixed path.  No
request field can select a command, module, executable, environment variable,
or output path.

The adapter is the deliberately small deployment-owned codec between this
wire format and ``B300MainnetWorker.run``.  It must write a closed
``cacheon-b300-adapter-result-v1`` result.  Transport, pod, and adapter
failures are returned as infrastructure ``no_decision`` records; this service
never manufactures a candidate FAIL, crown, settlement, or weight write.

Both daemons are ordinary foreground processes intended to be supervised by
tmux on the CPU validator and pod.  Their durable state is content addressed,
so either process may restart without rerunning an in-flight evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import hmac
import io
import json
import os
import re
import select
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn


SCHEMA_REGISTRATION = "cacheon-remote-worker-registration-v1"
SCHEMA_REQUEST = "cacheon-remote-evaluation-request-v1"
SCHEMA_ADAPTER_RESULT = "cacheon-b300-adapter-result-v1"
SCHEMA_RESULT_READY = "cacheon-remote-result-ready-v1"
SCHEMA_HEARTBEAT = "cacheon-remote-worker-heartbeat-v2"
SCHEMA_DISPATCH_STATE = "cacheon-remote-dispatch-state-v1"
SCHEMA_ADAPTER_COMMAND = "cacheon-b300-adapter-command-v1"
SCHEMA_ADAPTER_CONTROL = "cacheon-b300-adapter-control-v1"
SCHEMA_ADAPTER_LIFECYCLE_EVENT = "cacheon-b300-adapter-lifecycle-event-v1"

DOMAIN_REGISTRATION = "cacheon.chain.remote-worker-registration.v1"
DOMAIN_REQUEST = "cacheon.chain.remote-evaluation-request.v1"
DOMAIN_RESULT_READY = "cacheon.chain.remote-result-ready.v1"
DOMAIN_HEARTBEAT = "cacheon.chain.remote-worker-heartbeat.v1"
REQUEST_AUTH_DOMAIN = b"cacheon.remote-evaluation.request-auth.v1"
RESPONSE_AUTH_DOMAIN = b"cacheon.remote-evaluation.response-auth.v1"

POD_ROOT = Path("/data/cacheon-b300/remote-worker")
POD_READY_RECEIPT = Path(
    "/data/cacheon-b300/worker-bootstrap/ready-receipt.json"
)
POD_REGISTRATION = POD_ROOT / "registration.json"
POD_SERVICE = Path(
    "/data/cacheon-b300/worker-bootstrap/bin/remote_worker_service.py"
)
POD_ADAPTER = Path(
    "/data/cacheon-b300/worker-bootstrap/bin/"
    "cacheon-b300-evaluation-adapter"
)
POD_CREDENTIAL = POD_ROOT / "credential.secret"

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
EPOCH = re.compile(r"[0-9a-f]{32}\Z")
HOST = re.compile(r"[A-Za-z0-9.-]{1,253}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9._:@/+-]{1,512}\Z")
ROLE = re.compile(r"[a-z][a-z0-9._-]{0,63}\Z")
WIRE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
OWNER_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

READINESS_FIELDS = frozenset(
    {
        "arena_id",
        "gpu_count",
        "model_content_digest",
        "model_manifest_digest",
        "model_revision_digest",
        "provider_digest",
        "qualification_policy_digest",
        "ready_epoch",
        "ready_receipt_digest",
        "runtime_digest",
        "schema_version",
        "service_digest",
        "target_architecture",
        "tensor_parallel_size",
        "topology_class",
        "topology_digest",
        "worker_distribution_digest",
        "workload_digest",
    }
)
LEASE_FIELDS = frozenset(
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
LEASE_MEMBER_FIELDS = frozenset({"prior_status", "reservation_id"})
REGISTRATION_FIELDS = frozenset(
    {
        "adapter_sha256",
        "created_at_unix",
        "credential_digest",
        "credential_file_sha256",
        "credential_id",
        "credential_path",
        "known_hosts_path",
        "known_hosts_sha256",
        "lane_devices",
        "lane_digest",
        "pod_host",
        "pod_port",
        "pod_user",
        "python_executable",
        "python_executable_sha256",
        "ready_receipt_digest",
        "ready_receipt_file_sha256",
        "registration_digest",
        "remote_service_sha256",
        "schema",
        "service_identity",
        "transport_identity",
        "transport_identity_digest",
        "worker_epoch",
        "worker_readiness",
        "worker_readiness_digest",
    }
)
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
MAX_ARTIFACTS = 64
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
MAX_ARCHIVE_BYTES = 5 * 1024 * 1024 * 1024
MAX_JOB_SECONDS = 4 * 60 * 60
DEFAULT_POLL_SECONDS = 5
DEFAULT_HEARTBEAT_SECONDS = 10
DEFAULT_MAX_HEARTBEAT_AGE = 45
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


def canonical_json_bytes(value: object) -> bytes:
    """Strict canonical JSON compatible with Cacheon's identity helper."""

    def closed(item: object, location: str) -> object:
        if item is None or isinstance(item, (str, bool)):
            return item
        if type(item) is int:
            return item
        if isinstance(item, Mapping):
            result: dict[str, object] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    fail(f"{location} contains a non-string key")
                result[key] = closed(child, f"{location}.{key}")
            return result
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return [
                closed(child, f"{location}[{index}]")
                for index, child in enumerate(item)
            ]
        fail(f"{location} contains unsupported {type(item).__name__}")

    return json.dumps(
        closed(value, "payload"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def semantic_digest(domain: str, payload: object) -> str:
    envelope = {"domain": domain, "payload": payload, "schema_version": 1}
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


REMOTE_EVALUATION_PROTOCOL_DIGEST = semantic_digest(
    "cacheon.chain.remote-evaluation-protocol.v1",
    {
        "operations": ["screen", "qualification"],
        "request_auth": "hmac-sha256",
        "response_auth": "hmac-sha256",
        "result_encoding": "canonical-json",
        "shell_authority": False,
    },
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
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
    atomic_bytes(path, canonical_json_bytes(value) + b"\n", mode=mode)


def append_event(root: Path, event: str, **fields: object) -> None:
    row = {"event": event, "time_unix": int(time.time()), **fields}
    encoded = canonical_json_bytes(row) + b"\n"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "events.jsonl").open("ab", buffering=0) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(encoded)
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)
    print(encoded.decode("utf-8").rstrip(), flush=True)


def verify_ready_receipt(row: object) -> dict[str, Any]:
    if type(row) is dict and row.get("schema") == "cacheon-current-pod-commission-v1":
        fields = frozenset(
            {
                "created_at_unix",
                "gpu",
                "model",
                "lane",
                "provider",
                "python",
                "receipt_digest",
                "runtime",
                "schema",
                "source",
                "state",
                "worker_epoch",
                "worker_image",
            }
        )
        value = require_closed(row, fields, "current-pod commission receipt")
        if value["state"] != "READY_FOR_REGISTRATION":
            fail("current-pod commission is not READY_FOR_REGISTRATION")
        if not isinstance(value["worker_epoch"], str) or EPOCH.fullmatch(value["worker_epoch"]) is None:
            fail("commissioned worker epoch is malformed")
        supplied = require_digest(value["receipt_digest"], "commission receipt digest")
        unsigned = dict(value)
        del unsigned["receipt_digest"]
        if semantic_digest("cacheon.current-pod-commission.v1", unsigned) != supplied:
            fail("current-pod commission receipt semantic digest mismatch")
        gpu = value["gpu"]
        if (
            type(gpu) is not dict
            or set(gpu)
            != {"count", "inventory", "inventory_sha256", "topology_sha256"}
            or gpu["count"] != 8
            or type(gpu["inventory"]) is not list
            or len(gpu["inventory"]) != 8
            or any(
                type(item) is not dict
                or "B300" not in str(item.get("name", ""))
                for item in gpu["inventory"]
            )
        ):
            fail("current-pod commission GPU identity is not 8xB300")
        for section, expected in (
            ("source", {"path", "revision", "tree_digest"}),
            ("runtime", {"path", "tree_digest"}),
            (
                "model",
                {
                    "content_digest",
                    "path",
                    "receipt_digest",
                    "receipt_file_sha256",
                    "receipt_path",
                    "readonly_inventory_verified",
                },
            ),
            ("provider", {"hostname", "machine_id_sha256", "pod_endpoint"}),
            (
                "python",
                {"executable_sha256", "path", "resolved_path", "version"},
            ),
            (
                "lane",
                {"devices", "lane_digest", "tensor_parallel_size"},
            ),
        ):
            if type(value[section]) is not dict or set(value[section]) != expected:
                fail(f"current-pod commission {section} fields are not closed")
        for digest in (
            value["source"]["tree_digest"],
            value["runtime"]["tree_digest"],
            value["model"]["content_digest"],
            value["model"]["receipt_digest"],
            value["model"]["receipt_file_sha256"],
            value["python"]["executable_sha256"],
            value["lane"]["lane_digest"],
        ):
            require_digest(digest, "commission identity")
        if value["model"]["readonly_inventory_verified"] is not True:
            fail("commissioned model inventory is not read-only verified")
        python_path = Path(value["python"]["path"])
        resolved_python = Path(value["python"]["resolved_path"])
        if (
            not python_path.is_absolute()
            or not resolved_python.is_absolute()
            or not isinstance(value["python"]["version"], str)
            or not value["python"]["version"].startswith("Python 3.")
        ):
            fail("commissioned Python identity is malformed")
        lane_devices = value["lane"]["devices"]
        tp = require_int(
            value["lane"]["tensor_parallel_size"],
            "commissioned tensor parallel size",
            minimum=1,
            maximum=8,
        )
        if (
            type(lane_devices) is not list
            or len(lane_devices) != tp
            or any(type(device) is not int for device in lane_devices)
            or lane_devices != sorted(set(lane_devices))
            or any(device < 0 or device >= 8 for device in lane_devices)
        ):
            fail("commissioned lane devices are malformed")
        return value
    fields = frozenset(
        {
            "base_image",
            "build",
            "created_at",
            "gpu",
            "model",
            "provider",
            "receipt_digest",
            "runtime_seed",
            "schema",
            "source",
            "state",
            "venv",
            "worker_epoch",
            "worker_image",
        }
    )
    value = require_closed(row, fields, "READY receipt")
    if value["schema"] != "cacheon-lium-worker-ready-v1":
        fail("READY receipt schema is not supported")
    if value["state"] != "READY_FOR_REGISTRATION":
        fail("worker is not READY_FOR_REGISTRATION")
    if not isinstance(value["worker_epoch"], str) or EPOCH.fullmatch(value["worker_epoch"]) is None:
        fail("READY worker epoch is malformed")
    supplied = require_digest(value["receipt_digest"], "READY receipt digest")
    unsigned = dict(value)
    del unsigned["receipt_digest"]
    observed = hashlib.sha256(
        b"cacheon.lium-worker-ready.v1\0" + canonical_json_bytes(unsigned)
    ).hexdigest()
    if supplied != observed:
        fail("READY receipt semantic digest mismatch")
    gpu = value.get("gpu")
    if type(gpu) is not dict or gpu.get("count") != 8:
        fail("READY receipt is not for eight GPUs")
    inventory = gpu.get("inventory")
    if (
        type(inventory) is not list
        or len(inventory) != 8
        or any(type(item) is not dict or "B300" not in str(item.get("name", "")) for item in inventory)
    ):
        fail("READY receipt GPU inventory is not 8xB300")
    return value


def verify_readiness(row: object) -> tuple[dict[str, Any], str]:
    value = require_closed(row, READINESS_FIELDS, "WorkerReadiness")
    for field in (
        "ready_receipt_digest",
        "service_digest",
        "provider_digest",
        "runtime_digest",
        "worker_distribution_digest",
        "model_revision_digest",
        "model_manifest_digest",
        "model_content_digest",
        "topology_digest",
        "workload_digest",
        "qualification_policy_digest",
    ):
        require_digest(value[field], f"WorkerReadiness {field}")
    require_int(value["schema_version"], "WorkerReadiness schema_version", minimum=1, maximum=1)
    require_int(value["ready_epoch"], "WorkerReadiness ready_epoch")
    gpu_count = require_int(value["gpu_count"], "WorkerReadiness gpu_count", minimum=1)
    tensor_parallel = require_int(
        value["tensor_parallel_size"],
        "WorkerReadiness tensor_parallel_size",
        minimum=1,
    )
    if tensor_parallel > gpu_count:
        fail("WorkerReadiness tensor parallel exceeds GPU count")
    for field in ("arena_id", "target_architecture", "topology_class"):
        require_text(value[field], f"WorkerReadiness {field}")
    return value, semantic_digest("cacheon.chain.worker-readiness.v1", value)


def _sealed_tree_identity(root: Path, label: str) -> str:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        fail(f"{label} root is not an absolute regular directory")
    rows: list[dict[str, object]] = []
    total = 0
    count = 0
    for current, names, leaves in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        names[:] = sorted(name for name in names if name != ".git")
        for name in names:
            path = current_path / name
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                fail(f"{label} tree contains a non-directory carrier")
        for name in sorted(leaves):
            path = current_path / name
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_nlink != 1
            ):
                fail(f"{label} tree contains an unsafe file carrier")
            relative = path.relative_to(root).as_posix()
            if ".." in Path(relative).parts or "\x00" in relative:
                fail(f"{label} tree contains an unsafe logical path")
            size = info.st_size
            total += size
            count += 1
            if count > 200_000 or total > 50 * 1024 * 1024 * 1024:
                fail(f"{label} tree exceeds commissioning bounds")
            digest = file_sha256(path)
            after = path.lstat()
            if (
                after.st_dev != info.st_dev
                or after.st_ino != info.st_ino
                or after.st_size != info.st_size
                or after.st_mtime_ns != info.st_mtime_ns
            ):
                fail(f"{label} tree changed while hashing")
            rows.append(
                {
                    "mode": stat.S_IMODE(info.st_mode),
                    "path": relative,
                    "sha256": digest,
                    "size": size,
                }
            )
    if not rows:
        fail(f"{label} tree is empty")
    return semantic_digest(
        "cacheon.current-pod-tree.v1",
        {"files": rows, "label": label},
    )


def _verify_model_receipt(
    model_root: Path, receipt_path: Path
) -> tuple[str, str]:
    if (
        not model_root.is_absolute()
        or model_root.is_symlink()
        or not model_root.is_dir()
        or not receipt_path.is_absolute()
    ):
        fail("model or receipt path is not an absolute regular carrier")
    value = require_closed(
        load_json(receipt_path, maximum=64 << 20),
        frozenset({"content_digest", "files", "schema_version", "type"}),
        "model provision receipt",
    )
    if value["schema_version"] != 1 or value["type"] != "cacheon.model-provision":
        fail("model provision receipt schema/type is unsupported")
    files = value["files"]
    if type(files) is not list or not files:
        fail("model provision receipt file inventory is empty")
    canonical_files: list[dict[str, object]] = []
    prior = ""
    for raw in files:
        item = require_closed(
            raw, frozenset({"path", "sha256", "size"}), "model file record"
        )
        logical = require_text(item["path"], "model file path", maximum=4096)
        relative = Path(logical)
        if (
            relative.is_absolute()
            or relative.as_posix() != logical
            or ".." in relative.parts
            or logical <= prior
        ):
            fail("model provision receipt paths are not canonical and sorted")
        prior = logical
        digest = require_digest(item["sha256"], "model file digest")
        size = require_int(item["size"], "model file size")
        path = model_root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            fail(f"model receipt file is absent: {logical}")
        info = path.stat()
        if info.st_size != size or stat.S_IMODE(info.st_mode) & 0o222:
            fail(f"model file is changed-size or writable: {logical}")
        canonical_files.append({"path": logical, "sha256": digest, "size": size})
    content_digest = semantic_digest(
        "cacheon.model-content", {"files": canonical_files}
    )
    if content_digest != value["content_digest"]:
        fail("model provision content digest differs from inventory")
    receipt_digest = semantic_digest(
        "cacheon.model-provision-receipt", value
    )
    if receipt_path.name != f"model-provision-sha256-{receipt_digest}.json":
        fail("model receipt filename is not content addressed")
    if canonical_json_bytes(value) + b"\n" != receipt_path.read_bytes():
        fail("model receipt bytes are not canonical")
    return content_digest, receipt_digest


def install_source_archive(args: argparse.Namespace) -> None:
    """Install one exact Git archive under its content-addressed revision root."""

    if os.geteuid() != 0:
        fail("source installation must run as root")
    revision = require_text(
        args.source_revision,
        "source revision",
        maximum=40,
        pattern=re.compile(r"[0-9a-f]{40}\Z"),
    )
    archive_sha256 = require_digest(args.archive_sha256, "source archive digest")
    archive_path = Path(args.archive)
    incoming_root = Path("/data/cacheon-b300/worker-bootstrap/incoming")
    expected_name = f"source-{revision}-{archive_sha256}.tar"
    if (
        not archive_path.is_absolute()
        or archive_path.parent != incoming_root
        or archive_path.name != expected_name
        or archive_path.is_symlink()
        or not archive_path.is_file()
        or archive_path.stat().st_size > 2 * 1024 * 1024 * 1024
        or file_sha256(archive_path) != archive_sha256
    ):
        fail("source archive carrier or digest is invalid")
    destination = Path(f"/data/cacheon-b300/source-{revision}")
    if destination.exists():
        digest = _sealed_tree_identity(destination, "source")
        print(
            canonical_json_bytes(
                {
                    "path": str(destination),
                    "revision": revision,
                    "tree_digest": digest,
                }
            ).decode()
        )
        return
    temporary = Path(f"/data/cacheon-b300/.source-{revision}.{os.getpid()}")
    if temporary.exists():
        fail("source installation temporary path already exists")
    temporary.mkdir(mode=0o700)
    try:
        with tarfile.open(archive_path, "r:") as archive:
            members = archive.getmembers()
            if not members or len(members) > 200_000:
                fail("source archive member count is outside bounds")
            names: set[str] = set()
            total = 0
            for member in members:
                logical = Path(member.name)
                if (
                    member.name in names
                    or member.name.startswith("/")
                    or not member.name
                    or logical == Path(".")
                    or ".." in logical.parts
                    or any(part in {"", ".", ".git"} for part in logical.parts)
                    or OWNER_CONTROL.search(member.name) is not None
                    or not (member.isdir() or member.isfile())
                ):
                    fail("source archive contains an unsafe member")
                names.add(member.name)
                if member.isfile():
                    total += member.size
                    if member.size < 0 or total > 2 * 1024 * 1024 * 1024:
                        fail("source archive uncompressed bytes exceed bounds")
            for member in sorted(
                (row for row in members if row.isdir()),
                key=lambda row: len(Path(row.name).parts),
            ):
                temporary.joinpath(*Path(member.name).parts).mkdir(
                    parents=True, exist_ok=True, mode=0o700
                )
            for member in (row for row in members if row.isfile()):
                source = archive.extractfile(member)
                if source is None:
                    fail("source archive file is unreadable")
                target = temporary.joinpath(*Path(member.name).parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with target.open("xb") as output:
                    remaining = member.size
                    while remaining:
                        chunk = source.read(min(4 << 20, remaining))
                        if not chunk:
                            fail("source archive file was truncated")
                        output.write(chunk)
                        remaining -= len(chunk)
                    if source.read(1):
                        fail("source archive file exceeded its declared size")
                    output.flush()
                    os.fsync(output.fileno())
                os.chmod(target, 0o500 if member.mode & 0o111 else 0o400)
        if not (temporary / "cacheon" / "__init__.py").is_file():
            fail("source archive does not contain the Cacheon package")
        for current, directories, _files in os.walk(temporary, topdown=False):
            for name in directories:
                os.chmod(Path(current) / name, 0o500)
        os.chmod(temporary, 0o500)
        digest = _sealed_tree_identity(temporary, "source")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    print(
        canonical_json_bytes(
            {
                "path": str(destination),
                "revision": revision,
                "tree_digest": digest,
            }
        ).decode()
    )


def commission_current_pod(args: argparse.Namespace) -> None:
    if os.geteuid() != 0:
        fail("current-pod commissioning must run as root")
    source_root = Path(args.source_root)
    runtime_root = Path(args.runtime_root)
    model_root = Path(args.model_root)
    model_receipt = Path(args.model_receipt)
    python_executable = Path(args.python_executable)
    revision = require_text(
        args.source_revision,
        "source revision",
        maximum=40,
        pattern=re.compile(r"[0-9a-f]{40}\Z"),
    )
    worker_image = require_text(args.worker_image, "worker image", maximum=512)
    if re.fullmatch(r"[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}", worker_image) is None:
        fail("worker image must be an immutable repo digest")
    if (
        not python_executable.is_absolute()
        or not python_executable.exists()
        or not os.access(python_executable, os.X_OK)
    ):
        fail("commissioned Python executable is unavailable")
    try:
        resolved_python = python_executable.resolve(strict=True)
        python_version = subprocess.run(
            [str(python_executable), "--version"],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        fail(f"could not identify commissioned Python: {exc}")
    if not resolved_python.is_file() or not python_version.startswith("Python 3."):
        fail("commissioned Python is not a Python 3 executable")
    try:
        subprocess.run(
            ["docker", "image", "inspect", worker_image],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        fail(f"commissioned worker image is unavailable: {exc}")
    try:
        inventory_raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,pci.bus_id,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        ).stdout
        topology_raw = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        fail(f"could not capture B300 identity: {exc}")
    inventory = []
    for line in inventory_raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            fail("nvidia-smi inventory row is malformed")
        index, uuid, name, bus, memory = parts
        inventory.append(
            {
                "index": int(index),
                "memory_mib": int(memory),
                "name": name,
                "pci_bus_id": bus,
                "uuid": uuid,
            }
        )
    if (
        len(inventory) != 8
        or [row["index"] for row in inventory] != list(range(8))
        or any("B300" not in row["name"] for row in inventory)
    ):
        fail("commissioning requires exactly indexed 8xB300 inventory")
    try:
        lane_devices = [int(value) for value in args.lane_devices.split(",")]
    except (AttributeError, ValueError):
        fail("commissioned lane device list is malformed")
    if (
        not lane_devices
        or len(lane_devices) > 8
        or lane_devices != sorted(set(lane_devices))
        or any(device < 0 or device >= len(inventory) for device in lane_devices)
    ):
        fail("commissioned lane must be sorted unique physical GPU indexes")
    lane_digest = semantic_digest(
        "cacheon.current-pod-lane.v1",
        {
            "devices": lane_devices,
            "inventory": [inventory[device] for device in lane_devices],
            "topology_sha256": hashlib.sha256(topology_raw.encode()).hexdigest(),
        },
    )
    source_digest = _sealed_tree_identity(source_root, "source")
    runtime_digest = _sealed_tree_identity(runtime_root, "runtime")
    model_content, model_receipt_digest = _verify_model_receipt(
        model_root, model_receipt
    )
    output_path = Path(args.output)
    existing_receipt: dict[str, Any] | None = None
    if output_path.exists():
        existing_receipt = verify_ready_receipt(load_json(output_path))
        if existing_receipt["schema"] != "cacheon-current-pod-commission-v1":
            fail("refusing to replace a different READY receipt schema")
    epoch_path = POD_ROOT / "commission-worker-epoch"
    POD_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    if existing_receipt is not None:
        epoch = existing_receipt["worker_epoch"]
        if epoch_path.exists() and epoch_path.read_text(encoding="ascii").strip() != epoch:
            fail("commission epoch file differs from existing READY receipt")
        if not epoch_path.exists():
            atomic_bytes(epoch_path, (epoch + "\n").encode("ascii"), mode=0o400)
    elif epoch_path.exists():
        epoch = epoch_path.read_text(encoding="ascii").strip()
    else:
        epoch = os.urandom(16).hex()
        atomic_bytes(epoch_path, (epoch + "\n").encode("ascii"), mode=0o400)
    if EPOCH.fullmatch(epoch) is None:
        fail("commissioned worker epoch is malformed")
    machine = Path("/etc/machine-id")
    machine_digest = file_sha256(machine) if machine.is_file() else "0" * 64
    unsigned: dict[str, Any] = {
        "created_at_unix": (
            existing_receipt["created_at_unix"]
            if existing_receipt is not None
            else int(time.time())
        ),
        "gpu": {
            "count": 8,
            "inventory": inventory,
            "inventory_sha256": hashlib.sha256(inventory_raw.encode()).hexdigest(),
            "topology_sha256": hashlib.sha256(topology_raw.encode()).hexdigest(),
        },
        "model": {
            "content_digest": model_content,
            "path": str(model_root),
            "receipt_digest": model_receipt_digest,
            "receipt_file_sha256": file_sha256(model_receipt),
            "receipt_path": str(model_receipt),
            "readonly_inventory_verified": True,
        },
        "lane": {
            "devices": lane_devices,
            "lane_digest": lane_digest,
            "tensor_parallel_size": len(lane_devices),
        },
        "provider": {
            "hostname": os.uname().nodename,
            "machine_id_sha256": machine_digest,
            "pod_endpoint": args.pod_endpoint,
        },
        "python": {
            "executable_sha256": file_sha256(resolved_python),
            "path": str(python_executable),
            "resolved_path": str(resolved_python),
            "version": python_version,
        },
        "runtime": {"path": str(runtime_root), "tree_digest": runtime_digest},
        "schema": "cacheon-current-pod-commission-v1",
        "source": {
            "path": str(source_root),
            "revision": revision,
            "tree_digest": source_digest,
        },
        "state": "READY_FOR_REGISTRATION",
        "worker_epoch": epoch,
        "worker_image": worker_image,
    }
    receipt = {
        **unsigned,
        "receipt_digest": semantic_digest(
            "cacheon.current-pod-commission.v1", unsigned
        ),
    }
    verify_ready_receipt(receipt)
    if existing_receipt is not None and receipt != existing_receipt:
        fail("current pod identities differ from its existing commission receipt")
    atomic_json(output_path, receipt, mode=0o400)
    print(
        canonical_json_bytes(
            {
                "receipt_digest": receipt["receipt_digest"],
                "worker_epoch": epoch,
            }
        ).decode()
    )


def verify_lease(row: object) -> dict[str, Any]:
    value = require_closed(row, LEASE_FIELDS, "evaluation lease")
    require_digest(value["lease_id"], "lease_id")
    require_int(value["generation"], "lease generation", minimum=1)
    if value["stage"] not in {"screen", "qualification"}:
        fail("lease stage is malformed")
    require_text(value["owner"], "lease owner", maximum=256)
    claimed = require_int(value["claimed_block"], "claimed block")
    initial = require_int(value["initial_expires_block"], "initial expiry", minimum=1)
    expires = require_int(value["expires_block"], "lease expiry", minimum=1)
    if initial <= claimed or expires < initial:
        fail("lease block bounds are malformed")
    members = value["members"]
    if type(members) is not list or not members:
        fail("lease members are malformed")
    seen: set[str] = set()
    for item in members:
        member = require_closed(item, LEASE_MEMBER_FIELDS, "lease member")
        reservation = require_digest(member["reservation_id"], "reservation_id")
        if reservation in seen:
            fail("lease contains duplicate reservation")
        seen.add(reservation)
        if member["prior_status"] not in {
            "published",
            "promoted",
            "reproduction_pending",
        }:
            fail("lease member prior status is malformed")
    if value["stage"] == "screen" and len(members) != 1:
        fail("screen lease must contain one member")
    if value["stage"] == "qualification" and any(
        item["prior_status"] != "promoted" for item in members
    ):
        fail("qualification lease contains a non-promoted member")
    return value


def verify_registration(row: object) -> dict[str, Any]:
    value = require_closed(row, REGISTRATION_FIELDS, "worker registration")
    if value["schema"] != SCHEMA_REGISTRATION:
        fail("worker registration schema is not supported")
    unsigned = dict(value)
    supplied = require_digest(unsigned.pop("registration_digest"), "registration digest")
    if semantic_digest(DOMAIN_REGISTRATION, unsigned) != supplied:
        fail("worker registration semantic digest mismatch")
    for field in (
        "adapter_sha256",
        "credential_digest",
        "credential_file_sha256",
        "known_hosts_sha256",
        "lane_digest",
        "python_executable_sha256",
        "ready_receipt_digest",
        "ready_receipt_file_sha256",
        "remote_service_sha256",
        "transport_identity_digest",
        "worker_readiness_digest",
    ):
        require_digest(value[field], field)
    if not isinstance(value["worker_epoch"], str) or EPOCH.fullmatch(value["worker_epoch"]) is None:
        fail("registered worker epoch is malformed")
    require_text(value["pod_host"], "pod host", pattern=HOST)
    require_int(value["pod_port"], "pod port", minimum=1, maximum=65535)
    if value["pod_user"] != "root":
        fail("pod user must be root")
    readiness, digest = verify_readiness(value["worker_readiness"])
    if digest != value["worker_readiness_digest"]:
        fail("registered WorkerReadiness digest mismatch")
    if readiness["ready_receipt_digest"] != value["ready_receipt_digest"]:
        fail("registration and WorkerReadiness bind different READY receipts")
    lane_devices = value["lane_devices"]
    if (
        type(lane_devices) is not list
        or lane_devices != sorted(set(lane_devices))
        or len(lane_devices) != readiness["gpu_count"]
        or readiness["tensor_parallel_size"] != len(lane_devices)
        or any(type(device) is not int or device < 0 or device >= 8 for device in lane_devices)
    ):
        fail("registered physical lane differs from WorkerReadiness")
    python_executable = Path(value["python_executable"])
    if not python_executable.is_absolute():
        fail("registered Python executable must be absolute")
    require_text(value["service_identity"], "service identity")
    require_text(value["credential_id"], "credential id", maximum=128)
    credential_path = Path(value["credential_path"])
    if not credential_path.is_absolute():
        fail("credential path must be absolute")
    require_int(value["created_at_unix"], "registration creation time", minimum=1)
    transport = value["transport_identity"]
    transport_fields = frozenset(
        {
            "credential_digest",
            "endpoint_identity_digest",
            "max_response_bytes",
            "protocol_digest",
            "schema_version",
            "service_digest",
            "transport_id",
            "worker_readiness_digest",
        }
    )
    transport = require_closed(
        transport, transport_fields, "registered transport identity"
    )
    for field in (
        "credential_digest",
        "endpoint_identity_digest",
        "protocol_digest",
        "service_digest",
        "worker_readiness_digest",
    ):
        require_digest(transport[field], f"transport {field}")
    require_text(
        transport["transport_id"],
        "transport id",
        maximum=128,
        pattern=WIRE_IDENTIFIER,
    )
    require_int(
        transport["max_response_bytes"],
        "transport max response bytes",
        minimum=1,
        maximum=64 << 20,
    )
    require_int(
        transport["schema_version"],
        "transport schema version",
        minimum=1,
        maximum=1,
    )
    if transport["protocol_digest"] != REMOTE_EVALUATION_PROTOCOL_DIGEST:
        fail("registered remote evaluation protocol is unsupported")
    transport_digest = semantic_digest(
        "cacheon.chain.remote-worker-transport-identity.v1", transport
    )
    if transport_digest != value["transport_identity_digest"]:
        fail("registered transport identity digest mismatch")
    if (
        transport["credential_digest"] != value["credential_digest"]
        or transport["service_digest"] != readiness["service_digest"]
        or transport["worker_readiness_digest"] != digest
    ):
        fail("transport identity differs from credential/service/readiness")
    known_hosts = Path(value["known_hosts_path"])
    if not known_hosts.is_absolute():
        fail("known_hosts path must be absolute")
    return value


def make_registration(args: argparse.Namespace) -> None:
    ready_path = Path(args.ready_receipt)
    readiness_path = Path(args.worker_readiness)
    known_hosts_path = Path(args.known_hosts)
    service_path = Path(args.remote_service)
    adapter_path = Path(args.adapter)
    credential_path = Path(args.credential)
    ready = verify_ready_receipt(load_json(ready_path))
    readiness, readiness_digest = verify_readiness(load_json(readiness_path))
    if (
        args.bind_ready_receipt
        and readiness["ready_receipt_digest"] == "0" * 64
    ):
        readiness = {
            **readiness,
            "ready_receipt_digest": ready["receipt_digest"],
        }
        readiness, readiness_digest = verify_readiness(readiness)
    if ready["receipt_digest"] != readiness["ready_receipt_digest"]:
        fail("READY receipt and WorkerReadiness receipt digest differ")
    if ready["schema"] == "cacheon-current-pod-commission-v1":
        lane_devices = list(ready["lane"]["devices"])
        lane_digest = ready["lane"]["lane_digest"]
        python_executable = ready["python"]["path"]
        python_executable_sha256 = ready["python"]["executable_sha256"]
    else:
        try:
            lane_devices = [int(value) for value in args.lane_devices.split(",")]
        except (AttributeError, ValueError):
            fail("registered lane device list is malformed")
        if (
            not lane_devices
            or lane_devices != sorted(set(lane_devices))
            or any(device < 0 or device >= ready["gpu"]["count"] for device in lane_devices)
        ):
            fail("registered lane must be sorted unique physical GPU indexes")
        lane_digest = semantic_digest(
            "cacheon.registered-worker-lane.v1",
            {
                "devices": lane_devices,
                "ready_receipt_digest": ready["receipt_digest"],
            },
        )
        python_executable = require_text(
            args.python_executable, "registered Python executable", maximum=4096
        )
        python_path = Path(python_executable)
        if not python_path.is_absolute() or not python_path.exists():
            fail("registered Python executable is unavailable")
        python_executable_sha256 = file_sha256(python_path.resolve(strict=True))
    if (
        len(lane_devices) != readiness["gpu_count"]
        or readiness["tensor_parallel_size"] != len(lane_devices)
        or len(lane_devices) > ready["gpu"]["count"]
    ):
        fail("registered lane differs from WorkerReadiness GPU/TP identity")
    if not known_hosts_path.is_absolute() or known_hosts_path.is_symlink() or not known_hosts_path.is_file():
        fail("known_hosts must be an absolute regular file")
    if stat.S_IMODE(known_hosts_path.stat().st_mode) & 0o077:
        fail("known_hosts must not be group/world accessible")
    for path, name in (
        (service_path, "remote service"),
        (adapter_path, "adapter"),
        (credential_path, "credential"),
    ):
        if path.is_symlink() or not path.is_file():
            fail(f"{name} must be a regular file")
    credential_secret = credential_path.read_bytes()
    credential_id = require_text(
        args.credential_id,
        "credential id",
        maximum=128,
        pattern=WIRE_IDENTIFIER,
    )
    if not 32 <= len(credential_secret) <= 4096:
        fail("remote worker credential must contain 32 to 4096 bytes")
    credential_digest = semantic_digest(
        "cacheon.chain.remote-worker-credential.v1",
        {
            "credential_id": credential_id,
            "secret_sha256": hashlib.sha256(credential_secret).hexdigest(),
        },
    )
    endpoint_digest = semantic_digest(
        "cacheon.chain.remote-worker-endpoint.v1",
        {
            "known_hosts_sha256": file_sha256(known_hosts_path),
            "lane_digest": lane_digest,
            "pod_host": args.pod_host,
            "pod_port": args.pod_port,
            "pod_user": "root",
            "python_executable_sha256": python_executable_sha256,
            "ready_receipt_digest": ready["receipt_digest"],
            "worker_epoch": ready["worker_epoch"],
        },
    )
    max_response_bytes = require_int(
        args.max_response_bytes,
        "maximum response bytes",
        minimum=1,
        maximum=64 << 20,
    )
    transport = {
        "credential_digest": credential_digest,
        "endpoint_identity_digest": endpoint_digest,
        "max_response_bytes": max_response_bytes,
        "protocol_digest": REMOTE_EVALUATION_PROTOCOL_DIGEST,
        "schema_version": 1,
        "service_digest": readiness["service_digest"],
        "transport_id": f"lium-b300-{ready['worker_epoch'][:12]}",
        "worker_readiness_digest": readiness_digest,
    }
    transport_digest = semantic_digest(
        "cacheon.chain.remote-worker-transport-identity.v1", transport
    )
    endpoint = ready["provider"].get("pod_endpoint")
    if endpoint not in {"unknown", args.pod_host, f"{args.pod_host}:{args.pod_port}"}:
        fail("READY receipt was created for a different pod endpoint")
    value: dict[str, Any] = {
        "adapter_sha256": file_sha256(adapter_path),
        "created_at_unix": int(time.time()),
        "credential_digest": credential_digest,
        "credential_file_sha256": file_sha256(credential_path),
        "credential_id": credential_id,
        "credential_path": str(credential_path),
        "known_hosts_path": str(known_hosts_path),
        "known_hosts_sha256": file_sha256(known_hosts_path),
        "lane_devices": lane_devices,
        "lane_digest": lane_digest,
        "pod_host": require_text(args.pod_host, "pod host", pattern=HOST),
        "pod_port": require_int(args.pod_port, "pod port", minimum=1, maximum=65535),
        "pod_user": "root",
        "python_executable": python_executable,
        "python_executable_sha256": python_executable_sha256,
        "ready_receipt_digest": ready["receipt_digest"],
        "ready_receipt_file_sha256": file_sha256(ready_path),
        "remote_service_sha256": file_sha256(service_path),
        "schema": SCHEMA_REGISTRATION,
        "service_identity": require_text(args.service_identity, "service identity"),
        "transport_identity": transport,
        "transport_identity_digest": transport_digest,
        "worker_epoch": ready["worker_epoch"],
        "worker_readiness": readiness,
        "worker_readiness_digest": readiness_digest,
    }
    value["registration_digest"] = semantic_digest(DOMAIN_REGISTRATION, value)
    verify_registration(value)
    atomic_json(Path(args.output), value)
    print(
        canonical_json_bytes(
            {
                "registration_digest": value["registration_digest"],
                "worker_epoch": value["worker_epoch"],
            }
        ).decode()
    )


def verify_pod_registration(registration_path: Path) -> dict[str, Any]:
    registration = verify_registration(load_json(registration_path))
    ready = verify_ready_receipt(load_json(POD_READY_RECEIPT))
    if file_sha256(POD_READY_RECEIPT) != registration["ready_receipt_file_sha256"]:
        fail("pod READY receipt bytes differ from registration")
    if ready["receipt_digest"] != registration["ready_receipt_digest"]:
        fail("pod READY receipt digest differs from registration")
    if ready["worker_epoch"] != registration["worker_epoch"]:
        fail("pod worker epoch differs from registration")
    if ready.get("schema") == "cacheon-current-pod-commission-v1":
        if (
            ready["lane"]["devices"] != registration["lane_devices"]
            or ready["lane"]["lane_digest"] != registration["lane_digest"]
            or ready["python"]["path"] != registration["python_executable"]
            or ready["python"]["executable_sha256"]
            != registration["python_executable_sha256"]
        ):
            fail("pod Python/lane identity differs from registration")
    python_path = Path(registration["python_executable"])
    if not python_path.exists() or not os.access(python_path, os.X_OK):
        fail("registered pod Python executable is unavailable")
    if file_sha256(python_path.resolve(strict=True)) != registration["python_executable_sha256"]:
        fail("registered pod Python executable bytes changed")
    if file_sha256(POD_SERVICE) != registration["remote_service_sha256"]:
        fail("pod remote worker service differs from registration")
    if POD_CREDENTIAL.is_symlink() or not POD_CREDENTIAL.is_file():
        fail("pod remote worker credential is missing")
    credential_stat = POD_CREDENTIAL.stat()
    if credential_stat.st_uid != 0 or stat.S_IMODE(credential_stat.st_mode) & 0o077:
        fail("pod remote worker credential is not root-only")
    if file_sha256(POD_CREDENTIAL) != registration["credential_file_sha256"]:
        fail("pod remote worker credential bytes differ from registration")
    _credential_secret(registration, POD_CREDENTIAL)
    verify_fixed_adapter(registration)
    return registration


def _credential_secret(registration: Mapping[str, Any], path: Path) -> bytes:
    try:
        secret = path.read_bytes()
    except OSError as exc:
        fail(f"remote worker credential is unreadable: {exc}")
    if not 32 <= len(secret) <= 4096:
        fail("remote worker credential is outside its byte bounds")
    digest = semantic_digest(
        "cacheon.chain.remote-worker-credential.v1",
        {
            "credential_id": registration["credential_id"],
            "secret_sha256": hashlib.sha256(secret).hexdigest(),
        },
    )
    if digest != registration["credential_digest"]:
        fail("remote worker credential digest differs from registration")
    return secret


def _typed_credential(registration: Mapping[str, Any], path: Path):
    secret = _credential_secret(registration, path)
    try:
        from cacheon.chain.remote_evaluation_dispatcher import RemoteWorkerCredential

        credential = RemoteWorkerCredential(registration["credential_id"], secret)
    except (ImportError, TypeError, ValueError, RuntimeError) as exc:
        fail(f"typed remote worker credential is unavailable: {exc}")
    return credential


def _typed_transport_identity(registration: Mapping[str, Any]):
    try:
        from cacheon.chain.remote_evaluation_dispatcher import (
            RemoteWorkerTransportIdentity,
        )

        identity = RemoteWorkerTransportIdentity(**registration["transport_identity"])
    except (ImportError, TypeError, ValueError, RuntimeError) as exc:
        fail(f"remote worker transport identity is invalid: {exc}")
    if identity.digest != registration["transport_identity_digest"]:
        fail("remote worker transport identity digest differs")
    return identity


def verify_fixed_adapter(registration: Mapping[str, Any]) -> None:
    if POD_ADAPTER.is_symlink() or not POD_ADAPTER.is_file():
        fail(f"fixed deployment adapter is missing: {POD_ADAPTER}")
    metadata = POD_ADAPTER.stat()
    if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
        fail("fixed deployment adapter is not root-owned/read-only")
    if file_sha256(POD_ADAPTER) != registration["adapter_sha256"]:
        fail("fixed deployment adapter digest differs from registration")


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
        if not allow_output_roles and role in {
            "adapter_result",
            "evaluation_evidence",
            "worker_log",
        }:
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


def verify_request(row: object, root: Path, registration: Mapping[str, Any]) -> dict[str, Any]:
    value = require_closed(row, REQUEST_FIELDS, "remote evaluation request")
    if value["schema"] != SCHEMA_REQUEST:
        fail("remote evaluation request schema is unsupported")
    unsigned = dict(value)
    supplied = require_digest(unsigned.pop("request_id"), "request_id")
    if semantic_digest(DOMAIN_REQUEST, unsigned) != supplied:
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
    expected_kind = "screen_payload" if lease["stage"] == "screen" else "qualification_payload"
    artifacts = verify_artifacts(value["artifacts"], root, allow_output_roles=False)
    if not any(item["role"] == "candidate_publication" for item in artifacts):
        fail("request does not retain a candidate publication")
    if not any(item["role"] == expected_kind for item in artifacts):
        fail("request does not retain its stage payload")
    if lease["stage"] == "qualification":
        fail("remote qualification transport is gated pending evidence mirroring")
    wire = _authenticated_wire_request(value, root, registration)
    exact_members = wire["members"]
    if (
        wire["lease_id"] != lease["lease_id"]
        or wire["generation"] != lease["generation"]
        or wire["stage"] != lease["stage"]
        or wire["owner"] != lease["owner"]
        or exact_members != lease["members"]
        or wire["claimed_block"] != lease["claimed_block"]
        or wire["initial_expires_block"] != lease["initial_expires_block"]
        or wire["worker_readiness_digest"] != registration["worker_readiness_digest"]
        or wire["ready_receipt_digest"] != registration["ready_receipt_digest"]
        or wire["ready_epoch"] != registration["worker_readiness"]["ready_epoch"]
        or wire["service_identity"] != registration["service_identity"]
        or wire["transport_identity_digest"] != registration["transport_identity_digest"]
    ):
        fail("authenticated screen work differs from the transport lease")
    return value


def _artifact_for_role(
    request: Mapping[str, Any], root: Path, role: str
) -> Path:
    matches = [row for row in request["artifacts"] if row["role"] == role]
    if len(matches) != 1:
        fail(f"request must contain exactly one {role} artifact")
    return root / "blobs" / matches[0]["sha256"]


def _authenticated_wire_request(
    request: Mapping[str, Any],
    root: Path,
    registration: Mapping[str, Any],
):
    path = _artifact_for_role(request, root, "screen_payload")
    value = load_json(path, maximum=64 << 20)
    fields = frozenset(
        {
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
    )
    wire = require_closed(value, fields, "authenticated remote screen request")
    if wire["stage"] != "screen" or wire["body_kind"] != "screen_work":
        fail("authenticated remote request is not screen work")
    if wire["credential_id"] != registration["credential_id"]:
        fail("authenticated remote request credential id differs")
    for field in (
        "body_sha256",
        "lease_id",
        "ready_receipt_digest",
        "request_digest",
        "transport_identity_digest",
        "worker_readiness_digest",
    ):
        require_digest(wire[field], f"remote request {field}")
    require_digest(wire["auth_tag"], "remote request auth tag")
    require_int(wire["generation"], "remote request generation", minimum=1)
    require_int(wire["claimed_block"], "remote request claimed block")
    require_int(
        wire["initial_expires_block"],
        "remote request initial expiry",
        minimum=1,
    )
    require_int(wire["ready_epoch"], "remote request READY epoch")
    require_int(
        wire["schema_version"],
        "remote request schema version",
        minimum=1,
        maximum=1,
    )
    require_text(wire["owner"], "remote request owner", maximum=256)
    require_text(wire["service_identity"], "remote request service identity")
    if type(wire["members"]) is not list or not wire["members"]:
        fail("authenticated remote request members are malformed")
    for member in wire["members"]:
        require_closed(member, LEASE_MEMBER_FIELDS, "remote request member")
        require_digest(member["reservation_id"], "remote request reservation")
        if member["prior_status"] not in {"published", "reproduction_pending"}:
            fail("remote screen request member status is invalid")
    body = wire["body"]
    body_fields = frozenset(
        {
            "candidate_digest",
            "kind",
            "publication",
            "reservation",
            "schema_version",
            "screen_attempt",
            "screen_policy",
            "service_digest",
        }
    )
    body = require_closed(body, body_fields, "remote screen request body")
    if body["kind"] != "screen_work" or body["schema_version"] != 1:
        fail("remote screen request body kind/schema is invalid")
    require_digest(body["candidate_digest"], "remote candidate digest")
    require_digest(body["service_digest"], "remote service digest")
    require_int(body["screen_attempt"], "remote screen attempt", minimum=1)
    if body["service_digest"] != registration["worker_readiness"]["service_digest"]:
        fail("remote screen body service digest differs")
    if _contains_command_surface(body):
        fail("remote screen body contains a forbidden command surface")
    body_bytes = canonical_json_bytes(body)
    if hashlib.sha256(body_bytes).hexdigest() != wire["body_sha256"]:
        fail("remote screen body digest differs")
    unsigned = {
        "body_kind": wire["body_kind"],
        "body_sha256": wire["body_sha256"],
        "claimed_block": wire["claimed_block"],
        "credential_id": wire["credential_id"],
        "generation": wire["generation"],
        "initial_expires_block": wire["initial_expires_block"],
        "lease_id": wire["lease_id"],
        "members": wire["members"],
        "owner": wire["owner"],
        "ready_epoch": wire["ready_epoch"],
        "ready_receipt_digest": wire["ready_receipt_digest"],
        "schema_version": wire["schema_version"],
        "service_identity": wire["service_identity"],
        "stage": wire["stage"],
        "transport_identity_digest": wire["transport_identity_digest"],
        "worker_readiness_digest": wire["worker_readiness_digest"],
    }
    digest = semantic_digest("cacheon.chain.remote-evaluation-request.v1", unsigned)
    if digest != wire["request_digest"]:
        fail("authenticated remote request digest differs")
    credential_path = (
        POD_CREDENTIAL if POD_CREDENTIAL.exists() else Path(registration["credential_path"])
    )
    secret = _credential_secret(registration, credential_path)
    expected_auth = hmac.new(
        secret,
        REQUEST_AUTH_DOMAIN + b"\0" + digest.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(wire["auth_tag"], expected_auth):
        fail("authenticated remote request HMAC failed")
    return wire


def _contains_command_surface(value: object) -> bool:
    forbidden = {"argv", "command", "entrypoint", "env", "executable", "module", "shell"}
    if type(value) is dict:
        if forbidden & set(value):
            return True
        return any(_contains_command_surface(item) for item in value.values())
    if type(value) is list:
        return any(_contains_command_surface(item) for item in value)
    return False


def _parse_artifact_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        fail("artifact must be ROLE=/absolute/path")
    role, raw_path = value.split("=", 1)
    require_text(role, "artifact role", pattern=ROLE, maximum=64)
    if role not in ALLOWED_ARTIFACT_ROLES:
        fail(f"artifact role is not registered: {role}")
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        fail("artifact must be an absolute regular file")
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        fail("artifact exceeds per-file transfer limit")
    return role, path


def seal_request(args: argparse.Namespace) -> None:
    registration = verify_registration(load_json(Path(args.registration)))
    lease = verify_lease(load_json(Path(args.lease)))
    deadline_seconds = require_int(
        args.deadline_seconds,
        "deadline seconds",
        minimum=1,
        maximum=MAX_JOB_SECONDS,
    )
    artifact_inputs = [_parse_artifact_argument(item) for item in args.artifact]
    request_id, _ = enqueue_request(
        registration,
        lease,
        artifact_inputs,
        Path(args.outbox),
        deadline_seconds=deadline_seconds,
    )
    print(request_id)


def enqueue_request(
    registration: Mapping[str, Any],
    lease: Mapping[str, Any],
    artifact_inputs: Sequence[tuple[str, Path]],
    outbox: Path,
    *,
    deadline_seconds: int,
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
        request_id = semantic_digest(DOMAIN_REQUEST, unsigned)
        request = {**unsigned, "request_id": request_id}
        atomic_json(temporary / "request.json", request, mode=0o400)
        verify_request(request, temporary, registration)
        final = outbox / f"{queued_at:020d}-{request_id}"
        if final.exists():
            fail("request queue identity collision")
        os.replace(temporary, final)
        atomic_bytes(final / "REQUEST_READY", (request_id + "\n").encode(), mode=0o400)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return request_id, final


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o400
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def pack_directory(root: Path, manifest_name: str, artifacts: Iterable[Mapping[str, Any]], destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    try:
        with tarfile.open(temporary, "w") as archive:
            manifest = root / manifest_name
            data = manifest.read_bytes()
            archive.addfile(_tar_info(manifest_name, len(data)), io.BytesIO(data))
            seen: set[str] = set()
            for item in sorted(artifacts, key=lambda row: (str(row["sha256"]), str(row["role"]))):
                digest = str(item["sha256"])
                if digest in seen:
                    continue
                seen.add(digest)
                path = root / "blobs" / digest
                with path.open("rb") as handle:
                    archive.addfile(_tar_info(f"blobs/{digest}", path.stat().st_size), handle)
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


def run_checked(command: list[str], *, timeout: int = 60, capture: bool = False) -> subprocess.CompletedProcess[str]:
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
    registration: Mapping[str, Any], *arguments: str
) -> list[str]:
    for value in arguments:
        if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
            fail("remote service argument is not closed")
    return [registration["python_executable"], str(POD_SERVICE), *arguments]


def remote_json(registration: Mapping[str, Any], mode: str, request_id: str) -> dict[str, Any] | None:
    require_digest(request_id, "request id")
    completed = run_checked(
        ssh_base(registration)
        + remote_python_command(registration, mode, "--request-id", request_id),
        timeout=30,
        capture=True,
    )
    output = completed.stdout.strip()
    if not output:
        return None
    try:
        value = json.loads(output, object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError:
        fail("remote service returned malformed JSON")
    if type(value) is not dict:
        fail("remote service response is not an object")
    return value


def transfer_request(registration: Mapping[str, Any], job_dir: Path, request: Mapping[str, Any]) -> None:
    request_id = request["request_id"]
    archive_dir = job_dir / "transport"
    archive_dir.mkdir(exist_ok=True, mode=0o700)
    archive = archive_dir / f"{request_id}.tar"
    if not archive.exists():
        archive_sha = pack_directory(job_dir, "request.json", request["artifacts"], archive)
    else:
        archive_sha = file_sha256(archive)
    remote_part = f"{POD_ROOT}/incoming/.{request_id}.{archive_sha}.tar.part"
    run_checked(
        scp_base(registration)
        + ["--", str(archive), f"root@{registration['pod_host']}:{remote_part}"],
        timeout=1800,
    )
    run_checked(
        ssh_base(registration)
        + remote_python_command(
            registration,
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
    state = {
        "archive_sha256": archive_sha,
        "request_id": request_id,
        "schema": SCHEMA_DISPATCH_STATE,
        "state": "transferred",
        "updated_at_unix": int(time.time()),
        "worker_epoch": registration["worker_epoch"],
    }
    atomic_json(job_dir / "dispatch-state.json", state)


def pull_result(
    registration: Mapping[str, Any],
    job_dir: Path,
    request: Mapping[str, Any],
    results_root: Path,
) -> bool:
    request_id = request["request_id"]
    ready = remote_json(registration, "result-status", request_id)
    if ready is None or ready.get("state") != "ready":
        return False
    ready = verify_result_ready(ready, request, registration)
    archive_sha = ready["archive_sha256"]
    archive_size = ready["archive_size"]
    incoming = results_root / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive = incoming / f"{request_id}.{archive_sha}.tar"
    if not archive.exists():
        remote = f"root@{registration['pod_host']}:{POD_ROOT}/outgoing/{request_id}.{archive_sha}.tar"
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
    )
    final = results_root / request_id
    if final.exists():
        existing = verify_adapter_result(
            load_json(final / "result.json"),
            final,
            request,
            registration,
            request_root=job_dir,
        )
        if canonical_json_bytes(existing) != canonical_json_bytes(result):
            fail("local result identity collision")
        shutil.rmtree(temporary)
    else:
        os.replace(temporary, final)
        atomic_bytes(final / "RESULT_READY", (request_id + "\n").encode(), mode=0o400)
    run_checked(
        ssh_base(registration)
        + remote_python_command(
            registration,
            "ack-result",
            "--request-id",
            request_id,
            "--archive-sha256",
            archive_sha,
        ),
        timeout=60,
    )
    state = {
        "archive_sha256": archive_sha,
        "request_id": request_id,
        "schema": SCHEMA_DISPATCH_STATE,
        "state": "result_received",
        "updated_at_unix": int(time.time()),
        "worker_epoch": registration["worker_epoch"],
    }
    atomic_json(job_dir / "dispatch-state.json", state)
    return True


def verify_result_ready(
    row: object,
    request: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    fields = frozenset(
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
    value = require_closed(row, fields, "result-ready receipt")
    if value["schema"] != SCHEMA_RESULT_READY or value["state"] != "ready":
        fail("result-ready receipt state/schema is invalid")
    unsigned = dict(value)
    supplied = require_digest(unsigned.pop("ready_digest"), "result ready digest")
    if semantic_digest(DOMAIN_RESULT_READY, unsigned) != supplied:
        fail("result-ready receipt semantic digest mismatch")
    require_digest(value["archive_sha256"], "result archive digest")
    require_int(value["archive_size"], "result archive size", minimum=1, maximum=MAX_ARCHIVE_BYTES)
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


def verify_adapter_result(
    row: object,
    root: Path,
    request: Mapping[str, Any],
    registration: Mapping[str, Any],
    *,
    request_root: Path,
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
    response_path = _artifact_for_role(value, root, "adapter_result")
    if file_sha256(response_path) != response_sha:
        fail("authenticated response bytes differ from adapter result")
    wire_request = _authenticated_wire_request(request, request_root, registration)
    response_value = load_json(response_path, maximum=64 << 20)
    observed_response_digest = _authenticated_wire_response(
        response_value,
        wire_request,
        registration,
    )
    if observed_response_digest != response_digest:
        fail("authenticated response digest differs from adapter result")
    # The CPU deployment has the new typed protocol implementation and reopens
    # the receipt again before commit.  A pod built from the older frozen
    # controller source may not; its fixed adapter is the typed authority there.
    try:
        from cacheon.chain.remote_evaluation_dispatcher import (
            AuthenticatedRemoteEvaluationResponse,
            reopen_remote_response,
        )
        from cacheon.chain.remote_evaluation_dispatcher import RemoteEvaluationRequest

        typed_request = RemoteEvaluationRequest.from_dict(wire_request)
        response = AuthenticatedRemoteEvaluationResponse.from_dict(response_value)
        credential_path = (
            POD_CREDENTIAL if POD_CREDENTIAL.exists() else Path(registration["credential_path"])
        )
        payload = reopen_remote_response(
            typed_request,
            response,
            _typed_transport_identity(registration),
            _typed_credential(registration, credential_path),
        )
        if request["lease"]["stage"] != "screen" or type(payload).__name__ != "ArenaScreenReceipt":
            fail("completed response is not an exact screen receipt")
    except RemoteWorkerError:
        raise
    except ImportError:
        pass
    except (TypeError, ValueError, RuntimeError) as exc:
        fail(f"authenticated adapter response is invalid: {exc}")
    return value


def _authenticated_wire_response(
    value: object,
    request: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> str:
    fields = frozenset(
        {
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
    )
    response = require_closed(value, fields, "authenticated remote screen response")
    for field in (
        "auth_tag",
        "payload_digest",
        "payload_sha256",
        "ready_receipt_digest",
        "request_digest",
        "response_digest",
        "transport_identity_digest",
        "worker_readiness_digest",
    ):
        require_digest(response[field], f"remote response {field}")
    if (
        response["credential_id"] != registration["credential_id"]
        or response["request_digest"] != request["request_digest"]
        or response["transport_identity_digest"]
        != registration["transport_identity_digest"]
        or response["worker_readiness_digest"]
        != registration["worker_readiness_digest"]
        or response["ready_receipt_digest"]
        != registration["ready_receipt_digest"]
        or response["ready_epoch"]
        != registration["worker_readiness"]["ready_epoch"]
        or response["stage"] != "screen"
        or response["payload_kind"] != "arena_screen_receipt"
        or response["schema_version"] != 1
        or type(response["payload"]) is not dict
    ):
        fail("authenticated remote response changed request/worker authority")
    payload_bytes = canonical_json_bytes(response["payload"])
    if hashlib.sha256(payload_bytes).hexdigest() != response["payload_sha256"]:
        fail("authenticated remote response payload bytes differ")
    unsigned = {
        "credential_id": response["credential_id"],
        "payload_digest": response["payload_digest"],
        "payload_kind": response["payload_kind"],
        "payload_sha256": response["payload_sha256"],
        "ready_epoch": response["ready_epoch"],
        "ready_receipt_digest": response["ready_receipt_digest"],
        "request_digest": response["request_digest"],
        "schema_version": response["schema_version"],
        "stage": response["stage"],
        "transport_identity_digest": response["transport_identity_digest"],
        "worker_readiness_digest": response["worker_readiness_digest"],
    }
    digest = semantic_digest("cacheon.chain.remote-evaluation-response.v1", unsigned)
    if digest != response["response_digest"]:
        fail("authenticated remote response digest differs")
    credential_path = (
        POD_CREDENTIAL if POD_CREDENTIAL.exists() else Path(registration["credential_path"])
    )
    secret = _credential_secret(registration, credential_path)
    expected_auth = hmac.new(
        secret,
        RESPONSE_AUTH_DOMAIN + b"\0" + digest.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(response["auth_tag"], expected_auth):
        fail("authenticated remote response HMAC failed")
    return digest


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
    return {**unsigned, "heartbeat_digest": semantic_digest(DOMAIN_HEARTBEAT, unsigned)}


def verify_heartbeat(row: object, registration: Mapping[str, Any], max_age: int) -> dict[str, Any]:
    fields = frozenset(
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
    value = require_closed(row, fields, "worker heartbeat")
    unsigned = dict(value)
    supplied = require_digest(unsigned.pop("heartbeat_digest"), "heartbeat digest")
    if value["schema"] != SCHEMA_HEARTBEAT or semantic_digest(DOMAIN_HEARTBEAT, unsigned) != supplied:
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


def remote_heartbeat(registration: Mapping[str, Any], max_age: int) -> dict[str, Any]:
    completed = run_checked(
        ssh_base(registration)
        + remote_python_command(registration, "heartbeat-status"),
        timeout=30,
        capture=True,
    )
    output = completed.stdout.strip()
    if not output:
        fail("pod worker heartbeat is absent")
    try:
        value = json.loads(output, object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError:
        fail("pod worker heartbeat is malformed")
    return verify_heartbeat(value, registration, max_age)


def registration_is_current(registration: Mapping[str, Any], current_path: Path) -> bool:
    if current_path.is_symlink() or not current_path.is_file():
        return False
    try:
        current = verify_registration(load_json(current_path))
    except RemoteWorkerError:
        return False
    return current["registration_digest"] == registration["registration_digest"]


def iter_queue(outbox: Path, registration: Mapping[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[int, str, Path, dict[str, Any]]] = []
    if not outbox.exists():
        return []
    for path in outbox.iterdir():
        if not path.is_dir() or path.is_symlink() or not (path / "REQUEST_READY").is_file():
            continue
        request = verify_request(load_json(path / "request.json"), path, registration)
        rows.append((request["queued_at_unix_ns"], request["request_id"], path, request))
    rows.sort(key=lambda item: (item[0], item[1]))
    return [(path, request) for _, _, path, request in rows]


def cpu_serve(args: argparse.Namespace) -> None:
    registration_path = Path(args.registration)
    current_path = Path(args.current_registration)
    registration = verify_registration(load_json(registration_path))
    spool = Path(args.spool_root)
    outbox = spool / "outbox"
    results = spool / "results"
    state_root = spool / "state"
    for path in (outbox, results, state_root):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = state_root / f"dispatch-{registration['worker_epoch']}.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("another dispatcher already owns this worker epoch")
        append_event(spool, "dispatcher_started", worker_epoch=registration["worker_epoch"])
        while True:
            if not registration_is_current(registration, current_path):
                append_event(spool, "dispatcher_superseded", worker_epoch=registration["worker_epoch"])
                return
            try:
                heartbeat = remote_heartbeat(registration, args.max_heartbeat_age)
                if heartbeat["state"] == "epoch_failed":
                    append_event(
                        spool,
                        "dispatcher_epoch_failed",
                        adapter_start_count=heartbeat["adapter_start_count"],
                        consecutive_adapter_failures=heartbeat[
                            "consecutive_adapter_failures"
                        ],
                        worker_epoch=registration["worker_epoch"],
                    )
                    return
                queue = iter_queue(outbox, registration)
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
                        )
                        atomic_json(
                            state_path,
                            {
                                "request_id": request_id,
                                "schema": SCHEMA_DISPATCH_STATE,
                                "state": "result_received",
                                "updated_at_unix": int(time.time()),
                                "worker_epoch": registration["worker_epoch"],
                            },
                        )
                        continue
                    if pull_result(registration, job_dir, request, results):
                        append_event(spool, "result_received", request_id=request_id)
                        active = None
                        continue
                    if active not in {None, request_id}:
                        break
                    if state is None or state.get("state") != "transferred":
                        if request["deadline_unix"] <= int(time.time()):
                            write_local_no_decision(
                                results,
                                request,
                                "request_deadline_elapsed",
                            )
                            atomic_json(
                                state_path,
                                {
                                    "request_id": request_id,
                                    "schema": SCHEMA_DISPATCH_STATE,
                                    "state": "result_received",
                                    "updated_at_unix": int(time.time()),
                                    "worker_epoch": registration["worker_epoch"],
                                },
                            )
                            append_event(spool, "request_expired_before_transfer", request_id=request_id)
                            continue
                        transfer_request(registration, job_dir, request)
                        append_event(spool, "request_transferred", request_id=request_id)
                    break
                atomic_json(state_root / "heartbeat.json", heartbeat_payload(registration, "running", active))
            except RemoteWorkerError as exc:
                append_event(spool, "dispatcher_retry", error=str(exc)[:1024])
            time.sleep(args.poll_seconds)


def write_local_no_decision(results_root: Path, request: Mapping[str, Any], failure_code: str) -> None:
    if failure_code not in ALLOWED_FAILURE_CODES:
        fail("local failure code is not registered")
    result_id = request["request_id"]
    final = results_root / result_id
    if final.exists():
        return
    temporary = Path(tempfile.mkdtemp(prefix=f".{result_id}.", dir=results_root))
    try:
        payload = canonical_json_bytes(
            {"failure_code": failure_code, "request_id": result_id, "state": "no_decision"}
        ) + b"\n"
        digest = hashlib.sha256(payload).hexdigest()
        blobs = temporary / "blobs"
        blobs.mkdir(mode=0o700)
        atomic_bytes(blobs / digest, payload, mode=0o400)
        result = {
            "artifacts": [{"role": "adapter_result", "sha256": digest, "size": len(payload)}],
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


def accept_request(args: argparse.Namespace) -> None:
    registration = verify_pod_registration(POD_REGISTRATION)
    request_id = require_digest(args.request_id, "request id")
    archive_sha = require_digest(args.archive_sha256, "request archive digest")
    archive_size = require_int(args.archive_size, "request archive size", minimum=1, maximum=MAX_ARCHIVE_BYTES)
    incoming = POD_ROOT / "incoming"
    part = incoming / f".{request_id}.{archive_sha}.tar.part"
    final = incoming / f"{request_id}.{archive_sha}.tar"
    ready = incoming / f"{request_id}.ready.json"
    if final.exists() and ready.exists():
        if final.stat().st_size != archive_size or file_sha256(final) != archive_sha:
            fail("existing request archive identity collision")
        print("accepted")
        return
    if part.is_symlink() or not part.is_file():
        fail("incoming request archive part is absent")
    if part.stat().st_size != archive_size or file_sha256(part) != archive_sha:
        fail("incoming request archive checksum mismatch")
    scratch = POD_ROOT / "verify" / f".{request_id}.{os.getpid()}"
    if scratch.exists():
        shutil.rmtree(scratch)
    safe_extract(part, scratch)
    try:
        request = verify_request(load_json(scratch / "request.json"), scratch, registration)
        if request["request_id"] != request_id:
            fail("incoming request archive changed request id")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    os.chmod(part, 0o400)
    os.replace(part, final)
    atomic_json(
        ready,
        {
            "archive_sha256": archive_sha,
            "archive_size": archive_size,
            "request_id": request_id,
            "worker_epoch": registration["worker_epoch"],
        },
        mode=0o400,
    )
    append_event(POD_ROOT, "request_accepted", request_id=request_id)
    print("accepted")


def _adapter_environment(
    registration: Mapping[str, Any],
    request_id: str | None,
) -> dict[str, str]:
    environment = {
        "CACHEON_REMOTE_READY_RECEIPT_DIGEST": registration["ready_receipt_digest"],
        "CACHEON_REMOTE_CREDENTIAL_PATH": str(POD_CREDENTIAL),
        "CACHEON_REMOTE_REGISTRATION_PATH": str(POD_REGISTRATION),
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
        "TMPDIR": "/data/cacheon-b300/remote-worker/tmp",
    }
    if request_id is not None:
        environment["CACHEON_REMOTE_REQUEST_ID"] = request_id
    return environment


def infrastructure_result(request: Mapping[str, Any], result_root: Path, failure_code: str) -> None:
    if failure_code not in ALLOWED_FAILURE_CODES:
        fail("pod failure code is not registered")
    result_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    blobs = result_root / "blobs"
    blobs.mkdir(mode=0o700)
    payload = canonical_json_bytes(
        {
            "failure_code": failure_code,
            "request_id": request["request_id"],
            "state": "no_decision",
        }
    ) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    atomic_bytes(blobs / digest, payload, mode=0o400)
    atomic_json(
        result_root / "result.json",
        {
            "artifacts": [{"role": "adapter_result", "sha256": digest, "size": len(payload)}],
            "failure_code": failure_code,
            "request_id": request["request_id"],
            "response_digest": None,
            "response_sha256": None,
            "schema": SCHEMA_ADAPTER_RESULT,
            "state": "no_decision",
        },
        mode=0o400,
    )


def finalize_adapter_response(
    registration: Mapping[str, Any],
    request: Mapping[str, Any],
    request_root: Path,
    result_root: Path,
) -> None:
    """Turn the adapter's one canonical response.json into a closed result."""

    response_path = result_root / "response.json"
    if response_path.is_symlink() or not response_path.is_file():
        fail("fixed adapter did not write response.json")
    raw = response_path.read_bytes()
    if not raw or len(raw) > registration["transport_identity"]["max_response_bytes"]:
        fail("fixed adapter response is outside the registered bound")
    value = load_json(response_path, maximum=64 << 20)
    canonical = canonical_json_bytes(value) + b"\n"
    if raw != canonical:
        fail("fixed adapter response is not canonical JSON plus newline")
    wire_request = _authenticated_wire_request(request, request_root, registration)
    response_digest = _authenticated_wire_response(
        value, wire_request, registration
    )
    digest = hashlib.sha256(raw).hexdigest()
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


def publish_result(
    registration: Mapping[str, Any],
    request: Mapping[str, Any],
    result_root: Path,
    *,
    request_root: Path,
) -> None:
    result = verify_adapter_result(
        load_json(result_root / "result.json"),
        result_root,
        request,
        registration,
        request_root=request_root,
    )
    outgoing = POD_ROOT / "outgoing"
    outgoing.mkdir(parents=True, exist_ok=True, mode=0o700)
    request_id = request["request_id"]
    temporary_archive = outgoing / f".{request_id}.tar"
    archive_sha = pack_directory(result_root, "result.json", result["artifacts"], temporary_archive)
    archive_size = temporary_archive.stat().st_size
    final_archive = outgoing / f"{request_id}.{archive_sha}.tar"
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
    ready = {**unsigned, "ready_digest": semantic_digest(DOMAIN_RESULT_READY, unsigned)}
    atomic_json(outgoing / f"{request_id}.ready.json", ready, mode=0o400)
    append_event(POD_ROOT, "result_published", request_id=request_id, state=result["state"])


def recover_interrupted(registration: Mapping[str, Any]) -> None:
    processing = POD_ROOT / "processing"
    results = POD_ROOT / "results"
    if not processing.exists():
        return
    for job in sorted(processing.iterdir()):
        if not job.is_dir() or job.is_symlink() or HEX64.fullmatch(job.name) is None:
            continue
        request = verify_request(load_json(job / "request.json"), job, registration)
        result_root = results / request["request_id"]
        if not result_root.exists():
            infrastructure_result(request, result_root, "pod_service_restart")
        publish_result(
            registration,
            request,
            result_root,
            request_root=job,
        )
        completed = POD_ROOT / "completed" / request["request_id"]
        completed.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not completed.exists():
            os.replace(job, completed)


def _decode_adapter_control(raw: bytes) -> dict[str, Any]:
    if not raw or len(raw) > 64 * 1024 or not raw.endswith(b"\n"):
        fail("persistent adapter emitted a malformed control frame")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_float=_reject_constant,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        fail(f"persistent adapter emitted invalid JSON: {exc}")
    if (
        type(value) is not dict
        or raw != canonical_json_bytes(value) + b"\n"
        or value.get("schema") != SCHEMA_ADAPTER_CONTROL
    ):
        fail("persistent adapter control frame is not canonical")
    return value


class _PersistentAdapterProcess:
    """One fixed adapter process and commissioned worker per pod epoch."""

    def __init__(
        self,
        registration: Mapping[str, Any],
        *,
        heartbeat_seconds: int,
    ) -> None:
        self.registration = registration
        self.heartbeat_seconds = heartbeat_seconds
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
        self.consecutive_failures = (
            0 if completed else self.consecutive_failures + 1
        )

    def _heartbeat(self, request_id: str | None, state: str) -> None:
        atomic_json(
            POD_ROOT / "heartbeat.json",
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
                return _decode_adapter_control(raw)
            except RemoteWorkerError:
                return None
        return None

    def _start(self, *, deadline: int, request_id: str) -> bool:
        verify_fixed_adapter(self.registration)
        log_root = POD_ROOT / "logs"
        log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.log_handle = (log_root / "persistent-adapter.log").open(
            "ab", buffering=0
        )
        self.process = subprocess.Popen(
            [
                self.registration["python_executable"],
                str(POD_ADAPTER),
                "--serve",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_handle,
            env=_adapter_environment(self.registration, None),
            start_new_session=True,
        )
        self.start_count += 1
        append_event(
            POD_ROOT,
            "adapter_process_started",
            adapter_pid=self.process.pid,
            adapter_start_count=self.start_count,
            request_id=request_id,
            schema=SCHEMA_ADAPTER_LIFECYCLE_EVENT,
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
                POD_ROOT,
                "adapter_process_start_failed",
                adapter_start_count=self.start_count,
                request_id=request_id,
                schema=SCHEMA_ADAPTER_LIFECYCLE_EVENT,
                worker_epoch=self.registration["worker_epoch"],
            )
            self.close()
            return False
        append_event(
            POD_ROOT,
            "adapter_process_ready",
            adapter_pid=self.process.pid,
            adapter_start_count=self.start_count,
            request_id=request_id,
            schema=SCHEMA_ADAPTER_LIFECYCLE_EVENT,
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
            process.stdin.write(canonical_json_bytes(command) + b"\n")
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


def next_incoming(registration: Mapping[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    incoming = POD_ROOT / "incoming"
    candidates: list[tuple[int, str, Path, dict[str, Any]]] = []
    for marker in incoming.glob("*.ready.json"):
        request_id = marker.name[: -len(".ready.json")]
        if HEX64.fullmatch(request_id) is None:
            continue
        receipt = load_json(marker)
        if set(receipt) != {"archive_sha256", "archive_size", "request_id", "worker_epoch"}:
            fail("incoming request marker fields are not closed")
        if receipt["request_id"] != request_id or receipt["worker_epoch"] != registration["worker_epoch"]:
            fail("incoming request marker binding is invalid")
        archive_sha = require_digest(receipt["archive_sha256"], "incoming archive digest")
        archive = incoming / f"{request_id}.{archive_sha}.tar"
        scratch = POD_ROOT / "inbox" / request_id
        if scratch.exists():
            request = verify_request(load_json(scratch / "request.json"), scratch, registration)
        else:
            safe_extract(archive, scratch)
            request = verify_request(load_json(scratch / "request.json"), scratch, registration)
        candidates.append((request["queued_at_unix_ns"], request_id, scratch, request))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, request_id, scratch, request = candidates[0]
    processing = POD_ROOT / "processing" / request_id
    if not processing.exists():
        os.replace(scratch, processing)
    marker = incoming / f"{request_id}.ready.json"
    marker.unlink(missing_ok=True)
    return processing, request


def run_adapter(
    registration: Mapping[str, Any],
    job_root: Path,
    request: Mapping[str, Any],
    heartbeat_seconds: int,
    adapter_process: _PersistentAdapterProcess | None = None,
) -> Path:
    request_id = request["request_id"]
    results = POD_ROOT / "results"
    results.mkdir(parents=True, exist_ok=True, mode=0o700)
    final = results / request_id
    if final.exists():
        verify_adapter_result(
            load_json(final / "result.json"),
            final,
            request,
            registration,
            request_root=job_root,
        )
        return final
    temporary = results / f".{request_id}.{os.getpid()}"
    temporary.mkdir(mode=0o700)
    deadline = min(request["deadline_unix"], int(time.time()) + MAX_JOB_SECONDS)
    owns_process = adapter_process is None
    process = adapter_process or _PersistentAdapterProcess(
        registration,
        heartbeat_seconds=heartbeat_seconds,
    )
    try:
        failure = process.evaluate(
            request,
            job_root,
            temporary,
            deadline=deadline,
        )
    finally:
        if owns_process:
            process.close()
    if failure is not None:
        shutil.rmtree(temporary, ignore_errors=True)
        infrastructure_result(request, temporary, failure)
    else:
        try:
            finalize_adapter_response(
                registration,
                request,
                job_root,
                temporary,
            )
            verify_adapter_result(
                load_json(temporary / "result.json"),
                temporary,
                request,
                registration,
                request_root=job_root,
            )
        except RemoteWorkerError:
            shutil.rmtree(temporary, ignore_errors=True)
            infrastructure_result(request, temporary, "adapter_result_invalid")
    os.replace(temporary, final)
    return final


def pod_serve(args: argparse.Namespace) -> None:
    registration = verify_pod_registration(POD_REGISTRATION)
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
        (POD_ROOT / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = POD_ROOT / "service.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fail("another pod worker service already owns this root")
        recover_interrupted(registration)
        append_event(POD_ROOT, "pod_service_started", worker_epoch=registration["worker_epoch"])
        adapter_process = _PersistentAdapterProcess(
            registration,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        try:
            while True:
                verify_pod_registration(POD_REGISTRATION)
                adapter_process._heartbeat(None, "idle")
                item = next_incoming(registration)
                if item is not None:
                    job_root, request = item
                    request_id = request["request_id"]
                    append_event(POD_ROOT, "adapter_started", request_id=request_id)
                    result = run_adapter(
                        registration,
                        job_root,
                        request,
                        args.heartbeat_seconds,
                        adapter_process,
                    )
                    result_row = verify_adapter_result(
                        load_json(result / "result.json"),
                        result,
                        request,
                        registration,
                        request_root=job_root,
                    )
                    # A typed request-local refusal leaves the commissioned
                    # adapter and resident model healthy.  Every other adapter
                    # failure is epoch-fatal on its first occurrence.
                    adapter_process.record_result(
                        completed=(
                            result_row["state"] == "completed"
                            or result_row["failure_code"]
                            == "adapter_request_failed"
                        )
                    )
                    publish_result(
                        registration,
                        request,
                        result,
                        request_root=job_root,
                    )
                    completed = POD_ROOT / "completed" / request_id
                    if not completed.exists():
                        os.replace(job_root, completed)
                    append_event(POD_ROOT, "adapter_finished", request_id=request_id)
                    if (
                        adapter_process.consecutive_failures
                        >= MAX_CONSECUTIVE_ADAPTER_FAILURES
                    ):
                        append_event(
                            POD_ROOT,
                            "resident_epoch_failed",
                            adapter_start_count=adapter_process.start_count,
                            consecutive_adapter_failures=(
                                adapter_process.consecutive_failures
                            ),
                            schema=SCHEMA_ADAPTER_LIFECYCLE_EVENT,
                            worker_epoch=registration["worker_epoch"],
                        )
                        # Fail the epoch closed without tearing down a still-live
                        # resident model.  A human may inspect or recover the
                        # epoch, but this service must never turn a candidate or
                        # transport failure into an avoidable model unload.
                        while True:
                            verify_pod_registration(POD_REGISTRATION)
                            adapter_process._heartbeat(None, "epoch_failed")
                            time.sleep(args.poll_seconds)
                time.sleep(args.poll_seconds)
        finally:
            adapter_process.close()


def heartbeat_status() -> None:
    path = POD_ROOT / "heartbeat.json"
    if not path.exists():
        return
    sys.stdout.buffer.write(canonical_json_bytes(load_json(path)) + b"\n")


def result_status(args: argparse.Namespace) -> None:
    request_id = require_digest(args.request_id, "request id")
    path = POD_ROOT / "outgoing" / f"{request_id}.ready.json"
    if not path.exists():
        return
    sys.stdout.buffer.write(canonical_json_bytes(load_json(path)) + b"\n")


def ack_result(args: argparse.Namespace) -> None:
    request_id = require_digest(args.request_id, "request id")
    archive_sha = require_digest(args.archive_sha256, "archive digest")
    outgoing = POD_ROOT / "outgoing"
    archive = outgoing / f"{request_id}.{archive_sha}.tar"
    ready = outgoing / f"{request_id}.ready.json"
    if not archive.exists() or not ready.exists():
        fail("cannot acknowledge absent result")
    receipt = load_json(ready)
    if receipt.get("archive_sha256") != archive_sha or receipt.get("request_id") != request_id:
        fail("result acknowledgement identity differs")
    retained = POD_ROOT / "acked" / request_id
    retained.mkdir(parents=True, exist_ok=True, mode=0o700)
    for source in (archive, ready):
        destination = retained / source.name
        if destination.exists():
            if file_sha256(destination) != file_sha256(source):
                fail("acknowledged result collision")
            source.unlink()
        else:
            os.replace(source, destination)
    append_event(POD_ROOT, "result_acknowledged", request_id=request_id)
    print("acknowledged")


def _publication_archive(publication: object, destination: Path) -> None:
    """Copy one exact WorkerBundlePublication into a path-free worker archive."""

    try:
        from cacheon.chain.publication import (
            WorkerBundlePublication,
            reopen_worker_bundle,
        )
    except ImportError as exc:
        fail(f"worker publication type is unavailable: {exc}")
    if type(publication) is not WorkerBundlePublication:
        fail("screen transport requires an exact WorkerBundlePublication")
    root = publication.root
    if root.is_symlink() or not root.is_dir():
        fail("worker publication root is unavailable or symlinked")
    try:
        reopened = reopen_worker_bundle(
            root,
            publication.content_hash,
            expected_publication_digest=publication.publication_digest,
            expected_receipt_digest=publication.digest,
        )
    except Exception as exc:
        fail(f"worker publication cannot be reopened before transport: {exc}")
    if reopened.to_dict() != publication.to_dict():
        fail("reopened worker publication differs before transport")

    def stable_bytes(path: Path, *, size: int | None, sha256: str | None) -> bytes:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_size < 0
            or (size is not None and before.st_size != size)
        ):
            fail("worker publication file shape differs during transport")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
                or opened.st_mode != before.st_mode
                or opened.st_nlink != before.st_nlink
                or opened.st_size != before.st_size
            ):
                fail("worker publication file changed while opening")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(4 << 20, remaining))
                if not chunk:
                    fail("worker publication file was truncated")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                fail("worker publication file grew during transport")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        data = b"".join(chunks)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_mode != before.st_mode
            or after.st_nlink != before.st_nlink
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or (sha256 is not None and hashlib.sha256(data).hexdigest() != sha256)
        ):
            fail("worker publication bytes differ from retained inventory")
        return data

    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    try:
        manifest = canonical_json_bytes(
            {
                "publication": publication.to_dict(),
                "schema": "cacheon-remote-worker-publication-v1",
            }
        ) + b"\n"
        with tarfile.open(temporary, "w") as archive:
            archive.addfile(
                _tar_info("publication.json", len(manifest)), io.BytesIO(manifest)
            )
            native_manifest = stable_bytes(
                root / NATIVE_ARTIFACT_MANIFEST,
                size=None,
                sha256=None,
            )
            if len(native_manifest) > 16 << 20:
                fail("worker publication native manifest exceeds its hard bound")
            archive.addfile(
                _tar_info(
                    f"bundle/{NATIVE_ARTIFACT_MANIFEST}", len(native_manifest)
                ),
                io.BytesIO(native_manifest),
            )
            for row in publication.files:
                relative = Path(row.path)
                if relative.is_absolute() or ".." in relative.parts:
                    fail("worker publication inventory contains an unsafe path")
                path = root.joinpath(*relative.parts)
                data = stable_bytes(path, size=row.size, sha256=row.sha256)
                archive.addfile(
                    _tar_info(f"bundle/{relative.as_posix()}", len(data)),
                    io.BytesIO(data),
                )
        if Path(temporary).stat().st_size > MAX_ARTIFACT_BYTES:
            fail("worker publication archive exceeds transport limit")
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


class DurableSpoolAuthenticatedWorkerTransport:
    """AuthenticatedWorkerTransport backed by the standing VM/pod spool.

    ``RemoteEvaluationDispatcher`` owns the durable lease and its finalized-
    block heartbeat while this call waits.  The separate CPU tmux transfers
    the sealed request and response.  Only screen work is enabled; production
    qualification remains fail-closed until evidence mirroring is implemented.
    """

    def __init__(
        self,
        *,
        registration_path: str | os.PathLike[str],
        spool_root: str | os.PathLike[str],
        credential_path: str | os.PathLike[str],
        response_timeout_seconds: int = MAX_JOB_SECONDS,
        poll_seconds: int = 2,
    ):
        self.registration_path = Path(registration_path)
        self.registration = verify_registration(load_json(self.registration_path))
        self.spool_root = Path(spool_root)
        self.credential_path = Path(credential_path)
        if self.credential_path != Path(self.registration["credential_path"]):
            fail("transport credential path differs from registration")
        self.credential = _typed_credential(self.registration, self.credential_path)
        self.identity = _typed_transport_identity(self.registration)
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
        try:
            from cacheon.chain.evaluation_leases import EvaluationLease
        except ImportError as exc:
            fail(f"evaluation lease type is unavailable: {exc}")
        if type(lease) is not EvaluationLease:
            fail("screen transport lease is not exactly typed")
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

    def run_screen(self, request, *, job):
        try:
            from cacheon.chain.evaluation_coordinator import ClaimedScreenEvaluation
            from cacheon.chain.remote_evaluation_dispatcher import (
                AuthenticatedRemoteEvaluationResponse,
                RemoteEvaluationDispatcherError,
                RemoteEvaluationRequest,
                verify_remote_request,
            )
        except ImportError as exc:
            fail(f"authenticated remote dispatcher types are unavailable: {exc}")
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
                _publication_archive(job.publication, publication_path)
                request_id, job_dir = enqueue_request(
                    self.registration,
                    self._lease_dict(job.lease),
                    (
                        ("screen_payload", wire_path),
                        ("candidate_publication", publication_path),
                    ),
                    self.spool_root / "outbox",
                    deadline_seconds=self.response_timeout_seconds,
                )
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
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
                    )
                    if result["state"] != "completed":
                        raise RemoteEvaluationDispatcherError(
                            f"remote screen infrastructure: {result['failure_code']}"
                        )
                    response_path = _artifact_for_role(
                        result, result_root, "adapter_result"
                    )
                    response = AuthenticatedRemoteEvaluationResponse.from_dict(
                        load_json(response_path, maximum=64 << 20)
                    )
                    return response
                self._require_dispatcher_liveness()
                time.sleep(self.poll_seconds)
            raise RemoteEvaluationDispatcherError(
                "remote screen response exceeded the transport deadline"
            )
        except RemoteWorkerError as exc:
            raise RemoteEvaluationDispatcherError(str(exc)) from exc

    def run_qualification(self, request, *, job, work, prepared):
        try:
            from cacheon.chain.remote_evaluation_dispatcher import (
                RemoteEvaluationDispatcherError,
            )
        except ImportError as exc:
            fail(f"authenticated remote dispatcher type is unavailable: {exc}")
        raise RemoteEvaluationDispatcherError(
            "remote qualification is gated until evidence mirroring is closed"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    install_source = subparsers.add_parser("install-source")
    install_source.add_argument("--archive", required=True)
    install_source.add_argument("--archive-sha256", required=True)
    install_source.add_argument("--source-revision", required=True)
    install_source.set_defaults(function=install_source_archive)

    commission = subparsers.add_parser("commission-current-pod")
    commission.add_argument("--source-root", required=True)
    commission.add_argument("--source-revision", required=True)
    commission.add_argument("--runtime-root", required=True)
    commission.add_argument("--model-root", required=True)
    commission.add_argument("--model-receipt", required=True)
    commission.add_argument("--worker-image", required=True)
    commission.add_argument("--python-executable", required=True)
    commission.add_argument("--lane-devices", required=True)
    commission.add_argument("--pod-endpoint", required=True)
    commission.add_argument("--output", default=str(POD_READY_RECEIPT))
    commission.set_defaults(function=commission_current_pod)

    make = subparsers.add_parser("make-registration")
    make.add_argument("--ready-receipt", required=True)
    make.add_argument("--worker-readiness", required=True)
    make.add_argument("--known-hosts", required=True)
    make.add_argument("--pod-host", required=True)
    make.add_argument("--pod-port", required=True, type=int)
    make.add_argument("--service-identity", required=True)
    make.add_argument("--remote-service", required=True)
    make.add_argument("--adapter", required=True)
    make.add_argument("--credential", required=True)
    make.add_argument("--credential-id", required=True)
    make.add_argument("--python-executable", required=True)
    make.add_argument("--lane-devices", required=True)
    make.add_argument("--max-response-bytes", type=int, default=16 << 20)
    make.add_argument("--bind-ready-receipt", action="store_true")
    make.add_argument("--output", required=True)
    make.set_defaults(function=make_registration)

    seal = subparsers.add_parser("seal-request")
    seal.add_argument("--registration", required=True)
    seal.add_argument("--lease", required=True)
    seal.add_argument("--artifact", action="append", required=True)
    seal.add_argument("--deadline-seconds", type=int, default=MAX_JOB_SECONDS)
    seal.add_argument("--outbox", required=True)
    seal.set_defaults(function=seal_request)

    cpu = subparsers.add_parser("cpu-serve")
    cpu.add_argument("--registration", required=True)
    cpu.add_argument("--current-registration", required=True)
    cpu.add_argument("--spool-root", required=True)
    cpu.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    cpu.add_argument("--max-heartbeat-age", type=int, default=DEFAULT_MAX_HEARTBEAT_AGE)
    cpu.set_defaults(function=cpu_serve)

    pod = subparsers.add_parser("pod-serve")
    pod.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    pod.add_argument("--heartbeat-seconds", type=int, default=DEFAULT_HEARTBEAT_SECONDS)
    pod.set_defaults(function=pod_serve)

    accept = subparsers.add_parser("accept-request")
    accept.add_argument("--request-id", required=True)
    accept.add_argument("--archive-sha256", required=True)
    accept.add_argument("--archive-size", required=True, type=int)
    accept.set_defaults(function=accept_request)

    status = subparsers.add_parser("result-status")
    status.add_argument("--request-id", required=True)
    status.set_defaults(function=result_status)

    ack = subparsers.add_parser("ack-result")
    ack.add_argument("--request-id", required=True)
    ack.add_argument("--archive-sha256", required=True)
    ack.set_defaults(function=ack_result)

    heartbeat = subparsers.add_parser("heartbeat-status")
    heartbeat.set_defaults(function=lambda _args: heartbeat_status())
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "poll_seconds"):
        require_int(args.poll_seconds, "poll seconds", minimum=1, maximum=60)
    if hasattr(args, "heartbeat_seconds"):
        require_int(args.heartbeat_seconds, "heartbeat seconds", minimum=2, maximum=60)
    if hasattr(args, "max_heartbeat_age"):
        require_int(args.max_heartbeat_age, "maximum heartbeat age", minimum=10, maximum=300)
    try:
        args.function(args)
    except RemoteWorkerError as exc:
        print(f"REMOTE-WORKER-ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
