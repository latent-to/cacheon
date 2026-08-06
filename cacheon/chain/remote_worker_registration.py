"""Registration authority for one explicitly registered remote worker epoch.

A registration binds, under one semantic digest: the pod endpoint (host, port,
user, pinned ``known_hosts`` bytes), the commissioned READY receipt, the exact
``WorkerReadiness``, the physical GPU lane, the pod Python interpreter, the
frozen remote service and adapter bytes, the shared transport credential, and
the typed :class:`~cacheon.chain.remote_evaluation_dispatcher.RemoteWorkerTransportIdentity`.

This module verifies and constructs that contract.  It performs no evaluation
and opens no network connection.  Typed authority (credential, transport
identity, readiness, lease) is reopened through the tracked product types so
this module cannot drift from the authenticated protocol.

The registration schema is wire-compatible with the previously deployed
private service; field sets and digest domains are identity inputs and must
not change without a protocol review.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.chain.remote_evaluation_dispatcher import (
    REMOTE_EVALUATION_PROTOCOL_DIGEST,
    RemoteWorkerCredential,
    RemoteWorkerTransportIdentity,
)
from cacheon.chain.remote_worker_spool import (
    DOMAIN_REGISTRATION,
    EPOCH,
    HOST,
    SCHEMA_REGISTRATION,
    WIRE_IDENTIFIER,
    atomic_json,
    fail,
    file_sha256,
    load_json,
    require_closed,
    require_digest,
    require_int,
    require_text,
    spool_canonical_json,
    spool_digest,
)


READINESS_FIELDS = frozenset(WorkerReadiness.__dataclass_fields__)
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

@dataclass(frozen=True)
class PodPaths:
    """Closed pod-side filesystem contract for one installed worker epoch."""

    root: Path
    ready_receipt: Path
    registration: Path
    service: Path
    adapter: Path
    credential: Path

    def __post_init__(self) -> None:
        for name in (
            "root",
            "ready_receipt",
            "registration",
            "service",
            "adapter",
            "credential",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                fail(f"pod {name} path must be an absolute Path")


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
        if (
            not isinstance(value["worker_epoch"], str)
            or EPOCH.fullmatch(value["worker_epoch"]) is None
        ):
            fail("commissioned worker epoch is malformed")
        supplied = require_digest(value["receipt_digest"], "commission receipt digest")
        unsigned = dict(value)
        del unsigned["receipt_digest"]
        if spool_digest("cacheon.current-pod-commission.v1", unsigned) != supplied:
            fail("current-pod commission receipt semantic digest mismatch")
        gpu = value["gpu"]
        if (
            type(gpu) is not dict
            or set(gpu) != {"count", "inventory", "inventory_sha256", "topology_sha256"}
            or gpu["count"] != 8
            or type(gpu["inventory"]) is not list
            or len(gpu["inventory"]) != 8
            or any(
                type(item) is not dict or "B300" not in str(item.get("name", ""))
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
            ("python", {"executable_sha256", "path", "resolved_path", "version"}),
            ("lane", {"devices", "lane_digest", "tensor_parallel_size"}),
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
    if (
        not isinstance(value["worker_epoch"], str)
        or EPOCH.fullmatch(value["worker_epoch"]) is None
    ):
        fail("READY worker epoch is malformed")
    supplied = require_digest(value["receipt_digest"], "READY receipt digest")
    unsigned = dict(value)
    del unsigned["receipt_digest"]
    observed = hashlib.sha256(
        b"cacheon.lium-worker-ready.v1\0" + spool_canonical_json(unsigned)
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
        or any(
            type(item) is not dict or "B300" not in str(item.get("name", ""))
            for item in inventory
        )
    ):
        fail("READY receipt GPU inventory is not 8xB300")
    return value


def verify_readiness(row: object) -> tuple[dict[str, Any], str]:
    """Verify one WorkerReadiness projection through the exact tracked type."""

    value = require_closed(row, READINESS_FIELDS, "WorkerReadiness")
    try:
        readiness = WorkerReadiness(**value)
    except RuntimeError as exc:
        fail(f"WorkerReadiness is malformed: {exc}")
    return value, readiness.digest


def registration_credential(
    registration: Mapping[str, Any], path: Path
) -> RemoteWorkerCredential:
    """Load and bind the exact shared credential named by a registration."""

    try:
        secret = path.read_bytes()
    except OSError as exc:
        fail(f"remote worker credential is unreadable: {exc}")
    try:
        credential = RemoteWorkerCredential(registration["credential_id"], secret)
    except RuntimeError as exc:
        fail(f"typed remote worker credential is unavailable: {exc}")
    if credential.digest != registration["credential_digest"]:
        fail("remote worker credential digest differs from registration")
    return credential


def registration_transport_identity(
    registration: Mapping[str, Any],
) -> RemoteWorkerTransportIdentity:
    try:
        identity = RemoteWorkerTransportIdentity(**registration["transport_identity"])
    except (TypeError, RuntimeError) as exc:
        fail(f"remote worker transport identity is invalid: {exc}")
    if identity.digest != registration["transport_identity_digest"]:
        fail("remote worker transport identity digest differs")
    return identity


def verify_registration(row: object) -> dict[str, Any]:
    value = require_closed(row, REGISTRATION_FIELDS, "worker registration")
    if value["schema"] != SCHEMA_REGISTRATION:
        fail("worker registration schema is not supported")
    unsigned = dict(value)
    supplied = require_digest(unsigned.pop("registration_digest"), "registration digest")
    if spool_digest(DOMAIN_REGISTRATION, unsigned) != supplied:
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
    if (
        not isinstance(value["worker_epoch"], str)
        or EPOCH.fullmatch(value["worker_epoch"]) is None
    ):
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
        or any(
            type(device) is not int or device < 0 or device >= 8
            for device in lane_devices
        )
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
    identity = registration_transport_identity(value)
    if identity.protocol_digest != REMOTE_EVALUATION_PROTOCOL_DIGEST:
        fail("registered remote evaluation protocol is unsupported")
    if (
        identity.credential_digest != value["credential_digest"]
        or identity.service_digest != readiness["service_digest"]
        or identity.worker_readiness_digest != digest
    ):
        fail("transport identity differs from credential/service/readiness")
    known_hosts = Path(value["known_hosts_path"])
    if not known_hosts.is_absolute():
        fail("known_hosts path must be absolute")
    return value


def make_registration(
    *,
    ready_receipt: Path,
    worker_readiness: Path,
    known_hosts: Path,
    pod_host: str,
    pod_port: int,
    service_identity: str,
    remote_service: Path,
    adapter: Path,
    credential: Path,
    credential_id: str,
    output: Path,
    python_executable: str | None = None,
    lane_devices: str | None = None,
    max_response_bytes: int = 16 << 20,
    bind_ready_receipt: bool = False,
    transport_id: str | None = None,
) -> dict[str, Any]:
    """Construct, self-verify, and atomically write one registration."""

    ready = verify_ready_receipt(load_json(ready_receipt))
    readiness, readiness_digest = verify_readiness(load_json(worker_readiness))
    if bind_ready_receipt and readiness["ready_receipt_digest"] == "0" * 64:
        readiness = {**readiness, "ready_receipt_digest": ready["receipt_digest"]}
        readiness, readiness_digest = verify_readiness(readiness)
    if ready["receipt_digest"] != readiness["ready_receipt_digest"]:
        fail("READY receipt and WorkerReadiness receipt digest differ")
    if ready["schema"] == "cacheon-current-pod-commission-v1":
        registered_lane = list(ready["lane"]["devices"])
        lane_digest = ready["lane"]["lane_digest"]
        registered_python = ready["python"]["path"]
        python_executable_sha256 = ready["python"]["executable_sha256"]
    else:
        try:
            registered_lane = [int(value) for value in (lane_devices or "").split(",")]
        except ValueError:
            fail("registered lane device list is malformed")
        if (
            not registered_lane
            or registered_lane != sorted(set(registered_lane))
            or any(
                device < 0 or device >= ready["gpu"]["count"]
                for device in registered_lane
            )
        ):
            fail("registered lane must be sorted unique physical GPU indexes")
        lane_digest = spool_digest(
            "cacheon.registered-worker-lane.v1",
            {
                "devices": registered_lane,
                "ready_receipt_digest": ready["receipt_digest"],
            },
        )
        registered_python = require_text(
            python_executable, "registered Python executable", maximum=4096
        )
        python_path = Path(registered_python)
        if not python_path.is_absolute() or not python_path.exists():
            fail("registered Python executable is unavailable")
        python_executable_sha256 = file_sha256(python_path.resolve(strict=True))
    if (
        len(registered_lane) != readiness["gpu_count"]
        or readiness["tensor_parallel_size"] != len(registered_lane)
        or len(registered_lane) > ready["gpu"]["count"]
    ):
        fail("registered lane differs from WorkerReadiness GPU/TP identity")
    if (
        not known_hosts.is_absolute()
        or known_hosts.is_symlink()
        or not known_hosts.is_file()
    ):
        fail("known_hosts must be an absolute regular file")
    if stat.S_IMODE(known_hosts.stat().st_mode) & 0o077:
        fail("known_hosts must not be group/world accessible")
    for path, name in (
        (remote_service, "remote service"),
        (adapter, "adapter"),
        (credential, "credential"),
    ):
        if path.is_symlink() or not path.is_file():
            fail(f"{name} must be a regular file")
    credential_secret = credential.read_bytes()
    exact_credential_id = require_text(
        credential_id, "credential id", maximum=128, pattern=WIRE_IDENTIFIER
    )
    try:
        typed_credential = RemoteWorkerCredential(exact_credential_id, credential_secret)
    except RuntimeError as exc:
        fail(f"remote worker credential is invalid: {exc}")
    endpoint_digest = spool_digest(
        "cacheon.chain.remote-worker-endpoint.v1",
        {
            "known_hosts_sha256": file_sha256(known_hosts),
            "lane_digest": lane_digest,
            "pod_host": pod_host,
            "pod_port": pod_port,
            "pod_user": "root",
            "python_executable_sha256": python_executable_sha256,
            "ready_receipt_digest": ready["receipt_digest"],
            "worker_epoch": ready["worker_epoch"],
        },
    )
    exact_transport_id = (
        transport_id
        if transport_id is not None
        else f"lium-b300-{ready['worker_epoch'][:12]}"
    )
    try:
        identity = RemoteWorkerTransportIdentity(
            transport_id=exact_transport_id,
            endpoint_identity_digest=endpoint_digest,
            protocol_digest=REMOTE_EVALUATION_PROTOCOL_DIGEST,
            credential_digest=typed_credential.digest,
            service_digest=readiness["service_digest"],
            worker_readiness_digest=readiness_digest,
            max_response_bytes=require_int(
                max_response_bytes,
                "maximum response bytes",
                minimum=1,
                maximum=64 << 20,
            ),
        )
    except RuntimeError as exc:
        fail(f"remote worker transport identity is invalid: {exc}")
    provider = ready.get("provider")
    if type(provider) is not dict:
        fail("READY receipt provider identity is malformed")
    if provider.get("pod_endpoint") not in {"unknown", pod_host, f"{pod_host}:{pod_port}"}:
        fail("READY receipt was created for a different pod endpoint")
    value: dict[str, Any] = {
        "adapter_sha256": file_sha256(adapter),
        "created_at_unix": int(time.time()),
        "credential_digest": typed_credential.digest,
        "credential_file_sha256": file_sha256(credential),
        "credential_id": exact_credential_id,
        "credential_path": str(credential),
        "known_hosts_path": str(known_hosts),
        "known_hosts_sha256": file_sha256(known_hosts),
        "lane_devices": registered_lane,
        "lane_digest": lane_digest,
        "pod_host": require_text(pod_host, "pod host", pattern=HOST),
        "pod_port": require_int(pod_port, "pod port", minimum=1, maximum=65535),
        "pod_user": "root",
        "python_executable": registered_python,
        "python_executable_sha256": python_executable_sha256,
        "ready_receipt_digest": ready["receipt_digest"],
        "ready_receipt_file_sha256": file_sha256(ready_receipt),
        "remote_service_sha256": file_sha256(remote_service),
        "schema": SCHEMA_REGISTRATION,
        "service_identity": require_text(service_identity, "service identity"),
        "transport_identity": identity.to_dict(),
        "transport_identity_digest": identity.digest,
        "worker_epoch": ready["worker_epoch"],
        "worker_readiness": readiness,
        "worker_readiness_digest": readiness_digest,
    }
    value["registration_digest"] = spool_digest(DOMAIN_REGISTRATION, value)
    verify_registration(value)
    atomic_json(output, value)
    return value


def registration_is_current(
    registration: Mapping[str, Any], current_path: Path
) -> bool:
    if current_path.is_symlink() or not current_path.is_file():
        return False
    try:
        current = verify_registration(load_json(current_path))
    except RuntimeError:
        return False
    return current["registration_digest"] == registration["registration_digest"]


def verify_fixed_adapter(adapter_path: Path, registration: Mapping[str, Any]) -> None:
    if adapter_path.is_symlink() or not adapter_path.is_file():
        fail(f"fixed deployment adapter is missing: {adapter_path}")
    metadata = adapter_path.stat()
    if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
        fail("fixed deployment adapter is not root-owned/read-only")
    if file_sha256(adapter_path) != registration["adapter_sha256"]:
        fail("fixed deployment adapter digest differs from registration")


def verify_pod_registration(paths: PodPaths) -> dict[str, Any]:
    """Verify the installed pod authority against its own frozen carriers."""

    if type(paths) is not PodPaths:
        fail("pod paths are not exactly typed")
    registration = verify_registration(load_json(paths.registration))
    ready = verify_ready_receipt(load_json(paths.ready_receipt))
    if file_sha256(paths.ready_receipt) != registration["ready_receipt_file_sha256"]:
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
    if (
        file_sha256(python_path.resolve(strict=True))
        != registration["python_executable_sha256"]
    ):
        fail("registered pod Python executable bytes changed")
    if file_sha256(paths.service) != registration["remote_service_sha256"]:
        fail("pod remote worker service differs from registration")
    if paths.credential.is_symlink() or not paths.credential.is_file():
        fail("pod remote worker credential is missing")
    credential_stat = paths.credential.stat()
    if credential_stat.st_uid != 0 or stat.S_IMODE(credential_stat.st_mode) & 0o077:
        fail("pod remote worker credential is not root-only")
    if file_sha256(paths.credential) != registration["credential_file_sha256"]:
        fail("pod remote worker credential bytes differ from registration")
    registration_credential(registration, paths.credential)
    verify_fixed_adapter(paths.adapter, registration)
    return registration
