"""Command-line composition for the tracked remote worker transport.

One argument surface over the registration, spool, SSH transport, and pod
service modules.  Every filesystem coordinate is an explicit required flag:
there is no environment fallback and no built-in site path, so a deployment
wrapper supplies its exact installed locations.  Provisioning (source
installation and pod commissioning) is deliberately absent; it remains
deployment tooling outside the tracked package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cacheon.chain.remote_worker_pod_service import (
    accept_request,
    ack_result,
    heartbeat_status,
    pod_serve,
    result_status,
)
from cacheon.chain.remote_worker_registration import (
    PodPaths,
    make_registration,
    registration_credential,
    registration_transport_identity,
    verify_registration,
)
from cacheon.chain.remote_worker_spool import (
    ALLOWED_ARTIFACT_ROLES,
    MAX_ARTIFACT_BYTES,
    MAX_JOB_SECONDS,
    ROLE,
    RemoteWorkerError,
    enqueue_request,
    fail,
    load_json,
    require_int,
    require_text,
    verify_lease,
)
from cacheon.chain.ssh_worker_transport import (
    DEFAULT_HEARTBEAT_SECONDS,
    DEFAULT_MAX_HEARTBEAT_AGE,
    DEFAULT_POLL_SECONDS,
    RemotePodSite,
    cpu_serve,
)


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


def _add_pod_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pod-root", required=True)
    parser.add_argument("--pod-ready-receipt", required=True)
    parser.add_argument("--pod-registration", required=True)
    parser.add_argument("--pod-service", required=True)
    parser.add_argument("--pod-adapter", required=True)
    parser.add_argument("--pod-credential", required=True)


def _pod_paths(args: argparse.Namespace) -> PodPaths:
    return PodPaths(
        root=Path(args.pod_root),
        ready_receipt=Path(args.pod_ready_receipt),
        registration=Path(args.pod_registration),
        service=Path(args.pod_service),
        adapter=Path(args.pod_adapter),
        credential=Path(args.pod_credential),
    )


def _cmd_make_registration(args: argparse.Namespace) -> None:
    make_registration(
        ready_receipt=Path(args.ready_receipt),
        worker_readiness=Path(args.worker_readiness),
        known_hosts=Path(args.known_hosts),
        pod_host=args.pod_host,
        pod_port=args.pod_port,
        service_identity=args.service_identity,
        remote_service=Path(args.remote_service),
        adapter=Path(args.adapter),
        credential=Path(args.credential),
        credential_id=args.credential_id,
        output=Path(args.output),
        python_executable=args.python_executable,
        lane_devices=args.lane_devices,
        max_response_bytes=args.max_response_bytes,
        bind_ready_receipt=args.bind_ready_receipt,
        transport_id=args.transport_id,
    )


def _cmd_seal_request(args: argparse.Namespace) -> None:
    registration = verify_registration(load_json(Path(args.registration)))
    identity = registration_transport_identity(registration)
    credential = registration_credential(
        registration, Path(registration["credential_path"])
    )
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
        identity=identity,
        credential=credential,
    )
    print(request_id)


def _cmd_cpu_serve(args: argparse.Namespace) -> None:
    cpu_serve(
        registration_path=Path(args.registration),
        current_registration_path=Path(args.current_registration),
        spool_root=Path(args.spool_root),
        site=RemotePodSite(args.pod_root, args.pod_service_path),
        poll_seconds=args.poll_seconds,
        max_heartbeat_age=args.max_heartbeat_age,
    )


def _cmd_pod_serve(args: argparse.Namespace) -> None:
    pod_serve(
        _pod_paths(args),
        poll_seconds=args.poll_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        adapter_arguments=tuple(args.adapter_argument or ()),
    )


def _cmd_accept_request(args: argparse.Namespace) -> None:
    accept_request(
        _pod_paths(args),
        request_id=args.request_id,
        archive_sha256=args.archive_sha256,
        archive_size=args.archive_size,
    )


def _cmd_result_status(args: argparse.Namespace) -> None:
    result_status(_pod_paths(args), args.request_id)


def _cmd_ack_result(args: argparse.Namespace) -> None:
    ack_result(_pod_paths(args), args.request_id, args.archive_sha256)


def _cmd_heartbeat_status(args: argparse.Namespace) -> None:
    heartbeat_status(_pod_paths(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

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
    make.add_argument("--python-executable")
    make.add_argument("--lane-devices")
    make.add_argument("--max-response-bytes", type=int, default=16 << 20)
    make.add_argument("--bind-ready-receipt", action="store_true")
    make.add_argument("--transport-id")
    make.add_argument("--output", required=True)
    make.set_defaults(function=_cmd_make_registration)

    seal = subparsers.add_parser("seal-request")
    seal.add_argument("--registration", required=True)
    seal.add_argument("--lease", required=True)
    seal.add_argument("--artifact", action="append", required=True)
    seal.add_argument("--deadline-seconds", type=int, default=MAX_JOB_SECONDS)
    seal.add_argument("--outbox", required=True)
    seal.set_defaults(function=_cmd_seal_request)

    cpu = subparsers.add_parser("cpu-serve")
    cpu.add_argument("--registration", required=True)
    cpu.add_argument("--current-registration", required=True)
    cpu.add_argument("--spool-root", required=True)
    cpu.add_argument("--pod-root", required=True)
    cpu.add_argument("--pod-service-path", required=True)
    cpu.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    cpu.add_argument(
        "--max-heartbeat-age", type=int, default=DEFAULT_MAX_HEARTBEAT_AGE
    )
    cpu.set_defaults(function=_cmd_cpu_serve)

    pod = subparsers.add_parser("pod-serve")
    _add_pod_paths(pod)
    pod.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    pod.add_argument(
        "--heartbeat-seconds", type=int, default=DEFAULT_HEARTBEAT_SECONDS
    )
    pod.add_argument("--adapter-argument", action="append")
    pod.set_defaults(function=_cmd_pod_serve)

    accept = subparsers.add_parser("accept-request")
    _add_pod_paths(accept)
    accept.add_argument("--request-id", required=True)
    accept.add_argument("--archive-sha256", required=True)
    accept.add_argument("--archive-size", required=True, type=int)
    accept.set_defaults(function=_cmd_accept_request)

    status = subparsers.add_parser("result-status")
    _add_pod_paths(status)
    status.add_argument("--request-id", required=True)
    status.set_defaults(function=_cmd_result_status)

    ack = subparsers.add_parser("ack-result")
    _add_pod_paths(ack)
    ack.add_argument("--request-id", required=True)
    ack.add_argument("--archive-sha256", required=True)
    ack.set_defaults(function=_cmd_ack_result)

    heartbeat = subparsers.add_parser("heartbeat-status")
    _add_pod_paths(heartbeat)
    heartbeat.set_defaults(function=_cmd_heartbeat_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "poll_seconds"):
            require_int(args.poll_seconds, "poll seconds", minimum=1, maximum=60)
        if hasattr(args, "heartbeat_seconds"):
            require_int(
                args.heartbeat_seconds, "heartbeat seconds", minimum=2, maximum=60
            )
        if hasattr(args, "max_heartbeat_age"):
            require_int(
                args.max_heartbeat_age, "maximum heartbeat age", minimum=10, maximum=300
            )
        args.function(args)
    except RemoteWorkerError as exc:
        print(f"REMOTE-WORKER-ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
