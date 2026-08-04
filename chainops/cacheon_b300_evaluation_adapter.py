#!/usr/bin/env python3
"""Fixed authenticated screen codec for the Cacheon B300 worker service.

This executable has exactly one operation.  The supervising service invokes it
as::

    cacheon-b300-evaluation-adapter \
      --request-dir /data/cacheon-b300/remote-worker/processing/REQUEST_ID \
      --result-dir /data/cacheon-b300/remote-worker/results/.REQUEST_ID.PID

It never accepts a module, command, argv, environment, executable, source, or
output selector.  It authenticates the path-free ``RemoteEvaluationRequest``,
reconstructs the immutable publication under a content-addressed pod root,
calls the one built-in B300 screen deployment constructor, and emits one
canonical authenticated ``response.json``.  The CPU independently reopens the
typed receipt before committing the durable lease.

Qualification is intentionally absent until its evidence-mirroring authority
is closed in the CPU dispatcher.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import stat
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


SERVICE_PATH = Path(
    "/data/cacheon-b300/worker-bootstrap/bin/remote_worker_service.py"
)
REGISTRATION_PATH = Path("/data/cacheon-b300/remote-worker/registration.json")
READY_RECEIPT_PATH = Path(
    "/data/cacheon-b300/worker-bootstrap/ready-receipt.json"
)
CREDENTIAL_PATH = Path("/data/cacheon-b300/remote-worker/credential.secret")
PUBLICATION_ROOT = Path("/data/cacheon-b300/remote-worker/publications")
PROCESSING_ROOT = Path("/data/cacheon-b300/remote-worker/processing")
RESULTS_ROOT = Path("/data/cacheon-b300/remote-worker/results")
MAX_PUBLICATION_BYTES = 4 * 1024 * 1024 * 1024
NATIVE_ARTIFACT_MANIFEST = ".cacheon-native-artifact.json"


class AdapterError(RuntimeError):
    """The fixed adapter could not authenticate or execute screen work."""


def _load_service():
    if SERVICE_PATH.is_symlink() or not SERVICE_PATH.is_file():
        raise AdapterError("fixed remote worker service is unavailable")
    specification = importlib.util.spec_from_file_location(
        "cacheon_fixed_remote_worker_service", SERVICE_PATH
    )
    if specification is None or specification.loader is None:
        raise AdapterError("fixed remote worker service cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _closed_path(raw: str, root: Path, label: str, *, temporary: bool) -> Path:
    path = Path(raw)
    try:
        resolved_root = root.resolve(strict=True)
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise AdapterError(f"{label} parent is unavailable: {exc}") from None
    if (
        not path.is_absolute()
        or resolved_parent != resolved_root
        or (
            temporary
            and not (
                path.name.startswith(".")
                and len(path.name.split(".")) == 3
                and len(path.name.split(".")[1]) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in path.name.split(".")[1]
                )
                and path.name.split(".")[2].isdigit()
            )
        )
        or (
            not temporary
            and (
                len(path.name) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in path.name
                )
            )
        )
    ):
        raise AdapterError(f"{label} is outside its fixed content-addressed root")
    return path


def _source_from_ready(ready: dict[str, object]) -> Path:
    source = ready.get("source")
    if type(source) is not dict:
        raise AdapterError("READY receipt has no source identity")
    raw = source.get("path")
    if not isinstance(raw, str):
        raise AdapterError("READY source path is malformed")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise AdapterError("READY source root is unavailable")
    return path


def _safe_publication(
    archive_path: Path,
    expected_wire: object,
):
    from cacheon.chain.publication import reopen_worker_bundle
    from cacheon.eval.native_artifact import NativeArtifactFile

    if archive_path.is_symlink() or not archive_path.is_file():
        raise AdapterError("candidate publication archive is unavailable")
    if archive_path.stat().st_size > MAX_PUBLICATION_BYTES:
        raise AdapterError("candidate publication archive exceeds its bound")
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        by_name = {member.name: member for member in members}
        if len(by_name) != len(members) or "publication.json" not in by_name:
            raise AdapterError("candidate publication archive inventory is ambiguous")
        manifest_member = by_name["publication.json"]
        if not manifest_member.isfile() or manifest_member.size > 4 * 1024 * 1024:
            raise AdapterError("candidate publication manifest carrier is invalid")
        manifest_file = archive.extractfile(manifest_member)
        if manifest_file is None:
            raise AdapterError("candidate publication manifest is unreadable")
        manifest_raw = manifest_file.read()
        try:
            manifest = json.loads(manifest_raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AdapterError(f"candidate publication manifest is invalid: {exc}") from None
        if (
            type(manifest) is not dict
            or set(manifest) != {"publication", "schema"}
            or manifest["schema"] != "cacheon-remote-worker-publication-v1"
            or manifest["publication"] != expected_wire
            or manifest_raw != _canonical_json(manifest) + b"\n"
        ):
            raise AdapterError("candidate publication manifest changed wire authority")
        publication = manifest["publication"]
        if type(publication) is not dict:
            raise AdapterError("candidate publication identity is malformed")
        required = {
            "address_digest",
            "content_hash",
            "directories",
            "files",
            "publication_digest",
            "schema",
        }
        if set(publication) != required or publication["schema"] != "cacheon.worker-bundle-publication.v1":
            raise AdapterError("candidate publication fields are not closed")
        raw_files = publication["files"]
        raw_directories = publication["directories"]
        if type(raw_files) is not list or type(raw_directories) is not list:
            raise AdapterError("candidate publication inventory is malformed")
        try:
            files = tuple(NativeArtifactFile(**row) for row in raw_files)
        except (TypeError, ValueError) as exc:
            raise AdapterError(
                f"candidate publication file inventory is malformed: {exc}"
            ) from None
        if any(type(logical) is not str for logical in raw_directories):
            raise AdapterError("candidate publication directory inventory is malformed")
        directories = tuple(raw_directories)
        expected_names = {
            "publication.json",
            f"bundle/{NATIVE_ARTIFACT_MANIFEST}",
        } | {
            f"bundle/{row.path}" for row in files
        }
        if set(by_name) != expected_names:
            raise AdapterError("candidate archive bytes differ from publication inventory")
        for member in members:
            logical = PurePosixPath(member.name)
            if (
                not member.isfile()
                or logical.is_absolute()
                or any(part in {"", ".", ".."} for part in logical.parts)
                or member.issym()
                or member.islnk()
            ):
                raise AdapterError("candidate archive contains an unsafe carrier")
        address = publication["address_digest"]
        if not isinstance(address, str) or len(address) != 64:
            raise AdapterError("candidate publication address is malformed")
        parent = PUBLICATION_ROOT / address[:2]
        final = parent / address
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if final.exists():
            candidate = reopen_worker_bundle(
                final,
                publication["content_hash"],
                expected_publication_digest=publication["publication_digest"],
            )
            if candidate.to_dict() != publication:
                raise AdapterError(
                    "reopened candidate publication differs from wire authority"
                )
            return candidate
        temporary = Path(tempfile.mkdtemp(prefix=f".{address}.", dir=parent))
        try:
            for logical in sorted(
                directories,
                key=lambda value: (len(PurePosixPath(value).parts), value),
            ):
                relative = PurePosixPath(logical)
                if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                    raise AdapterError("candidate publication directory is unsafe")
                temporary.joinpath(*relative.parts).mkdir(
                    parents=True, exist_ok=True, mode=0o700
                )
            carriers = (
                (NATIVE_ARTIFACT_MANIFEST, None),
                *((row.path, row) for row in files),
            )
            for logical_name, row in carriers:
                member = by_name[f"bundle/{logical_name}"]
                source = archive.extractfile(member)
                if source is None:
                    raise AdapterError("candidate archive file is unreadable")
                target = temporary.joinpath(*PurePosixPath(logical_name).parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with target.open("xb") as output:
                    remaining = member.size if row is None else row.size
                    while remaining:
                        chunk = source.read(min(4 << 20, remaining))
                        if not chunk:
                            raise AdapterError("candidate archive file was truncated")
                        output.write(chunk)
                        remaining -= len(chunk)
                    if source.read(1):
                        raise AdapterError("candidate archive file exceeded its inventory")
                    output.flush()
                    os.fsync(output.fileno())
                os.chmod(target, 0o444)
            for row in files:
                path = temporary.joinpath(*PurePosixPath(row.path).parts)
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != row.size
                    or _file_sha256(path) != row.sha256
                ):
                    raise AdapterError(
                        "reconstructed candidate publication bytes differ"
                    )
            for logical in sorted(
                directories,
                key=lambda value: (-len(PurePosixPath(value).parts), value),
            ):
                os.chmod(
                    temporary.joinpath(*PurePosixPath(logical).parts), 0o555
                )
            os.chmod(temporary, 0o555)
            os.replace(temporary, final)
        finally:
            if temporary.exists():
                for current, directory_names, file_names in os.walk(
                    temporary, topdown=False
                ):
                    os.chmod(current, 0o700)
                    for name in file_names:
                        os.chmod(Path(current) / name, 0o600)
                shutil.rmtree(temporary)
    candidate = reopen_worker_bundle(
        final,
        publication["content_hash"],
        expected_publication_digest=publication["publication_digest"],
    )
    if candidate.to_dict() != publication:
        raise AdapterError("reconstructed candidate differs from wire authority")
    return candidate


def _verify_publication_bytes(publication) -> None:
    for row in publication.files:
        path = publication.root.joinpath(*PurePosixPath(row.path).parts)
        if path.is_symlink() or not path.is_file():
            raise AdapterError("reconstructed candidate publication is incomplete")
        info = path.stat()
        if info.st_size != row.size or _file_sha256(path) != row.sha256:
            raise AdapterError("reconstructed candidate publication bytes differ")


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4 << 20)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _run(request_dir: Path, result_dir: Path) -> None:
    service = _load_service()
    registration = service.verify_registration(service.load_json(REGISTRATION_PATH))
    ready = service.verify_ready_receipt(service.load_json(READY_RECEIPT_PATH))
    if ready["receipt_digest"] != registration["ready_receipt_digest"]:
        raise AdapterError("READY receipt differs from registration")
    source = _source_from_ready(ready)
    sys.path.insert(0, str(source))

    from cacheon.arena_service import ArenaCandidateBinding, ArenaScreenReceipt
    from cacheon.chain.evaluation_leases import (
        EvaluationLease,
        EvaluationLeaseMember,
    )
    from cacheon.chain.remote_evaluation_dispatcher import (
        RemoteEvaluationRequest,
        RemoteWorkerCredential,
        RemoteWorkerTransportIdentity,
        seal_remote_response,
        verify_remote_request,
    )
    from cacheon.eval.b300_screen_deployment import (
        build_commissioned_b300_screen_worker,
    )
    from cacheon.eval.qualification_intake import QualificationReservation

    outer = service.verify_request(
        service.load_json(request_dir / "request.json"),
        request_dir,
        registration,
    )
    wire_value = service.load_json(
        service._artifact_for_role(outer, request_dir, "screen_payload"),
        maximum=64 << 20,
    )
    wire = RemoteEvaluationRequest.from_dict(wire_value)
    credential = RemoteWorkerCredential(
        registration["credential_id"], CREDENTIAL_PATH.read_bytes()
    )
    identity = RemoteWorkerTransportIdentity(**registration["transport_identity"])
    verify_remote_request(wire, identity, credential)
    lease_value = outer["lease"]
    lease = EvaluationLease(
        lease_value["lease_id"],
        lease_value["generation"],
        lease_value["stage"],
        lease_value["owner"],
        tuple(EvaluationLeaseMember(**row) for row in lease_value["members"]),
        lease_value["claimed_block"],
        lease_value["initial_expires_block"],
        lease_value["expires_block"],
    )
    body = wire.body
    publication = _safe_publication(
        service._artifact_for_role(outer, request_dir, "candidate_publication"),
        body["publication"],
    )
    reservation = QualificationReservation.from_dict(body["reservation"])
    candidate = ArenaCandidateBinding(
        reservation,
        publication,
        body["screen_attempt"],
    )
    if (
        candidate.digest != body["candidate_digest"]
        or lease.reservation_ids != (reservation.reservation_digest,)
    ):
        raise AdapterError("reconstructed candidate differs from authenticated lease")
    worker = build_commissioned_b300_screen_worker(registration, ready)
    try:
        evaluation = worker.run_remote_screen(lease, candidate)
    finally:
        worker.close()
    receipt = evaluation.payload
    if type(receipt) is not ArenaScreenReceipt:
        raise AdapterError("B300 screen worker returned an untyped receipt")
    if (
        evaluation.lease != lease
        or evaluation.disposition != "completed"
        or evaluation.envelope.lease_id != lease.lease_id
        or evaluation.envelope.payload_digest != receipt.digest
    ):
        raise AdapterError("B300 screen worker changed the exact lease/result envelope")
    response = seal_remote_response(wire, receipt, identity, credential)
    output = result_dir / "response.json"
    with output.open("xb") as handle:
        handle.write(_canonical_json(response.to_dict()) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(output, 0o400)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args(argv)
    request_dir = _closed_path(
        args.request_dir, PROCESSING_ROOT, "request directory", temporary=False
    )
    result_dir = _closed_path(
        args.result_dir, RESULTS_ROOT, "result directory", temporary=True
    )
    if (
        request_dir.is_symlink()
        or not request_dir.is_dir()
        or result_dir.is_symlink()
        or not result_dir.is_dir()
        or any(result_dir.iterdir())
    ):
        raise AdapterError("request/result carrier state is invalid")
    _run(request_dir, result_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AdapterError as exc:
        print(f"CACHEON-B300-ADAPTER-ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
