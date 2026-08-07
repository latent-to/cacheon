"""Fixed authenticated evaluation codec for the Cacheon B300 worker service.

The pod service supervises exactly one instance of this adapter per
commissioned epoch.  The adapter authenticates the path-free
``RemoteEvaluationRequest``, reconstructs the immutable candidate publication
under a content-addressed pod root, invokes the commissioned resident B300
worker, and emits one canonical authenticated ``response.json``.  The CPU
independently reopens the typed receipt before committing the durable lease.

Stage authority: screen work executes through
:func:`cacheon.eval.b300_screen_deployment.build_commissioned_b300_screen_worker`.
Qualification executes only when construction receives one exact
``B300RemoteQualificationCommission`` carrying the fuller sealed deployment
authorities.  Each authenticated request safely materializes its own candidate
publication and derives a singleton ``B300RemoteQualificationAdapter`` from
that fixed commission.  Without it, qualification is refused as a typed
pre-resident ``AdapterRequestFailed``.  Refusing before any resident call keeps
the epoch healthy instead of failing it, and keeps an uncommissioned gap
visible instead of silently absent.

No request field can select a command, module, executable, environment,
source, or output path.  All filesystem coordinates come from one closed
:class:`AdapterPaths`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cacheon.chain.evaluation_coordinator import WorkerReadiness
    from cacheon.chain.publication import WorkerBundlePublication
    from cacheon.eval.b300_qualification_deployment import (
        B300QualificationConstructionAuthority,
        B300QualificationDeployment,
    )
    from cacheon.eval.b300_remote_qualification_adapter import (
        B300RemoteQualificationAdapter,
    )

from cacheon.chain.remote_worker_registration import (
    registration_credential,
    registration_transport_identity,
    verify_ready_receipt,
    verify_registration,
)
from cacheon.chain.remote_worker_execution_marker import publish_resident_entry
from cacheon.chain.remote_worker_spool import (
    NATIVE_ARTIFACT_MANIFEST,
    SCHEMA_ADAPTER_COMMAND,
    SCHEMA_ADAPTER_CONTROL,
    artifact_for_role,
    load_json,
    spool_canonical_json,
    strict_json_object,
    verify_request,
    RemoteWorkerError,
)


MAX_PUBLICATION_BYTES = 4 * 1024 * 1024 * 1024


class AdapterError(RuntimeError):
    """The fixed adapter could not authenticate or execute evaluation work."""


class AdapterRequestFailed(AdapterError):
    """One authenticated request failed before it could touch the resident lane."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


class AdapterEpochFailed(AdapterError):
    """The commissioned runtime can no longer safely accept another request."""


@dataclass(frozen=True)
class AdapterPaths:
    """Closed filesystem contract for one installed adapter epoch."""

    registration: Path
    ready_receipt: Path
    credential: Path
    publication_root: Path
    processing_root: Path
    results_root: Path
    continuation_root: Path

    def __post_init__(self) -> None:
        for name in (
            "registration",
            "ready_receipt",
            "credential",
            "publication_root",
            "processing_root",
            "results_root",
            "continuation_root",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise AdapterError(f"adapter {name} path must be an absolute Path")


@dataclass(frozen=True)
class B300RemoteQualificationCommission:
    """Fixed authorities which derive one candidate-local adapter per request."""

    deployment: B300QualificationDeployment
    construction: B300QualificationConstructionAuthority
    readiness: WorkerReadiness

    def __post_init__(self) -> None:
        from cacheon.chain.evaluation_coordinator import WorkerReadiness
        from cacheon.eval.b300_qualification_deployment import (
            B300QualificationConstructionAuthority,
            B300QualificationDeployment,
        )

        if (
            type(self.deployment) is not B300QualificationDeployment
            or type(self.construction)
            is not B300QualificationConstructionAuthority
            or type(self.readiness) is not WorkerReadiness
        ):
            raise AdapterError(
                "qualification commission authorities are not exactly typed"
            )

    def adapter_for(
        self,
        publication: WorkerBundlePublication,
        continuation_store,
    ) -> B300RemoteQualificationAdapter:
        from cacheon.chain.publication import WorkerBundlePublication
        from cacheon.eval.b300_remote_qualification_adapter import (
            B300RemoteQualificationAdapter,
            B300WorkerBundleResolver,
        )
        from cacheon.eval.qualification_continuation import (
            QualificationContinuationStore,
        )

        if (
            type(publication) is not WorkerBundlePublication
            or type(continuation_store) is not QualificationContinuationStore
        ):
            raise AdapterError(
                "qualification publication or continuation store is not exactly typed"
            )
        return B300RemoteQualificationAdapter(
            self.deployment,
            self.construction,
            self.readiness,
            B300WorkerBundleResolver((publication,)),
            continuation_store,
        )


def _closed_path(raw: object, root: Path, label: str, *, temporary: bool) -> Path:
    if not isinstance(raw, str):
        raise AdapterError(f"{label} is not a path string")
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
                    character not in "0123456789abcdef" for character in path.name
                )
            )
        )
    ):
        raise AdapterError(f"{label} is outside its fixed content-addressed root")
    return path


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def safe_publication(archive_path: Path, expected_wire: object, publication_root: Path):
    """Reconstruct one immutable publication under its content-addressed root."""

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
            raise AdapterError(
                f"candidate publication manifest is invalid: {exc}"
            ) from None
        if (
            type(manifest) is not dict
            or set(manifest) != {"publication", "schema"}
            or manifest["schema"] != "cacheon-remote-worker-publication-v1"
            or manifest["publication"] != expected_wire
            or manifest_raw != spool_canonical_json(manifest) + b"\n"
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
        if (
            set(publication) != required
            or publication["schema"] != "cacheon.worker-bundle-publication.v1"
        ):
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
        } | {f"bundle/{row.path}" for row in files}
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
        parent = publication_root / address[:2]
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
                if relative.is_absolute() or any(
                    part in {"", ".", ".."} for part in relative.parts
                ):
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
                        raise AdapterError(
                            "candidate archive file exceeded its inventory"
                        )
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
                os.chmod(temporary.joinpath(*PurePosixPath(logical).parts), 0o555)
            os.chmod(temporary, 0o555)
            os.replace(temporary, final)
        finally:
            if temporary.exists():
                for current, _directory_names, file_names in os.walk(
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


class AdapterRuntime:
    """One commissioned worker process retained across sequential requests."""

    def __init__(
        self,
        paths: AdapterPaths,
        qualification_commission: B300RemoteQualificationCommission | None = None,
        *,
        qualification_capabilities: object | None = None,
    ) -> None:
        if type(paths) is not AdapterPaths:
            raise AdapterError("adapter paths are not exactly typed")
        if (
            qualification_commission is not None
            and type(qualification_commission)
            is not B300RemoteQualificationCommission
        ):
            raise AdapterError("qualification commission is not exactly typed")
        if (
            qualification_commission is not None
            and qualification_capabilities is not None
        ):
            raise AdapterError(
                "qualification commission and capabilities are mutually exclusive"
            )
        if qualification_capabilities is not None:
            from cacheon.eval.b300_qualification_commission import (
                B300QualificationCapabilities,
            )

            if type(qualification_capabilities) is not B300QualificationCapabilities:
                raise AdapterError(
                    "qualification capabilities are not exactly typed"
                )
        qualification_enabled = (
            qualification_commission is not None
            or qualification_capabilities is not None
        )
        if qualification_enabled:
            from cacheon.eval.qualification_continuation import (
                QualificationContinuationStore,
            )

            qualification_continuation_store = QualificationContinuationStore(
                paths.continuation_root
            )
        else:
            qualification_continuation_store = None
        registration = verify_registration(load_json(paths.registration))
        ready = verify_ready_receipt(load_json(paths.ready_receipt))
        if ready["receipt_digest"] != registration["ready_receipt_digest"]:
            raise AdapterError("READY receipt differs from registration")

        self.paths = paths
        self.registration = registration
        self.ready = ready
        self.credential = registration_credential(registration, paths.credential)
        self.identity = registration_transport_identity(registration)
        self._commissioned_service = None
        if qualification_capabilities is not None:
            from cacheon.eval.b300_qualification_commission import (
                build_commissioned_b300_qualification_service,
            )

            # One replay yields both the screen worker and the qualification
            # commission over the same resident model lifetime.
            service = build_commissioned_b300_qualification_service(
                registration, ready, qualification_capabilities
            )
            self._commissioned_service = service
            self.worker = service.worker
            self.qualification_commission = service.commission
        else:
            from cacheon.eval.b300_screen_deployment import (
                build_commissioned_b300_screen_worker,
            )

            self.worker = build_commissioned_b300_screen_worker(registration, ready)
            self.qualification_commission = qualification_commission
        self.qualification_continuation_store = qualification_continuation_store
        self.closed = False

    def verify_current(self) -> None:
        registration = verify_registration(load_json(self.paths.registration))
        ready = verify_ready_receipt(load_json(self.paths.ready_receipt))
        if registration != self.registration or ready != self.ready:
            raise AdapterError("commissioned registration or READY authority changed")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._commissioned_service is not None:
            self._commissioned_service.close()
        else:
            self.worker.close()


def run_with_runtime(
    request_dir: Path,
    result_dir: Path,
    runtime: AdapterRuntime,
) -> None:
    if type(runtime) is not AdapterRuntime or runtime.closed:
        raise AdapterEpochFailed("persistent adapter runtime is unavailable")
    try:
        runtime.verify_current()
    except Exception as exc:
        raise AdapterEpochFailed(
            "commissioned adapter authority is no longer current"
        ) from exc

    from cacheon.arena_service import ArenaCandidateBinding, ArenaScreenReceipt
    from cacheon.chain.evaluation_leases import EvaluationLease, EvaluationLeaseMember
    from cacheon.chain.remote_evaluation_dispatcher import seal_remote_response
    from cacheon.eval.qualification_intake import QualificationReservation

    # Everything in this block consumes only the per-request carrier.  It has
    # not called the commissioned worker and therefore cannot have mutated the
    # standing model.  Reject that request without sacrificing residency.
    try:
        outer = verify_request(
            load_json(request_dir / "request.json"),
            request_dir,
            runtime.registration,
            identity=runtime.identity,
            credential=runtime.credential,
        )
        from cacheon.chain.remote_evaluation_dispatcher import (
            RemoteEvaluationRequest,
            verify_remote_request,
        )

        stage = outer["lease"]["stage"]
        wire_value = load_json(
            artifact_for_role(outer, request_dir, f"{stage}_payload"),
            maximum=64 << 20,
        )
        wire = RemoteEvaluationRequest.from_dict(wire_value)
        verify_remote_request(wire, runtime.identity, runtime.credential)
        if stage == "qualification":
            qualification_commission = runtime.qualification_commission
            if qualification_commission is None:
                raise AdapterError(
                    "qualification execution authority is not commissioned for this"
                    " adapter; the tracked qualification worker entrypoint awaits its"
                    " deployment authorities"
                )
            body = wire.body
            candidates = body.get("candidates")
            if (
                type(candidates) is not list
                or len(candidates) != 1
                or type(candidates[0]) is not dict
                or "publication" not in candidates[0]
            ):
                raise AdapterError(
                    "qualification request does not contain one closed candidate"
                )
            publication = safe_publication(
                artifact_for_role(outer, request_dir, "candidate_publication"),
                candidates[0]["publication"],
                runtime.paths.publication_root,
            )
            qualification_adapter = qualification_commission.adapter_for(
                publication,
                runtime.qualification_continuation_store,
            )
        else:
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
            publication = safe_publication(
                artifact_for_role(outer, request_dir, "candidate_publication"),
                body["publication"],
                runtime.paths.publication_root,
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
                raise AdapterError(
                    "reconstructed candidate differs from authenticated lease"
                )
    except Exception as exc:
        raise AdapterRequestFailed(
            "request carrier/authentication/staging failed before resident work"
        ) from exc

    try:
        publish_resident_entry(result_dir, outer)
    except Exception as exc:
        raise AdapterRequestFailed(
            "resident-entry marker failed before resident work"
        ) from exc

    # Once the worker is called, an exception is conservatively epoch-fatal:
    # it may have followed resident mutation.  Typed result products, including
    # NO_DECISION outcomes, complete normally through this path.
    try:
        if stage == "qualification":
            payload = qualification_adapter.run(wire)
        else:
            evaluation = runtime.worker.run_remote_screen(lease, candidate)
            payload = evaluation.payload
            if type(payload) is not ArenaScreenReceipt:
                raise AdapterError("B300 screen worker returned an untyped receipt")
            if (
                evaluation.lease != lease
                or evaluation.disposition != "completed"
                or evaluation.envelope.lease_id != lease.lease_id
                or evaluation.envelope.payload_digest != payload.digest
            ):
                raise AdapterError(
                    "B300 screen worker changed the exact lease/result envelope"
                )
        response = seal_remote_response(
            wire, payload, runtime.identity, runtime.credential
        )
        output = result_dir / "response.json"
        with output.open("xb") as handle:
            handle.write(spool_canonical_json(response.to_dict()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(output, 0o400)
    except Exception as exc:
        raise AdapterEpochFailed(
            "resident evaluation failed after entering commissioned worker"
        ) from exc


def _run(request_dir: Path, result_dir: Path, paths: AdapterPaths) -> None:
    runtime = AdapterRuntime(paths)
    try:
        run_with_runtime(request_dir, result_dir, runtime)
    finally:
        runtime.close()


def _decode_command(raw: bytes) -> dict[str, object]:
    if not raw or len(raw) > 64 * 1024 or not raw.endswith(b"\n"):
        raise AdapterError("adapter command frame is malformed")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeError as exc:
        raise AdapterError(f"adapter command is invalid JSON: {exc}") from None
    try:
        value = strict_json_object(decoded)
    except RemoteWorkerError as exc:
        raise AdapterError(f"adapter command is invalid JSON: {exc}") from None
    if raw != spool_canonical_json(value) + b"\n":
        raise AdapterError("adapter command is not canonical JSON")
    return value


def _emit_control(
    state: str,
    request_id: str | None = None,
    *,
    output=None,
) -> None:
    value: dict[str, object] = {
        "schema": SCHEMA_ADAPTER_CONTROL,
        "state": state,
    }
    if request_id is not None:
        value["request_id"] = request_id
    stream = sys.stdout.buffer if output is None else output
    stream.write(spool_canonical_json(value) + b"\n")
    stream.flush()


def validated_command_paths(
    raw: bytes, paths: AdapterPaths
) -> tuple[str, Path, Path]:
    command = _decode_command(raw)
    if set(command) != {
        "operation",
        "request_dir",
        "request_id",
        "result_dir",
        "schema",
    } or command.get("schema") != SCHEMA_ADAPTER_COMMAND:
        raise AdapterEpochFailed("adapter command fields are not closed")
    if command.get("operation") != "evaluate":
        raise AdapterEpochFailed("adapter command operation is unsupported")
    request_id = command.get("request_id")
    if (
        not isinstance(request_id, str)
        or len(request_id) != 64
        or any(character not in "0123456789abcdef" for character in request_id)
    ):
        raise AdapterEpochFailed("adapter command request ID is malformed")
    request_dir = _closed_path(
        command.get("request_dir"),
        paths.processing_root,
        "request directory",
        temporary=False,
    )
    result_dir = _closed_path(
        command.get("result_dir"),
        paths.results_root,
        "result directory",
        temporary=True,
    )
    if request_dir.name != request_id:
        raise AdapterEpochFailed("adapter command names another request")
    if (
        request_dir.is_symlink()
        or not request_dir.is_dir()
        or result_dir.is_symlink()
        or not result_dir.is_dir()
        or any(result_dir.iterdir())
    ):
        raise AdapterRequestFailed(
            "request/result carrier state is invalid", request_id=request_id
        )
    return request_id, request_dir, result_dir


def serve_runtime(
    runtime, paths: AdapterPaths, input_stream, control_output
) -> int:
    """Serve frames on exactly one runtime; never replace it in-process."""

    _emit_control("ready", output=control_output)
    for raw in input_stream:
        request_id: str | None = None
        try:
            request_id, request_dir, result_dir = validated_command_paths(raw, paths)
            run_with_runtime(request_dir, result_dir, runtime)
        except AdapterRequestFailed as exc:
            request_id = request_id or exc.request_id
            print(
                "CACHEON-B300-ADAPTER-REQUEST-FAILED: "
                f"request={request_id or 'unknown'} "
                f"type={type(exc.__cause__ or exc).__name__}",
                file=sys.stderr,
                flush=True,
            )
            _emit_control("request_failed", request_id, output=control_output)
            continue
        except Exception as exc:
            print(
                "CACHEON-B300-ADAPTER-EPOCH-FAILED: "
                f"request={request_id or 'unknown'} "
                f"type={type(exc.__cause__ or exc).__name__}",
                file=sys.stderr,
                flush=True,
            )
            _emit_control("epoch_failed", request_id, output=control_output)
            return 2
        _emit_control("completed", request_id, output=control_output)
    return 0


def _load_qualification_capabilities(specifier: str, source_sha256: str):
    """Load one private factory only after its exact source bytes verify."""

    from cacheon.eval.qualification_capability_loader import (
        QualificationCapabilityLoadError,
        load_qualification_capabilities,
    )

    try:
        return load_qualification_capabilities(specifier, source_sha256)
    except QualificationCapabilityLoadError as exc:
        raise AdapterError(str(exc)) from None


def _serve(paths: AdapterPaths, qualification_capabilities=None) -> int:
    # Reserve the original stdout exclusively for the tiny control protocol.
    # Imported controller/runtime code is redirected to stderr so an
    # incidental diagnostic cannot be mistaken for a completed request.
    control_output = sys.stdout.buffer
    sys.stdout = sys.stderr
    runtime = AdapterRuntime(
        paths, qualification_capabilities=qualification_capabilities
    )
    try:
        return serve_runtime(runtime, paths, sys.stdin.buffer, control_output)
    finally:
        runtime.close()


def _adapter_paths(args: argparse.Namespace) -> AdapterPaths:
    return AdapterPaths(
        registration=Path(args.registration),
        ready_receipt=Path(args.ready_receipt),
        credential=Path(args.credential),
        publication_root=Path(args.publication_root),
        processing_root=Path(args.processing_root),
        results_root=Path(args.results_root),
        continuation_root=Path(args.continuation_root),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--registration", required=True)
    parser.add_argument("--ready-receipt", required=True)
    parser.add_argument("--credential", required=True)
    parser.add_argument("--publication-root", required=True)
    parser.add_argument("--processing-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--continuation-root", required=True)
    parser.add_argument("--request-dir")
    parser.add_argument("--result-dir")
    parser.add_argument(
        "--qualification-capabilities",
        help=(
            "top-level MODULE:ATTRIBUTE naming a reviewed zero-argument factory"
            " returning B300QualificationCapabilities; requires the exact source"
            " digest and --serve"
        ),
    )
    parser.add_argument(
        "--qualification-capabilities-sha256",
        help="exact lowercase SHA-256 of the named qualification factory source",
    )
    args = parser.parse_args(argv)
    if (args.qualification_capabilities is None) != (
        args.qualification_capabilities_sha256 is None
    ):
        parser.error(
            "--qualification-capabilities and"
            " --qualification-capabilities-sha256 must be provided together"
        )
    paths = _adapter_paths(args)
    if args.serve:
        if args.request_dir is not None or args.result_dir is not None:
            parser.error("--serve does not accept one-shot request paths")
        capabilities = None
        if args.qualification_capabilities is not None:
            load_receipt = _load_qualification_capabilities(
                args.qualification_capabilities,
                args.qualification_capabilities_sha256,
            )
            capabilities = load_receipt.capabilities
        return _serve(paths, capabilities)
    if args.qualification_capabilities is not None:
        parser.error(
            "--qualification-capabilities requires the persistent --serve"
            " service; one-shot mode cannot commission qualification"
        )
    if args.request_dir is None or args.result_dir is None:
        parser.error("one-shot mode requires --request-dir and --result-dir")
    request_dir = _closed_path(
        args.request_dir, paths.processing_root, "request directory", temporary=False
    )
    result_dir = _closed_path(
        args.result_dir, paths.results_root, "result directory", temporary=True
    )
    if (
        request_dir.is_symlink()
        or not request_dir.is_dir()
        or result_dir.is_symlink()
        or not result_dir.is_dir()
        or any(result_dir.iterdir())
    ):
        raise AdapterError("request/result carrier state is invalid")
    _run(request_dir, result_dir, paths)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AdapterError as exc:
        print(f"CACHEON-B300-ADAPTER-ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
