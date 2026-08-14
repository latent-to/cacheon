"""Reconstruct wire-declared candidate publications from request carriers.

One authenticated qualification request carries one ``candidate_publication``
artifact per wire candidate, in the exact order of the body's ``candidates``
list.  Both lists are covered by the request authentication, so positional
pairing is exact: a swapped, missing, duplicated, or substituted carrier fails
the per-candidate publication verification before any resident work.  Screen
requests keep their single-carrier shape through
:func:`cacheon.chain.remote_worker_spool.artifact_for_role`.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Mapping

from cacheon.chain.remote_worker_spool import (
    NATIVE_ARTIFACT_MANIFEST,
    spool_canonical_json,
)

if TYPE_CHECKING:
    from cacheon.chain.publication import WorkerBundlePublication

MAX_PUBLICATION_BYTES = 4 * 1024 * 1024 * 1024


def artifacts_for_role(
    request: Mapping[str, Any], root: Path, role: str
) -> tuple[Path, ...]:
    """Return every artifact blob for one role in authenticated list order."""

    from cacheon.eval.b300_remote_worker_adapter import AdapterError

    matches = tuple(
        root / "blobs" / row["sha256"]
        for row in request["artifacts"]
        if row["role"] == role
    )
    if not matches:
        raise AdapterError(f"request does not contain a {role} artifact")
    return matches


def resolve_cohort_publications(
    request: Mapping[str, Any],
    request_dir: Path,
    candidates: list,
    publication_root: Path,
) -> tuple["WorkerBundlePublication", ...]:
    """Pair each wire candidate with its carrier archive, in cohort order."""

    from cacheon.eval.b300_remote_worker_adapter import AdapterError

    archives = artifacts_for_role(request, request_dir, "candidate_publication")
    if len(archives) != len(candidates):
        raise AdapterError(
            "qualification publication carriers differ from the request cohort"
        )
    return tuple(
        safe_publication(archive, row["publication"], publication_root)
        for archive, row in zip(archives, candidates, strict=True)
    )


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
    from cacheon.eval.b300_remote_worker_adapter import AdapterError
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


__all__ = [
    "MAX_PUBLICATION_BYTES",
    "artifacts_for_role",
    "resolve_cohort_publications",
    "safe_publication",
]
