"""Derive the next commission's incumbent authority from the durable store.

The commission packet measures every candidate against one authority
directory: ``incumbent-stack.json`` is the evaluation stack the pod boots as
its baseline, and ``sources/<artifact_digest>`` holds the frozen bundle bytes
behind each proposal entry.  Until 2026-09-02 that directory was assembled by
hand after every crown, so the store knew the winner while the pod kept
measuring against whatever an operator last copied.

This module writes the directory from the settlement row instead.  The newest
``STACK_TRANSITION`` event names the arena whose current durable stack is the
crown; the stack bytes are the store's own canonical encoding; and each
proposal entry's bytes are the reservation publication the intake retained when
the winning bundle was admitted, reopened through the same carrier the
dispatcher uses.  Anything missing fails closed.  Nothing here is typed by
hand, and nothing is fabricated.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from cacheon.bundle_hash import committed_content_hash
from cacheon.chain.intake import FinalizedIntakeStore, IntakeError
from cacheon.chain.publication import (
    WorkerBundlePublicationError,
    reopen_worker_bundle,
)
from cacheon.stack_manifest import EvaluationStackManifest, ProposalContributionRef

AUTHORITY_SCHEMA = "cacheon.incumbent-authority.v1"
STACK_FILE = "incumbent-stack.json"
RECEIPT_FILE = "incumbent-authority.json"
SOURCES_DIR = "sources"


class IncumbentAuthorityError(RuntimeError):
    """The settled crown cannot be turned into a closed authority directory."""


@dataclass(frozen=True)
class IncumbentSource:
    """One proposal entry and the retained publication that carries its bytes."""

    target_id: str
    artifact_digest: str
    reservation_id: str
    publication_digest: str

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_digest": self.artifact_digest,
            "publication_digest": self.publication_digest,
            "reservation_id": self.reservation_id,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class IncumbentAuthority:
    """The written authority directory and the settlement row it came from."""

    root: Path
    arena_digest: str
    generation: int
    stack_digest: str
    tree_digest: str
    transition_event_id: str
    settlement_sequence: int
    stack_sha256: str
    sources: tuple[IncumbentSource, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_digest": self.arena_digest,
            "generation": self.generation,
            "schema": AUTHORITY_SCHEMA,
            "settlement_sequence": self.settlement_sequence,
            "sources": [source.to_dict() for source in self.sources],
            "stack_digest": self.stack_digest,
            "stack_sha256": self.stack_sha256,
            "transition_event_id": self.transition_event_id,
            "tree_digest": self.tree_digest,
        }


def derive_incumbent_authority(
    store: FinalizedIntakeStore, output: str | os.PathLike[str]
) -> IncumbentAuthority:
    """Write ``output`` from the newest settled crown, or raise without writing."""

    root = Path(output)
    if not root.is_absolute():
        raise IncumbentAuthorityError("authority output must be an absolute path")
    if root.exists() or root.is_symlink():
        raise IncumbentAuthorityError(
            "authority output already exists; authorities are append-only"
        )
    if not root.parent.is_dir():
        raise IncumbentAuthorityError("authority output parent does not exist")
    transition = store.latest_stack_transition()
    if transition is None:
        raise IncumbentAuthorityError(
            "no settled crown: the store carries no STACK_TRANSITION event"
        )
    sequence, arena_digest = transition
    try:
        state = store.evaluation_stack(arena_digest)
    except IntakeError as exc:
        raise IncumbentAuthorityError(
            f"crowned arena stack cannot reopen: {exc}"
        ) from exc
    if state.generation < 1:
        raise IncumbentAuthorityError("crowned arena stack is still at genesis")
    entries = dict(state.manifest.entries)
    if not entries:
        raise IncumbentAuthorityError("crowned stack carries no contribution")
    sources = tuple(
        _retained_source(store, target_id, ref) for target_id, ref in sorted(entries.items())
    )
    encoded = json.dumps(
        state.manifest.to_dict(), separators=(",", ":"), sort_keys=True
    ).encode()
    if EvaluationStackManifest.from_dict(json.loads(encoded)).digest != state.manifest.digest:
        raise IncumbentAuthorityError("crowned stack does not round-trip byte for byte")
    authority = IncumbentAuthority(
        root,
        state.arena_digest,
        state.generation,
        state.manifest.digest,
        state.tree_digest,
        state.transition_event_id,
        sequence,
        hashlib.sha256(encoded).hexdigest(),
        sources,
    )
    incoming = root.parent / f"{root.name}.incoming"
    if incoming.exists() or incoming.is_symlink():
        raise IncumbentAuthorityError("authority staging path already exists")
    incoming.mkdir(mode=0o700)
    try:
        sources_root = incoming / SOURCES_DIR
        sources_root.mkdir(mode=0o700)
        for source in sources:
            _copy_retained_source(
                store, source, sources_root / source.artifact_digest
            )
        _write_once(incoming / STACK_FILE, encoded, mode=0o400)
        _write_once(
            incoming / RECEIPT_FILE,
            json.dumps(authority.to_dict(), separators=(",", ":"), sort_keys=True).encode(),
            mode=0o400,
        )
        sources_root.chmod(0o555)
        incoming.chmod(0o555)
        incoming.rename(root)
    except BaseException:
        _remove_tree(incoming)
        raise
    return authority


def _retained_source(
    store: FinalizedIntakeStore, target_id: str, ref: object
) -> IncumbentSource:
    if type(ref) is not ProposalContributionRef:
        raise IncumbentAuthorityError(
            f"{target_id}: only proposal contributions can be carried; an "
            "integrated entry needs its reviewed source tree"
        )
    rows = store.qualified_publications(ref.artifact_digest)
    if not rows:
        raise IncumbentAuthorityError(
            f"{target_id}: no qualified reservation retains bundle "
            f"{ref.artifact_digest[:12]}"
        )
    row = rows[0]
    return IncumbentSource(
        target_id, ref.artifact_digest, row.reservation_id, row.publication_digest
    )


def _copy_retained_source(
    store: FinalizedIntakeStore, source: IncumbentSource, destination: Path
) -> None:
    row = store.get(source.reservation_id)
    try:
        publication = reopen_worker_bundle(
            row.publication_root,
            source.artifact_digest,
            expected_receipt_digest=source.publication_digest,
        )
    except WorkerBundlePublicationError as exc:
        raise IncumbentAuthorityError(
            f"{source.target_id}: retained publication cannot reopen: {exc}"
        ) from exc
    shutil.copytree(publication.root, destination, symlinks=False, copy_function=shutil.copy2)
    paths = sorted(destination.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in paths:
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise IncumbentAuthorityError(
                f"{source.target_id}: retained publication carries a non-regular entry"
            )
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    observed = committed_content_hash(destination)
    if observed != source.artifact_digest:
        raise IncumbentAuthorityError(
            f"{source.target_id}: copied bytes hash to {observed[:12]}, "
            f"not the crowned artifact {source.artifact_digest[:12]}"
        )


def _write_once(path: Path, payload: bytes, *, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)
    root.chmod(0o700)
    shutil.rmtree(root, ignore_errors=True)


__all__ = [
    "AUTHORITY_SCHEMA",
    "IncumbentAuthority",
    "IncumbentAuthorityError",
    "IncumbentSource",
    "RECEIPT_FILE",
    "SOURCES_DIR",
    "STACK_FILE",
    "derive_incumbent_authority",
]
