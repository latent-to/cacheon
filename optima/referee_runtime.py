"""Resolve the immutable referee source mounted into crownable OCI launches.

Production commands are commonly invoked from a live checkout, but that checkout
must never be mounted into an untrusted candidate container.  This module turns
the checkout into the small, content-addressed release defined by
``optima.source_release`` and independently verifies the arena-owned identities
before returning a mountable path.
"""

from __future__ import annotations

import os
from pathlib import Path

from optima.source_release import (
    RELEASE_MANIFEST,
    RefereeReleaseError,
    RefereeSourceRelease,
    build_referee_source_release,
    verify_referee_source_release,
)


def _resolved_path(value: str | os.PathLike[str], *, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RefereeReleaseError(f"{name} does not resolve: {exc}") from None


def resolve_referee_runtime(
    source_root: str | os.PathLike[str],
    publish_root: str | os.PathLike[str],
    *,
    expected_tree_digest: str,
    expected_referee_source_digest: str,
) -> RefereeSourceRelease:
    """Return a verified release, building it from a checkout when necessary.

    A path containing a release manifest is always interpreted as a release.  If
    verification fails it is not silently reinterpreted as a checkout.  Publication
    is required to live outside the source tree so the candidate-visible mount and
    operator checkout cannot alias through a nested path.
    """

    source = _resolved_path(source_root, name="referee source root")
    if not source.is_dir():
        raise RefereeReleaseError("referee source root must be a directory")

    manifest = source / RELEASE_MANIFEST
    if os.path.lexists(manifest):
        return verify_referee_source_release(
            source,
            expected_tree_digest=expected_tree_digest,
            expected_referee_source_digest=expected_referee_source_digest,
        )

    publication = Path(publish_root).expanduser()
    if not publication.is_absolute():
        publication = publication.resolve()
    publication.mkdir(parents=True, mode=0o700, exist_ok=True)
    publication = _resolved_path(publication, name="referee publication root")
    if publication == source or publication in source.parents or source in publication.parents:
        raise RefereeReleaseError(
            "referee publication root and source checkout must be disjoint"
        )
    return build_referee_source_release(
        source,
        publication,
        expected_tree_digest=expected_tree_digest,
        expected_referee_source_digest=expected_referee_source_digest,
    )
