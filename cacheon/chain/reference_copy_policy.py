"""Validator-published reference bundles are copies by definition.

Every bundle this repository itself publishes (example bundles and test
fixtures carrying a manifest) is a known quantity: a submission that matches
one under the same authoritative rules used between miners is unoriginal
regardless of arrival order, and dies at the CPU screen instead of costing a
GPU qualification.  Matching stays exactly as strict as
submission-vs-submission copy detection — exact identity, normalized
identity, or symmetric containment — so genuinely modified work never
auto-demotes here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cacheon.copy_fingerprint import (
    SubmittedDeltaFingerprint,
    compare_submitted_deltas,
    fingerprint_submitted_delta,
)

_LOG = logging.getLogger(__name__)

_REFERENCE_DIRS = ("examples", "tests/fixtures")
_SKIP_STATUSES = {"failed", "expired"}
_CACHE: tuple[tuple[str, SubmittedDeltaFingerprint], ...] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bundle_roots(root: Path):
    for rel in _REFERENCE_DIRS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in sorted(base.iterdir()):
            if (path / "manifest.toml").is_file():
                yield path


def validator_reference_fingerprints() -> tuple[
    tuple[str, SubmittedDeltaFingerprint], ...
]:
    """Fingerprint every public reference bundle once per process.

    A reference that no longer parses (retired target, broken-on-purpose
    example) is skipped with a warning; the reconciliation must never wedge
    the standing loop over a fixture.
    """

    global _CACHE
    if _CACHE is None:
        rows: list[tuple[str, SubmittedDeltaFingerprint]] = []
        for path in _bundle_roots(_repo_root()):
            try:
                rows.append((path.name, fingerprint_submitted_delta(path)))
            except (OSError, TypeError, ValueError) as exc:
                _LOG.warning(
                    "reference bundle %s is not fingerprintable: %s",
                    path.name,
                    exc,
                )
        _CACHE = tuple(rows)
        if not _CACHE:
            _LOG.warning("no validator reference bundle was fingerprintable")
        else:
            _LOG.info(
                "validator reference corpus holds %d public bundles",
                len(_CACHE),
            )
    return _CACHE


def reference_copy_match(fingerprint: object) -> str | None:
    """Name the public reference this fingerprint authoritatively copies."""

    if type(fingerprint) is not SubmittedDeltaFingerprint:
        return None
    for name, reference in validator_reference_fingerprints():
        if compare_submitted_deltas(reference, fingerprint).authoritative:
            return name
    return None


def reconcile_reference_copies(store) -> tuple[tuple[str, str], ...]:
    """Idempotently demote every unresolved match against a public reference."""

    dispositions: list[tuple[str, str]] = []
    for row in store.all():
        if row.delta_fingerprint is None or row.status in _SKIP_STATUSES:
            continue
        name = reference_copy_match(row.delta_fingerprint)
        if name is not None:
            store.mark_reference_copy(row.reservation_id, name)
            dispositions.append((row.reservation_id, name))
    return tuple(dispositions)


__all__ = [
    "reconcile_reference_copies",
    "reference_copy_match",
    "validator_reference_fingerprints",
]
