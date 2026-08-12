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

import hashlib
import logging
from pathlib import Path

from cacheon.copy_fingerprint import (
    _SUBSTANTIAL_NORM_LEN,
    _top_level_definition_fingerprints,
    SubmittedDeltaFingerprint,
    compare_submitted_deltas,
    cuda_source_fingerprint,
    fingerprint_submitted_delta,
    normalized_source,
)

_LOG = logging.getLogger(__name__)

_REFERENCE_DIRS = ("examples", "tests/fixtures")
_SKIP_STATUSES = {"failed", "expired"}
_CACHE: tuple[tuple[str, SubmittedDeltaFingerprint], ...] | None = None

# Owner order 2026-08-13: a submission that contains validator-library code
# (cacheon_kernels is public; miners resubmitting it are returning our own
# work) dies at the CPU screen. Flip to False to return the library signal to
# advisory-only without disturbing the marker history.
_LIBRARY_KILL_ENABLED = True
_LIBRARY_DIR = "cacheon_kernels"
_LIBRARY_PY_SUFFIXES = {".py"}
_LIBRARY_CUDA_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp"}
_LIBRARY_CACHE: dict[str, str] | None = None


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


def validator_library_fingerprints() -> dict[str, str]:
    """Map every substantial validator-library fingerprint to its file.

    Uses the same relocation-proof projections the submission side folds into
    ``containment_fingerprints``: whole-file normalized source, per-definition
    dependency projections, and raw plus reformat-invariant CUDA text. A
    library file that fails to parse is skipped with a warning; the standing
    loop must never wedge over reference material.
    """

    global _LIBRARY_CACHE
    if _LIBRARY_CACHE is None:
        table: dict[str, str] = {}
        base = _repo_root() / _LIBRARY_DIR
        paths = sorted(base.rglob("*")) if base.is_dir() else []
        for path in paths:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(base).as_posix()
            fps: set[str] = set()
            if path.suffix in _LIBRARY_PY_SUFFIXES:
                try:
                    source = path.read_text(encoding="utf-8")
                    norm = normalized_source(source)
                    fps.update(_top_level_definition_fingerprints(source))
                except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
                    _LOG.warning("library file %s is not fingerprintable", rel)
                    continue
                if len(norm) >= _SUBSTANTIAL_NORM_LEN:
                    fps.add(hashlib.sha256(norm.encode("utf-8")).hexdigest())
            elif path.suffix in _LIBRARY_CUDA_SUFFIXES:
                try:
                    raw = path.read_bytes()
                except OSError:
                    continue
                fps.add(hashlib.sha256(raw).hexdigest())
                fps.add(cuda_source_fingerprint(raw.decode("utf-8", errors="replace")))
            for fingerprint in fps:
                table.setdefault(fingerprint, rel)
        _LIBRARY_CACHE = table
        _LOG.info("validator library corpus holds %d fingerprints", len(table))
    return _LIBRARY_CACHE


def _library_marker(rel: str) -> str:
    return ("library-" + rel.replace("/", "-"))[:80]


def reference_copy_match(fingerprint: object) -> str | None:
    """Name the public reference this fingerprint authoritatively copies."""

    if type(fingerprint) is not SubmittedDeltaFingerprint:
        return None
    for name, reference in validator_reference_fingerprints():
        if compare_submitted_deltas(reference, fingerprint).authoritative:
            return name
    return None


def library_copy_match(fingerprint: object) -> str | None:
    """Name the library file whose code this submission contains."""

    if not _LIBRARY_KILL_ENABLED or type(fingerprint) is not SubmittedDeltaFingerprint:
        return None
    table = validator_library_fingerprints()
    for fp in fingerprint.containment_fingerprints:
        rel = table.get(fp)
        if rel is not None:
            return rel
    return None


def reconcile_reference_copies(store) -> tuple[tuple[str, str], ...]:
    """Idempotently demote every unresolved match against validator material."""

    dispositions: list[tuple[str, str]] = []
    for row in store.all():
        if row.delta_fingerprint is None or row.status in _SKIP_STATUSES:
            continue
        name = reference_copy_match(row.delta_fingerprint)
        if name is None:
            library = library_copy_match(row.delta_fingerprint)
            if library is not None:
                name = _library_marker(library)
        if name is not None:
            store.mark_reference_copy(row.reservation_id, name)
            dispositions.append((row.reservation_id, name))
    return tuple(dispositions)


__all__ = [
    "library_copy_match",
    "reconcile_reference_copies",
    "reference_copy_match",
    "validator_library_fingerprints",
    "validator_reference_fingerprints",
]
