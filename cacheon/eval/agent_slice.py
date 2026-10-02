"""The sealed agent workload: a slice manifest and its directory of session windows.

A slice is S session windows cut from a pinned public trace corpus, ordered by
a seeded hash so the first l sessions are the fixed work at load l. The
manifest states the rules that produced it and, per session, how many main
turns and subagent inner requests the window contains; that per-root count is
what the scorer checks completed work against. The directory holds one trace
per file in the loader's format, named so that sorted filename order is the
sealed order, and the manifest's digest covers every named file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST_FIELDS = frozenset(
    {"dataset", "revision", "loader", "rules", "rejected", "dir", "files", "order",
     "sha256_of_named_files", "sessions"}
)
SESSION_FIELDS = frozenset(
    {"id", "n_main_total", "start", "k", "groups", "inner", "cold_prefix", "peak", "new_input",
     "reused_input", "output", "inner_input", "inner_output", "recorded_span_s"}
)


class SliceManifestError(ValueError):
    """The manifest, its directory, or their agreement is not as sealed."""


@dataclass(frozen=True)
class SliceSession:
    id: str
    start: int
    main_turns: int
    inner_requests: int
    cold_prefix_tokens: int
    peak_context_tokens: int


@dataclass(frozen=True)
class SliceManifest:
    dataset: str
    revision: str
    loader: str
    directory: Path
    digest: str
    rules: dict[str, object]
    sessions: tuple[SliceSession, ...]

    @property
    def max_load(self) -> int:
        return len(self.sessions)

    def expected_work(self, load: int) -> dict[str, tuple[int, int]]:
        """Completed (main, inner) turns per root that the fixed work at ``load`` requires."""
        if type(load) is not int or not 1 <= load <= self.max_load:
            raise SliceManifestError(f"load {load!r} is outside the slice's 1..{self.max_load}")
        return {s.id: (s.main_turns, s.inner_requests) for s in self.sessions[:load]}

    def turns(self, load: int) -> int:
        return sum(main + inner for main, inner in self.expected_work(load).values())


def load_slice_manifest(path: Path) -> SliceManifest:
    """Reopen a manifest and prove its directory is the one it was sealed over."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SliceManifestError(f"manifest unreadable: {exc}") from None
    if type(raw) is not dict or set(raw) != MANIFEST_FIELDS:
        raise SliceManifestError("manifest fields are not closed")
    rows = raw["sessions"]
    if type(rows) is not list or not rows or any(
        type(r) is not dict or set(r) != SESSION_FIELDS for r in rows
    ):
        raise SliceManifestError("manifest sessions are malformed")
    sessions = tuple(
        SliceSession(
            _text(r["id"]), _count(r["start"]), _count(r["k"], minimum=1), _count(r["inner"]),
            _count(r["cold_prefix"], minimum=1), _count(r["peak"], minimum=1),
        )
        for r in rows
    )
    if len({s.id for s in sessions}) != len(sessions):
        raise SliceManifestError("manifest repeats a session id")
    directory = Path(path).parent / _text(raw["dir"])
    if raw["files"] != len(sessions) or raw["order"] != "sorted filename = sealed order":
        raise SliceManifestError("manifest file accounting does not match its sessions")
    digest = _digest_directory(directory, sessions)
    if digest != raw["sha256_of_named_files"]:
        raise SliceManifestError("slice directory differs from the sealed digest")
    if type(raw["rules"]) is not dict:
        raise SliceManifestError("manifest rules must be a mapping")
    return SliceManifest(
        _text(raw["dataset"]), _text(raw["revision"]), _text(raw["loader"]),
        directory, digest, dict(raw["rules"]), sessions,
    )


def _digest_directory(directory: Path, sessions: tuple[SliceSession, ...]) -> str:
    """Hash the named files in sealed order; the names are part of what is sealed."""
    if not directory.is_dir():
        raise SliceManifestError(f"slice directory missing: {directory}")
    names = sorted(p.name for p in directory.glob("*.json"))
    expected = [f"{i:03d}_{s.id}.json" for i, s in enumerate(sessions)]
    if names != expected:
        raise SliceManifestError("slice directory files do not match the manifest order")
    digest = hashlib.sha256()
    for name, session in zip(names, sessions, strict=True):
        body = (directory / name).read_bytes()
        try:
            trace = json.loads(body)
        except ValueError:
            raise SliceManifestError(f"{name} is not JSON") from None
        if type(trace) is not dict or trace.get("id") != session.id:
            raise SliceManifestError(f"{name} does not hold session {session.id}")
        digest.update(name.encode())
        digest.update(body)
    return digest.hexdigest()


def _text(value: object) -> str:
    if type(value) is not str or not value:
        raise SliceManifestError("manifest text field is empty or untyped")
    return value


def _count(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SliceManifestError("manifest count field is invalid")
    return value


__all__ = ["SliceManifest", "SliceManifestError", "SliceSession", "load_slice_manifest"]
