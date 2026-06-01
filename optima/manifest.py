"""Bundle manifest parsing and validation.

A *bundle* is what a miner submits: a directory (or tarball) containing a
``manifest.toml`` plus kernel source and optional eligibility metadata. The
manifest is **data**, not code — it declares which slots the bundle claims to
implement and where the source lives. The validator reads it; the miner never
runs code at this stage.

This module is pure-Python (no torch/GPU) so the whole intake/validation path
runs anywhere.

Bundle layout::

    bundle/
      manifest.toml
      kernels/
        silu_and_mul.py        # exposes the slot's `entry` callable
      metadata/
        silu_and_mul.json      # optional eligibility (dtypes, arch, max dims)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

def _load_toml(p: Path) -> dict:
    """Parse a TOML file using whatever backend is available.

    Prefers stdlib ``tomllib`` (3.11+), then ``tomli`` (the de-facto 3.10
    backend, same binary-mode API), then the older ``toml`` package.
    """
    try:
        import tomllib  # type: ignore

        with p.open("rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        pass
    try:
        import tomli  # type: ignore

        with p.open("rb") as f:
            return tomli.load(f)
    except ModuleNotFoundError:
        pass
    import toml  # type: ignore

    return toml.loads(p.read_text())


ABI_VERSION = "optima-op-abi-v0"
_ID_RE = re.compile(r"^[0-9A-Za-z._\-]+$")


class ManifestError(ValueError):
    """Raised when a manifest is malformed or violates a structural rule."""


@dataclass(frozen=True)
class OpEntry:
    slot: str
    source: str  # bundle-relative path to the kernel module
    entry: str  # callable name inside the module
    dtypes: tuple[str, ...]
    architectures: tuple[str, ...]
    metadata: str | None
    prepare: str | None = None  # optional 2nd callable (weight-prep) for (prepare, forward) slots
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Manifest:
    bundle_id: str
    abi_version: str
    ops: tuple[OpEntry, ...]
    raw: dict[str, Any] = field(default_factory=dict)

    def op_for(self, slot: str) -> OpEntry | None:
        for op in self.ops:
            if op.slot == slot:
                return op
        return None


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ManifestError(msg)


def _safe_relpath(root: Path, rel: str, *, kind: str) -> Path:
    """Resolve ``rel`` under ``root`` and refuse to escape the bundle.

    Rejects absolute paths, ``..`` traversal, and symlinks that resolve outside
    ``root``. Returns the resolved path (which must exist).
    """
    _require(isinstance(rel, str) and rel != "", f"{kind} path must be a non-empty string")
    _require(not rel.startswith("/"), f"{kind} path must be relative: {rel!r}")
    p = (root / rel).resolve()
    root_resolved = root.resolve()
    _require(
        p == root_resolved or root_resolved in p.parents,
        f"{kind} path escapes bundle root: {rel!r}",
    )
    _require(p.exists(), f"{kind} not found: {rel!r}")
    _require(p.is_file(), f"{kind} must be a file: {rel!r}")
    return p


def load_manifest(bundle_root: str | Path) -> Manifest:
    """Load and structurally validate ``manifest.toml`` under ``bundle_root``.

    Validates schema and path-safety only. It does NOT import or execute any
    miner code (that is ``optima.sandbox``'s job) and it does NOT check the slot
    contract numerically (that is ``optima.verify``'s job).
    """
    root = Path(bundle_root).resolve()
    _require(root.is_dir(), f"bundle root is not a directory: {root}")
    manifest_path = root / "manifest.toml"
    _require(manifest_path.is_file(), f"manifest.toml not found in {root}")

    try:
        data = _load_toml(manifest_path)
    except Exception as exc:  # noqa: BLE001 - surface parse errors cleanly
        raise ManifestError(f"failed to parse manifest.toml: {exc}") from exc

    bundle_id = str(data.get("bundle_id", "")).strip()
    _require(bool(bundle_id), "manifest must set a non-empty bundle_id")
    _require(bool(_ID_RE.match(bundle_id)), f"bundle_id has illegal chars: {bundle_id!r}")

    abi = str(data.get("abi_version", "")).strip()
    _require(
        abi == ABI_VERSION,
        f"unsupported abi_version {abi!r}; this validator speaks {ABI_VERSION!r}",
    )

    ops_raw = data.get("ops")
    _require(isinstance(ops_raw, list) and ops_raw, "manifest must contain a non-empty [[ops]] list")

    ops: list[OpEntry] = []
    seen_slots: set[str] = set()
    for i, op in enumerate(ops_raw):
        _require(isinstance(op, dict), f"ops[{i}] must be a table")
        slot = str(op.get("slot", "")).strip()
        _require(bool(slot), f"ops[{i}] missing 'slot'")
        _require(slot not in seen_slots, f"duplicate slot in manifest: {slot!r}")
        seen_slots.add(slot)

        source = str(op.get("source", "")).strip()
        entry = str(op.get("entry", "")).strip()
        _require(bool(source), f"ops[{i}] ({slot}) missing 'source'")
        _require(bool(entry) and entry.isidentifier(), f"ops[{i}] ({slot}) 'entry' must be a python identifier")

        prepare = op.get("prepare")
        if prepare is not None:
            prepare = str(prepare).strip()
            _require(prepare.isidentifier(), f"ops[{i}] ({slot}) 'prepare' must be a python identifier")

        # Path-safety check now (existence + containment); content scanning later.
        _safe_relpath(root, source, kind="source")

        metadata = op.get("metadata")
        if metadata is not None:
            metadata = str(metadata).strip()
            _safe_relpath(root, metadata, kind="metadata")

        dtypes = tuple(str(d) for d in op.get("dtypes", ()))
        archs = tuple(str(a) for a in op.get("architectures", ()))

        known = {"slot", "source", "entry", "prepare", "dtypes", "architectures", "metadata"}
        extra = {k: v for k, v in op.items() if k not in known}

        ops.append(
            OpEntry(
                slot=slot,
                source=source,
                entry=entry,
                dtypes=dtypes,
                architectures=archs,
                metadata=metadata,
                prepare=prepare,
                extra=extra,
            )
        )

    return Manifest(bundle_id=bundle_id, abi_version=abi, ops=tuple(ops), raw=data)


def resolve_source(bundle_root: str | Path, op: OpEntry) -> Path:
    """Return the absolute, containment-checked path to an op's source file."""
    return _safe_relpath(Path(bundle_root), op.source, kind="source")
