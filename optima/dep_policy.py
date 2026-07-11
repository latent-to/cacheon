"""Which dependencies a bundle may patch, and where — VALIDATOR policy, never bundle data.

The ``dep_patches`` tier lets a bundle declare unified diffs against a PINNED
dependency tree (see optima/deppatch.py + optima/patchers/apply_dep_patch.py). This
module is the allowlist side of that contract: a patch target must have a row here,
and every file the patch touches must match this row's globs, or the one reviewed
applier hard-rejects the bundle. Nothing a bundle ships can widen this.

Consciously minimal and validator-owned. When the arena registry lands
(feat/arena-registry re-implementation — see the 2026-07-07 ledger), this table moves
onto ``Arena.patchable_deps`` so the allowlist is pinned per arena alongside the dep
versions it is valid against; keep the shape identical so that move is mechanical.

Why csrc-only for flashinfer: a .cu/.cuh/.h source patch is inspectable, fingerprints
like source (copy detection), and takes effect through a JIT rebuild the validator
controls (overlay + force-JIT — the runtime half). Patching PYTHON in a dependency is
NOT offered: dep Python executes in-process with validator privileges and would bypass
the sandbox scan that bundle Python goes through.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


_OVERLAY_CACHE_SCHEMA = "optima-dep-overlay-cache-v2"


def _overlay_root() -> Path:
    root = os.environ.get("OPTIMA_DEP_OVERLAY_CACHE", "")
    return Path(root) if root else Path.home() / ".cache" / "optima" / "dep_overlay"


def overlay_base(cache_key: str) -> Path:
    """Content-addressed overlay root.

    ``cache_key`` is produced by :func:`overlay_identity`; it is never a miner-
    controlled ``bundle_id``.  Keeping this one path constructor lets the reviewed
    applier, engine integration and FlashInfer JIT workspace agree exactly.
    """
    if len(cache_key) != 64 or any(c not in "0123456789abcdef" for c in cache_key):
        raise ValueError(f"invalid dep-overlay cache key: {cache_key!r}")
    return _overlay_root() / "v2" / cache_key[:2] / cache_key


@dataclass(frozen=True)
class DepPolicy:
    # Importable package whose site-packages install anchors the patch paths: a patch
    # path like "flashinfer/data/csrc/..." is resolved relative to the package's
    # site-root (the parent of the package directory).
    package: str
    # The subtree (site-root-relative, POSIX) that gets COPIED into the candidate-local
    # overlay. Must be broad enough that relative #includes inside it still resolve.
    overlay_subtree: str
    # fnmatch patterns (site-root-relative, POSIX; ``*`` crosses ``/``) for files a
    # patch may modify or create. Everything a patch touches must ALSO live under
    # overlay_subtree (else the patched file couldn't take effect via the overlay).
    allowed_globs: tuple[str, ...]
    # JitSpec names whose prebuilt AOT artifact must be bypassed at runtime so the
    # patched csrc actually compiles + loads (flashinfer prefers the AOT .so per spec
    # name; verified 2026-07-07 — see the ledger's overlay-assumptions report).
    force_jit_modules: tuple[str, ...] = ()
    # Runtime rebind coordinates: (module, attr) of the dependency's source-root
    # constant to repoint at ``<overlay>/<overlay_subtree>``. Must be LATE-BOUND at
    # every consumer site in the pinned dep (flashinfer's is, by upstream's own
    # documented design — env.py:17-19). None = the patched tree takes effect some
    # other way (no rebind step).
    env_rebind: tuple[str, str] | None = None


@dataclass(frozen=True)
class OverlayIdentity:
    cache_key: str
    payload: dict
    site_root: Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(root: Path) -> str:
    """Exact deterministic identity of an installed or materialized source tree."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise RuntimeError(f"dependency source subtree missing: {root}")
    h = hashlib.sha256()
    count = 0
    file_count = 0

    def record(kind: bytes, rel: str, data: bytes = b"") -> None:
        nonlocal count, file_count
        rel_b = rel.encode("utf-8")
        h.update(kind)
        h.update(len(rel_b).to_bytes(4, "big"))
        h.update(rel_b)
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
        count += 1
        if kind == b"F":
            file_count += 1

    def visit(path: Path, rel: str, ancestors: frozenset[tuple[int, int]]) -> None:
        if path.is_symlink():
            target_text = os.readlink(path)
            record(b"L", rel, target_text.encode("utf-8"))
            try:
                target = path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise RuntimeError(f"broken/cyclic dependency symlink: {path}") from exc
            # copytree(..., symlinks=False) dereferences links, so their target bytes
            # are compilation inputs too; hash them under the link's logical path.
            visit(target, rel, ancestors)
            return
        if path.is_dir():
            stat = path.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode in ancestors:
                raise RuntimeError(f"dependency source tree contains a directory cycle: {path}")
            record(b"D", rel)
            next_ancestors = ancestors | {inode}
            for child in sorted(path.iterdir(), key=lambda p: p.name):
                child_rel = f"{rel}/{child.name}" if rel else child.name
                visit(child, child_rel, next_ancestors)
            return
        if path.is_file():
            record(b"F", rel, path.read_bytes())
            return
        raise RuntimeError(f"dependency source tree contains a non-file entry: {path}")

    visit(root, "", frozenset())
    if file_count == 0:
        raise RuntimeError(f"dependency source subtree is empty: {root}")
    return h.hexdigest()


def dependency_site_root(policy: DepPolicy) -> Path | None:
    spec = importlib.util.find_spec(policy.package)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


def overlay_identity(
    bundle_path: str | Path,
    target: str,
    *,
    site_root: Path | None = None,
) -> OverlayIdentity:
    """Derive the cache identity independently in build and engine-load phases.

    It commits to the complete bundle tree, exact validator patcher, validator policy,
    and exact pinned dependency source subtree.  Thus a bundle-id collision, header or
    patch change, validator policy change, dependency drift, or patcher update cannot
    reuse an old overlay/JIT workspace.
    """
    policy = PATCHABLE_DEPS.get(target)
    if policy is None:
        raise RuntimeError(f"dependency target is not patchable: {target!r}")
    if site_root is None:
        site_root = dependency_site_root(policy)
    if site_root is None:
        raise RuntimeError(f"dependency package is not installed: {policy.package!r}")
    site_root = Path(site_root).resolve()
    dep_subtree = site_root / policy.overlay_subtree

    from optima.bundle_hash import content_hash

    patcher = Path(__file__).resolve().parent / "patchers" / "apply_dep_patch.py"
    # Normalize tuples to JSON arrays now so an in-memory identity compares byte-for-
    # byte with the same payload read back from overlay.json.
    policy_payload = json.loads(json.dumps(asdict(policy), sort_keys=True))
    payload = {
        "schema": _OVERLAY_CACHE_SCHEMA,
        "bundle_hash": content_hash(bundle_path),
        "target": target,
        "policy": policy_payload,
        "dependency_subtree_sha256": tree_hash(dep_subtree),
        "patcher_sha256": _sha256_file(patcher),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return OverlayIdentity(hashlib.sha256(encoded).hexdigest(), payload, site_root)


def overlay_workspace_base(bundle_path: str | Path, targets: list[str] | tuple[str, ...]) -> Path:
    """Private FlashInfer JIT workspace for one exact overlay stack."""
    identities = [overlay_identity(bundle_path, t).cache_key for t in sorted(set(targets))]
    if not identities:
        raise ValueError("overlay workspace requires at least one dependency target")
    digest = hashlib.sha256("\n".join(identities).encode("ascii")).hexdigest()
    return _overlay_root() / "jit_workspace" / "v2" / digest[:2] / digest


PATCHABLE_DEPS: dict[str, DepPolicy] = {
    "flashinfer": DepPolicy(
        package="flashinfer",
        overlay_subtree="flashinfer/data/csrc",
        # First occupant: the MoE cutlass backend (the fe_export deep seam lives in
        # fused_moe/cutlass_backend). Widen deliberately, per-arena, as slots demand.
        allowed_globs=("flashinfer/data/csrc/fused_moe/*",),
        force_jit_modules=("fused_moe_103",),
        env_rebind=("flashinfer.jit.env", "FLASHINFER_CSRC_DIR"),
    ),
}


def read_validated_overlay(
    bundle_path: str | Path, target: str
) -> tuple[OverlayIdentity, dict, Path]:
    """Locate and fully validate an overlay before an engine consumes it."""
    from optima.manifest import load_manifest

    bundle = Path(bundle_path).resolve()
    manifest = load_manifest(bundle)
    identity = overlay_identity(bundle, target)
    policy = PATCHABLE_DEPS[target]
    dest = overlay_base(identity.cache_key) / target
    stamp_path = dest / "overlay.json"
    if stamp_path.is_symlink() or not stamp_path.is_file():
        raise RuntimeError(
            f"dep overlay stamp missing for {target!r} at {stamp_path}"
        )
    try:
        data = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"dep overlay stamp unreadable at {stamp_path}: {exc}") from exc
    expected_keys = {
        "cache_key", "identity", "patch_shas", "subtree", "force_jit_modules",
        "files", "overlay_subtree_sha256",
    }
    if not isinstance(data, dict) or set(data) != expected_keys:
        raise RuntimeError(f"dep overlay stamp schema mismatch at {stamp_path}")
    patch_shas = {
        dp.path: _sha256_file(bundle / dp.path)
        for dp in manifest.dep_patches if dp.target == target
    }
    expected = {
        "cache_key": identity.cache_key,
        "identity": identity.payload,
        "patch_shas": patch_shas,
        "subtree": policy.overlay_subtree,
        "force_jit_modules": list(policy.force_jit_modules),
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(
                f"dep overlay stamp field {key!r} mismatches exact candidate identity"
            )
    actual = tree_hash(dest / policy.overlay_subtree)
    if data.get("overlay_subtree_sha256") != actual:
        raise RuntimeError(
            f"dep overlay source tree hash mismatch for {target!r} at {dest}"
        )
    return identity, data, dest
