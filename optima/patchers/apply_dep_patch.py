"""Reviewed rebuild patcher: apply a bundle's DECLARED dep patches to an OVERLAY copy.

The generic applier for the ``dep_patches`` tier — the ingestion path that lets a
bundle modify a PINNED dependency (first occupant: the flashinfer fe_export deep
seam) without any bespoke optima code per submission:

* The bundle ships INSPECTABLE text unified diffs, declared in its manifest
  (``[[dep_patches]]``), structurally validated at load (optima/deppatch.py: text
  modifications + new files only), scan-allowlisted only when declared, and folded
  into the per-slot copy fingerprints (raw sha + context-width-invariant normalized
  diff).
* This patcher is validator-shipped and reviewed (``optima/patchers/`` — the only
  place ``rebuild.json`` may select scripts from). It never executes bundle code:
  it parses the diffs and writes patched TEXT files, nothing else.
* WHERE a patch may land is arena policy (optima/dep_policy.py): the target must
  have a ``DepPolicy`` row and every touched path must match its globs. Policy
  violations are hard rejects on every box, CPU included — they are data
  validation, not build steps.
* The SHARED INSTALL IS NEVER MUTATED (unlike the campaign's patch-in-place +
  revert): the policy's subtree is copied into a candidate-local overlay keyed by
  bundle_id, patches apply to the COPY with byte-exact context (no fuzz — the dep
  is pinned; a mismatch means the bundle targets something else and must fail),
  and the runtime consume side repoints the dependency's (late-bound, upstream-
  sanctioned) csrc constant at the overlay + forces JIT for the policy's module
  names. This patcher materializes the overlay; the rebind happens at seam
  activation in the engine ranks.

Boxes where the dependency isn't installed (CPU intake / dry-run): validate
everything, then SKIP the overlay materialization with a notice. There is no
phantom-parity risk in that skip — without the dependency there is no engine run
to score; the validation half still fails closed on bad bundles.

Concurrent ranks (engine TP workers / distributed verify) all run the rebuild
plan: the overlay builds in a private temp dir and lands via an atomic rename;
losers of the race verify the winner's stamp and reuse it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath


def _log(msg: str) -> None:
    print(f"[optima.apply_dep_patch] {msg}", flush=True)


def _check_policy(target: str, file_patches) -> "object":
    from optima.dep_policy import PATCHABLE_DEPS

    policy = PATCHABLE_DEPS.get(target)
    if policy is None:
        raise RuntimeError(
            f"dep_patches target {target!r} is not on the validator's patchable-deps "
            f"allowlist ({sorted(PATCHABLE_DEPS)}) — rejecting the bundle"
        )
    subtree = PurePosixPath(policy.overlay_subtree)
    for fp in file_patches:
        p = PurePosixPath(fp.path)
        if subtree not in p.parents:
            raise RuntimeError(
                f"dep patch touches {fp.path!r} outside the overlay subtree "
                f"{policy.overlay_subtree!r} — rejecting the bundle"
            )
        if not any(fnmatch(fp.path, g) for g in policy.allowed_globs):
            raise RuntimeError(
                f"dep patch touches {fp.path!r} not matching the arena's allowed "
                f"globs {list(policy.allowed_globs)} — rejecting the bundle"
            )
    return policy


def _apply_to_overlay(policy, parsed_by_patch, site_root: Path, dest: Path) -> dict:
    """Copy the policy subtree into ``dest`` and apply every parsed patch. Returns the
    overlay manifest (written as overlay.json by the caller)."""
    from optima.deppatch import apply_file_patch

    src_subtree = site_root / policy.overlay_subtree
    if not src_subtree.is_dir():
        raise RuntimeError(f"pinned dependency subtree missing: {src_subtree}")
    dst_subtree = dest / policy.overlay_subtree
    shutil.copytree(src_subtree, dst_subtree, symlinks=False)

    touched: dict[str, str] = {}
    for _patch_rel, file_patches in parsed_by_patch:
        for fp in file_patches:
            target_file = dest / fp.path
            original = None
            if target_file.exists():
                original = target_file.read_text(encoding="utf-8")
            new_text = apply_file_patch(original, fp)  # raises on any context mismatch
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(new_text, encoding="utf-8")
            touched[fp.path] = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
    return {
        "subtree": policy.overlay_subtree,
        "force_jit_modules": list(policy.force_jit_modules),
        "files": touched,
    }


def _valid_overlay(dest: Path, want_stamp: dict, policy) -> tuple[bool, str, dict | None]:
    """Validate both identity metadata and every materialized source byte."""
    from optima.dep_policy import tree_hash

    stamp_path = dest / "overlay.json"
    if stamp_path.is_symlink() or not stamp_path.is_file():
        return False, "overlay stamp is missing or a symlink", None
    try:
        have = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"overlay stamp is unreadable: {exc}", None
    expected_keys = {
        "cache_key", "identity", "patch_shas", "subtree", "force_jit_modules",
        "files", "overlay_subtree_sha256",
    }
    if not isinstance(have, dict) or set(have) != expected_keys:
        return False, "overlay stamp schema is malformed", have if isinstance(have, dict) else None
    for key, value in want_stamp.items():
        if have.get(key) != value:
            return False, f"overlay stamp field {key!r} differs", have
    if have.get("subtree") != policy.overlay_subtree:
        return False, "overlay subtree policy differs", have
    if have.get("force_jit_modules") != list(policy.force_jit_modules):
        return False, "overlay force-JIT policy differs", have
    subtree = dest / policy.overlay_subtree
    try:
        actual = tree_hash(subtree)
    except RuntimeError as exc:
        return False, str(exc), have
    if have.get("overlay_subtree_sha256") != actual:
        return False, "materialized overlay bytes differ from stamp", have
    return True, "", have


def main() -> None:
    bundle = os.environ.get("OPTIMA_BUNDLE_PATH", "").strip()
    if not bundle:
        _log("no OPTIMA_BUNDLE_PATH set; nothing to apply")
        return

    from optima.deppatch import parse_patch_text
    from optima.manifest import load_manifest

    manifest = load_manifest(bundle)
    phase = os.environ.get("OPTIMA_REBUILD_PHASE", "all").strip().lower()
    if phase not in {"all", "build", "load"}:
        raise RuntimeError(f"unsupported OPTIMA_REBUILD_PHASE: {phase!r}")
    if not manifest.dep_patches:
        _log("bundle declares no dep_patches; nothing to apply")
        return

    # Group by target; parse + policy-check EVERYTHING first (hard rejects happen on
    # every box, dependency installed or not).
    by_target: dict[str, list[tuple[str, tuple]]] = {}
    patch_shas: dict[str, str] = {}
    for dp in manifest.dep_patches:
        raw = (Path(bundle) / dp.path).read_bytes()
        patch_shas[dp.path] = hashlib.sha256(raw).hexdigest()
        parsed = parse_patch_text(raw.decode("utf-8"))
        by_target.setdefault(dp.target, []).append((dp.path, parsed))

    policies = {t: _check_policy(t, [fp for _, parsed in entries for fp in parsed])
                for t, entries in by_target.items()}

    for target, entries in sorted(by_target.items()):
        policy = policies[target]
        from optima.dep_policy import dependency_site_root

        site_root = dependency_site_root(policy)
        if site_root is None:
            _log(f"dependency {policy.package!r} not installed on this box; policy checks "
                 "passed, SKIPPING overlay materialization (no engine run here to score)")
            continue

        from optima.dep_policy import overlay_base, overlay_identity, tree_hash

        identity = overlay_identity(bundle, target, site_root=site_root)
        dest = overlay_base(identity.cache_key) / target
        want_stamp = {
            "cache_key": identity.cache_key,
            "identity": identity.payload,
            "patch_shas": {rel: patch_shas[rel] for rel, _ in entries},
        }

        if phase == "load":
            # The candidate receives the content-addressed overlay cache read-only.
            # Validation must therefore be side-effect free (even creating a lock
            # file would require a writable mount and reopen cache-poisoning races).
            valid, why, _overlay_manifest = _valid_overlay(
                dest, want_stamp, policy
            )
            if not valid:
                raise RuntimeError(
                    f"refusing stale/missing dep overlay for {target}: {why}; "
                    "the trusted build worker must materialize this exact identity first"
                )
            _log(f"overlay cache hit for {target} ({dest})")
            continue

        # Same-identity processes serialize. Different bundle/dependency/policy
        # identities have different lock paths and can build concurrently safely.
        import fcntl

        lock_path = dest.parent / f".{target}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            valid, why, overlay_manifest = _valid_overlay(dest, want_stamp, policy)
            if valid:
                _log(f"overlay cache hit for {target} ({dest})")
                continue
            if dest.exists():
                _log(f"invalid overlay cache entry for {target} ({why}); rebuilding")
                shutil.rmtree(dest, ignore_errors=True)

            tmp = Path(tempfile.mkdtemp(prefix=f".{target}.", dir=dest.parent))
            try:
                overlay_manifest = _apply_to_overlay(policy, entries, site_root, tmp)
                overlay_manifest.update(want_stamp)
                overlay_manifest["overlay_subtree_sha256"] = tree_hash(
                    tmp / policy.overlay_subtree
                )
                (tmp / "overlay.json").write_text(
                    json.dumps(overlay_manifest, indent=2, sort_keys=True), encoding="utf-8"
                )
                os.rename(tmp, dest)  # same filesystem + identity lock => atomic landing
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            valid, why, overlay_manifest = _valid_overlay(dest, want_stamp, policy)
            if not valid:
                raise RuntimeError(f"newly materialized overlay failed self-check: {why}")
            _log(f"overlay ready for {target}: {dest / policy.overlay_subtree} "
                 f"(files patched: {len(overlay_manifest['files'])})")


main()
